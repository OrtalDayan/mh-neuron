#!/usr/bin/env python3
"""Run 1 ablation evaluation: apply ablation → save checkpoint → VLMEvalKit → clean up.

This script applies a single ablation configuration (category, fraction, trial),
saves the ablated model checkpoint, runs VLMEvalKit for BRV-identical evaluation,
parses the results, and cleans up the checkpoint.

Usage:
    python run1_ablation.py \
        --model_type llava-llama3 \
        --model_path llava-hf/llama3-llava-next-8b-hf \
        --model_name llava-next-llama3-8b \
        --hook_point gate_up \
        --label_dir results/3-classify/full/llava-next-llama3-8b/llm_permutation_gate_up_min100_max2048 \
        --ranking D_x_norm \
        --category visual \
        --fraction 0.10 \
        --trial_idx ranked \
        --benchmark MathVerse_MINI_Text_Dominant \
        --vlmeval_dir modern_vlms/VLMEvalKit \
        --vlmeval_python modern_vlms/.venv/bin/python \
        --judge gpt-4o-mini \
        --output_dir results/24-ranked-ablation/full/llava-next-llama3-8b/run1
"""
import argparse
import json
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import torch


def load_labels(label_dir):
    """Load merged neuron labels."""
    for name in ['merged_labels.json', 'neuron_labels_all.json', 'neuron_labels_permutation_all.json']:
        label_path = os.path.join(label_dir, name)
        if os.path.isfile(label_path):
            with open(label_path) as f:
                return json.load(f)
    raise FileNotFoundError(f'No label file found in {label_dir}')


def precompute_output_norms(model, hook_point, n_layers):
    """Precompute per-unit output-projection L2 norms from model weights.

    For gate/gate_up: L2 norm of each down_proj column → shape (n_neurons,).
        Measures how much neuron n's activation moves the residual stream.
    For attn: Frobenius norm of each head's block in o_proj → shape (n_heads,).

    Returns:
        dict: {layer_str: {neuron_idx_str: norm_float}}
    """
    # LLaMA/Qwen/Mistral use mlp.down_proj naming. InternLM (InternVL's backbone)
    # uses feed_forward.w2 / attention.wo. Probe parameter names to find which
    # convention this model follows so the same code works for both.
    if hook_point in ('gate', 'gate_up'):
        candidate_suffixes = ['.mlp.down_proj.weight', '.feed_forward.w2.weight']
    elif hook_point == 'attn':
        candidate_suffixes = ['.self_attn.o_proj.weight', '.attention.wo.weight']
    else:
        raise ValueError(f'Unknown hook_point: {hook_point}')

    param_names = [n for n, _ in model.named_parameters()]
    target_suffix = None
    for suffix in candidate_suffixes:
        if any(f'layers.0{suffix}' in n for n in param_names):
            target_suffix = suffix
            break
    if target_suffix is None:
        sample = [n for n in param_names if 'layers.0' in n][:5]
        raise ValueError(
            f'No down-projection / output-projection layer found for '
            f'hook_point={hook_point!r}. Tried suffixes {candidate_suffixes}. '
            f'Sample layer-0 param names: {sample}'
        )

    layer_params = {}
    for name, param in model.named_parameters():
        for l in range(n_layers):
            if f'layers.{l}{target_suffix}' in name:
                layer_params[l] = param
                break

    importance = {}
    with torch.no_grad():
        for l in range(n_layers):
            if l not in layer_params:
                continue
            w = layer_params[l].float()

            if hook_point in ('gate', 'gate_up'):
                # w shape: (hidden_dim, intermediate_dim)
                # Column n = neuron n → L2 norm per column
                norms = torch.norm(w, dim=0).cpu().numpy()
            elif hook_point == 'attn':
                # w shape: (hidden_dim, hidden_dim)
                hidden_dim = w.shape[0]
                for nh in [32, 28, 16, 40, 64]:
                    if hidden_dim % nh == 0:
                        n_heads = nh
                        break
                else:
                    n_heads = 32
                head_dim = hidden_dim // n_heads
                w_heads = w.reshape(hidden_dim, n_heads, head_dim)
                norms = torch.norm(w_heads, dim=(0, 2)).cpu().numpy()

            importance[str(l)] = {str(i): float(norms[i]) for i in range(len(norms))}

    sample_l = next(iter(importance))
    vals = list(importance[sample_l].values())
    print(f'  Output norms ({hook_point}): layer {sample_l} range [{min(vals):.4f}, {max(vals):.4f}], mean {sum(vals)/len(vals):.4f}')
    return importance


def rank_neurons(neurons, ranking='D_x_norm', importance=None):
    """Rank neurons by the specified strategy, descending importance."""
    if ranking == 'D':
        def key(n):
            cat = n.get('category', 'unknown')
            if cat == 'multimodal':
                # Most balanced first: high (1-|D|) = small |D|
                return 1.0 - abs(n.get('rate_diff', 0))
            return abs(n.get('rate_diff', 0))
    elif ranking == 'norm':
        def key(n):
            if importance is None:
                return 0
            layer = str(n['layer'])
            idx = n['neuron_idx']
            return importance.get(layer, {}).get(str(idx), 0)
    elif ranking == 'D_x_norm':
        def key(n):
            cat = n.get('category', 'unknown')
            if importance is None:
                norm = 1.0
            else:
                layer = str(n['layer'])
                idx = n['neuron_idx']
                norm = importance.get(layer, {}).get(str(idx), 0)
            if cat == 'multimodal':
                # Most balanced AND influential: (1-|D|) * norm
                return (1.0 - abs(n.get('rate_diff', 0))) * norm
            return abs(n.get('rate_diff', 0)) * norm
    else:
        raise ValueError(f'Unknown ranking: {ranking}')

    return sorted(neurons, key=key, reverse=True)


def get_vlmeval_model_name(model_type):
    """Map our model type to VLMEvalKit registered model name."""
    mapping = {
        'llava-llama3': ('LLaVA_Next', 'llava-next-llama3-8b_baseline'),
        'idefics2': ('Idefics2', 'idefics2_8b'),
        'qwen2vl': ('QwenVLChat', 'Qwen2-VL-7B-Instruct'),
        'llava-liuhaotian': ('LLaVA', 'llava_v1.5_7b'),
        'llava-ov': ('LLaVA_OneVision_HF', 'llava-onevision-qwen2-7b-ov-hf'),
        'internvl': ('InternVLChat', 'InternVL2_5-8B'),
        'qwen25vl-3b': ('QwenVLChat', 'Qwen2.5-VL-3B-Instruct'),
    }
    return mapping.get(model_type, (None, None))


def parse_vlmeval_score(score_csv):
    """Parse VLMEvalKit score CSV to get Overall accuracy and subcategory scores."""
    overall = None
    subcategories = {}
    with open(score_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get overall accuracy
            if overall is None:
                for key in ['Overall', 'accuracy']:
                    if key in row and row[key] not in ('nan', '', None):
                        overall = float(row[key])
                        break
            # Get subcategory scores (columns with dict-like values)
            for col, val in row.items():
                if col in ('Overall', 'accuracy', 'correct', 'total'):
                    continue
                if isinstance(val, str) and val.startswith('{'):
                    try:
                        import ast
                        sub = ast.literal_eval(val)
                        if isinstance(sub, dict) and 'accuracy' in sub:
                            subcategories[col] = {
                                'accuracy': float(sub['accuracy']),
                                'correct': int(sub['correct']),
                                'total': int(sub['total']),
                            }
                    except:
                        pass
    # Normalize units to percent-scale (0-100). Some VLMEvalKit benchmarks
    # write accuracy as a 0-1 fraction (TriviaQA), others as 0-100 percent
    # (POPE, MathVerse). Anything <= 1.0 is a fraction; multiply by 100.
    if overall is not None and overall <= 1.0:
        overall = overall * 100.0
    return overall, subcategories


def run_triviaqa_inline(model, model_type, model_path, triviaqa_path, triviaqa_num, seed=42):
    """Run TriviaQA evaluation inline on an ablated model (text-only, no VLMEvalKit needed).

    Loads the TriviaQA dataset, moves the (already-ablated) model to GPU, and scores
    each question via greedy generation with alias matching against gold answers.

    Returns:
        dict with 'accuracy' (0-1), 'correct', 'total'.
    """
    import numpy as np
    from transformers import AutoProcessor, AutoTokenizer

    # ── Load TriviaQA data ──
    print(f'  Loading TriviaQA from {triviaqa_path} (n={triviaqa_num})')
    with open(triviaqa_path) as fh:
        data = json.load(fh)

    items = []
    for entry in data['Data']:
        answer_obj = entry['Answer']
        aliases = set()
        aliases.add(answer_obj['Value'].strip().lower())
        for alias in answer_obj.get('Aliases', []):
            aliases.add(alias.strip().lower())
        items.append({
            'question': entry['Question'],
            'answer': answer_obj['Value'],
            'aliases': aliases,
        })

    if triviaqa_num is not None and triviaqa_num < len(items):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(items), size=triviaqa_num, replace=False)
        items = [items[i] for i in indices]

    print(f'  Loaded {len(items)} TriviaQA questions')

    # ── Load tokenizer/processor ──
    processor = AutoProcessor.from_pretrained(model_path)
    # Extract tokenizer from processor (works for LlavaNext/Idefics2/Qwen2VL)
    tokenizer = getattr(processor, 'tokenizer', processor)

    # ── Move model to GPU ──
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'  Moving model to {device}')
    model = model.to(device).eval()

    # ── Run evaluation ──
    from tqdm import tqdm
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(items, desc='TriviaQA'):
            prompt = (f"Answer the following question briefly.\n"
                      f"Question: {item['question']}\nAnswer:")

            # Text-only generation: use the full VLM wrapper's .generate() without
            # any pixel_values. LlamaModel/MistralModel (inner encoders) don't have
            # .generate() — only the CausalLM wrapper does. For LLaVA-Next/Idefics2/
            # Qwen2-VL, calling model.generate() with just input_ids dispatches through
            # the LM head and skips the vision path when pixel_values is absent.
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0, inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True)

            answer_lower = answer.strip().lower()
            if any(alias in answer_lower for alias in item['aliases']):
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    return {
        'accuracy': round(accuracy, 4),
        'accuracy_pct': round(accuracy * 100, 2),
        'correct': correct,
        'total': total,
    }


def main():
    p = argparse.ArgumentParser(description='Run 1 ablation: checkpoint + VLMEvalKit')
    p.add_argument('--model_type', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--n_layers', type=int, default=32)
    p.add_argument('--hook_point', default='gate_up')
    p.add_argument('--label_dir', required=True)
    p.add_argument('--label_dir_attn', default=None,
                   help='Path to attention labels dir. If set along with a gate/gate_up --hook_point, '
                        'ablate BOTH MLP neurons (down_proj cols) AND attention heads (o_proj blocks) '
                        'in the same category, at the same fraction.')
    p.add_argument('--ranking', default='D_x_norm')
    p.add_argument('--category', required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--trial_idx', required=True, help='ranked or 0-N for random')
    p.add_argument('--benchmark', required=True, help='VLMEvalKit dataset name')
    p.add_argument('--vlmeval_dir', default='modern_vlms/VLMEvalKit')
    p.add_argument('--vlmeval_python', default='modern_vlms/.venv/bin/python')
    p.add_argument('--judge', default='gpt-4o-mini')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint_dir', default=None, help='Temp dir for checkpoint (default: auto)')
    # TriviaQA-specific args (only used when --benchmark TriviaQA)
    p.add_argument('--triviaqa_path', default='data/triviaqa/qa/verified-web-dev.json',
                   help='Path to TriviaQA verified-web-dev.json')
    p.add_argument('--triviaqa_num', type=int, default=2000,
                   help='Number of TriviaQA questions to evaluate')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if already done
    trial_tag = 'ranked' if args.trial_idx == 'ranked' else f'r{args.trial_idx}'
    # Combined mode: mark hook as "combined" (mlp + attn)
    _hook_tag = args.hook_point
    if args.label_dir_attn and args.hook_point in ('gate', 'gate_up'):
        _hook_tag = f'{args.hook_point}_attn'
    out_name = f'run1_{args.ranking}_{_hook_tag}_{args.category}_{args.benchmark}_f{args.fraction:.2f}_{trial_tag}.json'
    out_path = os.path.join(args.output_dir, out_name)
    if os.path.isfile(out_path):
        print(f'  [skip] Already done: {out_path}')
        return

    # ── Load labels and select neurons ──
    print(f'\n  Loading labels from {args.label_dir}')
    labels = load_labels(args.label_dir)

    # Flatten all neurons with their category
    neurons = []
    for layer_key, layer_data in labels.items():
        layer_num = int(layer_key)
        if isinstance(layer_data, list):
            # Format: list of {label, p_value, observed_rate_diff, neuron_idx}
            for info in layer_data:
                neurons.append({
                    'layer': layer_num,
                    'neuron_idx': info['neuron_idx'],
                    'category': info['label'],
                    'rate_diff': info.get('observed_rate_diff', 0),
                    'p_value': info.get('p_value', 1.0),
                })
        elif isinstance(layer_data, dict):
            # Format: {neuron_idx_str: {category, rate_diff, p_value, ...}}
            for idx_str, info in layer_data.items():
                if isinstance(info, dict) and ('category' in info or 'label' in info):
                    neurons.append({
                        'layer': layer_num,
                        'neuron_idx': int(idx_str),
                        'category': info.get('category', info.get('label', 'unknown')),
                        'rate_diff': info.get('rate_diff', info.get('observed_rate_diff', 0)),
                        'p_value': info.get('p_value', 1.0),
                    })

    # Filter by category
    cat_neurons = [n for n in neurons if n['category'] == args.category]
    print(f'  {args.category} neurons: {len(cat_neurons)}')

    # ── Load model (needed for norms and ablation) ──
    print(f'\n  Loading model: {args.model_path}')
    if args.model_type == 'llava-llama3':
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map='cpu')
    elif args.model_type == 'idefics2':
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map='cpu')
    elif args.model_type == 'qwen2vl':
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map='cpu')
    elif args.model_type == 'llava-liuhaotian':
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map='cpu')
    elif args.model_type == 'llava-ov':
        from transformers import LlavaOnevisionForConditionalGeneration
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map='cpu')
    elif args.model_type == 'internvl':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True)
    elif args.model_type == 'qwen25vl-3b':
        from transformers import Qwen2_5_VLForConditionalGeneration
        # Qwen2.5-VL weights are stored in bfloat16. Loading with torch_dtype=float16
        # triggers an implicit conversion that deadlocks during shard loading. Use
        # bfloat16 to match the native dtype for instant load.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map='cpu')
    else:
        raise ValueError(f'Unsupported model type: {args.model_type}')

    # ── Compute importance norms from model weights ──
    importance = precompute_output_norms(model, args.hook_point, args.n_layers)

    # Rank and select top fraction
    ranked = rank_neurons(cat_neurons, args.ranking, importance)
    K = max(1, int(len(ranked) * args.fraction))

    if args.trial_idx == 'ranked':
        selected = ranked[:K]
        print(f'  Ranked ablation: top {K}/{len(ranked)} {args.category} neurons')
    else:
        seed_i = int(args.trial_idx)
        rng = np.random.RandomState(args.seed + seed_i * 1000)
        all_neurons_flat = neurons  # random from ALL neurons
        random_indices = rng.choice(len(all_neurons_flat), size=K, replace=False)
        selected = [all_neurons_flat[i] for i in random_indices]
        print(f'  Random trial {seed_i}: {K} random neurons (seed={args.seed + seed_i * 1000})')

    # Build neuron map {layer: [indices]}
    from collections import defaultdict
    nmap = defaultdict(list)
    for n in selected:
        nmap[n['layer']].append(n['neuron_idx'])

    # ── Apply ablation (zero weights permanently) ──
    print(f'  Applying ablation: zeroing {sum(len(v) for v in nmap.values())} neurons across {len(nmap)} layers')

    # Determine target weight suffix.
    # LLaMA/Qwen/Mistral use mlp.down_proj / self_attn.o_proj naming.
    # InternLM (InternVL's backbone) uses feed_forward.w2 / attention.wo.
    # Probe parameter names to find which convention this model follows
    # so the same ablation code works for both. (Mirrors the logic already
    # used in precompute_output_norms.)
    if args.hook_point in ('gate', 'gate_up'):
        candidate_suffixes = ['.mlp.down_proj.weight', '.feed_forward.w2.weight']
    elif args.hook_point == 'attn':
        candidate_suffixes = ['.self_attn.o_proj.weight', '.attention.wo.weight']
    else:
        raise ValueError(f'Unknown hook_point: {args.hook_point}')

    param_names = [n for n, _ in model.named_parameters()]
    target_suffix = None
    for suffix in candidate_suffixes:
        if any(f'layers.0{suffix}' in n for n in param_names):
            target_suffix = suffix
            break
    if target_suffix is None:
        sample = [n for n in param_names if 'layers.0' in n][:5]
        raise ValueError(
            f'No down-projection / output-projection layer found for '
            f'hook_point={args.hook_point!r}. Tried suffixes {candidate_suffixes}. '
            f'Sample layer-0 param names: {sample}'
        )
    print(f'  Detected target weight suffix: {target_suffix}')

    # Find weight parameters by name (model-agnostic, same as WeightZeroing)
    weight_params = {}
    for name, param in model.named_parameters():
        for layer_idx in nmap:
            pattern = f'layers.{layer_idx}{target_suffix}'
            if pattern in name:
                weight_params[layer_idx] = param
                break

    missing = set(nmap.keys()) - set(weight_params.keys())
    if missing:
        print(f'  WARNING: Could not find parameters for layers: {missing}')

    with torch.no_grad():
        for layer_idx, neuron_indices in nmap.items():
            if layer_idx not in weight_params:
                continue
            param = weight_params[layer_idx]

            if args.hook_point in ('gate', 'gate_up'):
                # Zero columns in down_proj
                param.data[:, neuron_indices] = 0
            elif args.hook_point == 'attn':
                # Zero blocks in o_proj per head
                head_dim = param.shape[1] // max(max(idx for idx in neuron_indices) + 1, 1)
                for idx in neuron_indices:
                    start = idx * head_dim
                    end = start + head_dim
                    if end <= param.shape[1]:
                        param.data[:, start:end] = 0

    # ── Combined ablation: also zero attention heads in the same category ──
    if args.label_dir_attn and args.hook_point in ('gate', 'gate_up'):
        print(f'\n  [combined] Loading attention labels from {args.label_dir_attn}')
        attn_labels = load_labels(args.label_dir_attn)

        # Flatten attention heads with their category
        attn_heads = []
        for layer_key, layer_data in attn_labels.items():
            layer_num = int(layer_key)
            if isinstance(layer_data, list):
                for info in layer_data:
                    attn_heads.append({
                        'layer': layer_num,
                        'neuron_idx': info['neuron_idx'],  # head index for attn labels
                        'category': info['label'],
                        'rate_diff': info.get('observed_rate_diff', 0),
                        'p_value': info.get('p_value', 1.0),
                    })
            elif isinstance(layer_data, dict):
                for idx_str, info in layer_data.items():
                    if isinstance(info, dict) and ('category' in info or 'label' in info):
                        attn_heads.append({
                            'layer': layer_num,
                            'neuron_idx': int(idx_str),
                            'category': info.get('category', info.get('label', 'unknown')),
                            'rate_diff': info.get('rate_diff', info.get('observed_rate_diff', 0)),
                            'p_value': info.get('p_value', 1.0),
                        })

        cat_heads = [h for h in attn_heads if h['category'] == args.category]
        print(f'  [combined] {args.category} attention heads: {len(cat_heads)}')

        # Rank attention heads (use pre-existing ranking on D/norm/D_x_norm; reuse same fn)
        attn_importance = precompute_output_norms(model, 'attn', args.n_layers)
        ranked_heads = rank_neurons(cat_heads, args.ranking, attn_importance)
        K_attn = max(1, int(len(ranked_heads) * args.fraction))

        if args.trial_idx == 'ranked':
            selected_heads = ranked_heads[:K_attn]
            print(f'  [combined] Ranked: top {K_attn}/{len(ranked_heads)} {args.category} heads')
        else:
            seed_i = int(args.trial_idx)
            rng = np.random.RandomState(args.seed + seed_i * 1000 + 1)  # +1 offset vs MLP rng
            random_h_idx = rng.choice(len(attn_heads), size=K_attn, replace=False)
            selected_heads = [attn_heads[i] for i in random_h_idx]
            print(f'  [combined] Random trial {seed_i}: {K_attn} random heads')

        # Build head map and zero o_proj blocks.
        # Same naming-convention probe as MLP block above: LLaMA uses
        # self_attn.o_proj, InternLM uses attention.wo.
        hmap = defaultdict(list)
        for h in selected_heads:
            hmap[h['layer']].append(h['neuron_idx'])

        attn_candidate_suffixes = ['.self_attn.o_proj.weight', '.attention.wo.weight']
        attn_param_names = [n for n, _ in model.named_parameters()]
        attn_target_suffix = None
        for suffix in attn_candidate_suffixes:
            if any(f'layers.0{suffix}' in n for n in attn_param_names):
                attn_target_suffix = suffix
                break
        if attn_target_suffix is None:
            raise ValueError(
                f'No attention output-projection layer found. '
                f'Tried suffixes {attn_candidate_suffixes}.'
            )

        attn_weight_params = {}
        for name, param in model.named_parameters():
            for layer_idx in hmap:
                pattern = f'layers.{layer_idx}{attn_target_suffix}'
                if pattern in name:
                    attn_weight_params[layer_idx] = param
                    break

        with torch.no_grad():
            for layer_idx, head_indices in hmap.items():
                if layer_idx not in attn_weight_params:
                    continue
                param = attn_weight_params[layer_idx]
                n_heads_total = max(max(idx for idx in head_indices) + 1, 1)
                head_dim = param.shape[1] // n_heads_total
                for idx in head_indices:
                    start = idx * head_dim
                    end = start + head_dim
                    if end <= param.shape[1]:
                        param.data[:, start:end] = 0
        print(f'  [combined] Zeroed {sum(len(v) for v in hmap.values())} heads across {len(hmap)} layers')

    # TriviaQA now routes through VLMEvalKit (same as POPE/MathVerse) via the
    # custom TriviaQA class in VLMEvalKit_brv. Falls through to the standard
    # checkpoint + VLMEvalKit --merge_model path below.

    # ── Save ablated language model state_dict ──
    # BRV's VLMEvalKit loads base model then replaces LM weights via --merge_model
    ckpt_dir = os.path.join(args.output_dir, '_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir,
        f'merged_model_{args.ranking}_{_hook_tag}_{args.category}_f{args.fraction:.2f}_{trial_tag}.pth')

    # Extract language model state_dict based on model type
    # Key format must match what BRV's load_state_dict target expects:
    #   LLaVA:   loads into LlamaForCausalLM   → keys need "model." prefix
    #   Idefics: loads into MistralModel        → keys need NO prefix
    #   Qwen:    loads into Qwen2Model          → keys need NO prefix
    if args.model_type == 'llava-llama3':
        lm_state = model.language_model.state_dict()
    elif args.model_type == 'idefics2':
        lm_state = model.model.text_model.state_dict()
    elif args.model_type == 'qwen2vl':
        lm_state = model.model.language_model.state_dict()
    elif args.model_type == 'llava-liuhaotian':
        lm_state = model.language_model.state_dict()
    elif args.model_type == 'llava-ov':
        lm_state = model.language_model.state_dict()
    elif args.model_type == 'internvl':
        lm_state = model.language_model.state_dict()
    elif args.model_type == 'qwen25vl-3b':
        lm_state = model.model.language_model.state_dict()
    else:
        raise ValueError(f'Unsupported model type: {args.model_type}')

    sample_key = list(lm_state.keys())[0]
    print(f'  State dict sample key: {sample_key}')
    print(f'  Total keys: {len(lm_state)}')

    if args.model_type in ('llava-llama3', 'llava-liuhaotian', 'llava-ov', 'internvl'):
        # BRV: model.model.language_model.load_state_dict() → LlamaForCausalLM-style
        # Needs: model.layers.0..., lm_head.weight
        if not sample_key.startswith('model.') and not sample_key.startswith('lm_head'):
            print(f'  Adding "model." prefix for LLaVA')
            lm_state = {f'model.{k}': v for k, v in lm_state.items()}
            # lm_head is at model.lm_head (not model.language_model.lm_head)
            if hasattr(model, 'lm_head'):
                lm_state['lm_head.weight'] = model.lm_head.weight.data
                print(f'  Added lm_head.weight from model.lm_head')
            elif hasattr(model.language_model, 'lm_head'):
                lm_state['lm_head.weight'] = model.language_model.lm_head.weight.data
                print(f'  Added lm_head.weight from model.language_model.lm_head')
    elif args.model_type in ('idefics2', 'qwen2vl', 'qwen25vl-3b'):
        # BRV: model.model.model.text_model.load_state_dict() → MistralModel/Qwen2Model
        # Needs: embed_tokens.weight, layers.0... (NO model. prefix, NO lm_head)
        if sample_key.startswith('model.'):
            tag = {'idefics2': 'Idefics2', 'qwen2vl': 'Qwen2VL', 'qwen25vl-3b': 'Qwen2.5VL'}[args.model_type]
            print(f'  Stripping "model." prefix for {tag}')
            lm_state = {k[len('model.'):]: v for k, v in lm_state.items()
                        if k.startswith('model.')}
        # Remove lm_head if present (inner Model doesn't have it)
        lm_state.pop('lm_head.weight', None)

    # Atomic write: write to .tmp, atomically rename, then fsync to force NFS commit.
    # Prevents VLMEvalKit's torch.load() from reading a partially-written checkpoint
    # under heavy concurrent load (PytorchStreamReader 'invalid header' errors).
    ckpt_path_tmp = ckpt_path + '.tmp'
    torch.save(lm_state, ckpt_path_tmp)
    os.replace(ckpt_path_tmp, ckpt_path)
    fd = os.open(ckpt_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    print(f'  Saved ablated LM weights to {ckpt_path} ({os.path.getsize(ckpt_path) / 1e9:.1f} GB)')

    # Free memory
    del model, lm_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Run VLMEvalKit with --merge_model ──
    _VE_MODEL = {
        'llava-llama3': 'llava_next_llama3',
        'idefics2': 'idefics2_8b',
        'qwen2vl': 'Qwen2-VL-7B-Instruct',
        'llava-liuhaotian': 'llava_v1.5_7b',
        'llava-ov': 'llava-onevision-qwen2-7b-ov-hf',
        'internvl': 'InternVL2_5-8B',
        'qwen25vl-3b': 'Qwen2.5-VL-3B-Instruct',
    }.get(args.model_type, 'UNKNOWN')

    # Read API key
    env_path = os.path.join(args.vlmeval_dir, '.env')
    api_key = ''
    if os.path.isfile(env_path):
        for line in open(env_path):
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()

    vlmeval_workdir = os.path.join(args.output_dir,
        f'vlmeval_work_{args.ranking}_{args.hook_point}_{args.category}_{args.benchmark}_f{args.fraction:.2f}_{trial_tag}')
    os.makedirs(vlmeval_workdir, exist_ok=True)

    cmd = (
        f'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        f'export OPENAI_API_KEY="{api_key}" && '
        f'{args.vlmeval_python} {os.path.join(args.vlmeval_dir, "run.py")} '
        f'--data {args.benchmark} '
        f'--model {_VE_MODEL} '
        f'--merge_model {ckpt_path} '
        f'--judge {args.judge} '
        f'--work-dir {vlmeval_workdir}'
    )

    print(f'\n  Running VLMEvalKit:')
    print(f'    Model: {_VE_MODEL}')
    print(f'    Merge: {ckpt_path}')
    print(f'    Benchmark: {args.benchmark}')
    print(f'    Judge: {args.judge}')
    sys.stdout.flush()

    result = subprocess.run(cmd, shell=True, text=True)
    print(f'    VLMEvalKit exit code: {result.returncode}')
    if result.returncode != 0:
        print(f'  VLMEvalKit FAILED (exit code {result.returncode})')
        with open(out_path, 'w') as f:
            json.dump({'error': f'VLMEvalKit exit code {result.returncode}', 'returncode': result.returncode}, f, indent=2)
    else:
        # ── Parse results ──
        score_files = []
        for root, dirs, files in os.walk(vlmeval_workdir):
            for fname in files:
                if 'score.csv' in fname and args.benchmark in fname:
                    score_files.append(os.path.join(root, fname))

        if score_files:
            accuracy, subcategories = parse_vlmeval_score(score_files[0])
            print(f'\n  VLMEvalKit result: {accuracy:.2f}%')

            save_data = {
                'model_type': args.model_type,
                'model_name': args.model_name,
                'hook_point': args.hook_point,
                'ranking': args.ranking,
                'category': args.category,
                'fraction': args.fraction,
                'trial': args.trial_idx,
                'benchmark': args.benchmark,
                'judge': args.judge,
                'accuracy': accuracy / 100.0,  # normalize to 0-1
                'accuracy_pct': accuracy,
                'subcategories': subcategories,
                'score_csv': score_files[0],
                'eval_mode': 'run1_vlmeval',
            }
            with open(out_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f'  Saved: {out_path}')

            # Copy score CSV to output dir before cleanup
            csv_copy = out_path.replace('.json', '_score.csv')
            shutil.copy2(score_files[0], csv_copy)
            print(f'  Score CSV copied: {csv_copy}')
        else:
            print(f'  WARNING: No score CSV found in {vlmeval_workdir}')
            # List what VLMEvalKit produced for debugging
            for root, dirs, files in os.walk(vlmeval_workdir):
                for fname in files:
                    print(f'    Found: {os.path.join(root, fname)}')
            with open(out_path, 'w') as f:
                json.dump({'error': 'No score CSV found'}, f, indent=2)

    # ── Clean up .pth file + VLMEvalKit workdir ──
    print(f'  Cleaning up checkpoint: {ckpt_path}')
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    print(f'  Cleaning up VLMEvalKit workdir: {vlmeval_workdir}')
    shutil.rmtree(vlmeval_workdir, ignore_errors=True)

    print(f'\n  Run 1 ablation complete.')


if __name__ == '__main__':
    main()