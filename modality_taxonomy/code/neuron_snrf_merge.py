#!/usr/bin/env python3
"""
neuron_snrf_merge.py — Layer 1b: PMBT-Guided Shared Neuron Low-Rank Fusion

Two-stage pipeline:
  Stage 1 (--stage profile): Run math LLM + VLM on reasoning prompts,
    record top-K% activated FFN neurons per layer, compute intersection
    → shared_neurons.json

  Stage 2 (--stage merge): Load shared neurons + PMBT text mask,
    intersect to get PMBT-guided shared set, compute ΔW = W_math - W_base,
    apply low-rank SVD truncation (rank 16), inject at intersection positions,
    save merged model.

Reference: "Do LLMs and VLMs Share Neurons for Inference?" (Cui et al., 2026)
  Adapted to use PMBT text neuron mask from our pipeline as an additional filter.

Compatible with run_pipeline.sh as step 23.

Usage:
  # Stage 1: profile shared neurons between math LLM and VLM
  python neuron_snrf_merge.py --stage profile \
      --vlm_path Qwen/Qwen2.5-VL-7B-Instruct \
      --math_llm_path Qwen/Qwen2.5-Math-7B \
      --model_type qwen2vl \
      --output_dir results/23-snrf/qwen2vl \
      --n_prompts 100 --top_k_pct 0.10

  # Stage 2: merge with PMBT mask
  python neuron_snrf_merge.py --stage merge \
      --vlm_path Qwen/Qwen2.5-VL-7B-Instruct \
      --base_llm_path Qwen/Qwen2.5-7B \
      --math_llm_path Qwen/Qwen2.5-Math-7B \
      --label_dir results/3-classify/full/qwen2.5-vl-7b/llm_permutation \
      --shared_neurons_json results/23-snrf/qwen2vl/shared_neurons.json \
      --model_type qwen2vl \
      --output_dir results/23-snrf/qwen2vl \
      --svd_rank 16 --beta 0.5

  # Both stages combined:
  python neuron_snrf_merge.py --stage both ...
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Argument parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='SNRF + PMBT: shared neuron low-rank fusion')

    # Stage selection
    p.add_argument('--stage', required=True,
                   choices=['profile', 'merge', 'both'],
                   help='profile = find shared neurons; '
                        'merge = SVD inject at PMBT∩shared; '
                        'both = run profile then merge')

    # Model paths
    p.add_argument('--vlm_path', required=True,
                   help='HuggingFace path to the target VLM')
    p.add_argument('--base_llm_path', default=None,
                   help='HuggingFace path to the base LLM '
                        '(only needed for merge stage)')
    p.add_argument('--math_llm_path', required=True,
                   help='HuggingFace path to the math-specialised LLM')

    # PMBT labels (only needed for merge)
    p.add_argument('--label_dir', default=None,
                   help='Directory containing PMBT neuron labels')
    p.add_argument('--shared_neurons_json', default=None,
                   help='Path to shared_neurons.json from profile stage '
                        '(auto-detected if output_dir matches)')

    # Architecture
    p.add_argument('--model_type', required=True,
                   choices=['llava-ov', 'qwen2vl', 'internvl',
                            'llava-hf', 'llava-liuhaotian', 'llava-llama3'],
                   help='Model type (determines weight path naming)')
    p.add_argument('--n_layers', type=int, default=None,
                   help='Number of transformer layers (auto-detected if None)')

    # Profile stage params
    p.add_argument('--n_prompts', type=int, default=100,
                   help='Number of reasoning prompts for shared neuron profiling')
    p.add_argument('--top_k_pct', type=float, default=0.10,
                   help='Top-K%% of neurons to consider "active" per layer '
                        '(default: 0.10 = top 10%%)')
    p.add_argument('--min_sample_freq', type=float, default=0.3,
                   help='Minimum fraction of prompts a neuron must be active in '
                        'to be considered consistently active (default: 0.3)')

    # Merge stage params
    p.add_argument('--svd_rank', type=int, default=16,
                   help='SVD truncation rank for low-rank delta (default: 16)')
    p.add_argument('--beta', type=float, default=0.5,
                   help='Injection strength (default: 0.5)')
    p.add_argument('--include_multimodal', action='store_true',
                   help='Include multimodal neurons in the PMBT mask '
                        '(default: text-only)')
    p.add_argument('--include_random', action='store_true',
                   help='Also run random mask (same count as intersection, sparsity control)')
    p.add_argument('--random_seed', type=int, default=42,
                   help='Seed for random mask generation')
    p.add_argument('--save_model', action='store_true', default=True,
                   help='Save the merged model to disk')

    # Output
    p.add_argument('--output_dir', required=True,
                   help='Output directory for shared neurons JSON and merged model')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Weight path helpers (duplicated from neuron_weight_merge.py
#   to keep this script self-contained on the cluster)
# ═══════════════════════════════════════════════════════════════════

def get_mlp_weight_paths(model_type, layer_idx):
    """Return (gate, up, down) weight paths for one VLM layer."""
    # model_type → VLM weight paths with language_model prefix
    if model_type in ('llava-ov', 'qwen2vl', 'llava-hf', 'llava-liuhaotian', 'llava-llama3'):
        prefix = f'model.language_model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')
    elif model_type == 'internvl':
        prefix = f'language_model.model.layers.{layer_idx}.feed_forward'
        return (f'{prefix}.w1.weight',
                f'{prefix}.w3.weight',
                f'{prefix}.w2.weight')
    else:
        prefix = f'model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')


def get_llm_mlp_weight_paths(layer_idx, model_type='qwen2vl'):
    """Return (gate, up, down) weight paths for a standalone LLM."""
    # standalone LLMs use simpler paths without language_model prefix
    if model_type == 'internvl':
        prefix = f'model.layers.{layer_idx}.feed_forward'
        return (f'{prefix}.w1.weight',
                f'{prefix}.w3.weight',
                f'{prefix}.w2.weight')
    else:
        prefix = f'model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')


# ═══════════════════════════════════════════════════════════════════
# Section 3 — PMBT label loading (duplicated from neuron_weight_merge.py)
# ═══════════════════════════════════════════════════════════════════

def load_pmbt_labels(label_dir, taxonomy='pmbt'):
    """Load PMBT neuron labels. Returns (labels_by_layer, n_layers, n_neurons)."""
    label_filename = ('neuron_labels_permutation.json' if taxonomy == 'pmbt'
                      else 'neuron_labels.json')
    # try merged file first, then per-layer
    merged_path = os.path.join(
        label_dir, label_filename.replace('.json', '_all.json'))
    labels_by_layer = {}
    if os.path.isfile(merged_path):
        print(f'  Loading merged PMBT labels from {merged_path}')
        with open(merged_path) as f:
            raw = json.load(f)
        labels_by_layer = {int(k): v for k, v in raw.items()}
    else:
        print(f'  Scanning {label_dir} for per-layer label files...')
        for entry in sorted(os.listdir(label_dir)):
            full = os.path.join(label_dir, entry)
            if not os.path.isdir(full):
                continue
            fpath = os.path.join(full, label_filename)
            if not os.path.isfile(fpath):
                continue
            m = re.search(r'layers?[._](\d+)', entry)
            if not m:
                m = re.search(r'(\d+)', entry)
            if m:
                layer_idx = int(m.group(1))
                with open(fpath) as f:
                    labels_by_layer[layer_idx] = json.load(f)
    if not labels_by_layer:
        raise FileNotFoundError(
            f'No PMBT labels found in {label_dir}. Run steps 1-4 first.')
    n_layers = max(labels_by_layer.keys()) + 1
    sample = labels_by_layer[min(labels_by_layer.keys())]
    n_neurons = len(sample)
    print(f'  Loaded PMBT labels: {n_layers} layers × {n_neurons} neurons')
    return labels_by_layer, n_layers, n_neurons


def build_text_mask(labels_by_layer, n_layers, n_neurons,
                    include_multimodal=False):
    """Build per-layer boolean masks for text (+ optionally multimodal) neurons.

    Returns dict {layer_idx: BoolTensor (n_neurons,)}.
    """
    masks = {}
    for layer_idx in range(n_layers):
        mask = torch.zeros(n_neurons, dtype=torch.bool)
        if layer_idx not in labels_by_layer:
            masks[layer_idx] = mask
            continue
        for entry in labels_by_layer[layer_idx]:
            idx = entry['neuron_idx']
            if idx >= n_neurons:
                continue
            label = entry.get('label', 'unknown')
            # include text neurons, and optionally multimodal
            if label == 'text':
                mask[idx] = True
            elif label == 'multimodal' and include_multimodal:
                mask[idx] = True
        masks[layer_idx] = mask
    total = sum(m.sum().item() for m in masks.values())
    print(f'  PMBT text mask: {total:,} neurons '
          f'({100 * total / (n_layers * n_neurons):.1f}%)')
    return masks


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Shared neuron profiling (Stage 1)
#   For each model (math LLM + VLM), run on reasoning prompts,
#   record top-K% activated gate neurons per layer via hooks.
#   Intersection = shared neurons.
# ═══════════════════════════════════════════════════════════════════

# Minimal GSM8K-style reasoning prompts for profiling
# (we only need the forward pass activations, not correct answers)
REASONING_PROMPTS = [
    "Solve step by step: If a store sells 240 apples in 6 days, how many apples does it sell per day?",
    "Solve step by step: A rectangle has a length of 12 cm and a width of 8 cm. What is the area?",
    "Solve step by step: If 3x + 7 = 22, what is x?",
    "Solve step by step: A train travels 180 km in 3 hours. What is its speed in km/h?",
    "Solve step by step: If you have 5 bags with 12 marbles each, how many marbles do you have?",
    "Solve step by step: The sum of two numbers is 50 and their difference is 10. Find the numbers.",
    "Solve step by step: A circle has radius 7. What is its area? Use pi = 3.14.",
    "Solve step by step: If 15% of a number is 45, what is the number?",
    "Solve step by step: How many ways can you arrange 4 books on a shelf?",
    "Solve step by step: A car uses 8 liters of fuel per 100km. How much fuel for 350km?",
]


def load_gsm8k_prompts(n_prompts):
    """Load GSM8K prompts for shared neuron profiling.

    Tries to load from the datasets library first (train split),
    falls back to built-in minimal prompts if unavailable.
    """
    prompts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for i, row in enumerate(ds):
            if i >= n_prompts:
                break
            prompts.append(f"Solve step by step: {row['question']}")
        print(f'  Loaded {len(prompts)} prompts from GSM8K train split')
    except Exception as e:
        print(f'  GSM8K dataset not available ({e}), using built-in prompts')
        # repeat built-in prompts to reach n_prompts
        while len(prompts) < n_prompts:
            prompts.extend(REASONING_PROMPTS)
        prompts = prompts[:n_prompts]
        print(f'  Using {len(prompts)} built-in reasoning prompts')
    return prompts


def _load_vlm_auto(model_path, model_type=None, device_map='auto', cpu_only=False):
    """Load VLM with the right class based on model_type, with fallback chain."""
    import torch
    _dmap = 'cpu' if cpu_only else device_map
    _kwargs = dict(torch_dtype=torch.float16, device_map=_dmap,
                   trust_remote_code=True, low_cpu_mem_usage=True)
    if model_type in ('llava-liuhaotian', 'llava-hf'):
        from transformers import LlavaForConditionalGeneration
        return LlavaForConditionalGeneration.from_pretrained(model_path, **_kwargs)
    elif model_type == 'llava-llama3':
        from transformers import LlavaNextForConditionalGeneration
        return LlavaNextForConditionalGeneration.from_pretrained(model_path, **_kwargs)
    elif model_type == 'llava-ov':
        from transformers import LlavaOnevisionForConditionalGeneration
        return LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, **_kwargs)
    elif model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(model_path, **_kwargs)
    else:
        # internvl and others
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_path, **_kwargs)


def profile_active_neurons(model_path, prompts, top_k_pct, min_freq,
                           is_vlm=False, model_type=None):
    """Profile which FFN neurons are consistently top-K% active on reasoning.

    Loads model, runs each prompt, records gate activation magnitudes,
    identifies neurons in top-K% per prompt, then retains neurons active
    in >= min_freq fraction of prompts.

    Returns:
        active_neurons: dict {layer_idx (int): set of neuron indices}
        n_layers: int
        n_neurons: int
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'\n  Loading model: {model_path}')
    # load model — for VLMs we load full model for state dict inspection,
    # then extract the LLM backbone for text-only profiling
    if is_vlm:
        full_model = _load_vlm_auto(model_path, model_type=model_type)
        # Extract LLM backbone — works for LLaVA, Qwen2VL, InternVL
        if hasattr(full_model, 'language_model'):
            model = full_model.language_model
            print(f'  Extracted language_model backbone for profiling')
        else:
            model = full_model
            print(f'  WARNING: no language_model attribute, using full model')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model.eval()

    # detect architecture
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None)
    if n_layers is None:
        # Try text_config for VLMs
        text_cfg = getattr(cfg, 'text_config', None)
        if text_cfg:
            n_layers = getattr(text_cfg, 'num_hidden_layers', None)
    if n_layers is None:
        n_layers = getattr(cfg, 'num_layers', 32)
    # detect d_ffn from first LLM layer (skip vision encoder weights)
    sd = model.state_dict()
    d_ffn = None
    for k, v in sd.items():
        # Only match LLM backbone weights, not vision encoder
        if ('vision' in k or 'encoder' in k or 'visual' in k):
            continue
        if 'gate_proj.weight' in k or '.w1.weight' in k:
            d_ffn = v.shape[0]
            break
    if d_ffn is None:
        raise RuntimeError('Could not detect d_ffn from model state dict')
    print(f'  Architecture: {n_layers} layers × {d_ffn} neurons')

    # identify MLP gate module paths for hooking
    gate_modules = {}  # layer_idx → module
    for name, module in model.named_modules():
        # match patterns like: model.layers.0.mlp.gate_proj
        # or: model.layers.0.feed_forward.w1
        m = re.search(r'layers\.(\d+)\.(?:mlp\.gate_proj|feed_forward\.w1)$',
                       name)
        if m:
            layer_idx = int(m.group(1))
            gate_modules[layer_idx] = module

    if not gate_modules:
        raise RuntimeError(
            f'No gate_proj modules found. Model structure may be unsupported.')
    print(f'  Found {len(gate_modules)} hookable gate modules')

    # per-layer activation counters: how many times each neuron is in top-K
    top_k_count = {l: torch.zeros(d_ffn, dtype=torch.long)
                   for l in gate_modules}
    n_prompts_run = 0
    k = max(1, int(d_ffn * top_k_pct))  # number of neurons in top-K%

    # hook function: records top-K neuron indices from gate output
    current_activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, d_ffn)
            # take max activation across sequence positions
            act = output.detach().float()
            max_act = act.abs().max(dim=1).values  # (batch, d_ffn)
            # get top-K neuron indices
            _, topk_idx = max_act.topk(k, dim=-1)  # (batch, k)
            current_activations[layer_idx] = topk_idx[0].cpu()
        return hook_fn

    # register hooks
    handles = []
    for layer_idx, module in gate_modules.items():
        h = module.register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    # run prompts
    print(f'  Profiling on {len(prompts)} prompts (top-K%={top_k_pct})...')
    for prompt in tqdm(prompts, desc='  Profiling'):
        current_activations.clear()
        inputs = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        # accumulate top-K counts
        for layer_idx, topk_idx in current_activations.items():
            top_k_count[layer_idx][topk_idx] += 1
        n_prompts_run += 1

    # remove hooks
    for h in handles:
        h.remove()

    # identify consistently active neurons (active in >= min_freq of prompts)
    threshold = int(n_prompts_run * min_freq)
    active_neurons = {}
    total_active = 0
    for layer_idx in sorted(top_k_count.keys()):
        counts = top_k_count[layer_idx]
        active = set(torch.where(counts >= threshold)[0].tolist())
        active_neurons[layer_idx] = active
        total_active += len(active)
    print(f'  Active neurons (freq >= {min_freq}): '
          f'{total_active:,} across {len(active_neurons)} layers')

    # free GPU memory
    del model
    torch.cuda.empty_cache()

    return active_neurons, n_layers, d_ffn


def compute_shared_neurons(active_vlm, active_llm, n_layers):
    """Compute intersection of active neurons from VLM and math LLM.

    Returns:
        shared: dict {layer_idx: sorted list of neuron indices}
        stats: dict with summary statistics
    """
    shared = {}
    total_shared = 0
    total_vlm = 0
    total_llm = 0
    for layer_idx in range(n_layers):
        vlm_set = active_vlm.get(layer_idx, set())
        llm_set = active_llm.get(layer_idx, set())
        intersection = vlm_set & llm_set
        shared[layer_idx] = sorted(intersection)
        total_shared += len(intersection)
        total_vlm += len(vlm_set)
        total_llm += len(llm_set)

    overlap_pct = (100 * total_shared / max(total_vlm, 1))
    print(f'  Shared neurons: {total_shared:,} '
          f'(VLM: {total_vlm:,}, LLM: {total_llm:,}, '
          f'overlap: {overlap_pct:.1f}%)')

    stats = {
        'total_shared': total_shared,
        'total_vlm_active': total_vlm,
        'total_llm_active': total_llm,
        'overlap_pct': overlap_pct,
    }
    return shared, stats


def run_profile_stage(args):
    """Stage 1: Profile shared neurons between math LLM and VLM."""
    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_gsm8k_prompts(args.n_prompts)

    # Profile math LLM
    print('\n' + '=' * 60)
    print('  Profiling math LLM active neurons')
    print('=' * 60)
    active_llm, n_layers_llm, d_ffn_llm = profile_active_neurons(
        args.math_llm_path, prompts, args.top_k_pct, args.min_sample_freq)

    # Profile VLM (using its LLM backbone)
    # For VLMs, we load as CausalLM to get only the text backbone
    # The VLM wrapper adds vision encoder but we want backbone activations
    print('\n' + '=' * 60)
    print('  Profiling VLM backbone active neurons')
    print('=' * 60)
    active_vlm, n_layers_vlm, d_ffn_vlm = profile_active_neurons(
        args.vlm_path, prompts, args.top_k_pct, args.min_sample_freq,
        is_vlm=True, model_type=args.model_type)

    # Compute intersection
    n_layers = min(n_layers_llm, n_layers_vlm)
    shared, stats = compute_shared_neurons(active_vlm, active_llm, n_layers)

    # Save shared neurons JSON
    # convert sets to lists for JSON serialisation
    shared_json = {str(k): v for k, v in shared.items()}
    out_path = os.path.join(args.output_dir, 'shared_neurons.json')
    with open(out_path, 'w') as f:
        json.dump(shared_json, f, indent=2)
    print(f'\n  Saved shared neurons → {out_path}')

    # Save stats
    stats['n_layers'] = n_layers
    stats['d_ffn'] = d_ffn_llm
    stats['n_prompts'] = len(prompts)
    stats['top_k_pct'] = args.top_k_pct
    stats['min_sample_freq'] = args.min_sample_freq
    stats['vlm_path'] = args.vlm_path
    stats['math_llm_path'] = args.math_llm_path
    stats_path = os.path.join(args.output_dir, 'profile_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  Saved profile stats → {stats_path}')

    return out_path


# ═══════════════════════════════════════════════════════════════════
# Section 5 — SVD low-rank merge (Stage 2)
#   Adapted from SNRF's merge_smart.py lora_stable method.
#   Key difference: mask is PMBT text neurons ∩ shared neurons.
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def svd_truncate(delta, rank):
    """Truncate a weight delta matrix to rank-r via SVD.

    delta: tensor of shape (out_features, in_features)
    rank:  target rank (e.g., 16)

    Returns: rank-r approximation of delta (same shape).
    """
    # SVD in float32 for numerical stability
    U, S, Vt = torch.linalg.svd(delta.float(), full_matrices=False)
    r = min(rank, len(S))
    # reconstruct with top-r singular values
    approx = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]
    return approx.to(delta.dtype)


def run_merge_stage(args):
    """Stage 2: SVD low-rank injection at PMBT ∩ shared neuron positions."""
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load shared neurons ──────────────────────────────────────
    shared_path = args.shared_neurons_json
    if shared_path is None:
        shared_path = os.path.join(args.output_dir, 'shared_neurons.json')
    if not os.path.isfile(shared_path):
        raise FileNotFoundError(
            f'Shared neurons not found at {shared_path}. '
            f'Run --stage profile first.')
    print(f'  Loading shared neurons from {shared_path}')
    with open(shared_path) as f:
        shared_raw = json.load(f)
    # convert to {int: set}
    shared_neurons = {int(k): set(v) for k, v in shared_raw.items()}
    total_shared = sum(len(v) for v in shared_neurons.values())
    print(f'  Shared neurons: {total_shared:,}')

    # ── Load PMBT labels and build text mask ─────────────────────
    if args.label_dir is None:
        raise ValueError('--label_dir is required for merge stage')
    labels_by_layer, n_layers, n_neurons = load_pmbt_labels(args.label_dir)
    if args.n_layers is not None:
        n_layers = args.n_layers
    text_masks = build_text_mask(
        labels_by_layer, n_layers, n_neurons,
        include_multimodal=args.include_multimodal)

    # ── Intersect: PMBT text ∩ shared ────────────────────────────
    intersection_masks = {}
    total_intersection = 0
    for layer_idx in range(n_layers):
        text_set = set(torch.where(text_masks[layer_idx])[0].tolist())
        shared_set = shared_neurons.get(layer_idx, set())
        inter = text_set & shared_set
        mask = torch.zeros(n_neurons, dtype=torch.bool)
        for idx in inter:
            mask[idx] = True
        intersection_masks[layer_idx] = mask
        total_intersection += len(inter)

    print(f'  PMBT text ∩ shared: {total_intersection:,} neurons '
          f'({100 * total_intersection / (n_layers * n_neurons):.1f}%)')

    # ── Load model state dicts ───────────────────────────────────
    print(f'\n  Loading VLM state dict: {args.vlm_path}')
    from transformers import AutoModelForCausalLM
    vlm = _load_vlm_auto(args.vlm_path, model_type=args.model_type, cpu_only=True)
    vlm_state = vlm.state_dict()
    del vlm
    torch.cuda.empty_cache()

    if args.base_llm_path is None:
        raise ValueError('--base_llm_path is required for merge stage')
    print(f'  Loading base LLM state dict: {args.base_llm_path}')
    base_llm = AutoModelForCausalLM.from_pretrained(
        args.base_llm_path, torch_dtype=torch.float16,
        device_map='cpu', trust_remote_code=True)
    base_state = base_llm.state_dict()
    del base_llm

    print(f'  Loading math LLM state dict: {args.math_llm_path}')
    math_llm = AutoModelForCausalLM.from_pretrained(
        args.math_llm_path, torch_dtype=torch.float16,
        device_map='cpu', trust_remote_code=True)
    math_state = math_llm.state_dict()
    del math_llm
    torch.cuda.empty_cache()

    # ── Per-layer: compute ΔW → SVD truncate → inject at mask ────
    n_updated = 0
    print(f'\n  Applying SVD low-rank merge (rank={args.svd_rank}, '
          f'β={args.beta})...')

    for layer_idx in tqdm(range(n_layers), desc='  Merging layers'):
        mask = intersection_masks[layer_idx]
        if not mask.any():
            continue

        # Get weight paths for VLM and standalone LLM
        g_vlm, u_vlm, d_vlm = get_mlp_weight_paths(
            args.model_type, layer_idx)
        g_llm, u_llm, d_llm = get_llm_mlp_weight_paths(
            layer_idx, args.model_type)

        # Process each projection: gate, up (rows=neurons), down (cols=neurons)
        for proj_name, vlm_key, llm_key, index_dim in [
            ('gate', g_vlm, g_llm, 0),
            ('up',   u_vlm, u_llm, 0),
            ('down', d_vlm, d_llm, 1),
        ]:
            # find actual keys (handle prefix mismatches)
            w_vlm = _find_weight(vlm_state, vlm_key, layer_idx)
            w_base = _find_weight(base_state, llm_key, layer_idx)
            w_math = _find_weight(math_state, llm_key, layer_idx)

            if w_vlm is None or w_base is None or w_math is None:
                continue

            # compute task vector: what math training changed
            delta = (w_math.float() - w_base.float())

            # SVD truncation: keep only top-r directions
            delta_lr = svd_truncate(delta, args.svd_rank)

            # inject at masked positions only
            if index_dim == 0:
                # gate/up: rows = neurons
                w_vlm_f = w_vlm.float()
                w_vlm_f[mask] += args.beta * delta_lr[mask]
                vlm_state[_resolve_key(vlm_state, vlm_key, layer_idx)] = \
                    w_vlm_f.half()
            else:
                # down: cols = neurons
                w_vlm_f = w_vlm.float()
                w_vlm_f[:, mask] += args.beta * delta_lr[:, mask]
                vlm_state[_resolve_key(vlm_state, vlm_key, layer_idx)] = \
                    w_vlm_f.half()

        n_updated += mask.sum().item()

    print(f'  Updated {n_updated:,} neuron positions across {n_layers} layers')

    # ── Save merged model ────────────────────────────────────────
    if args.save_model:
        merge_tag = f'snrf_r{args.svd_rank}_b{args.beta}'
        save_dir = os.path.join(args.output_dir, merge_tag)
        os.makedirs(save_dir, exist_ok=True)
        print(f'  Saving merged state dict → {save_dir}')

        # save state dict in shards
        torch.save(vlm_state, os.path.join(save_dir, 'pytorch_model.bin'))

        # copy tokenizer/config from original VLM
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(
            args.vlm_path, trust_remote_code=True)
        config.save_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.vlm_path, trust_remote_code=True)
        tokenizer.save_pretrained(save_dir)

        # save merge metadata
        meta = {
            'method': 'snrf_pmbt',
            'svd_rank': args.svd_rank,
            'beta': args.beta,
            'vlm_path': args.vlm_path,
            'base_llm_path': args.base_llm_path,
            'math_llm_path': args.math_llm_path,
            'n_neurons_updated': n_updated,
            'total_intersection': total_intersection,
            'total_shared': total_shared,
            'include_multimodal': args.include_multimodal,
        }
        with open(os.path.join(save_dir, 'merge_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'  ✓ Merged model saved to {save_dir}')

    # ── Optional: random baseline (same neuron count, random positions) ──
    if args.include_random:
        import random as _rng
        _rng.seed(args.random_seed)
        print(f'\n  ── Random baseline (count-matched sparsity control) ──')
        # Build random mask with same count as intersection
        random_masks = {l: torch.zeros(n_neurons, dtype=torch.bool) for l in range(n_layers)}
        all_positions = [(l, n) for l in range(n_layers) for n in range(n_neurons)]
        selected = _rng.sample(all_positions, min(total_intersection, len(all_positions)))
        for l, n in selected:
            random_masks[l][n] = True
        n_random = sum(m.sum().item() for m in random_masks.values())
        print(f'  Random mask: {n_random:,} neurons (matched to intersection count {total_intersection:,})')

        # Reload fresh VLM state
        print(f'  Reloading fresh VLM state dict...')
        vlm_reload = _load_vlm_auto(
            args.vlm_path, model_type=args.model_type, cpu_only=True)
        vlm_state_rnd = vlm_reload.state_dict()
        del vlm_reload
        torch.cuda.empty_cache()

        n_updated_rnd = 0
        print(f'  Applying SVD low-rank merge with random mask...')
        for layer_idx in tqdm(range(n_layers), desc='  Random merge'):
            mask = random_masks[layer_idx]
            if not mask.any():
                continue
            g_vlm, u_vlm, d_vlm = get_mlp_weight_paths(args.model_type, layer_idx)
            g_llm, u_llm, d_llm = get_llm_mlp_weight_paths(layer_idx, args.model_type)
            for proj_name, vlm_key, llm_key, index_dim in [
                ('gate', g_vlm, g_llm, 0),
                ('up',   u_vlm, u_llm, 0),
                ('down', d_vlm, d_llm, 1),
            ]:
                w_vlm = _find_weight(vlm_state_rnd, vlm_key, layer_idx)
                w_base = _find_weight(base_state, llm_key, layer_idx)
                w_math = _find_weight(math_state, llm_key, layer_idx)
                if w_vlm is None or w_base is None or w_math is None:
                    continue
                delta = (w_math.float() - w_base.float())
                delta_lr = svd_truncate(delta, args.svd_rank)
                if index_dim == 0:
                    w_vlm_f = w_vlm.float()
                    w_vlm_f[mask] += args.beta * delta_lr[mask]
                    vlm_state_rnd[_resolve_key(vlm_state_rnd, vlm_key, layer_idx)] = w_vlm_f.half()
                else:
                    w_vlm_f = w_vlm.float()
                    w_vlm_f[:, mask] += args.beta * delta_lr[:, mask]
                    vlm_state_rnd[_resolve_key(vlm_state_rnd, vlm_key, layer_idx)] = w_vlm_f.half()
            n_updated_rnd += mask.sum().item()

        if args.save_model:
            rnd_tag = f'snrf_random_r{args.svd_rank}_b{args.beta}'
            rnd_dir = os.path.join(args.output_dir, rnd_tag)
            os.makedirs(rnd_dir, exist_ok=True)
            torch.save(vlm_state_rnd, os.path.join(rnd_dir, 'pytorch_model.bin'))
            config.save_pretrained(rnd_dir)
            tokenizer.save_pretrained(rnd_dir)
            rnd_meta = {**meta, 'method': 'snrf_random', 'n_neurons_updated': n_updated_rnd}
            with open(os.path.join(rnd_dir, 'merge_meta.json'), 'w') as f:
                json.dump(rnd_meta, f, indent=2)
            print(f'  ✓ Random baseline saved to {rnd_dir}')
        del vlm_state_rnd

        return save_dir

    return save_dir if args.save_model else None


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Utility: find weight in state dict (handles prefix mismatches)
# ═══════════════════════════════════════════════════════════════════

def _find_weight(state_dict, key, layer_idx):
    """Find a weight tensor in state dict, trying suffix matching if needed."""
    if key in state_dict:
        return state_dict[key]
    # try suffix-based search for prefix mismatches between VLM and LLM
    suffix = key.split('.')[-2] + '.' + key.split('.')[-1]
    for k in state_dict:
        # Skip vision encoder weights to avoid dimension mismatches
        if 'vision' in k or 'encoder' in k or 'visual' in k:
            continue
        if k.endswith(suffix) and f'.{layer_idx}.' in k:
            return state_dict[k]
    return None


def _resolve_key(state_dict, key, layer_idx):
    """Resolve the actual key name in state dict."""
    if key in state_dict:
        return key
    suffix = key.split('.')[-2] + '.' + key.split('.')[-1]
    for k in state_dict:
        if 'vision' in k or 'encoder' in k or 'visual' in k:
            continue
        if k.endswith(suffix) and f'.{layer_idx}.' in k:
            return k
    return key


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()

    print('\n' + '═' * 60)
    print('  SNRF + PMBT: Shared Neuron Low-Rank Fusion (Layer 1b)')
    print('═' * 60)
    print(f'  Stage:     {args.stage}')
    print(f'  VLM:       {args.vlm_path}')
    print(f'  Math LLM:  {args.math_llm_path}')
    print(f'  Output:    {args.output_dir}')

    if args.stage in ('profile', 'both'):
        shared_path = run_profile_stage(args)
        if args.stage == 'both':
            args.shared_neurons_json = shared_path

    if args.stage in ('merge', 'both'):
        run_merge_stage(args)

    elapsed = time.time() - t0
    print(f'\n  Total time: {elapsed / 60:.1f} min')


if __name__ == '__main__':
    main()