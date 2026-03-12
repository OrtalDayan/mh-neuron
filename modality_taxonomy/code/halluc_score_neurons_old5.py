"""
hallucination_taxonomy.py — Full enrichment analysis across ALL neuron modality categories

Tests whether visual, text, multimodal, unknown, and random neuron sets
are disproportionately hallucination-driving, using Fisher's exact test
and bootstrap confidence intervals.

Pipeline:
  Phase 0 — (Optional) Contrastive POPE preprocessing (--contrastive).
             Loads 3 POPE splits (random, popular, adversarial), samples
             N responses per question at temperature=0.7, keeps only
             questions where the model is consistently correct or
             consistently wrong across all samples. Caps at 333 per
             split (~1000 total). Produces a high-confidence contrastive
             set that replaces the raw POPE data for Phase 1.

  Phase 1 — Identify hallucination-driving neurons via per-neuron ablation
             on POPE (Polling-based Object Probing Evaluation).
             Each neuron is zeroed out, and we measure the change in
             hallucination rate (ΔHallucination = ablated − baseline).
             Neurons with the largest positive ΔH are hallucination-driving.
             ** Multi-GPU: layers sharded across available GPUs **

  Phase 2 — Enrichment analysis: for each modality category
             (visual, text, multimodal, unknown, random), test whether
             that category is over-represented among hallucination-driving
             neurons compared to chance expectation (Fisher's exact test).

  Phase 3 — Visualisation: enrichment bar plots, odds-ratio forest plot,
             per-layer enrichment heatmap.

Inputs:
  --label_dir       : directory with neuron_labels.json or neuron_labels_permutation.json per layer
  --pope_path       : path to POPE evaluation JSON (coco_pope_random.json)
  --pope_img_dir    : directory with COCO val images
  --model_type      : llava-hf | llava-liuhaotian | internvl | qwen2vl | llava-ov
  --model_path      : HF model ID or local path
  --taxonomy         : ft | pmbt — which classification to use
  --top_k_pct       : top K% of neurons by ΔH to define "hallucination-driving"
  --n_pope_images   : number of POPE questions to evaluate per ablation
  --output_dir      : where to save results

Multi-GPU:
  Uses torch.multiprocessing to spawn one worker per available GPU.
  Each worker processes a disjoint subset of layers.
  Phase 2 and 3 are CPU-only and run after all GPU workers finish.

Usage:
    # Full analysis (auto-detect GPUs)
    python hallucination_taxonomy.py \
        --label_dir results/llava-1.5-7b/llm_permutation \
        --pope_path data/POPE/output/coco/coco_pope_random.json \
        --pope_img_dir data/val2014 \
        --model_path liuhaotian/llava-v1.5-7b \
        --model_type llava-liuhaotian

    # Restrict to 4 GPUs
    python hallucination_taxonomy.py --n_gpus 4 ...

    # Phase 2 only (skip ablation, load pre-computed scores)
    python hallucination_taxonomy.py --skip_ablation --ablation_scores results/ablation_scores.json ...
"""

import argparse                     # Line 1: parse command-line arguments
import json                         # Line 2: read/write JSON files for labels, POPE data, results
import os                           # Line 3: file path manipulation and directory creation
import sys                          # Line 4: modify Python import path for LLaVA repo
import time                         # Line 5: timing each phase of the pipeline
from collections import defaultdict # Line 6: default dictionaries for aggregating per-layer stats

import numpy as np                  # Line 7: numerical operations on activation arrays

# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Argument parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse all command-line arguments for the hallucination taxonomy pipeline.

    Groups: model config, data paths, ablation settings, enrichment settings,
            GPU config, output config.
    """
    p = argparse.ArgumentParser(
        description='Hallucination Taxonomy: enrichment analysis across '   # Line 1: description for --help
                    'ALL neuron modality categories (visual, text, '
                    'multimodal, unknown, random)')

    # ── Model configuration ───────────────────────────────────────
    p.add_argument('--model_type', default='llava-liuhaotian',              # Line 2: which model backend to use
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl',
                            'qwen2vl', 'llava-ov'],
                   help='Model backend (determines layer naming + loading)')
    p.add_argument('--model_path', default='liuhaotian/llava-v1.5-7b',     # Line 3: HF model ID or local path
                   help='HuggingFace model ID or local path to weights')
    p.add_argument('--model_name', default='llava-1.5-7b',                 # Line 4: short name for output dirs
                   help='Short model name for output directory naming')
    p.add_argument('--n_layers', type=int, default=32,                     # Line 5: total LLM layers in model
                   help='Total number of LLM layers (32 for LLaMA-2, 28 for Qwen2)')
    p.add_argument('--n_neurons', type=int, default=11008,                 # Line 6: neurons per FFN layer
                   help='Neurons per layer (11008 for LLaMA-2, 14336 for InternLM2)')

    # ── Data paths ────────────────────────────────────────────────
    p.add_argument('--label_dir', default=None,                             # Line 7: directory with classification results
                   help='Directory containing per-layer neuron label JSONs '
                        '(e.g. results/llava-1.5-7b/llm_permutation)')
    p.add_argument('--taxonomy', default='pmbt',                           # Line 8: use permutation-test labels (statistically principled)
                   help='Taxonomy to use: pmbt = permutation-test labels (default)')
    p.add_argument('--pope_path',                                          # Line 9: path to POPE evaluation data
                   default='data/POPE/output/coco/coco_pope_random.json',
                   help='Path to POPE evaluation JSONL')
    p.add_argument('--pope_img_dir', default='data/val2014',               # Line 10: COCO val images directory
                   help='Directory containing COCO val2014 images')
    p.add_argument('--coco_img_dir',                                       # Line 11: COCO train images for descriptions
                   default='/home/projects/bagon/shared/coco2017/images/train2017/',
                   help='COCO train2017 images directory')
    p.add_argument('--detail_23k_path',                                    # Line 12: detail_23k image list
                   default='data/detail_23k.json',
                   help='Path to detail_23k.json')
    p.add_argument('--generated_desc_path',                                # Line 13: LLaVA-generated descriptions
                   default='results/1-describe/generated_descriptions.json',
                   help='Path to generated descriptions JSON')

    # ── Ablation settings (Phase 1) ───────────────────────────────
    p.add_argument('--skip_ablation', action='store_true',                 # Line 14: skip Phase 1 if scores exist
                   help='Skip ablation; load pre-computed scores from --ablation_scores')
    p.add_argument('--ablation_scores', default=None,                      # Line 15: path to pre-computed ablation scores
                   help='Path to pre-computed ablation scores JSON (used with --skip_ablation)')
    p.add_argument('--n_pope_questions', type=int, default=500,            # Line 16: POPE questions per ablation eval
                   help='Number of POPE questions per ablation evaluation')
    p.add_argument('--contrastive', action='store_true',                   # Line 16b: enable contrastive filtering
                   help='Build contrastive POPE set by sampling multiple '
                        'responses per question and keeping only consistently '
                        'correct or consistently incorrect cases')
    p.add_argument('--contrastive_samples', type=int, default=10,          # Line 16c: samples per question
                   help='Number of responses to sample per POPE question '
                        'for contrastive filtering (default: 10)')
    p.add_argument('--contrastive_cap_per_split', type=int, default=333,   # Line 16d: max questions per split
                   help='Max clean questions to keep per POPE split '
                        '(default: 333, giving ~1000 total across 3 splits)')
    p.add_argument('--contrastive_start_per_split', type=int, default=1250, # Line 16e: starting questions per split
                   help='Number of POPE questions to sample from each split '
                        'before filtering (default: 1250)')
    p.add_argument('--pope_splits_dir', default=None,                      # Line 16f: directory with all 3 POPE split files
                   help='Directory containing coco_pope_random.json, '
                        'coco_pope_popular.json, coco_pope_adversarial.json. '
                        'Auto-detected from --pope_path parent if not set.')
    p.add_argument('--ablation_method', default='zero',                    # Line 17: how to ablate neurons
                   choices=['zero', 'mean'],
                   help='Ablation method: zero = set to 0, mean = set to dataset mean')
    p.add_argument('--batch_neurons', type=int, default=50,                # Line 18: ablate neurons in batches
                   help='Number of neurons to ablate simultaneously per evaluation '
                        '(grouped ablation for tractability)')

    # ── Enrichment settings (Phase 2) ─────────────────────────────
    p.add_argument('--top_k_pct', type=float, default=5.0,                 # Line 19: top K% as "hallucination-driving"
                   help='Top K%% of neurons by ΔHallucination to classify as '
                        'hallucination-driving')
    p.add_argument('--n_random_trials', type=int, default=1000,            # Line 20: random baseline repetitions
                   help='Number of random neuron sets for calibration')
    p.add_argument('--alpha', type=float, default=0.05,                    # Line 21: significance threshold
                   help='Significance level for enrichment tests')
    p.add_argument('--seed', type=int, default=42,                         # Line 22: random seed for reproducibility
                   help='Random seed')

    # ── GPU configuration ─────────────────────────────────────────
    p.add_argument('--n_gpus', type=int, default=0,                        # Line 23: number of GPUs (0 = auto-detect)
                   help='Number of GPUs to use (0 = auto-detect all available)')

    # ── Output ────────────────────────────────────────────────────
    p.add_argument('--output_dir', default='results/hallucination_taxonomy', # Line 24: output directory
                   help='Output directory for enrichment results and plots')
    p.add_argument('--plot_only', action='store_true',                      # Line 25: regenerate plots from saved results
                   help='Skip all computation, just regenerate plots from '
                        'existing enrichment_results.json and per_layer_enrichment.npy')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Load neuron modality labels
# ═══════════════════════════════════════════════════════════════════════

def get_layer_names(model_type, n_layers):
    """Return act_fn hook-point names for each LLM layer.

    Maps model_type to the correct module path prefix so we can
    attach baukit hooks at the right location in the model tree.
    """
    if model_type == 'llava-hf':                                           # Line 1: HF LLaVA wrapper
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'internvl':                                         # Line 2: InternVL2.5 architecture
        prefix = 'language_model.model.layers'
        suffix = 'feed_forward.act_fn'
    elif model_type == 'qwen2vl':                                          # Line 3: Qwen2.5-VL architecture
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'llava-ov':                                         # Line 4: LLaVA-OneVision architecture
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    else:                                                                  # Line 5: original liuhaotian LLaVA
        prefix = 'model.layers'
        suffix = 'mlp.act_fn'
    return [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]            # Line 6: list of layer hook names


def load_neuron_labels(label_dir, taxonomy, layer_names, n_layers):
    """Load per-layer neuron modality labels from the classification pipeline.

    Reads either neuron_labels.json (fixed-threshold) or
    neuron_labels_permutation.json (permutation-test) from each
    layer's subdirectory.

    Returns:
        labels: dict {layer_idx: list of label dicts}
                each label dict has at minimum {'neuron_idx': int, 'label': str}
        flat_labels: numpy array of shape (n_layers * n_neurons,) with
                     string labels for every neuron in the model
    """
    label_filename = ('neuron_labels.json' if taxonomy == 'ft'             # Line 1: select filename by taxonomy
                      else 'neuron_labels_permutation.json')

    # First try the merged "_all" file (single JSON with all layers)
    merged_name = label_filename.replace('.json', '_all.json')             # Line 2: e.g. neuron_labels_permutation_all.json
    merged_path = os.path.join(label_dir, merged_name)

    labels = {}                                                            # Line 3: {layer_idx_str: label_list}

    if os.path.isfile(merged_path):                                        # Line 4: merged file exists
        print(f'Loading merged labels from {merged_path}')
        with open(merged_path) as f:
            labels = json.load(f)                                          # Line 5: keys are layer index strings
        print(f'  Loaded {len(labels)} layers from merged file')
    else:                                                                   # Line 6: fall back to per-layer files
        print(f'Loading per-layer labels from {label_dir}')
        for l in range(n_layers):                                          # Line 7: iterate over all layers
            layer_name = layer_names[l]
            label_path = os.path.join(label_dir, layer_name, label_filename)
            if os.path.isfile(label_path):                                 # Line 8: file exists for this layer
                with open(label_path) as f:
                    labels[str(l)] = json.load(f)                          # Line 9: store under string key
            else:
                print(f'  WARNING: missing {label_path}')

    # Build flat label array: (n_total_neurons,) ordered by (layer, neuron_idx)
    flat_labels = []                                                       # Line 10: will be filled layer-by-layer
    for l in range(n_layers):                                              # Line 11: iterate in layer order
        key = str(l)
        if key in labels:                                                  # Line 12: layer data exists
            layer_labels = labels[key]
            # Sort by neuron_idx to ensure consistent ordering
            layer_labels_sorted = sorted(layer_labels,                     # Line 13: sort by neuron index
                                         key=lambda x: x['neuron_idx'])
            for entry in layer_labels_sorted:                              # Line 14: extract label strings
                flat_labels.append(entry['label'])
        else:
            print(f'  Layer {l}: no labels found, filling with "missing"')
            # Unknown number of neurons — use n_neurons from first available layer
            if labels:                                                     # Line 15: infer neuron count
                sample_key = next(iter(labels))
                n_neurons_layer = len(labels[sample_key])
            else:
                n_neurons_layer = 11008                                    # Line 16: fallback to LLaMA-2 default
            flat_labels.extend(['missing'] * n_neurons_layer)              # Line 17: placeholder for missing layers

    flat_labels = np.array(flat_labels)                                    # Line 18: convert to numpy for vectorized ops
    print(f'Loaded {len(flat_labels)} neuron labels total')

    # Print category distribution
    categories = ['visual', 'text', 'multimodal', 'unknown']              # Line 19: standard modality categories
    for cat in categories:                                                 # Line 20: print count and percentage
        count = (flat_labels == cat).sum()
        print(f'  {cat:12s}: {count:6,} ({100 * count / len(flat_labels):.1f}%)')

    return labels, flat_labels


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Phase 1: Identify hallucination-driving neurons (multi-GPU)
# ═══════════════════════════════════════════════════════════════════════

def load_pope_data(pope_path, n_questions=None):
    """Load POPE evaluation questions from JSONL.

    Each line: {"question": "...", "answer": "yes"/"no",
                "image": "COCO_val2014_000000XXXXXX.jpg", ...}

    Returns list of dicts, optionally truncated to n_questions.
    """
    questions = []                                                         # Line 1: accumulate parsed POPE entries
    with open(pope_path) as f:                                             # Line 2: open JSONL file
        for line in f:                                                     # Line 3: iterate line-by-line
            line = line.strip()
            if line:                                                       # Line 4: skip empty lines
                questions.append(json.loads(line))                         # Line 5: parse each JSON line
    if n_questions is not None and n_questions < len(questions):            # Line 6: optionally truncate
        questions = questions[:n_questions]
    print(f'Loaded {len(questions)} POPE questions from {pope_path}')
    return questions                                                       # Line 7: list of POPE question dicts


def _resolve_module(model, dotted_name):
    """Walk the model tree to find the module at a dotted attribute path.

    Example: _resolve_module(model, 'model.layers.0.mlp.act_fn')
    traverses model → model.model → model.model.layers → [0] → .mlp → .act_fn

    Args:
        model: root model object
        dotted_name: dot-separated path like 'model.layers.0.mlp.act_fn'

    Returns:
        module: the PyTorch module at the given path
    """
    parts = dotted_name.split('.')                                         # Line 1: split into path components
    mod = model                                                            # Line 2: start from root
    for p in parts:                                                        # Line 3: walk each component
        if p.isdigit():                                                    # Line 4: numeric index into ModuleList
            mod = mod[int(p)]
        else:                                                              # Line 5: named attribute
            mod = getattr(mod, p)
    return mod                                                             # Line 6: the resolved module


def _extract_down_proj_norms(model, model_type, layer_names, n_layers):
    """Extract L2 norm of each down_proj column for all layers.

    In the SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    Each neuron j's contribution to the residual stream is:
        contribution_j = activation_j * down_proj[:, j]
    The column norm ||down_proj[:, j]|| scales how much neuron j matters.

    Args:
        model: loaded VLM model
        model_type: backend string (e.g. 'llava-hf', 'internvl')
        layer_names: list of act_fn hook-point names from get_layer_names()
        n_layers: number of LLM layers

    Returns:
        norms: dict {layer_idx: tensor of shape (n_neurons,)}
    """
    import torch                                                           # Line 1: needed for detach/float/norm
    norms = {}                                                             # Line 2: output dict
    for layer_idx in range(n_layers):                                      # Line 3: iterate layers
        act_fn_path = layer_names[layer_idx]                               # Line 4: e.g. 'model.language_model.layers.0.mlp.act_fn'
        if model_type == 'internvl':                                       # Line 5: InternLM2 names down_proj as 'w2'
            dp_path = act_fn_path.replace('.act_fn', '.w2')
        else:                                                              # Line 6: all others use 'down_proj'
            dp_path = act_fn_path.replace('.act_fn', '.down_proj')
        dp_module = _resolve_module(model, dp_path)                        # Line 7: get the down_proj linear layer
        W_down = dp_module.weight.detach().float()                         # Line 8: shape (hidden_size, n_neurons)
        norms[layer_idx] = W_down.norm(dim=0).cpu()                        # Line 9: L2 norm per column → (n_neurons,)
    return norms                                                           # Line 10: {layer_idx: tensor}


def build_contrastive_pope_set(args, device='cuda:0'):
    """Build a contrastive POPE set and compute per-neuron CETT-diff scores.

    For each question, samples args.contrastive_samples responses at
    temperature=0.7. Keeps only questions where ALL samples give the
    same yes/no answer — either consistently correct or consistently
    wrong. Caps at args.contrastive_cap_per_split per POPE split.

    After filtering, computes CETT (Contribution of Each neuron To
    the hidden state) at the answer token position for each clean
    question, then computes CETT-diff = mean(CETT on hallucinating)
    − mean(CETT on correct). This provides a per-neuron hallucination
    score independent of the ablation-based ΔH scores.

    Loads all 3 POPE splits (random, popular, adversarial), samples
    args.contrastive_start_per_split from each, filters, caps, and
    saves the result to output_dir/contrastive_pope.jsonl.

    Args:
        args: parsed arguments namespace
        device: torch device string for model loading (default: 'cuda:0')

    Returns:
        contrastive_path: str — path to the saved contrastive JSONL file
        n_clean: int — total number of clean questions retained
        cett_diff: dict {(layer_idx, neuron_idx): float} — per-neuron
                   CETT-diff score (positive = higher contribution during
                   hallucination). None if computation fails.
    """
    import torch                                                           # Line 1: needed for model loading
    from PIL import Image                                                  # Line 2: image loading

    print(f'\n{"="*60}')
    print(f'CONTRASTIVE PREPROCESSING')
    print(f'  Splits: random, popular, adversarial')
    print(f'  Start per split: {args.contrastive_start_per_split}')
    print(f'  Samples per question: {args.contrastive_samples}')
    print(f'  Cap per split: {args.contrastive_cap_per_split}')
    print(f'{"="*60}\n')

    # ── Resolve POPE splits directory ──
    if args.pope_splits_dir is not None:                                   # Line 3: explicit dir
        splits_dir = args.pope_splits_dir
    else:                                                                  # Line 4: infer from pope_path parent
        splits_dir = os.path.dirname(args.pope_path)

    split_names = ['random', 'popular', 'adversarial']                     # Line 5: the 3 POPE difficulty levels
    split_files = {s: os.path.join(splits_dir, f'coco_pope_{s}.json')      # Line 6: expected filenames
                   for s in split_names}

    # Verify all split files exist
    for s, path in split_files.items():                                    # Line 7: check existence
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'POPE split file not found: {path}. '
                f'Set --pope_splits_dir to the directory containing all 3 splits.')

    # ── Load and sample from each split ──
    rng = np.random.RandomState(args.seed)                                 # Line 8: reproducible sampling
    all_candidates = []                                                    # Line 9: accumulate across splits

    for split_name in split_names:                                         # Line 10: iterate splits
        raw = load_pope_data(split_files[split_name])                      # Line 11: load full split
        if len(raw) > args.contrastive_start_per_split:                    # Line 12: subsample if needed
            indices = rng.choice(len(raw), size=args.contrastive_start_per_split,
                                 replace=False)
            raw = [raw[i] for i in sorted(indices)]                        # Line 13: keep order for reproducibility
        for q in raw:                                                      # Line 14: tag each question with its split
            q['pope_split'] = split_name
        all_candidates.append(raw)
        print(f'  {split_name}: {len(raw)} starting questions')

    # ── Load model for contrastive sampling ──
    print(f'\n  Loading model on {device} for contrastive sampling...')
    if args.model_type == 'llava-hf':
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
    elif args.model_type == 'llava-liuhaotian':
        _PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..'))
        _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')
        if _LLAVA_PATH not in sys.path:
            sys.path.insert(0, _LLAVA_PATH)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name,
            device_map=device, torch_dtype=torch.float16)
        processor = (tokenizer, image_processor)
    elif args.model_type == 'llava-ov':
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
    elif args.model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
        processor = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
    elif args.model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
    else:
        raise ValueError(f'Unsupported model_type: {args.model_type}')
    print(f'  Model loaded.')

    # ── Helper: generate one yes/no answer for a POPE question ──
    def generate_answer(q, do_sample=True, temperature=0.7):
        """Generate a single yes/no answer for a POPE question.

        Args:
            q: POPE question dict with 'image' and 'text'/'question' keys
            do_sample: if True, sample with temperature; if False, greedy
            temperature: sampling temperature (only used if do_sample=True)

        Returns:
            pred_yes: bool — True if model answered 'yes'
            success: bool — True if generation succeeded
        """
        img_filename = q['image']
        img_path = os.path.join(args.pope_img_dir, img_filename)
        question_text = q.get('text', q.get('question', ''))

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            return False, False                                            # failed to load image

        gen_kwargs = dict(max_new_tokens=10,
                          do_sample=do_sample,
                          temperature=temperature if do_sample else None)
        if not do_sample:
            gen_kwargs.pop('temperature', None)

        with torch.no_grad():
            if args.model_type == 'llava-hf':
                prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
                inputs = processor(text=prompt, images=img,
                                   return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, **gen_kwargs)
                generated = processor.decode(output_ids[0],
                                             skip_special_tokens=True)
            elif args.model_type == 'llava-liuhaotian':
                from llava.conversation import conv_templates
                from llava.mm_utils import (tokenizer_image_token,
                                            process_images)
                conv = conv_templates['v1'].copy()
                conv.append_message(conv.roles[0], f'<image>\n{question_text}')
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                tok = processor[0] if isinstance(processor, tuple) else processor
                img_proc = processor[1] if isinstance(processor, tuple) else None
                input_ids = tokenizer_image_token(
                    prompt, tok, return_tensors='pt').unsqueeze(0).to(device)
                img_tensor = process_images(
                    [img], img_proc, model.config
                ).to(device, dtype=torch.float16)
                output_ids = model.generate(
                    input_ids=input_ids, images=img_tensor, **gen_kwargs)
                generated = tok.decode(output_ids[0],
                                       skip_special_tokens=True)
            elif args.model_type == 'llava-ov':
                messages = [{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': question_text}]}]
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(images=img, text=prompt_text,
                                   return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, **gen_kwargs)
                prompt_len = inputs['input_ids'].shape[1]
                generated = processor.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True)
            elif args.model_type == 'internvl':
                import torchvision.transforms as T
                from torchvision.transforms.functional import InterpolationMode
                _tf = T.Compose([
                    T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                ])
                pixel_values = _tf(img).unsqueeze(0).to(
                    torch.bfloat16).to(device)
                question_prompt = f'<image>\n{question_text}'
                gen_config = dict(max_new_tokens=10, do_sample=do_sample)
                if do_sample:
                    gen_config['temperature'] = temperature
                response = model.chat(processor, pixel_values,
                                      question_prompt, gen_config)
                generated = response
            elif args.model_type == 'qwen2vl':
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': question_text}]}]
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(images=img, text=prompt_text,
                                   return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, **gen_kwargs)
                prompt_len = inputs['input_ids'].shape[1]
                generated = processor.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True)
            else:
                return False, False

        answer = generated.strip().lower()
        pred_yes = 'yes' in answer
        return pred_yes, True

    # ── Run contrastive sampling ──
    from tqdm import tqdm                                                  # Line 15: progress bar

    contrastive_questions = []                                             # Line 16: accumulate clean questions
    stats = {s: {'total': 0, 'consistent_correct': 0,                     # Line 17: per-split stats
                 'consistent_halluc': 0, 'inconsistent': 0}
             for s in split_names}

    for split_questions in all_candidates:                                 # Line 18: iterate each split's candidates
        split_name = split_questions[0]['pope_split']                       # Line 19: get split name from first q
        split_clean = []                                                   # Line 20: clean questions for this split

        for q in tqdm(split_questions,                                     # Line 21: iterate questions with progress
                       desc=f'Contrastive [{split_name}]'):
            gt_answer = q.get('label', q.get('answer', '')).strip().lower()  # Line 22: ground truth
            gt_yes = gt_answer == 'yes'                                    # Line 23: bool ground truth

            # Sample multiple responses
            predictions = []                                               # Line 24: collect per-sample predictions
            all_succeeded = True                                           # Line 25: track failures
            for _ in range(args.contrastive_samples):                      # Line 26: sample N times
                pred_yes, success = generate_answer(q, do_sample=True,
                                                     temperature=0.7)
                if not success:                                            # Line 27: image load failed
                    all_succeeded = False
                    break
                predictions.append(pred_yes)                               # Line 28: record this sample's answer

            if not all_succeeded:                                          # Line 29: skip broken images
                continue

            stats[split_name]['total'] += 1                                # Line 30: count evaluated

            # Check consistency: all predictions must agree
            all_yes = all(predictions)                                     # Line 31: all said yes
            all_no = all(not p for p in predictions)                       # Line 32: all said no

            if not (all_yes or all_no):                                    # Line 33: inconsistent — discard
                stats[split_name]['inconsistent'] += 1
                continue

            # Determine contrastive label
            pred_yes_consistent = all_yes                                  # Line 34: consistent prediction
            is_correct = (pred_yes_consistent == gt_yes)                   # Line 35: matches ground truth?

            if is_correct:                                                 # Line 36: consistently correct
                stats[split_name]['consistent_correct'] += 1
                q['contrastive_label'] = 'faithful'
            else:                                                          # Line 37: consistently wrong
                stats[split_name]['consistent_halluc'] += 1
                q['contrastive_label'] = 'hallucinated'

            split_clean.append(q)                                          # Line 38: add to clean set

        # Cap this split
        if len(split_clean) > args.contrastive_cap_per_split:             # Line 39: cap if too many
            rng.shuffle(split_clean)                                       # Line 40: shuffle before capping
            split_clean = split_clean[:args.contrastive_cap_per_split]

        contrastive_questions.extend(split_clean)                          # Line 41: add to global set
        print(f'  {split_name}: {stats[split_name]["total"]} evaluated → '
              f'{stats[split_name]["consistent_correct"]} correct, '
              f'{stats[split_name]["consistent_halluc"]} hallucinated, '
              f'{stats[split_name]["inconsistent"]} inconsistent → '
              f'{len(split_clean)} kept (capped at {args.contrastive_cap_per_split})')

    # ── Save contrastive set ──
    contrastive_path = os.path.join(args.output_dir,                       # Line 42: output path
                                     'contrastive_pope.jsonl')
    with open(contrastive_path, 'w') as f:                                 # Line 43: write JSONL
        for q in contrastive_questions:
            f.write(json.dumps(q) + '\n')                                  # Line 44: one JSON object per line

    # Save contrastive stats for reproducibility
    stats_path = os.path.join(args.output_dir, 'contrastive_stats.json')   # Line 45: stats output path
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\n  Contrastive set: {len(contrastive_questions)} questions '
          f'saved to {contrastive_path}')
    print(f'  Stats saved to {stats_path}')
    print(f'\n  Paper sentence: "We sampled {args.contrastive_samples} responses '
          f'per question from {args.contrastive_start_per_split} POPE items per '
          f'split (random, popular, adversarial), retaining '
          f'{args.contrastive_cap_per_split} per split where the model was '
          f'consistently correct or consistently incorrect, yielding '
          f'N={len(contrastive_questions)} high-confidence contrastive samples '
          f'balanced across difficulty levels."')

    # ── Compute CETT-diff from contrastive set ──────────────────────
    # CETT = |activation_j| × ||down_proj_col_j|| measures each neuron's
    # actual contribution to the layer output. By comparing CETT at the
    # answer token between hallucinating and correct questions, we get a
    # per-neuron hallucination score independent of ablation-based ΔH.
    # For POPE, the answer token is always the first generated token
    # ("yes"/"no"), so no GPT-4o is needed to identify it.
    cett_diff = None                                                       # Line C1: default if computation fails
    n_layers = args.n_layers                                               # Line C2: total LLM layers
    layer_names = get_layer_names(args.model_type, n_layers)               # Line C3: act_fn hook-point names

    try:
        print(f'\n{"="*60}')
        print(f'CETT-DIFF SCORING on {len(contrastive_questions)} '
              f'contrastive questions')
        print(f'{"="*60}\n')

        # Step 1: Extract down_proj column norms (once, from weights)
        print(f'  Extracting down_proj column norms ({n_layers} layers)...')
        down_norms = _extract_down_proj_norms(                             # Line C4: ||down_proj[:, j]|| per neuron
            model, args.model_type, layer_names, n_layers)
        n_neurons = down_norms[0].shape[0]                                 # Line C5: neurons per layer
        print(f'  {n_layers} layers × {n_neurons} neurons')

        # Step 2: Register hooks on act_fn at every layer.
        # During model.generate(), the first forward pass (prefill)
        # processes the full prompt. The activations at the last prompt
        # token determine the answer. We capture only the prefill pass
        # (seq_len > 1) and take the last position.
        act_captures = {}                                                  # Line C6: {layer_idx: tensor (n_neurons,)}
        hook_recorded = {}                                                 # Line C7: {layer_idx: bool} first-pass flag
        hooks = []                                                         # Line C8: hook handles for cleanup

        for layer_idx in range(n_layers):                                  # Line C9: register one hook per layer
            act_fn_module = _resolve_module(model, layer_names[layer_idx]) # Line C10: act_fn module
            hook_recorded[layer_idx] = False                               # Line C11: reset flag

            def make_hook(lidx):                                           # Line C12: closure to capture layer index
                def hook_fn(module, input, output):
                    if hook_recorded[lidx]:                                # Line C13: skip decode steps
                        return
                    out = output[0] if isinstance(output, tuple) else output  # Line C14: handle tuple output
                    act_captures[lidx] = out[0, -1, :].detach().cpu()     # Line C15: last prompt token activations
                    hook_recorded[lidx] = True                             # Line C16: mark as recorded
                return hook_fn

            h = act_fn_module.register_forward_hook(make_hook(layer_idx))  # Line C17: attach hook
            hooks.append(h)                                                # Line C18: save handle

        # Step 3: Run greedy forward pass per question, accumulate CETT
        # Use running sums to avoid storing per-question arrays.
        halluc_sum = {l: np.zeros(n_neurons, dtype=np.float64)             # Line C19: sum of CETT for halluc questions
                      for l in range(n_layers)}
        correct_sum = {l: np.zeros(n_neurons, dtype=np.float64)            # Line C20: sum of CETT for correct questions
                       for l in range(n_layers)}
        n_halluc = 0                                                       # Line C21: count of hallucinated questions
        n_correct = 0                                                      # Line C22: count of correct questions

        for q in tqdm(contrastive_questions, desc='CETT scoring'):         # Line C23: iterate clean questions
            # Reset hook flags for this question
            for lidx in hook_recorded:                                     # Line C24: reset all layer flags
                hook_recorded[lidx] = False
            act_captures.clear()                                           # Line C25: clear previous captures

            # Greedy forward pass (deterministic — activations at last
            # prompt position are identical regardless of temperature)
            pred_yes, success = generate_answer(q, do_sample=False)        # Line C26: greedy decode, hooks fire
            if not success:                                                # Line C27: skip failed images
                continue

            # Compute CETT per neuron and accumulate
            is_halluc = q['contrastive_label'] == 'hallucinated'           # Line C28: check contrastive label
            target = halluc_sum if is_halluc else correct_sum              # Line C29: pick accumulator

            for lidx in range(n_layers):                                   # Line C30: iterate layers
                if lidx not in act_captures:                               # Line C31: skip if hook didn't fire
                    continue
                acts = act_captures[lidx].float().numpy()                  # Line C32: (n_neurons,) activation values
                norms_np = down_norms[lidx].numpy()                        # Line C33: (n_neurons,) column norms
                cett = np.abs(acts) * norms_np                             # Line C34: CETT_j = |act_j| × ||down_proj_col_j||
                target[lidx] += cett                                       # Line C35: accumulate running sum

            if is_halluc:                                                  # Line C36: increment counter
                n_halluc += 1
            else:
                n_correct += 1

        # Step 4: Remove hooks
        for h in hooks:                                                    # Line C37: clean up all hooks
            h.remove()

        # Step 5: Compute CETT-diff = mean(halluc) − mean(correct)
        print(f'\n  CETT scored: {n_halluc} hallucinated, '
              f'{n_correct} correct questions')

        if n_halluc > 0 and n_correct > 0:                                # Line C38: need both groups
            cett_diff = {}                                                 # Line C39: {(layer, neuron): diff}
            for lidx in range(n_layers):                                   # Line C40: iterate layers
                mean_h = halluc_sum[lidx] / n_halluc                       # Line C41: mean CETT for halluc group
                mean_c = correct_sum[lidx] / n_correct                     # Line C42: mean CETT for correct group
                diff = mean_h - mean_c                                     # Line C43: per-neuron CETT-diff
                for n_idx in range(n_neurons):                             # Line C44: store per-neuron scores
                    cett_diff[(lidx, n_idx)] = float(diff[n_idx])

            # Save CETT-diff scores
            cett_diff_path = os.path.join(args.output_dir,                 # Line C45: output path
                                          'cett_diff_scores.json')
            cett_serializable = {f'{k[0]}_{k[1]}': v                       # Line C46: serialize tuple keys
                                 for k, v in cett_diff.items()}
            with open(cett_diff_path, 'w') as f:                           # Line C47: write JSON
                json.dump(cett_serializable, f, indent=2)

            # Print top CETT-diff neurons as sanity check
            top_neurons = sorted(cett_diff.items(),                        # Line C48: sort by CETT-diff descending
                                 key=lambda x: x[1], reverse=True)[:10]
            print(f'  Top 10 neurons by CETT-diff (positive = hallucination-driving):')
            for (l, n), score in top_neurons:                              # Line C49: print top neurons
                print(f'    layer {l:2d} neuron {n:5d}: CETT-diff = {score:.6f}')
            print(f'  CETT-diff scores saved to {cett_diff_path}')
        else:
            print(f'  WARNING: Insufficient data for CETT-diff '
                  f'(halluc={n_halluc}, correct={n_correct})')

    except Exception as e:                                                 # Line C50: catch any CETT errors
        print(f'  WARNING: CETT-diff computation failed: {e}')
        print(f'  Contrastive set is still valid. CETT-diff will be None.')
        import traceback
        traceback.print_exc()

    # ── Free model memory ──
    del model                                                              # Line 46: release model weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                                           # Line 47: free GPU memory
    print(f'  Contrastive model unloaded.\n')

    return contrastive_path, len(contrastive_questions), cett_diff


def compute_hallucination_rate(model, processor, pope_questions,
                               pope_img_dir, device, model_type,
                               ablation_hook=None):
    """Run POPE evaluation and compute hallucination rate.

    For each yes/no question, generates the model's answer and checks
    if the model says "yes" when the ground truth is "no" (hallucination).

    Args:
        model: loaded VLM model
        processor: tokenizer/processor for the model
        pope_questions: list of POPE question dicts
        pope_img_dir: directory with COCO val images
        device: torch device string (e.g. "cuda:0")
        model_type: backend identifier string
        ablation_hook: optional context manager that applies neuron ablation

    Returns:
        dict with keys: hallucination_rate, accuracy, n_correct, n_total,
                        n_hallucinated (said yes when answer is no)
    """
    import torch                                                           # Line 1: import here to avoid top-level GPU init
    from PIL import Image                                                  # Line 2: image loading

    n_correct = 0                                                          # Line 3: count of correct predictions
    n_hallucinated = 0                                                     # Line 4: count of false "yes" answers
    n_total = 0                                                            # Line 5: total questions evaluated

    model.eval()                                                           # Line 6: ensure eval mode

    for q in pope_questions:                                               # Line 7: iterate over POPE questions
        img_filename = q['image']                                          # Line 8: e.g. "COCO_val2014_000000XXXXXX.jpg"
        img_path = os.path.join(pope_img_dir, img_filename)                # Line 9: full path to image
        question_text = q.get("text", q.get("question", ""))                 # POPE uses text key
        gt_answer = q.get("label", q.get("answer", "")).strip().lower()    # POPE uses label key

        try:
            img = Image.open(img_path).convert('RGB')                      # Line 12: load and convert to RGB
        except Exception:
            continue                                                       # Line 13: skip broken images

        # Prepare inputs based on model type
        if model_type in ('llava-hf',):                                    # Line 14: HF LLaVA input format
            prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
            inputs = processor(text=prompt, images=img,                    # Line 15: tokenize text + preprocess image
                               return_tensors='pt').to(device)
        elif model_type == 'llava-liuhaotian':                             # Line 16: original LLaVA input format
            from llava.conversation import conv_templates                   # Line 17: LLaVA conversation template
            from llava.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria  # Line 18: LLaVA utils

            conv = conv_templates["v1"].copy()                             # Line 19: create fresh conversation
            conv.append_message(conv.roles[0],                             # Line 20: USER turn with image + question
                                f"<image>\n{question_text}")
            conv.append_message(conv.roles[1], None)                       # Line 21: empty ASSISTANT turn for generation
            prompt = conv.get_prompt()                                      # Line 22: format into model prompt string

            tokenizer = processor[0] if isinstance(processor, tuple) else processor  # Line 23: extract tokenizer
            image_proc = processor[1] if isinstance(processor, tuple) else None      # Line 24: extract image processor

            input_ids = tokenizer_image_token(                             # Line 25: tokenize with <image> placeholder handling
                prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
            img_tensor = process_images([img], image_proc,                 # Line 26: preprocess image to tensor
                                        model.config).to(device, dtype=torch.float16)
            inputs = {'input_ids': input_ids, 'images': img_tensor}        # Line 27: pack into inputs dict
        elif model_type == 'llava-ov':                                    # LLaVA-OneVision input format
            messages = [{'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question_text},
            ]}]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)     # Qwen2 chat template
            inputs = processor(
                images=img, text=prompt_text,
                return_tensors='pt').to(device)                           # processor handles image + text
        elif model_type == 'internvl':                                    # InternVL2.5 — uses model.chat()
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            _mean = (0.485, 0.456, 0.406)
            _std  = (0.229, 0.224, 0.225)
            _tf = T.Compose([
                T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=_mean, std=_std),
            ])
            pixel_values = _tf(img).unsqueeze(0).to(torch.bfloat16).to(device)
            question_prompt = f'<image>\n{question_text}'
            generation_config = dict(max_new_tokens=10, do_sample=False)
            with torch.no_grad():
                if ablation_hook is not None:
                    with ablation_hook:
                        response = model.chat(processor, pixel_values, question_prompt, generation_config)
                else:
                    response = model.chat(processor, pixel_values, question_prompt, generation_config)
            answer = response.strip().lower()
            pred_yes = 'yes' in answer
            if (pred_yes and gt_answer == 'yes') or (not pred_yes and gt_answer == 'no'):
                n_correct += 1
            if pred_yes and gt_answer == 'no':
                n_hallucinated += 1
            n_total += 1
            continue                                                       # skip common generate/decode
        elif model_type == 'qwen2vl':                                     # Qwen2.5-VL input format
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': question_text},
            ]}]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                images=img, text=prompt_text,
                return_tensors='pt').to(device)
        else:
            continue                                                       # skip unsupported model types

        # Generate answer with optional ablation hook
        with torch.no_grad():                                              # Line 29: no gradient computation needed
            if ablation_hook is not None:                                   # Line 30: if we're ablating neurons
                with ablation_hook:                                        # Line 31: apply the ablation context manager
                    output_ids = model.generate(**inputs, max_new_tokens=10,  # Line 32: generate short answer
                                                do_sample=False)
            else:                                                          # Line 33: baseline (no ablation)
                output_ids = model.generate(**inputs, max_new_tokens=10,   # Line 34: generate without ablation
                                            do_sample=False)

        # Decode generated answer
        if model_type == 'llava-hf':
            generated = processor.decode(output_ids[0], skip_special_tokens=True)
        elif model_type in ('llava-ov', 'qwen2vl'):
            prompt_len = inputs['input_ids'].shape[1]
            generated = processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)
        else:
            tokenizer = processor[0] if isinstance(processor, tuple) else processor
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract yes/no from generated text
        answer = generated.strip().lower()                                 # Line 39: normalise to lowercase
        pred_yes = 'yes' in answer                                         # Line 40: check if model said yes

        # Score: correct if prediction matches ground truth
        if (pred_yes and gt_answer == 'yes') or \
           (not pred_yes and gt_answer == 'no'):                           # Line 41: prediction matches ground truth
            n_correct += 1
        if pred_yes and gt_answer == 'no':                                 # Line 42: hallucination = said yes when should be no
            n_hallucinated += 1
        n_total += 1                                                       # Line 43: increment total counter

    hallucination_rate = n_hallucinated / max(n_total, 1)                  # Line 44: fraction of hallucinated answers
    accuracy = n_correct / max(n_total, 1)                                 # Line 45: overall accuracy

    return {                                                               # Line 46: return evaluation results
        'hallucination_rate': hallucination_rate,
        'accuracy': accuracy,
        'n_correct': n_correct,
        'n_hallucinated': n_hallucinated,
        'n_total': n_total,
    }


def ablation_worker(gpu_id, layer_range, args, return_dict):
    """Worker function for multi-GPU ablation — runs on a single GPU.

    Ablates neurons layer-by-layer in batches, measuring the change
    in hallucination rate (ΔH) for each batch. Assigns the batch-level
    ΔH equally to each neuron in the batch (approximation for tractability).

    Args:
        gpu_id: integer GPU index (0, 1, 2, ...)
        layer_range: tuple (start_layer, end_layer) — exclusive end
        args: parsed arguments namespace
        return_dict: multiprocessing.Manager dict for collecting results
    """
    import torch                                                           # Line 1: import torch inside worker (fork safety)
    from baukit import TraceDict                                           # Line 2: hook-based activation interception
    from PIL import Image                                                  # Line 3: image loading

    device = f'cuda:{gpu_id}'                                              # Line 4: assign this worker to its GPU
    print(f'[GPU {gpu_id}] Processing layers {layer_range[0]}-{layer_range[1]-1}')

    # Load model onto this GPU
    if args.model_type == 'llava-hf':                                      # Line 5: load HF LLaVA
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)         # Line 6: load tokenizer + image processor
        model = LlavaForConditionalGeneration.from_pretrained(             # Line 7: load model weights
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
    elif args.model_type == 'llava-liuhaotian':                            # Line 8: load original LLaVA
        _PROJECT_ROOT = os.path.abspath(os.path.join(                      # Line 9: compute project root
            os.path.dirname(__file__), '..', '..'))
        _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')                # Line 10: path to cloned LLaVA repo
        if _LLAVA_PATH not in sys.path:
            sys.path.insert(0, _LLAVA_PATH)                                # Line 11: add to import path
        from llava.model.builder import load_pretrained_model              # Line 12: LLaVA model loader
        from llava.mm_utils import get_model_name_from_path                # Line 13: derive model name from path
        model_name = get_model_name_from_path(args.model_path)             # Line 14: e.g. "llava-v1.5-7b"
        tokenizer, model, image_processor, _ = load_pretrained_model(      # Line 15: load all model components
            args.model_path, None, model_name,
            device_map=device, torch_dtype=torch.float16)
        processor = (tokenizer, image_processor)                           # Line 16: pack as tuple for consistency
    elif args.model_type == 'llava-ov':                                    # LLaVA-OneVision (Qwen2 backbone)
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)         # loads tokenizer + SigLIP image processor
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,                   # Qwen2 native dtype
            low_cpu_mem_usage=True).to(device).eval()
    elif args.model_type == 'internvl':                                    # InternVL2.5 (InternLM2 backbone)
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
        processor = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
    elif args.model_type == 'qwen2vl':                                     # Qwen2.5-VL
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
    else:
        raise ValueError(f'Unsupported model_type for ablation: {args.model_type}')

    # Load POPE data
    pope_questions = load_pope_data(args.pope_path, args.n_pope_questions)  # Line 17: load POPE evaluation questions

    # Get layer names for this model
    layer_names = get_layer_names(args.model_type, args.n_layers)          # Line 18: hook point names

    # Baseline hallucination rate (no ablation)
    print(f'[GPU {gpu_id}] Computing baseline hallucination rate...')
    baseline = compute_hallucination_rate(                                 # Line 19: evaluate without ablation
        model, processor, pope_questions,
        args.pope_img_dir, device, args.model_type)
    baseline_hr = baseline['hallucination_rate']                           # Line 20: extract baseline hallucination rate
    print(f'[GPU {gpu_id}] Baseline hallucination rate: {baseline_hr:.4f} '
          f'({baseline["n_hallucinated"]}/{baseline["n_total"]})')

    # Per-neuron ablation scores for this GPU's layers
    neuron_scores = {}                                                     # Line 21: {(layer, neuron_start): delta_h}

    for layer_idx in range(layer_range[0], layer_range[1]):                # Line 22: iterate layers assigned to this GPU
        layer_name = layer_names[layer_idx]                                # Line 23: hook point name for this layer
        print(f'[GPU {gpu_id}] Ablating layer {layer_idx} ({layer_name})')

        for batch_start in range(0, args.n_neurons, args.batch_neurons):   # Line 24: iterate neuron batches
            batch_end = min(batch_start + args.batch_neurons, args.n_neurons)  # Line 25: clamp to layer size
            batch_size = batch_end - batch_start                           # Line 26: actual batch size

            # Create ablation hook that zeros out neurons [batch_start:batch_end]
            # in the specified layer
            class AblationHook:
                """Context manager using register_forward_hook for neuron ablation.
                Avoids baukit signature issues across versions."""
                def __init__(self, model, layer_name, start, end, method='zero'):
                    self.model = model
                    self.layer_name = layer_name
                    self.start = start
                    self.end = end
                    self.method = method
                    self._handle = None

                def _resolve_module(self):
                    """Walk the model tree to find the module at layer_name."""
                    parts = self.layer_name.split('.')
                    mod = self.model
                    for p in parts:
                        if p.isdigit():
                            mod = mod[int(p)]
                        else:
                            mod = getattr(mod, p)
                    return mod

                def __enter__(self):
                    target = self._resolve_module()
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0]
                        else:
                            out = output
                        if self.method == 'zero':
                            out[:, :, self.start:self.end] = 0
                        return (out,) + output[1:] if isinstance(output, tuple) else out
                    self._handle = target.register_forward_hook(hook_fn)
                    return self

                def __exit__(self, *args):
                    if self._handle is not None:
                        self._handle.remove()
                        self._handle = None

            # Run POPE with this batch of neurons ablated
            hook = AblationHook(model, layer_name, batch_start, batch_end,
                                args.ablation_method)
            ablated = compute_hallucination_rate(                          # Line 46: evaluate with ablation active
                model, processor, pope_questions,
                args.pope_img_dir, device, args.model_type,
                ablation_hook=hook)

            delta_h = ablated['hallucination_rate'] - baseline_hr          # Line 47: compute ΔH (positive = more hallucinations)

            # Assign ΔH equally to each neuron in the batch (approximation)
            per_neuron_delta = delta_h / batch_size                        # Line 48: distribute equally
            for n in range(batch_start, batch_end):                        # Line 49: store per-neuron scores
                neuron_scores[(layer_idx, n)] = per_neuron_delta

            if batch_start % (args.batch_neurons * 5) == 0:               # Line 50: periodic progress log
                print(f'  [GPU {gpu_id}] Layer {layer_idx}: '
                      f'neurons {batch_start}-{batch_end-1}, '
                      f'ΔH={delta_h:.4f}')

    # Store results in shared dict
    return_dict[gpu_id] = {                                                # Line 51: pass results back to main process
        'neuron_scores': {f'{k[0]}_{k[1]}': v                             # Line 52: serialize tuple keys to strings
                          for k, v in neuron_scores.items()},
        'baseline': baseline,                                              # Line 53: include baseline for reference
    }
    print(f'[GPU {gpu_id}] Done — scored {len(neuron_scores)} neurons')


def run_ablation_multi_gpu(args):
    """Orchestrate multi-GPU ablation by spawning one worker per GPU.

    Splits layers evenly across available GPUs, spawns worker processes,
    collects results, and merges into a single scores dictionary.

    Returns:
        neuron_scores: dict {(layer_idx, neuron_idx): delta_h}
        baseline: dict with baseline evaluation results
    """
    import torch                                                           # Line 1: needed for GPU count detection
    import torch.multiprocessing as mp                                     # Line 2: multiprocessing with CUDA support

    n_gpus = args.n_gpus if args.n_gpus > 0 else torch.cuda.device_count()  # Line 3: auto-detect or use specified
    n_gpus = min(n_gpus, args.n_layers)                                    # Line 4: no more GPUs than layers
    print(f'\n{"="*60}')
    print(f'PHASE 1: Multi-GPU ablation ({n_gpus} GPUs, '
          f'{args.n_layers} layers)')
    print(f'{"="*60}\n')

    if n_gpus == 0:                                                        # Line 5: no GPUs available
        raise RuntimeError('No GPUs available for ablation. '
                           'Use --skip_ablation with pre-computed scores.')

    # Split layers across GPUs as evenly as possible
    layers_per_gpu = args.n_layers // n_gpus                               # Line 6: base layers per GPU
    remainder = args.n_layers % n_gpus                                     # Line 7: leftover layers to distribute
    layer_ranges = []                                                      # Line 8: (start, end) for each GPU
    start = 0
    for i in range(n_gpus):                                                # Line 9: assign ranges
        end = start + layers_per_gpu + (1 if i < remainder else 0)         # Line 10: first 'remainder' GPUs get +1 layer
        layer_ranges.append((start, end))
        start = end

    for i, (s, e) in enumerate(layer_ranges):                             # Line 11: log assignments
        print(f'  GPU {i}: layers {s}-{e-1} ({e-s} layers)')

    # Spawn workers
    mp.set_start_method('spawn', force=True)                               # Line 12: use spawn for CUDA compatibility
    manager = mp.Manager()                                                 # Line 13: shared memory manager
    return_dict = manager.dict()                                           # Line 14: dict for collecting worker results

    processes = []                                                         # Line 15: list of spawned processes
    for gpu_id in range(n_gpus):                                           # Line 16: one process per GPU
        p = mp.Process(target=ablation_worker,                             # Line 17: create process
                       args=(gpu_id, layer_ranges[gpu_id], args, return_dict))
        processes.append(p)
        p.start()                                                          # Line 18: launch process
        if gpu_id < n_gpus - 1:                                            # Line 18b: stagger model loading
            time.sleep(30)                                                 # Line 18c: 30s between starts to avoid OOM

    for p in processes:                                                     # Line 19: wait for all processes to finish
        p.join()

    # Merge results from all GPUs
    neuron_scores = {}                                                     # Line 20: merged score dict
    baseline = None
    for gpu_id in range(n_gpus):                                           # Line 21: iterate GPU results
        if gpu_id in return_dict:
            gpu_result = return_dict[gpu_id]
            for key_str, val in gpu_result['neuron_scores'].items():       # Line 22: deserialize keys
                parts = key_str.split('_')
                neuron_scores[(int(parts[0]), int(parts[1]))] = val        # Line 23: reconstruct tuple keys
            if baseline is None:
                baseline = gpu_result['baseline']                          # Line 24: grab baseline from first GPU

    print(f'\nMerged {len(neuron_scores)} neuron scores from {n_gpus} GPUs')
    return neuron_scores, baseline                                         # Line 25: return merged results


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — Phase 2: Enrichment analysis (CPU-only)
# ═══════════════════════════════════════════════════════════════════════

def fishers_exact_test(n_category_in_driving, n_driving, n_category_total,
                       n_total):
    """Run Fisher's exact test for enrichment of a category among driving neurons.

    Constructs a 2×2 contingency table:

                          In category    Not in category
        Halluc-driving      a                b
        Not driving         c                d

    Tests whether the category is over-represented among hallucination-driving
    neurons compared to the background proportion.

    Returns:
        odds_ratio: float — how much more likely driving neurons are to be
                    in this category vs not (>1 = enriched, <1 = depleted)
        p_value: float — significance of the association
    """
    from scipy.stats import fisher_exact                                   # Line 1: import statistical test

    a = n_category_in_driving                                              # Line 2: driving AND in category
    b = n_driving - a                                                      # Line 3: driving AND NOT in category
    c = n_category_total - a                                               # Line 4: not driving AND in category
    d = n_total - n_driving - c                                            # Line 5: not driving AND NOT in category

    table = np.array([[a, b], [c, d]])                                     # Line 6: 2×2 contingency table
    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')     # Line 7: two-sided Fisher's exact test

    return odds_ratio, p_value                                             # Line 8: return OR and p-value


def compute_enrichment(flat_labels, neuron_scores, top_k_pct,
                       n_random_trials=1000, alpha=0.05, seed=42):
    """Compute enrichment statistics for ALL neuron modality categories.

    For each category (visual, text, multimodal, unknown):
        1. Count how many neurons of that category fall in the top K%
           hallucination-driving set
        2. Run Fisher's exact test for over/under-representation
        3. Compute expected count under null (proportional) hypothesis

    Also runs random baseline: repeatedly sample random neuron sets of
    the same size as the top K% and compute enrichment, to calibrate
    the statistical test.

    Args:
        flat_labels: numpy array of string labels for all neurons
        neuron_scores: dict {(layer, neuron): delta_h} — hallucination scores
        top_k_pct: float — percentage of top neurons to consider "driving"
        n_random_trials: int — number of random baseline samples
        alpha: float — significance threshold
        seed: int — random seed

    Returns:
        results: dict with per-category enrichment statistics
    """
    rng = np.random.RandomState(seed)                                      # Line 1: reproducible randomness

    # Build score array aligned with flat_labels
    n_total = len(flat_labels)                                             # Line 2: total neurons in model
    score_array = np.zeros(n_total, dtype=np.float64)                      # Line 3: default 0 for neurons without scores

    # Infer n_neurons per layer from flat_labels and n_layers
    # (flat_labels should be n_layers × n_neurons_per_layer)
    n_layers = 0                                                           # Line 4: count how many layers contributed scores
    for (l, n) in neuron_scores:
        n_layers = max(n_layers, l + 1)
    n_neurons = n_total // n_layers if n_layers > 0 else 11008             # Line 5: infer neurons per layer

    for (layer_idx, neuron_idx), delta_h in neuron_scores.items():         # Line 6: fill score array
        flat_idx = layer_idx * n_neurons + neuron_idx                      # Line 7: linearised index
        if flat_idx < n_total:                                             # Line 8: bounds check
            score_array[flat_idx] = delta_h

    # Define hallucination-driving neurons: top K% by ΔH (largest positive ΔH)
    n_driving = max(1, int(n_total * top_k_pct / 100.0))                   # Line 9: number of driving neurons
    driving_indices = np.argsort(score_array)[-n_driving:]                  # Line 10: indices of top K% by score
    is_driving = np.zeros(n_total, dtype=bool)                             # Line 11: boolean mask
    is_driving[driving_indices] = True                                     # Line 12: mark top K% as driving

    print(f'\nTop {top_k_pct}% hallucination-driving neurons: {n_driving:,} / {n_total:,}')
    print(f'Score range in driving set: [{score_array[driving_indices].min():.6f}, '
          f'{score_array[driving_indices].max():.6f}]')

    # ── Enrichment for each modality category ─────────────────────
    categories = ['visual', 'text', 'multimodal', 'unknown']              # Line 13: categories to test
    results = {'top_k_pct': top_k_pct, 'n_driving': n_driving,            # Line 14: store metadata
               'n_total': n_total, 'categories': {}}

    for cat in categories:                                                 # Line 15: iterate over categories
        cat_mask = (flat_labels == cat)                                    # Line 16: boolean mask for this category
        n_cat = int(cat_mask.sum())                                        # Line 17: total neurons in category
        n_cat_in_driving = int((cat_mask & is_driving).sum())              # Line 18: driving neurons in category

        # Expected count under proportional null
        expected = n_driving * (n_cat / n_total)                           # Line 19: expected if driving were random
        fold_enrichment = n_cat_in_driving / expected if expected > 0 else 0  # Line 20: fold change

        # Fisher's exact test
        odds_ratio, p_value = fishers_exact_test(                          # Line 21: run Fisher's exact test
            n_cat_in_driving, n_driving, n_cat, n_total)

        # Determine enrichment direction
        if p_value < alpha:                                                # Line 22: statistically significant
            direction = 'ENRICHED' if fold_enrichment > 1 else 'DEPLETED' # Line 23: enriched or depleted
        else:
            direction = 'n.s.'                                             # Line 24: not significant

        results['categories'][cat] = {                                     # Line 25: store per-category results
            'n_in_category': n_cat,
            'n_in_driving': n_cat_in_driving,
            'expected': round(expected, 1),
            'fold_enrichment': round(fold_enrichment, 4),
            'odds_ratio': round(odds_ratio, 4) if not np.isinf(odds_ratio) else 'inf',
            'p_value': p_value,
            'significant': p_value < alpha,
            'direction': direction,
            'pct_of_category_in_driving': round(100 * n_cat_in_driving / max(n_cat, 1), 2),
            'pct_of_driving_from_category': round(100 * n_cat_in_driving / n_driving, 2),
        }

    # ── Random baseline: how often does a random set show enrichment? ──
    print(f'\nRunning {n_random_trials} random baseline trials...')
    random_enrichments = {cat: [] for cat in categories}                    # Line 26: store random fold enrichments

    for trial in range(n_random_trials):                                   # Line 27: iterate random trials
        random_indices = rng.choice(n_total, size=n_driving, replace=False)  # Line 28: random neuron set
        random_mask = np.zeros(n_total, dtype=bool)                        # Line 29: boolean mask
        random_mask[random_indices] = True                                 # Line 30: mark random set

        for cat in categories:                                             # Line 31: compute enrichment for each category
            cat_mask = (flat_labels == cat)
            n_cat = int(cat_mask.sum())
            n_cat_in_random = int((cat_mask & random_mask).sum())
            expected = n_driving * (n_cat / n_total)
            fold = n_cat_in_random / expected if expected > 0 else 0
            random_enrichments[cat].append(fold)                           # Line 32: store fold enrichment

    # Add random baseline statistics to results
    results['random_baseline'] = {}                                        # Line 33: dict for random baseline stats
    for cat in categories:                                                 # Line 34: compute per-category baseline stats
        re = np.array(random_enrichments[cat])                             # Line 35: convert to numpy array
        actual_fold = results['categories'][cat]['fold_enrichment']        # Line 36: actual fold enrichment

        # Empirical p-value: how often does random exceed actual
        if actual_fold > 1:                                                # Line 37: enrichment direction
            emp_p = (re >= actual_fold).mean()                             # Line 38: fraction of random trials ≥ actual
        else:
            emp_p = (re <= actual_fold).mean()                             # Line 39: fraction of random trials ≤ actual

        results['random_baseline'][cat] = {                                # Line 40: store baseline results
            'mean_fold': round(float(re.mean()), 4),
            'std_fold': round(float(re.std()), 4),
            'ci_95_low': round(float(np.percentile(re, 2.5)), 4),
            'ci_95_high': round(float(np.percentile(re, 97.5)), 4),
            'empirical_p_value': round(float(emp_p), 4),
        }

    return results                                                         # Line 41: return all enrichment results


def compute_per_layer_enrichment(labels_dict, neuron_scores, top_k_pct,
                                 n_layers, n_neurons):
    """Compute enrichment per layer for heatmap visualisation.

    For each layer, defines the top K% of that layer's neurons as
    hallucination-driving, then computes fold enrichment for each category.

    Returns:
        heatmap: numpy array (n_layers, 4) — fold enrichment per category per layer
        categories: list of category names corresponding to columns
    """
    categories = ['visual', 'text', 'multimodal', 'unknown']              # Line 1: column order
    heatmap = np.ones((n_layers, len(categories)), dtype=np.float64)       # Line 2: default fold=1.0 (no enrichment)

    for l in range(n_layers):                                              # Line 3: iterate layers
        key = str(l)
        if key not in labels_dict:                                         # Line 4: skip missing layers
            continue

        # Get labels for this layer
        layer_labels = sorted(labels_dict[key],                            # Line 5: sort by neuron index
                              key=lambda x: x['neuron_idx'])
        label_array = np.array([e['label'] for e in layer_labels])         # Line 6: extract labels

        # Get scores for this layer
        scores = np.zeros(len(label_array), dtype=np.float64)              # Line 7: score array for this layer
        for n_idx in range(len(label_array)):                              # Line 8: fill from neuron_scores
            if (l, n_idx) in neuron_scores:
                scores[n_idx] = neuron_scores[(l, n_idx)]

        # Top K% in this layer
        n_driving = max(1, int(len(label_array) * top_k_pct / 100.0))     # Line 9: driving count for this layer
        driving_idx = np.argsort(scores)[-n_driving:]                      # Line 10: top K% by score
        is_driving = np.zeros(len(label_array), dtype=bool)                # Line 11: boolean mask
        is_driving[driving_idx] = True                                     # Line 12: mark driving neurons

        for ci, cat in enumerate(categories):                              # Line 13: compute per-category enrichment
            cat_mask = (label_array == cat)                                # Line 14: category mask
            n_cat = int(cat_mask.sum())                                    # Line 15: total in category
            n_cat_driving = int((cat_mask & is_driving).sum())             # Line 16: driving in category
            expected = n_driving * (n_cat / len(label_array)) if len(label_array) > 0 else 0
            if expected > 0:                                               # Line 17: avoid division by zero
                heatmap[l, ci] = n_cat_driving / expected                  # Line 18: fold enrichment

    return heatmap, categories                                             # Line 19: return heatmap and column names


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — Phase 3: Visualisation
# ═══════════════════════════════════════════════════════════════════════

def plot_enrichment_results(results, per_layer_heatmap, categories,
                            output_dir, model_name):
    """Generate all enrichment visualisation plots.

    Creates four plots:
      1. Enrichment bar chart — fold enrichment per category with significance stars
      2. Odds-ratio forest plot — OR with 95% CI from random baseline
      3. Per-layer enrichment heatmap — fold enrichment by layer × category
      4. Summary table — text-based summary saved as PNG
    """
    import matplotlib                                                      # Line 1: matplotlib for plotting
    matplotlib.use('Agg')                                                  # Line 2: non-interactive backend
    import matplotlib.pyplot as plt                                        # Line 3: plotting API
    import matplotlib.colors as mcolors                                    # Line 4: colour normalisation for heatmap

    os.makedirs(output_dir, exist_ok=True)                                 # Line 5: ensure output directory exists

    cats = ['visual', 'text', 'multimodal', 'unknown']                    # Line 6: category order
    cat_colors = {                                                         # Line 7: colour scheme for categories
        'visual': '#3498db',                                               # Line 8: blue for visual
        'text': '#e74c3c',                                                 # Line 9: red for text
        'multimodal': '#2ecc71',                                           # Line 10: green for multimodal
        'unknown': '#95a5a6',                                              # Line 11: grey for unknown
    }

    # ── Plot 1: Enrichment bar chart ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))                                # Line 12: create figure
    x = np.arange(len(cats))                                               # Line 13: x positions for bars
    folds = [results['categories'][c]['fold_enrichment'] for c in cats]    # Line 14: fold enrichment values
    colors = [cat_colors[c] for c in cats]                                 # Line 15: bar colours
    sigs = [results['categories'][c]['significant'] for c in cats]         # Line 16: significance flags

    bars = ax.bar(x, folds, color=colors, edgecolor='white', linewidth=1.5,  # Line 17: draw bars
                  width=0.6)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1,         # Line 18: reference line at fold=1
               label='Expected (null)')

    # Add significance stars
    for i, (bar, sig) in enumerate(zip(bars, sigs)):                       # Line 19: annotate bars
        if sig:
            p = results['categories'][cats[i]]['p_value']
            stars = '***' if p < 0.001 else ('**' if p < 0.01 else '*')   # Line 20: significance level
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    stars, ha='center', va='bottom', fontsize=14,          # Line 21: place stars above bar
                    fontweight='bold')

    # Add random baseline CI as error bars
    for i, cat in enumerate(cats):                                         # Line 22: overlay random baseline range
        rb = results['random_baseline'][cat]
        ax.errorbar(i, rb['mean_fold'],                                    # Line 23: random baseline mean
                    yerr=[[rb['mean_fold'] - rb['ci_95_low']],             # Line 24: lower error bar
                          [rb['ci_95_high'] - rb['mean_fold']]],           # Line 25: upper error bar
                    fmt='D', color='black', markersize=6, capsize=5,       # Line 26: diamond marker
                    label='Random baseline 95% CI' if i == 0 else None)

    ax.set_xticks(x)                                                       # Line 27: set x-axis tick positions
    ax.set_xticklabels([c.capitalize() for c in cats], fontsize=12)        # Line 28: category labels
    ax.set_ylabel('Fold Enrichment', fontsize=13)                          # Line 29: y-axis label
    ax.set_title(f'{model_name} — Hallucination-Driving Neuron Enrichment\n'
                 f'(Top {results["top_k_pct"]}%, N={results["n_driving"]:,})',
                 fontsize=14, fontweight='bold')                           # Line 30: title with metadata
    ax.legend(fontsize=10)                                                 # Line 31: show legend
    ax.set_ylim(bottom=0)                                                  # Line 32: y-axis starts at 0
    plt.tight_layout()                                                     # Line 33: prevent label clipping
    fig.savefig(os.path.join(output_dir, 'enrichment_bar_chart.png'),      # Line 34: save figure
                dpi=200, bbox_inches='tight')
    plt.close(fig)                                                         # Line 35: free memory

    # ── Plot 2: Odds-ratio forest plot ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))                                 # Line 36: create figure
    y_pos = np.arange(len(cats))                                           # Line 37: y positions

    for i, cat in enumerate(cats):                                         # Line 38: iterate categories
        or_val = results['categories'][cat]['odds_ratio']                   # Line 39: odds ratio
        if or_val == 'inf':                                                # Line 40: handle infinite OR
            or_val = 10.0
        sig = results['categories'][cat]['significant']                    # Line 41: is it significant?
        marker = 's' if sig else 'o'                                       # Line 42: filled square if significant
        color = cat_colors[cat]
        ax.plot(or_val, i, marker, color=color, markersize=12,            # Line 43: plot OR point
                markeredgecolor='black', markeredgewidth=1)
        ax.annotate(f'  OR={or_val:.2f}', (or_val, i),                    # Line 44: annotate with OR value
                    fontsize=10, va='center')

    ax.axvline(x=1.0, color='grey', linestyle='--', linewidth=1)          # Line 45: null OR reference line
    ax.set_yticks(y_pos)                                                   # Line 46: set y-axis ticks
    ax.set_yticklabels([c.capitalize() for c in cats], fontsize=12)        # Line 47: category labels
    ax.set_xlabel('Odds Ratio', fontsize=13)                               # Line 48: x-axis label
    ax.set_title(f'{model_name} — Odds Ratios (Hallucination Enrichment)', # Line 49: title
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')                                                   # Line 50: log scale for OR
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'odds_ratio_forest.png'),         # Line 51: save figure
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 3: Per-layer enrichment heatmap ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 12))                                # Line 52: tall figure for all layers
    n_layers = per_layer_heatmap.shape[0]

    # Diverging colour map centred at 1.0 (no enrichment)
    vmin, vmax = 0.0, max(2.0, per_layer_heatmap.max())                    # Line 53: colour range
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)        # Line 54: centre at 1.0
    im = ax.imshow(per_layer_heatmap, aspect='auto', cmap='RdBu_r',       # Line 55: draw heatmap
                   norm=norm, interpolation='nearest')

    ax.set_xticks(range(len(categories)))                                  # Line 56: x-axis ticks
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)  # Line 57: x-axis labels
    ax.set_yticks(range(n_layers))                                         # Line 58: y-axis ticks
    ax.set_yticklabels([f'L{l}' for l in range(n_layers)], fontsize=8)    # Line 59: layer labels
    ax.set_xlabel('Neuron Category', fontsize=13)                          # Line 60: x-axis label
    ax.set_ylabel('Layer', fontsize=13)                                    # Line 61: y-axis label
    ax.set_title(f'{model_name} — Per-Layer Enrichment '                   # Line 62: title
                 f'(Fold, top {results["top_k_pct"]}%)',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Fold Enrichment', shrink=0.8)          # Line 63: add colour bar
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'per_layer_enrichment_heatmap.png'),  # Line 64: save
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'Plots saved to {output_dir}')


# ═══════════════════════════════════════════════════════════════════════
# Section 6 — Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()                                                    # Line 1: parse command-line arguments

    # ── Plot-only mode: regenerate plots from saved results ───────
    if args.plot_only:
        print(f'\n{"="*60}')
        print(f'PLOT-ONLY MODE: Regenerating plots for {args.model_name}')
        print(f'{"="*60}\n')

        results_path = os.path.join(args.output_dir, 'enrichment_results.json')
        heatmap_path = os.path.join(args.output_dir, 'per_layer_enrichment.npy')

        # Also check for legacy naming with model-type suffix (e.g. per_layer_enrichment-llava-ov.npy)
        if not os.path.isfile(heatmap_path):
            import glob
            legacy = glob.glob(os.path.join(args.output_dir, 'per_layer_enrichment-*.npy'))
            if legacy:
                heatmap_path = legacy[0]
                print(f'  Using legacy heatmap: {os.path.basename(heatmap_path)}')

        if not os.path.isfile(results_path):
            print(f'  ERROR: {results_path} not found. Run full pipeline first.')
            return
        if not os.path.isfile(heatmap_path):
            print(f'  ERROR: {heatmap_path} not found. Run full pipeline first.')
            return

        with open(results_path) as f:
            results = json.load(f)
        per_layer_heatmap = np.load(heatmap_path)
        heatmap_cats = ['visual', 'text', 'multimodal', 'unknown']

        # Save with standard name if loaded from legacy path
        std_path = os.path.join(args.output_dir, 'per_layer_enrichment.npy')
        if heatmap_path != std_path:
            np.save(std_path, per_layer_heatmap)
            print(f'  Saved standardized copy → {std_path}')

        try:
            plot_enrichment_results(results, per_layer_heatmap, heatmap_cats,
                                    args.output_dir, args.model_name)
        except ImportError as e:
            print(f'WARNING: Could not generate plots ({e}).')
            return

        print(f'  Plots saved to {args.output_dir}/*.png')
        return

    rng = np.random.RandomState(args.seed)                                 # Line 2: set global random seed
    os.makedirs(args.output_dir, exist_ok=True)                            # Line 3: create output directory

    if args.label_dir is None:
        print('ERROR: --label_dir is required (unless using --plot_only)')
        return

    # ── Load neuron modality labels ───────────────────────────────
    print(f'\n{"="*60}')
    print(f'Loading neuron labels ({args.taxonomy})')
    print(f'{"="*60}\n')

    layer_names = get_layer_names(args.model_type, args.n_layers)          # Line 4: hook point names
    labels_dict, flat_labels = load_neuron_labels(                         # Line 5: load all neuron labels
        args.label_dir, args.taxonomy, layer_names, args.n_layers)

    # ── Contrastive POPE preprocessing (if requested) ───────────────
    cett_diff_scores = None                                                # Line 5a: will hold CETT-diff if contrastive
    if args.contrastive:                                                   # Line 5b: build contrastive set + CETT-diff
        contrastive_path, n_clean, cett_diff_scores = build_contrastive_pope_set(args)
        if not args.skip_ablation:                                         # Line 5c: only override pope_path if ablation will run
            args.pope_path = contrastive_path                              # Line 5d: redirect workers to filtered JSONL
            args.n_pope_questions = None                                   # Line 5e: use all filtered questions (no truncation)
            print(f'  Ablation will use {n_clean} contrastive POPE questions')
        else:
            print(f'  Contrastive set built for CETT-diff only '
                  f'(ablation skipped, using pre-computed ΔH scores)')
        if cett_diff_scores is not None:
            print(f'  CETT-diff scores available ({len(cett_diff_scores)} neurons)')

    # ── Phase 1: Identify hallucination-driving neurons ───────────
    if args.skip_ablation:                                                 # Line 6: use pre-computed scores
        print(f'\nSkipping ablation — loading scores from {args.ablation_scores}')
        with open(args.ablation_scores) as f:                              # Line 7: load pre-computed JSON
            scores_raw = json.load(f)
        neuron_scores = {}                                                 # Line 8: reconstruct score dict
        for key_str, val in scores_raw.items():                            # Line 9: parse "layer_neuron" keys
            parts = key_str.split('_')
            neuron_scores[(int(parts[0]), int(parts[1]))] = val
        print(f'Loaded {len(neuron_scores)} pre-computed neuron scores')
    else:
        neuron_scores, baseline = run_ablation_multi_gpu(args)             # Line 10: run multi-GPU ablation

        # Save ablation scores for future re-use
        scores_path = os.path.join(args.output_dir, 'ablation_scores.json')  # Line 11: output path
        scores_serializable = {f'{k[0]}_{k[1]}': v                        # Line 12: serialize tuple keys
                               for k, v in neuron_scores.items()}
        with open(scores_path, 'w') as f:                                  # Line 13: write JSON
            json.dump(scores_serializable, f, indent=2)
        print(f'Ablation scores saved to {scores_path}')

        # Also save baseline results
        if baseline:                                                       # Line 14: save baseline evaluation
            baseline_path = os.path.join(args.output_dir, 'baseline_results.json')
            with open(baseline_path, 'w') as f:
                json.dump(baseline, f, indent=2)

    # ── Phase 2: Enrichment analysis ──────────────────────────────
    print(f'\n{"="*60}')
    print(f'PHASE 2: Enrichment analysis (all categories)')
    print(f'{"="*60}\n')

    results = compute_enrichment(                                          # Line 15: run enrichment for all categories
        flat_labels, neuron_scores,
        top_k_pct=args.top_k_pct,
        n_random_trials=args.n_random_trials,
        alpha=args.alpha,
        seed=args.seed)

    # Print enrichment summary table
    print(f'\n{"─"*80}')                                                   # Line 16: header line
    print(f'ENRICHMENT SUMMARY — Top {args.top_k_pct}% hallucination-driving neurons')
    print(f'{"─"*80}')
    print(f'{"Category":<14} {"In Driving":>10} {"Expected":>10} '        # Line 17: column headers
          f'{"Fold":>8} {"OR":>8} {"p-value":>10} {"Direction":>10}')
    print(f'{"─"*80}')

    for cat in ['visual', 'text', 'multimodal', 'unknown']:               # Line 18: print each category
        r = results['categories'][cat]
        or_str = f'{r["odds_ratio"]:.2f}' if r["odds_ratio"] != 'inf' else 'inf'
        print(f'{cat:<14} {r["n_in_driving"]:>10,} {r["expected"]:>10.1f} '
              f'{r["fold_enrichment"]:>8.3f} {or_str:>8} '
              f'{r["p_value"]:>10.2e} {r["direction"]:>10}')
    print(f'{"─"*80}')

    # Print random baseline summary
    print(f'\nRandom baseline (n={args.n_random_trials}):')                # Line 19: random baseline header
    for cat in ['visual', 'text', 'multimodal', 'unknown']:
        rb = results['random_baseline'][cat]
        print(f'  {cat:<12}: mean_fold={rb["mean_fold"]:.3f} '
              f'[{rb["ci_95_low"]:.3f}, {rb["ci_95_high"]:.3f}] '
              f'emp_p={rb["empirical_p_value"]:.4f}')

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'enrichment_results.json')  # Line 20: output path
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)                       # Line 21: write results
    print(f'\nResults saved to {results_path}')

    # ── CETT-diff enrichment (if contrastive was used) ────────────
    if cett_diff_scores is not None:
        print(f'\n{"="*60}')
        print(f'CETT-DIFF ENRICHMENT ANALYSIS (independent of ablation ΔH)')
        print(f'{"="*60}\n')

        cett_results = compute_enrichment(                                 # Line E1: enrichment using CETT-diff scores
            flat_labels, cett_diff_scores,
            top_k_pct=args.top_k_pct,
            n_random_trials=args.n_random_trials,
            alpha=args.alpha,
            seed=args.seed)

        # Print CETT-diff enrichment summary
        print(f'\n{"─"*80}')
        print(f'CETT-DIFF ENRICHMENT — Top {args.top_k_pct}% by CETT-diff')
        print(f'{"─"*80}')
        print(f'{"Category":<14} {"In Driving":>10} {"Expected":>10} '
              f'{"Fold":>8} {"OR":>8} {"p-value":>10} {"Direction":>10}')
        print(f'{"─"*80}')

        for cat in ['visual', 'text', 'multimodal', 'unknown']:
            r = cett_results['categories'][cat]
            or_str = f'{r["odds_ratio"]:.2f}' if r["odds_ratio"] != 'inf' else 'inf'
            print(f'{cat:<14} {r["n_in_driving"]:>10,} {r["expected"]:>10.1f} '
                  f'{r["fold_enrichment"]:>8.3f} {or_str:>8} '
                  f'{r["p_value"]:>10.2e} {r["direction"]:>10}')
        print(f'{"─"*80}')

        # Save CETT-diff enrichment
        cett_enrich_path = os.path.join(args.output_dir,
                                        'enrichment_results_cett_diff.json')
        with open(cett_enrich_path, 'w') as f:
            json.dump(cett_results, f, indent=2, default=str)
        print(f'CETT-diff enrichment saved to {cett_enrich_path}')

        # Compare top-5% overlap between ΔH and CETT-diff (Jaccard)
        n_total = len(neuron_scores)
        top_k = max(1, int(n_total * args.top_k_pct / 100.0))
        top_dh = set(sorted(neuron_scores, key=neuron_scores.get,          # Line E2: top by ΔH (most positive)
                            reverse=True)[:top_k])
        top_cett = set(sorted(cett_diff_scores,                            # Line E3: top by CETT-diff
                              key=cett_diff_scores.get,
                              reverse=True)[:top_k])
        overlap = len(top_dh & top_cett)                                   # Line E4: intersection size
        jaccard = overlap / len(top_dh | top_cett) if top_dh | top_cett else 0  # Line E5: Jaccard similarity
        print(f'\n  Top-{args.top_k_pct}% overlap (ΔH vs CETT-diff):')
        print(f'    Intersection: {overlap}/{top_k} neurons')
        print(f'    Jaccard similarity: {jaccard:.3f}')
        print(f'    (High Jaccard = both methods agree on which neurons '
              f'drive hallucination)')

        # ── Combined hallucination score: normalize(ΔH) × normalize(CETT-diff) ──
        print(f'\n{"="*60}')
        print(f'COMBINED HALLUCINATION SCORE (ΔH × CETT-diff)')
        print(f'{"="*60}\n')

        # Normalize ΔH to [0, 1]
        dh_vals = np.array(list(neuron_scores.values()))                   # Line H1: all ΔH values
        dh_min, dh_max = float(dh_vals.min()), float(dh_vals.max())        # Line H2: global range
        dh_range = dh_max - dh_min if dh_max > dh_min else 1.0            # Line H3: avoid division by zero

        # Normalize CETT-diff to [0, 1]
        cd_vals = np.array(list(cett_diff_scores.values()))                # Line H4: all CETT-diff values
        cd_min, cd_max = float(cd_vals.min()), float(cd_vals.max())        # Line H5: global range
        cd_range = cd_max - cd_min if cd_max > cd_min else 1.0            # Line H6: avoid division by zero

        # Compute combined = normalize(ΔH) × normalize(CETT-diff) for shared keys
        combined_halluc_scores = {}                                        # Line H7: {(layer, neuron): float}
        shared_keys = set(neuron_scores.keys()) & set(cett_diff_scores.keys())  # Line H8: neurons with both scores
        for key in shared_keys:                                            # Line H9: iterate shared neurons
            norm_dh = (neuron_scores[key] - dh_min) / dh_range            # Line H10: normalized ΔH
            norm_cd = (cett_diff_scores[key] - cd_min) / cd_range         # Line H11: normalized CETT-diff
            combined_halluc_scores[key] = norm_dh * norm_cd               # Line H12: product

        # Save all three rankings
        # 1. ΔH already saved as ablation_scores.json
        # 2. CETT-diff already saved as cett_diff_scores.json
        # 3. Save combined
        combined_path = os.path.join(args.output_dir,                      # Line H13: output path
                                     'combined_halluc_scores.json')
        combined_serializable = {f'{k[0]}_{k[1]}': v                       # Line H14: serialize tuple keys
                                 for k, v in combined_halluc_scores.items()}
        with open(combined_path, 'w') as f:                                # Line H15: write JSON
            json.dump(combined_serializable, f, indent=2)
        print(f'  {len(combined_halluc_scores)} neurons scored')
        print(f'  Saved → {combined_path}')

        # Run enrichment on combined scores
        combined_enrich = compute_enrichment(
            flat_labels, combined_halluc_scores,
            top_k_pct=args.top_k_pct,
            n_random_trials=args.n_random_trials,
            alpha=args.alpha,
            seed=args.seed)

        print(f'\n{"─"*80}')
        print(f'COMBINED ENRICHMENT — Top {args.top_k_pct}% by ΔH × CETT-diff')
        print(f'{"─"*80}')
        print(f'{"Category":<14} {"In Driving":>10} {"Expected":>10} '
              f'{"Fold":>8} {"OR":>8} {"p-value":>10} {"Direction":>10}')
        print(f'{"─"*80}')

        for cat in ['visual', 'text', 'multimodal', 'unknown']:
            r = combined_enrich['categories'][cat]
            or_str = f'{r["odds_ratio"]:.2f}' if r["odds_ratio"] != 'inf' else 'inf'
            print(f'{cat:<14} {r["n_in_driving"]:>10,} {r["expected"]:>10.1f} '
                  f'{r["fold_enrichment"]:>8.3f} {or_str:>8} '
                  f'{r["p_value"]:>10.2e} {r["direction"]:>10}')
        print(f'{"─"*80}')

        combined_enrich_path = os.path.join(args.output_dir,
                                            'enrichment_results_combined.json')
        with open(combined_enrich_path, 'w') as f:
            json.dump(combined_enrich, f, indent=2, default=str)
        print(f'Combined enrichment saved to {combined_enrich_path}')

        # Summary of all three methods
        print(f'\n{"="*60}')
        print(f'ENRICHMENT COMPARISON (visual neuron fold-enrichment)')
        print(f'{"="*60}')
        vis_dh = results['categories']['visual']['fold_enrichment']
        vis_cd = cett_results['categories']['visual']['fold_enrichment']
        vis_cb = combined_enrich['categories']['visual']['fold_enrichment']
        print(f'  ΔH (ablation):     {vis_dh:.3f}×')
        print(f'  CETT-diff:         {vis_cd:.3f}×')
        print(f'  Combined (ΔH×CD):  {vis_cb:.3f}×')
        print(f'\n  Output files for step 11 steering:')
        print(f'    --halluc_scores_path {combined_path}  (default: combined)')
        print(f'    --halluc_scores_path {os.path.join(args.output_dir, "ablation_scores.json")}  (ΔH only)')
        print(f'    --halluc_scores_path {os.path.join(args.output_dir, "cett_diff_scores.json")}  (CETT-diff only)')

    # ── Per-layer enrichment heatmap ──────────────────────────────
    per_layer_heatmap, heatmap_cats = compute_per_layer_enrichment(        # Line 22: compute per-layer enrichment
        labels_dict, neuron_scores, args.top_k_pct,
        args.n_layers, args.n_neurons)

    # Save heatmap data
    np.save(os.path.join(args.output_dir, 'per_layer_enrichment.npy'),     # Line 23: save numpy array
            per_layer_heatmap)

    # ── Phase 3: Visualisation ────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'PHASE 3: Generating plots')
    print(f'{"="*60}\n')

    try:
        plot_enrichment_results(results, per_layer_heatmap, heatmap_cats,  # Line 24: generate all plots
                                args.output_dir, args.model_name)
    except ImportError as e:                                               # Line 25: handle missing matplotlib
        print(f'WARNING: Could not generate plots ({e}). '
              f'Install matplotlib to enable visualisation.')

    print(f'\n{"="*60}')
    print(f'HALLUCINATION TAXONOMY COMPLETE')
    print(f'  Results:  {results_path}')
    print(f'  Plots:    {args.output_dir}/*.png')
    if cett_diff_scores is not None:
        print(f'  CETT-diff: {args.output_dir}/cett_diff_scores.json')
        print(f'  CETT-diff enrichment: {args.output_dir}/enrichment_results_cett_diff.json')
        print(f'  Combined:  {args.output_dir}/combined_halluc_scores.json')
        print(f'  Combined enrichment: {args.output_dir}/enrichment_results_combined.json')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()





# """
# hallucination_taxonomy.py — Full enrichment analysis across ALL neuron modality categories

# Tests whether visual, text, multimodal, unknown, and random neuron sets
# are disproportionately hallucination-driving, using Fisher's exact test
# and bootstrap confidence intervals.

# Pipeline:
#   Phase 0 — (Optional) Contrastive POPE preprocessing (--contrastive).
#              Loads 3 POPE splits (random, popular, adversarial), samples
#              N responses per question at temperature=0.7, keeps only
#              questions where the model is consistently correct or
#              consistently wrong across all samples. Caps at 333 per
#              split (~1000 total). Produces a high-confidence contrastive
#              set that replaces the raw POPE data for Phase 1.

#   Phase 1 — Identify hallucination-driving neurons via per-neuron ablation
#              on POPE (Polling-based Object Probing Evaluation).
#              Each neuron is zeroed out, and we measure the change in
#              hallucination rate (ΔHallucination = ablated − baseline).
#              Neurons with the largest positive ΔH are hallucination-driving.
#              ** Multi-GPU: layers sharded across available GPUs **

#   Phase 2 — Enrichment analysis: for each modality category
#              (visual, text, multimodal, unknown, random), test whether
#              that category is over-represented among hallucination-driving
#              neurons compared to chance expectation (Fisher's exact test).

#   Phase 3 — Visualisation: enrichment bar plots, odds-ratio forest plot,
#              per-layer enrichment heatmap.

# Inputs:
#   --label_dir       : directory with neuron_labels.json or neuron_labels_permutation.json per layer
#   --pope_path       : path to POPE evaluation JSON (coco_pope_random.json)
#   --pope_img_dir    : directory with COCO val images
#   --model_type      : llava-hf | llava-liuhaotian | internvl | qwen2vl | llava-ov
#   --model_path      : HF model ID or local path
#   --taxonomy         : ft | pmbt — which classification to use
#   --top_k_pct       : top K% of neurons by ΔH to define "hallucination-driving"
#   --n_pope_images   : number of POPE questions to evaluate per ablation
#   --output_dir      : where to save results

# Multi-GPU:
#   Uses torch.multiprocessing to spawn one worker per available GPU.
#   Each worker processes a disjoint subset of layers.
#   Phase 2 and 3 are CPU-only and run after all GPU workers finish.

# Usage:
#     # Full analysis (auto-detect GPUs)
#     python hallucination_taxonomy.py \
#         --label_dir results/llava-1.5-7b/llm_permutation \
#         --pope_path data/POPE/output/coco/coco_pope_random.json \
#         --pope_img_dir data/val2014 \
#         --model_path liuhaotian/llava-v1.5-7b \
#         --model_type llava-liuhaotian

#     # Restrict to 4 GPUs
#     python hallucination_taxonomy.py --n_gpus 4 ...

#     # Phase 2 only (skip ablation, load pre-computed scores)
#     python hallucination_taxonomy.py --skip_ablation --ablation_scores results/ablation_scores.json ...
# """

# import argparse                     # Line 1: parse command-line arguments
# import json                         # Line 2: read/write JSON files for labels, POPE data, results
# import os                           # Line 3: file path manipulation and directory creation
# import sys                          # Line 4: modify Python import path for LLaVA repo
# import time                         # Line 5: timing each phase of the pipeline
# from collections import defaultdict # Line 6: default dictionaries for aggregating per-layer stats

# import numpy as np                  # Line 7: numerical operations on activation arrays

# # ═══════════════════════════════════════════════════════════════════════
# # Section 1 — Argument parsing
# # ═══════════════════════════════════════════════════════════════════════

# def parse_args():
#     """Parse all command-line arguments for the hallucination taxonomy pipeline.

#     Groups: model config, data paths, ablation settings, enrichment settings,
#             GPU config, output config.
#     """
#     p = argparse.ArgumentParser(
#         description='Hallucination Taxonomy: enrichment analysis across '   # Line 1: description for --help
#                     'ALL neuron modality categories (visual, text, '
#                     'multimodal, unknown, random)')

#     # ── Model configuration ───────────────────────────────────────
#     p.add_argument('--model_type', default='llava-liuhaotian',              # Line 2: which model backend to use
#                    choices=['llava-hf', 'llava-liuhaotian', 'internvl',
#                             'qwen2vl', 'llava-ov'],
#                    help='Model backend (determines layer naming + loading)')
#     p.add_argument('--model_path', default='liuhaotian/llava-v1.5-7b',     # Line 3: HF model ID or local path
#                    help='HuggingFace model ID or local path to weights')
#     p.add_argument('--model_name', default='llava-1.5-7b',                 # Line 4: short name for output dirs
#                    help='Short model name for output directory naming')
#     p.add_argument('--n_layers', type=int, default=32,                     # Line 5: total LLM layers in model
#                    help='Total number of LLM layers (32 for LLaMA-2, 28 for Qwen2)')
#     p.add_argument('--n_neurons', type=int, default=11008,                 # Line 6: neurons per FFN layer
#                    help='Neurons per layer (11008 for LLaMA-2, 14336 for InternLM2)')

#     # ── Data paths ────────────────────────────────────────────────
#     p.add_argument('--label_dir', default=None,                             # Line 7: directory with classification results
#                    help='Directory containing per-layer neuron label JSONs '
#                         '(e.g. results/llava-1.5-7b/llm_permutation)')
#     p.add_argument('--taxonomy', default='pmbt',                           # Line 8: use permutation-test labels (statistically principled)
#                    help='Taxonomy to use: pmbt = permutation-test labels (default)')
#     p.add_argument('--pope_path',                                          # Line 9: path to POPE evaluation data
#                    default='data/POPE/output/coco/coco_pope_random.json',
#                    help='Path to POPE evaluation JSONL')
#     p.add_argument('--pope_img_dir', default='data/val2014',               # Line 10: COCO val images directory
#                    help='Directory containing COCO val2014 images')
#     p.add_argument('--coco_img_dir',                                       # Line 11: COCO train images for descriptions
#                    default='/home/projects/bagon/shared/coco2017/images/train2017/',
#                    help='COCO train2017 images directory')
#     p.add_argument('--detail_23k_path',                                    # Line 12: detail_23k image list
#                    default='data/detail_23k.json',
#                    help='Path to detail_23k.json')
#     p.add_argument('--generated_desc_path',                                # Line 13: LLaVA-generated descriptions
#                    default='results/1-describe/generated_descriptions.json',
#                    help='Path to generated descriptions JSON')

#     # ── Ablation settings (Phase 1) ───────────────────────────────
#     p.add_argument('--skip_ablation', action='store_true',                 # Line 14: skip Phase 1 if scores exist
#                    help='Skip ablation; load pre-computed scores from --ablation_scores')
#     p.add_argument('--ablation_scores', default=None,                      # Line 15: path to pre-computed ablation scores
#                    help='Path to pre-computed ablation scores JSON (used with --skip_ablation)')
#     p.add_argument('--n_pope_questions', type=int, default=500,            # Line 16: POPE questions per ablation eval
#                    help='Number of POPE questions per ablation evaluation')
#     p.add_argument('--contrastive', action='store_true',                   # Line 16b: enable contrastive filtering
#                    help='Build contrastive POPE set by sampling multiple '
#                         'responses per question and keeping only consistently '
#                         'correct or consistently incorrect cases')
#     p.add_argument('--contrastive_samples', type=int, default=10,          # Line 16c: samples per question
#                    help='Number of responses to sample per POPE question '
#                         'for contrastive filtering (default: 10)')
#     p.add_argument('--contrastive_cap_per_split', type=int, default=333,   # Line 16d: max questions per split
#                    help='Max clean questions to keep per POPE split '
#                         '(default: 333, giving ~1000 total across 3 splits)')
#     p.add_argument('--contrastive_start_per_split', type=int, default=1250, # Line 16e: starting questions per split
#                    help='Number of POPE questions to sample from each split '
#                         'before filtering (default: 1250)')
#     p.add_argument('--pope_splits_dir', default=None,                      # Line 16f: directory with all 3 POPE split files
#                    help='Directory containing coco_pope_random.json, '
#                         'coco_pope_popular.json, coco_pope_adversarial.json. '
#                         'Auto-detected from --pope_path parent if not set.')
#     p.add_argument('--ablation_method', default='zero',                    # Line 17: how to ablate neurons
#                    choices=['zero', 'mean'],
#                    help='Ablation method: zero = set to 0, mean = set to dataset mean')
#     p.add_argument('--batch_neurons', type=int, default=50,                # Line 18: ablate neurons in batches
#                    help='Number of neurons to ablate simultaneously per evaluation '
#                         '(grouped ablation for tractability)')

#     # ── Enrichment settings (Phase 2) ─────────────────────────────
#     p.add_argument('--top_k_pct', type=float, default=5.0,                 # Line 19: top K% as "hallucination-driving"
#                    help='Top K%% of neurons by ΔHallucination to classify as '
#                         'hallucination-driving')
#     p.add_argument('--n_random_trials', type=int, default=1000,            # Line 20: random baseline repetitions
#                    help='Number of random neuron sets for calibration')
#     p.add_argument('--alpha', type=float, default=0.05,                    # Line 21: significance threshold
#                    help='Significance level for enrichment tests')
#     p.add_argument('--seed', type=int, default=42,                         # Line 22: random seed for reproducibility
#                    help='Random seed')

#     # ── GPU configuration ─────────────────────────────────────────
#     p.add_argument('--n_gpus', type=int, default=0,                        # Line 23: number of GPUs (0 = auto-detect)
#                    help='Number of GPUs to use (0 = auto-detect all available)')

#     # ── Output ────────────────────────────────────────────────────
#     p.add_argument('--output_dir', default='results/hallucination_taxonomy', # Line 24: output directory
#                    help='Output directory for enrichment results and plots')
#     p.add_argument('--plot_only', action='store_true',                      # Line 25: regenerate plots from saved results
#                    help='Skip all computation, just regenerate plots from '
#                         'existing enrichment_results.json and per_layer_enrichment.npy')

#     return p.parse_args()


# # ═══════════════════════════════════════════════════════════════════════
# # Section 2 — Load neuron modality labels
# # ═══════════════════════════════════════════════════════════════════════

# def get_layer_names(model_type, n_layers):
#     """Return act_fn hook-point names for each LLM layer.

#     Maps model_type to the correct module path prefix so we can
#     attach baukit hooks at the right location in the model tree.
#     """
#     if model_type == 'llava-hf':                                           # Line 1: HF LLaVA wrapper
#         prefix = 'model.language_model.layers'
#         suffix = 'mlp.act_fn'
#     elif model_type == 'internvl':                                         # Line 2: InternVL2.5 architecture
#         prefix = 'language_model.model.layers'
#         suffix = 'feed_forward.act_fn'
#     elif model_type == 'qwen2vl':                                          # Line 3: Qwen2.5-VL architecture
#         prefix = 'model.language_model.layers'
#         suffix = 'mlp.act_fn'
#     elif model_type == 'llava-ov':                                         # Line 4: LLaVA-OneVision architecture
#         prefix = 'model.language_model.layers'
#         suffix = 'mlp.act_fn'
#     else:                                                                  # Line 5: original liuhaotian LLaVA
#         prefix = 'model.layers'
#         suffix = 'mlp.act_fn'
#     return [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]            # Line 6: list of layer hook names


# def load_neuron_labels(label_dir, taxonomy, layer_names, n_layers):
#     """Load per-layer neuron modality labels from the classification pipeline.

#     Reads either neuron_labels.json (fixed-threshold) or
#     neuron_labels_permutation.json (permutation-test) from each
#     layer's subdirectory.

#     Returns:
#         labels: dict {layer_idx: list of label dicts}
#                 each label dict has at minimum {'neuron_idx': int, 'label': str}
#         flat_labels: numpy array of shape (n_layers * n_neurons,) with
#                      string labels for every neuron in the model
#     """
#     label_filename = ('neuron_labels.json' if taxonomy == 'ft'             # Line 1: select filename by taxonomy
#                       else 'neuron_labels_permutation.json')

#     # First try the merged "_all" file (single JSON with all layers)
#     merged_name = label_filename.replace('.json', '_all.json')             # Line 2: e.g. neuron_labels_permutation_all.json
#     merged_path = os.path.join(label_dir, merged_name)

#     labels = {}                                                            # Line 3: {layer_idx_str: label_list}

#     if os.path.isfile(merged_path):                                        # Line 4: merged file exists
#         print(f'Loading merged labels from {merged_path}')
#         with open(merged_path) as f:
#             labels = json.load(f)                                          # Line 5: keys are layer index strings
#         print(f'  Loaded {len(labels)} layers from merged file')
#     else:                                                                   # Line 6: fall back to per-layer files
#         print(f'Loading per-layer labels from {label_dir}')
#         for l in range(n_layers):                                          # Line 7: iterate over all layers
#             layer_name = layer_names[l]
#             label_path = os.path.join(label_dir, layer_name, label_filename)
#             if os.path.isfile(label_path):                                 # Line 8: file exists for this layer
#                 with open(label_path) as f:
#                     labels[str(l)] = json.load(f)                          # Line 9: store under string key
#             else:
#                 print(f'  WARNING: missing {label_path}')

#     # Build flat label array: (n_total_neurons,) ordered by (layer, neuron_idx)
#     flat_labels = []                                                       # Line 10: will be filled layer-by-layer
#     for l in range(n_layers):                                              # Line 11: iterate in layer order
#         key = str(l)
#         if key in labels:                                                  # Line 12: layer data exists
#             layer_labels = labels[key]
#             # Sort by neuron_idx to ensure consistent ordering
#             layer_labels_sorted = sorted(layer_labels,                     # Line 13: sort by neuron index
#                                          key=lambda x: x['neuron_idx'])
#             for entry in layer_labels_sorted:                              # Line 14: extract label strings
#                 flat_labels.append(entry['label'])
#         else:
#             print(f'  Layer {l}: no labels found, filling with "missing"')
#             # Unknown number of neurons — use n_neurons from first available layer
#             if labels:                                                     # Line 15: infer neuron count
#                 sample_key = next(iter(labels))
#                 n_neurons_layer = len(labels[sample_key])
#             else:
#                 n_neurons_layer = 11008                                    # Line 16: fallback to LLaMA-2 default
#             flat_labels.extend(['missing'] * n_neurons_layer)              # Line 17: placeholder for missing layers

#     flat_labels = np.array(flat_labels)                                    # Line 18: convert to numpy for vectorized ops
#     print(f'Loaded {len(flat_labels)} neuron labels total')

#     # Print category distribution
#     categories = ['visual', 'text', 'multimodal', 'unknown']              # Line 19: standard modality categories
#     for cat in categories:                                                 # Line 20: print count and percentage
#         count = (flat_labels == cat).sum()
#         print(f'  {cat:12s}: {count:6,} ({100 * count / len(flat_labels):.1f}%)')

#     return labels, flat_labels


# # ═══════════════════════════════════════════════════════════════════════
# # Section 3 — Phase 1: Identify hallucination-driving neurons (multi-GPU)
# # ═══════════════════════════════════════════════════════════════════════

# def load_pope_data(pope_path, n_questions=None):
#     """Load POPE evaluation questions from JSONL.

#     Each line: {"question": "...", "answer": "yes"/"no",
#                 "image": "COCO_val2014_000000XXXXXX.jpg", ...}

#     Returns list of dicts, optionally truncated to n_questions.
#     """
#     questions = []                                                         # Line 1: accumulate parsed POPE entries
#     with open(pope_path) as f:                                             # Line 2: open JSONL file
#         for line in f:                                                     # Line 3: iterate line-by-line
#             line = line.strip()
#             if line:                                                       # Line 4: skip empty lines
#                 questions.append(json.loads(line))                         # Line 5: parse each JSON line
#     if n_questions is not None and n_questions < len(questions):            # Line 6: optionally truncate
#         questions = questions[:n_questions]
#     print(f'Loaded {len(questions)} POPE questions from {pope_path}')
#     return questions                                                       # Line 7: list of POPE question dicts


# def _resolve_module(model, dotted_name):
#     """Walk the model tree to find the module at a dotted attribute path.

#     Example: _resolve_module(model, 'model.layers.0.mlp.act_fn')
#     traverses model → model.model → model.model.layers → [0] → .mlp → .act_fn

#     Args:
#         model: root model object
#         dotted_name: dot-separated path like 'model.layers.0.mlp.act_fn'

#     Returns:
#         module: the PyTorch module at the given path
#     """
#     parts = dotted_name.split('.')                                         # Line 1: split into path components
#     mod = model                                                            # Line 2: start from root
#     for p in parts:                                                        # Line 3: walk each component
#         if p.isdigit():                                                    # Line 4: numeric index into ModuleList
#             mod = mod[int(p)]
#         else:                                                              # Line 5: named attribute
#             mod = getattr(mod, p)
#     return mod                                                             # Line 6: the resolved module


# def _extract_down_proj_norms(model, model_type, layer_names, n_layers):
#     """Extract L2 norm of each down_proj column for all layers.

#     In the SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
#     Each neuron j's contribution to the residual stream is:
#         contribution_j = activation_j * down_proj[:, j]
#     The column norm ||down_proj[:, j]|| scales how much neuron j matters.

#     Args:
#         model: loaded VLM model
#         model_type: backend string (e.g. 'llava-hf', 'internvl')
#         layer_names: list of act_fn hook-point names from get_layer_names()
#         n_layers: number of LLM layers

#     Returns:
#         norms: dict {layer_idx: tensor of shape (n_neurons,)}
#     """
#     import torch                                                           # Line 1: needed for detach/float/norm
#     norms = {}                                                             # Line 2: output dict
#     for layer_idx in range(n_layers):                                      # Line 3: iterate layers
#         act_fn_path = layer_names[layer_idx]                               # Line 4: e.g. 'model.language_model.layers.0.mlp.act_fn'
#         if model_type == 'internvl':                                       # Line 5: InternLM2 names down_proj as 'w2'
#             dp_path = act_fn_path.replace('.act_fn', '.w2')
#         else:                                                              # Line 6: all others use 'down_proj'
#             dp_path = act_fn_path.replace('.act_fn', '.down_proj')
#         dp_module = _resolve_module(model, dp_path)                        # Line 7: get the down_proj linear layer
#         W_down = dp_module.weight.detach().float()                         # Line 8: shape (hidden_size, n_neurons)
#         norms[layer_idx] = W_down.norm(dim=0).cpu()                        # Line 9: L2 norm per column → (n_neurons,)
#     return norms                                                           # Line 10: {layer_idx: tensor}


# def build_contrastive_pope_set(args, device='cuda:0'):
#     """Build a contrastive POPE set and compute per-neuron CETT-diff scores.

#     For each question, samples args.contrastive_samples responses at
#     temperature=0.7. Keeps only questions where ALL samples give the
#     same yes/no answer — either consistently correct or consistently
#     wrong. Caps at args.contrastive_cap_per_split per POPE split.

#     After filtering, computes CETT (Contribution of Each neuron To
#     the hidden state) at the answer token position for each clean
#     question, then computes CETT-diff = mean(CETT on hallucinating)
#     − mean(CETT on correct). This provides a per-neuron hallucination
#     score independent of the ablation-based ΔH scores.

#     Loads all 3 POPE splits (random, popular, adversarial), samples
#     args.contrastive_start_per_split from each, filters, caps, and
#     saves the result to output_dir/contrastive_pope.jsonl.

#     Args:
#         args: parsed arguments namespace
#         device: torch device string for model loading (default: 'cuda:0')

#     Returns:
#         contrastive_path: str — path to the saved contrastive JSONL file
#         n_clean: int — total number of clean questions retained
#         cett_diff: dict {(layer_idx, neuron_idx): float} — per-neuron
#                    CETT-diff score (positive = higher contribution during
#                    hallucination). None if computation fails.
#     """
#     import torch                                                           # Line 1: needed for model loading
#     from PIL import Image                                                  # Line 2: image loading

#     print(f'\n{"="*60}')
#     print(f'CONTRASTIVE PREPROCESSING')
#     print(f'  Splits: random, popular, adversarial')
#     print(f'  Start per split: {args.contrastive_start_per_split}')
#     print(f'  Samples per question: {args.contrastive_samples}')
#     print(f'  Cap per split: {args.contrastive_cap_per_split}')
#     print(f'{"="*60}\n')

#     # ── Resolve POPE splits directory ──
#     if args.pope_splits_dir is not None:                                   # Line 3: explicit dir
#         splits_dir = args.pope_splits_dir
#     else:                                                                  # Line 4: infer from pope_path parent
#         splits_dir = os.path.dirname(args.pope_path)

#     split_names = ['random', 'popular', 'adversarial']                     # Line 5: the 3 POPE difficulty levels
#     split_files = {s: os.path.join(splits_dir, f'coco_pope_{s}.json')      # Line 6: expected filenames
#                    for s in split_names}

#     # Verify all split files exist
#     for s, path in split_files.items():                                    # Line 7: check existence
#         if not os.path.exists(path):
#             raise FileNotFoundError(
#                 f'POPE split file not found: {path}. '
#                 f'Set --pope_splits_dir to the directory containing all 3 splits.')

#     # ── Load and sample from each split ──
#     rng = np.random.RandomState(args.seed)                                 # Line 8: reproducible sampling
#     all_candidates = []                                                    # Line 9: accumulate across splits

#     for split_name in split_names:                                         # Line 10: iterate splits
#         raw = load_pope_data(split_files[split_name])                      # Line 11: load full split
#         if len(raw) > args.contrastive_start_per_split:                    # Line 12: subsample if needed
#             indices = rng.choice(len(raw), size=args.contrastive_start_per_split,
#                                  replace=False)
#             raw = [raw[i] for i in sorted(indices)]                        # Line 13: keep order for reproducibility
#         for q in raw:                                                      # Line 14: tag each question with its split
#             q['pope_split'] = split_name
#         all_candidates.append(raw)
#         print(f'  {split_name}: {len(raw)} starting questions')

#     # ── Load model for contrastive sampling ──
#     print(f'\n  Loading model on {device} for contrastive sampling...')
#     if args.model_type == 'llava-hf':
#         from transformers import AutoProcessor, LlavaForConditionalGeneration
#         processor = AutoProcessor.from_pretrained(args.model_path)
#         model = LlavaForConditionalGeneration.from_pretrained(
#             args.model_path, torch_dtype=torch.float16,
#             low_cpu_mem_usage=True).to(device).eval()
#     elif args.model_type == 'llava-liuhaotian':
#         _PROJECT_ROOT = os.path.abspath(os.path.join(
#             os.path.dirname(__file__), '..', '..'))
#         _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')
#         if _LLAVA_PATH not in sys.path:
#             sys.path.insert(0, _LLAVA_PATH)
#         from llava.model.builder import load_pretrained_model
#         from llava.mm_utils import get_model_name_from_path
#         model_name = get_model_name_from_path(args.model_path)
#         tokenizer, model, image_processor, _ = load_pretrained_model(
#             args.model_path, None, model_name,
#             device_map=device, torch_dtype=torch.float16)
#         processor = (tokenizer, image_processor)
#     elif args.model_type == 'llava-ov':
#         from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
#         processor = AutoProcessor.from_pretrained(args.model_path)
#         model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#             args.model_path, torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True).to(device).eval()
#     elif args.model_type == 'internvl':
#         from transformers import AutoModel, AutoTokenizer
#         model = AutoModel.from_pretrained(
#             args.model_path, torch_dtype=torch.bfloat16,
#             trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
#         processor = AutoTokenizer.from_pretrained(
#             args.model_path, trust_remote_code=True)
#     elif args.model_type == 'qwen2vl':
#         from transformers import AutoModelForVision2Seq, AutoProcessor
#         processor = AutoProcessor.from_pretrained(args.model_path)
#         model = AutoModelForVision2Seq.from_pretrained(
#             args.model_path, torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True).to(device).eval()
#     else:
#         raise ValueError(f'Unsupported model_type: {args.model_type}')
#     print(f'  Model loaded.')

#     # ── Helper: generate one yes/no answer for a POPE question ──
#     def generate_answer(q, do_sample=True, temperature=0.7):
#         """Generate a single yes/no answer for a POPE question.

#         Args:
#             q: POPE question dict with 'image' and 'text'/'question' keys
#             do_sample: if True, sample with temperature; if False, greedy
#             temperature: sampling temperature (only used if do_sample=True)

#         Returns:
#             pred_yes: bool — True if model answered 'yes'
#             success: bool — True if generation succeeded
#         """
#         img_filename = q['image']
#         img_path = os.path.join(args.pope_img_dir, img_filename)
#         question_text = q.get('text', q.get('question', ''))

#         try:
#             img = Image.open(img_path).convert('RGB')
#         except Exception:
#             return False, False                                            # failed to load image

#         gen_kwargs = dict(max_new_tokens=10,
#                           do_sample=do_sample,
#                           temperature=temperature if do_sample else None)
#         if not do_sample:
#             gen_kwargs.pop('temperature', None)

#         with torch.no_grad():
#             if args.model_type == 'llava-hf':
#                 prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
#                 inputs = processor(text=prompt, images=img,
#                                    return_tensors='pt').to(device)
#                 output_ids = model.generate(**inputs, **gen_kwargs)
#                 generated = processor.decode(output_ids[0],
#                                              skip_special_tokens=True)
#             elif args.model_type == 'llava-liuhaotian':
#                 from llava.conversation import conv_templates
#                 from llava.mm_utils import (tokenizer_image_token,
#                                             process_images)
#                 conv = conv_templates['v1'].copy()
#                 conv.append_message(conv.roles[0], f'<image>\n{question_text}')
#                 conv.append_message(conv.roles[1], None)
#                 prompt = conv.get_prompt()
#                 tok = processor[0] if isinstance(processor, tuple) else processor
#                 img_proc = processor[1] if isinstance(processor, tuple) else None
#                 input_ids = tokenizer_image_token(
#                     prompt, tok, return_tensors='pt').unsqueeze(0).to(device)
#                 img_tensor = process_images(
#                     [img], img_proc, model.config
#                 ).to(device, dtype=torch.float16)
#                 output_ids = model.generate(
#                     input_ids=input_ids, images=img_tensor, **gen_kwargs)
#                 generated = tok.decode(output_ids[0],
#                                        skip_special_tokens=True)
#             elif args.model_type == 'llava-ov':
#                 messages = [{'role': 'user', 'content': [
#                     {'type': 'image'}, {'type': 'text', 'text': question_text}]}]
#                 prompt_text = processor.apply_chat_template(
#                     messages, tokenize=False, add_generation_prompt=True)
#                 inputs = processor(images=img, text=prompt_text,
#                                    return_tensors='pt').to(device)
#                 output_ids = model.generate(**inputs, **gen_kwargs)
#                 prompt_len = inputs['input_ids'].shape[1]
#                 generated = processor.decode(
#                     output_ids[0][prompt_len:], skip_special_tokens=True)
#             elif args.model_type == 'internvl':
#                 import torchvision.transforms as T
#                 from torchvision.transforms.functional import InterpolationMode
#                 _tf = T.Compose([
#                     T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
#                     T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
#                     T.ToTensor(),
#                     T.Normalize(mean=(0.485, 0.456, 0.406),
#                                 std=(0.229, 0.224, 0.225)),
#                 ])
#                 pixel_values = _tf(img).unsqueeze(0).to(
#                     torch.bfloat16).to(device)
#                 question_prompt = f'<image>\n{question_text}'
#                 gen_config = dict(max_new_tokens=10, do_sample=do_sample)
#                 if do_sample:
#                     gen_config['temperature'] = temperature
#                 response = model.chat(processor, pixel_values,
#                                       question_prompt, gen_config)
#                 generated = response
#             elif args.model_type == 'qwen2vl':
#                 messages = [{'role': 'user', 'content': [
#                     {'type': 'image', 'image': img},
#                     {'type': 'text', 'text': question_text}]}]
#                 prompt_text = processor.apply_chat_template(
#                     messages, tokenize=False, add_generation_prompt=True)
#                 inputs = processor(images=img, text=prompt_text,
#                                    return_tensors='pt').to(device)
#                 output_ids = model.generate(**inputs, **gen_kwargs)
#                 prompt_len = inputs['input_ids'].shape[1]
#                 generated = processor.decode(
#                     output_ids[0][prompt_len:], skip_special_tokens=True)
#             else:
#                 return False, False

#         answer = generated.strip().lower()
#         pred_yes = 'yes' in answer
#         return pred_yes, True

#     # ── Run contrastive sampling ──
#     from tqdm import tqdm                                                  # Line 15: progress bar

#     contrastive_questions = []                                             # Line 16: accumulate clean questions
#     stats = {s: {'total': 0, 'consistent_correct': 0,                     # Line 17: per-split stats
#                  'consistent_halluc': 0, 'inconsistent': 0}
#              for s in split_names}

#     for split_questions in all_candidates:                                 # Line 18: iterate each split's candidates
#         split_name = split_questions[0]['pope_split']                       # Line 19: get split name from first q
#         split_clean = []                                                   # Line 20: clean questions for this split

#         for q in tqdm(split_questions,                                     # Line 21: iterate questions with progress
#                        desc=f'Contrastive [{split_name}]'):
#             gt_answer = q.get('label', q.get('answer', '')).strip().lower()  # Line 22: ground truth
#             gt_yes = gt_answer == 'yes'                                    # Line 23: bool ground truth

#             # Sample multiple responses
#             predictions = []                                               # Line 24: collect per-sample predictions
#             all_succeeded = True                                           # Line 25: track failures
#             for _ in range(args.contrastive_samples):                      # Line 26: sample N times
#                 pred_yes, success = generate_answer(q, do_sample=True,
#                                                      temperature=0.7)
#                 if not success:                                            # Line 27: image load failed
#                     all_succeeded = False
#                     break
#                 predictions.append(pred_yes)                               # Line 28: record this sample's answer

#             if not all_succeeded:                                          # Line 29: skip broken images
#                 continue

#             stats[split_name]['total'] += 1                                # Line 30: count evaluated

#             # Check consistency: all predictions must agree
#             all_yes = all(predictions)                                     # Line 31: all said yes
#             all_no = all(not p for p in predictions)                       # Line 32: all said no

#             if not (all_yes or all_no):                                    # Line 33: inconsistent — discard
#                 stats[split_name]['inconsistent'] += 1
#                 continue

#             # Determine contrastive label
#             pred_yes_consistent = all_yes                                  # Line 34: consistent prediction
#             is_correct = (pred_yes_consistent == gt_yes)                   # Line 35: matches ground truth?

#             if is_correct:                                                 # Line 36: consistently correct
#                 stats[split_name]['consistent_correct'] += 1
#                 q['contrastive_label'] = 'faithful'
#             else:                                                          # Line 37: consistently wrong
#                 stats[split_name]['consistent_halluc'] += 1
#                 q['contrastive_label'] = 'hallucinated'

#             split_clean.append(q)                                          # Line 38: add to clean set

#         # Cap this split
#         if len(split_clean) > args.contrastive_cap_per_split:             # Line 39: cap if too many
#             rng.shuffle(split_clean)                                       # Line 40: shuffle before capping
#             split_clean = split_clean[:args.contrastive_cap_per_split]

#         contrastive_questions.extend(split_clean)                          # Line 41: add to global set
#         print(f'  {split_name}: {stats[split_name]["total"]} evaluated → '
#               f'{stats[split_name]["consistent_correct"]} correct, '
#               f'{stats[split_name]["consistent_halluc"]} hallucinated, '
#               f'{stats[split_name]["inconsistent"]} inconsistent → '
#               f'{len(split_clean)} kept (capped at {args.contrastive_cap_per_split})')

#     # ── Save contrastive set ──
#     contrastive_path = os.path.join(args.output_dir,                       # Line 42: output path
#                                      'contrastive_pope.jsonl')
#     with open(contrastive_path, 'w') as f:                                 # Line 43: write JSONL
#         for q in contrastive_questions:
#             f.write(json.dumps(q) + '\n')                                  # Line 44: one JSON object per line

#     # Save contrastive stats for reproducibility
#     stats_path = os.path.join(args.output_dir, 'contrastive_stats.json')   # Line 45: stats output path
#     with open(stats_path, 'w') as f:
#         json.dump(stats, f, indent=2)

#     print(f'\n  Contrastive set: {len(contrastive_questions)} questions '
#           f'saved to {contrastive_path}')
#     print(f'  Stats saved to {stats_path}')
#     print(f'\n  Paper sentence: "We sampled {args.contrastive_samples} responses '
#           f'per question from {args.contrastive_start_per_split} POPE items per '
#           f'split (random, popular, adversarial), retaining '
#           f'{args.contrastive_cap_per_split} per split where the model was '
#           f'consistently correct or consistently incorrect, yielding '
#           f'N={len(contrastive_questions)} high-confidence contrastive samples '
#           f'balanced across difficulty levels."')

#     # ── Compute CETT-diff from contrastive set ──────────────────────
#     # CETT = |activation_j| × ||down_proj_col_j|| measures each neuron's
#     # actual contribution to the layer output. By comparing CETT at the
#     # answer token between hallucinating and correct questions, we get a
#     # per-neuron hallucination score independent of ablation-based ΔH.
#     # For POPE, the answer token is always the first generated token
#     # ("yes"/"no"), so no GPT-4o is needed to identify it.
#     cett_diff = None                                                       # Line C1: default if computation fails
#     n_layers = args.n_layers                                               # Line C2: total LLM layers
#     layer_names = get_layer_names(args.model_type, n_layers)               # Line C3: act_fn hook-point names

#     try:
#         print(f'\n{"="*60}')
#         print(f'CETT-DIFF SCORING on {len(contrastive_questions)} '
#               f'contrastive questions')
#         print(f'{"="*60}\n')

#         # Step 1: Extract down_proj column norms (once, from weights)
#         print(f'  Extracting down_proj column norms ({n_layers} layers)...')
#         down_norms = _extract_down_proj_norms(                             # Line C4: ||down_proj[:, j]|| per neuron
#             model, args.model_type, layer_names, n_layers)
#         n_neurons = down_norms[0].shape[0]                                 # Line C5: neurons per layer
#         print(f'  {n_layers} layers × {n_neurons} neurons')

#         # Step 2: Register hooks on act_fn at every layer.
#         # During model.generate(), the first forward pass (prefill)
#         # processes the full prompt. The activations at the last prompt
#         # token determine the answer. We capture only the prefill pass
#         # (seq_len > 1) and take the last position.
#         act_captures = {}                                                  # Line C6: {layer_idx: tensor (n_neurons,)}
#         hook_recorded = {}                                                 # Line C7: {layer_idx: bool} first-pass flag
#         hooks = []                                                         # Line C8: hook handles for cleanup

#         for layer_idx in range(n_layers):                                  # Line C9: register one hook per layer
#             act_fn_module = _resolve_module(model, layer_names[layer_idx]) # Line C10: act_fn module
#             hook_recorded[layer_idx] = False                               # Line C11: reset flag

#             def make_hook(lidx):                                           # Line C12: closure to capture layer index
#                 def hook_fn(module, input, output):
#                     if hook_recorded[lidx]:                                # Line C13: skip decode steps
#                         return
#                     out = output[0] if isinstance(output, tuple) else output  # Line C14: handle tuple output
#                     act_captures[lidx] = out[0, -1, :].detach().cpu()     # Line C15: last prompt token activations
#                     hook_recorded[lidx] = True                             # Line C16: mark as recorded
#                 return hook_fn

#             h = act_fn_module.register_forward_hook(make_hook(layer_idx))  # Line C17: attach hook
#             hooks.append(h)                                                # Line C18: save handle

#         # Step 3: Run greedy forward pass per question, accumulate CETT
#         # Use running sums to avoid storing per-question arrays.
#         halluc_sum = {l: np.zeros(n_neurons, dtype=np.float64)             # Line C19: sum of CETT for halluc questions
#                       for l in range(n_layers)}
#         correct_sum = {l: np.zeros(n_neurons, dtype=np.float64)            # Line C20: sum of CETT for correct questions
#                        for l in range(n_layers)}
#         n_halluc = 0                                                       # Line C21: count of hallucinated questions
#         n_correct = 0                                                      # Line C22: count of correct questions

#         for q in tqdm(contrastive_questions, desc='CETT scoring'):         # Line C23: iterate clean questions
#             # Reset hook flags for this question
#             for lidx in hook_recorded:                                     # Line C24: reset all layer flags
#                 hook_recorded[lidx] = False
#             act_captures.clear()                                           # Line C25: clear previous captures

#             # Greedy forward pass (deterministic — activations at last
#             # prompt position are identical regardless of temperature)
#             pred_yes, success = generate_answer(q, do_sample=False)        # Line C26: greedy decode, hooks fire
#             if not success:                                                # Line C27: skip failed images
#                 continue

#             # Compute CETT per neuron and accumulate
#             is_halluc = q['contrastive_label'] == 'hallucinated'           # Line C28: check contrastive label
#             target = halluc_sum if is_halluc else correct_sum              # Line C29: pick accumulator

#             for lidx in range(n_layers):                                   # Line C30: iterate layers
#                 if lidx not in act_captures:                               # Line C31: skip if hook didn't fire
#                     continue
#                 acts = act_captures[lidx].float().numpy()                  # Line C32: (n_neurons,) activation values
#                 norms_np = down_norms[lidx].numpy()                        # Line C33: (n_neurons,) column norms
#                 cett = np.abs(acts) * norms_np                             # Line C34: CETT_j = |act_j| × ||down_proj_col_j||
#                 target[lidx] += cett                                       # Line C35: accumulate running sum

#             if is_halluc:                                                  # Line C36: increment counter
#                 n_halluc += 1
#             else:
#                 n_correct += 1

#         # Step 4: Remove hooks
#         for h in hooks:                                                    # Line C37: clean up all hooks
#             h.remove()

#         # Step 5: Compute CETT-diff = mean(halluc) − mean(correct)
#         print(f'\n  CETT scored: {n_halluc} hallucinated, '
#               f'{n_correct} correct questions')

#         if n_halluc > 0 and n_correct > 0:                                # Line C38: need both groups
#             cett_diff = {}                                                 # Line C39: {(layer, neuron): diff}
#             for lidx in range(n_layers):                                   # Line C40: iterate layers
#                 mean_h = halluc_sum[lidx] / n_halluc                       # Line C41: mean CETT for halluc group
#                 mean_c = correct_sum[lidx] / n_correct                     # Line C42: mean CETT for correct group
#                 diff = mean_h - mean_c                                     # Line C43: per-neuron CETT-diff
#                 for n_idx in range(n_neurons):                             # Line C44: store per-neuron scores
#                     cett_diff[(lidx, n_idx)] = float(diff[n_idx])

#             # Save CETT-diff scores
#             cett_diff_path = os.path.join(args.output_dir,                 # Line C45: output path
#                                           'cett_diff_scores.json')
#             cett_serializable = {f'{k[0]}_{k[1]}': v                       # Line C46: serialize tuple keys
#                                  for k, v in cett_diff.items()}
#             with open(cett_diff_path, 'w') as f:                           # Line C47: write JSON
#                 json.dump(cett_serializable, f, indent=2)

#             # Print top CETT-diff neurons as sanity check
#             top_neurons = sorted(cett_diff.items(),                        # Line C48: sort by CETT-diff descending
#                                  key=lambda x: x[1], reverse=True)[:10]
#             print(f'  Top 10 neurons by CETT-diff (positive = hallucination-driving):')
#             for (l, n), score in top_neurons:                              # Line C49: print top neurons
#                 print(f'    layer {l:2d} neuron {n:5d}: CETT-diff = {score:.6f}')
#             print(f'  CETT-diff scores saved to {cett_diff_path}')
#         else:
#             print(f'  WARNING: Insufficient data for CETT-diff '
#                   f'(halluc={n_halluc}, correct={n_correct})')

#     except Exception as e:                                                 # Line C50: catch any CETT errors
#         print(f'  WARNING: CETT-diff computation failed: {e}')
#         print(f'  Contrastive set is still valid. CETT-diff will be None.')
#         import traceback
#         traceback.print_exc()

#     # ── Free model memory ──
#     del model                                                              # Line 46: release model weights
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()                                           # Line 47: free GPU memory
#     print(f'  Contrastive model unloaded.\n')

#     return contrastive_path, len(contrastive_questions), cett_diff


# def compute_hallucination_rate(model, processor, pope_questions,
#                                pope_img_dir, device, model_type,
#                                ablation_hook=None):
#     """Run POPE evaluation and compute hallucination rate.

#     For each yes/no question, generates the model's answer and checks
#     if the model says "yes" when the ground truth is "no" (hallucination).

#     Args:
#         model: loaded VLM model
#         processor: tokenizer/processor for the model
#         pope_questions: list of POPE question dicts
#         pope_img_dir: directory with COCO val images
#         device: torch device string (e.g. "cuda:0")
#         model_type: backend identifier string
#         ablation_hook: optional context manager that applies neuron ablation

#     Returns:
#         dict with keys: hallucination_rate, accuracy, n_correct, n_total,
#                         n_hallucinated (said yes when answer is no)
#     """
#     import torch                                                           # Line 1: import here to avoid top-level GPU init
#     from PIL import Image                                                  # Line 2: image loading

#     n_correct = 0                                                          # Line 3: count of correct predictions
#     n_hallucinated = 0                                                     # Line 4: count of false "yes" answers
#     n_total = 0                                                            # Line 5: total questions evaluated

#     model.eval()                                                           # Line 6: ensure eval mode

#     for q in pope_questions:                                               # Line 7: iterate over POPE questions
#         img_filename = q['image']                                          # Line 8: e.g. "COCO_val2014_000000XXXXXX.jpg"
#         img_path = os.path.join(pope_img_dir, img_filename)                # Line 9: full path to image
#         question_text = q.get("text", q.get("question", ""))                 # POPE uses text key
#         gt_answer = q.get("label", q.get("answer", "")).strip().lower()    # POPE uses label key

#         try:
#             img = Image.open(img_path).convert('RGB')                      # Line 12: load and convert to RGB
#         except Exception:
#             continue                                                       # Line 13: skip broken images

#         # Prepare inputs based on model type
#         if model_type in ('llava-hf',):                                    # Line 14: HF LLaVA input format
#             prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
#             inputs = processor(text=prompt, images=img,                    # Line 15: tokenize text + preprocess image
#                                return_tensors='pt').to(device)
#         elif model_type == 'llava-liuhaotian':                             # Line 16: original LLaVA input format
#             from llava.conversation import conv_templates                   # Line 17: LLaVA conversation template
#             from llava.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria  # Line 18: LLaVA utils

#             conv = conv_templates["v1"].copy()                             # Line 19: create fresh conversation
#             conv.append_message(conv.roles[0],                             # Line 20: USER turn with image + question
#                                 f"<image>\n{question_text}")
#             conv.append_message(conv.roles[1], None)                       # Line 21: empty ASSISTANT turn for generation
#             prompt = conv.get_prompt()                                      # Line 22: format into model prompt string

#             tokenizer = processor[0] if isinstance(processor, tuple) else processor  # Line 23: extract tokenizer
#             image_proc = processor[1] if isinstance(processor, tuple) else None      # Line 24: extract image processor

#             input_ids = tokenizer_image_token(                             # Line 25: tokenize with <image> placeholder handling
#                 prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
#             img_tensor = process_images([img], image_proc,                 # Line 26: preprocess image to tensor
#                                         model.config).to(device, dtype=torch.float16)
#             inputs = {'input_ids': input_ids, 'images': img_tensor}        # Line 27: pack into inputs dict
#         elif model_type == 'llava-ov':                                    # LLaVA-OneVision input format
#             messages = [{'role': 'user', 'content': [
#                 {'type': 'image'},
#                 {'type': 'text', 'text': question_text},
#             ]}]
#             prompt_text = processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True)     # Qwen2 chat template
#             inputs = processor(
#                 images=img, text=prompt_text,
#                 return_tensors='pt').to(device)                           # processor handles image + text
#         elif model_type == 'internvl':                                    # InternVL2.5 — uses model.chat()
#             import torchvision.transforms as T
#             from torchvision.transforms.functional import InterpolationMode
#             _mean = (0.485, 0.456, 0.406)
#             _std  = (0.229, 0.224, 0.225)
#             _tf = T.Compose([
#                 T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
#                 T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#                 T.Normalize(mean=_mean, std=_std),
#             ])
#             pixel_values = _tf(img).unsqueeze(0).to(torch.bfloat16).to(device)
#             question_prompt = f'<image>\n{question_text}'
#             generation_config = dict(max_new_tokens=10, do_sample=False)
#             with torch.no_grad():
#                 if ablation_hook is not None:
#                     with ablation_hook:
#                         response = model.chat(processor, pixel_values, question_prompt, generation_config)
#                 else:
#                     response = model.chat(processor, pixel_values, question_prompt, generation_config)
#             answer = response.strip().lower()
#             pred_yes = 'yes' in answer
#             if (pred_yes and gt_answer == 'yes') or (not pred_yes and gt_answer == 'no'):
#                 n_correct += 1
#             if pred_yes and gt_answer == 'no':
#                 n_hallucinated += 1
#             n_total += 1
#             continue                                                       # skip common generate/decode
#         elif model_type == 'qwen2vl':                                     # Qwen2.5-VL input format
#             messages = [{'role': 'user', 'content': [
#                 {'type': 'image', 'image': img},
#                 {'type': 'text', 'text': question_text},
#             ]}]
#             prompt_text = processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True)
#             inputs = processor(
#                 images=img, text=prompt_text,
#                 return_tensors='pt').to(device)
#         else:
#             continue                                                       # skip unsupported model types

#         # Generate answer with optional ablation hook
#         with torch.no_grad():                                              # Line 29: no gradient computation needed
#             if ablation_hook is not None:                                   # Line 30: if we're ablating neurons
#                 with ablation_hook:                                        # Line 31: apply the ablation context manager
#                     output_ids = model.generate(**inputs, max_new_tokens=10,  # Line 32: generate short answer
#                                                 do_sample=False)
#             else:                                                          # Line 33: baseline (no ablation)
#                 output_ids = model.generate(**inputs, max_new_tokens=10,   # Line 34: generate without ablation
#                                             do_sample=False)

#         # Decode generated answer
#         if model_type == 'llava-hf':
#             generated = processor.decode(output_ids[0], skip_special_tokens=True)
#         elif model_type in ('llava-ov', 'qwen2vl'):
#             prompt_len = inputs['input_ids'].shape[1]
#             generated = processor.decode(
#                 output_ids[0][prompt_len:], skip_special_tokens=True)
#         else:
#             tokenizer = processor[0] if isinstance(processor, tuple) else processor
#             generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#         # Extract yes/no from generated text
#         answer = generated.strip().lower()                                 # Line 39: normalise to lowercase
#         pred_yes = 'yes' in answer                                         # Line 40: check if model said yes

#         # Score: correct if prediction matches ground truth
#         if (pred_yes and gt_answer == 'yes') or \
#            (not pred_yes and gt_answer == 'no'):                           # Line 41: prediction matches ground truth
#             n_correct += 1
#         if pred_yes and gt_answer == 'no':                                 # Line 42: hallucination = said yes when should be no
#             n_hallucinated += 1
#         n_total += 1                                                       # Line 43: increment total counter

#     hallucination_rate = n_hallucinated / max(n_total, 1)                  # Line 44: fraction of hallucinated answers
#     accuracy = n_correct / max(n_total, 1)                                 # Line 45: overall accuracy

#     return {                                                               # Line 46: return evaluation results
#         'hallucination_rate': hallucination_rate,
#         'accuracy': accuracy,
#         'n_correct': n_correct,
#         'n_hallucinated': n_hallucinated,
#         'n_total': n_total,
#     }


# def ablation_worker(gpu_id, layer_range, args, return_dict):
#     """Worker function for multi-GPU ablation — runs on a single GPU.

#     Ablates neurons layer-by-layer in batches, measuring the change
#     in hallucination rate (ΔH) for each batch. Assigns the batch-level
#     ΔH equally to each neuron in the batch (approximation for tractability).

#     Args:
#         gpu_id: integer GPU index (0, 1, 2, ...)
#         layer_range: tuple (start_layer, end_layer) — exclusive end
#         args: parsed arguments namespace
#         return_dict: multiprocessing.Manager dict for collecting results
#     """
#     import torch                                                           # Line 1: import torch inside worker (fork safety)
#     from baukit import TraceDict                                           # Line 2: hook-based activation interception
#     from PIL import Image                                                  # Line 3: image loading

#     device = f'cuda:{gpu_id}'                                              # Line 4: assign this worker to its GPU
#     print(f'[GPU {gpu_id}] Processing layers {layer_range[0]}-{layer_range[1]-1}')

#     # Load model onto this GPU
#     if args.model_type == 'llava-hf':                                      # Line 5: load HF LLaVA
#         from transformers import AutoProcessor, LlavaForConditionalGeneration
#         processor = AutoProcessor.from_pretrained(args.model_path)         # Line 6: load tokenizer + image processor
#         model = LlavaForConditionalGeneration.from_pretrained(             # Line 7: load model weights
#             args.model_path, torch_dtype=torch.float16,
#             low_cpu_mem_usage=True).to(device).eval()
#     elif args.model_type == 'llava-liuhaotian':                            # Line 8: load original LLaVA
#         _PROJECT_ROOT = os.path.abspath(os.path.join(                      # Line 9: compute project root
#             os.path.dirname(__file__), '..', '..'))
#         _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')                # Line 10: path to cloned LLaVA repo
#         if _LLAVA_PATH not in sys.path:
#             sys.path.insert(0, _LLAVA_PATH)                                # Line 11: add to import path
#         from llava.model.builder import load_pretrained_model              # Line 12: LLaVA model loader
#         from llava.mm_utils import get_model_name_from_path                # Line 13: derive model name from path
#         model_name = get_model_name_from_path(args.model_path)             # Line 14: e.g. "llava-v1.5-7b"
#         tokenizer, model, image_processor, _ = load_pretrained_model(      # Line 15: load all model components
#             args.model_path, None, model_name,
#             device_map=device, torch_dtype=torch.float16)
#         processor = (tokenizer, image_processor)                           # Line 16: pack as tuple for consistency
#     elif args.model_type == 'llava-ov':                                    # LLaVA-OneVision (Qwen2 backbone)
#         from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
#         processor = AutoProcessor.from_pretrained(args.model_path)         # loads tokenizer + SigLIP image processor
#         model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#             args.model_path, torch_dtype=torch.bfloat16,                   # Qwen2 native dtype
#             low_cpu_mem_usage=True).to(device).eval()
#     elif args.model_type == 'internvl':                                    # InternVL2.5 (InternLM2 backbone)
#         from transformers import AutoModel, AutoTokenizer
#         model = AutoModel.from_pretrained(
#             args.model_path, torch_dtype=torch.bfloat16,
#             trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
#         processor = AutoTokenizer.from_pretrained(
#             args.model_path, trust_remote_code=True)
#     elif args.model_type == 'qwen2vl':                                     # Qwen2.5-VL
#         from transformers import AutoModelForVision2Seq, AutoProcessor
#         processor = AutoProcessor.from_pretrained(args.model_path)
#         model = AutoModelForVision2Seq.from_pretrained(
#             args.model_path, torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True).to(device).eval()
#     else:
#         raise ValueError(f'Unsupported model_type for ablation: {args.model_type}')

#     # Load POPE data
#     pope_questions = load_pope_data(args.pope_path, args.n_pope_questions)  # Line 17: load POPE evaluation questions

#     # Get layer names for this model
#     layer_names = get_layer_names(args.model_type, args.n_layers)          # Line 18: hook point names

#     # Baseline hallucination rate (no ablation)
#     print(f'[GPU {gpu_id}] Computing baseline hallucination rate...')
#     baseline = compute_hallucination_rate(                                 # Line 19: evaluate without ablation
#         model, processor, pope_questions,
#         args.pope_img_dir, device, args.model_type)
#     baseline_hr = baseline['hallucination_rate']                           # Line 20: extract baseline hallucination rate
#     print(f'[GPU {gpu_id}] Baseline hallucination rate: {baseline_hr:.4f} '
#           f'({baseline["n_hallucinated"]}/{baseline["n_total"]})')

#     # Per-neuron ablation scores for this GPU's layers
#     neuron_scores = {}                                                     # Line 21: {(layer, neuron_start): delta_h}

#     for layer_idx in range(layer_range[0], layer_range[1]):                # Line 22: iterate layers assigned to this GPU
#         layer_name = layer_names[layer_idx]                                # Line 23: hook point name for this layer
#         print(f'[GPU {gpu_id}] Ablating layer {layer_idx} ({layer_name})')

#         for batch_start in range(0, args.n_neurons, args.batch_neurons):   # Line 24: iterate neuron batches
#             batch_end = min(batch_start + args.batch_neurons, args.n_neurons)  # Line 25: clamp to layer size
#             batch_size = batch_end - batch_start                           # Line 26: actual batch size

#             # Create ablation hook that zeros out neurons [batch_start:batch_end]
#             # in the specified layer
#             class AblationHook:
#                 """Context manager using register_forward_hook for neuron ablation.
#                 Avoids baukit signature issues across versions."""
#                 def __init__(self, model, layer_name, start, end, method='zero'):
#                     self.model = model
#                     self.layer_name = layer_name
#                     self.start = start
#                     self.end = end
#                     self.method = method
#                     self._handle = None

#                 def _resolve_module(self):
#                     """Walk the model tree to find the module at layer_name."""
#                     parts = self.layer_name.split('.')
#                     mod = self.model
#                     for p in parts:
#                         if p.isdigit():
#                             mod = mod[int(p)]
#                         else:
#                             mod = getattr(mod, p)
#                     return mod

#                 def __enter__(self):
#                     target = self._resolve_module()
#                     def hook_fn(module, input, output):
#                         if isinstance(output, tuple):
#                             out = output[0]
#                         else:
#                             out = output
#                         if self.method == 'zero':
#                             out[:, :, self.start:self.end] = 0
#                         return (out,) + output[1:] if isinstance(output, tuple) else out
#                     self._handle = target.register_forward_hook(hook_fn)
#                     return self

#                 def __exit__(self, *args):
#                     if self._handle is not None:
#                         self._handle.remove()
#                         self._handle = None

#             # Run POPE with this batch of neurons ablated
#             hook = AblationHook(model, layer_name, batch_start, batch_end,
#                                 args.ablation_method)
#             ablated = compute_hallucination_rate(                          # Line 46: evaluate with ablation active
#                 model, processor, pope_questions,
#                 args.pope_img_dir, device, args.model_type,
#                 ablation_hook=hook)

#             delta_h = ablated['hallucination_rate'] - baseline_hr          # Line 47: compute ΔH (positive = more hallucinations)

#             # Assign ΔH equally to each neuron in the batch (approximation)
#             per_neuron_delta = delta_h / batch_size                        # Line 48: distribute equally
#             for n in range(batch_start, batch_end):                        # Line 49: store per-neuron scores
#                 neuron_scores[(layer_idx, n)] = per_neuron_delta

#             if batch_start % (args.batch_neurons * 5) == 0:               # Line 50: periodic progress log
#                 print(f'  [GPU {gpu_id}] Layer {layer_idx}: '
#                       f'neurons {batch_start}-{batch_end-1}, '
#                       f'ΔH={delta_h:.4f}')

#     # Store results in shared dict
#     return_dict[gpu_id] = {                                                # Line 51: pass results back to main process
#         'neuron_scores': {f'{k[0]}_{k[1]}': v                             # Line 52: serialize tuple keys to strings
#                           for k, v in neuron_scores.items()},
#         'baseline': baseline,                                              # Line 53: include baseline for reference
#     }
#     print(f'[GPU {gpu_id}] Done — scored {len(neuron_scores)} neurons')


# def run_ablation_multi_gpu(args):
#     """Orchestrate multi-GPU ablation by spawning one worker per GPU.

#     Splits layers evenly across available GPUs, spawns worker processes,
#     collects results, and merges into a single scores dictionary.

#     Returns:
#         neuron_scores: dict {(layer_idx, neuron_idx): delta_h}
#         baseline: dict with baseline evaluation results
#     """
#     import torch                                                           # Line 1: needed for GPU count detection
#     import torch.multiprocessing as mp                                     # Line 2: multiprocessing with CUDA support

#     n_gpus = args.n_gpus if args.n_gpus > 0 else torch.cuda.device_count()  # Line 3: auto-detect or use specified
#     n_gpus = min(n_gpus, args.n_layers)                                    # Line 4: no more GPUs than layers
#     print(f'\n{"="*60}')
#     print(f'PHASE 1: Multi-GPU ablation ({n_gpus} GPUs, '
#           f'{args.n_layers} layers)')
#     print(f'{"="*60}\n')

#     if n_gpus == 0:                                                        # Line 5: no GPUs available
#         raise RuntimeError('No GPUs available for ablation. '
#                            'Use --skip_ablation with pre-computed scores.')

#     # Split layers across GPUs as evenly as possible
#     layers_per_gpu = args.n_layers // n_gpus                               # Line 6: base layers per GPU
#     remainder = args.n_layers % n_gpus                                     # Line 7: leftover layers to distribute
#     layer_ranges = []                                                      # Line 8: (start, end) for each GPU
#     start = 0
#     for i in range(n_gpus):                                                # Line 9: assign ranges
#         end = start + layers_per_gpu + (1 if i < remainder else 0)         # Line 10: first 'remainder' GPUs get +1 layer
#         layer_ranges.append((start, end))
#         start = end

#     for i, (s, e) in enumerate(layer_ranges):                             # Line 11: log assignments
#         print(f'  GPU {i}: layers {s}-{e-1} ({e-s} layers)')

#     # Spawn workers
#     mp.set_start_method('spawn', force=True)                               # Line 12: use spawn for CUDA compatibility
#     manager = mp.Manager()                                                 # Line 13: shared memory manager
#     return_dict = manager.dict()                                           # Line 14: dict for collecting worker results

#     processes = []                                                         # Line 15: list of spawned processes
#     for gpu_id in range(n_gpus):                                           # Line 16: one process per GPU
#         p = mp.Process(target=ablation_worker,                             # Line 17: create process
#                        args=(gpu_id, layer_ranges[gpu_id], args, return_dict))
#         processes.append(p)
#         p.start()                                                          # Line 18: launch process
#         if gpu_id < n_gpus - 1:                                            # Line 18b: stagger model loading
#             time.sleep(30)                                                 # Line 18c: 30s between starts to avoid OOM

#     for p in processes:                                                     # Line 19: wait for all processes to finish
#         p.join()

#     # Merge results from all GPUs
#     neuron_scores = {}                                                     # Line 20: merged score dict
#     baseline = None
#     for gpu_id in range(n_gpus):                                           # Line 21: iterate GPU results
#         if gpu_id in return_dict:
#             gpu_result = return_dict[gpu_id]
#             for key_str, val in gpu_result['neuron_scores'].items():       # Line 22: deserialize keys
#                 parts = key_str.split('_')
#                 neuron_scores[(int(parts[0]), int(parts[1]))] = val        # Line 23: reconstruct tuple keys
#             if baseline is None:
#                 baseline = gpu_result['baseline']                          # Line 24: grab baseline from first GPU

#     print(f'\nMerged {len(neuron_scores)} neuron scores from {n_gpus} GPUs')
#     return neuron_scores, baseline                                         # Line 25: return merged results


# # ═══════════════════════════════════════════════════════════════════════
# # Section 4 — Phase 2: Enrichment analysis (CPU-only)
# # ═══════════════════════════════════════════════════════════════════════

# def fishers_exact_test(n_category_in_driving, n_driving, n_category_total,
#                        n_total):
#     """Run Fisher's exact test for enrichment of a category among driving neurons.

#     Constructs a 2×2 contingency table:

#                           In category    Not in category
#         Halluc-driving      a                b
#         Not driving         c                d

#     Tests whether the category is over-represented among hallucination-driving
#     neurons compared to the background proportion.

#     Returns:
#         odds_ratio: float — how much more likely driving neurons are to be
#                     in this category vs not (>1 = enriched, <1 = depleted)
#         p_value: float — significance of the association
#     """
#     from scipy.stats import fisher_exact                                   # Line 1: import statistical test

#     a = n_category_in_driving                                              # Line 2: driving AND in category
#     b = n_driving - a                                                      # Line 3: driving AND NOT in category
#     c = n_category_total - a                                               # Line 4: not driving AND in category
#     d = n_total - n_driving - c                                            # Line 5: not driving AND NOT in category

#     table = np.array([[a, b], [c, d]])                                     # Line 6: 2×2 contingency table
#     odds_ratio, p_value = fisher_exact(table, alternative='two-sided')     # Line 7: two-sided Fisher's exact test

#     return odds_ratio, p_value                                             # Line 8: return OR and p-value


# def compute_enrichment(flat_labels, neuron_scores, top_k_pct,
#                        n_random_trials=1000, alpha=0.05, seed=42):
#     """Compute enrichment statistics for ALL neuron modality categories.

#     For each category (visual, text, multimodal, unknown):
#         1. Count how many neurons of that category fall in the top K%
#            hallucination-driving set
#         2. Run Fisher's exact test for over/under-representation
#         3. Compute expected count under null (proportional) hypothesis

#     Also runs random baseline: repeatedly sample random neuron sets of
#     the same size as the top K% and compute enrichment, to calibrate
#     the statistical test.

#     Args:
#         flat_labels: numpy array of string labels for all neurons
#         neuron_scores: dict {(layer, neuron): delta_h} — hallucination scores
#         top_k_pct: float — percentage of top neurons to consider "driving"
#         n_random_trials: int — number of random baseline samples
#         alpha: float — significance threshold
#         seed: int — random seed

#     Returns:
#         results: dict with per-category enrichment statistics
#     """
#     rng = np.random.RandomState(seed)                                      # Line 1: reproducible randomness

#     # Build score array aligned with flat_labels
#     n_total = len(flat_labels)                                             # Line 2: total neurons in model
#     score_array = np.zeros(n_total, dtype=np.float64)                      # Line 3: default 0 for neurons without scores

#     # Infer n_neurons per layer from flat_labels and n_layers
#     # (flat_labels should be n_layers × n_neurons_per_layer)
#     n_layers = 0                                                           # Line 4: count how many layers contributed scores
#     for (l, n) in neuron_scores:
#         n_layers = max(n_layers, l + 1)
#     n_neurons = n_total // n_layers if n_layers > 0 else 11008             # Line 5: infer neurons per layer

#     for (layer_idx, neuron_idx), delta_h in neuron_scores.items():         # Line 6: fill score array
#         flat_idx = layer_idx * n_neurons + neuron_idx                      # Line 7: linearised index
#         if flat_idx < n_total:                                             # Line 8: bounds check
#             score_array[flat_idx] = delta_h

#     # Define hallucination-driving neurons: top K% by ΔH (largest positive ΔH)
#     n_driving = max(1, int(n_total * top_k_pct / 100.0))                   # Line 9: number of driving neurons
#     driving_indices = np.argsort(score_array)[-n_driving:]                  # Line 10: indices of top K% by score
#     is_driving = np.zeros(n_total, dtype=bool)                             # Line 11: boolean mask
#     is_driving[driving_indices] = True                                     # Line 12: mark top K% as driving

#     print(f'\nTop {top_k_pct}% hallucination-driving neurons: {n_driving:,} / {n_total:,}')
#     print(f'Score range in driving set: [{score_array[driving_indices].min():.6f}, '
#           f'{score_array[driving_indices].max():.6f}]')

#     # ── Enrichment for each modality category ─────────────────────
#     categories = ['visual', 'text', 'multimodal', 'unknown']              # Line 13: categories to test
#     results = {'top_k_pct': top_k_pct, 'n_driving': n_driving,            # Line 14: store metadata
#                'n_total': n_total, 'categories': {}}

#     for cat in categories:                                                 # Line 15: iterate over categories
#         cat_mask = (flat_labels == cat)                                    # Line 16: boolean mask for this category
#         n_cat = int(cat_mask.sum())                                        # Line 17: total neurons in category
#         n_cat_in_driving = int((cat_mask & is_driving).sum())              # Line 18: driving neurons in category

#         # Expected count under proportional null
#         expected = n_driving * (n_cat / n_total)                           # Line 19: expected if driving were random
#         fold_enrichment = n_cat_in_driving / expected if expected > 0 else 0  # Line 20: fold change

#         # Fisher's exact test
#         odds_ratio, p_value = fishers_exact_test(                          # Line 21: run Fisher's exact test
#             n_cat_in_driving, n_driving, n_cat, n_total)

#         # Determine enrichment direction
#         if p_value < alpha:                                                # Line 22: statistically significant
#             direction = 'ENRICHED' if fold_enrichment > 1 else 'DEPLETED' # Line 23: enriched or depleted
#         else:
#             direction = 'n.s.'                                             # Line 24: not significant

#         results['categories'][cat] = {                                     # Line 25: store per-category results
#             'n_in_category': n_cat,
#             'n_in_driving': n_cat_in_driving,
#             'expected': round(expected, 1),
#             'fold_enrichment': round(fold_enrichment, 4),
#             'odds_ratio': round(odds_ratio, 4) if not np.isinf(odds_ratio) else 'inf',
#             'p_value': p_value,
#             'significant': p_value < alpha,
#             'direction': direction,
#             'pct_of_category_in_driving': round(100 * n_cat_in_driving / max(n_cat, 1), 2),
#             'pct_of_driving_from_category': round(100 * n_cat_in_driving / n_driving, 2),
#         }

#     # ── Random baseline: how often does a random set show enrichment? ──
#     print(f'\nRunning {n_random_trials} random baseline trials...')
#     random_enrichments = {cat: [] for cat in categories}                    # Line 26: store random fold enrichments

#     for trial in range(n_random_trials):                                   # Line 27: iterate random trials
#         random_indices = rng.choice(n_total, size=n_driving, replace=False)  # Line 28: random neuron set
#         random_mask = np.zeros(n_total, dtype=bool)                        # Line 29: boolean mask
#         random_mask[random_indices] = True                                 # Line 30: mark random set

#         for cat in categories:                                             # Line 31: compute enrichment for each category
#             cat_mask = (flat_labels == cat)
#             n_cat = int(cat_mask.sum())
#             n_cat_in_random = int((cat_mask & random_mask).sum())
#             expected = n_driving * (n_cat / n_total)
#             fold = n_cat_in_random / expected if expected > 0 else 0
#             random_enrichments[cat].append(fold)                           # Line 32: store fold enrichment

#     # Add random baseline statistics to results
#     results['random_baseline'] = {}                                        # Line 33: dict for random baseline stats
#     for cat in categories:                                                 # Line 34: compute per-category baseline stats
#         re = np.array(random_enrichments[cat])                             # Line 35: convert to numpy array
#         actual_fold = results['categories'][cat]['fold_enrichment']        # Line 36: actual fold enrichment

#         # Empirical p-value: how often does random exceed actual
#         if actual_fold > 1:                                                # Line 37: enrichment direction
#             emp_p = (re >= actual_fold).mean()                             # Line 38: fraction of random trials ≥ actual
#         else:
#             emp_p = (re <= actual_fold).mean()                             # Line 39: fraction of random trials ≤ actual

#         results['random_baseline'][cat] = {                                # Line 40: store baseline results
#             'mean_fold': round(float(re.mean()), 4),
#             'std_fold': round(float(re.std()), 4),
#             'ci_95_low': round(float(np.percentile(re, 2.5)), 4),
#             'ci_95_high': round(float(np.percentile(re, 97.5)), 4),
#             'empirical_p_value': round(float(emp_p), 4),
#         }

#     return results                                                         # Line 41: return all enrichment results


# def compute_per_layer_enrichment(labels_dict, neuron_scores, top_k_pct,
#                                  n_layers, n_neurons):
#     """Compute enrichment per layer for heatmap visualisation.

#     For each layer, defines the top K% of that layer's neurons as
#     hallucination-driving, then computes fold enrichment for each category.

#     Returns:
#         heatmap: numpy array (n_layers, 4) — fold enrichment per category per layer
#         categories: list of category names corresponding to columns
#     """
#     categories = ['visual', 'text', 'multimodal', 'unknown']              # Line 1: column order
#     heatmap = np.ones((n_layers, len(categories)), dtype=np.float64)       # Line 2: default fold=1.0 (no enrichment)

#     for l in range(n_layers):                                              # Line 3: iterate layers
#         key = str(l)
#         if key not in labels_dict:                                         # Line 4: skip missing layers
#             continue

#         # Get labels for this layer
#         layer_labels = sorted(labels_dict[key],                            # Line 5: sort by neuron index
#                               key=lambda x: x['neuron_idx'])
#         label_array = np.array([e['label'] for e in layer_labels])         # Line 6: extract labels

#         # Get scores for this layer
#         scores = np.zeros(len(label_array), dtype=np.float64)              # Line 7: score array for this layer
#         for n_idx in range(len(label_array)):                              # Line 8: fill from neuron_scores
#             if (l, n_idx) in neuron_scores:
#                 scores[n_idx] = neuron_scores[(l, n_idx)]

#         # Top K% in this layer
#         n_driving = max(1, int(len(label_array) * top_k_pct / 100.0))     # Line 9: driving count for this layer
#         driving_idx = np.argsort(scores)[-n_driving:]                      # Line 10: top K% by score
#         is_driving = np.zeros(len(label_array), dtype=bool)                # Line 11: boolean mask
#         is_driving[driving_idx] = True                                     # Line 12: mark driving neurons

#         for ci, cat in enumerate(categories):                              # Line 13: compute per-category enrichment
#             cat_mask = (label_array == cat)                                # Line 14: category mask
#             n_cat = int(cat_mask.sum())                                    # Line 15: total in category
#             n_cat_driving = int((cat_mask & is_driving).sum())             # Line 16: driving in category
#             expected = n_driving * (n_cat / len(label_array)) if len(label_array) > 0 else 0
#             if expected > 0:                                               # Line 17: avoid division by zero
#                 heatmap[l, ci] = n_cat_driving / expected                  # Line 18: fold enrichment

#     return heatmap, categories                                             # Line 19: return heatmap and column names


# # ═══════════════════════════════════════════════════════════════════════
# # Section 5 — Phase 3: Visualisation
# # ═══════════════════════════════════════════════════════════════════════

# def plot_enrichment_results(results, per_layer_heatmap, categories,
#                             output_dir, model_name):
#     """Generate all enrichment visualisation plots.

#     Creates four plots:
#       1. Enrichment bar chart — fold enrichment per category with significance stars
#       2. Odds-ratio forest plot — OR with 95% CI from random baseline
#       3. Per-layer enrichment heatmap — fold enrichment by layer × category
#       4. Summary table — text-based summary saved as PNG
#     """
#     import matplotlib                                                      # Line 1: matplotlib for plotting
#     matplotlib.use('Agg')                                                  # Line 2: non-interactive backend
#     import matplotlib.pyplot as plt                                        # Line 3: plotting API
#     import matplotlib.colors as mcolors                                    # Line 4: colour normalisation for heatmap

#     os.makedirs(output_dir, exist_ok=True)                                 # Line 5: ensure output directory exists

#     cats = ['visual', 'text', 'multimodal', 'unknown']                    # Line 6: category order
#     cat_colors = {                                                         # Line 7: colour scheme for categories
#         'visual': '#3498db',                                               # Line 8: blue for visual
#         'text': '#e74c3c',                                                 # Line 9: red for text
#         'multimodal': '#2ecc71',                                           # Line 10: green for multimodal
#         'unknown': '#95a5a6',                                              # Line 11: grey for unknown
#     }

#     # ── Plot 1: Enrichment bar chart ──────────────────────────────
#     fig, ax = plt.subplots(figsize=(10, 6))                                # Line 12: create figure
#     x = np.arange(len(cats))                                               # Line 13: x positions for bars
#     folds = [results['categories'][c]['fold_enrichment'] for c in cats]    # Line 14: fold enrichment values
#     colors = [cat_colors[c] for c in cats]                                 # Line 15: bar colours
#     sigs = [results['categories'][c]['significant'] for c in cats]         # Line 16: significance flags

#     bars = ax.bar(x, folds, color=colors, edgecolor='white', linewidth=1.5,  # Line 17: draw bars
#                   width=0.6)
#     ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1,         # Line 18: reference line at fold=1
#                label='Expected (null)')

#     # Add significance stars
#     for i, (bar, sig) in enumerate(zip(bars, sigs)):                       # Line 19: annotate bars
#         if sig:
#             p = results['categories'][cats[i]]['p_value']
#             stars = '***' if p < 0.001 else ('**' if p < 0.01 else '*')   # Line 20: significance level
#             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
#                     stars, ha='center', va='bottom', fontsize=14,          # Line 21: place stars above bar
#                     fontweight='bold')

#     # Add random baseline CI as error bars
#     for i, cat in enumerate(cats):                                         # Line 22: overlay random baseline range
#         rb = results['random_baseline'][cat]
#         ax.errorbar(i, rb['mean_fold'],                                    # Line 23: random baseline mean
#                     yerr=[[rb['mean_fold'] - rb['ci_95_low']],             # Line 24: lower error bar
#                           [rb['ci_95_high'] - rb['mean_fold']]],           # Line 25: upper error bar
#                     fmt='D', color='black', markersize=6, capsize=5,       # Line 26: diamond marker
#                     label='Random baseline 95% CI' if i == 0 else None)

#     ax.set_xticks(x)                                                       # Line 27: set x-axis tick positions
#     ax.set_xticklabels([c.capitalize() for c in cats], fontsize=12)        # Line 28: category labels
#     ax.set_ylabel('Fold Enrichment', fontsize=13)                          # Line 29: y-axis label
#     ax.set_title(f'{model_name} — Hallucination-Driving Neuron Enrichment\n'
#                  f'(Top {results["top_k_pct"]}%, N={results["n_driving"]:,})',
#                  fontsize=14, fontweight='bold')                           # Line 30: title with metadata
#     ax.legend(fontsize=10)                                                 # Line 31: show legend
#     ax.set_ylim(bottom=0)                                                  # Line 32: y-axis starts at 0
#     plt.tight_layout()                                                     # Line 33: prevent label clipping
#     fig.savefig(os.path.join(output_dir, 'enrichment_bar_chart.png'),      # Line 34: save figure
#                 dpi=200, bbox_inches='tight')
#     plt.close(fig)                                                         # Line 35: free memory

#     # ── Plot 2: Odds-ratio forest plot ────────────────────────────
#     fig, ax = plt.subplots(figsize=(8, 5))                                 # Line 36: create figure
#     y_pos = np.arange(len(cats))                                           # Line 37: y positions

#     for i, cat in enumerate(cats):                                         # Line 38: iterate categories
#         or_val = results['categories'][cat]['odds_ratio']                   # Line 39: odds ratio
#         if or_val == 'inf':                                                # Line 40: handle infinite OR
#             or_val = 10.0
#         sig = results['categories'][cat]['significant']                    # Line 41: is it significant?
#         marker = 's' if sig else 'o'                                       # Line 42: filled square if significant
#         color = cat_colors[cat]
#         ax.plot(or_val, i, marker, color=color, markersize=12,            # Line 43: plot OR point
#                 markeredgecolor='black', markeredgewidth=1)
#         ax.annotate(f'  OR={or_val:.2f}', (or_val, i),                    # Line 44: annotate with OR value
#                     fontsize=10, va='center')

#     ax.axvline(x=1.0, color='grey', linestyle='--', linewidth=1)          # Line 45: null OR reference line
#     ax.set_yticks(y_pos)                                                   # Line 46: set y-axis ticks
#     ax.set_yticklabels([c.capitalize() for c in cats], fontsize=12)        # Line 47: category labels
#     ax.set_xlabel('Odds Ratio', fontsize=13)                               # Line 48: x-axis label
#     ax.set_title(f'{model_name} — Odds Ratios (Hallucination Enrichment)', # Line 49: title
#                  fontsize=14, fontweight='bold')
#     ax.set_xscale('log')                                                   # Line 50: log scale for OR
#     plt.tight_layout()
#     fig.savefig(os.path.join(output_dir, 'odds_ratio_forest.png'),         # Line 51: save figure
#                 dpi=200, bbox_inches='tight')
#     plt.close(fig)

#     # ── Plot 3: Per-layer enrichment heatmap ──────────────────────
#     fig, ax = plt.subplots(figsize=(8, 12))                                # Line 52: tall figure for all layers
#     n_layers = per_layer_heatmap.shape[0]

#     # Diverging colour map centred at 1.0 (no enrichment)
#     vmin, vmax = 0.0, max(2.0, per_layer_heatmap.max())                    # Line 53: colour range
#     norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)        # Line 54: centre at 1.0
#     im = ax.imshow(per_layer_heatmap, aspect='auto', cmap='RdBu_r',       # Line 55: draw heatmap
#                    norm=norm, interpolation='nearest')

#     ax.set_xticks(range(len(categories)))                                  # Line 56: x-axis ticks
#     ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)  # Line 57: x-axis labels
#     ax.set_yticks(range(n_layers))                                         # Line 58: y-axis ticks
#     ax.set_yticklabels([f'L{l}' for l in range(n_layers)], fontsize=8)    # Line 59: layer labels
#     ax.set_xlabel('Neuron Category', fontsize=13)                          # Line 60: x-axis label
#     ax.set_ylabel('Layer', fontsize=13)                                    # Line 61: y-axis label
#     ax.set_title(f'{model_name} — Per-Layer Enrichment '                   # Line 62: title
#                  f'(Fold, top {results["top_k_pct"]}%)',
#                  fontsize=14, fontweight='bold')
#     plt.colorbar(im, ax=ax, label='Fold Enrichment', shrink=0.8)          # Line 63: add colour bar
#     plt.tight_layout()
#     fig.savefig(os.path.join(output_dir, 'per_layer_enrichment_heatmap.png'),  # Line 64: save
#                 dpi=200, bbox_inches='tight')
#     plt.close(fig)

#     print(f'Plots saved to {output_dir}')


# # ═══════════════════════════════════════════════════════════════════════
# # Section 6 — Main
# # ═══════════════════════════════════════════════════════════════════════

# def main():
#     args = parse_args()                                                    # Line 1: parse command-line arguments

#     # ── Plot-only mode: regenerate plots from saved results ───────
#     if args.plot_only:
#         print(f'\n{"="*60}')
#         print(f'PLOT-ONLY MODE: Regenerating plots for {args.model_name}')
#         print(f'{"="*60}\n')

#         results_path = os.path.join(args.output_dir, 'enrichment_results.json')
#         heatmap_path = os.path.join(args.output_dir, 'per_layer_enrichment.npy')

#         if not os.path.isfile(results_path):
#             print(f'  ERROR: {results_path} not found. Run full pipeline first.')
#             return
#         if not os.path.isfile(heatmap_path):
#             print(f'  ERROR: {heatmap_path} not found. Run full pipeline first.')
#             return

#         with open(results_path) as f:
#             results = json.load(f)
#         per_layer_heatmap = np.load(heatmap_path)
#         heatmap_cats = ['visual', 'text', 'multimodal', 'unknown']

#         try:
#             plot_enrichment_results(results, per_layer_heatmap, heatmap_cats,
#                                     args.output_dir, args.model_name)
#         except ImportError as e:
#             print(f'WARNING: Could not generate plots ({e}).')
#             return

#         print(f'  Plots saved to {args.output_dir}/*.png')
#         return

#     rng = np.random.RandomState(args.seed)                                 # Line 2: set global random seed
#     os.makedirs(args.output_dir, exist_ok=True)                            # Line 3: create output directory

#     if args.label_dir is None:
#         print('ERROR: --label_dir is required (unless using --plot_only)')
#         return

#     # ── Load neuron modality labels ───────────────────────────────
#     print(f'\n{"="*60}')
#     print(f'Loading neuron labels ({args.taxonomy})')
#     print(f'{"="*60}\n')

#     layer_names = get_layer_names(args.model_type, args.n_layers)          # Line 4: hook point names
#     labels_dict, flat_labels = load_neuron_labels(                         # Line 5: load all neuron labels
#         args.label_dir, args.taxonomy, layer_names, args.n_layers)

#     # ── Contrastive POPE preprocessing (if requested) ───────────────
#     cett_diff_scores = None                                                # Line 5a: will hold CETT-diff if contrastive
#     if args.contrastive:                                                   # Line 5b: build contrastive set + CETT-diff
#         contrastive_path, n_clean, cett_diff_scores = build_contrastive_pope_set(args)
#         if not args.skip_ablation:                                         # Line 5c: only override pope_path if ablation will run
#             args.pope_path = contrastive_path                              # Line 5d: redirect workers to filtered JSONL
#             args.n_pope_questions = None                                   # Line 5e: use all filtered questions (no truncation)
#             print(f'  Ablation will use {n_clean} contrastive POPE questions')
#         else:
#             print(f'  Contrastive set built for CETT-diff only '
#                   f'(ablation skipped, using pre-computed ΔH scores)')
#         if cett_diff_scores is not None:
#             print(f'  CETT-diff scores available ({len(cett_diff_scores)} neurons)')

#     # ── Phase 1: Identify hallucination-driving neurons ───────────
#     if args.skip_ablation:                                                 # Line 6: use pre-computed scores
#         print(f'\nSkipping ablation — loading scores from {args.ablation_scores}')
#         with open(args.ablation_scores) as f:                              # Line 7: load pre-computed JSON
#             scores_raw = json.load(f)
#         neuron_scores = {}                                                 # Line 8: reconstruct score dict
#         for key_str, val in scores_raw.items():                            # Line 9: parse "layer_neuron" keys
#             parts = key_str.split('_')
#             neuron_scores[(int(parts[0]), int(parts[1]))] = val
#         print(f'Loaded {len(neuron_scores)} pre-computed neuron scores')
#     else:
#         neuron_scores, baseline = run_ablation_multi_gpu(args)             # Line 10: run multi-GPU ablation

#         # Save ablation scores for future re-use
#         scores_path = os.path.join(args.output_dir, 'ablation_scores.json')  # Line 11: output path
#         scores_serializable = {f'{k[0]}_{k[1]}': v                        # Line 12: serialize tuple keys
#                                for k, v in neuron_scores.items()}
#         with open(scores_path, 'w') as f:                                  # Line 13: write JSON
#             json.dump(scores_serializable, f, indent=2)
#         print(f'Ablation scores saved to {scores_path}')

#         # Also save baseline results
#         if baseline:                                                       # Line 14: save baseline evaluation
#             baseline_path = os.path.join(args.output_dir, 'baseline_results.json')
#             with open(baseline_path, 'w') as f:
#                 json.dump(baseline, f, indent=2)

#     # ── Phase 2: Enrichment analysis ──────────────────────────────
#     print(f'\n{"="*60}')
#     print(f'PHASE 2: Enrichment analysis (all categories)')
#     print(f'{"="*60}\n')

#     results = compute_enrichment(                                          # Line 15: run enrichment for all categories
#         flat_labels, neuron_scores,
#         top_k_pct=args.top_k_pct,
#         n_random_trials=args.n_random_trials,
#         alpha=args.alpha,
#         seed=args.seed)

#     # Print enrichment summary table
#     print(f'\n{"─"*80}')                                                   # Line 16: header line
#     print(f'ENRICHMENT SUMMARY — Top {args.top_k_pct}% hallucination-driving neurons')
#     print(f'{"─"*80}')
#     print(f'{"Category":<14} {"In Driving":>10} {"Expected":>10} '        # Line 17: column headers
#           f'{"Fold":>8} {"OR":>8} {"p-value":>10} {"Direction":>10}')
#     print(f'{"─"*80}')

#     for cat in ['visual', 'text', 'multimodal', 'unknown']:               # Line 18: print each category
#         r = results['categories'][cat]
#         or_str = f'{r["odds_ratio"]:.2f}' if r["odds_ratio"] != 'inf' else 'inf'
#         print(f'{cat:<14} {r["n_in_driving"]:>10,} {r["expected"]:>10.1f} '
#               f'{r["fold_enrichment"]:>8.3f} {or_str:>8} '
#               f'{r["p_value"]:>10.2e} {r["direction"]:>10}')
#     print(f'{"─"*80}')

#     # Print random baseline summary
#     print(f'\nRandom baseline (n={args.n_random_trials}):')                # Line 19: random baseline header
#     for cat in ['visual', 'text', 'multimodal', 'unknown']:
#         rb = results['random_baseline'][cat]
#         print(f'  {cat:<12}: mean_fold={rb["mean_fold"]:.3f} '
#               f'[{rb["ci_95_low"]:.3f}, {rb["ci_95_high"]:.3f}] '
#               f'emp_p={rb["empirical_p_value"]:.4f}')

#     # Save results JSON
#     results_path = os.path.join(args.output_dir, 'enrichment_results.json')  # Line 20: output path
#     with open(results_path, 'w') as f:
#         json.dump(results, f, indent=2, default=str)                       # Line 21: write results
#     print(f'\nResults saved to {results_path}')

#     # ── CETT-diff enrichment (if contrastive was used) ────────────
#     if cett_diff_scores is not None:
#         print(f'\n{"="*60}')
#         print(f'CETT-DIFF ENRICHMENT ANALYSIS (independent of ablation ΔH)')
#         print(f'{"="*60}\n')

#         cett_results = compute_enrichment(                                 # Line E1: enrichment using CETT-diff scores
#             flat_labels, cett_diff_scores,
#             top_k_pct=args.top_k_pct,
#             n_random_trials=args.n_random_trials,
#             alpha=args.alpha,
#             seed=args.seed)

#         # Print CETT-diff enrichment summary
#         print(f'\n{"─"*80}')
#         print(f'CETT-DIFF ENRICHMENT — Top {args.top_k_pct}% by CETT-diff')
#         print(f'{"─"*80}')
#         print(f'{"Category":<14} {"In Driving":>10} {"Expected":>10} '
#               f'{"Fold":>8} {"OR":>8} {"p-value":>10} {"Direction":>10}')
#         print(f'{"─"*80}')

#         for cat in ['visual', 'text', 'multimodal', 'unknown']:
#             r = cett_results['categories'][cat]
#             or_str = f'{r["odds_ratio"]:.2f}' if r["odds_ratio"] != 'inf' else 'inf'
#             print(f'{cat:<14} {r["n_in_driving"]:>10,} {r["expected"]:>10.1f} '
#                   f'{r["fold_enrichment"]:>8.3f} {or_str:>8} '
#                   f'{r["p_value"]:>10.2e} {r["direction"]:>10}')
#         print(f'{"─"*80}')

#         # Save CETT-diff enrichment
#         cett_enrich_path = os.path.join(args.output_dir,
#                                         'enrichment_results_cett_diff.json')
#         with open(cett_enrich_path, 'w') as f:
#             json.dump(cett_results, f, indent=2, default=str)
#         print(f'CETT-diff enrichment saved to {cett_enrich_path}')

#         # Compare top-5% overlap between ΔH and CETT-diff (Jaccard)
#         n_total = len(neuron_scores)
#         top_k = max(1, int(n_total * args.top_k_pct / 100.0))
#         top_dh = set(sorted(neuron_scores, key=neuron_scores.get,          # Line E2: top by ΔH (most positive)
#                             reverse=True)[:top_k])
#         top_cett = set(sorted(cett_diff_scores,                            # Line E3: top by CETT-diff
#                               key=cett_diff_scores.get,
#                               reverse=True)[:top_k])
#         overlap = len(top_dh & top_cett)                                   # Line E4: intersection size
#         jaccard = overlap / len(top_dh | top_cett) if top_dh | top_cett else 0  # Line E5: Jaccard similarity
#         print(f'\n  Top-{args.top_k_pct}% overlap (ΔH vs CETT-diff):')
#         print(f'    Intersection: {overlap}/{top_k} neurons')
#         print(f'    Jaccard similarity: {jaccard:.3f}')
#         print(f'    (High Jaccard = both methods agree on which neurons '
#               f'drive hallucination)')

#         # ── Combined hallucination score: normalize(ΔH) × normalize(CETT-diff) ──
#         print(f'\n{"="*60}')
#         print(f'COMBINED HALLUCINATION SCORE (ΔH × CETT-diff)')
#         print(f'{"="*60}\n')

#         # Normalize ΔH to [0, 1]
#         dh_vals = np.array(list(neuron_scores.values()))                   # Line H1: all ΔH values
#         dh_min, dh_max = float(dh_vals.min()), float(dh_vals.max())        # Line H2: global range
#         dh_range = dh_max - dh_min if dh_max > dh_min else 1.0            # Line H3: avoid division by zero

#         # Normalize CETT-diff to [0, 1]
#         cd_vals = np.array(list(cett_diff_scores.values()))                # Line H4: all CETT-diff values
#         cd_min, cd_max = float(cd_vals.min()), float(cd_vals.max())        # Line H5: global range
#         cd_range = cd_max - cd_min if cd_max > cd_min else 1.0            # Line H6: avoid division by zero

#         # Compute combined = normalize(ΔH) × normalize(CETT-diff) for shared keys
#         combined_halluc_scores = {}                                        # Line H7: {(layer, neuron): float}
#         shared_keys = set(neuron_scores.keys()) & set(cett_diff_scores.keys())  # Line H8: neurons with both scores
#         for key in shared_keys:                                            # Line H9: iterate shared neurons
#             norm_dh = (neuron_scores[key] - dh_min) / dh_range            # Line H10: normalized ΔH
#             norm_cd = (cett_diff_scores[key] - cd_min) / cd_range         # Line H11: normalized CETT-diff
#             combined_halluc_scores[key] = norm_dh * norm_cd               # Line H12: product

#         # Save all three rankings
#         # 1. ΔH already saved as ablation_scores.json
#         # 2. CETT-diff already saved as cett_diff_scores.json
#         # 3. Save combined
#         combined_path = os.path.join(args.output_dir,                      # Line H13: output path
#                                      'combined_halluc_scores.json')
#         combined_serializable = {f'{k[0]}_{k[1]}': v                       # Line H14: serialize tuple keys
#                                  for k, v in combined_halluc_scores.items()}
#         with open(combined_path, 'w') as f:                                # Line H15: write JSON
#             json.dump(combined_serializable, f, indent=2)
#         print(f'  {len(combined_halluc_scores)} neurons scored')
#         print(f'  Saved → {combined_path}')

#         # Run enrichment on combined scores
#         combined_enrich = compute_enrichment(
#             flat_labels, combined_halluc_scores,
#             top_k_pct=args.top_k_pct,
#             n_random_trials=args.n_random_trials,
#             alpha=args.alpha,
#             seed=args.seed)

#         print(f'\n{"─"*80}')
#         print(f'COMBINED ENRICHMENT — Top {args.top_k_pct}% by ΔH × CETT-diff')
#         print(f'{"─"*80}')
#         print(f'{"Category":<14} {"In Driving":>10} {"Expected":>10} '
#               f'{"Fold":>8} {"OR":>8} {"p-value":>10} {"Direction":>10}')
#         print(f'{"─"*80}')

#         for cat in ['visual', 'text', 'multimodal', 'unknown']:
#             r = combined_enrich['categories'][cat]
#             or_str = f'{r["odds_ratio"]:.2f}' if r["odds_ratio"] != 'inf' else 'inf'
#             print(f'{cat:<14} {r["n_in_driving"]:>10,} {r["expected"]:>10.1f} '
#                   f'{r["fold_enrichment"]:>8.3f} {or_str:>8} '
#                   f'{r["p_value"]:>10.2e} {r["direction"]:>10}')
#         print(f'{"─"*80}')

#         combined_enrich_path = os.path.join(args.output_dir,
#                                             'enrichment_results_combined.json')
#         with open(combined_enrich_path, 'w') as f:
#             json.dump(combined_enrich, f, indent=2, default=str)
#         print(f'Combined enrichment saved to {combined_enrich_path}')

#         # Summary of all three methods
#         print(f'\n{"="*60}')
#         print(f'ENRICHMENT COMPARISON (visual neuron fold-enrichment)')
#         print(f'{"="*60}')
#         vis_dh = results['categories']['visual']['fold_enrichment']
#         vis_cd = cett_results['categories']['visual']['fold_enrichment']
#         vis_cb = combined_enrich['categories']['visual']['fold_enrichment']
#         print(f'  ΔH (ablation):     {vis_dh:.3f}×')
#         print(f'  CETT-diff:         {vis_cd:.3f}×')
#         print(f'  Combined (ΔH×CD):  {vis_cb:.3f}×')
#         print(f'\n  Output files for step 11 steering:')
#         print(f'    --halluc_scores_path {combined_path}  (default: combined)')
#         print(f'    --halluc_scores_path {os.path.join(args.output_dir, "ablation_scores.json")}  (ΔH only)')
#         print(f'    --halluc_scores_path {os.path.join(args.output_dir, "cett_diff_scores.json")}  (CETT-diff only)')

#     # ── Per-layer enrichment heatmap ──────────────────────────────
#     per_layer_heatmap, heatmap_cats = compute_per_layer_enrichment(        # Line 22: compute per-layer enrichment
#         labels_dict, neuron_scores, args.top_k_pct,
#         args.n_layers, args.n_neurons)

#     # Save heatmap data
#     np.save(os.path.join(args.output_dir, 'per_layer_enrichment.npy'),     # Line 23: save numpy array
#             per_layer_heatmap)

#     # ── Phase 3: Visualisation ────────────────────────────────────
#     print(f'\n{"="*60}')
#     print(f'PHASE 3: Generating plots')
#     print(f'{"="*60}\n')

#     try:
#         plot_enrichment_results(results, per_layer_heatmap, heatmap_cats,  # Line 24: generate all plots
#                                 args.output_dir, args.model_name)
#     except ImportError as e:                                               # Line 25: handle missing matplotlib
#         print(f'WARNING: Could not generate plots ({e}). '
#               f'Install matplotlib to enable visualisation.')

#     print(f'\n{"="*60}')
#     print(f'HALLUCINATION TAXONOMY COMPLETE')
#     print(f'  Results:  {results_path}')
#     print(f'  Plots:    {args.output_dir}/*.png')
#     if cett_diff_scores is not None:
#         print(f'  CETT-diff: {args.output_dir}/cett_diff_scores.json')
#         print(f'  CETT-diff enrichment: {args.output_dir}/enrichment_results_cett_diff.json')
#         print(f'  Combined:  {args.output_dir}/combined_halluc_scores.json')
#         print(f'  Combined enrichment: {args.output_dir}/enrichment_results_combined.json')
#     print(f'{"="*60}')


# if __name__ == '__main__':
#     main()