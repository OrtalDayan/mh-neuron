#!/usr/bin/env python3
"""
Counterfactual Perturbation Neuron Classifier (cf_classify.py)
================================================================

A standalone classifier that runs in parallel to neuron_modality_statistical.py
but uses content perturbation (vary image / vary text) instead of token-position
slicing for modality classification.

This script imports helpers from neuron_modality_statistical.py for:
  - Model loading (load_model_hf, load_model_internvl, load_model_qwen2vl, etc.)
  - Prompt building (prepare_inputs_*)
  - Layer name building (get_layer_names)
  - Hook-point post-processing (_postprocess_acts)
  - llava-llama3 normalization

That way model support, hook conventions, and tokenization are guaranteed
identical to the existing PMBT pipeline.

Output layout (parallel to existing pipeline):
  results/3-classify/<MODE>/<MODEL_NAME>/llm_cf_permutation<HOOK_SUFFIX><GEN_DIR_SUFFIX>/
      <layer_name>/neuron_labels_cf_permutation.json
      cf_permutation_stats_layers<S>-<E>.json

Usage:
  python cf_classify.py \\
      --model_type llava-llama3 \\
      --original_model_path llava-hf/llama3-llava-next-8b-hf \\
      --output_dir results/3-classify/full \\
      --model llava-next-llama3-8b \\
      --paired_data_path data/cf_paired_500.json \\
      --layer_start 14 --layer_end 23 \\
      --hook_point gate_up \\
      --K_image 5 --K_text 5 \\
      --n_noise_pairs 500 --noise_K 5 \\
      --noise_percentile 95 \\
      --n_permutations 1000 --alpha 0.05

Methodology:
  1. CF Pass: for each sample (canonical_image, canonical_caption,
     similar_images, captions), record activations under K image variants
     (text fixed) and K text variants (image fixed).
  2. Noise Pass: record activations on random (image, caption) pairings to
     calibrate the engagement threshold from the noise-variance distribution.
  3. Classifier: per neuron, variance ratio Δ_image - Δ_text with permutation
     test for visual/text discrimination, plus engagement check against the
     calibrated noise threshold for multimodal/unknown.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Import helpers from neuron_modality_statistical.py
# ═══════════════════════════════════════════════════════════════════
# We add the script directory to sys.path so we can import the existing
# classifier's helpers without duplicating model-loading code.

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

try:
    from neuron_modality_statistical import (
        load_coco_captions,
        load_generated_descriptions,
        get_layer_names,
        get_down_proj_names,
        get_o_proj_names,
        _postprocess_acts,
        _get_num_attention_heads,
        load_model_hf,
        load_model_original,
        load_model_internvl,
        load_model_llava_ov,
        load_model_qwen2vl,
        load_model_idefics2,
        prepare_inputs_hf,
        prepare_inputs_original,
        prepare_inputs_internvl,
        prepare_inputs_llava_ov,
        prepare_inputs_llava_next,
        prepare_inputs_qwen2vl,
        prepare_inputs_idefics2,
    )
except ImportError as e:
    sys.exit(
        f'ERROR: Could not import helpers from neuron_modality_statistical.py.\n'
        f'Make sure cf_classify.py is in the same directory '
        f'(typically code/) as neuron_modality_statistical.py.\n'
        f'Original error: {e}'
    )


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Argument parsing (compatible with run_pipeline.sh conventions)
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Counterfactual perturbation neuron classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ─── Compatible flags (same as neuron_modality_statistical.py) ───
    p.add_argument('--model_type', default='llava-hf',
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl',
                            'qwen25vl-7b', 'qwen25vl-3b', 'idefics2',
                            'llava-ov', 'llava-llama3'])
    p.add_argument('--model_path', default=None)
    p.add_argument('--model', default='llava-1.5-7b',
                   help='Short model name for output directory')
    p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf')
    p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b')
    p.add_argument('--coco_img_dir',
                   default='/home/projects/bagon/shared/coco2017/'
                           'images/train2017/')
    p.add_argument('--output_dir', default='results/3-classify/full')

    # Hook point (CF supports same hooks as PMBT)
    p.add_argument('--hook_point', default='gate_up',
                   choices=['gate', 'gate_up', 'attn'])
    p.add_argument('--importance_weight', action='store_true')

    # Layer sharding
    p.add_argument('--layer_start', type=int, default=0)
    p.add_argument('--layer_end', type=int, default=32)
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=42)

    # Output suffix (matches PMBT convention)
    p.add_argument('--output_suffix', default='',
                   help='Extra suffix for output dir (e.g. _min100_max2048)')

    # ─── CF-specific flags ───
    p.add_argument('--paired_data_path', required=True,
                   help='Path to paired-data JSON built by build_cf_paired_data.py. '
                        'Format: {img_id: {canonical_caption, captions, similar_images}}')
    p.add_argument('--K_image', type=int, default=5,
                   help='Number of image variants per sample (varying image, '
                        'caption fixed)')
    p.add_argument('--K_text', type=int, default=5,
                   help='Number of text variants per sample (varying caption, '
                        'image fixed)')
    p.add_argument('--n_noise_pairs', type=int, default=500,
                   help='Number of random (image, caption) groups for noise pass '
                        '(used to calibrate engagement threshold)')
    p.add_argument('--noise_K', type=int, default=5,
                   help='K random pairings per noise group (matches K_image/K_text)')
    p.add_argument('--noise_percentile', type=float, default=95.0,
                   help='Percentile of noise variance distribution that defines '
                        'the engagement threshold (95 = top 5%% of noise)')
    p.add_argument('--n_permutations', type=int, default=1000)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--skip_noise_pass', action='store_true',
                   help='Skip noise pass; use 3-way classification (visual/text/'
                        'non-specialized) instead of 4-way')

    # Sample size
    p.add_argument('--num_samples', type=int, default=None,
                   help='Limit number of samples from paired data (None = all)')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Activation recording (uses existing pipeline's helpers)
# ═══════════════════════════════════════════════════════════════════

def _normalize_model_args(args):
    """Apply the same normalization as neuron_modality_statistical.main().

    Handles qwen25vl-* → qwen2vl and llava-llama3 → llava-hf with chat template.

    Returns:
        (effective_model_type, is_llava_llama3, hf_id_resolved)
    """
    is_llava_llama3 = False
    effective = args.model_type

    if effective in ('qwen25vl-7b', 'qwen25vl-3b'):
        effective = 'qwen2vl'

    if effective == 'llava-llama3':
        is_llava_llama3 = True
        if args.hf_id == 'llava-hf/llava-1.5-7b-hf':
            args.hf_id = (args.model_path or args.original_model_path
                            or 'llava-hf/llama3-llava-next-8b-hf')
        effective = 'llava-hf'  # load via load_model_hf

    return effective, is_llava_llama3, args.hf_id


def _load_model_dispatch(args, effective_type, device):
    """Dispatch to the right model loader based on (effective) model type."""
    if effective_type == 'llava-hf':
        print(f'Loading HF model: {args.hf_id} …')
        return load_model_hf(args.hf_id, device)
    elif effective_type == 'internvl':
        path = args.model_path or args.original_model_path
        print(f'Loading InternVL: {path} …')
        return load_model_internvl(path, device)
    elif effective_type == 'llava-ov':
        path = args.model_path or args.original_model_path
        print(f'Loading LLaVA-OneVision: {path} …')
        return load_model_llava_ov(path, device)
    elif effective_type == 'qwen2vl':
        path = args.model_path or args.original_model_path
        print(f'Loading Qwen2VL: {path} …')
        return load_model_qwen2vl(path, device)
    elif effective_type == 'idefics2':
        path = args.model_path or args.original_model_path
        print(f'Loading Idefics2: {path} …')
        return load_model_idefics2(path, device)
    elif effective_type == 'llava-liuhaotian':
        path = args.original_model_path
        print(f'Loading original LLaVA: {path} …')
        return load_model_original(path, device)
    else:
        sys.exit(f'Unknown model_type: {effective_type}')


def _prepare_input_dispatch(args, effective_type, is_llava_llama3,
                              processor, tokenizer, model, img, text, device,
                              image_token_id):
    """Dispatch to the right prompt-builder."""
    if is_llava_llama3:
        # llava-llama3: load via hf, prompts via chat template
        return prepare_inputs_llava_next(processor, img, text, device, image_token_id)
    elif effective_type == 'llava-hf':
        return prepare_inputs_hf(processor, img, text, device, image_token_id)
    elif effective_type == 'internvl':
        return prepare_inputs_internvl(tokenizer, model, img, text, device)
    elif effective_type == 'llava-ov':
        return prepare_inputs_llava_ov(processor, img, text, device, image_token_id)
    elif effective_type == 'qwen2vl':
        return prepare_inputs_qwen2vl(processor, img, text, device, image_token_id)
    elif effective_type == 'idefics2':
        return prepare_inputs_idefics2(processor, img, text, device, image_token_id)
    elif effective_type == 'llava-liuhaotian':
        return prepare_inputs_original(processor, img, text, device, image_token_id)
    else:
        sys.exit(f'Unknown model_type for prompt: {effective_type}')


def _record_activations_one_pass(model, args, effective_type, is_llava_llama3,
                                    processor, tokenizer, img, text,
                                    layer_names, layer_indices, retain_input,
                                    hook_point, device, image_token_id,
                                    n_heads):
    """Run one forward pass; return max-over-content activations per layer.

    Aggregates over all content tokens (visual + text) within the prompt.
    """
    from baukit import TraceDict

    inputs = _prepare_input_dispatch(
        args, effective_type, is_llava_llama3,
        processor, tokenizer, model, img, text, device, image_token_id,
    )
    # The prepare_inputs_* helpers return a tuple — usually
    # (inputs_dict, visual_mask) or (inputs_dict, visual_mask, n_vis_tokens).
    # We only need the inputs dict for the forward pass; visual_mask and
    # n_vis_tokens are unused here because we max-pool over all positions.
    if isinstance(inputs, tuple):
        inputs = inputs[0]

    layer_acts = {}
    with torch.no_grad():
        with TraceDict(model, layer_names, retain_input=retain_input) as td:
            model(**inputs)

    for li in layer_indices:
        layer_name = layer_names[li]
        if retain_input:
            inp = td[layer_name].input
            raw = inp[0] if isinstance(inp, tuple) else inp
        else:
            out = td[layer_name].output
            raw = out[0] if isinstance(out, tuple) else out

        # Use existing post-processing for hook_point semantics
        # (handles attn head reduction, etc.)
        acts = _postprocess_acts(raw, hook_point, n_heads=n_heads)
        # acts shape: (1, seq_len, n_neurons)
        acts = acts[0].float()  # (seq_len, n_neurons)

        # Max over all positions (we want neuron's peak activation regardless
        # of where it occurred — the CF method is position-agnostic)
        max_per_neuron = acts.max(dim=0).values.cpu().numpy()
        layer_acts[li] = max_per_neuron.astype(np.float32)

    return layer_acts


# ═══════════════════════════════════════════════════════════════════
# Section 4 — CF Classifier (variance-ratio permutation test)
# ═══════════════════════════════════════════════════════════════════

def classify_neurons_cf_gpu(image_var, text_var, noise_var=None,
                             n_permutations=1000, alpha=0.05,
                             noise_percentile=95.0,
                             device='cuda:0', seed=42):
    """Classify all neurons in a layer using counterfactual perturbation.

    Args:
        image_var: (n_neurons, n_samples, K_image) — activations under image
                    variation, text held fixed
        text_var:  (n_neurons, n_samples, K_text)  — activations under text
                    variation, image held fixed
        noise_var: (n_neurons, n_groups, K_noise) — activations under random
                    pairing (None = skip noise pass, use 3-way classification)
        n_permutations, alpha, noise_percentile, seed: standard

    Returns:
        results: list of dicts (one per neuron) with label, p_value, etc.
        threshold: float — engagement threshold (None if skip_noise)
    """
    image = torch.tensor(image_var, device=device, dtype=torch.float32)
    text = torch.tensor(text_var, device=device, dtype=torch.float32)

    n_neurons = image.shape[0]

    # Within-group variance per neuron
    var_image = image.var(dim=2).mean(dim=1)
    var_text = text.var(dim=2).mean(dim=1)
    observed_D = var_image - var_text
    max_signal = torch.maximum(var_image, var_text)

    # ─── Noise threshold (if noise pass available) ───
    threshold = None
    if noise_var is not None:
        noise = torch.tensor(noise_var, device=device, dtype=torch.float32)
        var_noise = noise.var(dim=2).mean(dim=1)
        var_noise_np = var_noise.cpu().numpy()
        threshold = float(np.percentile(var_noise_np, noise_percentile))
        print(f'  Noise variance: median={np.median(var_noise_np):.6f}, '
              f'p95={np.percentile(var_noise_np, 95):.6f}, '
              f'max={var_noise_np.max():.6f}')
        print(f'  Engagement threshold (p{noise_percentile}): {threshold:.6f}')
    else:
        var_noise = None
        var_noise_np = None

    # ─── Permutation test (vectorized) ───
    n_image_total = image.shape[1] * image.shape[2]
    n_text_total = text.shape[1] * text.shape[2]
    pooled = torch.cat(
        [image.reshape(n_neurons, -1),
         text.reshape(n_neurons, -1)],
        dim=1
    )

    rng = torch.Generator(device=device).manual_seed(seed)
    null_Ds = torch.zeros(n_neurons, n_permutations, device=device)
    for p in range(n_permutations):
        perm = torch.randperm(n_image_total + n_text_total,
                                generator=rng, device=device)
        shuf = pooled[:, perm]
        i_shuf = shuf[:, :n_image_total].reshape(image.shape)
        t_shuf = shuf[:, n_image_total:].reshape(text.shape)
        null_Ds[:, p] = i_shuf.var(dim=2).mean(dim=1) - \
                       t_shuf.var(dim=2).mean(dim=1)

    p_values = (null_Ds.abs() >= observed_D.abs().unsqueeze(1)).float().mean(dim=1)

    # Move to CPU for label assignment
    var_image_np = var_image.cpu().numpy()
    var_text_np = var_text.cpu().numpy()
    observed_D_np = observed_D.cpu().numpy()
    p_values_np = p_values.cpu().numpy()
    max_signal_np = max_signal.cpu().numpy()

    # ─── Label assignment ───
    results = []
    for n in range(n_neurons):
        max_sig = max_signal_np[n]
        p = p_values_np[n]
        D = observed_D_np[n]

        if threshold is not None and max_sig <= threshold:
            label = 'unknown'
        elif p < alpha:
            label = 'visual' if D > 0 else 'text'
        else:
            if threshold is None:
                label = 'non-specialized'
            else:
                label = 'multimodal'

        entry = {
            'neuron_idx': n,
            'label': label,
            'p_value': float(p),
            'observed_D': float(D),
            'var_image': float(var_image_np[n]),
            'var_text': float(var_text_np[n]),
            'max_signal': float(max_sig),
        }
        if var_noise_np is not None:
            entry['var_noise'] = float(var_noise_np[n])
        results.append(entry)

    return results, threshold


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Helpers
# ═══════════════════════════════════════════════════════════════════

def _resolve_img_path(img_dir, img_id):
    """Find image file for a given ID. Tries multiple naming conventions."""
    # COCO standard: 12-digit zero-padded
    candidates = [
        os.path.join(img_dir, f'{int(img_id):012d}.jpg'),
        os.path.join(img_dir, f'COCO_train2017_{int(img_id):012d}.jpg'),
        os.path.join(img_dir, f'COCO_val2017_{int(img_id):012d}.jpg'),
        os.path.join(img_dir, f'{img_id}.jpg'),
        os.path.join(img_dir, f'{img_id}'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _build_random_pairs(paired, n_groups, K_per_group, seed):
    """Build random (image, caption) groups for noise pass."""
    rng = np.random.RandomState(seed)
    all_captions = []
    for iid, data in paired.items():
        for cap in data.get('captions', []):
            all_captions.append((iid, cap))
    available_imgs = list(paired.keys())

    groups = []
    for g in range(n_groups):
        group = []
        for _ in range(K_per_group):
            rand_img = available_imgs[rng.randint(len(available_imgs))]
            _, rand_cap = all_captions[rng.randint(len(all_captions))]
            group.append((rand_img, rand_cap))
        groups.append(group)
    return groups


def _hook_suffix(hook_point):
    """Hook suffix for output directory naming (matches run_pipeline.sh)."""
    return f'_{hook_point}'  # _gate, _gate_up, _attn


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print('═' * 70)
    print('  CF Classifier (cf_classify.py)')
    print('═' * 70)
    print(f'  Model:    {args.model_type} ({args.original_model_path})')
    print(f'  Layers:   {args.layer_start}-{args.layer_end}')
    print(f'  Hook:     {args.hook_point}')
    print(f'  K_image:  {args.K_image}    K_text: {args.K_text}')
    print(f'  Noise:    {args.n_noise_pairs} groups × {args.noise_K} '
          f'(skip={args.skip_noise_pass})')
    print(f'  Output:   {args.output_dir}')
    print()

    # Normalize model args
    effective_type, is_llava_llama3, hf_id = _normalize_model_args(args)

    # Setup device + RNG
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    layer_indices = list(range(args.layer_start, args.layer_end))
    # n_layers_total is the model's TOTAL layer count — computed below
    # once the model is loaded (its config tells us the truth).

    # Load paired data
    if not os.path.exists(args.paired_data_path):
        sys.exit(f'Paired data not found: {args.paired_data_path}\n'
                  f'Build it first with build_cf_paired_data.py')
    print(f'Loading paired data from {args.paired_data_path} …')
    with open(args.paired_data_path) as f:
        paired = json.load(f)
    sample_ids = list(paired.keys())
    if args.num_samples is not None:
        sample_ids = sample_ids[:args.num_samples]
    print(f'  Using {len(sample_ids)} samples')

    # Load model
    model_loaded = _load_model_dispatch(args, effective_type, device)
    # Different loaders return different tuples; standardize
    if effective_type == 'internvl':
        model, tokenizer, image_token_id = model_loaded
        processor = None
    else:
        model, processor, image_token_id = model_loaded
        tokenizer = None

    # Determine the model's actual total layer count from its config.
    # Pattern matches neuron_snrf_merge.py: VLMs (LLaVA-Next, LLaVA-OV,
    # InternVL, Qwen2VL) put the LM under `config.text_config`.
    cfg = model.config
    n_layers_total = getattr(cfg, 'num_hidden_layers', None)
    if n_layers_total is None:
        text_cfg = getattr(cfg, 'text_config', None)
        if text_cfg is not None:
            n_layers_total = getattr(text_cfg, 'num_hidden_layers', None)
    if n_layers_total is None:
        n_layers_total = getattr(cfg, 'num_layers', 32)
    print(f'  Model has {n_layers_total} transformer blocks; '
          f'requested layers {args.layer_start}-{args.layer_end - 1}')

    # Layer setup — uses existing pipeline's get_layer_names
    if args.hook_point == 'gate':
        layer_names = get_layer_names(effective_type, n_layers_total, model)
        retain_input = False
    elif args.hook_point == 'gate_up':
        layer_names = get_down_proj_names(effective_type, n_layers_total, model)
        retain_input = True
    elif args.hook_point == 'attn':
        layer_names = get_o_proj_names(effective_type, n_layers_total, model)
        retain_input = True
    else:
        sys.exit(f'Unknown hook_point: {args.hook_point}')

    # Resolve actual list of layer names for the requested range
    # (get_layer_names returns one per layer in the model's range)
    if len(layer_names) == n_layers_total:
        full_layer_names = []
        for li in layer_indices:
            # `layer_names` covers ALL transformer blocks (0..n_layers_total-1),
            # so use the absolute layer index `li` directly. Using
            # `li - args.layer_start` would silently hook layers.0 for the
            # first requested layer and shift every other layer downward.
            full_layer_names.append(layer_names[li])
        layer_names = full_layer_names

    # Number of attention heads (only relevant for attn hook)
    n_heads = None
    if args.hook_point == 'attn':
        n_heads = _get_num_attention_heads(model, effective_type)

    # Probe forward pass to get n_neurons
    print('Probing for n_neurons …')
    probe_id = sample_ids[0]
    probe_path = _resolve_img_path(args.coco_img_dir, probe_id)
    if probe_path is None:
        sys.exit(f'Could not find probe image: {probe_id} in {args.coco_img_dir}')
    probe_img = Image.open(probe_path).convert('RGB')
    probe_caption = paired[probe_id]['canonical_caption']
    probe_acts = _record_activations_one_pass(
        model, args, effective_type, is_llava_llama3,
        processor, tokenizer, probe_img, probe_caption,
        layer_names, list(range(len(layer_indices))), retain_input,
        args.hook_point, device, image_token_id, n_heads,
    )
    n_neurons = probe_acts[0].shape[0]
    print(f'  n_neurons per layer: {n_neurons}')

    # ─── Allocate storage ───
    K_v = args.K_image
    K_t = args.K_text
    K_n = args.noise_K
    n_samples = len(sample_ids)

    image_var_acts = {
        i: np.zeros((n_samples, K_v, n_neurons), dtype=np.float32)
        for i in range(len(layer_indices))
    }
    text_var_acts = {
        i: np.zeros((n_samples, K_t, n_neurons), dtype=np.float32)
        for i in range(len(layer_indices))
    }
    noise_var_acts = None
    if not args.skip_noise_pass:
        noise_var_acts = {
            i: np.zeros((args.n_noise_pairs, K_n, n_neurons), dtype=np.float32)
            for i in range(len(layer_indices))
        }

    # ─── CF Pass ───
    print('\nCF Pass: recording activations under controlled perturbation…')
    t0 = time.time()
    skipped = 0
    for s_idx, img_id in enumerate(tqdm(sample_ids, desc='CF Pass')):
        sample_data = paired[img_id]
        canonical_caption = sample_data['canonical_caption']
        captions = sample_data['captions'][:K_t]
        similar_imgs = sample_data['similar_images'][:K_v]

        canonical_path = _resolve_img_path(args.coco_img_dir, img_id)
        if canonical_path is None:
            skipped += 1
            continue
        canonical_img = Image.open(canonical_path).convert('RGB')

        # Vary text, hold image fixed
        for k, cap in enumerate(captions):
            try:
                acts = _record_activations_one_pass(
                    model, args, effective_type, is_llava_llama3,
                    processor, tokenizer, canonical_img, cap,
                    layer_names, list(range(len(layer_indices))),
                    retain_input, args.hook_point, device, image_token_id, n_heads,
                )
                for li_idx in range(len(layer_indices)):
                    text_var_acts[li_idx][s_idx, k] = acts[li_idx]
            except Exception as e:
                print(f'  [skip text {k} of {img_id}]: {e}')

        # Vary image, hold text fixed
        for k, sim_id in enumerate(similar_imgs):
            sim_path = _resolve_img_path(args.coco_img_dir, sim_id)
            if sim_path is None:
                continue
            try:
                sim_img = Image.open(sim_path).convert('RGB')
                acts = _record_activations_one_pass(
                    model, args, effective_type, is_llava_llama3,
                    processor, tokenizer, sim_img, canonical_caption,
                    layer_names, list(range(len(layer_indices))),
                    retain_input, args.hook_point, device, image_token_id, n_heads,
                )
                for li_idx in range(len(layer_indices)):
                    image_var_acts[li_idx][s_idx, k] = acts[li_idx]
            except Exception as e:
                print(f'  [skip image {k} of {img_id}]: {e}')

    print(f'CF Pass done in {(time.time()-t0)/60:.1f} min ({skipped} skipped)')

    # ─── Noise Pass ───
    if not args.skip_noise_pass:
        print('\nNoise Pass: recording activations on random pairs…')
        noise_groups = _build_random_pairs(
            paired, args.n_noise_pairs, K_n, args.seed + 1000)

        t0 = time.time()
        for g_idx, group in enumerate(tqdm(noise_groups, desc='Noise Pass')):
            for k, (rand_img_id, rand_cap) in enumerate(group):
                rand_path = _resolve_img_path(args.coco_img_dir, rand_img_id)
                if rand_path is None:
                    continue
                try:
                    rand_img = Image.open(rand_path).convert('RGB')
                    acts = _record_activations_one_pass(
                        model, args, effective_type, is_llava_llama3,
                        processor, tokenizer, rand_img, rand_cap,
                        layer_names, list(range(len(layer_indices))),
                        retain_input, args.hook_point, device, image_token_id, n_heads,
                    )
                    for li_idx in range(len(layer_indices)):
                        noise_var_acts[li_idx][g_idx, k] = acts[li_idx]
                except Exception as e:
                    print(f'  [skip noise {k} of {g_idx}]: {e}')
        print(f'Noise Pass done in {(time.time()-t0)/60:.1f} min')

    # ─── Free model memory before classification ───
    del model
    if processor is not None:
        del processor
    torch.cuda.empty_cache()

    # ─── Classify per layer ───
    suffix = _hook_suffix(args.hook_point) + args.output_suffix
    out_root = os.path.join(args.output_dir, args.model,
                              f'llm_cf_permutation{suffix}')
    os.makedirs(out_root, exist_ok=True)

    print(f'\nClassifying neurons; output to {out_root}')
    summary_per_layer = {}

    for li_idx, layer_idx in enumerate(layer_indices):
        layer_name = layer_names[li_idx]
        print(f'\n  Layer {layer_idx} ({layer_name})')

        img_acts = image_var_acts[li_idx].transpose(2, 0, 1)
        txt_acts = text_var_acts[li_idx].transpose(2, 0, 1)
        noi_acts = (noise_var_acts[li_idx].transpose(2, 0, 1)
                     if noise_var_acts is not None else None)

        results, threshold = classify_neurons_cf_gpu(
            img_acts, txt_acts, noi_acts,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            noise_percentile=args.noise_percentile,
            device=device,
            seed=args.seed + layer_idx,
        )

        # Save labels
        layer_dir = os.path.join(out_root, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        labels_path = os.path.join(layer_dir, 'neuron_labels_cf_permutation.json')
        with open(labels_path, 'w') as f:
            json.dump({
                'method': 'cf_perturbation_v1',
                'engaged_threshold': threshold,
                'noise_percentile': args.noise_percentile if not args.skip_noise_pass else None,
                'alpha': args.alpha,
                'n_permutations': args.n_permutations,
                'K_image': K_v,
                'K_text': K_t,
                'n_samples': n_samples,
                'hook_point': args.hook_point,
                'results': results,
            }, f, indent=2)
        print(f'  Saved {labels_path}')

        # Distribution
        label_counts = defaultdict(int)
        for r in results:
            label_counts[r['label']] += 1
        total = sum(label_counts.values())
        print(f'  Label distribution:')
        for lbl in sorted(label_counts.keys()):
            n = label_counts[lbl]
            print(f'    {lbl:<18} {n:>6} ({100*n/total:>5.1f}%)')
        summary_per_layer[layer_idx] = {
            'layer_name': layer_name,
            'engaged_threshold': threshold,
            'distribution': dict(label_counts),
        }

    # Per-shard stats file (matches PMBT naming convention)
    stats_path = os.path.join(
        out_root,
        f'cf_permutation_stats_layers{args.layer_start}-{args.layer_end}.json')
    with open(stats_path, 'w') as f:
        json.dump(summary_per_layer, f, indent=2)
    print(f'\nStats saved to {stats_path}')

    print('\n' + '═' * 70)
    print('CF classification complete')
    print('═' * 70)


if __name__ == '__main__':
    main()
