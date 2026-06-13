#!/usr/bin/env python3
"""
neuron_srf_merge.py — Layer 1c: PMBT-Guided Spectral Representation Filtering

Two-stage pipeline:
  Stage 1 (--stage profile): Run VLM on contrastive POPE sets (truthful vs
    hallucinated), collect per-layer hidden states (d_model), average over
    token positions, compute per-layer hallucination covariance
    Σ_H = (1/N) Σ d_i d_i^T where d_i = h_halluc - h_truthful,
    eigendecompose → full eigenspectrum → save.

  Stage 2 (--stage edit): Load eigenspectrum + PMBT visual mask, construct
    soft spectral suppression operator P_α = Q diag(1/(1+αλ_j)) Q^T
    using the FULL eigenspectrum (all eigenvalues, not top-k), then apply
    W_corr = P_α @ W_out to down_proj weights. For PMBT-guided mode,
    only visual neuron columns are modified.

The soft spectral filter (Eq 7-8 of Ali et al.) attenuates high-variance
hallucination modes proportionally to their eigenvalue magnitude while
preserving low-variance (non-hallucination) directions. This is in contrast
to Nullu's hard projection which completely removes top-k directions.

α is auto-computed per the paper's matheuristic: α = (1-η)/(η·λ₁)
with η=0.1, tying suppression strength to the dominant eigenvalue.

Reference: "Suppressing VLM Hallucinations with Spectral Representation
  Filtering" (Ali, Zoabi & Wolf, arXiv 2511.12220, Nov 2025).
  Adapted to restrict the filter to PMBT visual neurons only.

Compatible with run_pipeline.sh as step 24.

Usage:
  # Stage 1: profile hallucination modes from contrastive POPE
  python neuron_srf_merge.py --stage profile \
      --vlm_path llava-hf/llava-onevision-qwen2-7b-ov-hf \
      --model_type llava-ov \
      --pope_path data/POPE \
      --pope_img_dir data/coco/val2014 \
      --output_dir results/24-srf/llava-ov

  # Stage 2: apply spectral filter at visual neuron positions
  python neuron_srf_merge.py --stage edit \
      --vlm_path llava-hf/llava-onevision-qwen2-7b-ov-hf \
      --model_type llava-ov \
      --label_dir results/3-classify/full/llava-onevision-7b/llm_permutation \
      --eigenvecs_dir results/24-srf/llava-ov \
      --output_dir results/24-srf/llava-ov \
      --eta 0.1 --min_layer_pct 0.5
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
        description='PMBT-guided SRF: spectral filter on visual MH-Neurons')

    # Stage selection
    p.add_argument('--stage', required=True,
                   choices=['profile', 'edit', 'both', 'eval'],
                   help='profile = collect contrastive activations + '
                        'eigendecompose; edit = apply spectral filter; '
                        'both = run profile then edit; '
                        'eval = evaluate edited model on POPE + TriviaQA')

    # Model paths
    p.add_argument('--vlm_path', required=True,
                   help='HuggingFace path to the target VLM')
    p.add_argument('--model_type', required=True,
                   choices=['llava-ov', 'qwen2vl', 'internvl',
                            'llava-hf', 'llava-liuhaotian', 'llava-llama3'],
                   help='Model type (determines weight path naming)')
    p.add_argument('--n_layers', type=int, default=None,
                   help='Number of transformer layers (auto-detected if None)')

    # PMBT labels (only needed for edit, unless --all_neurons)
    p.add_argument('--label_dir', default=None,
                   help='Directory containing PMBT neuron labels')
    p.add_argument('--all_neurons', action='store_true',
                   help='Apply SRF to ALL down_proj columns (vanilla SRF, '
                        'no PMBT filtering). Use for reproducing paper results.')

    # Profile stage params — contrastive data
    p.add_argument('--lure_path', default=None,
                   help='Path to LURE filter_cap.json or hallucination5k_train.jsonl. '
                        'This is the contrastive dataset used in the SRF paper.')
    p.add_argument('--lure_img_dir', default=None,
                   help='Path to COCO train2014 images (for LURE). '
                        'If None, uses --pope_img_dir.')
    p.add_argument('--pope_path', default='data/POPE',
                   help='Path to POPE question files (fallback if no LURE)')
    p.add_argument('--pope_img_dir', default='data/coco/val2014',
                   help='Path to COCO val2014 images')
    p.add_argument('--contrastive_dir', default=None,
                   help='Path to precomputed contrastive sets from step 9 '
                        '(if available, skips POPE evaluation)')
    p.add_argument('--n_profile_samples', type=int, default=500,
                   help='Number of contrastive samples to use for profiling '
                        '(SRF paper uses LURE with ~5000 samples)')

    # Edit stage params
    p.add_argument('--eigenvecs_dir', default=None,
                   help='Directory containing hallucination eigenspectrum '
                        '(auto-detected if output_dir matches)')
    p.add_argument('--eta', type=float, default=0.1,
                   help='Retained variance fraction for dominant eigenvalue '
                        '(default: 0.1). α is auto-computed as '
                        '(1-η)/(η·λ₁) per SRF paper Eq. 12.')
    p.add_argument('--alpha_override', type=float, default=None,
                   help='If set, override auto-computed α with this value. '
                        'SRF paper uses α=70 for CHAIR, α=6 for POPE.')
    p.add_argument('--min_layer_pct', type=float, default=0.5,
                   help='Apply filter only to deeper layers (default: 0.5 = '
                        'top 50%% of layers). SRF paper applies to deeper '
                        'layers where hallucination modes are strongest.')
    p.add_argument('--save_model', action='store_true', default=True,
                   help='Save the edited model to disk')
    p.add_argument('--include_random', action='store_true',
                   help='Also run random mask (same count as visual, sparsity control)')
    p.add_argument('--random_seed', type=int, default=42,
                   help='Seed for random mask generation')

    # Output
    p.add_argument('--output_dir', required=True,
                   help='Output directory for eigenvectors and edited model')

    # Eval stage params
    p.add_argument('--edited_model_dir', default=None,
                   help='Path to edited model state dict directory '
                        '(for --stage eval). If None, auto-detected from '
                        'output_dir.')
    p.add_argument('--triviaqa_path', default=None,
                   help='Path to TriviaQA dataset for text evaluation. '
                        'If None, skips TriviaQA eval.')
    p.add_argument('--eval_pope_splits', default='random,popular,adversarial',
                   help='Comma-separated POPE splits to evaluate')
    p.add_argument('--eval_max_samples', type=int, default=500,
                   help='Max images per POPE split for evaluation')
    p.add_argument('--eval_max_tqa', type=int, default=200,
                   help='Max TriviaQA questions for evaluation')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Weight path helpers (shared with neuron_weight_merge.py)
# ═══════════════════════════════════════════════════════════════════

def get_mlp_weight_paths(model_type, layer_idx):
    """Return (gate, up, down) weight paths for one VLM layer."""
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


# ═══════════════════════════════════════════════════════════════════
# Section 3 — PMBT label loading
# ═══════════════════════════════════════════════════════════════════

def load_pmbt_labels(label_dir, taxonomy='pmbt'):
    """Load PMBT neuron labels. Returns (labels_by_layer, n_layers, n_neurons)."""
    label_filename = ('neuron_labels_permutation.json' if taxonomy == 'pmbt'
                      else 'neuron_labels.json')
    merged_path = os.path.join(
        label_dir, label_filename.replace('.json', '_all.json'))
    labels_by_layer = {}
    if os.path.isfile(merged_path):
        print(f'  Loading PMBT labels from {merged_path}')
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
                labels_by_layer[int(m.group(1))] = json.load(open(fpath))
    if not labels_by_layer:
        raise FileNotFoundError(f'No PMBT labels in {label_dir}')
    n_layers = max(labels_by_layer.keys()) + 1
    n_neurons = len(labels_by_layer[min(labels_by_layer.keys())])
    print(f'  Loaded: {n_layers} layers × {n_neurons} neurons')
    return labels_by_layer, n_layers, n_neurons


def build_visual_mask(labels_by_layer, n_layers, n_neurons):
    """Build per-layer boolean masks for visual neurons.

    Returns dict {layer_idx: BoolTensor (n_neurons,)}.
    """
    masks = {}
    for layer_idx in range(n_layers):
        mask = torch.zeros(n_neurons, dtype=torch.bool)
        if layer_idx in labels_by_layer:
            for entry in labels_by_layer[layer_idx]:
                if entry.get('label') == 'visual' and entry['neuron_idx'] < n_neurons:
                    mask[entry['neuron_idx']] = True
        masks[layer_idx] = mask
    total = sum(m.sum().item() for m in masks.values())
    print(f'  Visual neuron mask: {total:,} neurons '
          f'({100 * total / (n_layers * n_neurons):.1f}%)')
    return masks


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Hallucination mode profiling (Stage 1)
#   SRF paper method:
#   1. Load LURE dataset: paired (image, truthful_caption, halluc_caption)
#   2. Teacher-force each caption through the VLM (image + caption as input)
#   3. Extract per-layer hidden states, average over all token positions
#   4. Compute d_i = h_halluc_i - h_truthful_i for each sample
#   5. Build Σ_H = (1/N) Σ d_i d_i^T, eigendecompose
# ═══════════════════════════════════════════════════════════════════

def load_lure_dataset(lure_path, n_samples):
    """Load LURE contrastive caption dataset.

    Supports two formats from the LURE repo:
      1. filter_cap.json: {'image_id', 'caption' (truthful), 'h_caption' (halluc)}
      2. hallucination5k_train.jsonl: {'value' (truthful), 'h_value' (halluc),
         'image_id', ...}

    Returns:
        pairs: list of (image_id, truthful_caption, hallucinated_caption) tuples
    """
    pairs = []

    if lure_path.endswith('.jsonl'):
        # JSONL format (hallucination5k_train.jsonl)
        with open(lure_path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                img_id = item.get('image_id', item.get('image', ''))
                truthful = item.get('value', item.get('caption', ''))
                halluc = item.get('h_value', item.get('h_caption', ''))
                if truthful and halluc and img_id:
                    pairs.append((str(img_id), truthful, halluc))
    elif lure_path.endswith('.json'):
        # JSON format (filter_cap.json)
        with open(lure_path) as f:
            data = json.load(f)
        # filter_cap.json can be a dict or list
        if isinstance(data, dict):
            # dict keyed by image_id
            for img_id, item in data.items():
                if isinstance(item, dict):
                    truthful = item.get('caption', '')
                    halluc = item.get('h_caption', '')
                    if truthful and halluc:
                        pairs.append((str(img_id), truthful, halluc))
        elif isinstance(data, list):
            for item in data:
                img_id = item.get('image_id', item.get('image', ''))
                truthful = item.get('caption', item.get('value', ''))
                halluc = item.get('h_caption', item.get('h_value', ''))
                if truthful and halluc and img_id:
                    pairs.append((str(img_id), truthful, halluc))

    # truncate
    if n_samples and len(pairs) > n_samples:
        pairs = pairs[:n_samples]

    print(f'  Loaded {len(pairs)} LURE contrastive caption pairs')
    return pairs


def collect_hidden_states_paired(model, tokenizer, processor, pairs,
                                  img_dir, n_layers, model_type='auto'):
    """Collect sequence-averaged hidden states from paired image+caption inputs.

    SRF paper method (Eq. 3): for each sample i, feed image + caption through
    the model via teacher-forcing and extract the hidden state at each layer,
    averaged over all token positions.

    This runs TWO forward passes per sample:
      1. (image, truthful_caption) → truthful hidden states
      2. (image, hallucinated_caption) → hallucinated hidden states

    Args:
        pairs: list of (image_id, truthful_caption, hallucinated_caption)
        img_dir: path to image directory
        model_type: model type string for input format handling

    Returns:
        truthful_acts: dict {layer_idx: tensor (n_samples, d_model)}
        halluc_acts:   dict {layer_idx: tensor (n_samples, d_model)}
    """
    # Register hooks on transformer layer outputs
    layer_outputs = {}

    def make_layer_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # average over all token positions (SRF paper Eq. 3)
            hs_avg = hs.detach().float().mean(dim=1)  # (batch, d_model)
            if layer_idx not in layer_outputs:
                layer_outputs[layer_idx] = []
            layer_outputs[layer_idx].append(hs_avg.cpu())
        return hook_fn

    handles = []
    for name, module in model.named_modules():
        m = re.search(
            r'(?:language_model\.model|model)\.layers\.(\d+)$', name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx < n_layers:
                h = module.register_forward_hook(make_layer_hook(layer_idx))
                handles.append(h)

    print(f'  Registered hooks on {len(handles)} transformer layers')

    from PIL import Image

    # ── LLaVA-1.5 (liuhaotian) specific: uses original repo format ──
    is_llava15 = (model_type == 'llava-liuhaotian')
    if is_llava15:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        # processor is image_processor (CLIPImageProcessor) for llava-1.5
        image_processor = processor

    def _prepare_caption_input(image, caption):
        """Prepare inputs for teacher-forcing: image + caption as input text."""
        if is_llava15:
            # LLaVA-1.5 original repo format:
            #   prompt with <image> token → tokenize → process image separately
            prompt = (f'USER: {DEFAULT_IMAGE_TOKEN}\n'
                      f'Describe this image in detail.\n'
                      f'ASSISTANT: {caption}')
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors='pt').unsqueeze(0)  # (1, seq_len)
            img_tensor = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'].half()
            return {
                'input_ids': input_ids.to(model.device),
                'images': img_tensor.to(model.device),
            }
        elif processor is not None:
            try:
                # Chat-template models: format as assistant response
                msgs = [
                    {'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': 'Describe this image in detail.'}]},
                    {'role': 'assistant', 'content': [
                        {'type': 'text', 'text': caption}]}
                ]
                formatted = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
                inputs = processor(
                    text=formatted, images=image, return_tensors='pt')
            except Exception:
                try:
                    # LLaVA-1.5 HF style
                    formatted = (f'USER: <image>\nDescribe this image in detail.'
                                 f'\nASSISTANT: {caption}')
                    inputs = processor(
                        text=formatted, images=image, return_tensors='pt')
                except Exception:
                    # Fallback
                    inputs = processor(
                        text=caption, images=image, return_tensors='pt')
            return {k: v.to(model.device) if hasattr(v, 'to') else v
                    for k, v in inputs.items()}
        else:
            inputs = tokenizer(caption, return_tensors='pt')
            return {k: v.to(model.device) if hasattr(v, 'to') else v
                    for k, v in inputs.items()}

    # Collect hidden states for truthful and hallucinated captions
    truthful_layers = {l: [] for l in range(n_layers)}
    halluc_layers = {l: [] for l in range(n_layers)}

    for img_id, truthful_cap, halluc_cap in tqdm(pairs, desc='  Profiling'):
        # Resolve image path
        # LURE image_ids may be just numbers or full filenames
        img_name = img_id if '.' in img_id else f'COCO_train2014_{int(img_id):012d}.jpg'
        img_path = os.path.join(img_dir, img_name)
        if not os.path.isfile(img_path):
            # try without COCO prefix
            img_path = os.path.join(img_dir, f'{img_id}.jpg')
        if not os.path.isfile(img_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        # ── Pass 1: truthful caption ──
        layer_outputs.clear()
        try:
            inputs = _prepare_caption_input(image, truthful_cap)
            with torch.no_grad():
                model(**inputs)
            for l in layer_outputs:
                truthful_layers[l].append(layer_outputs[l][-1])
        except Exception:
            continue

        # ── Pass 2: hallucinated caption ──
        layer_outputs.clear()
        try:
            inputs = _prepare_caption_input(image, halluc_cap)
            with torch.no_grad():
                model(**inputs)
            for l in layer_outputs:
                halluc_layers[l].append(layer_outputs[l][-1])
        except Exception:
            # remove the truthful entry to keep pairs aligned
            for l in truthful_layers:
                if truthful_layers[l]:
                    truthful_layers[l].pop()
            continue

    # Remove hooks
    for h in handles:
        h.remove()

    # Stack: {layer_idx: (n_samples, d_model)}
    truthful_acts = {}
    halluc_acts = {}
    for l in range(n_layers):
        if truthful_layers[l] and halluc_layers[l]:
            truthful_acts[l] = torch.cat(truthful_layers[l], dim=0)
            halluc_acts[l] = torch.cat(halluc_layers[l], dim=0)

    n_paired = min(len(v) for v in truthful_acts.values()) if truthful_acts else 0
    print(f'  Collected {n_paired} paired samples across {len(truthful_acts)} layers')

    return truthful_acts, halluc_acts

def load_contrastive_pope(pope_path, n_samples):
    """Load POPE questions and split into truthful/hallucinated sets.

    Supports two formats:
      1. Contrastive JSONL from step 10 (preferred): each line has
         'consistent' field ('correct' or 'incorrect')
      2. Raw POPE directory: loads coco_pope_{split}.json files and
         uses ground truth labels (yes=truthful, no=hallucination-prone)

    Returns:
        truthful: list of (image_path, question) tuples
        hallucinated: list of (image_path, question) tuples
    """
    truthful = []
    hallucinated = []

    # Format 1: contrastive JSONL from step 10
    if pope_path.endswith('.jsonl') and os.path.isfile(pope_path):
        with open(pope_path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get('text', item.get('question', ''))
                img = item.get('image', '')
                consistency = item.get('contrastive_label', item.get('consistent', '')).lower()
                if consistency in ('faithful', 'correct'):
                    truthful.append((img, q))
                elif consistency in ('hallucinated', 'incorrect'):
                    hallucinated.append((img, q))
        print(f'  Loaded contrastive JSONL: {len(truthful)} correct, '
              f'{len(hallucinated)} hallucinated')

    # Format 2: raw POPE directory with per-split JSON files
    elif os.path.isdir(pope_path):
        for split in ['adversarial', 'popular', 'random']:
            pope_file = os.path.join(pope_path, f'coco_pope_{split}.json')
            if not os.path.isfile(pope_file):
                continue
            with open(pope_file) as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    data = json.load(f)
                else:
                    data = [json.loads(line) for line in f if line.strip()]
            for item in data:
                q = item.get('text', item.get('question', ''))
                img = item.get('image', '')
                answer = item.get('label', item.get('answer', '')).lower().strip()
                if answer == 'yes':
                    truthful.append((img, q))
                elif answer == 'no':
                    hallucinated.append((img, q))

    # Format 3: single raw POPE file (JSON array or JSONL)
    elif os.path.isfile(pope_path):
        with open(pope_path) as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # Standard JSON array
                data = json.load(f)
            else:
                # JSONL: one JSON object per line
                data = [json.loads(line) for line in f if line.strip()]
        for item in data:
            q = item.get('text', item.get('question', ''))
            img = item.get('image', '')
            answer = item.get('label', item.get('answer', '')).lower().strip()
            if answer == 'yes':
                truthful.append((img, q))
            elif answer == 'no':
                hallucinated.append((img, q))

    # balance and truncate
    n = min(n_samples // 2, len(truthful), len(hallucinated))
    if n == 0:
        raise ValueError(f'No POPE questions found in {pope_path}')
    truthful = truthful[:n]
    hallucinated = hallucinated[:n]
    print(f'  Loaded {n} truthful + {n} hallucinated POPE questions')
    return truthful, hallucinated


def collect_hidden_states(model, tokenizer, processor, questions,
                          img_dir, n_layers):
    """Collect sequence-averaged hidden states at each layer output.

    For each question, runs a forward pass and records the hidden state
    output of each transformer layer, averaged over all token positions.
    This produces vectors in d_model space (e.g., 4096-dim), matching
    the SRF paper Eq. 3.

    Returns:
        activations: dict {layer_idx: tensor of shape (n_questions, d_model)}
    """
    # Hook on each transformer layer's output to capture hidden states.
    # In HF transformers, each layer module returns (hidden_states, ...).
    # We register a hook on the full layer (not sub-modules) to get the
    # residual-stream output in d_model space.
    layer_outputs = {}  # layer_idx → list of (d_model,) tensors

    def make_layer_hook(layer_idx):
        """Create a hook that captures the layer's output hidden state,
        averages it over the sequence length dimension, and stores it."""
        def hook_fn(module, input, output):
            # output is typically a tuple: (hidden_states, ...)
            # hidden_states shape: (batch, seq_len, d_model)
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # average over all token positions (SRF paper Eq. 3)
            hs_avg = hs.detach().float().mean(dim=1)  # (batch, d_model)
            if layer_idx not in layer_outputs:
                layer_outputs[layer_idx] = []
            layer_outputs[layer_idx].append(hs_avg.cpu())
        return hook_fn

    # Find transformer layer modules and register hooks.
    # Pattern: model.language_model.model.layers.{i} or model.layers.{i}
    handles = []
    for name, module in model.named_modules():
        # Match the full decoder layer (not sub-modules like mlp or attn)
        m = re.search(
            r'(?:language_model\.model|model)\.layers\.(\d+)$', name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx < n_layers:
                h = module.register_forward_hook(make_layer_hook(layer_idx))
                handles.append(h)

    print(f'  Registered hooks on {len(handles)} transformer layers')

    # Run forward pass for each question
    from PIL import Image
    for img_name, question in tqdm(questions, desc='  Collecting hidden states'):
        img_path = os.path.join(img_dir, img_name) if img_dir else img_name
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        # Prepare inputs — must include image tokens for VLMs
        try:
            if processor is not None:
                try:
                    # Chat-template models (LLaVA-OV, Qwen2VL, LLaVA-LLaMA3)
                    msgs = [{'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': question}]}]
                    formatted = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                    inputs = processor(
                        text=formatted, images=image, return_tensors='pt')
                except Exception:
                    try:
                        # LLaVA-1.5 style
                        formatted = f'USER: <image>\n{question}\nASSISTANT:'
                        inputs = processor(
                            text=formatted, images=image, return_tensors='pt')
                    except Exception:
                        inputs = processor(
                            text=question, images=image, return_tensors='pt')
            else:
                inputs = tokenizer(question, return_tensors='pt')
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}
        except Exception:
            continue

        with torch.no_grad():
            try:
                model(**inputs)
            except TypeError:
                try:
                    text_inputs = tokenizer(question, return_tensors='pt')
                    text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}
                    if hasattr(model, 'language_model'):
                        model.language_model(**text_inputs)
                    else:
                        model(**text_inputs)
                except Exception:
                    continue

    # Remove hooks
    for h in handles:
        h.remove()

    # Stack activations: {layer_idx: (n_questions, d_model)}
    activations = {}
    for layer_idx in sorted(layer_outputs.keys()):
        acts = layer_outputs[layer_idx]
        if acts:
            activations[layer_idx] = torch.cat(acts, dim=0)

    return activations


def compute_hallucination_modes(truthful_acts, halluc_acts):
    """Compute full hallucination eigenspectrum from contrastive hidden states.

    For each layer (SRF paper Eq. 3-6):
    1. Compute difference: d_i = h_halluc_i - h_truthful_i  (per sample)
    2. Build hallucination covariance: Σ_H = (1/N) Σ d_i d_i^T
       Note: paper does NOT subtract the mean (Eq. 5 uses raw d_i d_i^T,
       not centered). This captures both the mean shift and variance.
    3. Eigendecompose: Σ_H = Q Λ Q^T (FULL spectrum, all eigenvalues)

    Returns:
        eigenvectors: dict {layer_idx: tensor (d_model, d_model)} — columns
            of Q, the full orthonormal eigenvector matrix
        eigenvalues: dict {layer_idx: tensor (d_model,)} — all eigenvalues
            in descending order
    """
    eigenvectors = {}
    eigenvalues = {}
    common_layers = set(truthful_acts.keys()) & set(halluc_acts.keys())

    for layer_idx in sorted(common_layers):
        h_truth = truthful_acts[layer_idx]   # (n, d_model)
        h_halluc = halluc_acts[layer_idx]    # (m, d_model)

        # Align sample counts
        n = min(h_truth.shape[0], h_halluc.shape[0])
        if n < 5:
            continue

        # Difference vectors d_i = h_halluc_i - h_truthful_i
        # SRF paper Eq. 4: d_i = x+_i - x-_i
        delta = h_halluc[:n].float() - h_truth[:n].float()  # (n, d_model)

        # Hallucination covariance Σ_H = (1/N) Σ d_i d_i^T
        # SRF paper Eq. 5 — note: NO mean subtraction (not centered)
        # Σ_H ∈ R^{d_model × d_model}
        sigma_h = (delta.T @ delta) / n  # (d_model, d_model)

        # Full eigendecomposition: Σ_H = Q Λ Q^T
        # SRF paper Eq. 6 — using ALL eigenvalues, not just top-k
        # torch.linalg.eigh returns eigenvalues in ascending order,
        # eigenvectors as columns of Q
        evals, evecs = torch.linalg.eigh(sigma_h)  # evals: (d,), evecs: (d, d)

        # Reverse to descending order (largest eigenvalue first)
        evals = evals.flip(0)
        evecs = evecs.flip(1)

        # Clamp tiny negative eigenvalues from numerical noise to zero
        evals = evals.clamp(min=0.0)

        eigenvectors[layer_idx] = evecs   # (d_model, d_model) — columns are eigenvectors
        eigenvalues[layer_idx] = evals    # (d_model,)

        # Log spectral summary
        d = evals.shape[0]
        top5 = evals[:5].tolist()
        energy_top10 = evals[:10].sum() / (evals.sum() + 1e-12)
        print(f'    Layer {layer_idx:2d}: d={d}, λ₁={top5[0]:.4f}, '
              f'top-5={[f"{v:.3f}" for v in top5]}, '
              f'top-10 energy={energy_top10:.2%}')

    print(f'  Computed full eigenspectrum for {len(eigenvectors)} layers')
    return eigenvectors, eigenvalues


def run_profile_stage(args):
    """Stage 1: Profile hallucination modes from contrastive data.

    SRF paper method:
    - Uses LURE dataset (paired truthful/hallucinated captions per image)
    - Teacher-forces each caption through VLM to extract hidden states
    - Computes per-layer hallucination covariance and eigendecomposes

    Falls back to POPE-based profiling if LURE is not available.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load VLM for activation profiling ────────────────────────
    print(f'\n  Loading VLM: {args.vlm_path}')
    model, tokenizer, processor = _load_vlm(args.vlm_path, args.model_type)

    # Detect n_layers
    cfg = model.config
    n_layers = args.n_layers
    if n_layers is None:
        for attr in ['num_hidden_layers', 'num_layers']:
            n_layers = getattr(cfg, attr, None)
            if n_layers is not None:
                break
        if n_layers is None:
            for sub in ['text_config', 'llm_config']:
                sub_cfg = getattr(cfg, sub, None)
                if sub_cfg:
                    n_layers = getattr(sub_cfg, 'num_hidden_layers', None)
                    if n_layers:
                        break
    if n_layers is None:
        n_layers = 32
    print(f'  Architecture: {n_layers} layers')

    # ── Load contrastive data (LURE preferred, POPE fallback) ────
    if args.lure_path and os.path.isfile(args.lure_path):
        # ── LURE path: paired captions (matches SRF paper exactly) ──
        print(f'\n  Using LURE dataset (matches SRF paper)')
        pairs = load_lure_dataset(args.lure_path, args.n_profile_samples)
        img_dir = args.lure_img_dir or args.pope_img_dir

        # Collect paired hidden states via teacher-forcing
        truthful_acts, halluc_acts = collect_hidden_states_paired(
            model, tokenizer, processor, pairs, img_dir, n_layers,
            model_type=args.model_type)

    else:
        # ── POPE fallback: separate question sets ──
        print(f'\n  LURE not found, falling back to POPE-based profiling')
        print(f'  NOTE: SRF paper uses LURE, not POPE. Results may differ.')
        truthful_qs, halluc_qs = load_contrastive_pope(
            args.pope_path, args.n_profile_samples)

        print('\n  Collecting truthful hidden states...')
        truthful_acts = collect_hidden_states(
            model, tokenizer, processor, truthful_qs, args.pope_img_dir,
            n_layers)

        print('\n  Collecting hallucinated hidden states...')
        halluc_acts = collect_hidden_states(
            model, tokenizer, processor, halluc_qs, args.pope_img_dir,
            n_layers)

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # ── Compute hallucination eigenspectrum (full, not top-k) ───
    eigenvectors, eigenvalues = compute_hallucination_modes(
        truthful_acts, halluc_acts)

    # ── Save full eigenspectrum ─────────────────────────────────
    # Save as .pt for efficient loading (eigenvectors can be large)
    pt_path = os.path.join(args.output_dir, 'hallucination_modes.pt')
    torch.save({'eigenvectors': eigenvectors, 'eigenvalues': eigenvalues},
               pt_path)
    print(f'  Saved full eigenspectrum (pt) → {pt_path}')

    # Save summary as JSON (eigenvalues only — eigenvectors too large)
    summary = {}
    for layer_idx in sorted(eigenvectors.keys()):
        ev = eigenvalues[layer_idx]
        summary[str(layer_idx)] = {
            'n_eigenvalues': len(ev),
            'top_10_eigenvalues': ev[:10].tolist(),
            'lambda_1': float(ev[0]),
            'total_energy': float(ev.sum()),
            'top_10_energy_fraction': float(ev[:10].sum() / (ev.sum() + 1e-12)),
        }

    json_path = os.path.join(args.output_dir, 'hallucination_spectrum_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved spectrum summary → {json_path}')

    # save stats
    stats = {
        'n_layers_profiled': len(eigenvectors),
        'n_truthful_samples': len(truthful_qs),
        'n_halluc_samples': len(halluc_qs),
        'vlm_path': args.vlm_path,
    }
    # per-layer dominant eigenvalue
    for layer_idx in sorted(eigenvectors.keys()):
        ev = eigenvalues[layer_idx]
        stats[f'layer_{layer_idx}_lambda_1'] = float(ev[0])
        stats[f'layer_{layer_idx}_top10_energy'] = float(
            ev[:10].sum() / (ev.sum() + 1e-12))

    stats_path = os.path.join(args.output_dir, 'srf_profile_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  Saved profile stats → {stats_path}')

    return pt_path


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Spectral filter application (Stage 2)
#   SRF paper Eq. 7-10:
#     P_α = Q diag(1/(1+αλ_j)) Q^T    (soft spectral suppression)
#     W_corr = P_α @ W_out             (left-multiply down_proj)
#
#   P_α operates in d_model space (left side of W_out).
#   For PMBT-guided mode, only visual neuron COLUMNS are modified:
#     W_corr[:, visual] = (P_α @ W_out)[:, visual]
#     W_corr[:, ~visual] = W_out[:, ~visual]   (unchanged)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_suppression_operator(Q, evals, alpha):
    """Build the soft spectral suppression matrix P_α (SRF paper Eq. 7-8).

    P_α = Q diag(1/(1+α·λ_j)) Q^T

    This attenuates directions proportionally to their eigenvalue:
    - Large λ_j (hallucination modes): strongly suppressed, f_α ≈ 1/(αλ_j)
    - Small λ_j (non-hallucination): mostly preserved, f_α ≈ 1 - αλ_j
    - λ_j = 0: completely preserved, f_α = 1

    Args:
        Q:      eigenvector matrix (d_model, d_model) — columns are eigenvectors
        evals:  eigenvalues (d_model,) — in descending order
        alpha:  damping strength (scalar)

    Returns:
        P_alpha: (d_model, d_model) suppression operator
    """
    # f_α(λ_j) = 1 / (1 + α·λ_j)  for each eigenvalue
    f_alpha = 1.0 / (1.0 + alpha * evals)  # (d_model,)

    # P_α = Q @ diag(f_α) @ Q^T
    # Efficient: (Q * f_alpha[None, :]) @ Q^T
    P_alpha = (Q * f_alpha.unsqueeze(0)) @ Q.T  # (d_model, d_model)

    return P_alpha


@torch.no_grad()
def apply_spectral_filter(w_down, Q, evals, mask, alpha):
    """Apply SRF soft spectral filter to down_proj weights.

    SRF paper Eq. 10: W_corr = P_α @ W_out
    where P_α = Q diag(1/(1+αλ_j)) Q^T is the soft suppression operator
    in d_model space (the row space of W_out / down_proj).

    For PMBT-guided mode, only columns corresponding to visual neurons
    are modified. Non-visual columns remain untouched.

    Args:
        w_down:  down_proj weight, shape (d_model, d_ffn)
        Q:       eigenvector matrix (d_model, d_model)
        evals:   eigenvalues (d_model,)
        mask:    boolean mask for visual neurons, shape (d_ffn,)
        alpha:   damping strength

    Returns:
        modified w_down (same shape)
    """
    if not mask.any() or Q is None:
        return w_down

    d_model, d_ffn = w_down.shape
    w = w_down.float()

    # Build P_α in d_model space
    Q_f = Q.float()
    evals_f = evals.float()
    P_alpha = build_suppression_operator(Q_f, evals_f, alpha)  # (d_model, d_model)

    # Apply P_α only to visual neuron columns:
    #   W_corr[:, j] = P_α @ W[:, j]  for j in visual mask
    visual_indices = torch.where(mask)[0]
    if len(visual_indices) == 0:
        return w_down

    # Extract visual columns, left-multiply by P_α, write back
    # w[:, visual] shape: (d_model, n_visual)
    # P_alpha @ w[:, visual] shape: (d_model, n_visual)
    w[:, visual_indices] = P_alpha @ w[:, visual_indices]

    return w.to(w_down.dtype)


def run_edit_stage(args):
    """Stage 2: Apply spectral filter at PMBT visual neuron positions."""
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load full hallucination eigenspectrum ────────────────────
    modes_dir = args.eigenvecs_dir or args.output_dir
    pt_path = os.path.join(modes_dir, 'hallucination_modes.pt')
    if os.path.isfile(pt_path):
        print(f'  Loading eigenspectrum from {pt_path}')
        data = torch.load(pt_path, map_location='cpu')
        # Support both old (modes) and new (eigenvectors) format
        if 'eigenvectors' in data:
            eigenvectors = data['eigenvectors']   # {layer: (d_model, d_model)}
            eigenvalues = data['eigenvalues']     # {layer: (d_model,)}
        else:
            raise ValueError(
                f'Old-format hallucination_modes.pt found. '
                f'Please re-run --stage profile to generate full eigenspectrum.')
    else:
        raise FileNotFoundError(
            f'No hallucination_modes.pt found in {modes_dir}. '
            f'Run --stage profile first.')

    print(f'  Loaded eigenspectrum for {len(eigenvectors)} layers')

    # ── Auto-compute α from dominant eigenvalue (SRF paper Eq. 12) ──
    # α = (1 - η) / (η · λ₁)  where η = retained variance fraction
    # Use the largest λ₁ across all layers for a global α
    eta = args.eta
    if args.alpha_override is not None:
        alpha = args.alpha_override
        print(f'  α = {alpha:.2f} (manual override)')
    else:
        # Find the largest dominant eigenvalue across all layers
        max_lambda1 = max(eigenvalues[l][0].item() for l in eigenvalues)
        alpha = (1.0 - eta) / (eta * max_lambda1 + 1e-12)
        print(f'  Auto-computed α = {alpha:.2f} '
              f'(η={eta}, max λ₁={max_lambda1:.4f})')

    # ── Load PMBT labels or use all-neurons mask ──────────────────
    if args.all_neurons:
        # Vanilla SRF: apply to ALL down_proj columns (no PMBT filtering)
        print(f'  [all_neurons] Vanilla SRF: applying to ALL neurons')
        # Determine d_ffn from the first down_proj weight we can find
        sample_key = None
        for sk in vlm_state:
            if ('down_proj' in sk or 'w2' in sk) and 'vision' not in sk:
                sample_key = sk
                break
        if sample_key is None:
            raise ValueError('Cannot find down_proj weight to determine d_ffn')
        d_ffn = vlm_state[sample_key].shape[1]
        # n_layers from eigenspectrum
        n_layers_edit = max(eigenvectors.keys()) + 1
        visual_masks = {l: torch.ones(d_ffn, dtype=torch.bool)
                        for l in range(n_layers_edit)}
        n_layers = n_layers_edit
        n_neurons = d_ffn
        print(f'  Full mask: {n_layers} layers × {d_ffn} neurons (all)')
    else:
        if args.label_dir is None:
            raise ValueError('--label_dir is required for edit stage '
                             '(or use --all_neurons for vanilla SRF)')
        labels_by_layer, n_layers, n_neurons = load_pmbt_labels(args.label_dir)
        if args.n_layers is not None:
            n_layers = args.n_layers
        visual_masks = build_visual_mask(labels_by_layer, n_layers, n_neurons)

    # ── Determine which layers to edit (deeper layers only) ──────
    min_layer = int(n_layers * args.min_layer_pct)
    edit_layers = [l for l in range(min_layer, n_layers) if l in eigenvectors]
    print(f'  Editing layers {min_layer}–{n_layers - 1} '
          f'({len(edit_layers)} layers with eigenspectrum)')

    # ── Load VLM state dict ──────────────────────────────────────
    print(f'\n  Loading VLM state dict: {args.vlm_path}')
    if args.model_type == 'llava-liuhaotian':
        # LLaVA-1.5 original repo loader
        try:
            from llava.model.builder import load_pretrained_model
            _tok, vlm, _proc, _ = load_pretrained_model(
                args.vlm_path, None, 'llava-v1.5-7b',
                device_map='cpu', torch_dtype=torch.float16)
        except ImportError:
            from transformers import AutoModelForCausalLM
            vlm = AutoModelForCausalLM.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16,
                device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True)
    else:
        from transformers import AutoModel
        vlm = AutoModel.from_pretrained(
            args.vlm_path, torch_dtype=torch.float16,
            device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True)
    vlm_state = vlm.state_dict()
    del vlm
    torch.cuda.empty_cache()

    # ── Apply spectral filter to down_proj at visual neuron positions ─
    n_edited = 0
    print(f'\n  Applying SRF spectral filter (α={alpha:.2f}, η={eta})...')

    for layer_idx in tqdm(edit_layers, desc='  Editing layers'):
        mask = visual_masks.get(layer_idx)
        if mask is None or not mask.any():
            continue

        Q = eigenvectors.get(layer_idx)    # (d_model, d_model)
        evals = eigenvalues.get(layer_idx)  # (d_model,)
        if Q is None or evals is None:
            continue

        # Get down_proj weight key
        _, _, d_key = get_mlp_weight_paths(args.model_type, layer_idx)
        if d_key not in vlm_state:
            # Try suffix matching (skip vision encoder keys)
            suffix = d_key.split('.')[-2] + '.' + d_key.split('.')[-1]
            found = False
            for sk in vlm_state:
                if 'vision' in sk or 'encoder' in sk or 'visual' in sk:
                    continue
                if sk.endswith(suffix) and f'.{layer_idx}.' in sk:
                    d_key = sk
                    found = True
                    break
            if not found:
                continue

        w_down = vlm_state[d_key]  # (d_model, d_ffn)

        # Apply spectral filter: W_corr[:, visual] = (P_α @ W)[:, visual]
        w_down_edited = apply_spectral_filter(
            w_down, Q, evals, mask, alpha)
        vlm_state[d_key] = w_down_edited

        n_visual = mask.sum().item()
        n_edited += n_visual

    print(f'  Edited {n_edited:,} visual neuron positions '
          f'across {len(edit_layers)} layers')

    # ── Save edited model ────────────────────────────────────────
    if args.save_model:
        edit_tag = f'srf_a{alpha:.1f}_eta{eta}'
        save_dir = os.path.join(args.output_dir, edit_tag)
        os.makedirs(save_dir, exist_ok=True)
        print(f'  Saving edited state dict → {save_dir}')

        torch.save(vlm_state, os.path.join(save_dir, 'pytorch_model.bin'))

        # copy config/tokenizer
        try:
            from transformers import AutoConfig, AutoTokenizer
            config = AutoConfig.from_pretrained(
                args.vlm_path, trust_remote_code=True)
            config.save_pretrained(save_dir)
        except Exception as e:
            print(f'  WARNING: Could not save config: {e}')
            config = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.vlm_path, trust_remote_code=True)
            tokenizer.save_pretrained(save_dir)
        except Exception:
            # LLaVA-1.5: use original repo tokenizer
            if args.model_type == 'llava-liuhaotian':
                try:
                    from transformers import LlamaTokenizer
                    tokenizer = LlamaTokenizer.from_pretrained(args.vlm_path)
                    tokenizer.save_pretrained(save_dir)
                except Exception as e2:
                    print(f'  WARNING: Could not save tokenizer: {e2}')
            else:
                print(f'  WARNING: Could not save tokenizer')

        # save metadata
        meta = {
            'method': 'srf_pmbt',
            'alpha': alpha,
            'eta': eta,
            'min_layer_pct': args.min_layer_pct,
            'min_layer': min_layer,
            'n_edit_layers': len(edit_layers),
            'n_visual_neurons_edited': n_edited,
            'vlm_path': args.vlm_path,
        }
        with open(os.path.join(save_dir, 'edit_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'  ✓ Edited model saved to {save_dir}')

    # ── Optional: random baseline (same neuron count, random positions) ──
    if args.include_random:
        import random as _rng
        _rng.seed(args.random_seed)
        n_visual_total = sum(
            visual_masks[l].sum().item() for l in edit_layers
            if l in visual_masks)
        print(f'\n  ── Random baseline (count-matched sparsity control) ──')
        random_masks = {l: torch.zeros(n_neurons, dtype=torch.bool)
                        for l in range(n_layers)}
        # Only randomize within edit_layers (same depth restriction as SRF)
        all_positions = [(l, n) for l in edit_layers for n in range(n_neurons)]
        selected = _rng.sample(all_positions, min(n_visual_total, len(all_positions)))
        for l, n in selected:
            random_masks[l][n] = True
        n_rnd = sum(m.sum().item() for m in random_masks.values())
        print(f'  Random mask: {n_rnd:,} neurons (matched to visual count {n_visual_total:,} in edit layers)')

        # Reload fresh VLM state
        print(f'  Reloading fresh VLM state dict...')
        if args.model_type == 'llava-liuhaotian':
            try:
                from llava.model.builder import load_pretrained_model
                _tok, vlm_reload, _proc, _ = load_pretrained_model(
                    args.vlm_path, None, 'llava-v1.5-7b',
                    device_map='cpu', torch_dtype=torch.float16)
            except ImportError:
                from transformers import AutoModelForCausalLM
                vlm_reload = AutoModelForCausalLM.from_pretrained(
                    args.vlm_path, torch_dtype=torch.float16,
                    device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True)
        else:
            vlm_reload = AutoModel.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16,
                device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True)
        vlm_state_rnd = vlm_reload.state_dict()
        del vlm_reload
        torch.cuda.empty_cache()

        n_edited_rnd = 0
        print(f'  Applying spectral filter with random mask...')
        for layer_idx in tqdm(edit_layers, desc='  Random edit'):
            mask = random_masks.get(layer_idx)
            if mask is None or not mask.any():
                continue
            Q = eigenvectors.get(layer_idx)
            evals = eigenvalues.get(layer_idx)
            if Q is None or evals is None:
                continue
            _, _, d_key = get_mlp_weight_paths(args.model_type, layer_idx)
            w_down = vlm_state_rnd.get(d_key)
            if w_down is None:
                continue
            w_down_edited = apply_spectral_filter(
                w_down, Q, evals, mask, alpha)
            vlm_state_rnd[d_key] = w_down_edited
            n_edited_rnd += mask.sum().item()

        if args.save_model:
            rnd_tag = f'srf_random_a{alpha:.1f}_eta{eta}'
            rnd_dir = os.path.join(args.output_dir, rnd_tag)
            os.makedirs(rnd_dir, exist_ok=True)
            torch.save(vlm_state_rnd, os.path.join(rnd_dir, 'pytorch_model.bin'))
            config.save_pretrained(rnd_dir)
            tokenizer.save_pretrained(rnd_dir)
            rnd_meta = {**meta, 'method': 'srf_random', 'n_visual_neurons_edited': n_edited_rnd}
            with open(os.path.join(rnd_dir, 'edit_meta.json'), 'w') as f:
                json.dump(rnd_meta, f, indent=2)
            print(f'  ✓ Random baseline saved to {rnd_dir}')
        del vlm_state_rnd

        return save_dir

    return save_dir if args.save_model else None


# ═══════════════════════════════════════════════════════════════════
# Section 6a — Evaluation stage (POPE + TriviaQA)
# ═══════════════════════════════════════════════════════════════════

def _load_edited_model(args, edited_state_path):
    """Load original model and replace weights with edited state dict."""
    print(f'  Loading base model: {args.vlm_path}')
    model, tokenizer, processor = _load_vlm(args.vlm_path, args.model_type)

    print(f'  Loading edited weights from: {edited_state_path}')
    edited_state = torch.load(
        os.path.join(edited_state_path, 'pytorch_model.bin'),
        map_location='cpu')

    # Replace matching keys in the model's state dict
    model_state = model.state_dict()
    n_replaced = 0
    for key in edited_state:
        if key in model_state:
            model_state[key] = edited_state[key].to(model_state[key].dtype)
            n_replaced += 1
    model.load_state_dict(model_state)
    print(f'  Replaced {n_replaced} weight tensors')
    del edited_state
    torch.cuda.empty_cache()

    return model, tokenizer, processor


def _generate_answer(model, tokenizer, processor, image, question,
                     model_type, max_new_tokens=64):
    """Generate a short answer from the model given image + question."""

    if model_type == 'llava-liuhaotian':
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        prompt = f'USER: {DEFAULT_IMAGE_TOKEN}\n{question}\nASSISTANT:'
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to(model.device)
        img_tensor = processor.preprocess(
            image, return_tensors='pt')['pixel_values'].half().to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids, images=img_tensor,
                max_new_tokens=max_new_tokens, do_sample=False)
        output_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    elif processor is not None:
        try:
            msgs = [{'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question}]}]
            formatted = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=formatted, images=image, return_tensors='pt')
        except Exception:
            formatted = f'USER: <image>\n{question}\nASSISTANT:'
            inputs = processor(
                text=formatted, images=image, return_tensors='pt')
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        output_text = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True)
    else:
        inputs = tokenizer(question, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output_text = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)

    return output_text.strip()


def eval_pope(model, tokenizer, processor, pope_dir, img_dir,
              model_type, splits, max_samples):
    """Evaluate on POPE benchmark (yes/no object probing).

    Returns dict of {split: {accuracy, precision, recall, f1}}.
    """
    from PIL import Image
    results = {}

    for split in splits:
        pope_file = os.path.join(pope_dir, f'coco_pope_{split}.json')
        if not os.path.isfile(pope_file):
            print(f'    POPE {split}: file not found, skipping')
            continue

        with open(pope_file) as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]

        # Limit by unique images
        if max_samples and len(data) > max_samples * 10:
            seen_imgs = set()
            filtered = []
            for item in data:
                img = item.get('image', '')
                if img not in seen_imgs:
                    seen_imgs.add(img)
                if len(seen_imgs) <= max_samples:
                    filtered.append(item)
            data = filtered

        tp = fp = tn = fn = 0
        n_total = 0

        for item in tqdm(data, desc=f'    POPE-{split}'):
            question = item.get('text', item.get('question', ''))
            img_name = item.get('image', '')
            gt = item.get('label', item.get('answer', '')).lower().strip()

            img_path = os.path.join(img_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            try:
                image = Image.open(img_path).convert('RGB')
                answer = _generate_answer(
                    model, tokenizer, processor, image, question,
                    model_type, max_new_tokens=10)
                pred = 'yes' if 'yes' in answer.lower() else 'no'
            except Exception:
                continue

            n_total += 1
            if pred == 'yes' and gt == 'yes':
                tp += 1
            elif pred == 'yes' and gt == 'no':
                fp += 1
            elif pred == 'no' and gt == 'no':
                tn += 1
            elif pred == 'no' and gt == 'yes':
                fn += 1

        acc = (tp + tn) / max(n_total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)

        results[split] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2),
            'n_total': n_total,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        }
        print(f'    {split}: Acc={acc*100:.2f}  Prec={prec*100:.2f}  '
              f'F1={f1*100:.2f}  (n={n_total})')

    return results


def eval_triviaqa(model, tokenizer, processor, triviaqa_path,
                  model_type, max_samples):
    """Evaluate on TriviaQA (text-only, no images).

    Returns dict with accuracy and error rate.
    """
    print(f'    Loading TriviaQA from {triviaqa_path}')

    if triviaqa_path.endswith('.jsonl'):
        with open(triviaqa_path) as f:
            data = [json.loads(line) for line in f if line.strip()]
    elif triviaqa_path.endswith('.json'):
        with open(triviaqa_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
    elif triviaqa_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(triviaqa_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f'Unsupported TriviaQA format: {triviaqa_path}')

    if max_samples and len(data) > max_samples:
        data = data[:max_samples]

    correct = 0
    total = 0

    for item in tqdm(data, desc='    TriviaQA'):
        question = item.get('question', item.get('Question', ''))
        answer_field = item.get('answer', item.get('Answer', {}))
        if isinstance(answer_field, dict):
            aliases = answer_field.get('aliases', [])
            main_ans = answer_field.get('value', '')
            accepted = [main_ans.lower()] + [a.lower() for a in aliases]
        elif isinstance(answer_field, str):
            accepted = [answer_field.lower()]
        elif isinstance(answer_field, list):
            accepted = [a.lower() for a in answer_field]
        else:
            continue

        accepted = [a for a in accepted if a]
        if not accepted or not question:
            continue

        try:
            if model_type == 'llava-liuhaotian':
                prompt = f'USER: {question}\nASSISTANT:'
                input_ids = tokenizer(
                    prompt, return_tensors='pt').input_ids.to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        inputs=input_ids,
                        max_new_tokens=32, do_sample=False)
                pred = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True).strip()
            else:
                inputs = tokenizer(question, return_tensors='pt')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=32, do_sample=False)
                pred = tokenizer.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True).strip()
        except Exception:
            continue

        total += 1
        pred_lower = pred.lower()
        if any(ans in pred_lower for ans in accepted):
            correct += 1

    acc = correct / max(total, 1)
    err = 1.0 - acc
    results = {
        'accuracy': round(acc * 100, 2),
        'error_rate': round(err * 100, 2),
        'correct': correct,
        'total': total,
    }
    print(f'    TriviaQA: Acc={acc*100:.2f}  Err={err*100:.2f}  '
          f'({correct}/{total})')
    return results


def run_eval_stage(args):
    """Stage eval: Evaluate edited model on POPE + TriviaQA."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Find edited model directory
    edited_dir = args.edited_model_dir
    if edited_dir is None:
        for entry in sorted(os.listdir(args.output_dir)):
            candidate = os.path.join(args.output_dir, entry)
            if (os.path.isdir(candidate) and entry.startswith('srf_') and
                    os.path.isfile(os.path.join(candidate, 'pytorch_model.bin'))):
                edited_dir = candidate
                break
    if edited_dir is None or not os.path.isfile(
            os.path.join(edited_dir, 'pytorch_model.bin')):
        raise FileNotFoundError(
            f'No edited model found. Specify --edited_model_dir or run '
            f'--stage edit first.')

    print(f'\n  ── Evaluation Stage ──')
    print(f'  Edited model: {edited_dir}')

    model, tokenizer, processor = _load_edited_model(args, edited_dir)
    model.eval()

    all_results = {'edited_model': edited_dir, 'vlm_path': args.vlm_path}

    # ── POPE evaluation ──
    if args.pope_path and os.path.isdir(args.pope_path):
        print(f'\n  Evaluating POPE...')
        splits = [s.strip() for s in args.eval_pope_splits.split(',')]
        pope_results = eval_pope(
            model, tokenizer, processor, args.pope_path,
            args.pope_img_dir, args.model_type, splits, args.eval_max_samples)
        all_results['pope'] = pope_results
    else:
        print(f'  Skipping POPE (no pope_path directory)')

    # ── TriviaQA evaluation ──
    if args.triviaqa_path and os.path.isfile(args.triviaqa_path):
        print(f'\n  Evaluating TriviaQA...')
        tqa_results = eval_triviaqa(
            model, tokenizer, processor, args.triviaqa_path,
            args.model_type, args.eval_max_tqa)
        all_results['triviaqa'] = tqa_results
    else:
        print(f'  Skipping TriviaQA (no triviaqa_path)')

    # ── Save results ──
    meta_path = os.path.join(edited_dir, 'edit_meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            all_results['edit_meta'] = json.load(f)

    eval_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(eval_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Results saved to {eval_path}')

    print(f'\n  ── Summary ──')
    if 'pope' in all_results:
        for split, r in all_results['pope'].items():
            print(f'  POPE {split:12s}: Acc={r["accuracy"]:.2f}  '
                  f'F1={r["f1"]:.2f}')
    if 'triviaqa' in all_results:
        r = all_results['triviaqa']
        print(f'  TriviaQA:        Acc={r["accuracy"]:.2f}  '
              f'Err={r["error_rate"]:.2f}')

    del model
    torch.cuda.empty_cache()
    return eval_path


# ═══════════════════════════════════════════════════════════════════
# Section 6b — VLM loading helper
# ═══════════════════════════════════════════════════════════════════

def _load_vlm(path, model_type):
    """Load VLM with appropriate class based on model_type.

    Returns:
        (model, tokenizer, processor) tuple.
        For most models, processor is an AutoProcessor.
        For llava-liuhaotian, processor is the image_processor from the
        original LLaVA repo, and tokenizer is the LLaMA tokenizer.
    """
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    # ── LLaVA-1.5 (liuhaotian): use original LLaVA repo loader ──
    if model_type == 'llava-liuhaotian':
        try:
            from llava.model.builder import load_pretrained_model
            tokenizer, model, image_processor, _ = load_pretrained_model(
                path, None, 'llava-v1.5-7b',
                device_map='auto', torch_dtype=torch.float16)
            # Pack (tokenizer, image_processor) as a pseudo-processor
            # so the rest of the code can use processor.apply_chat_template etc.
            # But LLaVA-1.5 doesn't use chat templates — we handle it in
            # the caption input preparation with the explicit format string.
            model.eval()
            return model, tokenizer, image_processor
        except ImportError:
            print('  WARNING: llava package not found, falling back to HF loader')
            # Fall through to generic HF loading below

    # ── Other LLaVA variants (OV, LLaMA3, HF) ──
    if model_type in ('llava-ov', 'llava-hf', 'llava-llama3'):
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception:
            pass
        tokenizer = processor.tokenizer if processor else AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)
        try:
            from transformers import LlavaOnevisionForConditionalGeneration
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.float16, device_map='auto',
                trust_remote_code=True)
        except Exception:
            try:
                from transformers import LlavaNextForConditionalGeneration
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    path, torch_dtype=torch.float16, device_map='auto',
                    trust_remote_code=True)
            except Exception:
                from transformers import LlavaForConditionalGeneration
                model = LlavaForConditionalGeneration.from_pretrained(
                    path, torch_dtype=torch.float16, device_map='auto',
                    trust_remote_code=True)
        model.eval()
        return model, tokenizer, processor

    elif model_type == 'qwen2vl':
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True)
        model.eval()
        return model, tokenizer, processor

    elif model_type == 'internvl':
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True)
        model.eval()
        return model, tokenizer, processor

    else:
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True)
        model.eval()
        return model, tokenizer, processor


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()

    print('\n' + '═' * 60)
    print('  PMBT-Guided SRF: Spectral Representation Filtering (Layer 1c)')
    print('═' * 60)
    print(f'  Stage:     {args.stage}')
    print(f'  VLM:       {args.vlm_path}')
    print(f'  Output:    {args.output_dir}')

    if args.stage in ('profile', 'both'):
        pt_path = run_profile_stage(args)
        if args.stage == 'both':
            args.eigenvecs_dir = args.output_dir

    if args.stage in ('edit', 'both'):
        run_edit_stage(args)

    if args.stage == 'eval':
        run_eval_stage(args)

    elapsed = time.time() - t0
    print(f'\n  Total time: {elapsed / 60:.1f} min')


if __name__ == '__main__':
    main()