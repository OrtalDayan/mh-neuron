"""
patch_fig3_activations.py — Fill missing Activation Pattern activations for Figure 3 neurons.

Lightweight GPU script that runs teacher-forcing forward passes on only the
6 Figure 3 images for their specific neurons/layers.  This avoids re-running
the full 23K-image pipeline when only the fig3 layers are missing.

Each model has its own neuron table.  LLaVA neurons are from Xu et al.;
InternVL neurons are TBD (placeholder stubs — update once identified).

Usage:
    python patch_fig3_activations.py \
        --data_dir results/full/llava-1.5-7b/llm_fixed_threshold \
        --coco_img_dir /path/to/coco/train2017 \
        --generated_desc_path generated_descriptions.json \
        --model_type llava-hf \
        --device 0

Called by run_pipeline.sh in the visualize step (--viz-fig3).
"""

import argparse                                                            # CLI argument parsing
import json                                                                # JSON I/O for descriptions, sampled_ids
import os                                                                  # file path operations
import sys                                                                 # sys.exit for early termination
import time                                                                # timing the forward passes

import numpy as np                                                         # array ops for Activation Pattern storage
import torch                                                               # PyTorch — model inference + tensor ops
from baukit import TraceDict                                               # hooks for capturing layer activations
from PIL import Image                                                      # loading COCO images from disk


# ═══════════════════════════════════════════════════════════════════
# Per-model Figure 3 neuron tables
# ═══════════════════════════════════════════════════════════════════
#
# Each entry: panel label, layer, neuron_idx, coco_image_id, neuron_type.
# Panels (e) and (f) share the SAME neuron shown on different images.

FIG3_NEURONS_LLAVA = [
    {
        'panel': '(a)',
        'layer': 27,
        'neuron_idx': 3900,
        'coco_id': '000000403170',
        'label': 'visual',
        'description': 'Visual neuron — airplane/motorcycles',
    },
    {
        'panel': '(b)',
        'layer': 2,
        'neuron_idx': 4450,
        'coco_id': '000000065793',
        'label': 'text',
        'description': 'Text neuron — teddy bears/stuffed animals',
    },
    {
        'panel': '(c)',
        'layer': 29,
        'neuron_idx': 600,
        'coco_id': '000000156852',
        'label': 'multimodal',
        'description': 'Multi-modal neuron — kitchen/thumbs up/tie',
    },
    {
        'panel': '(d)',
        'layer': 31,
        'neuron_idx': 1800,
        'coco_id': '000000323964',
        'label': 'multimodal',
        'description': 'Multi-modal neuron — doughnuts',
    },
    {
        'panel': '(e)',
        'layer': 21,
        'neuron_idx': 6100,
        'coco_id': '000000276332',
        'label': 'multimodal',
        'description': 'Multi-modal neuron — zebras (same neuron as f)',
    },
    {
        'panel': '(f)',
        'layer': 21,
        'neuron_idx': 6100,
        'coco_id': '000000060034',
        'label': 'multimodal',
        'description': 'Multi-modal neuron — fire hydrant/pigeons (same neuron as e)',
    },
]

# ─── 4-panel version for 28-layer models (skip layers 29, 31) ────────
FIG3_NEURONS_28L = [n for n in FIG3_NEURONS_LLAVA if n['layer'] < 28]

# ─── Proportional layer mapping for models with != 32 layers ─────────
# Xu's Figure 3 uses layers [2, 21, 27, 29, 31] from a 32-layer model.
# Layers 2, 21, 27 exist in all models (≥28 layers). Layers 29, 31 only
# exist in ≥32-layer models. For shorter/longer models, we map layers
# proportionally: layer_new = round(layer_xu / 31 * (n_layers - 1))
_XU_N_LAYERS = 32  # reference model layer count

def _map_layer(xu_layer, n_layers):
    """Map a layer index from 32-layer model to n_layers model.

    Only remaps if the original layer doesn't exist in the target model.
    Models with >= 32 layers keep the original Xu indices.
    """
    if xu_layer < n_layers:
        return xu_layer  # layer exists — keep original
    # Layer doesn't exist — map proportionally
    return min(round(xu_layer / (_XU_N_LAYERS - 1) * (n_layers - 1)), n_layers - 1)

_MODEL_N_LAYERS = {
    'llava-hf': 32, 'llava-liuhaotian': 32, 'llava-llama3': 32,
    'internvl': 32, 'idefics2': 32,
    'qwen25vl-3b': 36,
    'llava-ov': 28, 'qwen2vl': 28, 'qwen2vl-brv': 28, 'qwen25vl-7b': 28,
}

# ─── Attention head Figure 3: proportional head index ────────────────
# Xu's FFN neuron indices were chosen from LLaVA-1.5-7B (11008 neurons).
# For attention heads, we pick the head at the same relative position:
#   head_idx = round(ffn_neuron_idx / 11008 * n_heads)
_XU_FFN_DIM = 11008

_N_HEADS_MAP = {
    'llava-hf': 32, 'llava-liuhaotian': 32, 'llava-llama3': 32,
    'internvl': 32, 'idefics2': 32,
    'qwen25vl-3b': 16,
    'llava-ov': 28, 'qwen2vl': 28, 'qwen2vl-brv': 28, 'qwen25vl-7b': 28,
}


def _get_fig3_neurons(model_type, hook_point='gate'):
    """Return the Figure 3 neuron table for a given model and hook point.

    All models get 6 panels. Layers and neuron indices are mapped
    proportionally when the model has different depth or head count.
    """
    n_layers = _MODEL_N_LAYERS.get(model_type, 32)
    n_heads = _N_HEADS_MAP.get(model_type, 32)

    entries = []
    for n in FIG3_NEURONS_LLAVA:
        # Map layer proportionally
        mapped_layer = _map_layer(n['layer'], n_layers)

        # Map neuron index: FFN stays as-is, attn uses proportional head
        if hook_point == 'attn':
            ratio = n['neuron_idx'] / _XU_FFN_DIM
            mapped_idx = min(round(ratio * n_heads), n_heads - 1)
            desc = n['description'].replace('neuron', 'head')
        else:
            mapped_idx = n['neuron_idx']
            desc = n['description']

        entries.append({
            **n,
            'layer': mapped_layer,
            'neuron_idx': mapped_idx,
            'description': desc,
        })
    return entries


# ─── Registry: model_type → neuron table (gate_up only, legacy) ──────
FIG3_NEURONS_BY_MODEL = {
    'llava-hf':          FIG3_NEURONS_LLAVA,     # LLaVA-1.5-7B (32 layers, 11008 neurons)
    'llava-liuhaotian':  FIG3_NEURONS_LLAVA,     # LLaVA-1.5-7B original
    'llava-llama3':      FIG3_NEURONS_LLAVA,     # LLaVA-Next-LLaMA3-8B (32 layers, 14336)
    'internvl':          FIG3_NEURONS_LLAVA,     # InternVL2.5-8B (32 layers, 14336)
    'idefics2':          FIG3_NEURONS_LLAVA,     # Idefics2-8B (32 layers, 14336)
    'qwen25vl-3b':       FIG3_NEURONS_LLAVA,     # Qwen2.5-VL-3B (36 layers, 11008)
    'llava-ov':          FIG3_NEURONS_28L,       # LLaVA-OneVision-7B (28 layers — no panels c,d)
    'qwen2vl':           FIG3_NEURONS_28L,       # Qwen2-VL-7B (28 layers — no panels c,d)
    'qwen2vl-brv':       FIG3_NEURONS_28L,       # Qwen2.5-VL-7B (28 layers — no panels c,d)
    'qwen25vl-7b':       FIG3_NEURONS_28L,       # Qwen2.5-VL-7B (28 layers — no panels c,d)
}


# ═══════════════════════════════════════════════════════════════════
# Helper: layer names (copied from neuron_modality_statistical.py)
# ═══════════════════════════════════════════════════════════════════

def get_layer_name(model_type, layer_idx, hook_point='gate'):
    """Return the baukit hook path for a specific layer and hook point.

    hook_point:
        'gate'    → mlp.act_fn output = SiLU(gate_proj(x))
        'gate_up' → mlp.down_proj input = SiLU(gate_proj(x)) * up_proj(x)
        'attn'    → self_attn.o_proj input = softmax(QK^T/√d)V
    """
    # Determine prefix per model_type
    if model_type in ('llava-hf', 'llava-llama3'):
        prefix = f'model.language_model.layers.{layer_idx}'
    elif model_type == 'internvl':
        prefix = f'language_model.model.layers.{layer_idx}'
    elif model_type in ('qwen2vl', 'llava-ov'):
        prefix = f'model.language_model.layers.{layer_idx}'
    elif model_type == 'idefics2':
        prefix = f'model.text_model.layers.{layer_idx}'
    else:  # llava-liuhaotian
        prefix = f'model.layers.{layer_idx}'

    # Determine suffix per hook_point
    if hook_point == 'attn':
        suffix = 'attention.wo' if model_type == 'internvl' else 'self_attn.o_proj'
    elif hook_point == 'gate_up':
        suffix = 'feed_forward.w2' if model_type == 'internvl' else 'mlp.down_proj'
    else:  # gate
        suffix = 'feed_forward.act_fn' if model_type == 'internvl' else 'mlp.act_fn'

    return f'{prefix}.{suffix}'


# ═══════════════════════════════════════════════════════════════════
# Model loading (thin wrappers — import from neuron_modality_statistical
# if available, otherwise inline)
# ═══════════════════════════════════════════════════════════════════

def load_model(model_type, model_path, device):
    """Load the model based on model_type.

    Line-by-line:
        - Imports the appropriate loading function
        - 'llava-hf': loads HF LLaVA via AutoProcessor + LlavaForConditionalGeneration
        - 'llava-liuhaotian': loads original LLaVA via the cloned repo's builder
        - 'internvl': loads InternVL2.5-8B with monkey-patches for torch.linspace
                      and all_tied_weights_keys
        - Returns (model, processor_or_tokenizer, image_token_id) in all cases
    """
    try:
        # Try importing from the classify script (same code/ directory)
        from neuron_modality_statistical import (
            load_model_hf, load_model_original, load_model_internvl
        )
        if model_type in ('llava-hf', 'llava-llama3'):
            return load_model_hf(model_path, device)
        elif model_type == 'internvl':
            return load_model_internvl(model_path, device)
        elif model_type == 'llava-ov':
            from neuron_modality_statistical import load_model_llava_ov
            return load_model_llava_ov(model_path, device)
        elif model_type == 'qwen2vl':
            from neuron_modality_statistical import load_model_qwen2vl
            return load_model_qwen2vl(model_path, device)
        elif model_type == 'idefics2':
            from neuron_modality_statistical import load_model_idefics2
            return load_model_idefics2(model_path, device)
        else:
            return load_model_original(model_path, device)
    except ImportError:
        # Fallback: inline HF loading (most common case)
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device).eval()
        image_token_id = model.config.image_token_index
        return model, processor, image_token_id


def prepare_inputs(model_type, processor, model, img, text, device,
                   image_token_id):
    """Prepare teacher-forcing inputs based on model_type.

    Line-by-line:
        - 'llava-hf': uses HF AutoProcessor to tokenize text+image together,
                 returns (inputs_dict, visual_mask)
        - 'llava-liuhaotian': uses LLaVA repo's tokenizer_image_token,
                         returns (inputs_dict, visual_mask)
        - 'internvl': uses prepare_inputs_internvl with dynamic tiling,
                       returns (inputs_dict, visual_mask, n_vis_tokens)
        - For hf/liuhaotian, we append a dummy n_vis=576 to unify the interface
    """
    try:
        from neuron_modality_statistical import (
            prepare_inputs_hf, prepare_inputs_original, prepare_inputs_internvl
        )
        if model_type in ('llava-hf', 'llava-llama3'):
            inputs, vis_mask = prepare_inputs_hf(
                processor, img, text, device, image_token_id)
            return inputs, vis_mask, 576                                   # LLaVA fixed 576 visual tokens
        elif model_type == 'internvl':
            return prepare_inputs_internvl(
                processor, model, img, text, device, image_token_id)
        elif model_type == 'llava-ov':
            from neuron_modality_statistical import prepare_inputs_llava_ov
            return prepare_inputs_llava_ov(
                processor, img, text, device, image_token_id)
        elif model_type == 'qwen2vl':
            from neuron_modality_statistical import prepare_inputs_qwen2vl
            return prepare_inputs_qwen2vl(
                processor, img, text, device, image_token_id)
        elif model_type == 'idefics2':
            from neuron_modality_statistical import prepare_inputs_idefics2
            return prepare_inputs_idefics2(
                processor, img, text, device, image_token_id)
        else:
            inputs, vis_mask = prepare_inputs_original(
                processor, img, text, device, image_token_id)
            return inputs, vis_mask, 576
    except ImportError:
        # Fallback: inline HF preparation
        inputs = processor(images=img, text=text,
                           return_tensors='pt').to(device)
        ids_cpu = inputs['input_ids'][0].cpu().numpy()
        vis_mask = (ids_cpu == image_token_id)
        return inputs, vis_mask, 576


# ═══════════════════════════════════════════════════════════════════
# Build teacher-forcing prompt
# ═══════════════════════════════════════════════════════════════════

def build_prompt_text(model_type, description):
    """Build the full teacher-forcing text prompt.

    Line-by-line:
        - 'llava-hf' / 'llava-liuhaotian': USER: <image>\\nCould you describe the image?\\nASSISTANT: {desc}
        - 'internvl': description only (IMG_CONTEXT template built in prepare_inputs_internvl)
    """
    if model_type in ('internvl', 'qwen2vl', 'llava-ov', 'idefics2'):
        return description                                                 # template built in prepare_inputs
    else:
        return (f'USER: <image>\nCould you describe the image?\n'
                f'ASSISTANT: {description}')


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments.

    Line-by-line:
        - --data_dir: path to the Top-N Heap/2 output directory (e.g. results/.../llm_fixed_threshold)
        - --coco_img_dir: path to COCO train2017 images
        - --generated_desc_path: JSON file with per-image generated descriptions
        - --model_type: which VLM backend (llava-hf, llava-liuhaotian, internvl)
        - --original_model_path: path to model weights (HF hub ID or local directory)
        - --device: GPU index
    """
    p = argparse.ArgumentParser(
        description='Patch Activation Pattern activations for Figure 3 neurons')

    p.add_argument('--data_dir', required=True,
                   help='Top-N Heap/2 data directory (e.g. .../llm_fixed_threshold)')
    p.add_argument('--coco_img_dir', required=True,
                   help='COCO train2017 image directory')
    p.add_argument('--generated_desc_path', required=True,
                   help='JSON file with generated descriptions')
    p.add_argument('--model_type', required=True,
                   choices=['llava-hf', 'llava-liuhaotian', 'llava-llama3',
                            'internvl', 'qwen2vl', 'qwen2vl-brv',
                            'qwen25vl-7b', 'qwen25vl-3b',
                            'llava-ov', 'idefics2'],
                   help='VLM backend type')
    p.add_argument('--original_model_path',
                   default='llava-hf/llava-1.5-7b-hf',
                   help='Model weights path (HF ID or local directory)')
    p.add_argument('--device', default='0',
                   help='GPU device index')
    p.add_argument('--hook_point', default='gate',
                   choices=['gate', 'gate_up', 'attn'],
                   help='Hook point: gate=SiLU(gate_proj(x)), '
                        'gate_up=gate*up, attn=o_proj input (attention heads)')
    p.add_argument('--output_dir', default=None,
                   help='Output directory for Figure 3 panels '
                        '(default: <data_dir>/../fig3_panels)')
    p.add_argument('--skip_viz', action='store_true',
                   help='Skip visualization — only patch activations')
    p.add_argument('--model_name', default='',
                   help='Human-readable model name passed to visualize '
                        'script for figure titles and filenames')
    p.add_argument('--fig3_json', default=None,
                   help='Path to fig3_neurons.json (output of '
                        'find_fig3_neurons.py). Auto-detects at '
                        '<data_dir>/../fig3_neurons.json if not set.')

    return p.parse_args()


def main():
    args = parse_args()
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device

    # ── Normalize model_type for loading/layer names ─────────────
    # Save original for FIG3_NEURONS_BY_MODEL lookup, then normalize
    # qwen25vl variants → qwen2vl (same architecture, same hooks)
    # llava-llama3 stays as-is (handled in load_model/prepare_inputs)
    _original_model_type = args.model_type
    if args.model_type in ('qwen25vl-7b', 'qwen25vl-3b'):
        args.model_type = 'qwen2vl'

    # ── Select the neuron table for this model ────────────────────
    # Priority: --fig3_json → auto-detect JSON → hardcoded list
    fig3_neurons = None

    # 1. Explicit --fig3_json path
    json_path = args.fig3_json

    # 2. Auto-detect at <data_dir>/../fig3_neurons.json
    #    Skip for attn hook — cached JSON has FFN indices, not head indices
    if json_path is None and args.hook_point != 'attn':
        auto_path = os.path.join(
            os.path.dirname(args.data_dir.rstrip('/')),
            'fig3_neurons.json')
        if os.path.isfile(auto_path):
            json_path = auto_path

    # 3. Load from JSON if found
    if json_path and os.path.isfile(json_path):
        print(f'  Loading fig3 neurons from {json_path}')
        with open(json_path) as f:
            fig3_neurons = json.load(f)
        print(f'  Loaded {len(fig3_neurons)} panel entries')

    # 4. Fallback to hardcoded list (hook-point aware)
    if not fig3_neurons:
        fig3_neurons = _get_fig3_neurons(_original_model_type, args.hook_point)

    if not fig3_neurons:
        print(f'[patch_fig3] No Figure 3 neurons defined for '
              f'model_type={args.model_type}. Skipping.')
        print('  Run find_fig3_neurons.py first:')
        print(f'    bash code/run_pipeline.sh --model-type '
              f'{args.model_type} --step find_fig3 --local')
        sys.exit(0)

    # ── Resolve paths ─────────────────────────────────────────────
    topn_heap_dir = os.path.join(args.data_dir, 'topn_heap')                     # Top-N Heap top_n_sids, global_max
    act_pattern_dir = os.path.join(args.data_dir, 'act_pattern_raw')                 # Activation Pattern raw_acts npz files
    os.makedirs(act_pattern_dir, exist_ok=True)                                 # create if missing (test mode)

    print(f'\n{"="*60}')
    print(f'PATCH FIGURE 3 ACTIVATIONS')
    print(f'{"="*60}')
    print(f'  Model type:    {args.model_type}')
    print(f'  Model path:    {args.original_model_path}')
    print(f'  Top-N Heap dir:   {topn_heap_dir}')
    print(f'  Activation Pattern dir:   {act_pattern_dir}')
    print(f'  COCO img dir:  {args.coco_img_dir}')
    print(f'  Neurons to patch: {len(fig3_neurons)}')

    # ── Check which layers actually need patching ─────────────────
    required_layers = sorted(set(e['layer'] for e in fig3_neurons))        # unique layers from the table
    layers_to_patch = []                                                   # layers where npz is missing

    for l in required_layers:
        npz_path = os.path.join(act_pattern_dir, f'raw_acts_layer{l}.npz')     # expected Activation Pattern file
        if os.path.isfile(npz_path):
            print(f'  Layer {l}: Activation Pattern data exists — will update in-place')
            layers_to_patch.append(l)                                      # still patch (neuron might be zero)
        else:
            print(f'  Layer {l}: Activation Pattern data MISSING — will create')
            layers_to_patch.append(l)

    if not layers_to_patch:
        print('\nAll layers already patched. Nothing to do.')
        return

    # ── Load descriptions ─────────────────────────────────────────
    print('\nLoading generated descriptions...')
    with open(args.generated_desc_path) as f:
        raw_desc = json.load(f)                                            # JSON format varies

    # Normalise: descriptions[image_id_str] = text_string
    descriptions = {}                                                      # unified lookup
    if isinstance(raw_desc, list):                                         # list-of-dicts format
        for entry in raw_desc:
            img_id = str(entry.get('id', entry.get('image_id', '')))
            descriptions[img_id] = entry.get('text', entry.get('description', ''))
    elif isinstance(raw_desc, dict):                                       # dict format
        for k, v in raw_desc.items():
            if isinstance(v, dict):
                descriptions[str(k)] = v.get('text', v.get('description', ''))
            else:
                descriptions[str(k)] = str(v)
    print(f'  {len(descriptions)} descriptions loaded')

    # ── Load sampled_ids (sample_idx → image_id mapping) ──────────
    sampled_ids_path = os.path.join(topn_heap_dir, 'sampled_ids.json')
    if os.path.isfile(sampled_ids_path):
        with open(sampled_ids_path) as f:
            sampled_ids = json.load(f)                                     # list: idx → image_id_str
        print(f'  {len(sampled_ids)} sampled images')
    else:
        sampled_ids = None                                                 # test mode: no sampled_ids
        print('  WARNING: sampled_ids.json not found (test mode?)')

    # ── Load model ────────────────────────────────────────────────
    print(f'\nLoading model ({args.model_type})...')
    t0 = time.time()
    model, processor, image_token_id = load_model(
        args.model_type, args.original_model_path, device)
    print(f'  Model loaded in {time.time() - t0:.1f}s')

    # ── Determine n_neurons via probe ─────────────────────────────
    _retain_input = (args.hook_point in ('gate_up', 'attn'))
    first_layer_name = get_layer_name(args.model_type, required_layers[0], args.hook_point)
    first_entry = fig3_neurons[0]

    # Quick probe: load first fig3 image, run forward, get neuron count
    probe_img = Image.open(
        os.path.join(args.coco_img_dir,
                     f'{first_entry["coco_id"]}.jpg')).convert('RGB')
    probe_desc = descriptions.get(first_entry['coco_id'], 'A photograph.')
    probe_text = build_prompt_text(args.model_type, probe_desc)

    probe_inputs, _, _ = prepare_inputs(
        args.model_type, processor, model, probe_img, probe_text,
        device, image_token_id)

    with torch.no_grad():
        with TraceDict(model, [first_layer_name], retain_input=_retain_input) as td:
            model(**probe_inputs)

    if _retain_input:
        out = td[first_layer_name].input[0]                                # gate_up/attn: input to down_proj/o_proj
    else:
        out = td[first_layer_name].output                                  # gate: act_fn output
    if isinstance(out, tuple):
        out = out[0]
    # Squeeze batch dim if present
    if out.dim() == 3:
        out = out[0]                                                       # (batch, seq, hidden) → (seq, hidden)

    if args.hook_point == 'attn':
        # Reshape to per-head and L2-norm: (seq, hidden) → n_heads
        from neuron_modality_statistical import _get_num_attention_heads
        _n_heads = _get_num_attention_heads(model, args.model_type)
        _head_dim = out.shape[-1] // _n_heads
        n_neurons = _n_heads
        print(f'  Attention heads per layer: {n_neurons} (head_dim={_head_dim})')
    else:
        n_neurons = out.shape[-1]                                          # 11008 for LLaVA, 14336 for InternVL
        _n_heads = None
        print(f'  Neurons per layer: {n_neurons}')

    # ── Determine array sizes ─────────────────────────────────────
    if args.model_type in ('internvl', 'qwen2vl', 'llava-ov', 'idefics2', 'llava-llama3'):
        N_VIS = 3072                                                       # dynamic visual tokens (up to 3072)
    else:
        N_VIS = 576                                                        # LLaVA-1.5 fixed 576 CLIP patches
    MAX_TXT = 300                                                          # max text token slots
    TOP_N = 50                                                             # default top-N from pipeline

    # ── Group fig3 entries by layer ───────────────────────────────
    from collections import defaultdict
    entries_by_layer = defaultdict(list)                                    # layer_idx → list of entries
    for entry in fig3_neurons:
        entries_by_layer[entry['layer']].append(entry)

    # ── Process each layer ────────────────────────────────────────
    t0 = time.time()

    for l in layers_to_patch:
        layer_name = get_layer_name(args.model_type, l, args.hook_point)    # baukit hook path
        npz_path = os.path.join(act_pattern_dir, f'raw_acts_layer{l}.npz')

        print(f'\n{"─"*60}')
        print(f'Layer {l}: {layer_name}')

        # Load existing Activation Pattern data or create empty arrays
        if os.path.isfile(npz_path):
            data = np.load(npz_path)                                       # load existing Activation Pattern
            vis_acts = data['vis_acts'].copy()                             # (n_neurons, top_n, N_VIS)
            txt_acts = data['txt_acts'].copy()                             # (n_neurons, top_n, MAX_TXT)
            txt_lengths = data['txt_lengths'].copy()                       # (n_neurons, top_n)
            # Handle vis_lengths (added for InternVL)
            if 'vis_lengths' in data:
                vis_lengths = data['vis_lengths'].copy()
            else:
                vis_lengths = np.full(vis_acts.shape[:2], N_VIS, dtype=np.int16)
            actual_top_n = vis_acts.shape[1]                               # might differ from TOP_N
            actual_n_vis = vis_acts.shape[2]                               # use array dim, not hardcoded N_VIS
            print(f'  Loaded existing: shape={vis_acts.shape}')
        else:
            actual_top_n = TOP_N
            actual_n_vis = N_VIS                                           # use hardcoded default for new arrays
            vis_acts = np.zeros((n_neurons, actual_top_n, actual_n_vis), dtype=np.float16)
            txt_acts = np.zeros((n_neurons, actual_top_n, MAX_TXT), dtype=np.float16)
            txt_lengths = np.zeros((n_neurons, actual_top_n), dtype=np.int16)
            vis_lengths = np.full((n_neurons, actual_top_n), actual_n_vis, dtype=np.int16)
            print(f'  Created new arrays: ({n_neurons}, {actual_top_n}, {actual_n_vis})')

        # Load Top-N Heap global_max for normalisation
        gmax_path = os.path.join(topn_heap_dir, f'global_max_layer{l}.npy')
        if os.path.isfile(gmax_path):
            global_max = np.load(gmax_path)                                # (n_neurons,) float32
        else:
            global_max = None                                              # no normalisation available
            print('  WARNING: global_max not found — activations will be raw')

        # Process each fig3 entry for this layer
        _neuron_rank_counter = {}                                          # track rank per neuron for (e)/(f)
        for entry in entries_by_layer[l]:
            nidx = entry['neuron_idx']                                     # neuron index within FFN
            coco_id = entry['coco_id']                                     # target COCO image ID
            panel = entry['panel']

            print(f'\n  Panel {panel}: neuron {nidx}, image {coco_id}')

            # Validate neuron index
            if nidx >= n_neurons:
                print(f'    ERROR: neuron_idx {nidx} >= n_neurons {n_neurons}. '
                      f'Wrong model? Skipping.')
                continue

            # Load COCO image
            img_path = os.path.join(args.coco_img_dir, f'{coco_id}.jpg')
            if not os.path.isfile(img_path):
                print(f'    ERROR: Image not found: {img_path}. Skipping.')
                continue
            img = Image.open(img_path).convert('RGB')

            # Get generated description
            desc = descriptions.get(coco_id, '')
            if not desc:
                print(f'    WARNING: No description for {coco_id}. '
                      f'Using placeholder.')
                desc = 'A photograph.'

            # Build teacher-forcing prompt
            text = build_prompt_text(args.model_type, desc)

            # Prepare inputs
            inputs, vis_mask, n_vis = prepare_inputs(
                args.model_type, processor, model, img, text,
                device, image_token_id)

            # Forward pass with baukit hook
            with torch.no_grad():
                with TraceDict(model, [layer_name], retain_input=_retain_input) as td:
                    model(**inputs)

            if _retain_input:
                out = td[layer_name].input[0]                              # gate_up/attn: input to projection
            else:
                out = td[layer_name].output                                # gate: act_fn output
            if isinstance(out, tuple):
                out = out[0]
            # Squeeze batch dim if present: (batch, seq, hidden) → (seq, hidden)
            if out.dim() == 3:
                out = out[0]                                               # remove batch dim

            if args.hook_point == 'attn' and _n_heads is not None:
                # Reshape (seq, hidden) → (seq, n_heads) via L2 norm per head
                seq_len = out.shape[0]
                reshaped = out.reshape(seq_len, _n_heads, -1)
                acts = torch.norm(reshaped.float(), dim=-1).cpu().numpy()  # (seq_len, n_heads)
            else:
                acts = out.float().cpu().numpy()                           # (seq_len, n_neurons)

            # Extract this neuron's activations across all positions
            neuron_acts = acts[:, nidx]                                    # (seq_len,)

            # Normalise to 0–10 scale using global_max
            if global_max is not None and global_max[nidx] > 0:
                neuron_acts = neuron_acts / global_max[nidx] * 10.0        # same scale as Activation Pattern
            else:
                # No global_max: use this sample's max for normalisation
                local_max = neuron_acts.max()
                if local_max > 0:
                    neuron_acts = neuron_acts / local_max * 10.0

            # Split into visual and text activations
            vis_positions = np.where(vis_mask)[0]                          # indices of visual tokens
            txt_start = vis_positions[-1] + 1 if len(vis_positions) > 0 else 0

            # Visual activations
            n_vis_actual = min(len(vis_positions), actual_n_vis)            # clip to array size
            vis_vals = neuron_acts[vis_positions[:n_vis_actual]]            # extract visual token acts
            vis_padded = np.zeros(actual_n_vis, dtype=np.float16)          # zero-padded array
            vis_padded[:n_vis_actual] = vis_vals.astype(np.float16)

            # Text activations (description tokens only — after visual + prompt tokens)
            # Compute description token count
            if args.model_type in ('internvl', 'qwen2vl', 'llava-ov'):
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(
                    args.original_model_path, trust_remote_code=True)
                desc_token_ids = _tok(desc, add_special_tokens=False).input_ids
                n_desc_tokens = len(desc_token_ids)
            else:
                # For LLaVA: count by tokenizing desc alone
                if hasattr(processor, 'tokenizer'):
                    _tok = processor.tokenizer                             # HF processor wraps tokenizer
                elif isinstance(processor, tuple):
                    _tok = processor[0]                                    # original: (tokenizer, img_proc)
                else:
                    _tok = processor
                desc_token_ids = _tok(desc, add_special_tokens=False).input_ids
                n_desc_tokens = len(desc_token_ids)

            # Description tokens are at the END of the sequence
            seq_len = len(neuron_acts)
            txt_start_pos = seq_len - n_desc_tokens                        # first description token position
            txt_end_pos = seq_len                                          # exclusive end

            n_txt_actual = min(n_desc_tokens, MAX_TXT)                     # clip to array size
            txt_vals = neuron_acts[txt_start_pos:txt_start_pos + n_txt_actual]
            txt_padded = np.zeros(MAX_TXT, dtype=np.float16)
            txt_padded[:n_txt_actual] = txt_vals.astype(np.float16)

            # Store into next available rank slot for this neuron
            # Panels (e) and (f) share same neuron — (e)→rank 0, (f)→rank 1
            rank_slot = _neuron_rank_counter.get(nidx, 0)
            _neuron_rank_counter[nidx] = rank_slot + 1
            vis_acts[nidx, rank_slot, :] = vis_padded
            txt_acts[nidx, rank_slot, :] = txt_padded
            txt_lengths[nidx, rank_slot] = n_txt_actual
            vis_lengths[nidx, rank_slot] = n_vis_actual

            print(f'    seq_len={seq_len}, n_vis={n_vis_actual}, '
                  f'n_txt={n_txt_actual}')
            print(f'    vis_max={vis_padded[:n_vis_actual].max():.1f}, '
                  f'txt_max={txt_padded[:n_txt_actual].max():.1f}')
            print(f'    → Written to rank {rank_slot}')

        # Save updated Activation Pattern data
        np.savez_compressed(
            npz_path,
            vis_acts=vis_acts,                                             # (n_neurons, top_n, N_VIS) float16
            vis_lengths=vis_lengths,                                       # (n_neurons, top_n) int16
            txt_acts=txt_acts,                                             # (n_neurons, top_n, MAX_TXT) float16
            txt_lengths=txt_lengths,                                       # (n_neurons, top_n) int16
        )
        print(f'  Saved: {npz_path}')

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'Done. Patched {len(layers_to_patch)} layers in {elapsed:.1f}s')
    print(f'{"="*60}')

    # ── Run visualize_neuron_activations.py --fig3 ───────────────
    if args.skip_viz:
        print('\n  --skip_viz: skipping Figure 3 panel generation')
        return

    viz_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'visualize_neuron_activations.py')
    if not os.path.isfile(viz_script):
        print(f'\n  SKIP visualize: {viz_script} not found')
        return

    viz_out = args.output_dir or os.path.join(
        os.path.dirname(args.data_dir), 'fig3_panels')

    print(f'\n{"="*60}')
    print(f'Generating Figure 3 panels → {viz_out}')
    print(f'{"="*60}\n')

    import subprocess
    viz_cmd = [
        sys.executable, viz_script,
        '--data_dir', args.data_dir,
        '--coco_img_dir', args.coco_img_dir,
        '--model_type', args.model_type,
        '--output_dir', viz_out,
        '--hook_point', args.hook_point,
        '--fig3',
    ]
    if args.generated_desc_path:
        viz_cmd += ['--generated_desc_path', args.generated_desc_path]
    if args.model_name:
        viz_cmd += ['--model_name', args.model_name]
    if json_path:
        viz_cmd += ['--fig3_json', json_path]

    viz_result = subprocess.run(viz_cmd, capture_output=False)
    if viz_result.returncode != 0:
        print(f'\n  WARNING: visualize exited with code {viz_result.returncode}')
    else:
        print(f'\n  Figure 3 panels saved to {viz_out}')


if __name__ == '__main__':
    main()