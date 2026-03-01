#!/usr/bin/env python3
"""
visualize_neuron_activations.py — Generate Xu et al. Figure 3-style visualizations

Reads Top-N Heap / Activation Pattern data saved by neuron_modality_statistical.py and
produces activation-overlay figures for selected neurons.  Each figure shows:

    ┌───────────────────────────────────────────────────────────────────┐
    │  Header: neuron type, layer, neuron index, max vis/txt act       │
    ├──────────────────────┬────────────────────────────────────────────┤
    │   Original image     │   Activation-modulated image              │
    ├──────────────────────┴────────────────────────────────────────────┤
    │   Text tokens with green activation highlighting                 │
    └───────────────────────────────────────────────────────────────────┘

The activation-modulated image follows Xu's formula (Section 6.1):
    pixel = pixel_value × (act/10) + 255 × (1 − act/10)
    → high activation = original colour; low activation = white

Text tokens are rendered with green background intensity ∝ normalised
activation (darker green = higher activation), matching Xu Figure 1/3.

Modes:
    --fig3       Reproduce exact Xu Figure 3 panels (a)-(f) with paper's neurons + images
    --fig89      Reproduce Xu Figures 8 & 9: two-sample panels (top-2 images per neuron)
    --supplementary  Reproduce supplementary Figures 15-17 (9 neurons, two-sample layout)
    --two_samples    Use two-sample layout for auto-selected or specific neurons
    --auto       Auto-select one high-confidence neuron per type (default)
    --neuron     Specify exact layer + neuron index

Requires:  Top-N Heap + Activation Pattern outputs from neuron_modality_statistical.py,
           generated_descriptions.json, detail_23k.json, COCO images,
           HuggingFace tokenizer (CPU only, no GPU needed).

Usage:
    # Auto-select best example of each type (no GPU needed)
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --coco_img_dir /path/to/train2017/ \\
        --generated_desc_path generated_descriptions.json \\
        --detail_23k_path detail_23k.json

    # Reproduce Xu et al. Figure 3 panels (a)-(f) with exact neurons + images
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --coco_img_dir /path/to/train2017/ \\
        --generated_desc_path generated_descriptions.json \\
        --detail_23k_path detail_23k.json \\
        --fig3

    # Reproduce Xu Figures 8 & 9: two-sample panels per neuron
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --coco_img_dir /path/to/train2017/ \\
        --generated_desc_path generated_descriptions.json \\
        --fig89

    # Reproduce supplementary Figures 15-17 (9 neurons)
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --coco_img_dir /path/to/train2017/ \\
        --generated_desc_path generated_descriptions.json \\
        --supplementary

    # Auto-select neurons but use two-sample layout
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --coco_img_dir /path/to/train2017/ \\
        --generated_desc_path generated_descriptions.json \\
        --two_samples

    # Specific neuron
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --layer 27 --neuron_idx 3900 \\
        --coco_img_dir /path/to/train2017/

    # Specific neuron, specific sample rank (0 = top-1 activated)
    python visualize_neuron_activations.py \\
        --data_dir outputs/llava-1.5-7b/llm \\
        --layer 27 --neuron_idx 3900 --rank 0 \\
        --coco_img_dir /path/to/train2017/
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use('Agg')                                               # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# Xu et al. Figure 3 — exact neuron + image pairings from the paper
# ═══════════════════════════════════════════════════════════════════
#
# Each entry: (panel_label, layer, neuron_idx, coco_image_id, neuron_type)
#
# Panels (a)-(f) in Figure 3 show one sample per neuron.  Panels (e)
# and (f) are the SAME neuron (layer 21, neuron 6100) shown on two
# different top-activated images (zebras and fire hydrant/pigeons).
#
# The COCO image IDs match FIG3_IMAGES in generate_descriptions.py.

FIG3_NEURONS_LLAVA = [
    {
        'panel': '(a)',                                             # Figure 3 panel label
        'layer': 27,                                                # LLaVA-1.5 layer index
        'neuron_idx': 3900,                                         # neuron index within FFN (11,008 per layer)
        'coco_id': '000000403170',                                  # COCO image shown in the panel
        'label': 'visual',                                          # neuron type
        'description': 'Visual neuron — airplane/motorcycles',      # human-readable description
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
        'neuron_idx': 6100,                                         # same neuron as (e)
        'coco_id': '000000060034',                                  # different image
        'label': 'multimodal',
        'description': 'Multi-modal neuron — fire hydrant/pigeons (same neuron as e)',
    },
]

# ─── InternVL2.5-8B: placeholder — UPDATE once interesting neurons are identified ──
# 32 layers, 14,336 neurons per layer. Same dict keys as LLaVA entries above.
FIG3_NEURONS_INTERNVL = [
    # EXAMPLE (replace with real neurons after InternVL classification run):
    # {'panel': '(a)', 'layer': 27, 'neuron_idx': 5000,
    #  'coco_id': '000000403170', 'label': 'visual',
    #  'description': 'Visual neuron — TBD'},
]

# ─── LLaVA-OneVision-7B: placeholder — UPDATE once interesting neurons are identified ──
# 28 layers (Qwen2-7B backbone), ~18,944 neurons per layer.
FIG3_NEURONS_LLAVA_OV = [
    # EXAMPLE (replace with real neurons after LLaVA-OV classification run):
    # {'panel': '(a)', 'layer': 20, 'neuron_idx': 3000,
    #  'coco_id': '000000403170', 'label': 'visual',
    #  'description': 'Visual neuron — TBD'},
]

# ─── Qwen2.5-VL-7B: placeholder — UPDATE once interesting neurons are identified ──
# 28 layers (Qwen2.5-7B backbone), ~18,944 neurons per layer.
FIG3_NEURONS_QWEN2VL = [
    # EXAMPLE (replace with real neurons after Qwen2.5-VL classification run):
    # {'panel': '(a)', 'layer': 20, 'neuron_idx': 3000,
    #  'coco_id': '000000403170', 'label': 'visual',
    #  'description': 'Visual neuron — TBD'},
]

# ─── Registry: model_type → neuron table ──────────────────────────
FIG3_NEURONS_BY_MODEL = {
    'llava-hf':          FIG3_NEURONS_LLAVA,                              # HF LLaVA-1.5-7b
    'llava-liuhaotian':  FIG3_NEURONS_LLAVA,                              # original LLaVA-1.5-7b
    'internvl':    FIG3_NEURONS_INTERNVL,                           # InternVL2.5-8B (TBD)
    'qwen2vl':     FIG3_NEURONS_QWEN2VL,                            # Qwen2.5-VL-7B (TBD)
    'llava-ov':    FIG3_NEURONS_LLAVA_OV,                           # LLaVA-OneVision-7B (TBD)
}

# Backward compatibility alias
FIG3_NEURONS = FIG3_NEURONS_LLAVA


# ═══════════════════════════════════════════════════════════════════
# Xu et al. Figures 8 & 9 — two-sample neuron visualization
# ═══════════════════════════════════════════════════════════════════
#
# Figures 8 and 9 show TWO top-activated samples for the SAME neuron.
# Each sample is displayed as: Origin | Groundtruth | Prediction.
# The Prediction column requires the simulator (Grounded SAM 2 +
# GPT-4o), which we don't have — so we render Origin + Groundtruth
# and leave a clean placeholder for Prediction.
#
# Supplementary Figures 15, 16, 17 use the same layout for more
# neurons (3 neurons × 2 samples each per figure).

FIG89_NEURONS = [
    # ── Figure 8: Visual neuron ──────────────────────────────
    {
        'figure': 'fig8',
        'layer': 0,
        'neuron_idx': 6098,
        'label': 'visual',
        'explanation': 'the presence of a train or train-related elements.',
    },
    # ── Figure 9: Multi-modal neuron ─────────────────────────
    {
        'figure': 'fig9',
        'layer': 7,
        'neuron_idx': 1410,
        'label': 'multimodal',
        'explanation': 'words related to airplanes and airports.',
    },
]

# Supplementary figures — same 2-sample layout
FIG_SUPPLEMENTARY_NEURONS = [
    # ── Figure 15: Visual neurons (supplementary) ────────────
    {
        'figure': 'fig15a',
        'layer': 2,
        'neuron_idx': 8997,
        'label': 'visual',
        'explanation': 'scenes depicting groups of people gathered together.',
    },
    {
        'figure': 'fig15b',
        'layer': 16,
        'neuron_idx': 4347,
        'label': 'visual',
        'explanation': 'words related to roads, pathways, or walkways.',
    },
    {
        'figure': 'fig15c',
        'layer': 3,
        'neuron_idx': 8142,
        'label': 'visual',
        'explanation': 'references to food, particularly cakes and desserts.',
    },
    # ── Figure 16: Text neurons (supplementary) ──────────────
    {
        'figure': 'fig16a',
        'layer': 29,
        'neuron_idx': 7693,
        'label': 'text',
        'explanation': 'the beginning of sentences or paragraphs.',
    },
    {
        'figure': 'fig16b',
        'layer': 20,
        'neuron_idx': 2063,
        'label': 'text',
        'explanation': 'situations or contexts related to resting or breaks '
                       'during sports activities.',
    },
    {
        'figure': 'fig16c',
        'layer': 6,
        'neuron_idx': 5298,
        'label': 'text',
        'explanation': 'references to the position or location of objects '
                       'within an image.',
    },
    # ── Figure 17: Multi-modal neurons (supplementary) ───────
    {
        'figure': 'fig17a',
        'layer': 24,
        'neuron_idx': 8912,
        'label': 'multimodal',
        'explanation': 'references to rivers and water bodies.',
    },
    {
        'figure': 'fig17b',
        'layer': 23,
        'neuron_idx': 6568,
        'label': 'multimodal',
        'explanation': 'animals, particularly focusing on birds and large mammals.',
    },
    {
        'figure': 'fig17c',
        'layer': 23,
        'neuron_idx': 844,
        'label': 'multimodal',
        'explanation': 'references to trains and train-related settings.',
    },
]


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Data loading helpers
# ═══════════════════════════════════════════════════════════════════

def load_sampled_ids(topn_heap_dir):
    """Load the sample index → COCO image ID mapping from Top-N Heap.

    Top-N Heap saves sampled_ids.json: a list where sampled_ids[idx] = image_id_str.
    This lets us map the integer sample indices stored in top_n_sids back to
    actual COCO image filenames.

    Returns:
        list[str] — sampled_ids[sample_idx] = image_id_string (e.g. "000000323964")
    """
    path = os.path.join(topn_heap_dir, 'sampled_ids.json')             # saved during Top-N Heap
    with open(path) as f:
        return json.load(f)                                         # list of image ID strings


def load_topn_heap_layer(topn_heap_dir, layer):
    """Load Top-N Heap data for a single layer.

    Top-N Heap saves per-layer:
        top_n_sids_layer{l}.npy — (n_neurons, top_n) int32, sample indices
        top_n_acts_layer{l}.npy — (n_neurons, top_n) float32, raw activation values
        global_max_layer{l}.npy — (n_neurons,) float32, global max per neuron

    Returns:
        top_n_sids: (n_neurons, top_n) — which samples are in each neuron's top-N
        top_n_acts: (n_neurons, top_n) — the raw activation values
        global_max: (n_neurons,) — global max for normalisation
    """
    sids = np.load(os.path.join(topn_heap_dir, f'top_n_sids_layer{layer}.npy'))
    acts = np.load(os.path.join(topn_heap_dir, f'top_n_acts_layer{layer}.npy'))
    gmax = np.load(os.path.join(topn_heap_dir, f'global_max_layer{layer}.npy'))
    return sids, acts, gmax


def load_act_pattern_layer(act_pattern_dir, layer):
    """Load Activation Pattern raw normalised activations for a single layer.

    Activation Pattern saves per-layer .npz files:
        vis_acts:    (n_neurons, top_n, 576)     float16 — normalised [0-10] visual activations
        txt_acts:    (n_neurons, top_n, MAX_TXT) float16 — normalised [0-10] text activations
        txt_lengths: (n_neurons, top_n)          int16   — actual text token count per sample

    Returns:
        vis_acts, txt_acts, txt_lengths
    """
    data = np.load(os.path.join(act_pattern_dir, f'raw_acts_layer{layer}.npz'))
    return (data['vis_acts'].astype(np.float32),                    # upcast for arithmetic
            data['txt_acts'].astype(np.float32),
            data['txt_lengths'].astype(np.int32))


def load_neuron_labels(data_dir, layer, model_type='llava-hf'):
    """Load neuron classification labels for a layer.

    neuron_modality_statistical.py saves per-layer:
        {layer_name}/neuron_labels.json — list of dicts with keys:
            neuron_idx, label, pv, pt, pm, pu, global_max_activation, top_n_valid

    Returns:
        list[dict] — one entry per neuron
    """
    # Determine layer name format based on model_type
    if model_type == 'llava-hf':                                          # HF naming convention
        layer_name = f'model.language_model.model.layers.{layer}.mlp.act_fn'
    elif model_type == 'internvl':                                  # InternVL naming convention
        layer_name = f'language_model.model.layers.{layer}.feed_forward.act_fn'
    elif model_type == 'llava-ov':                                   # LLaVA-OneVision (Qwen2 backbone)
        layer_name = f'model.language_model.layers.{layer}.mlp.act_fn'
    elif model_type == 'qwen2vl':                                    # Qwen2.5-VL (Qwen2.5 backbone)
        layer_name = f'model.layers.{layer}.mlp.act_fn'
    else:                                                           # Original LLaVA naming
        layer_name = f'model.layers.{layer}.mlp.act_fn'

    label_path = os.path.join(data_dir, layer_name, 'neuron_labels.json')
    with open(label_path) as f:
        return json.load(f)


def load_id_to_filename(detail_23k_path):
    """Build image_id → filename mapping from detail_23k.json.

    detail_23k.json entries have 'id' (e.g. "000000323964") and
    'image' (e.g. "000000323964.jpg" or "train2017/000000323964.jpg").

    Returns:
        dict — {image_id_str: filename_str}
    """
    with open(detail_23k_path) as f:
        detail_data = json.load(f)
    id_to_fn = {}
    for item in detail_data:
        img_id = item['id']
        fname = os.path.basename(item['image'])                     # strip any directory prefix
        id_to_fn[img_id] = fname
    return id_to_fn


def load_descriptions(desc_path):
    """Load generated descriptions from JSON.

    Supports two formats:
        flat:   {"000000323964": "The image features..."}
        nested: {"000000323964": {"token_ids": [...], "text": "...", ...}}

    Returns:
        dict — {image_id_str: description_text}
    """
    with open(desc_path) as f:
        raw = json.load(f)
    descs = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            descs[k] = v['text']                                    # use decoded text, not joined subwords
        else:
            descs[k] = v
    return descs


def get_description_tokens(tokenizer, img_id, descriptions, model_type='llava-hf'):
    """Re-tokenize the description to recover subword tokens for display.

    Activation Pattern stores activation values per text position but not the token
    strings themselves. We re-tokenize the description exactly as
    neuron_modality_statistical.py does to recover the alignment.

    The "text tokens" are only the generated description tokens (after
    ASSISTANT:), not template tokens. We compute desc_count by subtracting
    the prefix token count from the full token count.

    Returns:
        list[str] — subword token strings for the description only
    """
    desc_text = descriptions[img_id]

    if model_type == 'llava-hf':                                          # HF tokenizer from AutoProcessor
        prefix = "USER: <image>\nCould you describe the image?\nASSISTANT:"
        full = f"USER: <image>\nCould you describe the image?\nASSISTANT: {desc_text}"
    elif model_type in ('internvl', 'qwen2vl', 'llava-ov'):
        # For modern VLMs, tokenise description standalone (no template subtraction needed)
        desc_ids = tokenizer.encode(desc_text, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in desc_ids]
        return tokens
    else:                                                           # original LLaVA v1 template
        from llava.conversation import conv_templates
        conv_p = conv_templates["v1"].copy()
        conv_p.append_message(conv_p.roles[0],
                              "<image>\nCould you describe the image?")
        conv_p.append_message(conv_p.roles[1], None)
        prefix = conv_p.get_prompt()

        conv_f = conv_templates["v1"].copy()
        conv_f.append_message(conv_f.roles[0],
                              "<image>\nCould you describe the image?")
        conv_f.append_message(conv_f.roles[1], desc_text)
        full = conv_f.get_prompt()

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)  # [BOS, ..., ASSISTANT, :]
    full_ids = tokenizer.encode(full, add_special_tokens=True)      # [BOS, ..., ASSISTANT, :, desc...]

    # Description token IDs are the suffix after the prefix
    desc_ids = full_ids[len(prefix_ids):]                           # just the description portion

    # Decode each token individually to get subword strings
    tokens = [tokenizer.decode([tid]) for tid in desc_ids]          # e.g. ["The", " image", " features", ...]

    return tokens


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Image rendering helpers
# ═══════════════════════════════════════════════════════════════════

def make_activation_modulated_image(img_array, vis_acts, patch_grid=24):
    """Create Xu-style activation-modulated image.

    Xu Section 6.1:
        pixel = pixel_value × (activation/10) + 255 × (1 − activation/10)
    So high activation → original colour, low activation → white.

    The 576 visual token activations correspond to a 24×24 grid of CLIP
    ViT-L/14 patches (336×336 input ÷ 14 = 24 patches per side).

    Args:
        img_array: (H, W, 3) uint8 — original image
        vis_acts:  (576,) float — normalised activations [0-10]
        patch_grid: int — grid size (24 for LLaVA-1.5 CLIP ViT-L/14)

    Returns:
        (H, W, 3) uint8 — activation-modulated image
    """
    H, W = img_array.shape[:2]

    # Reshape 576 activations → 24×24 grid
    act_grid = vis_acts[:patch_grid * patch_grid].reshape(           # (24, 24) activation values
        patch_grid, patch_grid)
    act_grid = np.clip(act_grid / 10.0, 0.0, 1.0)                  # normalise to [0, 1]

    # Upsample to image resolution using nearest-neighbor
    # (each patch covers a 14×14 pixel region in the 336×336 CLIP input)
    act_map = np.repeat(np.repeat(act_grid, H // patch_grid + 1,    # repeat rows
                                  axis=0),
                        W // patch_grid + 1, axis=1)                 # repeat cols
    act_map = act_map[:H, :W]                                       # crop to exact image size

    # Apply Xu's blending formula
    act_map_3d = act_map[:, :, np.newaxis]                          # (H, W, 1) for broadcasting
    img_float = img_array.astype(np.float32)
    modulated = img_float * act_map_3d + 255.0 * (1.0 - act_map_3d)  # blend with white
    modulated = np.clip(modulated, 0, 255).astype(np.uint8)

    return modulated


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Text token rendering with green highlighting
# ═══════════════════════════════════════════════════════════════════

def render_text_with_activations(ax, tokens, activations, max_chars_per_line=90):
    """Render text tokens with green background proportional to activation.

    Xu Figures 1, 3: darker green = higher activation. We use a white-to-green
    colour ramp where activation 0 → white (#FFFFFF), activation 10 → dark
    green (#006400).

    Each token is drawn as a coloured rectangle with the token text on top.
    Tokens wrap to the next line when the character count exceeds max_chars_per_line.

    Args:
        ax:          matplotlib Axes to draw on
        tokens:      list[str] — subword token strings
        activations: 1-D array — normalised [0-10] activation per token
        max_chars_per_line: int — approximate characters before wrapping
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    if len(tokens) == 0:
        return

    # Truncate activations to match token count (Activation Pattern may have padding)
    n = min(len(tokens), len(activations))
    tokens = tokens[:n]
    acts = activations[:n]

    # Layout parameters
    font_size = 7                                                   # small font to fit many tokens
    line_height = 0.045                                             # vertical spacing between lines
    char_width = 0.0075                                             # approximate width per character
    x_margin = 0.02                                                 # left margin
    y_start = 0.97                                                  # start near top

    x = x_margin
    y = y_start
    pad_x = 0.003                                                   # horizontal padding inside box
    pad_y = 0.003                                                   # vertical padding inside box

    for i, (tok, act) in enumerate(zip(tokens, acts)):
        tok_display = tok.replace('\n', '↵')                        # show newlines as visible char
        tok_width = len(tok_display) * char_width + 2 * pad_x      # estimated box width

        # Wrap to next line if token would overflow
        if x + tok_width > 1.0 - x_margin:
            x = x_margin
            y -= line_height

        if y < 0.02:                                                # stop if we've run out of vertical space
            # Draw "..." indicator
            ax.text(x, y, '...', fontsize=font_size, va='top',
                    fontfamily='monospace', color='gray')
            break

        # Compute green intensity: act/10 → 0=white, 10=dark green
        intensity = np.clip(act / 10.0, 0.0, 1.0)
        # Interpolate: white (1,1,1) → dark green (0, 0.39, 0)
        r = 1.0 - intensity * 1.0                                  # 1.0 → 0.0
        g = 1.0 - intensity * 0.61                                  # 1.0 → 0.39
        b = 1.0 - intensity * 1.0                                   # 1.0 → 0.0
        bg_color = (r, g, b)

        # Choose text colour: white on dark backgrounds, black on light
        text_color = 'white' if intensity > 0.5 else 'black'

        # Draw background rectangle
        rect = mpatches.FancyBboxPatch(
            (x, y - line_height + pad_y),                           # bottom-left corner
            tok_width, line_height - 2 * pad_y,                     # width, height
            boxstyle="round,pad=0.001",
            facecolor=bg_color, edgecolor='none',
            transform=ax.transAxes, clip_on=True)
        ax.add_patch(rect)

        # Draw token text
        ax.text(x + pad_x, y - line_height / 2 + pad_y,            # position inside box
                tok_display,
                fontsize=font_size, fontfamily='monospace',
                va='center', ha='left', color=text_color,
                transform=ax.transAxes, clip_on=True)

        x += tok_width + 0.002                                     # advance cursor past this token


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Figure composition (one neuron, one sample)
# ═══════════════════════════════════════════════════════════════════

def create_neuron_panel(img_array, vis_acts, txt_acts, txt_len,
                        tokens, layer, neuron_idx, label, pv, pt, pm, pu,
                        patch_grid=24):
    """Create a single Figure 3-style panel for one neuron + one sample.

    Layout:
        Row 0: header text (neuron type, layer, index, max activations)
        Row 1: [original image] [activation-modulated image]
        Row 2: text tokens with green highlighting

    Args:
        img_array:   (H, W, 3) uint8 — original image
        vis_acts:    (576,) float — normalised visual activations [0-10]
        txt_acts:    (MAX_TXT,) float — normalised text activations [0-10]
        txt_len:     int — actual number of text tokens (rest is padding)
        tokens:      list[str] — subword token strings for the description
        layer:       int — layer index
        neuron_idx:  int — neuron index within layer
        label:       str — 'visual', 'text', 'multimodal', 'unknown'
        pv, pt, pm, pu: float — classification probabilities
        patch_grid:  int — CLIP patch grid size (24 for LLaVA-1.5)

    Returns:
        matplotlib Figure
    """
    # Compute max activation values (normalised 0-10 scale)
    max_vis_act = int(round(vis_acts.max()))                        # max visual token activation
    txt_actual = txt_acts[:txt_len] if txt_len > 0 else np.array([0.0])
    max_txt_act = int(round(txt_actual.max()))                      # max text token activation

    # Create activation-modulated image
    mod_img = make_activation_modulated_image(img_array, vis_acts, patch_grid)

    # Pretty label for header
    label_display = {                                               # formatting for display
        'visual': 'Visual neuron',
        'text': 'Text neuron',
        'multimodal': 'Multi-modal neuron',
        'unknown': 'Unknown neuron',
    }.get(label, label.capitalize() + ' neuron')

    # ── Build figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 2, figure=fig,                                 # 3 rows, 2 columns
                  height_ratios=[0.06, 0.54, 0.40],                 # header, images, text
                  hspace=0.15, wspace=0.05)

    # Row 0: Header (spans both columns)
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    header_text = (
        f'{label_display}: layer:{layer}, neuron:{neuron_idx}\n'
        f'Max visual activation:{max_vis_act}, '
        f'max text activation:{max_txt_act}'
    )
    ax_header.text(0.5, 0.5, header_text,                          # centered header
                   fontsize=13, fontweight='bold',
                   ha='center', va='center',
                   transform=ax_header.transAxes)

    # Row 1 left: Original image
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(img_array)
    ax_orig.set_title('Original image', fontsize=10, pad=4)
    ax_orig.axis('off')

    # Row 1 right: Activation-modulated image
    ax_mod = fig.add_subplot(gs[1, 1])
    ax_mod.imshow(mod_img)
    ax_mod.set_title('Activation-modulated image', fontsize=10, pad=4)
    ax_mod.axis('off')

    # Row 2: Text with green highlighting (spans both columns)
    ax_text = fig.add_subplot(gs[2, :])
    render_text_with_activations(ax_text, tokens, txt_actual)

    return fig


# ═══════════════════════════════════════════════════════════════════
# Section 4b — Two-sample panel (Figures 8/9 layout)
# ═══════════════════════════════════════════════════════════════════

def create_two_sample_panel(samples, layer, neuron_idx, label,
                            explanation='', patch_grid=24):
    """Create a Xu Figure 8/9-style panel: two samples for one neuron.

    Layout (matching Xu Figures 8, 9, 15-17):
        Row 0: header — neuron type, layer, index, explanation
        Row 1: [sample (a) images] | [sample (b) images]
               each side: Origin | Groundtruth [| Prediction if avail]
        Row 2: [sample (a) text]   | [sample (b) text]
               Groundtruth text with green highlighting
               [Prediction text with green highlighting, if avail]
        Row 3: score labels (a) and (b) — shown if prediction provided

    The Prediction column is only rendered when the sample dict contains
    'pred_vis_acts' and/or 'pred_txt_acts'.  Without predictions, the
    figure shows Origin + Groundtruth (2 columns per sample).

    Args:
        samples:     list of 2 dicts, each with keys:
                       img_array   — (H, W, 3) uint8
                       vis_acts    — (576,) float [0-10]
                       txt_acts    — (MAX_TXT,) float [0-10]
                       txt_len     — int
                       tokens      — list[str]
                     Optional prediction keys (for future simulator):
                       pred_vis_acts — (576,) float [0-10]
                       pred_txt_acts — (MAX_TXT,) float [0-10]
                       score         — float (Pearson correlation)
        layer:       int — layer index
        neuron_idx:  int — neuron index
        label:       str — 'visual', 'text', 'multimodal', 'unknown'
        explanation: str — neuron explanation text (empty if unavailable)
        patch_grid:  int — CLIP patch grid size

    Returns:
        matplotlib Figure
    """
    n_samples = len(samples)
    assert n_samples == 2, f'Expected 2 samples, got {n_samples}'

    # Check if predictions are available
    has_pred = any('pred_vis_acts' in s for s in samples)

    # Pretty label
    label_display = {
        'visual': 'Visual neuron',
        'text': 'Text neuron',
        'multimodal': 'Multi-modal neuron',
        'unknown': 'Unknown neuron',
    }.get(label, label.capitalize() + ' neuron')

    # Number of image columns per sample: 2 (Origin + GT) or 3 (+Pred)
    n_img_cols = 3 if has_pred else 2
    total_cols = n_img_cols * 2                                     # doubled for side-by-side samples

    # ── Build figure ──────────────────────────────────────────
    fig_width = 7 * n_img_cols                                      # scale width by column count
    fig = plt.figure(figsize=(fig_width, 10))

    # Outer grid: header, images, text_a/b, [scores]
    n_rows = 4 if has_pred else 3                                   # add score row if predictions
    height_ratios = [0.06, 0.38, 0.52] if not has_pred \
                    else [0.05, 0.35, 0.48, 0.04]
    outer_gs = GridSpec(n_rows, 1, figure=fig,
                        height_ratios=height_ratios,
                        hspace=0.12)

    # Row 0: Header
    ax_header = fig.add_subplot(outer_gs[0])
    ax_header.axis('off')
    header_line1 = f'{label_display}: Layer: {layer}, neuron: {neuron_idx}'
    header_line2 = f'Explanation: {explanation}' if explanation else ''
    header_text = header_line1 + ('\n' + header_line2 if header_line2 else '')
    ax_header.text(0.5, 0.5, header_text,
                   fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   transform=ax_header.transAxes)

    # Row 1: Images — inner grid [1 row, total_cols] split evenly
    img_gs = outer_gs[1].subgridspec(1, total_cols + 1,             # +1 for gap column
                                      width_ratios=[1]*n_img_cols + [0.15] + [1]*n_img_cols,
                                      wspace=0.08)

    for si, sample in enumerate(samples):
        img_array = sample['img_array']
        vis_acts = sample['vis_acts']

        # Column offset: sample 0 uses cols 0..n_img_cols-1,
        # sample 1 uses cols n_img_cols+1..total_cols  (skip gap col)
        col_offset = si * (n_img_cols + 1)                          # +1 to skip gap column
        if si == 1:
            col_offset = n_img_cols + 1                             # after gap

        # Origin
        ax_orig = fig.add_subplot(img_gs[0, col_offset])
        ax_orig.imshow(img_array)
        ax_orig.set_title('Origin', fontsize=9, pad=3)
        ax_orig.axis('off')

        # Groundtruth (activation-modulated)
        gt_mod = make_activation_modulated_image(img_array, vis_acts, patch_grid)
        ax_gt = fig.add_subplot(img_gs[0, col_offset + 1])
        ax_gt.imshow(gt_mod)
        ax_gt.set_title('Groundtruth', fontsize=9, pad=3)
        ax_gt.axis('off')

        # Prediction (if available)
        if has_pred and 'pred_vis_acts' in sample:
            pred_mod = make_activation_modulated_image(
                img_array, sample['pred_vis_acts'], patch_grid)
            ax_pred = fig.add_subplot(img_gs[0, col_offset + 2])
            ax_pred.imshow(pred_mod)
            ax_pred.set_title('Prediction', fontsize=9, pad=3)
            ax_pred.axis('off')
        elif has_pred:
            # Placeholder for missing prediction on this sample
            ax_pred = fig.add_subplot(img_gs[0, col_offset + 2])
            ax_pred.text(0.5, 0.5, 'No prediction', fontsize=9,
                         ha='center', va='center', color='gray',
                         transform=ax_pred.transAxes)
            ax_pred.set_title('Prediction', fontsize=9, pad=3)
            ax_pred.axis('off')

    # Row 2: Text highlighting — 2 columns (one per sample)
    txt_gs = outer_gs[2].subgridspec(1, 2, wspace=0.08)

    for si, sample in enumerate(samples):
        txt_acts = sample['txt_acts']
        txt_len = sample['txt_len']
        tokens = sample['tokens']
        txt_actual = txt_acts[:txt_len] if txt_len > 0 else np.array([0.0])

        # How many text rows do we need? GT only, or GT + Pred?
        has_pred_txt = has_pred and 'pred_txt_acts' in sample
        n_txt_rows = 2 if has_pred_txt else 1
        inner_txt_gs = txt_gs[si].subgridspec(
            n_txt_rows, 1, hspace=0.08)

        # Groundtruth text
        ax_gt_txt = fig.add_subplot(inner_txt_gs[0])
        render_text_with_activations(ax_gt_txt, tokens, txt_actual)
        # Subtle label in corner
        ax_gt_txt.text(0.0, 1.0, 'Groundtruth:', fontsize=6,
                       fontweight='bold', color='#555555',
                       va='top', ha='left',
                       transform=ax_gt_txt.transAxes)

        # Prediction text (if available)
        if has_pred_txt:
            pred_txt = sample['pred_txt_acts'][:txt_len] \
                       if txt_len > 0 else np.array([0.0])
            ax_pred_txt = fig.add_subplot(inner_txt_gs[1])
            render_text_with_activations(ax_pred_txt, tokens, pred_txt)
            ax_pred_txt.text(0.0, 1.0, 'Prediction:', fontsize=6,
                             fontweight='bold', color='#555555',
                             va='top', ha='left',
                             transform=ax_pred_txt.transAxes)

    # Row 3: Score labels (only if predictions available)
    if has_pred:
        score_gs = outer_gs[3].subgridspec(1, 2, wspace=0.08)
        for si, sample in enumerate(samples):
            ax_score = fig.add_subplot(score_gs[si])
            ax_score.axis('off')
            score = sample.get('score', None)
            panel_label = '(a)' if si == 0 else '(b)'
            score_text = f'{panel_label} Score={score:.2f}' if score is not None \
                         else f'{panel_label}'
            ax_score.text(0.5, 0.5, score_text,
                          fontsize=11, fontweight='bold',
                          ha='center', va='center',
                          transform=ax_score.transAxes)
    else:
        # Even without scores, add panel labels below images
        # Add them as text annotations on the text axes
        pass

    return fig


def load_two_samples_for_neuron(topn_heap_dir, act_pattern_dir, sampled_ids,
                                id_to_fn, coco_img_dir, descriptions,
                                tokenizer, model_type, layer, neuron_idx,
                                patch_grid=24):
    """Load Top-N Heap/2 data and images for the top-2 activated samples of a neuron.

    This is a convenience function that encapsulates the repeated logic of
    loading activations + images for two samples. Used by --fig89,
    --supplementary, and --two_samples modes.

    Args:
        topn_heap_dir, act_pattern_dir: str — directories with saved Phase data
        sampled_ids:    list[str] — sample_idx → COCO image ID
        id_to_fn:       dict — image_id → filename
        coco_img_dir:   str — path to COCO images
        descriptions:   dict — image_id → description text
        tokenizer:      tokenizer object (for re-tokenizing descriptions)
        model_type:     str — 'llava-hf' or 'llava-liuhaotian'
        layer:          int — layer index
        neuron_idx:     int — neuron index
        patch_grid:     int — CLIP patch grid size

    Returns:
        list[dict] — up to 2 sample dicts, each with keys:
            img_array, vis_acts, txt_acts, txt_len, tokens, img_id
        Returns fewer than 2 if samples are missing/invalid.
    """
    top_n_sids, top_n_acts, global_max = load_topn_heap_layer(topn_heap_dir, layer)
    vis_acts_all, txt_acts_all, txt_len_all = load_act_pattern_layer(act_pattern_dir, layer)

    # Sort by activation to get top-1 and top-2
    neuron_acts = top_n_acts[neuron_idx]                            # (top_n,)
    sorted_ranks = np.argsort(-neuron_acts)                         # descending

    samples = []
    for rank in range(min(2, len(sorted_ranks))):
        rank_idx = sorted_ranks[rank]
        sample_idx = int(top_n_sids[neuron_idx, rank_idx])
        if sample_idx < 0:
            print(f'    Rank {rank}: no valid sample, skipping')
            continue

        img_id = sampled_ids[sample_idx]
        filename = id_to_fn.get(img_id)
        if filename is None:
            print(f'    Rank {rank}: no filename for {img_id}, skipping')
            continue

        img_path = os.path.join(coco_img_dir, filename)
        if not os.path.exists(img_path):
            print(f'    Rank {rank}: image not found {img_path}, skipping')
            continue

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        vis_acts = vis_acts_all[neuron_idx, rank_idx, :]
        txt_acts = txt_acts_all[neuron_idx, rank_idx, :]
        txt_len = int(txt_len_all[neuron_idx, rank_idx])

        if img_id in descriptions:
            tokens = get_description_tokens(
                tokenizer, img_id, descriptions, model_type)
            if len(tokens) != txt_len and txt_len > 0:
                print(f'    Rank {rank}: token mismatch '
                      f'({len(tokens)} vs {txt_len})')
        else:
            tokens = [f'tok_{i}' for i in range(txt_len)]
            print(f'    Rank {rank}: no description for {img_id}')

        samples.append({
            'img_array': img_array,
            'vis_acts': vis_acts,
            'txt_acts': txt_acts,
            'txt_len': txt_len,
            'tokens': tokens,
            'img_id': img_id,
        })
        print(f'    Rank {rank}: image={img_id}, '
              f'vis_max={vis_acts.max():.1f}, '
              f'txt_max={txt_acts[:txt_len].max():.1f} '
              f'({txt_len} tokens)')

    return samples

def create_composite_figure(panels, suptitle='Neuron Activation Visualization'):
    """Combine multiple single-neuron panels into one composite figure.

    Each panel is a dict with keys matching create_neuron_panel args.
    Arranges panels in a vertical stack, similar to Xu Figure 3 (a)-(f).

    Args:
        panels: list[dict] — each dict has keys:
            img_array, vis_acts, txt_acts, txt_len, tokens,
            layer, neuron_idx, label, pv, pt, pm, pu
        suptitle: str — overall figure title

    Returns:
        list[matplotlib.Figure] — one Figure per panel (for saving individually)
    """
    figs = []
    for panel in panels:
        fig = create_neuron_panel(**panel)
        figs.append(fig)
    return figs


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Auto-selection of high-confidence neurons
# ═══════════════════════════════════════════════════════════════════

def auto_select_neurons(data_dir, layers, model_type='llava-hf',
                        target_types=('visual', 'text', 'multimodal')):
    """Find one high-confidence neuron per type across all layers.

    Strategy: for each target type, scan all layers and find the neuron
    with the highest probability for that type (e.g. highest pv for visual).
    This gives us the clearest example of each category.

    Args:
        data_dir:     str — base output dir (e.g. outputs/llava-1.5-7b/llm)
        layers:       list[int] — layer indices to scan
        model_type:   str — 'llava-hf' or 'llava-liuhaotian'
        target_types: tuple — which types to find

    Returns:
        list[dict] — one entry per type: {layer, neuron_idx, label, pv, pt, pm, pu}
    """
    best = {t: {'prob': -1, 'info': None} for t in target_types}    # track best per type

    prob_key = {'visual': 'pv', 'text': 'pt',                      # map type → probability key
                'multimodal': 'pm', 'unknown': 'pu'}

    for layer in layers:
        try:
            labels = load_neuron_labels(data_dir, layer, model_type)
        except FileNotFoundError:
            continue                                                # skip layers without data

        for entry in labels:
            lbl = entry['label']
            if lbl in target_types:
                p = entry[prob_key[lbl]]                            # probability for this type
                if p > best[lbl]['prob']:
                    best[lbl]['prob'] = p
                    best[lbl]['info'] = {
                        'layer': layer,
                        'neuron_idx': entry['neuron_idx'],
                        'label': lbl,
                        'pv': entry['pv'],
                        'pt': entry['pt'],
                        'pm': entry['pm'],
                        'pu': entry['pu'],
                    }

    selected = []
    for t in target_types:
        if best[t]['info'] is not None:
            print(f'  Auto-selected {t}: layer {best[t]["info"]["layer"]}, '
                  f'neuron {best[t]["info"]["neuron_idx"]}, '
                  f'p={best[t]["prob"]:.3f}')
            selected.append(best[t]['info'])
        else:
            print(f'  WARNING: No {t} neuron found')

    return selected


def find_rank_for_image(top_n_sids, neuron_idx, sampled_ids, target_coco_id):
    """Find the rank in a neuron's top-N that corresponds to a specific COCO image.

    When using --fig3 mode, we know which image Xu displayed for each panel.
    This function searches the neuron's top-N sample indices to find which
    rank contains that specific image.

    Args:
        top_n_sids:     (n_neurons, top_n) int32 — sample indices from Top-N Heap
        neuron_idx:     int — which neuron to search
        sampled_ids:    list[str] — sample_idx → COCO image ID mapping
        target_coco_id: str — the COCO image ID to find (e.g. "000000323964")

    Returns:
        int or None — rank index (0-based) if found, None if image not in top-N
    """
    sids = top_n_sids[neuron_idx]                                   # (top_n,) sample indices for this neuron
    for rank_idx in range(len(sids)):
        sid = int(sids[rank_idx])
        if sid < 0:                                                 # unfilled slot
            continue
        if sampled_ids[sid] == target_coco_id:                      # found the target image
            return rank_idx
    return None                                                     # image not in this neuron's top-N


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Main pipeline
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate Xu et al. Figure 3-style neuron visualizations')

    # Data paths
    p.add_argument('--data_dir', required=True,
                   help='Base output dir from neuron_modality_statistical.py '
                        '(e.g. outputs/llava-1.5-7b/llm)')
    p.add_argument('--coco_img_dir', required=True,
                   help='Path to COCO train2017 images directory')
    p.add_argument('--generated_desc_path',
                   default='generated_descriptions.json',
                   help='Path to generated_descriptions.json')
    p.add_argument('--detail_23k_path',
                   default=os.path.join(os.path.dirname(__file__), '..', 'data', 'detail_23k.json'),
                   help='Path to detail_23k.json')

    # Model / tokenizer
    p.add_argument('--model_type', default='llava-hf',
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
                   help='Model type (determines layer naming + tokenizer)')
    p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
                   help='HuggingFace model ID (for tokenizer loading)')

    # Neuron selection
    p.add_argument('--layer', type=int, default=None,
                   help='Specific layer index (use with --neuron_idx)')
    p.add_argument('--neuron_idx', type=int, default=None,
                   help='Specific neuron index within layer')
    p.add_argument('--rank', type=int, default=0,
                   help='Which top-N sample to visualize (0=highest activation)')
    p.add_argument('--auto', action='store_true', default=True,
                   help='Auto-select best example of each type (default)')
    p.add_argument('--fig3', action='store_true',
                   help='Reproduce Xu et al. Figure 3 panels (a)-(f) using '
                        'the exact neurons and COCO images from the paper. '
                        'Requires full pipeline data (all 32 layers).')
    p.add_argument('--pmbt_data_dir', default=None,
                   help='Path to PMBT classification data dir (for side-by-side '
                        'FT vs PMBT figures). If provided, generates both FT '
                        'and PMBT panels plus combined comparison.')
    p.add_argument('--fig89', action='store_true',
                   help='Reproduce Xu et al. Figures 8 & 9: two-sample '
                        'panels showing top-2 activated images per neuron. '
                        'Uses exact neurons from the paper (layer 0/6098 '
                        'and layer 7/1410).')
    p.add_argument('--supplementary', action='store_true',
                   help='Reproduce supplementary Figures 15-17: two-sample '
                        'panels for 9 additional neurons (3 visual, 3 text, '
                        '3 multimodal). Requires full pipeline data.')
    p.add_argument('--two_samples', action='store_true',
                   help='Use two-sample layout (Figures 8/9 style) for '
                        'auto-selected neurons instead of single-sample '
                        'layout (Figure 3 style).')
    p.add_argument('--types', nargs='+',
                   default=['visual', 'text', 'multimodal'],
                   help='Which neuron types to auto-select')

    # Layer range (for auto-selection scanning)
    p.add_argument('--layer_start', type=int, default=0,
                   help='First layer to scan')
    p.add_argument('--layer_end', type=int, default=31,
                   help='Last layer to scan (inclusive)')

    # Output
    p.add_argument('--output_dir', default='figure3_outputs',
                   help='Directory to save output figures')
    p.add_argument('--dpi', type=int, default=200,
                   help='Figure DPI for saved images')
    p.add_argument('--format', default='png',
                   choices=['png', 'pdf', 'svg'],
                   help='Output figure format')

    # Display
    p.add_argument('--patch_grid', type=int, default=24,
                   help='CLIP patch grid size (24 for ViT-L/14 at 336px)')
    p.add_argument('--n_vis', type=int, default=576,
                   help='Number of visual tokens (576 for LLaVA-1.5)')

    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve directories ───────────────────────────────────
    topn_heap_dir = os.path.join(args.data_dir, 'topn_heap')              # Top-N Heap saved here
    act_pattern_dir = os.path.join(args.data_dir, 'act_pattern_raw')          # Activation Pattern raw acts saved here
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Data dir:   {args.data_dir}')
    print(f'Top-N Heap:    {topn_heap_dir}')
    print(f'Activation Pattern:    {act_pattern_dir}')
    print(f'COCO imgs:  {args.coco_img_dir}')
    print(f'Output:     {args.output_dir}')

    # ── Load shared data ──────────────────────────────────────
    print('\nLoading sampled_ids...')
    sampled_ids = load_sampled_ids(topn_heap_dir)                      # list: sample_idx → image_id_str
    print(f'  {len(sampled_ids)} samples in dataset')

    print('Loading id_to_filename...')
    id_to_fn = load_id_to_filename(args.detail_23k_path)            # image_id → filename
    print(f'  {len(id_to_fn)} images in detail_23k')

    print('Loading descriptions...')
    descriptions = load_descriptions(args.generated_desc_path)      # image_id → text
    print(f'  {len(descriptions)} descriptions loaded')

    # ── Load tokenizer (CPU only, no GPU) ─────────────────────
    print('Loading tokenizer (CPU only)...')
    # Override hf_id for model types whose tokenizer path differs from default
    _hf_id_defaults = {
        'llava-ov': 'modern_vlms/pretrained/llava-onevision-qwen2-7b-ov-hf',
        'qwen2vl':  'modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct',
        'internvl': 'modern_vlms/pretrained/InternVL2_5-8B',
    }
    if args.hf_id == 'llava-hf/llava-1.5-7b-hf' and args.model_type in _hf_id_defaults:
        args.hf_id = _hf_id_defaults[args.model_type]
    if args.model_type == 'llava-hf':
        from transformers import AutoTokenizer                      # lightweight, no model weights
        tokenizer = AutoTokenizer.from_pretrained(args.hf_id)
    elif args.model_type == 'internvl':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_id, trust_remote_code=True)
    elif args.model_type == 'qwen2vl':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_id)       # Qwen2.5-VL tokenizer
    elif args.model_type == 'llava-ov':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_id)       # Qwen2 tokenizer (from LLaVA-OV)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'liuhaotian/llava-v1.5-7b', use_fast=False)
    print(f'  Tokenizer loaded: vocab_size={tokenizer.vocab_size}')

    # ── Determine which neurons to visualize ──────────────────
    if args.fig3:
        # ── Fig3 mode: reproduce exact panels from Xu et al. ──
        # Select per-model neuron table
        fig3_neurons = FIG3_NEURONS_BY_MODEL.get(args.model_type, [])
        if not fig3_neurons:
            print(f'\nNo Figure 3 neurons defined for '
                  f'model_type={args.model_type}.')
            print(f'Update the corresponding FIG3_NEURONS_* list in '
                  'visualize_neuron_activations.py once neurons '
                  'are identified.')
            return

        print('\n' + '═'*60)
        print('  FIG3 MODE: Reproducing Xu et al. Figure 3 panels (a)-(f)')
        print('  Using exact neurons + COCO images from the paper')
        print('═'*60)

        required_layers = sorted(set(e['layer'] for e in fig3_neurons))
        print(f'  Required layers: {required_layers}')
        print(f'  Panels: {len(fig3_neurons)}')

        has_pmbt = (args.pmbt_data_dir is not None
                    and os.path.isdir(args.pmbt_data_dir))
        if has_pmbt:
            print(f'  PMBT data dir: {args.pmbt_data_dir}')
            print(f'  Will generate FT + PMBT + comparison figures')
        else:
            print(f'  PMBT data dir: not provided — FT figures only')

        ft_panel_paths = []                                             # collect FT panel paths
        pmbt_panel_paths = []                                           # collect PMBT panel paths

        for entry in fig3_neurons:
            layer = entry['layer']
            nidx = entry['neuron_idx']
            target_coco_id = entry['coco_id']
            label = entry['label']
            panel = entry['panel']

            print(f'\n{"─"*60}')
            print(f'Panel {panel}: {entry["description"]}')
            print(f'  layer={layer}, neuron={nidx}, '
                  f'target image={target_coco_id}')

            # Load Top-N Heap data for this layer
            try:
                top_n_sids, top_n_acts, global_max = load_topn_heap_layer(
                    topn_heap_dir, layer)
            except FileNotFoundError:
                print(f'  ERROR: Top-N Heap data not found for layer {layer}. '
                      f'Run full pipeline (all 32 layers) first.')
                continue

            # Load Activation Pattern raw activations for this layer
            try:
                vis_acts_all, txt_acts_all, txt_len_all = load_act_pattern_layer(
                    act_pattern_dir, layer)
            except FileNotFoundError:
                print(f'  ERROR: Activation Pattern data not found for layer {layer}. '
                      f'Run full pipeline (all 32 layers) first.')
                continue

            # Find the rank corresponding to the target COCO image
            rank_idx = find_rank_for_image(
                top_n_sids, nidx, sampled_ids, target_coco_id)

            if rank_idx is not None:
                print(f'  Found target image at rank {rank_idx}')
            else:
                # Image not in this neuron's top-N — fall back to top-1
                print(f'  WARNING: Image {target_coco_id} not in neuron\'s '
                      f'top-N. Falling back to rank 0 (top-1 sample).')
                sorted_ranks = np.argsort(-top_n_acts[nidx])
                rank_idx = sorted_ranks[0]

            sample_idx = int(top_n_sids[nidx, rank_idx])
            if sample_idx < 0:
                print(f'  ERROR: Invalid sample at rank {rank_idx}, skipping')
                continue

            img_id = sampled_ids[sample_idx]
            filename = id_to_fn.get(img_id)
            if filename is None:
                print(f'  ERROR: No filename for image {img_id}, skipping')
                continue

            img_path = os.path.join(args.coco_img_dir, filename)
            if not os.path.exists(img_path):
                print(f'  ERROR: Image not found: {img_path}, skipping')
                continue

            print(f'  Using: sample_idx={sample_idx}, image={img_id}, '
                  f'file={filename}')

            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # Extract activations (shared between FT and PMBT)
            vis_acts = vis_acts_all[nidx, rank_idx, :]
            txt_acts = txt_acts_all[nidx, rank_idx, :]
            txt_len = int(txt_len_all[nidx, rank_idx])

            # Get text tokens
            if img_id in descriptions:
                tokens = get_description_tokens(
                    tokenizer, img_id, descriptions, args.model_type)
                if len(tokens) != txt_len and txt_len > 0:
                    print(f'  NOTE: token count mismatch: '
                          f'tokenizer={len(tokens)}, saved={txt_len}')
            else:
                tokens = [f'tok_{i}' for i in range(txt_len)]
                print(f'  WARNING: No description for {img_id}, '
                      f'using placeholders')

            print(f'  Visual acts: max={vis_acts.max():.1f}, '
                  f'mean={vis_acts.mean():.2f}')
            if txt_len > 0:
                print(f'  Text acts ({txt_len} tokens): '
                      f'max={txt_acts[:txt_len].max():.1f}, '
                      f'mean={txt_acts[:txt_len].mean():.2f}')

            panel_tag = panel.strip('()')                              # "a", "b", etc.
            base_name = (f'fig3_{panel_tag}_{label}_layer{layer}'
                         f'_neuron{nidx}_{args.model_type}')

            # ── FT panel ──
            try:
                ft_labels = load_neuron_labels(
                    args.data_dir, layer, args.model_type)
                ft_info = ft_labels[nidx]
                ft_pv, ft_pt = ft_info['pv'], ft_info['pt']
                ft_pm, ft_pu = ft_info['pm'], ft_info['pu']
            except (FileNotFoundError, IndexError):
                ft_pv = ft_pt = ft_pm = ft_pu = 0.0
                print(f'  WARNING: No FT labels for layer {layer}')

            fig_ft = create_neuron_panel(
                img_array=img_array, vis_acts=vis_acts,
                txt_acts=txt_acts, txt_len=txt_len, tokens=tokens,
                layer=layer, neuron_idx=nidx, label=label,
                pv=ft_pv, pt=ft_pt, pm=ft_pm, pu=ft_pu,
                patch_grid=args.patch_grid,
            )
            ft_path = os.path.join(
                args.output_dir, f'{base_name}_ft.{args.format}')
            fig_ft.savefig(ft_path, dpi=args.dpi,
                           bbox_inches='tight', facecolor='white')
            plt.close(fig_ft)
            ft_panel_paths.append(ft_path)
            print(f'  Saved FT:   {ft_path}')

            # ── PMBT panel (if available) ──
            if has_pmbt:
                try:
                    pmbt_labels = load_neuron_labels(
                        args.pmbt_data_dir, layer, args.model_type)
                    pmbt_info = pmbt_labels[nidx]
                    pmbt_pv, pmbt_pt = pmbt_info['pv'], pmbt_info['pt']
                    pmbt_pm, pmbt_pu = pmbt_info['pm'], pmbt_info['pu']
                    pmbt_label = pmbt_info.get('label', label)
                except (FileNotFoundError, IndexError):
                    pmbt_pv = pmbt_pt = pmbt_pm = pmbt_pu = 0.0
                    pmbt_label = label
                    print(f'  WARNING: No PMBT labels for layer {layer}')

                fig_pmbt = create_neuron_panel(
                    img_array=img_array, vis_acts=vis_acts,
                    txt_acts=txt_acts, txt_len=txt_len, tokens=tokens,
                    layer=layer, neuron_idx=nidx, label=pmbt_label,
                    pv=pmbt_pv, pt=pmbt_pt, pm=pmbt_pm, pu=pmbt_pu,
                    patch_grid=args.patch_grid,
                )
                pmbt_path = os.path.join(
                    args.output_dir, f'{base_name}_pmbt.{args.format}')
                fig_pmbt.savefig(pmbt_path, dpi=args.dpi,
                                 bbox_inches='tight', facecolor='white')
                plt.close(fig_pmbt)
                pmbt_panel_paths.append(pmbt_path)
                print(f'  Saved PMBT: {pmbt_path}')

        # ── Create combined figures ──
        def _make_combined_grid(paths, entries, suffix, title_extra=''):
            """Build a 2×3 grid from 6 panel image paths."""
            if len(paths) != 6:
                print(f'\n  WARNING: Only {len(paths)}/6 {suffix} panels, '
                      f'skipping combined {suffix} figure.')
                return None
            print(f'\n{"─"*60}')
            print(f'Creating combined {suffix.upper()} figure (2×3)...')
            fig_c, axes_c = plt.subplots(2, 3, figsize=(36, 20))
            for ax, path, entry in zip(axes_c.flatten(), paths, entries):
                ax.imshow(np.array(Image.open(path)))
                ax.set_title(
                    f'{entry["panel"]} {entry["description"]}{title_extra}',
                    fontsize=14, fontweight='bold')
                ax.axis('off')
            fig_c.tight_layout(pad=2.0)
            cpath = os.path.join(
                args.output_dir,
                f'fig3_combined_{args.model_type}_{suffix}.{args.format}')
            fig_c.savefig(cpath, dpi=args.dpi,
                          bbox_inches='tight', facecolor='white')
            plt.close(fig_c)
            print(f'  Saved: {cpath}')
            return cpath

        _make_combined_grid(ft_panel_paths, fig3_neurons, 'ft')

        if has_pmbt:
            _make_combined_grid(pmbt_panel_paths, fig3_neurons, 'llm_pmbt')

            # ── Side-by-side comparison: PMBT (left) | FT (right) ──
            if len(ft_panel_paths) == 6 and len(pmbt_panel_paths) == 6:
                print(f'\n{"─"*60}')
                print('Creating comparison figure (PMBT left | FT right)...')
                fig_cmp, axes_cmp = plt.subplots(
                    2, 6, figsize=(60, 20))                            # 2 rows × 6 cols

                for i, entry in enumerate(fig3_neurons):
                    row = i // 3                                       # 0 or 1
                    col = (i % 3) * 2                                  # 0, 2, 4

                    # Left: PMBT
                    ax_pmbt = axes_cmp[row, col]
                    ax_pmbt.imshow(np.array(Image.open(pmbt_panel_paths[i])))
                    ax_pmbt.set_title(
                        f'{entry["panel"]} PMBT', fontsize=12,
                        fontweight='bold', color='#2196F3')
                    ax_pmbt.axis('off')

                    # Right: FT
                    ax_ft = axes_cmp[row, col + 1]
                    ax_ft.imshow(np.array(Image.open(ft_panel_paths[i])))
                    ax_ft.set_title(
                        f'{entry["panel"]} Fixed Threshold', fontsize=12,
                        fontweight='bold', color='#FF9800')
                    ax_ft.axis('off')

                fig_cmp.tight_layout(pad=1.5)
                cmp_path = os.path.join(
                    args.output_dir,
                    f'fig3_combined_{args.model_type}_comparison.{args.format}')
                fig_cmp.savefig(cmp_path, dpi=args.dpi,
                                bbox_inches='tight', facecolor='white')
                plt.close(fig_cmp)
                print(f'  Saved comparison: {cmp_path}')

        print(f'\n{"═"*60}')
        print(f'Done. Figure 3 panels saved to {args.output_dir}/')
        print(f'{"═"*60}')
        return                                                      # exit after fig3 mode

    elif args.fig89 or args.supplementary:
        # ── Fig89 / Supplementary mode: two-sample panels ─────
        if args.fig89:
            neuron_list = FIG89_NEURONS
            mode_label = 'FIG89 MODE: Reproducing Xu Figures 8 & 9'
        else:
            neuron_list = FIG_SUPPLEMENTARY_NEURONS
            mode_label = 'SUPPLEMENTARY MODE: Reproducing Xu Figures 15-17'

        print('\n' + '═'*60)
        print(f'  {mode_label}')
        print(f'  Two-sample panels (top-2 activated images per neuron)')
        print('═'*60)

        required_layers = sorted(set(e['layer'] for e in neuron_list))
        print(f'  Required layers: {required_layers}')
        print(f'  Neurons: {len(neuron_list)}')

        for entry in neuron_list:
            layer = entry['layer']
            nidx = entry['neuron_idx']
            label = entry['label']
            fig_tag = entry['figure']
            explanation = entry.get('explanation', '')

            print(f'\n{"─"*60}')
            print(f'{fig_tag}: {label} neuron — layer {layer}, '
                  f'neuron {nidx}')
            if explanation:
                print(f'  Explanation: {explanation}')

            # Load top-2 samples using helper
            try:
                samples = load_two_samples_for_neuron(
                    topn_heap_dir, act_pattern_dir, sampled_ids,
                    id_to_fn, args.coco_img_dir, descriptions,
                    tokenizer, args.model_type, layer, nidx,
                    args.patch_grid)
            except FileNotFoundError as e:
                print(f'  ERROR: Data not found for layer {layer}: {e}')
                continue

            if len(samples) < 2:
                print(f'  WARNING: Only {len(samples)} valid sample(s), '
                      f'need 2 — skipping')
                continue

            # Create two-sample panel
            fig = create_two_sample_panel(
                samples=samples,
                layer=layer,
                neuron_idx=nidx,
                label=label,
                explanation=explanation,
                patch_grid=args.patch_grid,
            )

            out_name = f'{fig_tag}_{label}_layer{layer}_neuron{nidx}.{args.format}'
            out_path = os.path.join(args.output_dir, out_name)
            fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                        facecolor='white')
            plt.close(fig)
            print(f'  Saved: {out_path}')

        which = 'Figures 8-9' if args.fig89 else 'Figures 15-17'
        print(f'\n{"═"*60}')
        print(f'Done. {which} panels saved to {args.output_dir}/')
        print(f'{"═"*60}')
        return                                                      # exit after fig89/supplementary

    elif args.layer is not None and args.neuron_idx is not None:
        # User specified exact neuron
        labels = load_neuron_labels(args.data_dir, args.layer, args.model_type)
        entry = labels[args.neuron_idx]                             # dict with label, pv, pt, etc.
        neurons_to_viz = [{
            'layer': args.layer,
            'neuron_idx': args.neuron_idx,
            'label': entry['label'],
            'pv': entry['pv'], 'pt': entry['pt'],
            'pm': entry['pm'], 'pu': entry['pu'],
        }]
        print(f'\nVisualizing specific neuron: layer={args.layer}, '
              f'idx={args.neuron_idx}, type={entry["label"]}')
    else:
        # Auto-select high-confidence neurons
        print('\nAuto-selecting neurons...')
        all_layers = list(range(args.layer_start, args.layer_end + 1))
        neurons_to_viz = auto_select_neurons(
            args.data_dir, all_layers, args.model_type, tuple(args.types))

    if not neurons_to_viz:
        print('ERROR: No neurons selected. Check data paths and layer range.')
        return

    # ── Generate visualizations ───────────────────────────────
    for info in neurons_to_viz:
        layer = info['layer']
        nidx = info['neuron_idx']
        label = info['label']

        print(f'\n{"─"*60}')
        print(f'Generating: {label} neuron — layer {layer}, neuron {nidx}')
        print(f'  pv={info["pv"]:.3f}  pt={info["pt"]:.3f}  '
              f'pm={info["pm"]:.3f}  pu={info["pu"]:.3f}')

        # ── Two-sample mode (--two_samples flag) ──────────────
        if args.two_samples:
            try:
                samples = load_two_samples_for_neuron(
                    topn_heap_dir, act_pattern_dir, sampled_ids,
                    id_to_fn, args.coco_img_dir, descriptions,
                    tokenizer, args.model_type, layer, nidx,
                    args.patch_grid)
            except FileNotFoundError as e:
                print(f'  ERROR: Data not found for layer {layer}: {e}')
                continue

            if len(samples) < 2:
                print(f'  WARNING: Only {len(samples)} valid sample(s), '
                      f'need 2 — skipping')
                continue

            fig = create_two_sample_panel(
                samples=samples,
                layer=layer,
                neuron_idx=nidx,
                label=label,
                explanation='',
                patch_grid=args.patch_grid,
            )

            out_name = (f'{label}_layer{layer}_neuron{nidx}'
                        f'_2samples.{args.format}')
            out_path = os.path.join(args.output_dir, out_name)
            fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                        facecolor='white')
            plt.close(fig)
            print(f'  Saved: {out_path}')
            continue                                                # skip single-sample logic below

        # ── Single-sample mode (default) ──────────────────────
        # Load Top-N Heap data for this layer
        top_n_sids, top_n_acts, global_max = load_topn_heap_layer(topn_heap_dir, layer)

        # Load Activation Pattern raw activations for this layer
        vis_acts_all, txt_acts_all, txt_len_all = load_act_pattern_layer(act_pattern_dir, layer)

        # Sort ranks by activation value to find the true top-1
        neuron_acts = top_n_acts[nidx]                              # (top_n,) raw activation values
        sorted_ranks = np.argsort(-neuron_acts)                     # descending by activation
        rank = min(args.rank, len(sorted_ranks) - 1)                # requested rank (default 0 = top)
        best_rank = sorted_ranks[rank]                              # index into the top_n arrays

        sample_idx = int(top_n_sids[nidx, best_rank])              # sample index into sampled_ids
        if sample_idx < 0:
            print(f'  WARNING: No valid sample at rank {rank}, skipping')
            continue

        img_id = sampled_ids[sample_idx]                            # COCO image ID string
        filename = id_to_fn.get(img_id)
        if filename is None:
            print(f'  WARNING: No filename for image {img_id}, skipping')
            continue

        img_path = os.path.join(args.coco_img_dir, filename)
        if not os.path.exists(img_path):
            print(f'  WARNING: Image not found: {img_path}, skipping')
            continue

        print(f'  Sample: rank={rank}, sample_idx={sample_idx}, '
              f'image={img_id}, file={filename}')

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)                                   # (H, W, 3) uint8

        # Extract activations for this neuron + sample
        vis_acts = vis_acts_all[nidx, best_rank, :]                 # (576,)
        txt_acts = txt_acts_all[nidx, best_rank, :]                 # (MAX_TXT,)
        txt_len = int(txt_len_all[nidx, best_rank])                 # actual text token count

        # Get text tokens by re-tokenizing the description
        if img_id in descriptions:
            tokens = get_description_tokens(
                tokenizer, img_id, descriptions, args.model_type)
            # Align token count with saved txt_len
            # (minor differences possible due to tokenizer version)
            if len(tokens) != txt_len and txt_len > 0:
                print(f'  NOTE: token count mismatch: '
                      f'tokenizer={len(tokens)}, saved={txt_len}. '
                      f'Using min({len(tokens)}, {txt_len}) tokens.')
        else:
            tokens = [f'tok_{i}' for i in range(txt_len)]           # fallback placeholder
            print(f'  WARNING: No description for {img_id}, using placeholders')

        print(f'  Visual acts: max={vis_acts.max():.1f}, '
              f'mean={vis_acts.mean():.2f}')
        print(f'  Text acts ({txt_len} tokens): max={txt_acts[:txt_len].max():.1f}, '
              f'mean={txt_acts[:txt_len].mean():.2f}')

        # Create the figure
        fig = create_neuron_panel(
            img_array=img_array,
            vis_acts=vis_acts,
            txt_acts=txt_acts,
            txt_len=txt_len,
            tokens=tokens,
            layer=layer,
            neuron_idx=nidx,
            label=label,
            pv=info['pv'], pt=info['pt'],
            pm=info['pm'], pu=info['pu'],
            patch_grid=args.patch_grid,
        )

        # Save
        out_name = f'{label}_layer{layer}_neuron{nidx}_rank{rank}.{args.format}'
        out_path = os.path.join(args.output_dir, out_name)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f'  Saved: {out_path}')

    print(f'\n{"═"*60}')
    print(f'Done. Figures saved to {args.output_dir}/')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()




# #!/usr/bin/env python3
# """
# visualize_neuron_activations.py — Generate Xu et al. Figure 3-style visualizations

# Reads Top-N Heap / Activation Pattern data saved by neuron_modality_statistical.py and
# produces activation-overlay figures for selected neurons.  Each figure shows:

#     ┌───────────────────────────────────────────────────────────────────┐
#     │  Header: neuron type, layer, neuron index, max vis/txt act       │
#     ├──────────────────────┬────────────────────────────────────────────┤
#     │   Original image     │   Activation-modulated image              │
#     ├──────────────────────┴────────────────────────────────────────────┤
#     │   Text tokens with green activation highlighting                 │
#     └───────────────────────────────────────────────────────────────────┘

# The activation-modulated image follows Xu's formula (Section 6.1):
#     pixel = pixel_value × (act/10) + 255 × (1 − act/10)
#     → high activation = original colour; low activation = white

# Text tokens are rendered with green background intensity ∝ normalised
# activation (darker green = higher activation), matching Xu Figure 1/3.

# Modes:
#     --fig3       Reproduce exact Xu Figure 3 panels (a)-(f) with paper's neurons + images
#     --fig89      Reproduce Xu Figures 8 & 9: two-sample panels (top-2 images per neuron)
#     --supplementary  Reproduce supplementary Figures 15-17 (9 neurons, two-sample layout)
#     --two_samples    Use two-sample layout for auto-selected or specific neurons
#     --auto       Auto-select one high-confidence neuron per type (default)
#     --neuron     Specify exact layer + neuron index

# Requires:  Top-N Heap + Activation Pattern outputs from neuron_modality_statistical.py,
#            generated_descriptions.json, detail_23k.json, COCO images,
#            HuggingFace tokenizer (CPU only, no GPU needed).

# Usage:
#     # Auto-select best example of each type (no GPU needed)
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --coco_img_dir /path/to/train2017/ \\
#         --generated_desc_path generated_descriptions.json \\
#         --detail_23k_path detail_23k.json

#     # Reproduce Xu et al. Figure 3 panels (a)-(f) with exact neurons + images
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --coco_img_dir /path/to/train2017/ \\
#         --generated_desc_path generated_descriptions.json \\
#         --detail_23k_path detail_23k.json \\
#         --fig3

#     # Reproduce Xu Figures 8 & 9: two-sample panels per neuron
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --coco_img_dir /path/to/train2017/ \\
#         --generated_desc_path generated_descriptions.json \\
#         --fig89

#     # Reproduce supplementary Figures 15-17 (9 neurons)
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --coco_img_dir /path/to/train2017/ \\
#         --generated_desc_path generated_descriptions.json \\
#         --supplementary

#     # Auto-select neurons but use two-sample layout
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --coco_img_dir /path/to/train2017/ \\
#         --generated_desc_path generated_descriptions.json \\
#         --two_samples

#     # Specific neuron
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --layer 27 --neuron_idx 3900 \\
#         --coco_img_dir /path/to/train2017/

#     # Specific neuron, specific sample rank (0 = top-1 activated)
#     python visualize_neuron_activations.py \\
#         --data_dir outputs/llava-1.5-7b/llm \\
#         --layer 27 --neuron_idx 3900 --rank 0 \\
#         --coco_img_dir /path/to/train2017/
# """

# import argparse
# import json
# import os
# import textwrap

# import matplotlib
# matplotlib.use('Agg')                                               # non-interactive backend for server use
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec
# import numpy as np
# from PIL import Image


# # ═══════════════════════════════════════════════════════════════════
# # Xu et al. Figure 3 — exact neuron + image pairings from the paper
# # ═══════════════════════════════════════════════════════════════════
# #
# # Each entry: (panel_label, layer, neuron_idx, coco_image_id, neuron_type)
# #
# # Panels (a)-(f) in Figure 3 show one sample per neuron.  Panels (e)
# # and (f) are the SAME neuron (layer 21, neuron 6100) shown on two
# # different top-activated images (zebras and fire hydrant/pigeons).
# #
# # The COCO image IDs match FIG3_IMAGES in generate_descriptions.py.

# FIG3_NEURONS_LLAVA = [
#     {
#         'panel': '(a)',                                             # Figure 3 panel label
#         'layer': 27,                                                # LLaVA-1.5 layer index
#         'neuron_idx': 3900,                                         # neuron index within FFN (11,008 per layer)
#         'coco_id': '000000403170',                                  # COCO image shown in the panel
#         'label': 'visual',                                          # neuron type
#         'description': 'Visual neuron — airplane/motorcycles',      # human-readable description
#     },
#     {
#         'panel': '(b)',
#         'layer': 2,
#         'neuron_idx': 4450,
#         'coco_id': '000000065793',
#         'label': 'text',
#         'description': 'Text neuron — teddy bears/stuffed animals',
#     },
#     {
#         'panel': '(c)',
#         'layer': 29,
#         'neuron_idx': 600,
#         'coco_id': '000000156852',
#         'label': 'multimodal',
#         'description': 'Multi-modal neuron — kitchen/thumbs up/tie',
#     },
#     {
#         'panel': '(d)',
#         'layer': 31,
#         'neuron_idx': 1800,
#         'coco_id': '000000323964',
#         'label': 'multimodal',
#         'description': 'Multi-modal neuron — doughnuts',
#     },
#     {
#         'panel': '(e)',
#         'layer': 21,
#         'neuron_idx': 6100,
#         'coco_id': '000000276332',
#         'label': 'multimodal',
#         'description': 'Multi-modal neuron — zebras (same neuron as f)',
#     },
#     {
#         'panel': '(f)',
#         'layer': 21,
#         'neuron_idx': 6100,                                         # same neuron as (e)
#         'coco_id': '000000060034',                                  # different image
#         'label': 'multimodal',
#         'description': 'Multi-modal neuron — fire hydrant/pigeons (same neuron as e)',
#     },
# ]

# # ─── InternVL2.5-8B: placeholder — UPDATE once interesting neurons are identified ──
# # 32 layers, 14,336 neurons per layer. Same dict keys as LLaVA entries above.
# FIG3_NEURONS_INTERNVL = [
#     # EXAMPLE (replace with real neurons after InternVL classification run):
#     # {'panel': '(a)', 'layer': 27, 'neuron_idx': 5000,
#     #  'coco_id': '000000403170', 'label': 'visual',
#     #  'description': 'Visual neuron — TBD'},
# ]

# # ─── LLaVA-OneVision-7B: placeholder — UPDATE once interesting neurons are identified ──
# # 28 layers (Qwen2-7B backbone), ~18,944 neurons per layer.
# FIG3_NEURONS_LLAVA_OV = [
#     # EXAMPLE (replace with real neurons after LLaVA-OV classification run):
#     # {'panel': '(a)', 'layer': 20, 'neuron_idx': 3000,
#     #  'coco_id': '000000403170', 'label': 'visual',
#     #  'description': 'Visual neuron — TBD'},
# ]

# # ─── Qwen2.5-VL-7B: placeholder — UPDATE once interesting neurons are identified ──
# # 28 layers (Qwen2.5-7B backbone), ~18,944 neurons per layer.
# FIG3_NEURONS_QWEN2VL = [
#     # EXAMPLE (replace with real neurons after Qwen2.5-VL classification run):
#     # {'panel': '(a)', 'layer': 20, 'neuron_idx': 3000,
#     #  'coco_id': '000000403170', 'label': 'visual',
#     #  'description': 'Visual neuron — TBD'},
# ]

# # ─── Registry: model_type → neuron table ──────────────────────────
# FIG3_NEURONS_BY_MODEL = {
#     'llava-hf':          FIG3_NEURONS_LLAVA,                              # HF LLaVA-1.5-7b
#     'llava-liuhaotian':  FIG3_NEURONS_LLAVA,                              # original LLaVA-1.5-7b
#     'internvl':    FIG3_NEURONS_INTERNVL,                           # InternVL2.5-8B (TBD)
#     'qwen2vl':     FIG3_NEURONS_QWEN2VL,                            # Qwen2.5-VL-7B (TBD)
#     'llava-ov':    FIG3_NEURONS_LLAVA_OV,                           # LLaVA-OneVision-7B (TBD)
# }

# # Backward compatibility alias
# FIG3_NEURONS = FIG3_NEURONS_LLAVA


# # ═══════════════════════════════════════════════════════════════════
# # Xu et al. Figures 8 & 9 — two-sample neuron visualization
# # ═══════════════════════════════════════════════════════════════════
# #
# # Figures 8 and 9 show TWO top-activated samples for the SAME neuron.
# # Each sample is displayed as: Origin | Groundtruth | Prediction.
# # The Prediction column requires the simulator (Grounded SAM 2 +
# # GPT-4o), which we don't have — so we render Origin + Groundtruth
# # and leave a clean placeholder for Prediction.
# #
# # Supplementary Figures 15, 16, 17 use the same layout for more
# # neurons (3 neurons × 2 samples each per figure).

# FIG89_NEURONS = [
#     # ── Figure 8: Visual neuron ──────────────────────────────
#     {
#         'figure': 'fig8',
#         'layer': 0,
#         'neuron_idx': 6098,
#         'label': 'visual',
#         'explanation': 'the presence of a train or train-related elements.',
#     },
#     # ── Figure 9: Multi-modal neuron ─────────────────────────
#     {
#         'figure': 'fig9',
#         'layer': 7,
#         'neuron_idx': 1410,
#         'label': 'multimodal',
#         'explanation': 'words related to airplanes and airports.',
#     },
# ]

# # Supplementary figures — same 2-sample layout
# FIG_SUPPLEMENTARY_NEURONS = [
#     # ── Figure 15: Visual neurons (supplementary) ────────────
#     {
#         'figure': 'fig15a',
#         'layer': 2,
#         'neuron_idx': 8997,
#         'label': 'visual',
#         'explanation': 'scenes depicting groups of people gathered together.',
#     },
#     {
#         'figure': 'fig15b',
#         'layer': 16,
#         'neuron_idx': 4347,
#         'label': 'visual',
#         'explanation': 'words related to roads, pathways, or walkways.',
#     },
#     {
#         'figure': 'fig15c',
#         'layer': 3,
#         'neuron_idx': 8142,
#         'label': 'visual',
#         'explanation': 'references to food, particularly cakes and desserts.',
#     },
#     # ── Figure 16: Text neurons (supplementary) ──────────────
#     {
#         'figure': 'fig16a',
#         'layer': 29,
#         'neuron_idx': 7693,
#         'label': 'text',
#         'explanation': 'the beginning of sentences or paragraphs.',
#     },
#     {
#         'figure': 'fig16b',
#         'layer': 20,
#         'neuron_idx': 2063,
#         'label': 'text',
#         'explanation': 'situations or contexts related to resting or breaks '
#                        'during sports activities.',
#     },
#     {
#         'figure': 'fig16c',
#         'layer': 6,
#         'neuron_idx': 5298,
#         'label': 'text',
#         'explanation': 'references to the position or location of objects '
#                        'within an image.',
#     },
#     # ── Figure 17: Multi-modal neurons (supplementary) ───────
#     {
#         'figure': 'fig17a',
#         'layer': 24,
#         'neuron_idx': 8912,
#         'label': 'multimodal',
#         'explanation': 'references to rivers and water bodies.',
#     },
#     {
#         'figure': 'fig17b',
#         'layer': 23,
#         'neuron_idx': 6568,
#         'label': 'multimodal',
#         'explanation': 'animals, particularly focusing on birds and large mammals.',
#     },
#     {
#         'figure': 'fig17c',
#         'layer': 23,
#         'neuron_idx': 844,
#         'label': 'multimodal',
#         'explanation': 'references to trains and train-related settings.',
#     },
# ]


# # ═══════════════════════════════════════════════════════════════════
# # Section 1 — Data loading helpers
# # ═══════════════════════════════════════════════════════════════════

# def load_sampled_ids(topn_heap_dir):
#     """Load the sample index → COCO image ID mapping from Top-N Heap.

#     Top-N Heap saves sampled_ids.json: a list where sampled_ids[idx] = image_id_str.
#     This lets us map the integer sample indices stored in top_n_sids back to
#     actual COCO image filenames.

#     Returns:
#         list[str] — sampled_ids[sample_idx] = image_id_string (e.g. "000000323964")
#     """
#     path = os.path.join(topn_heap_dir, 'sampled_ids.json')             # saved during Top-N Heap
#     with open(path) as f:
#         return json.load(f)                                         # list of image ID strings


# def load_topn_heap_layer(topn_heap_dir, layer):
#     """Load Top-N Heap data for a single layer.

#     Top-N Heap saves per-layer:
#         top_n_sids_layer{l}.npy — (n_neurons, top_n) int32, sample indices
#         top_n_acts_layer{l}.npy — (n_neurons, top_n) float32, raw activation values
#         global_max_layer{l}.npy — (n_neurons,) float32, global max per neuron

#     Returns:
#         top_n_sids: (n_neurons, top_n) — which samples are in each neuron's top-N
#         top_n_acts: (n_neurons, top_n) — the raw activation values
#         global_max: (n_neurons,) — global max for normalisation
#     """
#     sids = np.load(os.path.join(topn_heap_dir, f'top_n_sids_layer{layer}.npy'))
#     acts = np.load(os.path.join(topn_heap_dir, f'top_n_acts_layer{layer}.npy'))
#     gmax = np.load(os.path.join(topn_heap_dir, f'global_max_layer{layer}.npy'))
#     return sids, acts, gmax


# def load_act_pattern_layer(act_pattern_dir, layer):
#     """Load Activation Pattern raw normalised activations for a single layer.

#     Activation Pattern saves per-layer .npz files:
#         vis_acts:    (n_neurons, top_n, 576)     float16 — normalised [0-10] visual activations
#         txt_acts:    (n_neurons, top_n, MAX_TXT) float16 — normalised [0-10] text activations
#         txt_lengths: (n_neurons, top_n)          int16   — actual text token count per sample

#     Returns:
#         vis_acts, txt_acts, txt_lengths
#     """
#     data = np.load(os.path.join(act_pattern_dir, f'raw_acts_layer{layer}.npz'))
#     return (data['vis_acts'].astype(np.float32),                    # upcast for arithmetic
#             data['txt_acts'].astype(np.float32),
#             data['txt_lengths'].astype(np.int32))


# def load_neuron_labels(data_dir, layer, model_type='llava-hf'):
#     """Load neuron classification labels for a layer.

#     neuron_modality_statistical.py saves per-layer:
#         {layer_name}/neuron_labels.json — list of dicts with keys:
#             neuron_idx, label, pv, pt, pm, pu, global_max_activation, top_n_valid

#     Returns:
#         list[dict] — one entry per neuron
#     """
#     # Determine layer name format based on model_type
#     if model_type == 'llava-hf':                                          # HF naming convention
#         layer_name = f'model.language_model.model.layers.{layer}.mlp.act_fn'
#     elif model_type == 'internvl':                                  # InternVL naming convention
#         layer_name = f'language_model.model.layers.{layer}.feed_forward.act_fn'
#     elif model_type == 'llava-ov':                                   # LLaVA-OneVision (Qwen2 backbone)
#         layer_name = f'language_model.model.layers.{layer}.mlp.act_fn'
#     elif model_type == 'qwen2vl':                                    # Qwen2.5-VL (Qwen2.5 backbone)
#         layer_name = f'model.layers.{layer}.mlp.act_fn'
#     else:                                                           # Original LLaVA naming
#         layer_name = f'model.layers.{layer}.mlp.act_fn'

#     label_path = os.path.join(data_dir, layer_name, 'neuron_labels.json')
#     with open(label_path) as f:
#         return json.load(f)


# def load_id_to_filename(detail_23k_path):
#     """Build image_id → filename mapping from detail_23k.json.

#     detail_23k.json entries have 'id' (e.g. "000000323964") and
#     'image' (e.g. "000000323964.jpg" or "train2017/000000323964.jpg").

#     Returns:
#         dict — {image_id_str: filename_str}
#     """
#     with open(detail_23k_path) as f:
#         detail_data = json.load(f)
#     id_to_fn = {}
#     for item in detail_data:
#         img_id = item['id']
#         fname = os.path.basename(item['image'])                     # strip any directory prefix
#         id_to_fn[img_id] = fname
#     return id_to_fn


# def load_descriptions(desc_path):
#     """Load generated descriptions from JSON.

#     Supports two formats:
#         flat:   {"000000323964": "The image features..."}
#         nested: {"000000323964": {"token_ids": [...], "text": "...", ...}}

#     Returns:
#         dict — {image_id_str: description_text}
#     """
#     with open(desc_path) as f:
#         raw = json.load(f)
#     descs = {}
#     for k, v in raw.items():
#         if isinstance(v, dict):
#             descs[k] = v['text']                                    # use decoded text, not joined subwords
#         else:
#             descs[k] = v
#     return descs


# def get_description_tokens(tokenizer, img_id, descriptions, model_type='llava-hf'):
#     """Re-tokenize the description to recover subword tokens for display.

#     Activation Pattern stores activation values per text position but not the token
#     strings themselves. We re-tokenize the description exactly as
#     neuron_modality_statistical.py does to recover the alignment.

#     The "text tokens" are only the generated description tokens (after
#     ASSISTANT:), not template tokens. We compute desc_count by subtracting
#     the prefix token count from the full token count.

#     Returns:
#         list[str] — subword token strings for the description only
#     """
#     desc_text = descriptions[img_id]

#     if model_type == 'llava-hf':                                          # HF tokenizer from AutoProcessor
#         prefix = "USER: <image>\nCould you describe the image?\nASSISTANT:"
#         full = f"USER: <image>\nCould you describe the image?\nASSISTANT: {desc_text}"
#     elif model_type in ('internvl', 'qwen2vl', 'llava-ov'):
#         # For modern VLMs, tokenise description standalone (no template subtraction needed)
#         desc_ids = tokenizer.encode(desc_text, add_special_tokens=False)
#         tokens = [tokenizer.decode([tid]) for tid in desc_ids]
#         return tokens
#     else:                                                           # original LLaVA v1 template
#         from llava.conversation import conv_templates
#         conv_p = conv_templates["v1"].copy()
#         conv_p.append_message(conv_p.roles[0],
#                               "<image>\nCould you describe the image?")
#         conv_p.append_message(conv_p.roles[1], None)
#         prefix = conv_p.get_prompt()

#         conv_f = conv_templates["v1"].copy()
#         conv_f.append_message(conv_f.roles[0],
#                               "<image>\nCould you describe the image?")
#         conv_f.append_message(conv_f.roles[1], desc_text)
#         full = conv_f.get_prompt()

#     prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)  # [BOS, ..., ASSISTANT, :]
#     full_ids = tokenizer.encode(full, add_special_tokens=True)      # [BOS, ..., ASSISTANT, :, desc...]

#     # Description token IDs are the suffix after the prefix
#     desc_ids = full_ids[len(prefix_ids):]                           # just the description portion

#     # Decode each token individually to get subword strings
#     tokens = [tokenizer.decode([tid]) for tid in desc_ids]          # e.g. ["The", " image", " features", ...]

#     return tokens


# # ═══════════════════════════════════════════════════════════════════
# # Section 2 — Image rendering helpers
# # ═══════════════════════════════════════════════════════════════════

# def make_activation_modulated_image(img_array, vis_acts, patch_grid=24):
#     """Create Xu-style activation-modulated image.

#     Xu Section 6.1:
#         pixel = pixel_value × (activation/10) + 255 × (1 − activation/10)
#     So high activation → original colour, low activation → white.

#     The 576 visual token activations correspond to a 24×24 grid of CLIP
#     ViT-L/14 patches (336×336 input ÷ 14 = 24 patches per side).

#     Args:
#         img_array: (H, W, 3) uint8 — original image
#         vis_acts:  (576,) float — normalised activations [0-10]
#         patch_grid: int — grid size (24 for LLaVA-1.5 CLIP ViT-L/14)

#     Returns:
#         (H, W, 3) uint8 — activation-modulated image
#     """
#     H, W = img_array.shape[:2]

#     # Reshape 576 activations → 24×24 grid
#     act_grid = vis_acts[:patch_grid * patch_grid].reshape(           # (24, 24) activation values
#         patch_grid, patch_grid)
#     act_grid = np.clip(act_grid / 10.0, 0.0, 1.0)                  # normalise to [0, 1]

#     # Upsample to image resolution using nearest-neighbor
#     # (each patch covers a 14×14 pixel region in the 336×336 CLIP input)
#     act_map = np.repeat(np.repeat(act_grid, H // patch_grid + 1,    # repeat rows
#                                   axis=0),
#                         W // patch_grid + 1, axis=1)                 # repeat cols
#     act_map = act_map[:H, :W]                                       # crop to exact image size

#     # Apply Xu's blending formula
#     act_map_3d = act_map[:, :, np.newaxis]                          # (H, W, 1) for broadcasting
#     img_float = img_array.astype(np.float32)
#     modulated = img_float * act_map_3d + 255.0 * (1.0 - act_map_3d)  # blend with white
#     modulated = np.clip(modulated, 0, 255).astype(np.uint8)

#     return modulated


# # ═══════════════════════════════════════════════════════════════════
# # Section 3 — Text token rendering with green highlighting
# # ═══════════════════════════════════════════════════════════════════

# def render_text_with_activations(ax, tokens, activations, max_chars_per_line=90):
#     """Render text tokens with green background proportional to activation.

#     Xu Figures 1, 3: darker green = higher activation. We use a white-to-green
#     colour ramp where activation 0 → white (#FFFFFF), activation 10 → dark
#     green (#006400).

#     Each token is drawn as a coloured rectangle with the token text on top.
#     Tokens wrap to the next line when the character count exceeds max_chars_per_line.

#     Args:
#         ax:          matplotlib Axes to draw on
#         tokens:      list[str] — subword token strings
#         activations: 1-D array — normalised [0-10] activation per token
#         max_chars_per_line: int — approximate characters before wrapping
#     """
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.axis('off')

#     if len(tokens) == 0:
#         return

#     # Truncate activations to match token count (Activation Pattern may have padding)
#     n = min(len(tokens), len(activations))
#     tokens = tokens[:n]
#     acts = activations[:n]

#     # Layout parameters
#     font_size = 7                                                   # small font to fit many tokens
#     line_height = 0.045                                             # vertical spacing between lines
#     char_width = 0.0075                                             # approximate width per character
#     x_margin = 0.02                                                 # left margin
#     y_start = 0.97                                                  # start near top

#     x = x_margin
#     y = y_start
#     pad_x = 0.003                                                   # horizontal padding inside box
#     pad_y = 0.003                                                   # vertical padding inside box

#     for i, (tok, act) in enumerate(zip(tokens, acts)):
#         tok_display = tok.replace('\n', '↵')                        # show newlines as visible char
#         tok_width = len(tok_display) * char_width + 2 * pad_x      # estimated box width

#         # Wrap to next line if token would overflow
#         if x + tok_width > 1.0 - x_margin:
#             x = x_margin
#             y -= line_height

#         if y < 0.02:                                                # stop if we've run out of vertical space
#             # Draw "..." indicator
#             ax.text(x, y, '...', fontsize=font_size, va='top',
#                     fontfamily='monospace', color='gray')
#             break

#         # Compute green intensity: act/10 → 0=white, 10=dark green
#         intensity = np.clip(act / 10.0, 0.0, 1.0)
#         # Interpolate: white (1,1,1) → dark green (0, 0.39, 0)
#         r = 1.0 - intensity * 1.0                                  # 1.0 → 0.0
#         g = 1.0 - intensity * 0.61                                  # 1.0 → 0.39
#         b = 1.0 - intensity * 1.0                                   # 1.0 → 0.0
#         bg_color = (r, g, b)

#         # Choose text colour: white on dark backgrounds, black on light
#         text_color = 'white' if intensity > 0.5 else 'black'

#         # Draw background rectangle
#         rect = mpatches.FancyBboxPatch(
#             (x, y - line_height + pad_y),                           # bottom-left corner
#             tok_width, line_height - 2 * pad_y,                     # width, height
#             boxstyle="round,pad=0.001",
#             facecolor=bg_color, edgecolor='none',
#             transform=ax.transAxes, clip_on=True)
#         ax.add_patch(rect)

#         # Draw token text
#         ax.text(x + pad_x, y - line_height / 2 + pad_y,            # position inside box
#                 tok_display,
#                 fontsize=font_size, fontfamily='monospace',
#                 va='center', ha='left', color=text_color,
#                 transform=ax.transAxes, clip_on=True)

#         x += tok_width + 0.002                                     # advance cursor past this token


# # ═══════════════════════════════════════════════════════════════════
# # Section 4 — Figure composition (one neuron, one sample)
# # ═══════════════════════════════════════════════════════════════════

# def create_neuron_panel(img_array, vis_acts, txt_acts, txt_len,
#                         tokens, layer, neuron_idx, label, pv, pt, pm, pu,
#                         patch_grid=24):
#     """Create a single Figure 3-style panel for one neuron + one sample.

#     Layout:
#         Row 0: header text (neuron type, layer, index, max activations)
#         Row 1: [original image] [activation-modulated image]
#         Row 2: text tokens with green highlighting

#     Args:
#         img_array:   (H, W, 3) uint8 — original image
#         vis_acts:    (576,) float — normalised visual activations [0-10]
#         txt_acts:    (MAX_TXT,) float — normalised text activations [0-10]
#         txt_len:     int — actual number of text tokens (rest is padding)
#         tokens:      list[str] — subword token strings for the description
#         layer:       int — layer index
#         neuron_idx:  int — neuron index within layer
#         label:       str — 'visual', 'text', 'multimodal', 'unknown'
#         pv, pt, pm, pu: float — classification probabilities
#         patch_grid:  int — CLIP patch grid size (24 for LLaVA-1.5)

#     Returns:
#         matplotlib Figure
#     """
#     # Compute max activation values (normalised 0-10 scale)
#     max_vis_act = int(round(vis_acts.max()))                        # max visual token activation
#     txt_actual = txt_acts[:txt_len] if txt_len > 0 else np.array([0.0])
#     max_txt_act = int(round(txt_actual.max()))                      # max text token activation

#     # Create activation-modulated image
#     mod_img = make_activation_modulated_image(img_array, vis_acts, patch_grid)

#     # Pretty label for header
#     label_display = {                                               # formatting for display
#         'visual': 'Visual neuron',
#         'text': 'Text neuron',
#         'multimodal': 'Multi-modal neuron',
#         'unknown': 'Unknown neuron',
#     }.get(label, label.capitalize() + ' neuron')

#     # ── Build figure ──────────────────────────────────────────
#     fig = plt.figure(figsize=(12, 8))
#     gs = GridSpec(3, 2, figure=fig,                                 # 3 rows, 2 columns
#                   height_ratios=[0.06, 0.54, 0.40],                 # header, images, text
#                   hspace=0.15, wspace=0.05)

#     # Row 0: Header (spans both columns)
#     ax_header = fig.add_subplot(gs[0, :])
#     ax_header.axis('off')
#     header_text = (
#         f'{label_display}: layer:{layer}, neuron:{neuron_idx}\n'
#         f'Max visual activation:{max_vis_act}, '
#         f'max text activation:{max_txt_act}'
#     )
#     ax_header.text(0.5, 0.5, header_text,                          # centered header
#                    fontsize=13, fontweight='bold',
#                    ha='center', va='center',
#                    transform=ax_header.transAxes)

#     # Row 1 left: Original image
#     ax_orig = fig.add_subplot(gs[1, 0])
#     ax_orig.imshow(img_array)
#     ax_orig.set_title('Original image', fontsize=10, pad=4)
#     ax_orig.axis('off')

#     # Row 1 right: Activation-modulated image
#     ax_mod = fig.add_subplot(gs[1, 1])
#     ax_mod.imshow(mod_img)
#     ax_mod.set_title('Activation-modulated image', fontsize=10, pad=4)
#     ax_mod.axis('off')

#     # Row 2: Text with green highlighting (spans both columns)
#     ax_text = fig.add_subplot(gs[2, :])
#     render_text_with_activations(ax_text, tokens, txt_actual)

#     return fig


# # ═══════════════════════════════════════════════════════════════════
# # Section 4b — Two-sample panel (Figures 8/9 layout)
# # ═══════════════════════════════════════════════════════════════════

# def create_two_sample_panel(samples, layer, neuron_idx, label,
#                             explanation='', patch_grid=24):
#     """Create a Xu Figure 8/9-style panel: two samples for one neuron.

#     Layout (matching Xu Figures 8, 9, 15-17):
#         Row 0: header — neuron type, layer, index, explanation
#         Row 1: [sample (a) images] | [sample (b) images]
#                each side: Origin | Groundtruth [| Prediction if avail]
#         Row 2: [sample (a) text]   | [sample (b) text]
#                Groundtruth text with green highlighting
#                [Prediction text with green highlighting, if avail]
#         Row 3: score labels (a) and (b) — shown if prediction provided

#     The Prediction column is only rendered when the sample dict contains
#     'pred_vis_acts' and/or 'pred_txt_acts'.  Without predictions, the
#     figure shows Origin + Groundtruth (2 columns per sample).

#     Args:
#         samples:     list of 2 dicts, each with keys:
#                        img_array   — (H, W, 3) uint8
#                        vis_acts    — (576,) float [0-10]
#                        txt_acts    — (MAX_TXT,) float [0-10]
#                        txt_len     — int
#                        tokens      — list[str]
#                      Optional prediction keys (for future simulator):
#                        pred_vis_acts — (576,) float [0-10]
#                        pred_txt_acts — (MAX_TXT,) float [0-10]
#                        score         — float (Pearson correlation)
#         layer:       int — layer index
#         neuron_idx:  int — neuron index
#         label:       str — 'visual', 'text', 'multimodal', 'unknown'
#         explanation: str — neuron explanation text (empty if unavailable)
#         patch_grid:  int — CLIP patch grid size

#     Returns:
#         matplotlib Figure
#     """
#     n_samples = len(samples)
#     assert n_samples == 2, f'Expected 2 samples, got {n_samples}'

#     # Check if predictions are available
#     has_pred = any('pred_vis_acts' in s for s in samples)

#     # Pretty label
#     label_display = {
#         'visual': 'Visual neuron',
#         'text': 'Text neuron',
#         'multimodal': 'Multi-modal neuron',
#         'unknown': 'Unknown neuron',
#     }.get(label, label.capitalize() + ' neuron')

#     # Number of image columns per sample: 2 (Origin + GT) or 3 (+Pred)
#     n_img_cols = 3 if has_pred else 2
#     total_cols = n_img_cols * 2                                     # doubled for side-by-side samples

#     # ── Build figure ──────────────────────────────────────────
#     fig_width = 7 * n_img_cols                                      # scale width by column count
#     fig = plt.figure(figsize=(fig_width, 10))

#     # Outer grid: header, images, text_a/b, [scores]
#     n_rows = 4 if has_pred else 3                                   # add score row if predictions
#     height_ratios = [0.06, 0.38, 0.52] if not has_pred \
#                     else [0.05, 0.35, 0.48, 0.04]
#     outer_gs = GridSpec(n_rows, 1, figure=fig,
#                         height_ratios=height_ratios,
#                         hspace=0.12)

#     # Row 0: Header
#     ax_header = fig.add_subplot(outer_gs[0])
#     ax_header.axis('off')
#     header_line1 = f'{label_display}: Layer: {layer}, neuron: {neuron_idx}'
#     header_line2 = f'Explanation: {explanation}' if explanation else ''
#     header_text = header_line1 + ('\n' + header_line2 if header_line2 else '')
#     ax_header.text(0.5, 0.5, header_text,
#                    fontsize=12, fontweight='bold',
#                    ha='center', va='center',
#                    transform=ax_header.transAxes)

#     # Row 1: Images — inner grid [1 row, total_cols] split evenly
#     img_gs = outer_gs[1].subgridspec(1, total_cols + 1,             # +1 for gap column
#                                       width_ratios=[1]*n_img_cols + [0.15] + [1]*n_img_cols,
#                                       wspace=0.08)

#     for si, sample in enumerate(samples):
#         img_array = sample['img_array']
#         vis_acts = sample['vis_acts']

#         # Column offset: sample 0 uses cols 0..n_img_cols-1,
#         # sample 1 uses cols n_img_cols+1..total_cols  (skip gap col)
#         col_offset = si * (n_img_cols + 1)                          # +1 to skip gap column
#         if si == 1:
#             col_offset = n_img_cols + 1                             # after gap

#         # Origin
#         ax_orig = fig.add_subplot(img_gs[0, col_offset])
#         ax_orig.imshow(img_array)
#         ax_orig.set_title('Origin', fontsize=9, pad=3)
#         ax_orig.axis('off')

#         # Groundtruth (activation-modulated)
#         gt_mod = make_activation_modulated_image(img_array, vis_acts, patch_grid)
#         ax_gt = fig.add_subplot(img_gs[0, col_offset + 1])
#         ax_gt.imshow(gt_mod)
#         ax_gt.set_title('Groundtruth', fontsize=9, pad=3)
#         ax_gt.axis('off')

#         # Prediction (if available)
#         if has_pred and 'pred_vis_acts' in sample:
#             pred_mod = make_activation_modulated_image(
#                 img_array, sample['pred_vis_acts'], patch_grid)
#             ax_pred = fig.add_subplot(img_gs[0, col_offset + 2])
#             ax_pred.imshow(pred_mod)
#             ax_pred.set_title('Prediction', fontsize=9, pad=3)
#             ax_pred.axis('off')
#         elif has_pred:
#             # Placeholder for missing prediction on this sample
#             ax_pred = fig.add_subplot(img_gs[0, col_offset + 2])
#             ax_pred.text(0.5, 0.5, 'No prediction', fontsize=9,
#                          ha='center', va='center', color='gray',
#                          transform=ax_pred.transAxes)
#             ax_pred.set_title('Prediction', fontsize=9, pad=3)
#             ax_pred.axis('off')

#     # Row 2: Text highlighting — 2 columns (one per sample)
#     txt_gs = outer_gs[2].subgridspec(1, 2, wspace=0.08)

#     for si, sample in enumerate(samples):
#         txt_acts = sample['txt_acts']
#         txt_len = sample['txt_len']
#         tokens = sample['tokens']
#         txt_actual = txt_acts[:txt_len] if txt_len > 0 else np.array([0.0])

#         # How many text rows do we need? GT only, or GT + Pred?
#         has_pred_txt = has_pred and 'pred_txt_acts' in sample
#         n_txt_rows = 2 if has_pred_txt else 1
#         inner_txt_gs = txt_gs[si].subgridspec(
#             n_txt_rows, 1, hspace=0.08)

#         # Groundtruth text
#         ax_gt_txt = fig.add_subplot(inner_txt_gs[0])
#         render_text_with_activations(ax_gt_txt, tokens, txt_actual)
#         # Subtle label in corner
#         ax_gt_txt.text(0.0, 1.0, 'Groundtruth:', fontsize=6,
#                        fontweight='bold', color='#555555',
#                        va='top', ha='left',
#                        transform=ax_gt_txt.transAxes)

#         # Prediction text (if available)
#         if has_pred_txt:
#             pred_txt = sample['pred_txt_acts'][:txt_len] \
#                        if txt_len > 0 else np.array([0.0])
#             ax_pred_txt = fig.add_subplot(inner_txt_gs[1])
#             render_text_with_activations(ax_pred_txt, tokens, pred_txt)
#             ax_pred_txt.text(0.0, 1.0, 'Prediction:', fontsize=6,
#                              fontweight='bold', color='#555555',
#                              va='top', ha='left',
#                              transform=ax_pred_txt.transAxes)

#     # Row 3: Score labels (only if predictions available)
#     if has_pred:
#         score_gs = outer_gs[3].subgridspec(1, 2, wspace=0.08)
#         for si, sample in enumerate(samples):
#             ax_score = fig.add_subplot(score_gs[si])
#             ax_score.axis('off')
#             score = sample.get('score', None)
#             panel_label = '(a)' if si == 0 else '(b)'
#             score_text = f'{panel_label} Score={score:.2f}' if score is not None \
#                          else f'{panel_label}'
#             ax_score.text(0.5, 0.5, score_text,
#                           fontsize=11, fontweight='bold',
#                           ha='center', va='center',
#                           transform=ax_score.transAxes)
#     else:
#         # Even without scores, add panel labels below images
#         # Add them as text annotations on the text axes
#         pass

#     return fig


# def load_two_samples_for_neuron(topn_heap_dir, act_pattern_dir, sampled_ids,
#                                 id_to_fn, coco_img_dir, descriptions,
#                                 tokenizer, model_type, layer, neuron_idx,
#                                 patch_grid=24):
#     """Load Top-N Heap/2 data and images for the top-2 activated samples of a neuron.

#     This is a convenience function that encapsulates the repeated logic of
#     loading activations + images for two samples. Used by --fig89,
#     --supplementary, and --two_samples modes.

#     Args:
#         topn_heap_dir, act_pattern_dir: str — directories with saved Phase data
#         sampled_ids:    list[str] — sample_idx → COCO image ID
#         id_to_fn:       dict — image_id → filename
#         coco_img_dir:   str — path to COCO images
#         descriptions:   dict — image_id → description text
#         tokenizer:      tokenizer object (for re-tokenizing descriptions)
#         model_type:     str — 'llava-hf' or 'llava-liuhaotian'
#         layer:          int — layer index
#         neuron_idx:     int — neuron index
#         patch_grid:     int — CLIP patch grid size

#     Returns:
#         list[dict] — up to 2 sample dicts, each with keys:
#             img_array, vis_acts, txt_acts, txt_len, tokens, img_id
#         Returns fewer than 2 if samples are missing/invalid.
#     """
#     top_n_sids, top_n_acts, global_max = load_topn_heap_layer(topn_heap_dir, layer)
#     vis_acts_all, txt_acts_all, txt_len_all = load_act_pattern_layer(act_pattern_dir, layer)

#     # Sort by activation to get top-1 and top-2
#     neuron_acts = top_n_acts[neuron_idx]                            # (top_n,)
#     sorted_ranks = np.argsort(-neuron_acts)                         # descending

#     samples = []
#     for rank in range(min(2, len(sorted_ranks))):
#         rank_idx = sorted_ranks[rank]
#         sample_idx = int(top_n_sids[neuron_idx, rank_idx])
#         if sample_idx < 0:
#             print(f'    Rank {rank}: no valid sample, skipping')
#             continue

#         img_id = sampled_ids[sample_idx]
#         filename = id_to_fn.get(img_id)
#         if filename is None:
#             print(f'    Rank {rank}: no filename for {img_id}, skipping')
#             continue

#         img_path = os.path.join(coco_img_dir, filename)
#         if not os.path.exists(img_path):
#             print(f'    Rank {rank}: image not found {img_path}, skipping')
#             continue

#         img = Image.open(img_path).convert('RGB')
#         img_array = np.array(img)

#         vis_acts = vis_acts_all[neuron_idx, rank_idx, :]
#         txt_acts = txt_acts_all[neuron_idx, rank_idx, :]
#         txt_len = int(txt_len_all[neuron_idx, rank_idx])

#         if img_id in descriptions:
#             tokens = get_description_tokens(
#                 tokenizer, img_id, descriptions, model_type)
#             if len(tokens) != txt_len and txt_len > 0:
#                 print(f'    Rank {rank}: token mismatch '
#                       f'({len(tokens)} vs {txt_len})')
#         else:
#             tokens = [f'tok_{i}' for i in range(txt_len)]
#             print(f'    Rank {rank}: no description for {img_id}')

#         samples.append({
#             'img_array': img_array,
#             'vis_acts': vis_acts,
#             'txt_acts': txt_acts,
#             'txt_len': txt_len,
#             'tokens': tokens,
#             'img_id': img_id,
#         })
#         print(f'    Rank {rank}: image={img_id}, '
#               f'vis_max={vis_acts.max():.1f}, '
#               f'txt_max={txt_acts[:txt_len].max():.1f} '
#               f'({txt_len} tokens)')

#     return samples

# def create_composite_figure(panels, suptitle='Neuron Activation Visualization'):
#     """Combine multiple single-neuron panels into one composite figure.

#     Each panel is a dict with keys matching create_neuron_panel args.
#     Arranges panels in a vertical stack, similar to Xu Figure 3 (a)-(f).

#     Args:
#         panels: list[dict] — each dict has keys:
#             img_array, vis_acts, txt_acts, txt_len, tokens,
#             layer, neuron_idx, label, pv, pt, pm, pu
#         suptitle: str — overall figure title

#     Returns:
#         list[matplotlib.Figure] — one Figure per panel (for saving individually)
#     """
#     figs = []
#     for panel in panels:
#         fig = create_neuron_panel(**panel)
#         figs.append(fig)
#     return figs


# # ═══════════════════════════════════════════════════════════════════
# # Section 6 — Auto-selection of high-confidence neurons
# # ═══════════════════════════════════════════════════════════════════

# def auto_select_neurons(data_dir, layers, model_type='llava-hf',
#                         target_types=('visual', 'text', 'multimodal')):
#     """Find one high-confidence neuron per type across all layers.

#     Strategy: for each target type, scan all layers and find the neuron
#     with the highest probability for that type (e.g. highest pv for visual).
#     This gives us the clearest example of each category.

#     Args:
#         data_dir:     str — base output dir (e.g. outputs/llava-1.5-7b/llm)
#         layers:       list[int] — layer indices to scan
#         model_type:   str — 'llava-hf' or 'llava-liuhaotian'
#         target_types: tuple — which types to find

#     Returns:
#         list[dict] — one entry per type: {layer, neuron_idx, label, pv, pt, pm, pu}
#     """
#     best = {t: {'prob': -1, 'info': None} for t in target_types}    # track best per type

#     prob_key = {'visual': 'pv', 'text': 'pt',                      # map type → probability key
#                 'multimodal': 'pm', 'unknown': 'pu'}

#     for layer in layers:
#         try:
#             labels = load_neuron_labels(data_dir, layer, model_type)
#         except FileNotFoundError:
#             continue                                                # skip layers without data

#         for entry in labels:
#             lbl = entry['label']
#             if lbl in target_types:
#                 p = entry[prob_key[lbl]]                            # probability for this type
#                 if p > best[lbl]['prob']:
#                     best[lbl]['prob'] = p
#                     best[lbl]['info'] = {
#                         'layer': layer,
#                         'neuron_idx': entry['neuron_idx'],
#                         'label': lbl,
#                         'pv': entry['pv'],
#                         'pt': entry['pt'],
#                         'pm': entry['pm'],
#                         'pu': entry['pu'],
#                     }

#     selected = []
#     for t in target_types:
#         if best[t]['info'] is not None:
#             print(f'  Auto-selected {t}: layer {best[t]["info"]["layer"]}, '
#                   f'neuron {best[t]["info"]["neuron_idx"]}, '
#                   f'p={best[t]["prob"]:.3f}')
#             selected.append(best[t]['info'])
#         else:
#             print(f'  WARNING: No {t} neuron found')

#     return selected


# def find_rank_for_image(top_n_sids, neuron_idx, sampled_ids, target_coco_id):
#     """Find the rank in a neuron's top-N that corresponds to a specific COCO image.

#     When using --fig3 mode, we know which image Xu displayed for each panel.
#     This function searches the neuron's top-N sample indices to find which
#     rank contains that specific image.

#     Args:
#         top_n_sids:     (n_neurons, top_n) int32 — sample indices from Top-N Heap
#         neuron_idx:     int — which neuron to search
#         sampled_ids:    list[str] — sample_idx → COCO image ID mapping
#         target_coco_id: str — the COCO image ID to find (e.g. "000000323964")

#     Returns:
#         int or None — rank index (0-based) if found, None if image not in top-N
#     """
#     sids = top_n_sids[neuron_idx]                                   # (top_n,) sample indices for this neuron
#     for rank_idx in range(len(sids)):
#         sid = int(sids[rank_idx])
#         if sid < 0:                                                 # unfilled slot
#             continue
#         if sampled_ids[sid] == target_coco_id:                      # found the target image
#             return rank_idx
#     return None                                                     # image not in this neuron's top-N


# # ═══════════════════════════════════════════════════════════════════
# # Section 7 — Main pipeline
# # ═══════════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser(
#         description='Generate Xu et al. Figure 3-style neuron visualizations')

#     # Data paths
#     p.add_argument('--data_dir', required=True,
#                    help='Base output dir from neuron_modality_statistical.py '
#                         '(e.g. outputs/llava-1.5-7b/llm)')
#     p.add_argument('--coco_img_dir', required=True,
#                    help='Path to COCO train2017 images directory')
#     p.add_argument('--generated_desc_path',
#                    default='generated_descriptions.json',
#                    help='Path to generated_descriptions.json')
#     p.add_argument('--detail_23k_path',
#                    default=os.path.join(os.path.dirname(__file__), '..', 'data', 'detail_23k.json'),
#                    help='Path to detail_23k.json')

#     # Model / tokenizer
#     p.add_argument('--model_type', default='llava-hf',
#                    choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
#                    help='Model type (determines layer naming + tokenizer)')
#     p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
#                    help='HuggingFace model ID (for tokenizer loading)')

#     # Neuron selection
#     p.add_argument('--layer', type=int, default=None,
#                    help='Specific layer index (use with --neuron_idx)')
#     p.add_argument('--neuron_idx', type=int, default=None,
#                    help='Specific neuron index within layer')
#     p.add_argument('--rank', type=int, default=0,
#                    help='Which top-N sample to visualize (0=highest activation)')
#     p.add_argument('--auto', action='store_true', default=True,
#                    help='Auto-select best example of each type (default)')
#     p.add_argument('--fig3', action='store_true',
#                    help='Reproduce Xu et al. Figure 3 panels (a)-(f) using '
#                         'the exact neurons and COCO images from the paper. '
#                         'Requires full pipeline data (all 32 layers).')
#     p.add_argument('--pmbt_data_dir', default=None,
#                    help='Path to PMBT classification data dir (for side-by-side '
#                         'FT vs PMBT figures). If provided, generates both FT '
#                         'and PMBT panels plus combined comparison.')
#     p.add_argument('--fig89', action='store_true',
#                    help='Reproduce Xu et al. Figures 8 & 9: two-sample '
#                         'panels showing top-2 activated images per neuron. '
#                         'Uses exact neurons from the paper (layer 0/6098 '
#                         'and layer 7/1410).')
#     p.add_argument('--supplementary', action='store_true',
#                    help='Reproduce supplementary Figures 15-17: two-sample '
#                         'panels for 9 additional neurons (3 visual, 3 text, '
#                         '3 multimodal). Requires full pipeline data.')
#     p.add_argument('--two_samples', action='store_true',
#                    help='Use two-sample layout (Figures 8/9 style) for '
#                         'auto-selected neurons instead of single-sample '
#                         'layout (Figure 3 style).')
#     p.add_argument('--types', nargs='+',
#                    default=['visual', 'text', 'multimodal'],
#                    help='Which neuron types to auto-select')

#     # Layer range (for auto-selection scanning)
#     p.add_argument('--layer_start', type=int, default=0,
#                    help='First layer to scan')
#     p.add_argument('--layer_end', type=int, default=31,
#                    help='Last layer to scan (inclusive)')

#     # Output
#     p.add_argument('--output_dir', default='figure3_outputs',
#                    help='Directory to save output figures')
#     p.add_argument('--dpi', type=int, default=200,
#                    help='Figure DPI for saved images')
#     p.add_argument('--format', default='png',
#                    choices=['png', 'pdf', 'svg'],
#                    help='Output figure format')

#     # Display
#     p.add_argument('--patch_grid', type=int, default=24,
#                    help='CLIP patch grid size (24 for ViT-L/14 at 336px)')
#     p.add_argument('--n_vis', type=int, default=576,
#                    help='Number of visual tokens (576 for LLaVA-1.5)')

#     return p.parse_args()


# def main():
#     args = parse_args()

#     # ── Resolve directories ───────────────────────────────────
#     topn_heap_dir = os.path.join(args.data_dir, 'topn_heap')              # Top-N Heap saved here
#     act_pattern_dir = os.path.join(args.data_dir, 'act_pattern_raw')          # Activation Pattern raw acts saved here
#     os.makedirs(args.output_dir, exist_ok=True)

#     print(f'Data dir:   {args.data_dir}')
#     print(f'Top-N Heap:    {topn_heap_dir}')
#     print(f'Activation Pattern:    {act_pattern_dir}')
#     print(f'COCO imgs:  {args.coco_img_dir}')
#     print(f'Output:     {args.output_dir}')

#     # ── Load shared data ──────────────────────────────────────
#     print('\nLoading sampled_ids...')
#     sampled_ids = load_sampled_ids(topn_heap_dir)                      # list: sample_idx → image_id_str
#     print(f'  {len(sampled_ids)} samples in dataset')

#     print('Loading id_to_filename...')
#     id_to_fn = load_id_to_filename(args.detail_23k_path)            # image_id → filename
#     print(f'  {len(id_to_fn)} images in detail_23k')

#     print('Loading descriptions...')
#     descriptions = load_descriptions(args.generated_desc_path)      # image_id → text
#     print(f'  {len(descriptions)} descriptions loaded')

#     # ── Load tokenizer (CPU only, no GPU) ─────────────────────
#     print('Loading tokenizer (CPU only)...')
#     if args.model_type == 'llava-hf':
#         from transformers import AutoTokenizer                      # lightweight, no model weights
#         tokenizer = AutoTokenizer.from_pretrained(args.hf_id)
#     elif args.model_type == 'internvl':
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#             args.hf_id, trust_remote_code=True)
#     elif args.model_type == 'qwen2vl':
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(args.hf_id)       # Qwen2.5-VL tokenizer
#     elif args.model_type == 'llava-ov':
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(args.hf_id)       # Qwen2 tokenizer (from LLaVA-OV)
#     else:
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#             'liuhaotian/llava-v1.5-7b', use_fast=False)
#     print(f'  Tokenizer loaded: vocab_size={tokenizer.vocab_size}')

#     # ── Determine which neurons to visualize ──────────────────
#     if args.fig3:
#         # ── Fig3 mode: reproduce exact panels from Xu et al. ──
#         # Select per-model neuron table
#         fig3_neurons = FIG3_NEURONS_BY_MODEL.get(args.model_type, [])
#         if not fig3_neurons:
#             print(f'\nNo Figure 3 neurons defined for '
#                   f'model_type={args.model_type}.')
#             print(f'Update the corresponding FIG3_NEURONS_* list in '
#                   'visualize_neuron_activations.py once neurons '
#                   'are identified.')
#             return

#         print('\n' + '═'*60)
#         print('  FIG3 MODE: Reproducing Xu et al. Figure 3 panels (a)-(f)')
#         print('  Using exact neurons + COCO images from the paper')
#         print('═'*60)

#         required_layers = sorted(set(e['layer'] for e in fig3_neurons))
#         print(f'  Required layers: {required_layers}')
#         print(f'  Panels: {len(fig3_neurons)}')

#         has_pmbt = (args.pmbt_data_dir is not None
#                     and os.path.isdir(args.pmbt_data_dir))
#         if has_pmbt:
#             print(f'  PMBT data dir: {args.pmbt_data_dir}')
#             print(f'  Will generate FT + PMBT + comparison figures')
#         else:
#             print(f'  PMBT data dir: not provided — FT figures only')

#         ft_panel_paths = []                                             # collect FT panel paths
#         pmbt_panel_paths = []                                           # collect PMBT panel paths

#         for entry in fig3_neurons:
#             layer = entry['layer']
#             nidx = entry['neuron_idx']
#             target_coco_id = entry['coco_id']
#             label = entry['label']
#             panel = entry['panel']

#             print(f'\n{"─"*60}')
#             print(f'Panel {panel}: {entry["description"]}')
#             print(f'  layer={layer}, neuron={nidx}, '
#                   f'target image={target_coco_id}')

#             # Load Top-N Heap data for this layer
#             try:
#                 top_n_sids, top_n_acts, global_max = load_topn_heap_layer(
#                     topn_heap_dir, layer)
#             except FileNotFoundError:
#                 print(f'  ERROR: Top-N Heap data not found for layer {layer}. '
#                       f'Run full pipeline (all 32 layers) first.')
#                 continue

#             # Load Activation Pattern raw activations for this layer
#             try:
#                 vis_acts_all, txt_acts_all, txt_len_all = load_act_pattern_layer(
#                     act_pattern_dir, layer)
#             except FileNotFoundError:
#                 print(f'  ERROR: Activation Pattern data not found for layer {layer}. '
#                       f'Run full pipeline (all 32 layers) first.')
#                 continue

#             # Find the rank corresponding to the target COCO image
#             rank_idx = find_rank_for_image(
#                 top_n_sids, nidx, sampled_ids, target_coco_id)

#             if rank_idx is not None:
#                 print(f'  Found target image at rank {rank_idx}')
#             else:
#                 # Image not in this neuron's top-N — fall back to top-1
#                 print(f'  WARNING: Image {target_coco_id} not in neuron\'s '
#                       f'top-N. Falling back to rank 0 (top-1 sample).')
#                 sorted_ranks = np.argsort(-top_n_acts[nidx])
#                 rank_idx = sorted_ranks[0]

#             sample_idx = int(top_n_sids[nidx, rank_idx])
#             if sample_idx < 0:
#                 print(f'  ERROR: Invalid sample at rank {rank_idx}, skipping')
#                 continue

#             img_id = sampled_ids[sample_idx]
#             filename = id_to_fn.get(img_id)
#             if filename is None:
#                 print(f'  ERROR: No filename for image {img_id}, skipping')
#                 continue

#             img_path = os.path.join(args.coco_img_dir, filename)
#             if not os.path.exists(img_path):
#                 print(f'  ERROR: Image not found: {img_path}, skipping')
#                 continue

#             print(f'  Using: sample_idx={sample_idx}, image={img_id}, '
#                   f'file={filename}')

#             # Load image
#             img = Image.open(img_path).convert('RGB')
#             img_array = np.array(img)

#             # Extract activations (shared between FT and PMBT)
#             vis_acts = vis_acts_all[nidx, rank_idx, :]
#             txt_acts = txt_acts_all[nidx, rank_idx, :]
#             txt_len = int(txt_len_all[nidx, rank_idx])

#             # Get text tokens
#             if img_id in descriptions:
#                 tokens = get_description_tokens(
#                     tokenizer, img_id, descriptions, args.model_type)
#                 if len(tokens) != txt_len and txt_len > 0:
#                     print(f'  NOTE: token count mismatch: '
#                           f'tokenizer={len(tokens)}, saved={txt_len}')
#             else:
#                 tokens = [f'tok_{i}' for i in range(txt_len)]
#                 print(f'  WARNING: No description for {img_id}, '
#                       f'using placeholders')

#             print(f'  Visual acts: max={vis_acts.max():.1f}, '
#                   f'mean={vis_acts.mean():.2f}')
#             if txt_len > 0:
#                 print(f'  Text acts ({txt_len} tokens): '
#                       f'max={txt_acts[:txt_len].max():.1f}, '
#                       f'mean={txt_acts[:txt_len].mean():.2f}')

#             panel_tag = panel.strip('()')                              # "a", "b", etc.
#             base_name = (f'fig3_{panel_tag}_{label}_layer{layer}'
#                          f'_neuron{nidx}_{args.model_type}')

#             # ── FT panel ──
#             try:
#                 ft_labels = load_neuron_labels(
#                     args.data_dir, layer, args.model_type)
#                 ft_info = ft_labels[nidx]
#                 ft_pv, ft_pt = ft_info['pv'], ft_info['pt']
#                 ft_pm, ft_pu = ft_info['pm'], ft_info['pu']
#             except (FileNotFoundError, IndexError):
#                 ft_pv = ft_pt = ft_pm = ft_pu = 0.0
#                 print(f'  WARNING: No FT labels for layer {layer}')

#             fig_ft = create_neuron_panel(
#                 img_array=img_array, vis_acts=vis_acts,
#                 txt_acts=txt_acts, txt_len=txt_len, tokens=tokens,
#                 layer=layer, neuron_idx=nidx, label=label,
#                 pv=ft_pv, pt=ft_pt, pm=ft_pm, pu=ft_pu,
#                 patch_grid=args.patch_grid,
#             )
#             ft_path = os.path.join(
#                 args.output_dir, f'{base_name}_ft.{args.format}')
#             fig_ft.savefig(ft_path, dpi=args.dpi,
#                            bbox_inches='tight', facecolor='white')
#             plt.close(fig_ft)
#             ft_panel_paths.append(ft_path)
#             print(f'  Saved FT:   {ft_path}')

#             # ── PMBT panel (if available) ──
#             if has_pmbt:
#                 try:
#                     pmbt_labels = load_neuron_labels(
#                         args.pmbt_data_dir, layer, args.model_type)
#                     pmbt_info = pmbt_labels[nidx]
#                     pmbt_pv, pmbt_pt = pmbt_info['pv'], pmbt_info['pt']
#                     pmbt_pm, pmbt_pu = pmbt_info['pm'], pmbt_info['pu']
#                     pmbt_label = pmbt_info.get('label', label)
#                 except (FileNotFoundError, IndexError):
#                     pmbt_pv = pmbt_pt = pmbt_pm = pmbt_pu = 0.0
#                     pmbt_label = label
#                     print(f'  WARNING: No PMBT labels for layer {layer}')

#                 fig_pmbt = create_neuron_panel(
#                     img_array=img_array, vis_acts=vis_acts,
#                     txt_acts=txt_acts, txt_len=txt_len, tokens=tokens,
#                     layer=layer, neuron_idx=nidx, label=pmbt_label,
#                     pv=pmbt_pv, pt=pmbt_pt, pm=pmbt_pm, pu=pmbt_pu,
#                     patch_grid=args.patch_grid,
#                 )
#                 pmbt_path = os.path.join(
#                     args.output_dir, f'{base_name}_pmbt.{args.format}')
#                 fig_pmbt.savefig(pmbt_path, dpi=args.dpi,
#                                  bbox_inches='tight', facecolor='white')
#                 plt.close(fig_pmbt)
#                 pmbt_panel_paths.append(pmbt_path)
#                 print(f'  Saved PMBT: {pmbt_path}')

#         # ── Create combined figures ──
#         def _make_combined_grid(paths, entries, suffix, title_extra=''):
#             """Build a 2×3 grid from 6 panel image paths."""
#             if len(paths) != 6:
#                 print(f'\n  WARNING: Only {len(paths)}/6 {suffix} panels, '
#                       f'skipping combined {suffix} figure.')
#                 return None
#             print(f'\n{"─"*60}')
#             print(f'Creating combined {suffix.upper()} figure (2×3)...')
#             fig_c, axes_c = plt.subplots(2, 3, figsize=(36, 20))
#             for ax, path, entry in zip(axes_c.flatten(), paths, entries):
#                 ax.imshow(np.array(Image.open(path)))
#                 ax.set_title(
#                     f'{entry["panel"]} {entry["description"]}{title_extra}',
#                     fontsize=14, fontweight='bold')
#                 ax.axis('off')
#             fig_c.tight_layout(pad=2.0)
#             cpath = os.path.join(
#                 args.output_dir,
#                 f'fig3_combined_{args.model_type}_{suffix}.{args.format}')
#             fig_c.savefig(cpath, dpi=args.dpi,
#                           bbox_inches='tight', facecolor='white')
#             plt.close(fig_c)
#             print(f'  Saved: {cpath}')
#             return cpath

#         _make_combined_grid(ft_panel_paths, fig3_neurons, 'ft')

#         if has_pmbt:
#             _make_combined_grid(pmbt_panel_paths, fig3_neurons, 'llm_pmbt')

#             # ── Side-by-side comparison: PMBT (left) | FT (right) ──
#             if len(ft_panel_paths) == 6 and len(pmbt_panel_paths) == 6:
#                 print(f'\n{"─"*60}')
#                 print('Creating comparison figure (PMBT left | FT right)...')
#                 fig_cmp, axes_cmp = plt.subplots(
#                     2, 6, figsize=(60, 20))                            # 2 rows × 6 cols

#                 for i, entry in enumerate(fig3_neurons):
#                     row = i // 3                                       # 0 or 1
#                     col = (i % 3) * 2                                  # 0, 2, 4

#                     # Left: PMBT
#                     ax_pmbt = axes_cmp[row, col]
#                     ax_pmbt.imshow(np.array(Image.open(pmbt_panel_paths[i])))
#                     ax_pmbt.set_title(
#                         f'{entry["panel"]} PMBT', fontsize=12,
#                         fontweight='bold', color='#2196F3')
#                     ax_pmbt.axis('off')

#                     # Right: FT
#                     ax_ft = axes_cmp[row, col + 1]
#                     ax_ft.imshow(np.array(Image.open(ft_panel_paths[i])))
#                     ax_ft.set_title(
#                         f'{entry["panel"]} Fixed Threshold', fontsize=12,
#                         fontweight='bold', color='#FF9800')
#                     ax_ft.axis('off')

#                 fig_cmp.tight_layout(pad=1.5)
#                 cmp_path = os.path.join(
#                     args.output_dir,
#                     f'fig3_combined_{args.model_type}_comparison.{args.format}')
#                 fig_cmp.savefig(cmp_path, dpi=args.dpi,
#                                 bbox_inches='tight', facecolor='white')
#                 plt.close(fig_cmp)
#                 print(f'  Saved comparison: {cmp_path}')

#         print(f'\n{"═"*60}')
#         print(f'Done. Figure 3 panels saved to {args.output_dir}/')
#         print(f'{"═"*60}')
#         return                                                      # exit after fig3 mode

#     elif args.fig89 or args.supplementary:
#         # ── Fig89 / Supplementary mode: two-sample panels ─────
#         if args.fig89:
#             neuron_list = FIG89_NEURONS
#             mode_label = 'FIG89 MODE: Reproducing Xu Figures 8 & 9'
#         else:
#             neuron_list = FIG_SUPPLEMENTARY_NEURONS
#             mode_label = 'SUPPLEMENTARY MODE: Reproducing Xu Figures 15-17'

#         print('\n' + '═'*60)
#         print(f'  {mode_label}')
#         print(f'  Two-sample panels (top-2 activated images per neuron)')
#         print('═'*60)

#         required_layers = sorted(set(e['layer'] for e in neuron_list))
#         print(f'  Required layers: {required_layers}')
#         print(f'  Neurons: {len(neuron_list)}')

#         for entry in neuron_list:
#             layer = entry['layer']
#             nidx = entry['neuron_idx']
#             label = entry['label']
#             fig_tag = entry['figure']
#             explanation = entry.get('explanation', '')

#             print(f'\n{"─"*60}')
#             print(f'{fig_tag}: {label} neuron — layer {layer}, '
#                   f'neuron {nidx}')
#             if explanation:
#                 print(f'  Explanation: {explanation}')

#             # Load top-2 samples using helper
#             try:
#                 samples = load_two_samples_for_neuron(
#                     topn_heap_dir, act_pattern_dir, sampled_ids,
#                     id_to_fn, args.coco_img_dir, descriptions,
#                     tokenizer, args.model_type, layer, nidx,
#                     args.patch_grid)
#             except FileNotFoundError as e:
#                 print(f'  ERROR: Data not found for layer {layer}: {e}')
#                 continue

#             if len(samples) < 2:
#                 print(f'  WARNING: Only {len(samples)} valid sample(s), '
#                       f'need 2 — skipping')
#                 continue

#             # Create two-sample panel
#             fig = create_two_sample_panel(
#                 samples=samples,
#                 layer=layer,
#                 neuron_idx=nidx,
#                 label=label,
#                 explanation=explanation,
#                 patch_grid=args.patch_grid,
#             )

#             out_name = f'{fig_tag}_{label}_layer{layer}_neuron{nidx}.{args.format}'
#             out_path = os.path.join(args.output_dir, out_name)
#             fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
#                         facecolor='white')
#             plt.close(fig)
#             print(f'  Saved: {out_path}')

#         which = 'Figures 8-9' if args.fig89 else 'Figures 15-17'
#         print(f'\n{"═"*60}')
#         print(f'Done. {which} panels saved to {args.output_dir}/')
#         print(f'{"═"*60}')
#         return                                                      # exit after fig89/supplementary

#     elif args.layer is not None and args.neuron_idx is not None:
#         # User specified exact neuron
#         labels = load_neuron_labels(args.data_dir, args.layer, args.model_type)
#         entry = labels[args.neuron_idx]                             # dict with label, pv, pt, etc.
#         neurons_to_viz = [{
#             'layer': args.layer,
#             'neuron_idx': args.neuron_idx,
#             'label': entry['label'],
#             'pv': entry['pv'], 'pt': entry['pt'],
#             'pm': entry['pm'], 'pu': entry['pu'],
#         }]
#         print(f'\nVisualizing specific neuron: layer={args.layer}, '
#               f'idx={args.neuron_idx}, type={entry["label"]}')
#     else:
#         # Auto-select high-confidence neurons
#         print('\nAuto-selecting neurons...')
#         all_layers = list(range(args.layer_start, args.layer_end + 1))
#         neurons_to_viz = auto_select_neurons(
#             args.data_dir, all_layers, args.model_type, tuple(args.types))

#     if not neurons_to_viz:
#         print('ERROR: No neurons selected. Check data paths and layer range.')
#         return

#     # ── Generate visualizations ───────────────────────────────
#     for info in neurons_to_viz:
#         layer = info['layer']
#         nidx = info['neuron_idx']
#         label = info['label']

#         print(f'\n{"─"*60}')
#         print(f'Generating: {label} neuron — layer {layer}, neuron {nidx}')
#         print(f'  pv={info["pv"]:.3f}  pt={info["pt"]:.3f}  '
#               f'pm={info["pm"]:.3f}  pu={info["pu"]:.3f}')

#         # ── Two-sample mode (--two_samples flag) ──────────────
#         if args.two_samples:
#             try:
#                 samples = load_two_samples_for_neuron(
#                     topn_heap_dir, act_pattern_dir, sampled_ids,
#                     id_to_fn, args.coco_img_dir, descriptions,
#                     tokenizer, args.model_type, layer, nidx,
#                     args.patch_grid)
#             except FileNotFoundError as e:
#                 print(f'  ERROR: Data not found for layer {layer}: {e}')
#                 continue

#             if len(samples) < 2:
#                 print(f'  WARNING: Only {len(samples)} valid sample(s), '
#                       f'need 2 — skipping')
#                 continue

#             fig = create_two_sample_panel(
#                 samples=samples,
#                 layer=layer,
#                 neuron_idx=nidx,
#                 label=label,
#                 explanation='',
#                 patch_grid=args.patch_grid,
#             )

#             out_name = (f'{label}_layer{layer}_neuron{nidx}'
#                         f'_2samples.{args.format}')
#             out_path = os.path.join(args.output_dir, out_name)
#             fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
#                         facecolor='white')
#             plt.close(fig)
#             print(f'  Saved: {out_path}')
#             continue                                                # skip single-sample logic below

#         # ── Single-sample mode (default) ──────────────────────
#         # Load Top-N Heap data for this layer
#         top_n_sids, top_n_acts, global_max = load_topn_heap_layer(topn_heap_dir, layer)

#         # Load Activation Pattern raw activations for this layer
#         vis_acts_all, txt_acts_all, txt_len_all = load_act_pattern_layer(act_pattern_dir, layer)

#         # Sort ranks by activation value to find the true top-1
#         neuron_acts = top_n_acts[nidx]                              # (top_n,) raw activation values
#         sorted_ranks = np.argsort(-neuron_acts)                     # descending by activation
#         rank = min(args.rank, len(sorted_ranks) - 1)                # requested rank (default 0 = top)
#         best_rank = sorted_ranks[rank]                              # index into the top_n arrays

#         sample_idx = int(top_n_sids[nidx, best_rank])              # sample index into sampled_ids
#         if sample_idx < 0:
#             print(f'  WARNING: No valid sample at rank {rank}, skipping')
#             continue

#         img_id = sampled_ids[sample_idx]                            # COCO image ID string
#         filename = id_to_fn.get(img_id)
#         if filename is None:
#             print(f'  WARNING: No filename for image {img_id}, skipping')
#             continue

#         img_path = os.path.join(args.coco_img_dir, filename)
#         if not os.path.exists(img_path):
#             print(f'  WARNING: Image not found: {img_path}, skipping')
#             continue

#         print(f'  Sample: rank={rank}, sample_idx={sample_idx}, '
#               f'image={img_id}, file={filename}')

#         # Load image
#         img = Image.open(img_path).convert('RGB')
#         img_array = np.array(img)                                   # (H, W, 3) uint8

#         # Extract activations for this neuron + sample
#         vis_acts = vis_acts_all[nidx, best_rank, :]                 # (576,)
#         txt_acts = txt_acts_all[nidx, best_rank, :]                 # (MAX_TXT,)
#         txt_len = int(txt_len_all[nidx, best_rank])                 # actual text token count

#         # Get text tokens by re-tokenizing the description
#         if img_id in descriptions:
#             tokens = get_description_tokens(
#                 tokenizer, img_id, descriptions, args.model_type)
#             # Align token count with saved txt_len
#             # (minor differences possible due to tokenizer version)
#             if len(tokens) != txt_len and txt_len > 0:
#                 print(f'  NOTE: token count mismatch: '
#                       f'tokenizer={len(tokens)}, saved={txt_len}. '
#                       f'Using min({len(tokens)}, {txt_len}) tokens.')
#         else:
#             tokens = [f'tok_{i}' for i in range(txt_len)]           # fallback placeholder
#             print(f'  WARNING: No description for {img_id}, using placeholders')

#         print(f'  Visual acts: max={vis_acts.max():.1f}, '
#               f'mean={vis_acts.mean():.2f}')
#         print(f'  Text acts ({txt_len} tokens): max={txt_acts[:txt_len].max():.1f}, '
#               f'mean={txt_acts[:txt_len].mean():.2f}')

#         # Create the figure
#         fig = create_neuron_panel(
#             img_array=img_array,
#             vis_acts=vis_acts,
#             txt_acts=txt_acts,
#             txt_len=txt_len,
#             tokens=tokens,
#             layer=layer,
#             neuron_idx=nidx,
#             label=label,
#             pv=info['pv'], pt=info['pt'],
#             pm=info['pm'], pu=info['pu'],
#             patch_grid=args.patch_grid,
#         )

#         # Save
#         out_name = f'{label}_layer{layer}_neuron{nidx}_rank{rank}.{args.format}'
#         out_path = os.path.join(args.output_dir, out_name)
#         fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
#                     facecolor='white')
#         plt.close(fig)
#         print(f'  Saved: {out_path}')

#     print(f'\n{"═"*60}')
#     print(f'Done. Figures saved to {args.output_dir}/')
#     print(f'{"═"*60}')


# if __name__ == '__main__':
#     main()