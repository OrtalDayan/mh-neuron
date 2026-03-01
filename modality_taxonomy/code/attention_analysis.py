"""
attention_analysis.py — Analyze attention patterns for reclassified neurons.

Hypothesis: Text tokens highlighted by Xu's threshold method (e.g., "hot",
"white", "blue", "tie") attend heavily to visual patches, meaning their
activations reflect leaked visual information — not genuine text processing.

Method:
  1. Load LLaVA (HF or original backend) with output_attentions=True
  2. Run teacher-forcing forward pass (same as classification pipeline)
  3. For each layer, compute what fraction of each text token's attention
     goes to visual patches vs other text tokens
  4. Compare highlighted tokens (Xu's false positives) vs non-highlighted

Outputs:
  - Heatmap: attention from highlighted text tokens to 24×24 visual grid
  - Bar chart: highlighted vs non-highlighted visual attention per layer
  - Summary table printed to stdout

Supports three model backends:
  --model_type hf         : llava-hf/llava-1.5-7b-hf (default)
  --model_type liuhaotian : liuhaotian/llava-v1.5-7b
  --model_type llava-ov   : llava-hf/llava-onevision-qwen2-7b-ov-hf

Usage:
    python attention_analysis.py \\
        --image_id 000000189475 \\
        --highlighted_words hot white blue tie \\
        --model_type liuhaotian \\
        --device 0

    # LLaVA-OneVision
    python attention_analysis.py \\
        --image_id 000000156852 \\
        --highlighted_words hot white blue tie \\
        --model_type llava-ov \\
        --device 0

    # Or provide a COCO image path directly
    python attention_analysis.py \\
        --image_path /path/to/image.jpg \\
        --highlighted_words hot white blue tie \\
        --device 0
"""

import argparse                                        # Command-line argument parsing
import json                                            # JSON file loading
import os                                              # File path operations
import sys                                             # Path manipulation

# ── Project path setup (same as patch_fig3_activations.py) ────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # Directory of this script
_PROJECT_ROOT = os.path.abspath(os.path.join(                # Navigate up to project root
    _SCRIPT_DIR, '..', '..'))
_LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')          # Path to cloned LLaVA repo
if _LLAVA_PATH not in sys.path:                              # Add LLaVA to Python path
    sys.path.insert(0, _LLAVA_PATH)                          # (needed for liuhaotian backend)

import numpy as np                                     # Numerical array operations
import torch                                           # PyTorch tensor operations
import matplotlib.pyplot as plt                        # Plotting library
from PIL import Image                                  # Image loading


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Self-contained helpers (from existing pipeline)
# ═══════════════════════════════════════════════════════════════════

def load_generated_descriptions(desc_path, detail_23k_path):
    """Load LLaVA-generated descriptions and detail_23k image list.

    Copied from patch_fig3_activations.py to avoid circular imports.
    Handles both dict-of-dicts and dict-of-strings formats.
    """
    with open(desc_path) as f:                                 # Open the descriptions JSON
        raw = json.load(f)                                     # Parse into Python dict
    descriptions = {}                                          # Will hold id → text mapping
    for k, v in raw.items():                                   # Iterate over entries
        if isinstance(v, dict):                                # Some entries are {text: ..., ...}
            descriptions[k] = v['text']                        # Extract the text field
        else:
            descriptions[k] = v                                # Already a plain string
    with open(detail_23k_path) as f:                           # Open image list JSON
        detail_data = json.load(f)                             # Parse the list of image dicts
    image_ids = []                                             # Ordered list of image IDs
    id_to_filename = {}                                        # Map: image_id → filename
    for item in detail_data:                                   # Iterate over images
        img_id = item['id']                                    # Extract the image identifier
        fname = os.path.basename(item['image'])                # Extract just the filename
        image_ids.append(img_id)                               # Add to ordered list
        id_to_filename[img_id] = fname                         # Add to lookup dict
    return descriptions, image_ids, id_to_filename


def build_prompt(img_id, model_type='hf', descriptions=None, processor=None):
    """Build teacher-forcing prompt for a given image.

    Same prompt construction as neuron_modality_statistical.py:
    the full generated description is appended as the ASSISTANT response,
    so the model processes it in a single forward pass (no generation).
    """
    text = descriptions[img_id]                                # Get the pre-generated description
    if model_type == 'llava-ov':                                     # LLaVA-OneVision uses chat template
        conversation = [                                             # Build conversation in HF format
            {'role': 'user', 'content': [                            # User turn with image + question
                {'type': 'image'},
                {'type': 'text', 'text': 'Could you describe the image in detail?'},
            ]},
            {'role': 'assistant', 'content': [                       # Assistant turn (teacher forcing)
                {'type': 'text', 'text': text},
            ]},
        ]
        return processor.apply_chat_template(                        # Format via Qwen2 chat template
            conversation, add_generation_prompt=False)
    elif model_type == 'hf':                                         # HF uses simple string template
        return (f"USER: <image>\n"                             # User turn with image placeholder
                f"Could you describe the image?\n"             # Question prompt
                f"ASSISTANT: {text}")                           # Pre-filled assistant response
    else:                                                      # Original LLaVA uses conv_templates
        from llava.conversation import conv_templates          # Import conversation formatter
        conv = conv_templates["v1"].copy()                     # LLaVA-v1.5 uses "v1" template
        conv.append_message(conv.roles[0],                     # USER turn
                            "<image>\nCould you describe the image?")
        conv.append_message(conv.roles[1], text)               # ASSISTANT turn (teacher forcing)
        return conv.get_prompt()                               # Format into final string


def count_description_tokens(img_id, model_type, processor, descriptions):
    """Count how many tokens the generated description occupies.

    Same as in neuron_modality_statistical.py: tokenize full prompt
    minus template prefix = number of description-only tokens.
    """
    desc_text = descriptions[img_id]                           # Get description text
    if model_type == 'llava-ov':                                     # LLaVA-OneVision: use processor
        tokenizer = processor.tokenizer                              # Access underlying tokenizer
        # Build prefix (user turn only) and full (user + assistant) via chat template
        conv_prefix = [                                              # Conversation without description
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Could you describe the image in detail?'},
            ]},
        ]
        conv_full = [                                                # Conversation with description
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Could you describe the image in detail?'},
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': desc_text},
            ]},
        ]
        prefix = processor.apply_chat_template(conv_prefix, add_generation_prompt=True)
        full = processor.apply_chat_template(conv_full, add_generation_prompt=False)
    elif model_type == 'hf':                                     # HF processor wraps a tokenizer
        tokenizer = processor.tokenizer                        # Access the underlying tokenizer
        prefix = "USER: <image>\nCould you describe the image?\nASSISTANT:"
        full = f"USER: <image>\nCould you describe the image?\nASSISTANT: {desc_text}"
    else:                                                      # Original LLaVA backend
        from llava.conversation import conv_templates          # Import for template formatting
        tokenizer = processor[0]                               # Unpack (tokenizer, image_processor)
        conv_p = conv_templates["v1"].copy()                   # Template without assistant text
        conv_p.append_message(conv_p.roles[0],
                              "<image>\nCould you describe the image?")
        conv_p.append_message(conv_p.roles[1], None)           # Empty assistant turn
        prefix = conv_p.get_prompt()
        conv_f = conv_templates["v1"].copy()                   # Template with full description
        conv_f.append_message(conv_f.roles[0],
                              "<image>\nCould you describe the image?")
        conv_f.append_message(conv_f.roles[1], desc_text)      # Full assistant response
        full = conv_f.get_prompt()
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)   # Tokenize prefix only
    full_ids = tokenizer.encode(full, add_special_tokens=True)       # Tokenize full prompt
    return len(full_ids) - len(prefix_ids)                           # Difference = description tokens


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Model loading (from existing pipeline)
# ═══════════════════════════════════════════════════════════════════

def load_model_hf(hf_id, device):
    """Load the HuggingFace LLaVA model.

    Uses attn_implementation='eager' so attention weights are returned
    when output_attentions=True is passed to the forward call.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(hf_id)           # Text + image preprocessing
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,                             # Half precision for memory
        low_cpu_mem_usage=True,                                # Reduce RAM during loading
        attn_implementation='eager',                           # Required for attention output
    ).to(device).eval()                                        # Move to GPU, set eval mode
    image_token_id = model.config.image_token_index            # Token ID for <image> (e.g. 32000)
    return model, processor, image_token_id


def load_model_original(model_path, device):
    """Load the original LLaVA model via the LLaVA repo.

    The original model uses LlamaForCausalLM internals, which support
    output_attentions natively through HuggingFace's modeling_llama.py.
    """
    from llava.model.builder import load_pretrained_model      # LLaVA's model loader
    from llava.mm_utils import get_model_name_from_path        # Derive name from path
    from llava.constants import IMAGE_TOKEN_INDEX              # Special token ID (-200)

    model_name = get_model_name_from_path(model_path)          # e.g. "llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=device,
        torch_dtype=torch.float16
    )
    model.eval()                                               # Set to evaluation mode

    # Patch attention implementation to eager if using SDPA (SDPA doesn't return weights)
    llm = model.model if hasattr(model, 'model') else model    # Access the underlying LlamaModel
    if hasattr(llm, 'config'):                                 # Check if config is accessible
        llm.config._attn_implementation = 'eager'              # Force eager attention

    processor = (tokenizer, image_processor)                   # Pack into tuple like pipeline does
    return model, processor, IMAGE_TOKEN_INDEX


def load_model_llava_ov(hf_id, device):
    """Load LLaVA-OneVision model with eager attention for attention extraction.

    Uses attn_implementation='eager' so that attention weights are computed
    and returned when output_attentions=True is passed to forward().
    Flash/SDPA attention does not support returning attention weights.
    """
    from transformers import (                                       # Import HF model classes
        AutoProcessor,
        LlavaOnevisionForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(hf_id)                 # Load tokenizer + image processor

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,                                   # Half precision to save GPU memory
        low_cpu_mem_usage=True,                                      # Avoid full CPU copy before GPU transfer
        attn_implementation='eager',                                 # Required for output_attentions=True
    ).to(device).eval()                                              # Move to GPU and set eval mode

    image_token_id = model.config.image_token_index                  # 151646 for llava-onevision-qwen2
    return model, processor, image_token_id


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Input preparation (from existing pipeline)
# ═══════════════════════════════════════════════════════════════════

def prepare_inputs_hf(processor, img, text, device, image_token_id=32000):
    """Prepare model inputs for HF LLaVA.

    Returns: (inputs_dict, visual_mask, expanded_tokens)
    """
    inputs = processor(images=img, text=text,
                       return_tensors='pt').to(device)         # Tokenize + preprocess image
    input_ids = inputs['input_ids'][0].cpu()                   # (seq_len,) on CPU for analysis

    # In HF, <image> is expanded to 576 repeated image_token_id tokens
    visual_mask = (input_ids.numpy() == image_token_id)        # Bool mask: True at image positions

    # Decode each token for display purposes
    tokens = processor.tokenizer.convert_ids_to_tokens(
        input_ids.tolist())                                    # List of token strings

    return inputs, visual_mask, tokens


def prepare_inputs_original(processor, img, text, device, image_token_id):
    """Prepare model inputs for original LLaVA.

    The original model replaces the single IMAGE_TOKEN_INDEX placeholder
    with 576 image features during forward pass, so the expanded sequence
    is longer than the tokenized input.
    """
    from llava.constants import IMAGE_TOKEN_INDEX              # -200
    from llava.mm_utils import tokenizer_image_token           # Tokenize with image placeholder

    tokenizer, image_processor = processor                     # Unpack tuple

    # Tokenize text with <image> placeholder
    input_ids = tokenizer_image_token(
        text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)                                  # (1, seq_len) on GPU

    # Preprocess image via CLIP processor
    image_tensor = image_processor.preprocess(
        img, return_tensors='pt'
    )['pixel_values'].half().to(device)                        # (1, 3, 336, 336)

    # Build visual mask for the EXPANDED sequence
    ids_cpu = input_ids[0].cpu().numpy()                       # Token IDs on CPU
    img_pos = int(np.where(ids_cpu == IMAGE_TOKEN_INDEX)[0][0])  # Position of <image>
    n_image_features = 576                                     # LLaVA uses 576 CLIP patches
    expanded_len = len(ids_cpu) - 1 + n_image_features         # -1 placeholder + 576 features
    visual_mask = np.zeros(expanded_len, dtype=bool)           # Initialize all False
    visual_mask[img_pos:img_pos + n_image_features] = True     # Mark image feature positions

    # Build expanded token list for display
    tokens = tokenizer.convert_ids_to_tokens(ids_cpu.tolist()) # Tokenize for labels
    expanded_tokens = []                                       # Will match expanded_len
    for i, tok in enumerate(tokens):                           # Replace <image> with patches
        if ids_cpu[i] == IMAGE_TOKEN_INDEX:
            expanded_tokens.extend(
                [f'<vis_{j}>' for j in range(n_image_features)])  # 576 visual placeholders
        else:
            expanded_tokens.append(tok)                        # Keep text token as-is

    inputs_dict = {'input_ids': input_ids, 'images': image_tensor}
    return inputs_dict, visual_mask, expanded_tokens


def prepare_inputs_llava_ov(processor, img, text, device, image_token_id):
    """Prepare model inputs for LLaVA-OneVision.

    The processor tokenizes the prompt and preprocesses the image in one call.
    The image placeholder token is expanded to N image_token_id tokens
    (one per visual patch from SigLIP), similar to how the HF LLaVA backend
    expands to 576 tokens but with a variable patch count.

    Returns: (inputs_dict, visual_mask, expanded_tokens)
    """
    inputs = processor(                                              # Tokenize text + preprocess image
        images=img,                                                  # PIL image
        text=text,                                                   # Formatted prompt string
        return_tensors='pt',                                         # Return PyTorch tensors
    ).to(device)                                                     # Move all tensors to GPU

    input_ids = inputs['input_ids'][0].cpu()                         # (seq_len,) on CPU for analysis

    # Build visual mask: True where input_ids == image_token_id
    # The processor expands the <image> placeholder into N repeated
    # image_token_id tokens (one per visual patch after SigLIP encoding)
    visual_mask = (input_ids.numpy() == image_token_id)              # Bool array: True at image positions

    # Decode each token for display / matching purposes
    expanded_tokens = processor.tokenizer.convert_ids_to_tokens(     # Convert IDs → token strings
        input_ids.tolist()
    )

    return inputs, visual_mask, expanded_tokens


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Attention extraction and analysis
# ═══════════════════════════════════════════════════════════════════

def forward_with_attention(model, inputs, model_type):
    """Run teacher-forcing forward pass with output_attentions=True.

    Returns list of (num_heads, seq_len, seq_len) numpy arrays, one per layer.
    """
    with torch.no_grad():                                      # No gradient computation
        outputs = model(**inputs, output_attentions=True)      # Forward pass with attention output

    # outputs.attentions is a tuple of num_layers tensors
    # Each tensor: (batch=1, num_heads, seq_len, seq_len)
    attn_per_layer = []                                        # Will store per-layer attention
    for layer_attn in outputs.attentions:                      # Iterate over layers
        attn = layer_attn.squeeze(0)                           # Remove batch dim → (heads, S, S)
        attn = attn.float().cpu().numpy()                      # Convert to float32 numpy
        attn_per_layer.append(attn)                            # Append to list
    return attn_per_layer


def find_highlighted_tokens(expanded_tokens, desc_start_idx, highlighted_words):
    """Identify which description token positions match the highlighted words.

    Only searches within the description portion of the sequence
    (from desc_start_idx to end), since template tokens are not relevant.

    Returns: (highlighted_indices, non_highlighted_indices)
    """
    highlighted_indices = []                                   # Positions matching highlighted words
    non_highlighted_indices = []                                # All other description token positions

    for idx in range(desc_start_idx, len(expanded_tokens)):    # Only look at description tokens
        tok = expanded_tokens[idx]                             # Get the token string
        tok_clean = tok.lower().replace('▁', '').replace('Ġ', '')  # Normalize (sentencepiece/BPE)

        if not tok_clean.strip():                              # Skip empty/whitespace tokens
            continue

        # Check if token matches any highlighted word
        is_match = any(hw.lower() in tok_clean                 # Substring match
                       for hw in highlighted_words)

        if is_match:
            highlighted_indices.append(idx)                    # Add to highlighted set
        else:
            non_highlighted_indices.append(idx)                # Add to non-highlighted set

    return highlighted_indices, non_highlighted_indices


def compute_attention_stats(attn_per_layer, visual_indices, text_indices,
                            highlighted_indices, non_highlighted_indices):
    """Compute per-layer attention statistics.

    For highlighted and non-highlighted text tokens, computes the fraction
    of their total attention that is directed at visual patch positions.
    """
    results = {
        'highlighted_to_visual': [],                           # Per-layer mean attention to visual
        'non_highlighted_to_visual': [],                       # Per-layer mean attention to visual
        'highlighted_to_text': [],                             # Per-layer mean attention to text
        'non_highlighted_to_text': [],                         # Per-layer mean attention to text
    }

    for layer_idx, attn in enumerate(attn_per_layer):          # Iterate over layers
        head_avg = attn.mean(axis=0)                           # Average over heads → (S, S)

        # Highlighted tokens → visual patches
        if len(highlighted_indices) > 0:                       # Only compute if tokens exist
            h_to_vis = head_avg[highlighted_indices][:, visual_indices].sum(axis=1).mean()
            h_to_txt = head_avg[highlighted_indices][:, text_indices].sum(axis=1).mean()
        else:
            h_to_vis = 0.0                                     # No highlighted tokens found
            h_to_txt = 0.0

        # Non-highlighted tokens → visual patches
        if len(non_highlighted_indices) > 0:                   # Only compute if tokens exist
            nh_to_vis = head_avg[non_highlighted_indices][:, visual_indices].sum(axis=1).mean()
            nh_to_txt = head_avg[non_highlighted_indices][:, text_indices].sum(axis=1).mean()
        else:
            nh_to_vis = 0.0                                    # No non-highlighted tokens found
            nh_to_txt = 0.0

        results['highlighted_to_visual'].append(float(h_to_vis))
        results['non_highlighted_to_visual'].append(float(nh_to_vis))
        results['highlighted_to_text'].append(float(h_to_txt))
        results['non_highlighted_to_text'].append(float(nh_to_txt))

    return results


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_heatmap(attn_per_layer, visual_indices, highlighted_indices,
                 expanded_tokens, target_layers, save_path):
    """Plot heatmap: attention from highlighted text tokens to 24×24 visual grid.

    Each row is one highlighted token, each column is one target layer.
    The 576 visual attention values are reshaped to a 24×24 grid matching
    the ViT patch layout.
    """
    if len(highlighted_indices) == 0:                          # Guard against empty input
        print('No highlighted tokens found — skipping heatmap.')
        return

    # Filter to layers that exist in the data
    target_layers = [l for l in target_layers if l < len(attn_per_layer)]
    n_layers = len(target_layers)                              # Number of columns
    n_tokens = len(highlighted_indices)                        # Number of rows

    fig, axes = plt.subplots(
        n_tokens, n_layers,
        figsize=(3 * n_layers, 3 * n_tokens),                 # Scale figure size
        squeeze=False                                          # Always return 2D array of axes
    )

    for col, layer_idx in enumerate(target_layers):            # Iterate over selected layers
        head_avg = attn_per_layer[layer_idx].mean(axis=0)      # Average over heads → (S, S)

        for row, tok_idx in enumerate(highlighted_indices):    # Iterate over highlighted tokens
            attn_to_visual = head_avg[tok_idx, visual_indices] # Attention to all visual patches
            n_vis = len(visual_indices)                        # Number of visual patches
            side = int(np.sqrt(n_vis))                         # Approximate square grid side
            if side * side == n_vis:                            # Perfect square (e.g. 576=24×24)
                attn_map = attn_to_visual.reshape(side, side)  # Reshape to 2D grid
            else:
                attn_map = attn_to_visual.reshape(1, -1)       # Fallback: single row

            ax = axes[row, col]                                # Get the subplot axis
            ax.imshow(attn_map, cmap='hot', interpolation='nearest')  # Heatmap

            if col == 0:                                       # Label rows with token strings
                label = expanded_tokens[tok_idx] if tok_idx < len(expanded_tokens) else '?'
                ax.set_ylabel(label.replace('▁', '').replace('Ġ', ''),
                              fontsize=10, fontweight='bold')
            if row == 0:                                       # Label columns with layer numbers
                ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')

            ax.set_xticks([])                                  # Hide tick marks
            ax.set_yticks([])

    fig.suptitle(
        'Attention from highlighted text tokens to visual patches (24×24)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Heatmap saved to: {save_path}')


def plot_bar_chart(results, save_path):
    """Bar chart comparing visual attention for highlighted vs non-highlighted tokens.

    X-axis: layer index (0-31), Y-axis: fraction of attention to visual patches.
    Two bars per layer: red = highlighted tokens, blue = non-highlighted.
    """
    n_layers = len(results['highlighted_to_visual'])           # Number of layers
    layers = np.arange(n_layers)                               # Layer indices for x-axis
    bar_width = 0.35                                           # Width of each bar

    fig, ax = plt.subplots(figsize=(14, 5))                    # Wide figure for 32 layers

    ax.bar(layers - bar_width / 2,                             # Left-shifted bars
           results['highlighted_to_visual'],
           bar_width,
           label='Highlighted tokens → visual',
           color='#d62728', alpha=0.85)                        # Red

    ax.bar(layers + bar_width / 2,                             # Right-shifted bars
           results['non_highlighted_to_visual'],
           bar_width,
           label='Non-highlighted tokens → visual',
           color='#1f77b4', alpha=0.85)                        # Blue

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Fraction of attention to visual patches', fontsize=12)
    ax.set_title(
        'Attention to visual patches: highlighted vs non-highlighted text tokens',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved to: {save_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Argument parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Attention analysis for reclassified neurons')

    # Image source — provide either image_id (looks up from pipeline data) or image_path
    p.add_argument('--image_id', type=str, default=None,
                   help='COCO image ID (e.g. 000000189475). Looks up filename '
                        'from detail_23k.json and loads from --coco_img_dir')
    p.add_argument('--image_path', type=str, default=None,
                   help='Direct path to an image file (overrides --image_id)')

    # Words to highlight (the ones Xu's method incorrectly flagged)
    p.add_argument('--highlighted_words', type=str, nargs='+',
                   default=['hot', 'white', 'blue', 'tie'],
                   help='Words that Xu highlighted as text-activated '
                        '(default: hot white blue tie)')

    # Model settings — same as rest of pipeline
    p.add_argument('--model_type', default='hf', choices=['hf', 'liuhaotian', 'llava-ov'],
                   help='"hf" for llava-hf/llava-1.5-7b-hf, '
                        '"liuhaotian" for liuhaotian/llava-v1.5-7b, '
                        '"llava-ov" for llava-hf/llava-onevision-qwen2-7b-ov-hf')
    p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
                   help='HuggingFace model ID (for --model_type hf)')
    p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b',
                   help='Original LLaVA model path (for --model_type liuhaotian)')
    p.add_argument('--device', default='0',
                   help='GPU device index (default: 0)')

    # Data paths — same defaults as rest of pipeline
    p.add_argument('--coco_img_dir',
                   default='/home/projects/bagon/shared/coco2017/images/train2017/')
    p.add_argument('--generated_desc_path',
                   default='results/generated_descriptions/generated_descriptions.json',
                   help='Path to LLaVA-generated descriptions JSON')
    _project_root = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
    p.add_argument('--detail_23k_path',
                   default=os.path.join(_project_root, 'neuron_taxonomy',
                                        'data', 'detail_23k.json'),
                   help='Path to detail_23k.json')

    # Output settings
    p.add_argument('--output_dir', type=str,
                   default='results/attention_analysis',
                   help='Directory to save output figures')
    p.add_argument('--heatmap_layers', type=int, nargs='+',
                   default=[0, 7, 15, 23, 28, 31],
                   help='Layers to show in heatmap (default: 0 7 15 23 28 31)')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    os.makedirs(args.output_dir, exist_ok=True)                # Create output directory

    # ── Validate image source ─────────────────────────────────
    if args.image_path is None and args.image_id is None:      # Must provide one
        print('ERROR: provide either --image_path or --image_id')
        sys.exit(1)

    # ── Load descriptions + image list ────────────────────────
    print('Loading descriptions and image list...')
    descriptions, image_ids, id_to_filename = load_generated_descriptions(
        args.generated_desc_path, args.detail_23k_path)
    print(f'  {len(descriptions)} descriptions loaded')

    # ── Resolve image path and image ID ───────────────────────
    if args.image_path is not None:                            # Direct path provided
        img_path = args.image_path
        # Try to find matching image_id from filename
        fname = os.path.basename(img_path)
        args.image_id = None
        for iid, fn in id_to_filename.items():                # Reverse lookup
            if fn == fname:
                args.image_id = iid
                break
        if args.image_id is None:                              # Couldn't match — fall back
            print(f'WARNING: could not find image_id for {fname}.')
            print('         Using first available description for prompt.')
            args.image_id = image_ids[0]                       # Fallback (won't match perfectly)
    else:                                                      # image_id provided
        fname = id_to_filename.get(args.image_id)
        if fname is None:
            print(f'ERROR: image_id {args.image_id} not found in detail_23k.json')
            sys.exit(1)
        img_path = os.path.join(args.coco_img_dir, fname)

    if not os.path.exists(img_path):                           # Check file exists
        print(f'ERROR: image not found at {img_path}')
        sys.exit(1)

    print(f'  Image: {img_path}')
    print(f'  Image ID: {args.image_id}')
    print(f'  Description: {descriptions[args.image_id][:80]}...')

    # ── Load model ────────────────────────────────────────────
    if args.model_type == 'llava-ov':
        print(f'\nLoading LLaVA-OneVision model: {args.hf_id} on {device}...')
        model, processor, image_token_id = load_model_llava_ov(args.hf_id, device)
    elif args.model_type == 'hf':
        print(f'\nLoading HF model: {args.hf_id} on {device}...')
        model, processor, image_token_id = load_model_hf(args.hf_id, device)
    else:
        print(f'\nLoading original model: {args.original_model_path} on {device}...')
        model, processor, image_token_id = load_model_original(
            args.original_model_path, device)
    print('Model loaded.')

    # ── Prepare inputs (teacher-forcing, same as pipeline) ────
    print('\nPreparing inputs (teacher-forcing mode)...')
    img = Image.open(img_path).convert('RGB')                  # Load image as RGB
    prompt = build_prompt(args.image_id, model_type=args.model_type,
                          descriptions=descriptions, processor=processor)

    if args.model_type == 'llava-ov':
        inputs, visual_mask, expanded_tokens = prepare_inputs_llava_ov(
            processor, img, prompt, device, image_token_id)
    elif args.model_type == 'hf':
        inputs, visual_mask, expanded_tokens = prepare_inputs_hf(
            processor, img, prompt, device, image_token_id)
    else:
        inputs, visual_mask, expanded_tokens = prepare_inputs_original(
            processor, img, prompt, device, image_token_id)

    seq_len = len(visual_mask)                                 # Total sequence length
    visual_indices = np.where(visual_mask)[0].tolist()         # Positions of visual tokens
    text_mask = ~visual_mask                                   # Invert to get text positions
    all_text_indices = np.where(text_mask)[0].tolist()         # Positions of all text tokens

    # Identify description tokens (at the end of the sequence)
    desc_count = count_description_tokens(
        args.image_id, args.model_type, processor, descriptions)
    desc_start_idx = seq_len - desc_count                      # Description starts here

    print(f'  Sequence length: {seq_len}')
    print(f'  Visual patches: {len(visual_indices)}')
    print(f'  Total text tokens: {len(all_text_indices)}')
    print(f'  Description tokens: {desc_count} (starting at position {desc_start_idx})')

    # ── Find highlighted vs non-highlighted tokens ────────────
    highlighted_idx, non_highlighted_idx = find_highlighted_tokens(
        expanded_tokens, desc_start_idx, args.highlighted_words)

    print(f'\n  Highlighted tokens ({len(highlighted_idx)}):')
    for idx in highlighted_idx:                                # Show which tokens matched
        tok = expanded_tokens[idx] if idx < len(expanded_tokens) else '?'
        print(f'    pos {idx}: "{tok}"')
    print(f'  Non-highlighted description tokens: {len(non_highlighted_idx)}')

    # ── Forward pass with attention ───────────────────────────
    print('\nRunning forward pass with output_attentions=True...')
    attn_per_layer = forward_with_attention(model, inputs, args.model_type)
    print(f'  Extracted attention from {len(attn_per_layer)} layers')
    print(f'  Attention shape per layer: {attn_per_layer[0].shape}')

    # ── Compute statistics ────────────────────────────────────
    print('\nComputing attention statistics...')
    results = compute_attention_stats(
        attn_per_layer, visual_indices, all_text_indices,
        highlighted_idx, non_highlighted_idx)

    # Print summary table
    print(f'\n{"─"*75}')
    print(f'{"Layer":<8} {"Highlighted→Vis":<18} {"NonHighl→Vis":<18} {"Δ (H − NH)":<14} {"Ratio":<10}')
    print(f'{"─"*75}')
    for i in range(len(results['highlighted_to_visual'])):
        h = results['highlighted_to_visual'][i]
        nh = results['non_highlighted_to_visual'][i]
        diff = h - nh
        ratio = h / nh if nh > 1e-9 else float('inf')
        print(f'{i:<8} {h:<18.4f} {nh:<18.4f} {diff:<14.4f} {ratio:<10.2f}')

    # Overall average across layers
    avg_h = np.mean(results['highlighted_to_visual'])
    avg_nh = np.mean(results['non_highlighted_to_visual'])
    print(f'{"─"*75}')
    print(f'{"Avg":<8} {avg_h:<18.4f} {avg_nh:<18.4f} '
          f'{avg_h - avg_nh:<14.4f} {avg_h / avg_nh if avg_nh > 1e-9 else float("inf"):<10.2f}')

    # ── Save numerical results ────────────────────────────────
    np.savez(
        os.path.join(args.output_dir, 'attention_stats.npz'),
        highlighted_to_visual=results['highlighted_to_visual'],
        non_highlighted_to_visual=results['non_highlighted_to_visual'],
        highlighted_to_text=results['highlighted_to_text'],
        non_highlighted_to_text=results['non_highlighted_to_text'],
        highlighted_tokens=[expanded_tokens[i] for i in highlighted_idx],
        highlighted_words=args.highlighted_words,
        image_id=args.image_id,
    )
    print(f'\nNumerical results saved to: {args.output_dir}/attention_stats.npz')

    # ── Plot heatmap ──────────────────────────────────────────
    print('\nGenerating heatmap...')
    heatmap_path = os.path.join(args.output_dir, 'attention_heatmap.png')
    plot_heatmap(
        attn_per_layer, visual_indices, highlighted_idx,
        expanded_tokens, args.heatmap_layers, heatmap_path)

    # ── Plot bar chart ────────────────────────────────────────
    print('Generating bar chart...')
    bar_path = os.path.join(args.output_dir, 'attention_bar_chart.png')
    plot_bar_chart(results, bar_path)

    print(f'\nDone! All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()



# """
# attention_analysis.py — Analyze attention patterns for reclassified neurons.

# Hypothesis: Text tokens highlighted by Xu's threshold method (e.g., "hot",
# "white", "blue", "tie") attend heavily to visual patches, meaning their
# activations reflect leaked visual information — not genuine text processing.

# Method:
#   1. Load LLaVA (HF or original backend) with output_attentions=True
#   2. Run teacher-forcing forward pass (same as classification pipeline)
#   3. For each layer, compute what fraction of each text token's attention
#      goes to visual patches vs other text tokens
#   4. Compare highlighted tokens (Xu's false positives) vs non-highlighted

# Outputs:
#   - Heatmap: attention from highlighted text tokens to 24×24 visual grid
#   - Bar chart: highlighted vs non-highlighted visual attention per layer
#   - Summary table printed to stdout

# Supports both model backends:
#   --model_type hf         : llava-hf/llava-1.5-7b-hf (default)
#   --model_type liuhaotian : liuhaotian/llava-v1.5-7b

# Usage:
#     python attention_analysis.py \\
#         --image_id 000000189475 \\
#         --highlighted_words hot white blue tie \\
#         --model_type liuhaotian \\
#         --device 0

#     # Or provide a COCO image path directly
#     python attention_analysis.py \\
#         --image_path /path/to/image.jpg \\
#         --highlighted_words hot white blue tie \\
#         --device 0
# """

# import argparse                                        # Command-line argument parsing
# import json                                            # JSON file loading
# import os                                              # File path operations
# import sys                                             # Path manipulation

# # ── Project path setup (same as patch_fig3_activations.py) ────
# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # Directory of this script
# _PROJECT_ROOT = os.path.abspath(os.path.join(                # Navigate up to project root
#     _SCRIPT_DIR, '..', '..'))
# _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')          # Path to cloned LLaVA repo
# if _LLAVA_PATH not in sys.path:                              # Add LLaVA to Python path
#     sys.path.insert(0, _LLAVA_PATH)                          # (needed for liuhaotian backend)

# import numpy as np                                     # Numerical array operations
# import torch                                           # PyTorch tensor operations
# import matplotlib.pyplot as plt                        # Plotting library
# from PIL import Image                                  # Image loading


# # ═══════════════════════════════════════════════════════════════════
# # Section 1 — Self-contained helpers (from existing pipeline)
# # ═══════════════════════════════════════════════════════════════════

# def load_generated_descriptions(desc_path, detail_23k_path):
#     """Load LLaVA-generated descriptions and detail_23k image list.

#     Copied from patch_fig3_activations.py to avoid circular imports.
#     Handles both dict-of-dicts and dict-of-strings formats.
#     """
#     with open(desc_path) as f:                                 # Open the descriptions JSON
#         raw = json.load(f)                                     # Parse into Python dict
#     descriptions = {}                                          # Will hold id → text mapping
#     for k, v in raw.items():                                   # Iterate over entries
#         if isinstance(v, dict):                                # Some entries are {text: ..., ...}
#             descriptions[k] = v['text']                        # Extract the text field
#         else:
#             descriptions[k] = v                                # Already a plain string
#     with open(detail_23k_path) as f:                           # Open image list JSON
#         detail_data = json.load(f)                             # Parse the list of image dicts
#     image_ids = []                                             # Ordered list of image IDs
#     id_to_filename = {}                                        # Map: image_id → filename
#     for item in detail_data:                                   # Iterate over images
#         img_id = item['id']                                    # Extract the image identifier
#         fname = os.path.basename(item['image'])                # Extract just the filename
#         image_ids.append(img_id)                               # Add to ordered list
#         id_to_filename[img_id] = fname                         # Add to lookup dict
#     return descriptions, image_ids, id_to_filename


# def build_prompt(img_id, model_type='hf', descriptions=None):
#     """Build teacher-forcing prompt for a given image.

#     Same prompt construction as neuron_modality_statistical.py:
#     the full generated description is appended as the ASSISTANT response,
#     so the model processes it in a single forward pass (no generation).
#     """
#     text = descriptions[img_id]                                # Get the pre-generated description
#     if model_type == 'hf':                                     # HF uses simple string template
#         return (f"USER: <image>\n"                             # User turn with image placeholder
#                 f"Could you describe the image?\n"             # Question prompt
#                 f"ASSISTANT: {text}")                           # Pre-filled assistant response
#     else:                                                      # Original LLaVA uses conv_templates
#         from llava.conversation import conv_templates          # Import conversation formatter
#         conv = conv_templates["v1"].copy()                     # LLaVA-v1.5 uses "v1" template
#         conv.append_message(conv.roles[0],                     # USER turn
#                             "<image>\nCould you describe the image?")
#         conv.append_message(conv.roles[1], text)               # ASSISTANT turn (teacher forcing)
#         return conv.get_prompt()                               # Format into final string


# def count_description_tokens(img_id, model_type, processor, descriptions):
#     """Count how many tokens the generated description occupies.

#     Same as in neuron_modality_statistical.py: tokenize full prompt
#     minus template prefix = number of description-only tokens.
#     """
#     desc_text = descriptions[img_id]                           # Get description text
#     if model_type == 'hf':                                     # HF processor wraps a tokenizer
#         tokenizer = processor.tokenizer                        # Access the underlying tokenizer
#         prefix = "USER: <image>\nCould you describe the image?\nASSISTANT:"
#         full = f"USER: <image>\nCould you describe the image?\nASSISTANT: {desc_text}"
#     else:                                                      # Original LLaVA backend
#         from llava.conversation import conv_templates          # Import for template formatting
#         tokenizer = processor[0]                               # Unpack (tokenizer, image_processor)
#         conv_p = conv_templates["v1"].copy()                   # Template without assistant text
#         conv_p.append_message(conv_p.roles[0],
#                               "<image>\nCould you describe the image?")
#         conv_p.append_message(conv_p.roles[1], None)           # Empty assistant turn
#         prefix = conv_p.get_prompt()
#         conv_f = conv_templates["v1"].copy()                   # Template with full description
#         conv_f.append_message(conv_f.roles[0],
#                               "<image>\nCould you describe the image?")
#         conv_f.append_message(conv_f.roles[1], desc_text)      # Full assistant response
#         full = conv_f.get_prompt()
#     prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)   # Tokenize prefix only
#     full_ids = tokenizer.encode(full, add_special_tokens=True)       # Tokenize full prompt
#     return len(full_ids) - len(prefix_ids)                           # Difference = description tokens


# # ═══════════════════════════════════════════════════════════════════
# # Section 2 — Model loading (from existing pipeline)
# # ═══════════════════════════════════════════════════════════════════

# def load_model_hf(hf_id, device):
#     """Load the HuggingFace LLaVA model.

#     Uses attn_implementation='eager' so attention weights are returned
#     when output_attentions=True is passed to the forward call.
#     """
#     from transformers import AutoProcessor, LlavaForConditionalGeneration

#     processor = AutoProcessor.from_pretrained(hf_id)           # Text + image preprocessing
#     model = LlavaForConditionalGeneration.from_pretrained(
#         hf_id,
#         torch_dtype=torch.float16,                             # Half precision for memory
#         low_cpu_mem_usage=True,                                # Reduce RAM during loading
#         attn_implementation='eager',                           # Required for attention output
#     ).to(device).eval()                                        # Move to GPU, set eval mode
#     image_token_id = model.config.image_token_index            # Token ID for <image> (e.g. 32000)
#     return model, processor, image_token_id


# def load_model_original(model_path, device):
#     """Load the original LLaVA model via the LLaVA repo.

#     The original model uses LlamaForCausalLM internals, which support
#     output_attentions natively through HuggingFace's modeling_llama.py.
#     """
#     from llava.model.builder import load_pretrained_model      # LLaVA's model loader
#     from llava.mm_utils import get_model_name_from_path        # Derive name from path
#     from llava.constants import IMAGE_TOKEN_INDEX              # Special token ID (-200)

#     model_name = get_model_name_from_path(model_path)          # e.g. "llava-v1.5-7b"
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, None, model_name, device_map=device,
#         torch_dtype=torch.float16
#     )
#     model.eval()                                               # Set to evaluation mode

#     # Patch attention implementation to eager if using SDPA (SDPA doesn't return weights)
#     llm = model.model if hasattr(model, 'model') else model    # Access the underlying LlamaModel
#     if hasattr(llm, 'config'):                                 # Check if config is accessible
#         llm.config._attn_implementation = 'eager'              # Force eager attention

#     processor = (tokenizer, image_processor)                   # Pack into tuple like pipeline does
#     return model, processor, IMAGE_TOKEN_INDEX


# # ═══════════════════════════════════════════════════════════════════
# # Section 3 — Input preparation (from existing pipeline)
# # ═══════════════════════════════════════════════════════════════════

# def prepare_inputs_hf(processor, img, text, device, image_token_id=32000):
#     """Prepare model inputs for HF LLaVA.

#     Returns: (inputs_dict, visual_mask, expanded_tokens)
#     """
#     inputs = processor(images=img, text=text,
#                        return_tensors='pt').to(device)         # Tokenize + preprocess image
#     input_ids = inputs['input_ids'][0].cpu()                   # (seq_len,) on CPU for analysis

#     # In HF, <image> is expanded to 576 repeated image_token_id tokens
#     visual_mask = (input_ids.numpy() == image_token_id)        # Bool mask: True at image positions

#     # Decode each token for display purposes
#     tokens = processor.tokenizer.convert_ids_to_tokens(
#         input_ids.tolist())                                    # List of token strings

#     return inputs, visual_mask, tokens


# def prepare_inputs_original(processor, img, text, device, image_token_id):
#     """Prepare model inputs for original LLaVA.

#     The original model replaces the single IMAGE_TOKEN_INDEX placeholder
#     with 576 image features during forward pass, so the expanded sequence
#     is longer than the tokenized input.
#     """
#     from llava.constants import IMAGE_TOKEN_INDEX              # -200
#     from llava.mm_utils import tokenizer_image_token           # Tokenize with image placeholder

#     tokenizer, image_processor = processor                     # Unpack tuple

#     # Tokenize text with <image> placeholder
#     input_ids = tokenizer_image_token(
#         text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
#     ).unsqueeze(0).to(device)                                  # (1, seq_len) on GPU

#     # Preprocess image via CLIP processor
#     image_tensor = image_processor.preprocess(
#         img, return_tensors='pt'
#     )['pixel_values'].half().to(device)                        # (1, 3, 336, 336)

#     # Build visual mask for the EXPANDED sequence
#     ids_cpu = input_ids[0].cpu().numpy()                       # Token IDs on CPU
#     img_pos = int(np.where(ids_cpu == IMAGE_TOKEN_INDEX)[0][0])  # Position of <image>
#     n_image_features = 576                                     # LLaVA uses 576 CLIP patches
#     expanded_len = len(ids_cpu) - 1 + n_image_features         # -1 placeholder + 576 features
#     visual_mask = np.zeros(expanded_len, dtype=bool)           # Initialize all False
#     visual_mask[img_pos:img_pos + n_image_features] = True     # Mark image feature positions

#     # Build expanded token list for display
#     tokens = tokenizer.convert_ids_to_tokens(ids_cpu.tolist()) # Tokenize for labels
#     expanded_tokens = []                                       # Will match expanded_len
#     for i, tok in enumerate(tokens):                           # Replace <image> with patches
#         if ids_cpu[i] == IMAGE_TOKEN_INDEX:
#             expanded_tokens.extend(
#                 [f'<vis_{j}>' for j in range(n_image_features)])  # 576 visual placeholders
#         else:
#             expanded_tokens.append(tok)                        # Keep text token as-is

#     inputs_dict = {'input_ids': input_ids, 'images': image_tensor}
#     return inputs_dict, visual_mask, expanded_tokens


# # ═══════════════════════════════════════════════════════════════════
# # Section 4 — Attention extraction and analysis
# # ═══════════════════════════════════════════════════════════════════

# def forward_with_attention(model, inputs, model_type):
#     """Run teacher-forcing forward pass with output_attentions=True.

#     Returns list of (num_heads, seq_len, seq_len) numpy arrays, one per layer.
#     """
#     with torch.no_grad():                                      # No gradient computation
#         outputs = model(**inputs, output_attentions=True)      # Forward pass with attention output

#     # outputs.attentions is a tuple of num_layers tensors
#     # Each tensor: (batch=1, num_heads, seq_len, seq_len)
#     attn_per_layer = []                                        # Will store per-layer attention
#     for layer_attn in outputs.attentions:                      # Iterate over layers
#         attn = layer_attn.squeeze(0)                           # Remove batch dim → (heads, S, S)
#         attn = attn.float().cpu().numpy()                      # Convert to float32 numpy
#         attn_per_layer.append(attn)                            # Append to list
#     return attn_per_layer


# def find_highlighted_tokens(expanded_tokens, desc_start_idx, highlighted_words):
#     """Identify which description token positions match the highlighted words.

#     Only searches within the description portion of the sequence
#     (from desc_start_idx to end), since template tokens are not relevant.

#     Returns: (highlighted_indices, non_highlighted_indices)
#     """
#     highlighted_indices = []                                   # Positions matching highlighted words
#     non_highlighted_indices = []                                # All other description token positions

#     for idx in range(desc_start_idx, len(expanded_tokens)):    # Only look at description tokens
#         tok = expanded_tokens[idx]                             # Get the token string
#         tok_clean = tok.lower().replace('▁', '').replace('Ġ', '')  # Normalize (sentencepiece/BPE)

#         if not tok_clean.strip():                              # Skip empty/whitespace tokens
#             continue

#         # Check if token matches any highlighted word
#         is_match = any(hw.lower() in tok_clean                 # Substring match
#                        for hw in highlighted_words)

#         if is_match:
#             highlighted_indices.append(idx)                    # Add to highlighted set
#         else:
#             non_highlighted_indices.append(idx)                # Add to non-highlighted set

#     return highlighted_indices, non_highlighted_indices


# def compute_attention_stats(attn_per_layer, visual_indices, text_indices,
#                             highlighted_indices, non_highlighted_indices):
#     """Compute per-layer attention statistics.

#     For highlighted and non-highlighted text tokens, computes the fraction
#     of their total attention that is directed at visual patch positions.
#     """
#     results = {
#         'highlighted_to_visual': [],                           # Per-layer mean attention to visual
#         'non_highlighted_to_visual': [],                       # Per-layer mean attention to visual
#         'highlighted_to_text': [],                             # Per-layer mean attention to text
#         'non_highlighted_to_text': [],                         # Per-layer mean attention to text
#     }

#     for layer_idx, attn in enumerate(attn_per_layer):          # Iterate over layers
#         head_avg = attn.mean(axis=0)                           # Average over heads → (S, S)

#         # Highlighted tokens → visual patches
#         if len(highlighted_indices) > 0:                       # Only compute if tokens exist
#             h_to_vis = head_avg[highlighted_indices][:, visual_indices].sum(axis=1).mean()
#             h_to_txt = head_avg[highlighted_indices][:, text_indices].sum(axis=1).mean()
#         else:
#             h_to_vis = 0.0                                     # No highlighted tokens found
#             h_to_txt = 0.0

#         # Non-highlighted tokens → visual patches
#         if len(non_highlighted_indices) > 0:                   # Only compute if tokens exist
#             nh_to_vis = head_avg[non_highlighted_indices][:, visual_indices].sum(axis=1).mean()
#             nh_to_txt = head_avg[non_highlighted_indices][:, text_indices].sum(axis=1).mean()
#         else:
#             nh_to_vis = 0.0                                    # No non-highlighted tokens found
#             nh_to_txt = 0.0

#         results['highlighted_to_visual'].append(float(h_to_vis))
#         results['non_highlighted_to_visual'].append(float(nh_to_vis))
#         results['highlighted_to_text'].append(float(h_to_txt))
#         results['non_highlighted_to_text'].append(float(nh_to_txt))

#     return results


# # ═══════════════════════════════════════════════════════════════════
# # Section 5 — Plotting
# # ═══════════════════════════════════════════════════════════════════

# def plot_heatmap(attn_per_layer, visual_indices, highlighted_indices,
#                  expanded_tokens, target_layers, save_path):
#     """Plot heatmap: attention from highlighted text tokens to 24×24 visual grid.

#     Each row is one highlighted token, each column is one target layer.
#     The 576 visual attention values are reshaped to a 24×24 grid matching
#     the ViT patch layout.
#     """
#     if len(highlighted_indices) == 0:                          # Guard against empty input
#         print('No highlighted tokens found — skipping heatmap.')
#         return

#     # Filter to layers that exist in the data
#     target_layers = [l for l in target_layers if l < len(attn_per_layer)]
#     n_layers = len(target_layers)                              # Number of columns
#     n_tokens = len(highlighted_indices)                        # Number of rows

#     fig, axes = plt.subplots(
#         n_tokens, n_layers,
#         figsize=(3 * n_layers, 3 * n_tokens),                 # Scale figure size
#         squeeze=False                                          # Always return 2D array of axes
#     )

#     for col, layer_idx in enumerate(target_layers):            # Iterate over selected layers
#         head_avg = attn_per_layer[layer_idx].mean(axis=0)      # Average over heads → (S, S)

#         for row, tok_idx in enumerate(highlighted_indices):    # Iterate over highlighted tokens
#             attn_to_visual = head_avg[tok_idx, visual_indices] # Attention to all 576 visual patches
#             attn_map = attn_to_visual.reshape(24, 24)          # Reshape to ViT grid layout

#             ax = axes[row, col]                                # Get the subplot axis
#             ax.imshow(attn_map, cmap='hot', interpolation='nearest')  # Heatmap

#             if col == 0:                                       # Label rows with token strings
#                 label = expanded_tokens[tok_idx] if tok_idx < len(expanded_tokens) else '?'
#                 ax.set_ylabel(label.replace('▁', '').replace('Ġ', ''),
#                               fontsize=10, fontweight='bold')
#             if row == 0:                                       # Label columns with layer numbers
#                 ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')

#             ax.set_xticks([])                                  # Hide tick marks
#             ax.set_yticks([])

#     fig.suptitle(
#         'Attention from highlighted text tokens to visual patches (24×24)',
#         fontsize=14, fontweight='bold', y=1.02
#     )
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f'Heatmap saved to: {save_path}')


# def plot_bar_chart(results, save_path):
#     """Bar chart comparing visual attention for highlighted vs non-highlighted tokens.

#     X-axis: layer index (0-31), Y-axis: fraction of attention to visual patches.
#     Two bars per layer: red = highlighted tokens, blue = non-highlighted.
#     """
#     n_layers = len(results['highlighted_to_visual'])           # Number of layers
#     layers = np.arange(n_layers)                               # Layer indices for x-axis
#     bar_width = 0.35                                           # Width of each bar

#     fig, ax = plt.subplots(figsize=(14, 5))                    # Wide figure for 32 layers

#     ax.bar(layers - bar_width / 2,                             # Left-shifted bars
#            results['highlighted_to_visual'],
#            bar_width,
#            label='Highlighted tokens → visual',
#            color='#d62728', alpha=0.85)                        # Red

#     ax.bar(layers + bar_width / 2,                             # Right-shifted bars
#            results['non_highlighted_to_visual'],
#            bar_width,
#            label='Non-highlighted tokens → visual',
#            color='#1f77b4', alpha=0.85)                        # Blue

#     ax.set_xlabel('Layer', fontsize=12)
#     ax.set_ylabel('Fraction of attention to visual patches', fontsize=12)
#     ax.set_title(
#         'Attention to visual patches: highlighted vs non-highlighted text tokens',
#         fontsize=13, fontweight='bold'
#     )
#     ax.legend(fontsize=11)
#     ax.set_xticks(layers)
#     ax.set_xticklabels([str(l) for l in layers], fontsize=8)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f'Bar chart saved to: {save_path}')


# # ═══════════════════════════════════════════════════════════════════
# # Section 6 — Argument parsing
# # ═══════════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser(
#         description='Attention analysis for reclassified neurons')

#     # Image source — provide either image_id (looks up from pipeline data) or image_path
#     p.add_argument('--image_id', type=str, default=None,
#                    help='COCO image ID (e.g. 000000189475). Looks up filename '
#                         'from detail_23k.json and loads from --coco_img_dir')
#     p.add_argument('--image_path', type=str, default=None,
#                    help='Direct path to an image file (overrides --image_id)')

#     # Words to highlight (the ones Xu's method incorrectly flagged)
#     p.add_argument('--highlighted_words', type=str, nargs='+',
#                    default=['hot', 'white', 'blue', 'tie'],
#                    help='Words that Xu highlighted as text-activated '
#                         '(default: hot white blue tie)')

#     # Model settings — same as rest of pipeline
#     p.add_argument('--model_type', default='hf', choices=['hf', 'liuhaotian'],
#                    help='"hf" for llava-hf/llava-1.5-7b-hf, '
#                         '"liuhaotian" for liuhaotian/llava-v1.5-7b')
#     p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
#                    help='HuggingFace model ID (for --model_type hf)')
#     p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b',
#                    help='Original LLaVA model path (for --model_type liuhaotian)')
#     p.add_argument('--device', default='0',
#                    help='GPU device index (default: 0)')

#     # Data paths — same defaults as rest of pipeline
#     p.add_argument('--coco_img_dir',
#                    default='/home/projects/bagon/shared/coco2017/images/train2017/')
#     p.add_argument('--generated_desc_path',
#                    default='results/generated_descriptions/generated_descriptions.json',
#                    help='Path to LLaVA-generated descriptions JSON')
#     _project_root = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
#     p.add_argument('--detail_23k_path',
#                    default=os.path.join(_project_root, 'neuron_taxonomy',
#                                         'data', 'detail_23k.json'),
#                    help='Path to detail_23k.json')

#     # Output settings
#     p.add_argument('--output_dir', type=str,
#                    default='results/attention_analysis',
#                    help='Directory to save output figures')
#     p.add_argument('--heatmap_layers', type=int, nargs='+',
#                    default=[0, 7, 15, 23, 28, 31],
#                    help='Layers to show in heatmap (default: 0 7 15 23 28 31)')

#     return p.parse_args()


# # ═══════════════════════════════════════════════════════════════════
# # Section 7 — Main
# # ═══════════════════════════════════════════════════════════════════

# def main():
#     args = parse_args()
#     device = f'cuda:{args.device}' if args.device.isdigit() else args.device
#     os.makedirs(args.output_dir, exist_ok=True)                # Create output directory

#     # ── Validate image source ─────────────────────────────────
#     if args.image_path is None and args.image_id is None:      # Must provide one
#         print('ERROR: provide either --image_path or --image_id')
#         sys.exit(1)

#     # ── Load descriptions + image list ────────────────────────
#     print('Loading descriptions and image list...')
#     descriptions, image_ids, id_to_filename = load_generated_descriptions(
#         args.generated_desc_path, args.detail_23k_path)
#     print(f'  {len(descriptions)} descriptions loaded')

#     # ── Resolve image path and image ID ───────────────────────
#     if args.image_path is not None:                            # Direct path provided
#         img_path = args.image_path
#         # Try to find matching image_id from filename
#         fname = os.path.basename(img_path)
#         args.image_id = None
#         for iid, fn in id_to_filename.items():                # Reverse lookup
#             if fn == fname:
#                 args.image_id = iid
#                 break
#         if args.image_id is None:                              # Couldn't match — fall back
#             print(f'WARNING: could not find image_id for {fname}.')
#             print('         Using first available description for prompt.')
#             args.image_id = image_ids[0]                       # Fallback (won't match perfectly)
#     else:                                                      # image_id provided
#         fname = id_to_filename.get(args.image_id)
#         if fname is None:
#             print(f'ERROR: image_id {args.image_id} not found in detail_23k.json')
#             sys.exit(1)
#         img_path = os.path.join(args.coco_img_dir, fname)

#     if not os.path.exists(img_path):                           # Check file exists
#         print(f'ERROR: image not found at {img_path}')
#         sys.exit(1)

#     print(f'  Image: {img_path}')
#     print(f'  Image ID: {args.image_id}')
#     print(f'  Description: {descriptions[args.image_id][:80]}...')

#     # ── Load model ────────────────────────────────────────────
#     if args.model_type == 'hf':
#         print(f'\nLoading HF model: {args.hf_id} on {device}...')
#         model, processor, image_token_id = load_model_hf(args.hf_id, device)
#     else:
#         print(f'\nLoading original model: {args.original_model_path} on {device}...')
#         model, processor, image_token_id = load_model_original(
#             args.original_model_path, device)
#     print('Model loaded.')

#     # ── Prepare inputs (teacher-forcing, same as pipeline) ────
#     print('\nPreparing inputs (teacher-forcing mode)...')
#     img = Image.open(img_path).convert('RGB')                  # Load image as RGB
#     prompt = build_prompt(args.image_id, model_type=args.model_type,
#                           descriptions=descriptions)

#     if args.model_type == 'hf':
#         inputs, visual_mask, expanded_tokens = prepare_inputs_hf(
#             processor, img, prompt, device, image_token_id)
#     else:
#         inputs, visual_mask, expanded_tokens = prepare_inputs_original(
#             processor, img, prompt, device, image_token_id)

#     seq_len = len(visual_mask)                                 # Total sequence length
#     visual_indices = np.where(visual_mask)[0].tolist()         # Positions of visual tokens
#     text_mask = ~visual_mask                                   # Invert to get text positions
#     all_text_indices = np.where(text_mask)[0].tolist()         # Positions of all text tokens

#     # Identify description tokens (at the end of the sequence)
#     desc_count = count_description_tokens(
#         args.image_id, args.model_type, processor, descriptions)
#     desc_start_idx = seq_len - desc_count                      # Description starts here

#     print(f'  Sequence length: {seq_len}')
#     print(f'  Visual patches: {len(visual_indices)}')
#     print(f'  Total text tokens: {len(all_text_indices)}')
#     print(f'  Description tokens: {desc_count} (starting at position {desc_start_idx})')

#     # ── Find highlighted vs non-highlighted tokens ────────────
#     highlighted_idx, non_highlighted_idx = find_highlighted_tokens(
#         expanded_tokens, desc_start_idx, args.highlighted_words)

#     print(f'\n  Highlighted tokens ({len(highlighted_idx)}):')
#     for idx in highlighted_idx:                                # Show which tokens matched
#         tok = expanded_tokens[idx] if idx < len(expanded_tokens) else '?'
#         print(f'    pos {idx}: "{tok}"')
#     print(f'  Non-highlighted description tokens: {len(non_highlighted_idx)}')

#     # ── Forward pass with attention ───────────────────────────
#     print('\nRunning forward pass with output_attentions=True...')
#     attn_per_layer = forward_with_attention(model, inputs, args.model_type)
#     print(f'  Extracted attention from {len(attn_per_layer)} layers')
#     print(f'  Attention shape per layer: {attn_per_layer[0].shape}')

#     # ── Compute statistics ────────────────────────────────────
#     print('\nComputing attention statistics...')
#     results = compute_attention_stats(
#         attn_per_layer, visual_indices, all_text_indices,
#         highlighted_idx, non_highlighted_idx)

#     # Print summary table
#     print(f'\n{"─"*75}')
#     print(f'{"Layer":<8} {"Highlighted→Vis":<18} {"NonHighl→Vis":<18} {"Δ (H − NH)":<14} {"Ratio":<10}')
#     print(f'{"─"*75}')
#     for i in range(len(results['highlighted_to_visual'])):
#         h = results['highlighted_to_visual'][i]
#         nh = results['non_highlighted_to_visual'][i]
#         diff = h - nh
#         ratio = h / nh if nh > 1e-9 else float('inf')
#         print(f'{i:<8} {h:<18.4f} {nh:<18.4f} {diff:<14.4f} {ratio:<10.2f}')

#     # Overall average across layers
#     avg_h = np.mean(results['highlighted_to_visual'])
#     avg_nh = np.mean(results['non_highlighted_to_visual'])
#     print(f'{"─"*75}')
#     print(f'{"Avg":<8} {avg_h:<18.4f} {avg_nh:<18.4f} '
#           f'{avg_h - avg_nh:<14.4f} {avg_h / avg_nh if avg_nh > 1e-9 else float("inf"):<10.2f}')

#     # ── Save numerical results ────────────────────────────────
#     np.savez(
#         os.path.join(args.output_dir, 'attention_stats.npz'),
#         highlighted_to_visual=results['highlighted_to_visual'],
#         non_highlighted_to_visual=results['non_highlighted_to_visual'],
#         highlighted_to_text=results['highlighted_to_text'],
#         non_highlighted_to_text=results['non_highlighted_to_text'],
#         highlighted_tokens=[expanded_tokens[i] for i in highlighted_idx],
#         highlighted_words=args.highlighted_words,
#         image_id=args.image_id,
#     )
#     print(f'\nNumerical results saved to: {args.output_dir}/attention_stats.npz')

#     # ── Plot heatmap ──────────────────────────────────────────
#     print('\nGenerating heatmap...')
#     heatmap_path = os.path.join(args.output_dir, 'attention_heatmap.png')
#     plot_heatmap(
#         attn_per_layer, visual_indices, highlighted_idx,
#         expanded_tokens, args.heatmap_layers, heatmap_path)

#     # ── Plot bar chart ────────────────────────────────────────
#     print('Generating bar chart...')
#     bar_path = os.path.join(args.output_dir, 'attention_bar_chart.png')
#     plot_bar_chart(results, bar_path)

#     print(f'\nDone! All outputs in: {args.output_dir}')


# if __name__ == '__main__':
#     main()
