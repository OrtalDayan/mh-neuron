"""
neuron_classify_shabi.py — Neuron classification with permutation testing

Extends Xu et al. (MM 2025) Phases 1-2 with principled statistical testing.

  Top-N Heap — (From Xu, unchanged) For each neuron, find top-N (default 50)
      most-activating samples from the 23K detail image subset. Also track
      each neuron's global maximum activation for normalisation.
      ** SAVES: top_n_sids, top_n_acts, global_max per layer **

  Activation Pattern — Re-process the top-N images. Record normalised per-token
      activations for BOTH visual and text positions per neuron.
      Also runs Xu's original threshold-based classification for comparison.
      ** SAVES: raw normalised activations per neuron (vis + txt) per layer **

  Fixed-threshold — (Xu's method) Compute (pv, pt, pm, pu) proportions, assign
      neuron label = argmax. Kept for comparison with permutation results.

  Permutation-test — (NEW) For each neuron:
      1. Apply Otsu's method to find adaptive high-activation threshold
      2. Count observed visual/text tokens above threshold
      3. Shuffle modality labels 1000×, build null distribution
      4. Compute p-values, classify at α=0.05
      ** Can be re-run offline from saved Activation Pattern data (no GPU needed) **

Scope: LLM FFN neurons only — output of act_fn = SiLU(gate_proj(x)),
       11,008 neurons/layer × 32 layers = 352,256 neurons total.
       Per refs [22, 29]: "output from the activation function of the
       first linear transformation (neurons) of each FFN layer."

Dataset: Xu et al. used 23K images from COCO train2017 (the detail_23k
         subset from LLaVA training data). Text tokens come from LLaVA's
         own generated descriptions via model.generate() with the prompt
         "Could you describe the image?" (confirmed by comparing Figure
         text against LLaVA outputs). Teacher forcing is then used with
         the generated text for efficient activation recording.

         Requires: 1) detail_23k.json (image list)
                   2) generated_descriptions.json (from generate_descriptions.py)

GPU sharding: by layer range. Each GPU processes all images for its assigned
layers. With 32 layers, use up to 32 GPUs for max parallelism.

Supports two model backends:
  --model_type llava-hf         : HuggingFace llava-hf/llava-1.5-7b-hf (default)
  --model_type llava-liuhaotian : Original liuhaotian/llava-v1.5-7b via LLaVA repo

Usage:
    # Step 0 — generate descriptions (run once)
    python generate_descriptions.py

    # Step 1 — classify with HF model (default)
    python neuron_classify_xu.py --text_source generated --device 0

    # Classify with original LLaVA model
    python neuron_classify_xu.py --model_type llava-liuhaotian --text_source generated --device 0

    # GPU sharding — one layer per GPU
    python neuron_classify_xu.py --text_source generated --device 0 --layer_start 0 --layer_end 1

    # Fallback: use COCO captions instead
    python neuron_classify_xu.py --text_source coco --device 0
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Add LLaVA repo to path (needed for original model backend)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')
if _LLAVA_PATH not in sys.path:
    sys.path.insert(0, _LLAVA_PATH)

import numpy as np
import torch
from baukit import TraceDict
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Helpers
# ═══════════════════════════════════════════════════════════════════

def load_coco_captions(ann_path):
    """Load COCO captions → {image_id: [captions]}, {id: filename}."""
    with open(ann_path) as f:
        data = json.load(f)
    captions = {}
    for ann in data['annotations']:
        captions.setdefault(ann['image_id'], []).append(ann['caption'])
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    return captions, id_to_filename


def load_generated_descriptions(desc_path, detail_23k_path):
    """Load LLaVA-generated descriptions and detail_23k image list.

    Returns:
        descriptions: {image_id_str: generated_text}
        image_ids:    list of image ID strings from detail_23k
        id_to_filename: {image_id_str: filename}
    """
    # Load generated descriptions
    # Supports two formats:
    #   flat:   {"000000323964": "The image features..."}
    #   nested: {"000000323964": {"token_ids": [...], "tokens": [...], "text": "...", ...}}
    #
    # Always use the 'text' field — it contains the properly decoded description.
    # Joining subword tokens with spaces corrupts the text (e.g. "d ough n uts"
    # instead of "doughnuts"), which produces wrong tokenisation during teacher forcing.
    with open(desc_path) as f:
        raw = json.load(f)
    descriptions = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            descriptions[k] = v['text']      # use properly decoded text, NOT joined subword tokens
        else:
            descriptions[k] = v

    # Load detail_23k to get the image list and filenames
    with open(detail_23k_path) as f:
        detail_data = json.load(f)

    image_ids = []
    id_to_filename = {}
    for item in detail_data:
        img_id = item['id']           # e.g. "000000323964"
        fname = os.path.basename(item['image'])  # e.g. "000000323964.jpg"
        image_ids.append(img_id)
        id_to_filename[img_id] = fname

    return descriptions, image_ids, id_to_filename


def get_layer_names(model_type='llava-hf', n_layers=32):
    """Return act_fn layer names — we capture their OUTPUT = SiLU(gate_proj(x)).

    Per Xu et al. Section 3.2 and refs [22, 29]:
        "We use the output from the activation function of the first
         linear transformation (neurons) of each FFN layer."

    In LLaMA-2 / Vicuna SwiGLU MLP (LlamaMLP.forward):
        gate = act_fn(gate_proj(x))     ← 11,008-dim — we hook HERE (act_fn output)
        up   = up_proj(x)               ← 11,008-dim — parallel branch, NOT included
        out  = down_proj(gate * up)      ← 4,096-dim  — final projection

    Previously we hooked down_proj with retain_input=True, which captured
    gate * up (the element-wise product of BOTH branches). The paper
    specifies only the first linear + activation, i.e. SiLU(gate_proj(x)).

    Layer name prefix differs by model type:
        HF:       model.language_model.layers.{i}.mlp.act_fn
        Original: model.layers.{i}.mlp.act_fn
    """
    if model_type == 'llava-hf':
        # HF LlavaForConditionalGeneration — verified working on cluster
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'internvl':
        # InternVLChatModel → .language_model (InternLM2ForCausalLM)
        #   → .model (InternLM2Model) → .layers[i] → .feed_forward (InternLM2MLP)
        #   → .act_fn (nn.SiLU) — output of SiLU(gate_proj(x)), same as LLaMA
        prefix = 'language_model.model.layers'
        suffix = 'feed_forward.act_fn'
    elif model_type == 'qwen2vl':
        # Qwen2_5_VLForConditionalGeneration → .model (Qwen2_5_VLModel)
        #   → .layers[i] → .mlp (Qwen2MLP) → .act_fn (SiLU)
        prefix = 'model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'llava-ov':
        # LlavaOnevisionForConditionalGeneration → .language_model (Qwen2ForCausalLM)
        #   → .model (Qwen2Model) → .layers[i] → .mlp (Qwen2MLP)
        #   → .act_fn (nn.SiLU) — same SwiGLU structure as LLaMA
        prefix = 'language_model.model.layers'
        suffix = 'mlp.act_fn'
    else:
        # Original LlavaLlamaForCausalLM inherits LlamaForCausalLM
        # → .model (LlamaModel) → .layers
        prefix = 'model.layers'
        suffix = 'mlp.act_fn'
    return [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]


def build_prompt(img_id, text_source, model_type='llava-hf',
                 descriptions=None, captions_dict=None):
    """Build teacher-forcing prompt for a given image.

    For 'generated' mode (Xu's method):
        The full generated description is pre-filled as the assistant
        response, so the model processes it in a single forward pass.

    For 'coco' mode (fallback):
        The COCO caption is used as the text input.

    Prompt format depends on model_type:
        HF:       "USER: <image>\\n...\\nASSISTANT: {text}"
        Original: Uses conv_templates["v1"] for proper role formatting
    """
    if text_source == 'generated':
        text = descriptions[img_id]
        question = "Could you describe the image?"
    else:
        text = captions_dict[img_id][0]
        question = text

    if model_type == 'llava-hf':
        # HF uses simple string with <image> placeholder
        if text_source == 'generated':
            return f"USER: <image>\nCould you describe the image?\nASSISTANT: {text}"
        else:
            return f"USER: <image>\n{text}\nASSISTANT:"
    elif model_type == 'internvl':
        # InternVL: prompt is built inside prepare_inputs_internvl()
        # because the number of <IMG_CONTEXT> tokens depends on the image.
        # Return just the description text for reference / token counting.
        return text
    elif model_type == 'llava-ov':
        # LLaVA-OneVision: prompt is built inside prepare_inputs_llava_ov()
        # using processor.apply_chat_template(). Return description text
        # for reference / token counting.
        return text
    elif model_type == 'qwen2vl':
        # Qwen2.5-VL: prompt is built inside prepare_inputs_qwen2vl()
        # using processor.apply_chat_template(). Return description text
        # for reference / token counting.
        return text
    else:
        # Original uses conv_templates for proper role formatting
        from llava.conversation import conv_templates
        conv = conv_templates["v1"].copy()                              # LLaVA-v1.5 uses "v1" template
        if text_source == 'generated':
            conv.append_message(conv.roles[0],
                                "<image>\nCould you describe the image?")  # USER turn
            conv.append_message(conv.roles[1], text)                    # ASSISTANT turn (teacher forcing)
        else:
            conv.append_message(conv.roles[0], f"<image>\n{text}")
            conv.append_message(conv.roles[1], None)
        return conv.get_prompt()


def count_description_tokens(img_id, model_type, processor, descriptions):
    """Count how many tokens the generated description occupies.

    Per Xu et al. Figures 1-3, only the generated description tokens
    count as "text tokens" for classification. Template tokens (BOS,
    USER:, prompt, ASSISTANT:) are excluded — they are identical
    across all samples and would add noise to text-activation counts.

    Method: tokenize the full prompt and the template-only prefix.
    The difference = number of description tokens. Since the
    description sits at the END of the token sequence (after image
    expansion), we can mark the last desc_count positions as text.

    Returns:
        int — number of description tokens in the tokenized sequence.
    """
    desc_text = descriptions[img_id]

    if model_type == 'llava-hf':
        tokenizer = processor.tokenizer                                # HF AutoProcessor wraps tokenizer
        prefix = "USER: <image>\nCould you describe the image?\nASSISTANT:"
        full = f"USER: <image>\nCould you describe the image?\nASSISTANT: {desc_text}"
    elif model_type == 'internvl':
        # For InternVL, processor IS the tokenizer (AutoTokenizer).
        # Tokenise the description text standalone — avoids needing to
        # construct the full IMG_CONTEXT-laden template just for counting.
        tokenizer = processor
        desc_ids = tokenizer.encode(desc_text, add_special_tokens=False)
        return len(desc_ids)
    elif model_type == 'llava-ov':
        # For LLaVA-OV, processor is AutoProcessor — access .tokenizer
        # Tokenise the description text standalone for counting.
        tokenizer = processor.tokenizer
        desc_ids = tokenizer.encode(desc_text, add_special_tokens=False)
        return len(desc_ids)
    elif model_type == 'qwen2vl':
        # For Qwen2.5-VL, processor is AutoProcessor — access .tokenizer
        # Tokenise the description text standalone for counting.
        tokenizer = processor.tokenizer
        desc_ids = tokenizer.encode(desc_text, add_special_tokens=False)
        return len(desc_ids)
    else:
        from llava.conversation import conv_templates
        tokenizer = processor[0]                                       # (tokenizer, image_processor) tuple

        # Prefix: assistant turn left empty (ends with "ASSISTANT:")
        conv_p = conv_templates["v1"].copy()
        conv_p.append_message(conv_p.roles[0],
                              "<image>\nCould you describe the image?")
        conv_p.append_message(conv_p.roles[1], None)
        prefix = conv_p.get_prompt()

        # Full: assistant turn filled with description
        conv_f = conv_templates["v1"].copy()
        conv_f.append_message(conv_f.roles[0],
                              "<image>\nCould you describe the image?")
        conv_f.append_message(conv_f.roles[1], desc_text)
        full = conv_f.get_prompt()

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)     # [BOS, ..., ASSISTANT, :]
    full_ids = tokenizer.encode(full, add_special_tokens=True)         # [BOS, ..., ASSISTANT, :, desc...]

    return len(full_ids) - len(prefix_ids)


# ═══════════════════════════════════════════════════════════════════
# Section 1a — Permutation test helpers
# ═══════════════════════════════════════════════════════════════════

def otsu_threshold(values):
    """Compute Otsu's threshold on a 1-D array of activation values.

    Finds the threshold that minimises within-class variance when
    splitting values into "low" and "high" groups.  This adaptively
    determines what counts as "high activation" for each neuron.

    Args:
        values: 1-D numpy array of normalised activation values (0-10).

    Returns:
        float — the optimal threshold.  Values > threshold are "high".
    """
    # Line 1: remove zeros for cleaner bimodal split (many tokens are ~0)
    vals = values[values > 0]
    if len(vals) < 2:
        return 5.0                                  # fallback if neuron is nearly silent

    # Line 2: build histogram (256 bins over the 0-10 range)
    hist, bin_edges = np.histogram(vals, bins=256, range=(0.0, 10.0))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Line 3: normalise histogram to probabilities
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 5.0
    hist /= total

    # Line 4: cumulative sums for between-class variance
    w0 = np.cumsum(hist)                            # weight of class 0 (below threshold)
    w1 = 1.0 - w0                                   # weight of class 1 (above threshold)
    mu0_cum = np.cumsum(hist * bin_centres)          # cumulative mean of class 0
    mu_total = mu0_cum[-1]                           # total mean

    # Line 5: avoid division by zero
    valid = (w0 > 1e-10) & (w1 > 1e-10)
    if not valid.any():
        return 5.0

    # Line 6: between-class variance (Otsu's criterion)
    mu0 = mu0_cum[valid] / w0[valid]
    mu1 = (mu_total - mu0_cum[valid]) / w1[valid]
    between_var = w0[valid] * w1[valid] * (mu0 - mu1) ** 2

    # Line 7: optimal threshold = bin centre that maximises between-class variance
    best_idx = np.argmax(between_var)
    return float(bin_centres[valid][best_idx])


def classify_neuron_permutation(vis_acts, txt_acts, n_permutations=1000,
                                 alpha=0.05, min_high_tokens=5, rng=None):
    """Classify a single neuron using a rate-difference permutation test.

    Instead of testing each modality independently (zero-sum: one wins,
    the other loses), we test whether the neuron *prefers* one modality
    over the other.  The test statistic is:

        D = high_vis_rate - high_txt_rate
          = (n_high_vis / n_vis) - (n_high_txt / n_txt)

    This rate normalisation accounts for unequal token counts between
    modalities. Under the null hypothesis (neuron is indifferent to
    modality), shuffling modality labels gives the null distribution
    of D.  A two-tailed test then classifies:

        |D| significant & D > 0  →  visual   (prefers visual)
        |D| significant & D < 0  →  text     (prefers text)
        |D| NOT significant      →  multimodal (responds equally)
        too few high tokens      →  unknown   (neuron inactive)

    Args:
        vis_acts:  1-D numpy array — normalised activations at visual token
                   positions across all top-k samples (concatenated).
        txt_acts:  1-D numpy array — normalised activations at text token
                   positions across all top-k samples (concatenated).
        n_permutations: number of label shuffles for null distribution.
        alpha:     significance level for the two-tailed test.
        min_high_tokens: minimum total high-activation tokens required to
                   classify as visual/text/multimodal (below → unknown).
        rng:       numpy RandomState for reproducibility.

    Returns:
        dict with keys:
            'label':    str — 'visual', 'text', 'multimodal', or 'unknown'
            'p_value':  float — two-tailed p-value for the rate difference
            'otsu_threshold': float — adaptive threshold used
            'observed_visual': int — count of visual tokens above threshold
            'observed_text':   int — count of text tokens above threshold
            'observed_rate_diff': float — D = vis_rate - txt_rate
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Line 1: pool all activations and build modality label array
    all_acts = np.concatenate([vis_acts, txt_acts])       # (n_vis + n_txt,)
    n_vis = len(vis_acts)                                  # total visual token count
    n_txt = len(txt_acts)                                  # total text token count
    n_total = n_vis + n_txt
    is_visual = np.zeros(n_total, dtype=bool)
    is_visual[:n_vis] = True                               # first n_vis are visual

    # Line 2: compute Otsu threshold on pooled activations
    threshold = otsu_threshold(all_acts)

    # Line 3: binary high-activation mask
    is_high = all_acts > threshold                          # (n_total,) bool

    # Line 4: observed counts
    observed_visual = int((is_high & is_visual).sum())      # visual tokens above threshold
    observed_text = int((is_high & ~is_visual).sum())       # text tokens above threshold
    total_high = observed_visual + observed_text

    # Line 5: activity gate — if too few tokens exceed threshold,
    #          the neuron is effectively inactive → unknown
    if total_high < min_high_tokens:
        return {
            'label': 'unknown',
            'p_value': 1.0,
            'otsu_threshold': float(threshold),
            'observed_visual': observed_visual,
            'observed_text': observed_text,
            'observed_rate_diff': 0.0,
        }

    # Line 6: observed rate difference D = vis_rate - txt_rate
    #          rate normalisation lets us compare modalities fairly
    #          even when n_vis >> n_txt (e.g. 576 image tokens vs ~100 text)
    rate_vis = observed_visual / n_vis
    rate_txt = observed_text / n_txt
    observed_diff = rate_vis - rate_txt

    # Line 7: permutation — shuffle modality labels, build null
    #          distribution of D under the indifference hypothesis
    null_diffs = np.empty(n_permutations, dtype=np.float64)
    for p in range(n_permutations):
        rng.shuffle(is_visual)                              # in-place shuffle of labels
        perm_vis = (is_high & is_visual).sum()
        perm_txt = (is_high & ~is_visual).sum()
        null_diffs[p] = perm_vis / n_vis - perm_txt / n_txt

    # Line 8: two-tailed p-value — how often does the null produce
    #          a rate difference as extreme as (or more than) observed?
    p_value = (np.abs(null_diffs) >= np.abs(observed_diff)).sum() / n_permutations

    # Line 9: classify based on significance and direction
    #          significant → neuron has a modality preference (visual or text)
    #          not significant → neuron responds equally → multimodal
    if p_value < alpha:
        label = 'visual' if observed_diff > 0 else 'text'
    else:
        label = 'multimodal'

    return {
        'label': label,
        'p_value': float(p_value),
        'otsu_threshold': float(threshold),
        'observed_visual': observed_visual,
        'observed_text': observed_text,
        'observed_rate_diff': float(observed_diff),
    }


def classify_neuron_permutation_gpu(vis_acts, txt_acts, n_permutations=1000,
                                    alpha=0.05, min_high_tokens=5, device='cuda',
                                    seed=42):
    """GPU-accelerated permutation test — vectorises all shuffles in one pass.

    Same statistical test as classify_neuron_permutation, but instead of
    1000 sequential numpy shuffles, generates all permutation indices on
    GPU and computes the null distribution via a single batched gather +
    element-wise multiply + sum.

    Memory: O(n_permutations × n_total) floats on GPU.  For a typical
    neuron with ~2000 tokens and 1000 perms, this is ~8 MB — negligible.

    Speed: ~100× faster per neuron than the CPU loop.  On a full layer of
    18,944 neurons the permutation test drops from ~22 min → ~15 sec.
    """
    # 1. Pool activations, build labels (CPU — tiny, fast)
    all_acts = np.concatenate([vis_acts, txt_acts])
    n_vis = len(vis_acts)
    n_txt = len(txt_acts)
    n_total = n_vis + n_txt

    # 2. Otsu threshold (CPU — histogram on a few thousand values)
    threshold = otsu_threshold(all_acts)

    # 3. Binary masks
    is_high = all_acts > threshold
    observed_visual = int((is_high[:n_vis]).sum())             # high visual tokens
    observed_text   = int((is_high[n_vis:]).sum())             # high text tokens
    total_high = observed_visual + observed_text

    if total_high < min_high_tokens:
        return {
            'label': 'unknown', 'p_value': 1.0,
            'otsu_threshold': float(threshold),
            'observed_visual': observed_visual,
            'observed_text': observed_text,
            'observed_rate_diff': 0.0,
        }

    # 4. Observed rate difference
    rate_vis = observed_visual / n_vis
    rate_txt = observed_text / n_txt
    observed_diff = rate_vis - rate_txt

    # 5. GPU-vectorised permutations
    #    Generate (n_perm, n_total) random permutation indices on GPU.
    #    For each permutation, the first n_vis positions in the shuffled
    #    order are treated as "visual" — equivalent to shuffling labels.
    is_high_gpu = torch.tensor(is_high, dtype=torch.float32, device=device)  # (n_total,)

    gen = torch.Generator(device=device).manual_seed(seed)
    # Build all permutations at once: (n_perm, n_total)
    perm_indices = torch.stack([
        torch.randperm(n_total, device=device, generator=gen)
        for _ in range(n_permutations)
    ])

    # Gather: shuffled_high[i, j] = is_high[perm_indices[i, j]]
    shuffled_high = is_high_gpu[perm_indices]                 # (n_perm, n_total)

    # Count high tokens that landed in "visual" positions (first n_vis)
    perm_vis_high = shuffled_high[:, :n_vis].sum(dim=1)       # (n_perm,)
    perm_txt_high = shuffled_high[:, n_vis:].sum(dim=1)       # (n_perm,)

    # Null distribution of rate differences
    null_diffs = perm_vis_high / n_vis - perm_txt_high / n_txt  # (n_perm,)

    # 6. Two-tailed p-value
    p_value = float((null_diffs.abs() >= abs(observed_diff)).sum()) / n_permutations

    # 7. Classify
    if p_value < alpha:
        label = 'visual' if observed_diff > 0 else 'text'
    else:
        label = 'multimodal'

    return {
        'label': label, 'p_value': float(p_value),
        'otsu_threshold': float(threshold),
        'observed_visual': observed_visual,
        'observed_text': observed_text,
        'observed_rate_diff': float(observed_diff),
    }


# ═══════════════════════════════════════════════════════════════════
# Section 1b — Model loading helpers
# ═══════════════════════════════════════════════════════════════════

def load_model_hf(hf_id, device):
    """Load the HuggingFace LLaVA model.

    Returns: (model, processor, image_token_id)
        - processor: HF AutoProcessor (handles both text + image)
        - image_token_id: token ID used for <image> placeholders
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(hf_id)                    # handles text + image tokenisation
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    image_token_id = model.config.image_token_index                     # e.g. 32000

    return model, processor, image_token_id


def load_model_original(model_path, device):
    """Load the original LLaVA model via the LLaVA repo.

    Returns: (model, (tokenizer, image_processor), image_token_id)
        - tokenizer: the LLaMA tokenizer
        - image_processor: CLIP image processor (stored in tuple)
        - image_token_id: IMAGE_TOKEN_INDEX from llava.constants
    """
    from llava.model.builder import load_pretrained_model               # from cloned LLaVA repo
    from llava.mm_utils import get_model_name_from_path                 # derives model name from path
    from llava.constants import IMAGE_TOKEN_INDEX                       # typically -200

    model_name = get_model_name_from_path(model_path)                   # e.g. "llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=device,
        torch_dtype=torch.float16
    )
    model.eval()

    # Pack tokenizer + image_processor into a tuple to match HF "processor" slot
    processor = (tokenizer, image_processor)
    return model, processor, IMAGE_TOKEN_INDEX


# ═══════════════════════════════════════════════════════════════════
# Section 1c — Input preparation helpers
# ═══════════════════════════════════════════════════════════════════

def prepare_inputs_hf(processor, img, text, device, image_token_id=32000):
    """Prepare model inputs for HF LLaVA.

    Returns: (inputs_dict, visual_mask)
        - inputs_dict: can be unpacked as model(**inputs_dict)
        - visual_mask: bool array (seq_len,) — True at image token positions
    """
    inputs = processor(images=img, text=text,
                       return_tensors='pt').to(device)                  # tokenise text + preprocess image
    input_ids = inputs['input_ids'][0].cpu()                            # (seq_len,)
    # HF processor expands <image> into 576 repeated image_token_id tokens
    visual_mask = (input_ids.numpy() == image_token_id)
    return inputs, visual_mask


def prepare_inputs_original(processor, img, text, device, image_token_id):
    """Prepare model inputs for original LLaVA.

    The original model tokenises text and preprocesses images separately.
    The model's forward method replaces the single IMAGE_TOKEN_INDEX
    placeholder with 576 image feature vectors.

    Returns: (inputs_dict, visual_mask)
        - inputs_dict: {'input_ids': ..., 'images': ...}
        - visual_mask: bool array (expanded_seq_len,) — True at 576 image positions
    """
    from llava.constants import IMAGE_TOKEN_INDEX                       # special token ID for <image>
    from llava.mm_utils import tokenizer_image_token                    # tokenise with image placeholder

    tokenizer, image_processor = processor                              # unpack from tuple

    # Tokenise text, replacing <image> with IMAGE_TOKEN_INDEX (-200)
    input_ids = tokenizer_image_token(
        text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)                                           # (1, seq_len)

    # Preprocess image using CLIP processor (resize + normalise)
    image_tensor = image_processor.preprocess(
        img, return_tensors='pt'
    )['pixel_values'].half().to(device)                                 # (1, 3, 336, 336)

    # Build visual mask for the EXPANDED sequence
    # The model replaces 1 placeholder token with 576 image features
    ids_cpu = input_ids[0].cpu().numpy()
    img_pos = int(np.where(ids_cpu == IMAGE_TOKEN_INDEX)[0][0])         # position of <image> placeholder
    n_image_features = 576                                              # LLaVA-1.5 uses 576 CLIP patches
    expanded_len = len(ids_cpu) - 1 + n_image_features                  # -1 placeholder + 576 features
    visual_mask = np.zeros(expanded_len, dtype=bool)
    visual_mask[img_pos:img_pos + n_image_features] = True              # mark image feature positions

    inputs_dict = {'input_ids': input_ids, 'images': image_tensor}
    return inputs_dict, visual_mask


# ═══════════════════════════════════════════════════════════════════
# Section 1d — InternVL backend helpers
# ═══════════════════════════════════════════════════════════════════

# --- InternVL image preprocessing (replicated from model repo) ---

IMAGENET_MEAN = (0.485, 0.456, 0.406)                                    # ImageNet normalisation constants
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _internvl_build_transform(input_size=448):
    """Build the image transform for InternVL tiles.

    Each tile is resized to (input_size × input_size), converted to
    a float tensor, and normalised with ImageNet statistics.  This
    matches the transform used during InternVL training.
    """
    import torchvision.transforms as T                                    # lazy import — only needed for internvl
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),                                # resize to exact tile size
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),                                                     # HWC uint8 → CHW float [0,1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),                # ImageNet normalisation
    ])

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the optimal tile grid for an image's aspect ratio.

    Given the image aspect ratio and a set of candidate grid ratios
    (e.g. 1×1, 1×2, 2×1, …), returns the grid that best matches
    while maximising resolution.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height                                                 # original image area
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]                               # aspect ratio of this grid
        ratio_diff = abs(aspect_ratio - target_aspect)                    # how close to original
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Tie-break: prefer the grid that uses more pixels
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Split an image into tiles using InternVL's dynamic resolution strategy.

    The image is divided into a grid of (rows × cols) tiles, each of size
    image_size × image_size.  A thumbnail (full image resized to tile size)
    is optionally appended.  This lets the model see both fine details
    (from tiles) and global context (from thumbnail).

    Returns:
        list[PIL.Image] — the tiles (+ optional thumbnail).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height                               # width / height

    # Build all candidate grid shapes up to max_num tiles
    target_ratios = set()
    for n in range(min_num, max_num + 1):                                 # n = total number of tiles
        for i in range(1, n + 1):                                         # i = rows
            for j in range(1, n + 1):                                     # j = cols
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])     # sort by total tiles

    # Pick the best grid for this image's aspect ratio
    best_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width  = best_ratio[0] * image_size                            # total pixel width of the grid
    target_height = best_ratio[1] * image_size                            # total pixel height of the grid

    # Resize the full image to fit the chosen grid exactly
    resized = image.resize((target_width, target_height))

    # Slice into individual tiles
    tiles = []
    for i in range(best_ratio[1]):                                        # row index
        for j in range(best_ratio[0]):                                    # col index
            box = (j * image_size, i * image_size,
                   (j + 1) * image_size, (i + 1) * image_size)
            tiles.append(resized.crop(box))

    # Optionally append a thumbnail (full image → single tile)
    if use_thumbnail and len(tiles) != 1:
        thumbnail = image.resize((image_size, image_size))
        tiles.append(thumbnail)

    return tiles


def load_model_internvl(model_path, device):
    """Load InternVL2.5-8B from a local directory.

    Uses AutoTokenizer + AutoModel with trust_remote_code=True (InternVL
    ships its own modelling code inside the weights folder).

    Workarounds for transformers ≥4.48 + accelerate:
      1. torch.linspace monkey-patch: InternVL's vision encoder calls
         torch.linspace(...).item() during __init__.  accelerate creates
         tensors on a meta device, where .item() is illegal.  We force
         CPU creation.
      2. all_tied_weights_keys patch: newer transformers expects this
         attribute on the model, but InternVL's custom code only defines
         _tied_weights_keys.  We patch the method that reads it.

    Returns: (model, tokenizer, image_token_id)
        - model         : InternVLChatModel (vision encoder + LLM)
        - tokenizer     : InternLM2 tokenizer
        - image_token_id: token ID for <IMG_CONTEXT>
    """
    from transformers import AutoTokenizer, AutoModel                      # standard HF AutoClasses
    import transformers.modeling_utils as _mu                              # for PreTrainedModel internals

    tokenizer = AutoTokenizer.from_pretrained(                             # load bundled tokenizer
        model_path, trust_remote_code=True)

    # -- Patch 1: force torch.linspace onto CPU (avoid meta-tensor .item() crash) --
    _orig_linspace = torch.linspace                                        # save original
    def _cpu_linspace(*args, **kwargs):                                    # replacement: always CPU
        kwargs['device'] = 'cpu'
        return _orig_linspace(*args, **kwargs)
    torch.linspace = _cpu_linspace

    # -- Patch 2: add missing all_tied_weights_keys attribute ----------------
    _orig_adjust = getattr(_mu.PreTrainedModel,
                           '_adjust_tied_keys_with_tied_pointers', None)
    if _orig_adjust is not None:
        def _safe_adjust(self, *a, **kw):
            if not hasattr(self, 'all_tied_weights_keys'):
                self.all_tied_weights_keys = set()                         # create as empty set
            return _orig_adjust(self, *a, **kw)
        _mu.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,                                    # InternVL's native dtype
            trust_remote_code=True,                                        # required: custom model code
        ).to(device).eval()
    finally:
        torch.linspace = _orig_linspace                                    # always restore
        if _orig_adjust is not None:
            _mu.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _orig_adjust

    # Resolve IMG_CONTEXT token ID — used to identify visual positions
    image_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

    return model, tokenizer, image_token_id


def prepare_inputs_internvl(tokenizer, model, img, desc_text, device,
                            image_token_id, text_source='generated'):
    """Prepare teacher-forcing inputs for InternVL.

    Constructs the full chat-template input sequence with:
      <|im_start|>user\\n<img>{IMG_CONTEXT * n_patches}</img>\\n{question}<|im_end|>
      <|im_start|>assistant\\n{description}<|im_end|>

    The image is split into tiles via dynamic resolution, each tile
    producing 256 visual tokens through the vision encoder + projector.

    Returns: (inputs_dict, visual_mask, n_vis_tokens)
        - inputs_dict  : {pixel_values, input_ids, attention_mask} for model()
        - visual_mask  : bool array (seq_len,) — True at IMG_CONTEXT positions
        - n_vis_tokens : int — number of visual token positions
    """
    # 1. Preprocess image into tiles
    tiles = _dynamic_preprocess(img, min_num=1, max_num=12, image_size=448)
    transform = _internvl_build_transform(input_size=448)
    pixel_values = torch.stack([transform(t) for t in tiles])             # (n_tiles, 3, 448, 448)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

    # 2. Compute number of visual tokens
    #    InternViT-300M: 448/14 = 32 patches per side → 32² = 1024
    #    pixel_shuffle 2× downsample: 1024 / 4 = 256 tokens per tile
    #    Prefer model.num_image_token if available (future-proof).
    num_patches = pixel_values.shape[0]                                    # number of tiles
    tokens_per_tile = getattr(model, 'num_image_token', 256)              # InternVL2.5-8B: 256
    n_vis_tokens = num_patches * tokens_per_tile

    # 3. Build the full text prompt — split around the image block so we
    #    can inject IMG_CONTEXT token IDs manually (the tokenizer may not
    #    recognize <IMG_CONTEXT> as a single special token, causing a
    #    size mismatch between the prompt and vision-encoder output).
    question = 'Could you describe the image?'

    if text_source == 'generated':
        pre_img  = '<|im_start|>user\n<img>'                              # text before image tokens
        post_img = (f'</img>\n{question}<|im_end|>\n'                     # text after image tokens
                    f'<|im_start|>assistant\n{desc_text}<|im_end|>')
    else:
        pre_img  = '<|im_start|>user\n<img>'
        post_img = (f'</img>\n{desc_text}<|im_end|>\n'
                    f'<|im_start|>assistant\n<|im_end|>')

    # 4. Tokenize text parts separately, inject image token IDs in between
    pre_ids  = tokenizer.encode(pre_img,  add_special_tokens=False)       # token IDs before image
    post_ids = tokenizer.encode(post_img, add_special_tokens=False)       # token IDs after image
    img_ids  = [image_token_id] * n_vis_tokens                            # explicit IMG_CONTEXT IDs

    all_ids  = pre_ids + img_ids + post_ids                               # concatenate
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)  # (1, seq_len)

    # 5. Build visual mask: True where input_ids == IMG_CONTEXT token ID
    ids_cpu = input_ids[0].cpu().numpy()
    visual_mask = (ids_cpu == image_token_id)

    # 6. Assemble inputs dict for model.forward()
    inputs_dict = {
        'pixel_values': pixel_values,                                      # (n_tiles, 3, 448, 448)
        'input_ids': input_ids,                                            # (1, seq_len)
        'attention_mask': torch.ones_like(input_ids),                      # all ones (no padding)
        'image_flags': torch.ones(num_patches, dtype=torch.long,           # 1 = real tile (not padding)
                                  device=device),                          # (n_tiles,)
    }
    return inputs_dict, visual_mask, n_vis_tokens


def load_model_llava_ov(model_path, device):
    """Load LLaVA-OneVision from HuggingFace.

    Architecture: SigLIP vision encoder → 2-layer MLP projector → Qwen2-7B backbone.
    Uses LlavaOnevisionForConditionalGeneration + AutoProcessor.

    Returns: (model, processor, image_token_id)
        - model          : LlavaOnevisionForConditionalGeneration
        - processor      : AutoProcessor (text tokenizer + SigLIP image processor)
        - image_token_id : int — token ID for <image> placeholders (expanded by processor)
    """
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)                  # loads tokenizer + image processor
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(        # load model
        model_path,
        torch_dtype=torch.bfloat16,                                        # Qwen2 native dtype
        low_cpu_mem_usage=True,
    ).to(device).eval()

    image_token_id = model.config.image_token_index                        # e.g. 151646 for Qwen2 vocab
    return model, processor, image_token_id


def prepare_inputs_llava_ov(processor, img, desc_text, device,
                            image_token_id, text_source='generated'):
    """Prepare teacher-forcing inputs for LLaVA-OneVision.

    Constructs the full Qwen2 chat-template input with apply_chat_template(),
    then runs the processor to expand image placeholders into visual tokens
    (SigLIP patches from anyres-9 tiling).

    Returns: (inputs_dict, visual_mask, n_vis_tokens)
        - inputs_dict  : dict for model(**inputs_dict)
        - visual_mask  : bool array (seq_len,) — True at image token positions
        - n_vis_tokens : int — number of visual token positions
    """
    question = 'Could you describe the image?'

    if text_source == 'generated':
        # Teacher forcing: user asks question, assistant provides full description
        messages = [
            {"role": "user", "content": [
                {"type": "image"},                                         # image placeholder
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": desc_text},                      # pre-filled description
            ]},
        ]
    else:
        # COCO caption mode: caption as user input
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": desc_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": ""},
            ]},
        ]

    # Apply Qwen2 chat template → formatted string with special tokens
    text = processor.apply_chat_template(                                  # converts messages → model input string
        messages, tokenize=False, add_generation_prompt=False)            # no extra assistant prefix (already in messages)

    inputs = processor(                                                    # processor handles image tiling + tokenisation
        images=img,
        text=text,
        return_tensors='pt',
    ).to(device)

    # Build visual mask: True where input_ids == image_token_id
    input_ids = inputs['input_ids'][0].cpu().numpy()                       # (seq_len,)
    visual_mask = (input_ids == image_token_id)
    n_vis_tokens = int(visual_mask.sum())

    return inputs, visual_mask, n_vis_tokens


def load_model_qwen2vl(model_path, device):
    """Load Qwen2.5-VL from a local directory.

    Returns: (model, processor, image_token_id)
        - model          : Qwen2_5_VLForConditionalGeneration
        - processor      : AutoProcessor (text + dynamic-resolution image preprocessing)
        - image_token_id : int — token ID for <|image_pad|>
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)                  # loads tokenizer + image processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,                                        # Qwen's native dtype
    ).to(device).eval()

    # Qwen2.5-VL uses <|image_pad|> as the visual placeholder token
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')

    return model, processor, image_token_id


def prepare_inputs_qwen2vl(processor, img, desc_text, device,
                           image_token_id, text_source='generated'):
    """Prepare teacher-forcing inputs for Qwen2.5-VL.

    Constructs the Qwen2-style chat template with the generated description
    as the assistant response (teacher forcing). The processor handles
    dynamic-resolution image tiling internally via process_vision_info().

    Returns: (inputs_dict, visual_mask, n_vis_tokens)
    """
    from qwen_vl_utils import process_vision_info                          # Qwen helper: PIL → tensors

    # 1. Build chat messages — teacher forcing puts description as assistant reply
    if text_source == 'generated':
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},                           # PIL Image passed directly
                {"type": "text",  "text": "Could you describe the image?"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": desc_text},                       # teacher-forcing: full description
            ]},
        ]
    else:
        # COCO caption mode: caption as user text, no assistant reply
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": desc_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": ""},
            ]},
        ]

    # 2. Apply chat template → formatted string with special tokens
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)

    # 3. Extract + preprocess images via Qwen helper
    image_inputs, _ = process_vision_info(messages)

    # 4. Tokenize text + preprocess image through processor
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors='pt',
    ).to(device)

    # 5. Build visual mask: True where input_ids == image_pad token
    ids_cpu = inputs['input_ids'][0].cpu().numpy()
    visual_mask = (ids_cpu == image_token_id)
    n_vis_tokens = int(visual_mask.sum())

    return inputs, visual_mask, n_vis_tokens


def parse_args():
    p = argparse.ArgumentParser(
        description='Neuron classification following Xu et al. (MM 2025)')

    # Model type and paths
    p.add_argument('--model_type', default='llava-hf',
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
                   help='"llava-hf" for llava-hf/llava-1.5-7b-hf, '
                        '"llava-liuhaotian" for liuhaotian/llava-v1.5-7b, '
                        '"internvl" for InternVL2.5-8B, '
                        '"qwen2vl" for Qwen2.5-VL-7B, '
                        '"llava-ov" for llava-hf/llava-onevision-qwen2-7b-ov-hf')
    p.add_argument('--model_path', default=None,
                   help='Local path to model weights (used by internvl, llava-ov backends)')
    p.add_argument('--model', default='llava-1.5-7b',
                   help='Short model name for output directory')
    p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
                   help='HuggingFace model ID (for --model_type llava-hf)')
    p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b',
                   help='Original LLaVA model path (for --model_type llava-liuhaotian)')
    p.add_argument('--coco_img_dir',
                   default='/home/projects/bagon/shared/coco2017/'
                           'images/train2017/')
    p.add_argument('--coco_ann_path',
                   default='/home/projects/bagon/shared/coco2017/'
                           'annotations/captions_train2017.json')
    p.add_argument('--generated_desc_path',
                   default='results/1-describe/generated_descriptions.json',
                   help='Path to LLaVA-generated descriptions JSON '
                        '(from generate_descriptions.py)')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--detail_23k_path',
                   default=os.path.join(_script_dir, '..', 'data', 'detail_23k.json'),
                   help='Path to detail_23k.json (defines image subset)')
    p.add_argument('--text_source', default='generated',
                   choices=['generated', 'coco'],
                   help='Text source: "generated" = LLaVA-generated '
                        'descriptions (Xu method), "coco" = COCO captions')
    p.add_argument('--output_dir', default='results')

    # Xu et al. hyper-parameters (Section 3.2, defaults match paper)
    p.add_argument('--num_images', type=int, default=23000,
                   help='Number of images to use (Xu used all 23K)')
    p.add_argument('--top_n', type=int, default=50,
                   help='Top-N activated samples per neuron (N in paper)')
    p.add_argument('--Tv', type=float, default=2.0,
                   help='Visual token threshold on 0-10 scale')
    p.add_argument('--nv', type=int, default=4,
                   help='Min visual tokens above Tv for visual-activated')
    p.add_argument('--Tt', type=float, default=3.0,
                   help='Text token threshold on 0-10 scale')
    p.add_argument('--nt', type=int, default=2,
                   help='Min text tokens above Tt for text-activated')

    # Permutation test parameters (permutation-test)
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Significance level for permutation test')
    p.add_argument('--n_permutations', type=int, default=1000,
                   help='Number of permutations for null distribution')

    # GPU sharding
    p.add_argument('--layer_start', type=int, default=0,
                   help='First layer to process (inclusive)')
    p.add_argument('--layer_end', type=int, default=32,
                   help='Last layer to process (exclusive)')
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image_ids', default=None,
                   help='Comma-separated COCO image IDs to process '
                        '(e.g. "000000403170,000000065793"). '
                        'When set, only these images are used instead of '
                        'the full detail_23k subset.')

    # Merge mode
    p.add_argument('--merge', action='store_true',
                   help='Merge per-shard stats and neuron labels into summary files, then exit')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Main
# ═══════════════════════════════════════════════════════════════════

def main(args=None):
    if args is None:
        args = parse_args()
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    rng = np.random.RandomState(args.seed)
    n_layers_total = args.layer_end - args.layer_start

    # ───────────────────────────────────────────────────────────
    # 3a. Load dataset
    # ───────────────────────────────────────────────────────────
    descriptions = None
    captions_dict = None

    if args.text_source == 'generated':
        # Xu's method: use LLaVA-generated descriptions as text tokens
        print(f'Loading generated descriptions from {args.generated_desc_path} ...')
        descriptions, all_image_ids, id_to_filename = load_generated_descriptions(
            args.generated_desc_path, args.detail_23k_path)
        # Only keep images that have a generated description
        image_ids = [iid for iid in all_image_ids if iid in descriptions]
        print(f'Images with descriptions: {len(image_ids)} / {len(all_image_ids)}')
    else:
        # Fallback: COCO captions
        print(f'Loading COCO captions from {args.coco_ann_path} ...')
        captions_dict, id_to_filename = load_coco_captions(args.coco_ann_path)
        image_ids = sorted([iid for iid, caps in captions_dict.items()
                            if len(caps) >= 1])

    # Use all images from detail_23k (no sampling — Xu et al. used the full subset)
    sampled_ids = list(image_ids)

    # ─── Optional: restrict to specific image IDs (e.g. viz-fig-3 mode) ───
    if args.image_ids is not None:
        requested = [x.strip() for x in args.image_ids.split(',')]
        available = set(sampled_ids)
        missing = [r for r in requested if r not in available]
        if missing:
            print(f'WARNING: {len(missing)} requested image IDs not found '
                  f'in dataset: {missing}')
        sampled_ids = [r for r in requested if r in available]
        print(f'Filtered to {len(sampled_ids)} requested image IDs')
    else:
        print(f'Using all {len(sampled_ids)} images from detail_23k')

    # ───────────────────────────────────────────────────────────
    # 3b. Load model
    # ───────────────────────────────────────────────────────────
    if args.model_type == 'llava-hf':
        print(f'Loading HF model: {args.hf_id} …')
        model, processor, image_token_id = load_model_hf(args.hf_id, device)
    elif args.model_type == 'internvl':
        _path = args.model_path or args.original_model_path                # --model_path preferred, fall back to --original_model_path
        print(f'Loading InternVL model: {_path} …')
        model, processor, image_token_id = load_model_internvl(_path, device)
        # processor = tokenizer for InternVL (used by count_description_tokens)
    elif args.model_type == 'llava-ov':
        _path = args.model_path or 'modern_vlms/pretrained/llava-onevision-qwen2-7b-ov-hf'
        print(f'Loading LLaVA-OneVision model: {_path} …')
        model, processor, image_token_id = load_model_llava_ov(_path, device)
    elif args.model_type == 'qwen2vl':
        _path = args.model_path or 'modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct'
        print(f'Loading Qwen2.5-VL model: {_path} …')
        model, processor, image_token_id = load_model_qwen2vl(_path, device)
    else:
        print(f'Loading original model: {args.original_model_path} …')
        model, processor, image_token_id = load_model_original(
            args.original_model_path, device)

    print(f'Image token ID: {image_token_id}')

    # Layer names for this GPU's assigned layers
    # Per-model layer counts:
    #   LLaVA-1.5 / LLaMA-2: 32,  InternVL2.5 / InternLM2.5: 32
    #   Qwen2.5-VL / Qwen2.5: 28,  LLaVA-OneVision / Qwen2: 28
    _n_layers_map = {
        'llava-liuhaotian': 32, 'llava-hf': 32,
        'internvl': 32, 'qwen2vl': 28, 'llava-ov': 28,
    }
    n_layers_model = _n_layers_map.get(args.model_type, 32)
    if args.layer_end > n_layers_model:
        print(f'⚠ --layer_end {args.layer_end} > {n_layers_model} layers '
              f'for {args.model_type}; clamping to {n_layers_model}')
        args.layer_end = n_layers_model
        n_layers_total = args.layer_end - args.layer_start
    all_layer_names = get_layer_names(args.model_type, n_layers_model)
    my_layers = list(range(args.layer_start, args.layer_end))
    my_layer_names = [all_layer_names[l] for l in my_layers]
    print(f'Processing layers {args.layer_start}–{args.layer_end - 1} '
          f'({n_layers_total} layers)')

    # ───────────────────────────────────────────────────────────
    # 3c. Probe to get neuron count
    # ───────────────────────────────────────────────────────────
    print('Probing model …')
    probe_id = sampled_ids[0]
    probe_img = Image.open(os.path.join(
        args.coco_img_dir, id_to_filename[probe_id])).convert('RGB')
    probe_text = build_prompt(
        probe_id, args.text_source, model_type=args.model_type,
        descriptions=descriptions if args.text_source == 'generated' else None,
        captions_dict=captions_dict if args.text_source == 'coco' else None)

    # Prepare inputs using the appropriate backend
    if args.model_type == 'llava-hf':
        probe_inputs, _ = prepare_inputs_hf(
            processor, probe_img, probe_text, device, image_token_id)
    elif args.model_type == 'internvl':
        _desc = descriptions.get(probe_id, '') if descriptions else ''
        probe_inputs, _, _probe_nvis = prepare_inputs_internvl(
            processor, model, probe_img, _desc, device,
            image_token_id, args.text_source)
    elif args.model_type == 'llava-ov':
        _desc = descriptions.get(probe_id, '') if descriptions else ''
        probe_inputs, _, _probe_nvis = prepare_inputs_llava_ov(
            processor, probe_img, _desc, device,
            image_token_id, args.text_source)
    elif args.model_type == 'qwen2vl':
        _desc = descriptions.get(probe_id, '') if descriptions else ''
        probe_inputs, _, _probe_nvis = prepare_inputs_qwen2vl(
            processor, probe_img, _desc, device,
            image_token_id, args.text_source)
    else:
        probe_inputs, _ = prepare_inputs_original(
            processor, probe_img, probe_text, device, image_token_id)

    with torch.no_grad():
        with TraceDict(model, my_layer_names) as td:
            model(**probe_inputs)

    out = td[my_layer_names[0]].output
    if isinstance(out, tuple):
        out = out[0]
    n_neurons = out.shape[-1]   # 11,008 for LLaVA, 14,336 for InternVL
    print(f'Neurons per layer: {n_neurons}')

    # ═══════════════════════════════════════════════════════════
    # Section 4 — Top-N Heap: find top-N samples + global max
    # ═══════════════════════════════════════════════════════════
    #
    # For each (layer, neuron) we maintain:
    #   top_n_acts[li][neuron, rank] — activation value
    #   top_n_sids[li][neuron, rank] — sample index into sampled_ids
    #   global_max[li][neuron]       — max activation seen so far
    #
    # Ranking criterion (same as Xu): max activation across ALL token
    # positions in one forward pass.

    print(f'\n{"="*60}')
    print(f'TOP-N HEAP: Scanning {len(sampled_ids)} images → '
          f'top-{args.top_n} per neuron + global max')
    print(f'{"="*60}\n')

    top_n_acts = [np.full((n_neurons, args.top_n), -np.inf, dtype=np.float32)
                  for _ in my_layers]
    top_n_sids = [np.full((n_neurons, args.top_n), -1, dtype=np.int32)
                  for _ in my_layers]
    global_max = [np.full(n_neurons, 0.0, dtype=np.float32)
                  for _ in my_layers]

    t0 = time.time()
    for sample_idx, img_id in enumerate(tqdm(sampled_ids, desc='Top-N Heap')):
        filename = id_to_filename.get(img_id)
        if filename is None:
            continue
        img_path = os.path.join(args.coco_img_dir, filename)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            tqdm.write(f'⚠ skip {img_path}: {e}')
            continue

        text = build_prompt(
            img_id, args.text_source, model_type=args.model_type,
            descriptions=descriptions if args.text_source == 'generated' else None,
            captions_dict=captions_dict if args.text_source == 'coco' else None)

        # Prepare inputs using the appropriate backend
        if args.model_type == 'internvl':
            _desc = descriptions.get(img_id, '') if descriptions else \
                    (captions_dict[img_id][0] if captions_dict else '')
            inputs, visual_positions, _ = prepare_inputs_internvl(
                processor, model, img, _desc, device,
                image_token_id, args.text_source)
        elif args.model_type == 'llava-ov':
            _desc = descriptions.get(img_id, '') if descriptions else \
                    (captions_dict[img_id][0] if captions_dict else '')
            inputs, visual_positions, _ = prepare_inputs_llava_ov(
                processor, img, _desc, device,
                image_token_id, args.text_source)
        elif args.model_type == 'qwen2vl':
            _desc = descriptions.get(img_id, '') if descriptions else \
                    (captions_dict[img_id][0] if captions_dict else '')
            inputs, visual_positions, _ = prepare_inputs_qwen2vl(
                processor, img, _desc, device,
                image_token_id, args.text_source)
        elif args.model_type == 'llava-hf':
            inputs, visual_positions = prepare_inputs_hf(
                processor, img, text, device, image_token_id)
        else:
            inputs, visual_positions = prepare_inputs_original(
                processor, img, text, device, image_token_id)

        # Build content mask: visual + description positions only.
        # Template tokens (BOS, USER:, prompt, ASSISTANT:) are excluded
        # so they cannot influence top-N rankings or global_max.
        if args.text_source == 'generated':
            desc_count = count_description_tokens(
                img_id, args.model_type, processor, descriptions)
            desc_positions = np.zeros(len(visual_positions), dtype=bool)
            desc_positions[-desc_count:] = True                        # last desc_count positions = description
            content_mask = visual_positions | desc_positions
        else:
            content_mask = np.ones(len(visual_positions), dtype=bool)   # fallback: use all positions

        with torch.no_grad():
            with TraceDict(model, my_layer_names) as td:
                model(**inputs)

        for li, layer_name in enumerate(my_layer_names):
            # Extract act_fn output = SiLU(gate_proj(x)), the first-linear activations
            out = td[layer_name].output
            if isinstance(out, tuple):
                out = out[0]
            # out: (1, seq_len, n_neurons)
            acts = out[0].float()                              # (seq_len, n_neurons)
            acts_content = acts[content_mask]                   # (n_content, n_neurons)
            max_per_neuron = acts_content.max(dim=0).values.cpu().numpy()  # (n_neurons,) — for top-N ranking

            # Update global max from ALL token positions (incl. template tokens).
            # Xu Section 3.2: "the maximum activation value of the neuron is
            # mapped to 10" — this should be the unconditional max across the
            # entire sequence, not just content tokens. Template tokens (BOS,
            # USER:, ASSISTANT:) can carry high activations; excluding them
            # deflates global_max and inflates Activation Pattern normalised values,
            # causing over-classification as multimodal.
            max_per_neuron_all = acts.max(dim=0).values.cpu().numpy()      # (n_neurons,) — all positions
            np.maximum(global_max[li], max_per_neuron_all, out=global_max[li])

            # Update top-N heaps (vectorised check, scalar insert)
            #   Compare each neuron's current activation against the
            #   weakest entry in its top-N. Only loop over neurons
            #   that actually need updating.
            min_in_topn = top_n_acts[li].min(axis=1)           # (n_neurons,)
            needs_update = max_per_neuron > min_in_topn
            for nidx in np.where(needs_update)[0]:
                worst = np.argmin(top_n_acts[li][nidx])
                top_n_acts[li][nidx, worst] = max_per_neuron[nidx]
                top_n_sids[li][nidx, worst] = sample_idx

    topn_heap_time = time.time() - t0
    print(f'\nTop-N Heap done in {topn_heap_time/60:.1f} min')
    for li, l in enumerate(my_layers):
        filled = (top_n_sids[li] >= 0).all(axis=1).sum()
        print(f'  Layer {l}: {filled}/{n_neurons} neurons have full '
              f'top-{args.top_n}')

    # ─── Save Top-N Heap outputs (allows re-running Activation Pattern+ without Top-N Heap) ───
    topn_heap_dir = os.path.join(args.output_dir, args.model, 'llm_fixed_threshold', 'topn_heap')
    os.makedirs(topn_heap_dir, exist_ok=True)
    for li, l in enumerate(my_layers):
        np.save(os.path.join(topn_heap_dir, f'top_n_sids_layer{l}.npy'), top_n_sids[li])
        np.save(os.path.join(topn_heap_dir, f'top_n_acts_layer{l}.npy'), top_n_acts[li])
        np.save(os.path.join(topn_heap_dir, f'global_max_layer{l}.npy'), global_max[li])
    # Save the image_id list so we can map sample_idx back to image IDs
    with open(os.path.join(topn_heap_dir, 'sampled_ids.json'), 'w') as f:
        json.dump(sampled_ids, f)
    print(f'Top-N Heap saved to {topn_heap_dir}')

    # ═══════════════════════════════════════════════════════════
    # Section 5 — Activation Pattern: classify each top-N sample
    # ═══════════════════════════════════════════════════════════
    #
    # For each neuron's top-N samples, re-run the forward pass and:
    #   1. Identify visual token positions (image_token_id in input_ids)
    #   2. Normalise activations: clamp(act / global_max * 10, 0, 10)
    #   3. Count visual tokens with norm_act > Tv
    #   4. Count text tokens with norm_act > Tt
    #   5. Classify sample as visual / text / multimodal / unknown
    #
    # Optimisation: build reverse index (sample → neurons that need it)
    # so we only process relevant neurons per forward pass.

    print(f'\n{"="*60}')
    print(f'ACTIVATION PATTERN: Classifying top-{args.top_n} samples per neuron')
    print(f'{"="*60}\n')

    # 5a. Build reverse index: sample_idx → [(li, nidx, rank), ...]
    #     This avoids a costly search during the forward-pass loop.
    sample_to_entries = defaultdict(list)
    for li in range(n_layers_total):
        for nidx in range(n_neurons):
            for rank in range(args.top_n):
                sid = int(top_n_sids[li][nidx, rank])
                if sid >= 0:
                    sample_to_entries[sid].append((li, nidx, rank))

    needed_samples = sorted(sample_to_entries.keys())
    print(f'Need to re-process {len(needed_samples)} unique images')

    # 5b. Allocate storage for per-sample classification results (Xu method)
    sample_vis_act = [np.zeros((n_neurons, args.top_n), dtype=bool)
                      for _ in my_layers]
    sample_txt_act = [np.zeros((n_neurons, args.top_n), dtype=bool)
                      for _ in my_layers]

    # 5c. Allocate storage for raw normalised activations (permutation test)
    #     vis: image patch tokens per sample (576 for LLaVA, variable for InternVL)
    #     txt: variable length, padded to MAX_TXT; actual length in txt_len_all
    if args.model_type == 'internvl':
        N_VIS = 3072                                          # max 12 tiles × 256 tokens/tile
    elif args.model_type == 'llava-ov':
        N_VIS = 4096                                          # SigLIP anyres-9: up to ~10 crops, variable tokens
    elif args.model_type == 'qwen2vl':
        N_VIS = 4096                                          # dynamic resolution ViT, variable token count
    else:
        N_VIS = 576                                           # LLaVA-1.5 fixed 576 CLIP patches
    MAX_TXT = 300                                             # max text tokens (descriptions ~100-200)
    vis_norm_all = [np.zeros((n_neurons, args.top_n, N_VIS), dtype=np.float16)
                    for _ in my_layers]
    vis_len_all = [np.full((n_neurons, args.top_n), N_VIS, dtype=np.int16)
                   for _ in my_layers]                        # actual vis token count per sample
    txt_norm_all = [np.zeros((n_neurons, args.top_n, MAX_TXT), dtype=np.float16)
                    for _ in my_layers]
    txt_len_all = [np.zeros((n_neurons, args.top_n), dtype=np.int16)
                   for _ in my_layers]

    t0 = time.time()
    for sample_idx in tqdm(needed_samples, desc='Activation Pattern'):
        img_id = sampled_ids[sample_idx]
        filename = id_to_filename.get(img_id)
        if filename is None:
            continue
        img_path = os.path.join(args.coco_img_dir, filename)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        text = build_prompt(
            img_id, args.text_source, model_type=args.model_type,
            descriptions=descriptions if args.text_source == 'generated' else None,
            captions_dict=captions_dict if args.text_source == 'coco' else None)

        # Prepare inputs and get visual/text position mask
        n_vis_this = N_VIS                                    # default (LLaVA: always 576)
        if args.model_type == 'internvl':
            _desc = descriptions.get(img_id, '') if descriptions else \
                    (captions_dict[img_id][0] if captions_dict else '')
            inputs, visual_positions, n_vis_this = prepare_inputs_internvl(
                processor, model, img, _desc, device,
                image_token_id, args.text_source)
        elif args.model_type == 'llava-ov':
            _desc = descriptions.get(img_id, '') if descriptions else \
                    (captions_dict[img_id][0] if captions_dict else '')
            inputs, visual_positions, n_vis_this = prepare_inputs_llava_ov(
                processor, img, _desc, device,
                image_token_id, args.text_source)
        elif args.model_type == 'qwen2vl':
            _desc = descriptions.get(img_id, '') if descriptions else \
                    (captions_dict[img_id][0] if captions_dict else '')
            inputs, visual_positions, n_vis_this = prepare_inputs_qwen2vl(
                processor, img, _desc, device,
                image_token_id, args.text_source)
        elif args.model_type == 'llava-hf':
            inputs, visual_positions = prepare_inputs_hf(
                processor, img, text, device, image_token_id)
        else:
            inputs, visual_positions = prepare_inputs_original(
                processor, img, text, device, image_token_id)

        # Build text mask: description tokens ONLY (Xu Figures 1-3).
        # Template tokens (BOS, USER:, prompt, ASSISTANT:) are excluded
        # — they are constant across all samples and not "text content".
        if args.text_source == 'generated':
            desc_count = count_description_tokens(
                img_id, args.model_type, processor, descriptions)
            text_positions = np.zeros(len(visual_positions), dtype=bool)
            text_positions[-desc_count:] = True                        # last desc_count positions = description
        else:
            # Fallback for coco mode: no clear description boundary,
            # treat all non-visual positions as text (original behaviour)
            text_positions = ~visual_positions

        with torch.no_grad():
            with TraceDict(model, my_layer_names) as td:
                model(**inputs)

        # Group this sample's entries by layer for efficient processing
        entries_by_layer = defaultdict(list)
        for li, nidx, rank in sample_to_entries[sample_idx]:
            entries_by_layer[li].append((nidx, rank))

        for li, nr_list in entries_by_layer.items():
            layer_name = my_layer_names[li]
            out = td[layer_name].output
            if isinstance(out, tuple):
                out = out[0]
            acts = out[0].float().cpu().numpy()        # (seq_len, n_neurons)

            # Extract only the neurons we need for this sample
            neuron_indices = np.array([nidx for nidx, _ in nr_list])
            ranks = [rank for _, rank in nr_list]

            # Per-token activations for relevant neurons: (seq_len, n_rel)
            rel_acts = acts[:, neuron_indices]

            # Normalise to [0, 10] per neuron (Xu Section 3.2)
            #   "negative values are assigned to 0 and the maximum
            #    activation value of the neuron is mapped to 10"
            gmax = global_max[li][neuron_indices]               # (n_rel,)
            gmax = np.maximum(gmax, 1e-8)
            rel_norm = np.clip(rel_acts / gmax[None, :] * 10.0, 0.0, 10.0)

            # Count visual tokens with norm activation > Tv
            vis_tok_acts = rel_norm[visual_positions, :]        # (n_vis, n_rel)
            n_vis_high = (vis_tok_acts > args.Tv).sum(axis=0)   # (n_rel,)

            # Count text tokens with norm activation > Tt
            txt_tok_acts = rel_norm[text_positions, :]          # (n_txt, n_rel)
            n_txt_high = (txt_tok_acts > args.Tt).sum(axis=0)   # (n_rel,)

            # Classify this sample for each neuron
            vis_activated = n_vis_high >= args.nv                # (n_rel,)
            txt_activated = n_txt_high >= args.nt                # (n_rel,)

            # Number of text tokens in this sample (for padding)
            n_txt_this = txt_tok_acts.shape[0]
            n_txt_clipped = min(n_txt_this, MAX_TXT)

            # Store results (Xu booleans + raw activations for permutation)
            for i, (nidx, rank) in enumerate(nr_list):
                # Xu classification booleans
                sample_vis_act[li][nidx, rank] = vis_activated[i]
                sample_txt_act[li][nidx, rank] = txt_activated[i]
                # Raw normalised activations for permutation test
                n_vis_actual = vis_tok_acts.shape[0]
                n_vis_clipped = min(n_vis_actual, N_VIS)
                vis_norm_all[li][nidx, rank, :n_vis_clipped] = \
                    vis_tok_acts[:n_vis_clipped, i].astype(np.float16)
                vis_len_all[li][nidx, rank] = n_vis_clipped
                txt_norm_all[li][nidx, rank, :n_txt_clipped] = \
                    txt_tok_acts[:n_txt_clipped, i].astype(np.float16)
                txt_len_all[li][nidx, rank] = n_txt_clipped

    act_pattern_time = time.time() - t0
    print(f'\nActivation Pattern done in {act_pattern_time/60:.1f} min')

    # ─── Save Activation Pattern raw activations (allows re-running permutation-test without GPU) ───
    act_pattern_dir = os.path.join(args.output_dir, args.model, 'llm_fixed_threshold', 'act_pattern_raw')
    os.makedirs(act_pattern_dir, exist_ok=True)
    for li, l in enumerate(my_layers):
        np.savez_compressed(
            os.path.join(act_pattern_dir, f'raw_acts_layer{l}.npz'),
            vis_acts=vis_norm_all[li],      # (n_neurons, top_n, N_VIS) float16
            vis_lengths=vis_len_all[li],    # (n_neurons, top_n) int16
            txt_acts=txt_norm_all[li],      # (n_neurons, top_n, MAX_TXT) float16
            txt_lengths=txt_len_all[li],    # (n_neurons, top_n) int16
        )
    print(f'Activation Pattern raw activations saved to {act_pattern_dir}')

    # ═══════════════════════════════════════════════════════════
    # Section 6 — Fixed-threshold: compute (pv, pt, pm, pu), assign labels
    # ═══════════════════════════════════════════════════════════
    #
    # Per Xu et al.:
    #   Visual sample     = visual-activated AND NOT text-activated
    #   Text sample       = text-activated AND NOT visual-activated
    #   Multimodal sample = both
    #   Unknown sample    = neither
    #   pv = n_visual_samples / N,  pt = n_text / N, etc.
    #   Label = argmax(pv, pt, pm, pu)  →  "X-prone" neuron

    print(f'\n{"="*60}')
    print(f'FIXED-THRESHOLD: Computing probabilities → neuron labels')
    print(f'{"="*60}\n')

    out_base = os.path.join(args.output_dir, args.model, 'llm_fixed_threshold')
    os.makedirs(out_base, exist_ok=True)

    total_stats = {'visual': 0, 'text': 0, 'multimodal': 0, 'unknown': 0}

    for li, l in enumerate(my_layers):
        layer_name = all_layer_names[l]
        va = sample_vis_act[li]          # (n_neurons, top_n) bool
        ta = sample_txt_act[li]          # (n_neurons, top_n) bool
        valid = top_n_sids[li] >= 0      # (n_neurons, top_n) bool

        # Count each sample type per neuron
        n_valid = valid.sum(axis=1).clip(min=1).astype(np.float32)
        n_vis   = ( va & ~ta & valid).sum(axis=1)     # visual-only
        n_txt   = (~va &  ta & valid).sum(axis=1)     # text-only
        n_multi = ( va &  ta & valid).sum(axis=1)     # both
        n_unkn  = (~va & ~ta & valid).sum(axis=1)     # neither

        # Probabilities
        pv = n_vis   / n_valid
        pt = n_txt   / n_valid
        pm = n_multi / n_valid
        pu = n_unkn  / n_valid

        # Label = argmax
        probs = np.stack([pv, pt, pm, pu], axis=1)    # (n_neurons, 4)
        label_idx = np.argmax(probs, axis=1)
        label_map = ['visual', 'text', 'multimodal', 'unknown']

        # Build per-neuron output and layer stats
        neuron_labels = []
        layer_stats = {'visual': 0, 'text': 0, 'multimodal': 0, 'unknown': 0}

        for nidx in range(n_neurons):
            label = label_map[label_idx[nidx]]
            layer_stats[label] += 1
            neuron_labels.append({
                'neuron_idx': nidx,
                'label': label,
                'pv': round(float(pv[nidx]), 4),
                'pt': round(float(pt[nidx]), 4),
                'pm': round(float(pm[nidx]), 4),
                'pu': round(float(pu[nidx]), 4),
                'global_max_activation': round(float(global_max[li][nidx]), 6),
                'top_n_valid': int(n_valid[nidx]),
            })

        # Save per-layer results
        layer_dir = os.path.join(out_base, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        label_path = os.path.join(layer_dir, 'neuron_labels.json')
        with open(label_path, 'w') as f:
            json.dump(neuron_labels, f, indent=2)

        t = sum(layer_stats.values())
        print(f'Layer {l:2d}: '
              f'vis={layer_stats["visual"]:5d} ({100*layer_stats["visual"]/t:.1f}%)  '
              f'txt={layer_stats["text"]:5d} ({100*layer_stats["text"]/t:.1f}%)  '
              f'multi={layer_stats["multimodal"]:5d} ({100*layer_stats["multimodal"]/t:.1f}%)  '
              f'unkn={layer_stats["unknown"]:5d} ({100*layer_stats["unknown"]/t:.1f}%)')

        for k in total_stats:
            total_stats[k] += layer_stats[k]

    # ═══════════════════════════════════════════════════════════
    # Section 7 — Save config and summary
    # ═══════════════════════════════════════════════════════════
    total = sum(total_stats.values())
    print(f'\n{"─"*60}')
    print(f'OVERALL ({total:,} neurons):')
    print(f'  Visual:     {total_stats["visual"]:6,} '
          f'({100*total_stats["visual"]/total:.1f}%)')
    print(f'  Text:       {total_stats["text"]:6,} '
          f'({100*total_stats["text"]/total:.1f}%)')
    print(f'  Multimodal: {total_stats["multimodal"]:6,} '
          f'({100*total_stats["multimodal"]/total:.1f}%)')
    print(f'  Unknown:    {total_stats["unknown"]:6,} '
          f'({100*total_stats["unknown"]/total:.1f}%)')
    print(f'{"─"*60}')
    print(f'Total time: {(topn_heap_time + act_pattern_time)/60:.1f} min '
          f'(Top-N Heap: {topn_heap_time/60:.1f}, Activation Pattern: {act_pattern_time/60:.1f})')

    config = {
        'method': 'xu_et_al_mm2025',
        'reference': 'Deciphering Functions of Neurons in VLMs, Xu et al., MM 2025',
        'model': args.model,
        'model_type': args.model_type,
        'hf_id': args.hf_id if args.model_type == 'llava-hf'
                else (args.model_path or args.original_model_path),
        'dataset': f'detail_23k_generated' if args.text_source == 'generated'
                  else 'coco_train2017',
        'text_source': args.text_source,
        'num_images_processed': len(sampled_ids),
        'top_n': args.top_n,
        'thresholds': {
            'Tv': args.Tv, 'nv': args.nv,
            'Tt': args.Tt, 'nt': args.nt,
        },
        'layer_start': args.layer_start,
        'layer_end': args.layer_end,
        'scope': 'llm_act_fn_output (SiLU(gate_proj(x)), 11008-dim, per Xu refs [22,29])',
        'neurons_per_layer': n_neurons,
        'stats': total_stats,
        'seed': args.seed,
        'topn_heap_time_sec': round(topn_heap_time, 1),
        'act_pattern_time_sec': round(act_pattern_time, 1),
    }

    config_path = os.path.join(
        out_base,
        f'classification_stats_layers{args.layer_start}-{args.layer_end}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'\nConfig saved to {config_path}')

    # ═══════════════════════════════════════════════════════════
    # Section 8 — Permutation-test classification
    # ═══════════════════════════════════════════════════════════
    #
    # For each neuron:
    #   1. Pool visual + text activations across all top-N samples
    #   2. Apply Otsu's method for adaptive threshold
    #   3. Run permutation test (shuffle modality labels 1000×)
    #   4. Classify based on p-values at α=0.05

    _perm_device = 'gpu' if device.startswith('cuda') else 'cpu'
    print(f'\n{"="*60}')
    print(f'PERMUTATION-TEST: classification (α={args.alpha}, {_perm_device})')
    print(f'{"="*60}\n')

    perm_out_base = os.path.join(args.output_dir, args.model, 'llm_permutation')
    os.makedirs(perm_out_base, exist_ok=True)

    perm_total_stats = {'visual': 0, 'text': 0, 'multimodal': 0, 'unknown': 0}
    t0_perm = time.time()

    for li, l in enumerate(my_layers):
        layer_name = all_layer_names[l]
        valid = top_n_sids[li] >= 0      # (n_neurons, top_n) bool

        perm_labels = []
        layer_stats = {'visual': 0, 'text': 0, 'multimodal': 0, 'unknown': 0}

        for nidx in tqdm(range(n_neurons), desc=f'Layer {l} perm'):
            # Gather this neuron's visual activations across valid top-N samples
            valid_ranks = np.where(valid[nidx])[0]
            if len(valid_ranks) == 0:
                perm_labels.append({
                    'neuron_idx': nidx, 'label': 'unknown',
                    'p_value': 1.0,
                    'otsu_threshold': 0.0,
                    'observed_visual': 0, 'observed_text': 0,
                    'observed_rate_diff': 0.0,
                })
                layer_stats['unknown'] += 1
                continue

            # Concatenate visual activations: (n_valid_samples × actual_vis_len,)
            vis_acts_list = []
            for r in valid_ranks:
                vlen = int(vis_len_all[li][nidx, r])
                if vlen > 0:
                    vis_acts_list.append(
                        vis_norm_all[li][nidx, r, :vlen].astype(np.float32))
            vis_acts = np.concatenate(vis_acts_list) if vis_acts_list else np.array([], dtype=np.float32)

            # Concatenate text activations: (n_valid_samples × actual_txt_len,)
            txt_acts_list = []
            for r in valid_ranks:
                tlen = int(txt_len_all[li][nidx, r])
                if tlen > 0:
                    txt_acts_list.append(
                        txt_norm_all[li][nidx, r, :tlen].astype(np.float32))
            txt_acts = np.concatenate(txt_acts_list) if txt_acts_list else np.array([], dtype=np.float32)

            if len(txt_acts) == 0 or len(vis_acts) == 0:
                perm_labels.append({
                    'neuron_idx': nidx, 'label': 'unknown',
                    'p_value': 1.0,
                    'otsu_threshold': 0.0,
                    'observed_visual': 0, 'observed_text': 0,
                    'observed_rate_diff': 0.0,
                })
                layer_stats['unknown'] += 1
                continue

            # Run permutation test (GPU-accelerated when available)
            if device.startswith('cuda'):
                result = classify_neuron_permutation_gpu(
                    vis_acts, txt_acts,
                    n_permutations=args.n_permutations,
                    alpha=args.alpha,
                    device=device,
                    seed=args.seed + nidx,
                )
            else:
                result = classify_neuron_permutation(
                    vis_acts, txt_acts,
                    n_permutations=args.n_permutations,
                    alpha=args.alpha,
                    rng=np.random.RandomState(args.seed + nidx),
                )
            result['neuron_idx'] = nidx
            perm_labels.append(result)
            layer_stats[result['label']] += 1

        # Save per-layer permutation results
        layer_dir = os.path.join(perm_out_base, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        label_path = os.path.join(layer_dir, 'neuron_labels_permutation.json')
        with open(label_path, 'w') as f:
            json.dump(perm_labels, f, indent=2)

        t = sum(layer_stats.values())
        print(f'Layer {l:2d}: '
              f'vis={layer_stats["visual"]:5d} ({100*layer_stats["visual"]/t:.1f}%)  '
              f'txt={layer_stats["text"]:5d} ({100*layer_stats["text"]/t:.1f}%)  '
              f'multi={layer_stats["multimodal"]:5d} ({100*layer_stats["multimodal"]/t:.1f}%)  '
              f'unkn={layer_stats["unknown"]:5d} ({100*layer_stats["unknown"]/t:.1f}%)')

        for k in perm_total_stats:
            perm_total_stats[k] += layer_stats[k]

    perm_time = time.time() - t0_perm
    total_perm = sum(perm_total_stats.values())
    print(f'\n{"─"*60}')
    print(f'PERMUTATION TEST ({total_perm:,} neurons, {perm_time/60:.1f} min):')
    print(f'  Visual:     {perm_total_stats["visual"]:6,} '
          f'({100*perm_total_stats["visual"]/total_perm:.1f}%)')
    print(f'  Text:       {perm_total_stats["text"]:6,} '
          f'({100*perm_total_stats["text"]/total_perm:.1f}%)')
    print(f'  Multimodal: {perm_total_stats["multimodal"]:6,} '
          f'({100*perm_total_stats["multimodal"]/total_perm:.1f}%)')
    print(f'  Unknown:    {perm_total_stats["unknown"]:6,} '
          f'({100*perm_total_stats["unknown"]/total_perm:.1f}%)')
    print(f'{"─"*60}')

    # Save permutation config
    perm_config = {
        'method': 'permutation_test_otsu',
        'model': args.model,
        'alpha': args.alpha,
        'n_permutations': args.n_permutations,
        'layer_start': args.layer_start,
        'layer_end': args.layer_end,
        'neurons_per_layer': n_neurons,
        'stats': perm_total_stats,
        'time_sec': round(perm_time, 1),
    }
    perm_config_path = os.path.join(
        perm_out_base,
        f'permutation_stats_layers{args.layer_start}-{args.layer_end}.json')
    with open(perm_config_path, 'w') as f:
        json.dump(perm_config, f, indent=2)
    print(f'Permutation config saved to {perm_config_path}')


def merge_results(args):
    """Merge per-shard classification stats and neuron labels into summary files.

    Handles both fixed-threshold (llm/) and permutation-test (permutation/) outputs.
    """
    import glob as glob_mod

    layer_names = get_layer_names(args.model_type, args.layer_end)
    n_layers = args.layer_end - args.layer_start

    # ── Helper to merge one output directory ──────────────────
    def _merge_dir(out_base, stats_glob, label_filename):
        if not os.path.isdir(out_base):
            print(f'  SKIP: directory not found: {out_base}')
            return

        stats_files = sorted(
            glob_mod.glob(os.path.join(out_base, stats_glob)),
            key=lambda f: int(os.path.basename(f).split('layers')[1].split('-')[0]))

        if not stats_files:
            print(f'  No {stats_glob} files found in {out_base}')
            return

        total_stats = {'visual': 0, 'text': 0, 'multimodal': 0, 'unknown': 0}
        per_layer_stats = {}
        meta = None

        for f in stats_files:
            with open(f) as fp:
                data = json.load(fp)
            if meta is None:
                meta = {k: data[k] for k in data if k not in (
                    'stats', 'layer_start', 'layer_end',
                    'topn_heap_time_sec', 'act_pattern_time_sec', 'time_sec')}
            for layer_idx in range(data['layer_start'], data['layer_end']):
                per_layer_stats[layer_idx] = data['stats']
            for k in total_stats:
                total_stats[k] += data['stats'].get(k, 0)

        found_layers = sorted(per_layer_stats.keys())
        missing = sorted(set(range(args.layer_start, args.layer_end)) - set(found_layers))
        print(f'  Found stats for {len(found_layers)}/{n_layers} layers from {len(stats_files)} shard files')
        if missing:
            print(f'  WARNING: missing layers: {missing}')
            print('  Re-run classification for the missing layers before merging.')
            return

        # Save merged stats
        summary = dict(meta) if meta else {}
        summary['layer_start'] = args.layer_start
        summary['layer_end'] = args.layer_end
        summary['stats'] = total_stats
        summary['per_layer_stats'] = {str(l): per_layer_stats[l] for l in sorted(per_layer_stats)}

        summary_path = os.path.join(out_base, stats_glob.replace('layers*', 'all').replace('_layers*', '_all'))
        # Normalise the path
        summary_path = os.path.join(out_base,
            'classification_stats_all.json' if 'classification' in stats_glob else 'permutation_stats_all.json')
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=2)

        total = sum(total_stats.values())
        print(f'  Overall ({total:,} neurons across {len(found_layers)} layers):')
        for k in ['visual', 'text', 'multimodal', 'unknown']:
            print(f'    {k:12s}: {total_stats[k]:6,} ({100*total_stats[k]/total:.1f}%)')
        print(f'  Saved → {summary_path}')

        # Merge neuron labels
        merged_labels = {}
        for l in sorted(per_layer_stats.keys()):
            layer_name = layer_names[l]
            label_path = os.path.join(out_base, layer_name, label_filename)
            if os.path.isfile(label_path):
                with open(label_path) as fp:
                    merged_labels[str(l)] = json.load(fp)
            else:
                print(f'    WARNING: missing {label_path}')

        labels_all = label_filename.replace('.json', '_all.json')
        labels_path = os.path.join(out_base, labels_all)
        with open(labels_path, 'w') as fp:
            json.dump(merged_labels, fp, indent=2)
        print(f'  Merged neuron labels for {len(merged_labels)} layers → {labels_path}')

    # ── Fixed-threshold: Xu-style classification (fixed_threshold/) ───────
    out_base_llm = os.path.join(args.output_dir, args.model, 'llm_fixed_threshold')
    print(f'\nMerging fixed-threshold (Xu-style) results from {out_base_llm}')
    _merge_dir(out_base_llm, 'classification_stats_layers*.json', 'neuron_labels.json')

    # ── Permutation-test (permutation/) ────────────────
    out_base_perm = os.path.join(args.output_dir, args.model, 'llm_permutation')
    print(f'\nMerging permutation-test results from {out_base_perm}')
    _merge_dir(out_base_perm, 'permutation_stats_layers*.json', 'neuron_labels_permutation.json')


if __name__ == '__main__':
    args = parse_args()
    if args.merge:
        merge_results(args)
    else:
        main(args)