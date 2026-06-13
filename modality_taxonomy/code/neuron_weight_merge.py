#!/usr/bin/env python3
"""
neuron_weight_merge.py — PMBT-guided neuron-selective weight merging

Implements two methods:

  Method 1 (--method text_inject):  Task Arithmetic + PMBT text mask
    Injects a math task vector (W_math - W_base) ONLY into text-classified
    neurons of the VLM backbone.  Visual and multimodal neurons are never
    touched, protecting the neurons that drive hallucination.
    Inspired by "Bring Reason to Vision" (Chen et al., ICML 2025,
    arXiv 2505.05464) but with PMBT-guided masking.

  Method 2 (--method visual_transplant): Cross-VLM Visual Neuron Transplant
    Injects a visual task vector (W_better_vlm - W_base) ONLY into
    visual-classified neurons of a weaker VLM.  Text and multimodal
    neurons are never touched.  Symmetric complement to Method 1.

Both methods:
  - Load PMBT neuron labels from your existing step-10 pipeline output
  - Apply masked update only at the relevant neuron positions
  - Save the merged model to disk
  - Optionally evaluate on POPE + VQAv2 (hallucination benchmarks)

Compatible model pairs for Method 1:
  VLM                     Base LLM          Math LLM
  ─────────────────────── ────────────────  ─────────────────────────
  LLaVA-OV-7B (Qwen2)    Qwen/Qwen2-7B    Qwen/Qwen2-Math-7B-Base
  Qwen2.5-VL-7B           Qwen2.5-7B        Qwen/Qwen2.5-Math-7B-Base

Usage examples:

  # Method 1: inject math into text neurons of LLaVA-OV
  python neuron_weight_merge.py \\
      --method text_inject \\
      --vlm_path llava-hf/llava-onevision-qwen2-7b-ov-hf \\
      --base_llm_path Qwen/Qwen2-7B \\
      --math_llm_path Qwen/Qwen2-Math-7B-Base \\
      --label_dir results/3-classify/full/llava-onevision-7b/llm_permutation \\
      --model_type llava-ov \\
      --output_dir results/16-weight-merge/llava-ov/text_inject \\
      --lambda_sweep 0.95 0.9 0.8

  # Method 2: transplant visual neurons from Qwen2.5-VL into LLaVA-OV
  python neuron_weight_merge.py \\
      --method visual_transplant \\
      --vlm_path llava-hf/llava-onevision-qwen2-7b-ov-hf \\
      --donor_vlm_path Qwen/Qwen2.5-VL-7B-Instruct \\
      --base_llm_path Qwen/Qwen2-7B \\
      --label_dir results/3-classify/full/llava-onevision-7b/llm_permutation \\
      --donor_label_dir results/3-classify/full/qwen2.5-vl-7b/llm_permutation \\
      --model_type llava-ov \\
      --output_dir results/17-weight-merge/llava-ov/visual_transplant \\
      --lambda_sweep 0.95 0.9 0.8
"""

import argparse
import json
import os
import re
import copy
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
        description='PMBT-guided neuron-selective VLM weight merging')

    # Method
    p.add_argument('--method', required=True,
                   choices=['text_inject', 'visual_transplant', 'compose'],
                   help='text_inject = Method 1 (math LLM → text neurons); '
                        'visual_transplant = Method 5 (better VLM → visual neurons); '
                        'compose = combine saved step-16 and step-17 state dicts')

    # Model paths
    p.add_argument('--vlm_path', required=True,
                   help='HF path or local path to the target VLM')
    p.add_argument('--base_llm_path', default=None,
                   help='HF path to the base LLM (shared backbone origin); '
                        'required for text_inject and visual_transplant')
    p.add_argument('--math_llm_path', default=None,
                   help='[Method 1 only] HF path to math-finetuned LLM')
    p.add_argument('--donor_vlm_path', default=None,
                   help='[Method 5 only] HF path to the donor VLM '
                        '(should have lower hallucination rate)')
    p.add_argument('--donor_label_dir', default=None,
                   help='[Method 5 only] PMBT label dir for donor VLM '
                        '(to identify its visual neurons)')
    p.add_argument('--step16_model_path', default=None,
                   help='[compose only] Path to saved step-16 state dict directory '
                        '(contains merged_state_dict.pt)')
    p.add_argument('--step17_model_path', default=None,
                   help='[compose only] Path to saved step-17 state dict directory '
                        '(contains merged_state_dict.pt)')

    # Model type
    p.add_argument('--model_type', default='llava-ov',
                   choices=['llava-ov', 'qwen2vl', 'internvl',
                            'llava-hf', 'llava-liuhaotian', 'llava-llama3',
                            'idefics2'],
                   help='VLM architecture (determines weight path naming)')

    # PMBT labels
    p.add_argument('--label_dir', default=None,
                   help='Directory with PMBT neuron_labels_permutation.json '
                        'per layer; required for text_inject and visual_transplant')
    p.add_argument('--taxonomy', default='pmbt',
                   choices=['pmbt', 'ft'],
                   help='Which taxonomy to use: pmbt (default) or ft')

    # Layer settings (auto-detected from label_dir if not specified)
    p.add_argument('--n_layers', type=int, default=None,
                   help='Number of LLM layers (auto-detected if not set)')
    p.add_argument('--n_neurons', type=int, default=None,
                   help='Neurons per MLP layer (auto-detected if not set)')

    # Merge settings
    p.add_argument('--lambda_sweep', type=float, nargs='+',
                   default=[0.95, 0.9, 0.85, 0.8],
                   help='Lambda values to sweep (BRV convention). '
                        'λ = VLM retention weight. λ=0.9 means 90%% VLM + 10%% math. '
                        'BRV best for LLaVA: λ=0.9. '
                        'Formula: θ\' = λ·θ_vlm + (1-λ)·θ_reason')

    # Merge formula
    p.add_argument('--merge_formula', default='brv',
                   choices=['additive', 'brv'],
                   help='Merge formula to use. '
                        '"brv" (default): θ\' = λ·θ_vlm + (1-λ)·θ_reason '
                        '(BRV paper exact, trades VLM capability for reasoning). '
                        '"additive": θ\' = θ_vlm + (1-λ)·τ_reason '
                        '(keeps full VLM, adds reasoning on top).')

    # Baseline: also run uniform merge (no mask) for comparison
    p.add_argument('--include_uniform_baseline', action='store_true',
                   help='Also run uniform (unmasked) merge as ablation baseline')
    p.add_argument('--include_multimodal', action='store_true',
                   help='Also run text+multimodal mask variant')
    p.add_argument('--include_visual_only', action='store_true',
                   help='Also run visual-only mask variant (negative control)')
    p.add_argument('--include_visual_multi', action='store_true',
                   help='Also run visual+multimodal mask variant')
    p.add_argument('--include_multimodal_only', action='store_true',
                   help='Also run multimodal-only mask variant')
    p.add_argument('--include_random', action='store_true',
                   help='Also run random mask with same neuron count as text (sparsity control)')
    p.add_argument('--random_seed', type=int, default=42,
                   help='Seed for random mask generation')

    # Evaluation (optional — set to skip heavy eval during debugging)
    p.add_argument('--eval_pope', action='store_true', default=True,
                   help='Evaluate on POPE after merging')
    p.add_argument('--no_eval_pope', dest='eval_pope', action='store_false')
    p.add_argument('--pope_path',
                   default='data/POPE/output/coco/coco_pope_random.json')
    p.add_argument('--pope_img_dir', default='data/val2014')
    p.add_argument('--n_pope_questions', type=int, default=500,
                   help='POPE questions to evaluate (500 is fast but sufficient)')

    p.add_argument('--eval_vqa', action='store_true', default=False,
                   help='Evaluate on VQAv2 (slow, optional)')
    p.add_argument('--vqa_path', default=None,
                   help='Path to VQAv2 validation questions JSON')
    p.add_argument('--vqa_img_dir', default=None,
                   help='Directory with VQAv2 val images')
    p.add_argument('--n_vqa_questions', type=int, default=500)

    # ── VLMEvalKit evaluation (BRV paper benchmarks) ─────────────
    # Following https://github.com/shiqichen17/VLM_Merging evaluation protocol
    p.add_argument('--eval_vlmevalkit', action='store_true', default=False,
                   help='Evaluate on BRV paper benchmarks via VLMEvalKit '
                        '(MathVista, MathVerse all splits, MathVision_MINI, DynaMath, MMStar, MM-Math)')
    p.add_argument('--vlmevalkit_benchmarks', type=str, nargs='+',
                   default=['MathVista_MINI',
                            'MathVerse_MINI',
                            'MathVerse_MINI_Vision_Only',
                            'MathVerse_MINI_Vision_Dominant',
                            'MathVerse_MINI_Vision_Intensive',
                            'MathVerse_MINI_Text_Lite',
                            'MathVerse_MINI_Text_Dominant',
                            'MathVision_MINI', 'DynaMath', 'MMStar',
                            'MM-Math'],
                   help='VLMEvalKit benchmark names (default: BRV paper benchmarks). '
                        'Sub-scores (MathVista General/Math, MMStar Math) '
                        'are reported automatically from each parent run.')
    p.add_argument('--vlmevalkit_dir', default=None,
                   help='Path to VLMEvalKit repo (auto-detected if not set)')
    p.add_argument('--vlmevalkit_n_gpus', type=int, default=1,
                   help='Number of GPUs for VLMEvalKit inference')

    # ── CHAIR evaluation (caption-based hallucination) ───────────
    p.add_argument('--eval_chair', action='store_true', default=False,
                   help='Evaluate CHAIR (caption hallucination on COCO val2014)')
    p.add_argument('--coco_ann_dir', default='data/annotations',
                   help='Path to COCO annotations dir '
                        '(containing instances_val2014.json)')
    p.add_argument('--chair_n_images', type=int, default=500,
                   help='Number of COCO images for CHAIR captioning')
    p.add_argument('--chair_max_tokens', type=int, default=64,
                   help='Max tokens for CHAIR caption generation')

    # Output
    p.add_argument('--output_dir', required=True,
                   help='Directory to save merged models and results')
    p.add_argument('--save_model', action='store_true', default=True,
                   help='Save the merged model weights (large, ~14GB per lambda). '
                        'Default: True. Use --no_save_model to disable.')
    p.add_argument('--no_save_model', dest='save_model', action='store_false',
                   help='Disable saving merged model weights (evaluation only)')
    p.add_argument('--device', default='cuda:0',
                   help='Device for evaluation (merging is always CPU)')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 2 — MLP weight path resolution
# ═══════════════════════════════════════════════════════════════════

def get_mlp_weight_paths(model_type, layer_idx):
    """Return (gate_proj_path, up_proj_path, down_proj_path) for one layer.

    These are dot-notation paths used to navigate the model state dict.
    gate_proj / up_proj: shape (d_ffn, d_model)   — rows = neurons
    down_proj:           shape (d_model, d_ffn)   — cols = neurons

    InternVL uses SwiGLU with w1/w3/w2 naming convention.
    """
    if model_type in ('llava-ov', 'qwen2vl', 'llava-hf', 'llava-liuhaotian', 'llava-llama3'):
        # Qwen2 backbone: model.language_model.layers.{i}.mlp.*
        prefix = f'model.language_model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')
    elif model_type == 'idefics2':
        # Idefics2 (Mistral backbone): model.text_model.layers.{i}.mlp.*
        prefix = f'model.text_model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')
    elif model_type == 'internvl':
        # InternLM2 backbone: language_model.model.layers.{i}.feed_forward.*
        prefix = f'language_model.model.layers.{layer_idx}.feed_forward'
        return (f'{prefix}.w1.weight',   # gate
                f'{prefix}.w3.weight',   # up
                f'{prefix}.w2.weight')   # down
    else:
        # Original LLaVA / LLaMA-2 backbone: model.layers.{i}.mlp.*
        prefix = f'model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')


def get_llm_mlp_weight_paths(layer_idx, model_type='qwen2vl'):
    """Return MLP weight paths for a standalone LLM.

    Used for the base LLM and math LLM, which have simpler path structures
    than the VLM wrappers.

    Qwen2/LLaMA style:   model.layers.{i}.mlp.gate_proj/up_proj/down_proj
    InternLM2 style:      model.layers.{i}.feed_forward.w1/w3/w2
    """
    if model_type == 'internvl':
        prefix = f'model.layers.{layer_idx}.feed_forward'
        return (f'{prefix}.w1.weight',   # gate
                f'{prefix}.w3.weight',   # up
                f'{prefix}.w2.weight')   # down
    else:
        prefix = f'model.layers.{layer_idx}.mlp'
        return (f'{prefix}.gate_proj.weight',
                f'{prefix}.up_proj.weight',
                f'{prefix}.down_proj.weight')


def get_attn_weight_paths(model_type, layer_idx):
    """Return attention weight paths for a VLM layer.

    Returns dict of {name: path} for q_proj, k_proj, v_proj, o_proj.
    InternVL uses fused wqkv, so returns different keys.
    """
    if model_type in ('llava-ov', 'qwen2vl', 'llava-hf', 'llava-liuhaotian', 'llava-llama3'):
        prefix = f'model.language_model.layers.{layer_idx}.self_attn'
        return {
            'q': f'{prefix}.q_proj.weight',
            'k': f'{prefix}.k_proj.weight',
            'v': f'{prefix}.v_proj.weight',
            'o': f'{prefix}.o_proj.weight',
        }
    elif model_type == 'idefics2':
        prefix = f'model.text_model.layers.{layer_idx}.self_attn'
        return {
            'q': f'{prefix}.q_proj.weight',
            'k': f'{prefix}.k_proj.weight',
            'v': f'{prefix}.v_proj.weight',
            'o': f'{prefix}.o_proj.weight',
        }
    elif model_type == 'internvl':
        prefix = f'language_model.model.layers.{layer_idx}.attention'
        return {
            'qkv': f'{prefix}.wqkv.weight',
            'o':   f'{prefix}.wo.weight',
        }
    else:
        prefix = f'model.layers.{layer_idx}.self_attn'
        return {
            'q': f'{prefix}.q_proj.weight',
            'k': f'{prefix}.k_proj.weight',
            'v': f'{prefix}.v_proj.weight',
            'o': f'{prefix}.o_proj.weight',
        }


def get_llm_attn_weight_paths(layer_idx, model_type='qwen2vl'):
    """Return attention weight paths for a standalone LLM."""
    if model_type == 'internvl':
        prefix = f'model.layers.{layer_idx}.attention'
        return {
            'qkv': f'{prefix}.wqkv.weight',
            'o':   f'{prefix}.wo.weight',
        }
    else:
        prefix = f'model.layers.{layer_idx}.self_attn'
        return {
            'q': f'{prefix}.q_proj.weight',
            'k': f'{prefix}.k_proj.weight',
            'v': f'{prefix}.v_proj.weight',
            'o': f'{prefix}.o_proj.weight',
        }


def get_layernorm_paths(model_type, layer_idx):
    """Return layer norm weight paths for a VLM layer.

    Returns dict of {name: path} for input_layernorm, post_attention_layernorm.
    """
    if model_type in ('llava-ov', 'qwen2vl', 'llava-hf', 'llava-liuhaotian', 'llava-llama3'):
        prefix = f'model.language_model.layers.{layer_idx}'
        return {
            'input_ln': f'{prefix}.input_layernorm.weight',
            'post_attn_ln': f'{prefix}.post_attention_layernorm.weight',
        }
    elif model_type == 'idefics2':
        prefix = f'model.text_model.layers.{layer_idx}'
        return {
            'input_ln': f'{prefix}.input_layernorm.weight',
            'post_attn_ln': f'{prefix}.post_attention_layernorm.weight',
        }
    elif model_type == 'internvl':
        prefix = f'language_model.model.layers.{layer_idx}'
        return {
            'input_ln': f'{prefix}.attention_norm.weight',
            'post_attn_ln': f'{prefix}.ffn_norm.weight',
        }
    else:
        prefix = f'model.layers.{layer_idx}'
        return {
            'input_ln': f'{prefix}.input_layernorm.weight',
            'post_attn_ln': f'{prefix}.post_attention_layernorm.weight',
        }


def get_llm_layernorm_paths(layer_idx, model_type='qwen2vl'):
    """Return layer norm weight paths for a standalone LLM."""
    if model_type == 'internvl':
        prefix = f'model.layers.{layer_idx}'
        return {
            'input_ln': f'{prefix}.attention_norm.weight',
            'post_attn_ln': f'{prefix}.ffn_norm.weight',
        }
    else:
        prefix = f'model.layers.{layer_idx}'
        return {
            'input_ln': f'{prefix}.input_layernorm.weight',
            'post_attn_ln': f'{prefix}.post_attention_layernorm.weight',
        }


def get_final_norm_paths(model_type):
    """Return final model norm paths for VLM and standalone LLM.

    Returns (vlm_path, llm_path) for the final RMSNorm/LayerNorm.
    """
    if model_type in ('llava-ov', 'qwen2vl', 'llava-hf', 'llava-liuhaotian', 'llava-llama3'):
        return ('model.language_model.model.norm.weight', 'model.norm.weight')
    elif model_type == 'idefics2':
        return ('model.text_model.norm.weight', 'norm.weight')
    elif model_type == 'internvl':
        return ('language_model.model.tok_embeddings.norm.weight', 'model.tok_embeddings.norm.weight')
    else:
        return ('model.norm.weight', 'model.norm.weight')


# ═══════════════════════════════════════════════════════════════════
# Section 3 — PMBT label loading
# ═══════════════════════════════════════════════════════════════════

def load_pmbt_labels(label_dir, taxonomy='pmbt'):
    """Load PMBT neuron labels from the pipeline's classification output.

    Returns:
        labels_by_layer: dict {layer_idx (int): list of label dicts}
        n_layers: int
        n_neurons: int (inferred from first layer)
    """
    label_filename = ('neuron_labels_permutation.json' if taxonomy == 'pmbt'
                      else 'neuron_labels.json')

    # Try merged _all file first (single JSON with all layers)
    merged_path = os.path.join(
        label_dir,
        label_filename.replace('.json', '_all.json'))

    labels_by_layer = {}

    if os.path.isfile(merged_path):
        print(f'  Loading merged PMBT labels from {merged_path}')
        with open(merged_path) as f:
            raw = json.load(f)
        # Keys are layer index strings
        labels_by_layer = {int(k): v for k, v in raw.items()}
    else:
        # Per-layer files — scan the directory for layer subdirs
        print(f'  Scanning {label_dir} for per-layer label files...')
        for entry in sorted(os.listdir(label_dir)):
            full = os.path.join(label_dir, entry)
            if not os.path.isdir(full):
                continue
            fpath = os.path.join(full, label_filename)
            if not os.path.isfile(fpath):
                continue
            # Extract layer index from directory name
            import re
            m = re.search(r'layers?[._](\d+)', entry)
            if not m:
                m = re.search(r'(\d+)', entry)
            if m:
                layer_idx = int(m.group(1))
                with open(fpath) as f:
                    labels_by_layer[layer_idx] = json.load(f)

    if not labels_by_layer:
        raise FileNotFoundError(
            f'No PMBT labels found in {label_dir}. '
            f'Run steps 1-4 of the pipeline first.')

    n_layers = max(labels_by_layer.keys()) + 1

    # Infer n_neurons from first available layer
    sample_layer = labels_by_layer[min(labels_by_layer.keys())]
    n_neurons = len(sample_layer)

    print(f'  Loaded PMBT labels: {n_layers} layers × {n_neurons} neurons')

    # Print category distribution
    all_labels = [e['label'] for layer_data in labels_by_layer.values()
                  for e in layer_data]
    total = len(all_labels)
    for cat in ['visual', 'text', 'multimodal', 'unknown']:
        count = sum(1 for l in all_labels if l == cat)
        print(f'    {cat:12s}: {count:6,} ({100 * count / total:.1f}%)')

    return labels_by_layer, n_layers, n_neurons


def build_neuron_masks(labels_by_layer, n_layers, n_neurons):
    """Build per-layer boolean masks for text and visual neurons.

    Returns:
        text_masks:   dict {layer_idx: torch.BoolTensor of shape (n_neurons,)}
        visual_masks: dict {layer_idx: torch.BoolTensor of shape (n_neurons,)}
    """
    text_masks = {}
    visual_masks = {}

    for layer_idx in range(n_layers):
        if layer_idx not in labels_by_layer:
            # Missing layer — use empty masks (no neurons modified)
            text_masks[layer_idx] = torch.zeros(n_neurons, dtype=torch.bool)
            visual_masks[layer_idx] = torch.zeros(n_neurons, dtype=torch.bool)
            continue

        layer_labels = sorted(labels_by_layer[layer_idx],
                              key=lambda x: x['neuron_idx'])

        text_mask = torch.zeros(n_neurons, dtype=torch.bool)
        visual_mask = torch.zeros(n_neurons, dtype=torch.bool)

        for entry in layer_labels:
            idx = entry['neuron_idx']
            if idx >= n_neurons:
                continue
            label = entry.get('label', 'unknown')
            if label == 'text':
                text_mask[idx] = True
            elif label == 'visual':
                visual_mask[idx] = True

        text_masks[layer_idx] = text_mask
        visual_masks[layer_idx] = visual_mask

    # Summary
    total_text = sum(m.sum().item() for m in text_masks.values())
    total_visual = sum(m.sum().item() for m in visual_masks.values())
    total = n_layers * n_neurons
    print(f'  Masks built:')
    print(f'    Text neurons:   {total_text:,} / {total:,} '
          f'({100 * total_text / total:.1f}%)')
    print(f'    Visual neurons: {total_visual:,} / {total:,} '
          f'({100 * total_visual / total:.1f}%)')

    return text_masks, visual_masks


# ═══════════════════════════════════════════════════════════════════
# Section 3b — BRV-style text backbone extraction & direct merge
#
# Matches BRV merge.py exactly: extract just the text/language model
# from the VLM and math LLM so their state dict keys align.
# No base LLM needed — direct α·W_vlm + (1-α)·W_math.
# ═══════════════════════════════════════════════════════════════════

def extract_text_backbone_state_dicts(vlm_path, math_llm_path, model_type):
    """Extract text backbone state dicts matching BRV's merge.py exactly."""
    import time
    from transformers import AutoModelForCausalLM

    t0 = time.time()

    if model_type in ('llava-llama3', 'llava-hf'):
        from transformers import LlavaNextForConditionalGeneration
        vlm = LlavaNextForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True)
        vlm_text_sd = vlm.language_model.state_dict()
        del vlm
        math_model = AutoModelForCausalLM.from_pretrained(
            math_llm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
        math_sd = math_model.state_dict()
        del math_model
        excluded_keys = {'model.embed_tokens.weight', 'lm_head.weight'}

    elif model_type == 'idefics2':
        # BRV: AutoModelForVision2Seq(...).model.text_model → MistralModel
        from transformers import AutoModelForVision2Seq
        vlm = AutoModelForVision2Seq.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        vlm_text_sd = vlm.model.text_model.state_dict()
        del vlm
        # BRV: AutoModelForCausalLM(...).model → MistralModel
        math_model = AutoModelForCausalLM.from_pretrained(
            math_llm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
        math_sd = math_model.model.state_dict()
        del math_model
        excluded_keys = {'embed_tokens.weight'}

    elif model_type in ('qwen2vl',):
        # BRV: Qwen2VLForConditionalGeneration(...).model
        # Handle both Qwen2-VL and Qwen2.5-VL
        try:
            from transformers import Qwen2VLForConditionalGeneration
            vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                vlm_path, torch_dtype='auto', low_cpu_mem_usage=True)
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration
            vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_path, torch_dtype='auto', low_cpu_mem_usage=True)
        vlm_text_sd = vlm.model.state_dict()
        del vlm
        math_model = AutoModelForCausalLM.from_pretrained(
            math_llm_path, torch_dtype='auto', low_cpu_mem_usage=True)
        math_sd = math_model.model.state_dict()
        del math_model
        excluded_keys = {'embed_tokens.weight'}

    elif model_type == 'llava-ov':
        from transformers import LlavaOnevisionForConditionalGeneration
        vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        vlm_text_sd = vlm.language_model.state_dict()
        del vlm
        math_model = AutoModelForCausalLM.from_pretrained(
            math_llm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        math_sd = math_model.state_dict()
        del math_model
        excluded_keys = {'model.embed_tokens.weight', 'lm_head.weight'}

    elif model_type == 'llava-liuhaotian':
        from transformers import LlavaForConditionalGeneration
        vlm = LlavaForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        vlm_text_sd = vlm.language_model.state_dict()
        del vlm
        math_model = AutoModelForCausalLM.from_pretrained(
            math_llm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
        math_sd = math_model.state_dict()
        del math_model
        excluded_keys = {'model.embed_tokens.weight', 'lm_head.weight'}

    elif model_type == 'internvl':
        from transformers import AutoModel
        vlm = AutoModel.from_pretrained(
            vlm_path, torch_dtype=torch.float16,
            trust_remote_code=True, low_cpu_mem_usage=True)
        vlm_text_sd = vlm.language_model.state_dict()
        del vlm
        math_model = AutoModelForCausalLM.from_pretrained(
            math_llm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, trust_remote_code=True)
        math_sd = math_model.state_dict()
        del math_model
        excluded_keys = {'model.embed_tokens.weight',
                         'model.tok_embeddings.weight', 'output.weight'}

    else:
        raise ValueError(f'Unsupported model_type for BRV extraction: {model_type}')

    torch.cuda.empty_cache()

    # ── Normalize key prefixes ────────────────────────────────────
    # Different extractors produce different key prefixes:
    #   .language_model.state_dict() → "layers.0..." (LlamaModel)
    #   AutoModelForCausalLM.state_dict() → "model.layers.0..." (LlamaForCausalLM)
    #   .model.state_dict() → "language_model.layers.0..." (Qwen2.5-VL)
    # Detect mismatch and remap math_sd keys to match vlm_text_sd.
    vlm_sample = [k for k in vlm_text_sd if 'layers.0.' in k]
    math_sample = [k for k in math_sd if 'layers.0.' in k]

    if vlm_sample and math_sample:
        vlm_prefix = vlm_sample[0].split('layers.0.')[0]   # e.g. "" or "model." or "language_model."
        math_prefix = math_sample[0].split('layers.0.')[0]  # e.g. "model." or "" or "language_model."

        if vlm_prefix != math_prefix:
            print(f'  Key prefix mismatch: VLM="{vlm_prefix}" vs Math="{math_prefix}"')
            print(f'  Remapping math keys: "{math_prefix}" → "{vlm_prefix}"')
            remapped = {}
            remapped_excluded = set()
            for k, v in math_sd.items():
                if k.startswith(math_prefix):
                    new_key = vlm_prefix + k[len(math_prefix):]
                else:
                    new_key = k
                remapped[new_key] = v
                # Also remap excluded keys
                if k in excluded_keys:
                    remapped_excluded.add(new_key)
            math_sd = remapped
            # Update excluded keys with remapped versions
            excluded_keys = excluded_keys | remapped_excluded

    vlm_keys = set(vlm_text_sd.keys()) - excluded_keys
    math_keys = set(math_sd.keys()) - excluded_keys
    shared = vlm_keys & math_keys
    vlm_only = vlm_keys - math_keys
    math_only = math_keys - vlm_keys

    elapsed = time.time() - t0
    print(f'  BRV extraction complete in {elapsed:.1f}s:')
    print(f'    VLM text backbone: {len(vlm_text_sd)} keys')
    print(f'    Math LLM:          {len(math_sd)} keys')
    print(f'    Shared (mergeable): {len(shared)} keys')
    if vlm_only:
        print(f'    VLM-only (skipped): {len(vlm_only)} keys')
        for k in sorted(vlm_only)[:3]:
            print(f'      {k}  shape={list(vlm_text_sd[k].shape)}')
    if math_only:
        print(f'    Math-only (skipped): {len(math_only)} keys')
        for k in sorted(math_only)[:3]:
            print(f'      {k}  shape={list(math_sd[k].shape)}')

    return vlm_text_sd, math_sd, excluded_keys


def apply_brv_direct_merge(vlm_text_sd, math_sd, excluded_keys, alpha,
                           masks=None, n_layers=None, n_neurons=None,
                           model_type=None):
    """Apply BRV direct weighted average: θ' = α·θ_vlm + (1-α)·θ_math.

    masks=None → uniform (all keys merged, exact BRV).
    masks provided → PMBT selective (MLP masked, attention+norms uniform).
    """
    from tqdm import tqdm

    n_merged = 0
    n_masked_neurons = 0
    is_masked = masks is not None

    masked_mlp_keys = set()
    if is_masked and n_layers and model_type:
        # Detect actual key prefix from vlm_text_sd
        _vlm_prefix = ''
        for k in vlm_text_sd:
            if 'layers.0.' in k and 'mlp' in k:
                _vlm_prefix = k.split('layers.0.')[0]
                break

        for layer_idx in range(n_layers):
            if model_type == 'internvl':
                suffixes = ['feed_forward.w1.weight',
                            'feed_forward.w3.weight',
                            'feed_forward.w2.weight']
            else:
                suffixes = ['mlp.gate_proj.weight',
                            'mlp.up_proj.weight',
                            'mlp.down_proj.weight']
            for s in suffixes:
                p = f'{_vlm_prefix}layers.{layer_idx}.{s}'
                if p in vlm_text_sd:
                    masked_mlp_keys.add(p)

    for key in tqdm(list(math_sd.keys()), desc='  Merging'):
        if key in excluded_keys:
            continue
        if key not in vlm_text_sd:
            continue
        if vlm_text_sd[key].shape != math_sd[key].shape:
            print(f'  [skip] Shape mismatch: {key} '
                  f'vlm={list(vlm_text_sd[key].shape)} '
                  f'math={list(math_sd[key].shape)}')
            continue

        if is_masked and key in masked_mlp_keys:
            layer_match = re.search(r'(\d+)', key)
            if not layer_match:
                continue
            layer_idx = int(layer_match.group())
            if layer_idx not in masks:
                continue
            mask = masks[layer_idx]
            w_vlm = vlm_text_sd[key].float()
            w_math = math_sd[key].float()
            is_down = ('down_proj' in key or 'w2.weight' in key)
            if is_down:
                mask_dev = mask[:w_vlm.shape[1]]
                w_vlm[:, mask_dev] = alpha * w_vlm[:, mask_dev] + (1 - alpha) * w_math[:, mask_dev]
            else:
                mask_dev = mask[:w_vlm.shape[0]]
                w_vlm[mask_dev] = alpha * w_vlm[mask_dev] + (1 - alpha) * w_math[mask_dev]
            vlm_text_sd[key] = w_vlm.half()
            n_masked_neurons += mask_dev.sum().item()
        else:
            vlm_text_sd[key].copy_(
                (alpha * vlm_text_sd[key].float() +
                 (1 - alpha) * math_sd[key].float()).half()
            )
        n_merged += 1

    return n_merged, n_masked_neurons


def _inject_text_backbone_into_vlm(vlm_state, text_backbone_sd, model_type):
    """Copy merged text backbone weights back into the full VLM state dict."""
    sample_text_key = next(iter(text_backbone_sd))

    if model_type in ('llava-ov', 'llava-llama3', 'llava-hf', 'llava-liuhaotian'):
        if sample_text_key in vlm_state:
            prefix = ''
        elif f'language_model.{sample_text_key}' in vlm_state:
            prefix = 'language_model.'
        else:
            prefix = ''
            for vlm_key in vlm_state:
                if vlm_key.endswith(sample_text_key):
                    prefix = vlm_key[:-len(sample_text_key)]
                    break
    elif model_type == 'idefics2':
        prefix = 'model.text_model.'
    elif model_type == 'qwen2vl':
        prefix = 'model.'
    elif model_type == 'internvl':
        prefix = 'language_model.'
    else:
        prefix = ''

    n_injected = 0
    for text_key, value in text_backbone_sd.items():
        vlm_key = f'{prefix}{text_key}' if prefix else text_key
        vlm_key = vlm_key.replace('model.model.', 'model.')
        if vlm_key in vlm_state:
            vlm_state[vlm_key] = value
            n_injected += 1

    if n_injected == 0:
        print(f'  WARNING: 0 keys injected! Prefix="{prefix}", '
              f'sample key="{sample_text_key}"')
        for k in list(vlm_state.keys())[:5]:
            print(f'    VLM key sample: {k}')
    else:
        print(f'  Injected {n_injected} merged keys back into VLM state dict')


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Task vector computation
# ═══════════════════════════════════════════════════════════════════

def compute_task_vectors(vlm_state, base_state, finetuned_state,
                         model_type, n_layers, source='llm'):
    """Compute per-layer task vectors: delta = W_finetuned - W_base.

    For the LLM-side vectors (Method 1), both base and finetuned are
    standalone LLMs whose weight paths use the simpler naming.

    For the VLM-side vectors (Method 5), both models are VLMs and we
    compare their VLM backbone weights directly.

    Returns:
        task_vectors: dict {layer_idx: {'gate': tensor, 'up': tensor, 'down': tensor,
                                        'attn': {name: tensor, ...}}}
        The 'attn' sub-dict contains task vectors for attention projections
        (q, k, v, o or qkv, o for InternVL).
    """
    task_vectors = {}

    print(f'  Computing task vectors (MLP + attention) across {n_layers} layers...')

    for layer_idx in tqdm(range(n_layers), desc='  Task vectors'):
        if source == 'llm':
            # Both base and finetuned are standalone LLMs
            g_path, u_path, d_path = get_llm_mlp_weight_paths(layer_idx, model_type)
            attn_paths = get_llm_attn_weight_paths(layer_idx, model_type)
        else:
            # Both are VLMs — use VLM-style paths
            g_path, u_path, d_path = get_mlp_weight_paths(model_type, layer_idx)
            attn_paths = get_attn_weight_paths(model_type, layer_idx)

        if g_path not in base_state:
            # For source='vlm', base_state is a standalone LLM (Qwen2-7B etc.)
            # whose paths are simpler: model.layers.{i}.mlp.* (no language_model prefix).
            # Try LLM-style paths for the base first.
            g_path_llm, u_path_llm, d_path_llm = get_llm_mlp_weight_paths(layer_idx, model_type)
            if g_path_llm in base_state:
                # Base found via LLM paths; donor mismatch handled below by suffix search
                g_path, u_path, d_path = g_path_llm, u_path_llm, d_path_llm
                attn_paths = get_llm_attn_weight_paths(layer_idx, model_type)
            else:
                # True fallback: try VLM paths for both sides
                g_path, u_path, d_path = get_mlp_weight_paths(model_type, layer_idx)
                if g_path not in finetuned_state:
                    print(f'    WARNING: layer {layer_idx} not found in state dicts, skipping')
                    continue

        tv = {}
        for key, path_base, path_ft in [
            ('gate', g_path, g_path),
            ('up',   u_path, u_path),
            ('down', d_path, d_path),
        ]:
            # Align: finetuned state may have different path prefix than base
            # Try exact path first, then search for matching suffix
            if path_ft in finetuned_state and path_base in base_state:
                w_ft   = finetuned_state[path_ft].float()
                w_base = base_state[path_base].float()
            else:
                # Search for matching key by suffix (handles Qwen2 LLM vs VLM mismatch)
                suffix = '.' + path_base.split('.')[-2] + '.' + path_base.split('.')[-1]
                ft_matches   = [k for k in finetuned_state if k.endswith(suffix)
                                and f'.{layer_idx}.' in k]
                base_matches = [k for k in base_state   if k.endswith(suffix)
                                and f'.{layer_idx}.' in k]
                if not ft_matches or not base_matches:
                    print(f'    WARNING: {path_ft} not found, skipping layer {layer_idx}')
                    continue
                w_ft   = finetuned_state[ft_matches[0]].float()
                w_base = base_state[base_matches[0]].float()

            tv[key] = w_ft - w_base  # delta: what finetuning changed

        # Compute attention task vectors
        attn_tv = {}
        for attn_name, attn_path in attn_paths.items():
            if attn_path in finetuned_state and attn_path in base_state:
                attn_tv[attn_name] = finetuned_state[attn_path].float() - base_state[attn_path].float()
            else:
                # Suffix search for attention weights — use full self_attn.X.weight
                # to avoid matching vision encoder attention (different dimensions)
                parts = attn_path.split('.')
                # Build suffix like 'self_attn.q_proj.weight' (3 parts)
                suffix = '.'.join(parts[-3:])
                ft_matches   = [k for k in finetuned_state if k.endswith(suffix)
                                and f'.{layer_idx}.' in k
                                and 'vision' not in k and 'visual' not in k
                                and 'encoder' not in k]
                base_matches = [k for k in base_state   if k.endswith(suffix)
                                and f'.{layer_idx}.' in k
                                and 'vision' not in k and 'visual' not in k
                                and 'encoder' not in k]
                if ft_matches and base_matches:
                    w_ft = finetuned_state[ft_matches[0]].float()
                    w_base = base_state[base_matches[0]].float()
                    if w_ft.shape == w_base.shape:
                        attn_tv[attn_name] = w_ft - w_base
                    else:
                        print(f'    WARNING: attn {attn_name} shape mismatch '
                              f'{w_ft.shape} vs {w_base.shape}, skipping')

        if len(tv) == 3:
            tv['attn'] = attn_tv if attn_tv else {}
            task_vectors[layer_idx] = tv

    n_attn = sum(1 for k, v in task_vectors.items()
                 if isinstance(k, int) and isinstance(v, dict) and v.get('attn'))
    print(f'  Task vectors computed for {len(task_vectors)} layers '
          f'({n_attn} with attention).')
    return task_vectors


def compute_remaining_task_vectors(vlm_state, base_state, finetuned_state,
                                    model_type, n_layers):
    """Compute task vectors for all LLM keys NOT covered by MLP or attention.

    This captures layer norms, final norms, and any other parameters that
    BRV merges but our MLP+attention path doesn't. BRV's merge.py iterates
    over ALL keys except embeddings — we need to match that.

    Returns:
        dict {vlm_key: task_vector_tensor} for remaining keys
        (both reason and vlm task vectors if needed for BRV formula)
    """
    # Collect all VLM language-model keys we already handle
    handled_suffixes = set()
    for layer_idx in range(n_layers):
        g, u, d = get_mlp_weight_paths(model_type, layer_idx)
        handled_suffixes.update([g, u, d])
        for _, path in get_attn_weight_paths(model_type, layer_idx).items():
            handled_suffixes.add(path)

    # Excluded keys (embeddings — same as BRV)
    exclude_patterns = ['embed_tokens', 'lm_head']

    # Find VLM keys that are LLM parameters but NOT MLP/attention/embeddings
    remaining = {}
    n_found = 0

    for vlm_key in vlm_state:
        # Skip if already handled by MLP or attention
        if vlm_key in handled_suffixes:
            continue
        # Skip vision encoder keys
        if any(x in vlm_key for x in ['vision', 'visual', 'encoder', 'image',
                                        'multi_modal_projector', 'connector']):
            continue
        # Skip embeddings
        if any(x in vlm_key for x in exclude_patterns):
            continue
        # Must be a language model key (layernorm, norm, etc.)
        # Check if we can find matching keys in finetuned and base state dicts
        # Try suffix matching (strip VLM prefix)
        suffix_parts = vlm_key.split('.')
        # Build candidate suffixes: try progressively shorter suffixes
        ft_key = None
        base_key = None

        # For layer norms: vlm key like model.language_model.layers.0.input_layernorm.weight
        # base/ft key like model.layers.0.input_layernorm.weight
        # Strategy: try progressively shorter suffixes, AND also prepend 'model.' 
        for start in range(len(suffix_parts)):
            candidate = '.'.join(suffix_parts[start:])
            if candidate in finetuned_state and candidate in base_state:
                ft_key = candidate
                base_key = candidate
                break
            # Also try with 'model.' prefix (LLM state dicts often have model.layers.*)
            model_candidate = 'model.' + candidate
            if model_candidate in finetuned_state and model_candidate in base_state:
                ft_key = model_candidate
                base_key = model_candidate
                break

        if ft_key is None or base_key is None:
            continue

        w_ft = finetuned_state[ft_key].float()
        w_base = base_state[base_key].float()
        w_vlm = vlm_state[vlm_key].float()

        if w_ft.shape != w_base.shape or w_ft.shape != w_vlm.shape:
            continue

        remaining[vlm_key] = {
            'reason_tv': w_ft - w_base,     # τ_reason
            'vlm_tv': w_vlm - w_base,       # τ_vlm (for BRV formula)
        }
        n_found += 1

    print(f'  Remaining (non-MLP/attn) task vectors: {n_found} keys '
          f'(layer norms, final norm, etc.)')
    return remaining


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Masked weight update (the core merge operation)
# ═══════════════════════════════════════════════════════════════════

def apply_masked_merge(vlm_state, task_vectors, masks,
                       model_type, n_layers, lam,
                       formula='additive', vlm_task_vectors=None,
                       remaining_tvs=None):
    """Apply task vector injection to a subset of neurons defined by masks.

    For gate_proj and up_proj: weight shape is (d_ffn, d_model)
      → rows correspond to neurons → index on dim 0

    For down_proj: weight shape is (d_model, d_ffn)
      → columns correspond to neurons → index on dim 1

    Attention layers (q_proj, k_proj, v_proj, o_proj) are merged
    UNCONDITIONALLY for all variants (matching BRV's full-model merge).
    Only MLP layers use the PMBT mask for selective merging.

    Two formulas are supported (λ = BRV convention, VLM retention weight):

      "brv" (default, BRV paper, Chen et al. ICML 2025):
        θ'[mask] = λ · θ_vlm[mask] + (1-λ) · θ_reason[mask]
                 = θ_vlm[mask] + (1-λ) · τ_reason[mask] - (1-λ) · τ_vlm[mask]
        λ=0.9 means 90% VLM, 10% math (BRV best for LLaVA).
        Requires vlm_task_vectors (τ_vlm = θ_vlm - θ_base).

      "additive" (ours):
        θ'[mask] = θ_vlm[mask] + (1-λ) · τ_reason[mask]
        Keeps full VLM weights, purely adds reasoning on top.

    Args:
        vlm_state: state dict of the target VLM (will be modified in-place)
        task_vectors: dict {layer_idx: {'gate', 'up', 'down', 'attn': {...}}} — τ_reason
        masks: dict {layer_idx: BoolTensor (n_neurons,)}
        model_type: determines weight path naming
        n_layers: number of layers
        lam: injection strength (lambda)
        formula: 'additive' or 'brv'
        vlm_task_vectors: dict {layer_idx: {'gate', 'up', 'down', 'attn': {...}}} — τ_vlm
                          Required only when formula='brv'.

    Returns:
        n_neurons_updated: total number of neurons modified
    """
    if formula == 'brv' and vlm_task_vectors is None:
        raise ValueError('BRV formula requires vlm_task_vectors (τ_vlm)')

    n_updated = 0
    n_attn_merged = 0

    for layer_idx in range(n_layers):
        if layer_idx not in task_vectors:
            continue
        if layer_idx not in masks:
            continue

        mask = masks[layer_idx]
        if not mask.any():
            continue

        tv = task_vectors[layer_idx]
        g_path, u_path, d_path = get_mlp_weight_paths(model_type, layer_idx)

        # Find actual keys in state dict (handles slight naming variations)
        def find_key(path):
            if path in vlm_state:
                return path
            # Try searching by suffix
            for k in vlm_state:
                if k.endswith(path.split('model.')[-1]):
                    return k
            return None

        g_key = find_key(g_path)
        u_key = find_key(u_path)
        d_key = find_key(d_path)

        if g_key is None:
            continue

        tv_gate = tv['gate'].to(vlm_state[g_key].device)
        tv_up   = tv['up'].to(vlm_state[u_key].device)
        tv_down = tv['down'].to(vlm_state[d_key].device)

        mask_dev = mask.to(vlm_state[g_key].device)

        # For BRV formula, also get the VLM task vectors to subtract
        if formula == 'brv' and layer_idx in vlm_task_vectors:
            vlm_tv = vlm_task_vectors[layer_idx]
            vlm_tv_gate = vlm_tv['gate'].to(vlm_state[g_key].device)
            vlm_tv_up   = vlm_tv['up'].to(vlm_state[u_key].device)
            vlm_tv_down = vlm_tv['down'].to(vlm_state[d_key].device)
        else:
            vlm_tv_gate = vlm_tv_up = vlm_tv_down = None

        # Injection strength = (1 - λ) since λ is VLM retention weight
        alpha = 1.0 - lam

        # ── MLP: masked merge (selective via PMBT) ────────────────
        # gate_proj + up_proj: shape (d_ffn, d_model) — rows = neurons
        vlm_state[g_key] = vlm_state[g_key].float()
        vlm_state[g_key][mask_dev] += alpha * tv_gate[mask_dev]
        if vlm_tv_gate is not None:
            vlm_state[g_key][mask_dev] -= alpha * vlm_tv_gate[mask_dev]
        vlm_state[g_key] = vlm_state[g_key].half()  # restore dtype

        vlm_state[u_key] = vlm_state[u_key].float()
        vlm_state[u_key][mask_dev] += alpha * tv_up[mask_dev]
        if vlm_tv_up is not None:
            vlm_state[u_key][mask_dev] -= alpha * vlm_tv_up[mask_dev]
        vlm_state[u_key] = vlm_state[u_key].half()

        # down_proj: shape (d_model, d_ffn) — cols = neurons
        vlm_state[d_key] = vlm_state[d_key].float()
        vlm_state[d_key][:, mask_dev] += alpha * tv_down[:, mask_dev]
        if vlm_tv_down is not None:
            vlm_state[d_key][:, mask_dev] -= alpha * vlm_tv_down[:, mask_dev]
        vlm_state[d_key] = vlm_state[d_key].half()

        n_updated += mask.sum().item()

        # ── Attention: unconditional merge (all variants) ─────────
        # Matches BRV which merges all attention weights uniformly.
        # Formula: θ' = λ·θ_vlm + (1-λ)·θ_reason = θ_vlm + α·τ_reason - α·τ_vlm
        attn_tv = tv.get('attn', {})
        vlm_attn_tv = (vlm_task_vectors[layer_idx].get('attn', {})
                        if formula == 'brv' and layer_idx in vlm_task_vectors
                        else {})

        attn_paths = get_attn_weight_paths(model_type, layer_idx)
        for attn_name, attn_path in attn_paths.items():
            if attn_name not in attn_tv:
                continue
            attn_key = find_key(attn_path)
            if attn_key is None:
                continue

            attn_delta = attn_tv[attn_name].to(vlm_state[attn_key].device)
            vlm_state[attn_key] = vlm_state[attn_key].float()
            vlm_state[attn_key] += alpha * attn_delta
            if formula == 'brv' and attn_name in vlm_attn_tv:
                vlm_attn_delta = vlm_attn_tv[attn_name].to(vlm_state[attn_key].device)
                vlm_state[attn_key] -= alpha * vlm_attn_delta
            vlm_state[attn_key] = vlm_state[attn_key].half()
            n_attn_merged += 1

    if n_attn_merged > 0:
        print(f'    (+ {n_attn_merged} attention projections merged unconditionally)')

    # ── Remaining keys: layer norms, final norm, etc. ─────────
    # BRV merges ALL parameters except embeddings. This handles
    # everything not covered by MLP or attention above.
    n_remaining = 0
    if remaining_tvs:
        alpha = 1.0 - lam
        for vlm_key, tvs in remaining_tvs.items():
            if vlm_key not in vlm_state:
                continue
            reason_tv = tvs['reason_tv'].to(vlm_state[vlm_key].device)
            vlm_state[vlm_key] = vlm_state[vlm_key].float()
            vlm_state[vlm_key] += alpha * reason_tv
            if formula == 'brv':
                vlm_tv = tvs['vlm_tv'].to(vlm_state[vlm_key].device)
                vlm_state[vlm_key] -= alpha * vlm_tv
            vlm_state[vlm_key] = vlm_state[vlm_key].half()
            n_remaining += 1
        if n_remaining > 0:
            print(f'    (+ {n_remaining} remaining keys merged: layer norms, etc.)')

    return n_updated


# ═══════════════════════════════════════════════════════════════════
# Section 6 — POPE evaluation (reuses logic from your existing pipeline)
# ═══════════════════════════════════════════════════════════════════

def load_model_for_eval(model_type, model_path_or_state,
                        merged_state=None, device='cuda:0'):
    """Load a VLM for evaluation, optionally injecting merged weights.

    If merged_state is provided, it is loaded into the model after
    the base model is loaded (in-place weight replacement).
    """
    if model_type == 'llava-ov':
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                   AutoProcessor)
        processor = AutoProcessor.from_pretrained(model_path_or_state)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path_or_state,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).eval()
        if merged_state is not None:
            model.load_state_dict(merged_state, strict=False)
        model = model.to(device)

    elif model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path_or_state)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path_or_state,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).eval()
        if merged_state is not None:
            model.load_state_dict(merged_state, strict=False)
        model = model.to(device)

    elif model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_path_or_state,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True).eval()
        if merged_state is not None:
            model.load_state_dict(merged_state, strict=False)
        model = model.to(device)
        processor = AutoTokenizer.from_pretrained(
            model_path_or_state, trust_remote_code=True)

    elif model_type in ('llava-hf', 'llava-liuhaotian'):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path_or_state)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path_or_state,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval()
        if merged_state is not None:
            model.load_state_dict(merged_state, strict=False)
        model = model.to(device)

    elif model_type == 'idefics2':
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_path_or_state)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path_or_state,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval()
        if merged_state is not None:
            model.load_state_dict(merged_state, strict=False)
        model = model.to(device)

    elif model_type == 'llava-llama3':
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path_or_state)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path_or_state,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval()
        if merged_state is not None:
            model.load_state_dict(merged_state, strict=False)
        model = model.to(device)

    else:
        raise ValueError(f'Unsupported model_type: {model_type}')

    return model, processor


def eval_pope(model, processor, pope_path, pope_img_dir,
              n_questions, model_type, device):
    """Evaluate POPE hallucination rate. Returns dict with metrics."""
    from PIL import Image

    questions = []
    with open(pope_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if n_questions and n_questions < len(questions):
        questions = questions[:n_questions]

    n_correct = n_halluc = n_total = 0
    n_tp = n_fp = n_fn = 0
    model.eval()

    for q in tqdm(questions, desc='    POPE eval', leave=False):
        img_path = os.path.join(pope_img_dir, q['image'])
        qtext    = q.get('text', q.get('question', ''))
        gt       = q.get('label', q.get('answer', '')).strip().lower()

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        try:
            with torch.no_grad():
                if model_type == 'llava-ov':
                    msgs = [{'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': qtext}]}]
                    prompt = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                    inputs = processor(images=img, text=prompt,
                                       return_tensors='pt').to(device)
                    out = model.generate(**inputs, max_new_tokens=10,
                                         do_sample=False)
                    plen = inputs['input_ids'].shape[1]
                    generated = processor.decode(out[0][plen:],
                                                  skip_special_tokens=True)

                elif model_type == 'qwen2vl':
                    msgs = [{'role': 'user', 'content': [
                        {'type': 'image', 'image': img},
                        {'type': 'text', 'text': qtext}]}]
                    prompt = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                    inputs = processor(images=img, text=prompt,
                                       return_tensors='pt').to(device)
                    out = model.generate(**inputs, max_new_tokens=10,
                                         do_sample=False)
                    plen = inputs['input_ids'].shape[1]
                    generated = processor.decode(out[0][plen:],
                                                  skip_special_tokens=True)

                elif model_type == 'internvl':
                    import torchvision.transforms as T
                    from torchvision.transforms.functional import InterpolationMode
                    tf = T.Compose([
                        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                    pv = tf(img).unsqueeze(0).to(torch.bfloat16).to(device)
                    generated = model.chat(
                        processor, pv, f'<image>\n{qtext}',
                        dict(max_new_tokens=10, do_sample=False))

                elif model_type in ('llava-hf', 'llava-liuhaotian'):
                    prompt = f'USER: <image>\n{qtext}\nASSISTANT:'
                    inputs = processor(text=prompt, images=img,
                                       return_tensors='pt').to(device)
                    out = model.generate(**inputs, max_new_tokens=10,
                                         do_sample=False)
                    generated = processor.decode(out[0], skip_special_tokens=True)

                elif model_type == 'idefics2':
                    msgs = [{'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': qtext}]}]
                    prompt = processor.apply_chat_template(
                        msgs, add_generation_prompt=True)
                    inputs = processor(images=img, text=prompt,
                                       return_tensors='pt').to(device)
                    out = model.generate(**inputs, max_new_tokens=10,
                                         do_sample=False)
                    plen = inputs['input_ids'].shape[1]
                    generated = processor.decode(out[0][plen:],
                                                  skip_special_tokens=True)

                elif model_type == 'llava-llama3':
                    msgs = [{'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': qtext}]}]
                    prompt = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                    inputs = processor(images=img, text=prompt,
                                       return_tensors='pt').to(device)
                    out = model.generate(**inputs, max_new_tokens=10,
                                         do_sample=False)
                    plen = inputs['input_ids'].shape[1]
                    generated = processor.decode(out[0][plen:],
                                                  skip_special_tokens=True)

                else:
                    continue

        except Exception as e:
            print(f'    Warning: generation failed: {e}')
            continue

        answer = generated.strip().lower()
        pred_yes = 'yes' in answer

        if (pred_yes and gt == 'yes') or (not pred_yes and gt == 'no'):
            n_correct += 1
        if pred_yes and gt == 'no':
            n_halluc += 1
        # Track TP/FP/FN for precision/recall/F1
        if pred_yes and gt == 'yes':
            n_tp += 1
        elif pred_yes and gt == 'no':
            n_fp += 1
        elif not pred_yes and gt == 'yes':
            n_fn += 1
        n_total += 1

    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_tp + n_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'hallucination_rate': n_halluc / max(n_total, 1),
        'accuracy': n_correct / max(n_total, 1),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'n_hallucinated': n_halluc,
        'n_total': n_total,
    }


# ═══════════════════════════════════════════════════════════════════
# Section 6b — Save merged model as HF checkpoint (for VLMEvalKit)
# ═══════════════════════════════════════════════════════════════════

def save_as_hf_checkpoint(merged_state, model_type, vlm_path, save_dir):
    """Save merged weights as a full HuggingFace checkpoint.

    VLMEvalKit needs a loadable HF model directory (with config.json,
    tokenizer, etc).  We load the base VLM architecture, inject the
    merged state dict, then save_pretrained.

    Following the BRV repo pattern:
    https://github.com/shiqichen17/VLM_Merging/blob/main/merge.py
    """
    if os.path.isfile(os.path.join(save_dir, 'config.json')):
        print(f'    HF checkpoint already exists: {save_dir}')
        return save_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f'    Saving HF checkpoint → {save_dir} ...')
    t0 = time.time()

    if model_type == 'llava-ov':
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                  AutoProcessor)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(vlm_path)
    elif model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(vlm_path)
    elif model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            vlm_path, torch_dtype=torch.float16,
            trust_remote_code=True, low_cpu_mem_usage=True)
        processor = AutoTokenizer.from_pretrained(
            vlm_path, trust_remote_code=True)
    elif model_type in ('llava-hf', 'llava-liuhaotian'):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(vlm_path)
    elif model_type == 'idefics2':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(vlm_path)
    elif model_type == 'llava-llama3':
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        model = LlavaNextForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(vlm_path)
    else:
        raise ValueError(f'Unsupported model_type: {model_type}')

    model.load_state_dict(merged_state, strict=False)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    del model
    torch.cuda.empty_cache()
    print(f'    HF checkpoint saved in {time.time()-t0:.0f}s')
    return save_dir


# ═══════════════════════════════════════════════════════════════════
# Section 6c — VLMEvalKit evaluation (BRV paper benchmarks)
#
# Following the BRV repo: https://github.com/shiqichen17/VLM_Merging
# which uses VLMEvalKit (pip install vlmeval) for:
#   MathVista_MINI, MathVerse_MINI (5 sub-splits run individually),
#   MathVision_MINI, DynaMath, MMStar, MM-Math
# ═══════════════════════════════════════════════════════════════════

def eval_vlmevalkit(hf_checkpoint_dir, benchmarks, model_type,
                    n_gpus=1, vlmevalkit_dir=None, work_dir=None,
                    model_name=None):
    """Run VLMEvalKit benchmarks on a saved HF checkpoint.

    Auto-registers the model in VLMEvalKit's config.py if needed,
    then uses --data + --model CLI flags per benchmark.
    """
    import subprocess
    import csv
    import glob

    if work_dir is None:
        work_dir = os.path.join(hf_checkpoint_dir, 'vlmevalkit_results')

    if model_name is None:
        model_name = os.path.basename(os.path.normpath(hf_checkpoint_dir))
    abs_model_path = os.path.abspath(hf_checkpoint_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Find VLMEvalKit dir and run.py
    run_py = None
    _vlmeval_root = vlmevalkit_dir
    if vlmevalkit_dir and os.path.isfile(os.path.join(vlmevalkit_dir, 'run.py')):
        run_py = os.path.join(vlmevalkit_dir, 'run.py')
    else:
        for candidate in ['VLMEvalKit', '../VLMEvalKit',
                          'modern_vlms/VLMEvalKit']:
            if os.path.isfile(os.path.join(candidate, 'run.py')):
                run_py = os.path.join(candidate, 'run.py')
                _vlmeval_root = candidate
                break

    # Auto-register model in VLMEvalKit config.py
    vlmevalkit_class = {
        'qwen2vl': 'Qwen2VLChat',
        'llava-ov': 'LLaVA_OneVision_HF',
        'internvl': 'InternVLChat',
        'llava-hf': 'LLaVA',
        'llava-liuhaotian': 'LLaVA',
        'llava-llama3': 'LLaVA_Next',
        'idefics2': 'idefics2_8b',
    }.get(model_type, model_type)

    if _vlmeval_root:
        config_py = os.path.join(_vlmeval_root, 'vlmeval', 'config.py')
        if os.path.isfile(config_py):
            import fcntl
            lock_path = config_py + '.lock'
            try:
                lock_fd = open(lock_path, 'w')
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                with open(config_py) as f:
                    cfg_text = f.read()
                if f'"{model_name}"' not in cfg_text:
                    anchors = {
                        'LLaVA_Next': '"llava_next_mistral_7b"',
                        'Qwen2VLChat': '"Qwen2-VL-7B-Instruct"',
                        'InternVLChat': '"InternVL2_5-8B"',
                        'LLaVA_OneVision_HF': '"llava-onevision-qwen2-7b-ov-hf"',
                        'LLaVA': '"llava-v1.5-7b"',
                    }
                    anchor = anchors.get(vlmevalkit_class, '')
                    entry = f'    "{model_name}": partial({vlmevalkit_class}, model_path="{abs_model_path}"),\n'
                    if anchor and anchor in cfg_text:
                        cfg_text = cfg_text.replace(
                            anchor, entry + '    ' + anchor)
                        with open(config_py, 'w') as f:
                            f.write(cfg_text)
                        print(f'    Registered "{model_name}" in VLMEvalKit config.py')
                    else:
                        print(f'    WARNING: anchor {anchor} not found in config.py')
                else:
                    print(f'    Model "{model_name}" already in VLMEvalKit config.py')
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

    # Execute one benchmark at a time via CLI flags
    all_results = {}

    for bench in benchmarks:
        print(f'    VLMEvalKit: {bench} on {model_name} ({n_gpus} GPU)')

        if run_py:
            cmd = [sys.executable, run_py,
                   '--data', bench,
                   '--model', model_name,
                   '--work-dir', work_dir,
                   '--reuse',
                   '--verbose']
        else:
            cmd = [sys.executable, '-m', 'vlmeval',
                   '--data', bench,
                   '--model', model_name,
                   '--work-dir', work_dir,
                   '--reuse',
                   '--verbose']

        print(f'    cmd: {" ".join(cmd)}')

        try:
            result = subprocess.run(cmd, timeout=7200)
            if result.returncode != 0:
                print(f'    WARNING: {bench} failed (rc={result.returncode})')
        except subprocess.TimeoutExpired:
            print(f'    WARNING: {bench} timed out after 2h')
        except FileNotFoundError:
            print(f'    ERROR: VLMEvalKit not found')
            return {}

        # ── Parse results: try CSV first (GPT-judged), xlsx fallback ──
        found = False

        # 1. CSV files from full eval (scored by GPT judge — BRV method)
        for search_dir in [os.path.join(work_dir, model_name), work_dir]:
            for pattern in [f'{bench}.csv', f'{bench}_result.csv',
                            f'*{bench}*.csv']:
                csv_matches = glob.glob(os.path.join(search_dir, '**', pattern),
                                        recursive=True)
                if csv_matches:
                    try:
                        with open(csv_matches[0]) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                all_results[bench] = {
                                    k: _try_float(v)
                                    for k, v in row.items() if v}
                        if bench in all_results:
                            print(f'    {bench}: {all_results[bench]}')
                            found = True
                    except Exception as e:
                        print(f'    WARNING: CSV parse error: {e}')
                    break
            if found:
                break

        # 2. Fallback: xlsx predictions (rule-based local scoring)
        if not found:
            xlsx_matches = glob.glob(
                os.path.join(work_dir, '**', f'*{bench}*.xlsx'),
                recursive=True)
            if xlsx_matches:
                try:
                    score = _score_vlmeval_xlsx(xlsx_matches[0], bench)
                    if score:
                        all_results[bench] = score
                        print(f'    {bench} (rule-based): accuracy='
                              f'{score["accuracy"]:.4f} '
                              f'({score["correct"]}/{score["total"]})')
                except Exception as e:
                    print(f'    WARNING: xlsx scoring error: {e}')

    return all_results


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _score_vlmeval_xlsx(xlsx_path, benchmark):
    """Score a VLMEvalKit prediction xlsx file using rule-based matching."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    if 'prediction' not in df.columns or 'answer' not in df.columns:
        return None

    correct = 0
    total = 0
    for _, row in df.iterrows():
        pred = str(row.get('prediction', '')).strip()
        answer = str(row.get('answer', '')).strip()
        qtype = str(row.get('question_type', '')).lower()
        atype = str(row.get('answer_type', '')).lower()
        if not answer:
            continue
        total += 1

        if 'multi_choice' in qtype or benchmark == 'MMStar':
            # MCQ: compare extracted letter to answer_option (the correct letter)
            answer_option = str(row.get('answer_option', '')).strip().upper()
            pred_letter = _extract_mcq_letter(pred)
            if answer_option and pred_letter == answer_option:
                correct += 1
            elif not answer_option:
                # Fallback: compare prediction text to answer text
                if pred.lower() == answer.lower():
                    correct += 1
        elif atype in ('integer', 'float'):
            # Numeric: extract last number from prediction, compare to answer
            pred_num = _extract_number(pred)
            ans_num = _extract_number(answer)
            if pred_num is not None and ans_num is not None:
                if ans_num == 0:
                    if abs(pred_num) < 1e-3:
                        correct += 1
                elif abs(pred_num - ans_num) / max(abs(ans_num), 1e-6) < 0.01:
                    correct += 1
            elif pred.strip() == answer.strip():
                correct += 1
        else:
            # Text free-form: exact match (case-insensitive)
            if pred.lower().strip() == answer.lower().strip():
                correct += 1

    if total == 0:
        return None
    return {'accuracy': round(correct / total, 4),
            'correct': correct, 'total': total}


def _extract_mcq_letter(text):
    """Extract MCQ answer letter from model response."""
    text = text.strip()
    if text.upper() in ('A', 'B', 'C', 'D', 'E'):
        return text.upper()
    m = re.search(r'answer\s+is\s+\(?([A-E])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\(([A-E])\)', text)
    if m:
        return m.group(1).upper()
    if len(text) <= 3 and text and text[0].upper() in 'ABCDE':
        return text[0].upper()
    return text[:1].upper() if text else ''


def _extract_number(text):
    """Extract the last number from text."""
    text = str(text).replace(',', '').replace('$', '').replace('%', '')
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def _match_freeform(pred, answer):
    """Match free-form numerical/text answers."""
    pred = pred.strip().lower()
    answer = answer.strip().lower()
    if pred == answer:
        return True
    pred_num = _extract_number(pred)
    ans_num = _extract_number(answer)
    if pred_num is not None and ans_num is not None:
        if ans_num == 0:
            return abs(pred_num) < 1e-3
        return abs(pred_num - ans_num) / max(abs(ans_num), 1e-6) < 0.01
    return False


# ═══════════════════════════════════════════════════════════════════
# Section 6d — CHAIR evaluation (caption-based hallucination)
#
# CHAIR (Rohrbach et al., EMNLP 2018) measures object hallucination
# in free-form image captions:
#   CHAIRi = fraction of mentioned objects that are hallucinated
#   CHAIRs = fraction of captions containing any hallucinated object
#
# Implementation follows Maxlinn/CHAIR-metric-standalone and
# LALBJ/PAI (ECCV 2024) patterns.
# ═══════════════════════════════════════════════════════════════════

def eval_chair(model, processor, model_type, device,
               coco_img_dir, coco_ann_dir,
               n_images=500, max_tokens=64, seed=42):
    """Evaluate CHAIR metrics by generating captions on COCO val2014.

    Returns:
        dict with CHAIRi, CHAIRs, Recall, and counts
    """
    from PIL import Image
    import random
    import re

    caption_prompt = ('Please describe this image in detail. '
                      'List all the objects you can see.')

    # ── Load COCO instance annotations ───────────────────────────
    instances_file = os.path.join(coco_ann_dir, 'instances_val2014.json')
    if not os.path.isfile(instances_file):
        print(f'    CHAIR: {instances_file} not found, skipping')
        return None

    with open(instances_file) as f:
        inst = json.load(f)

    cat_map = {c['id']: c['name'].lower() for c in inst['categories']}
    imid_to_objects = {}
    for ann in inst['annotations']:
        imid = ann['image_id']
        name = cat_map.get(ann['category_id'], '')
        if name:
            imid_to_objects.setdefault(imid, set()).add(name)

    # COCO synonyms (from CHAIR-metric-standalone)
    synonyms = {
        'motorbike': 'motorcycle', 'aeroplane': 'airplane',
        'sofa': 'couch', 'tv': 'television',
        'cell phone': 'cellphone', 'mobile phone': 'cellphone',
        'laptop computer': 'laptop', 'hot dog': 'hotdog',
        'teddy bear': 'teddybear', 'fire hydrant': 'firehydrant',
        'stop sign': 'stopsign', 'parking meter': 'parkingmeter',
        'wine glass': 'wineglass', 'baseball bat': 'baseballbat',
        'baseball glove': 'baseballglove', 'tennis racket': 'tennisracket',
        'sports ball': 'sportsball', 'potted plant': 'pottedplant',
        'hair drier': 'hairdryer', 'motor bike': 'motorcycle',
        'motor cycle': 'motorcycle', 'air plane': 'airplane',
        'suit case': 'suitcase', 'traffic light': 'trafficlight',
        'street light': 'trafficlight', 'bow tie': 'bowtie',
        'stove top oven': 'oven',
    }
    all_coco_objects = set(cat_map.values())
    inv_syn = {obj: obj for obj in all_coco_objects}
    inv_syn.update(synonyms)

    # ── Select images (≥3 objects, like POPE) ────────────────────
    eligible = [i for i, o in imid_to_objects.items() if len(o) >= 3]
    random.seed(seed)
    selected = random.sample(eligible, min(n_images, len(eligible)))

    # ── Generate captions ────────────────────────────────────────
    print(f'    CHAIR: captioning {len(selected)} images...')
    model.eval()
    captions_out = []

    for imid in tqdm(selected, desc='    CHAIR captions', leave=False):
        img_path = os.path.join(coco_img_dir, f'COCO_val2014_{imid:012d}.jpg')
        if not os.path.isfile(img_path):
            continue
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        try:
            with torch.no_grad():
                gen = _generate_text(model, processor, img, caption_prompt,
                                     model_type, device, max_tokens)
        except Exception as e:
            continue

        captions_out.append({'image_id': imid, 'caption': gen.strip()})

    if not captions_out:
        return None

    # ── Score ─────────────────────────────────────────────────────
    total_mentioned = total_halluc = total_caps = total_halluc_caps = 0
    total_gt = total_recalled = 0

    for entry in captions_out:
        imid = entry['image_id']
        caption = entry['caption'].lower()
        gt = imid_to_objects.get(imid, set())
        words = caption.split()

        mentioned = set()
        hallucinated = set()

        # single-word matches
        for w in words:
            w_clean = re.sub(r'[^\w]', '', w)
            if w_clean in inv_syn:
                canon = inv_syn[w_clean]
                mentioned.add(canon)
                if w_clean not in gt and canon not in gt:
                    hallucinated.add(canon)

        # two-word phrase matches
        for i in range(len(words) - 1):
            phrase = (re.sub(r'[^\w]', '', words[i]) + ' ' +
                      re.sub(r'[^\w]', '', words[i+1]))
            if phrase in inv_syn:
                canon = inv_syn[phrase]
                mentioned.add(canon)
                if phrase not in gt and canon not in gt:
                    hallucinated.add(canon)

        total_mentioned += len(mentioned)
        total_halluc += len(hallucinated)
        total_caps += 1
        if hallucinated:
            total_halluc_caps += 1
        total_gt += len(gt)
        total_recalled += len(mentioned & gt)

    chair_i = total_halluc / max(total_mentioned, 1)
    chair_s = total_halluc_caps / max(total_caps, 1)
    recall = total_recalled / max(total_gt, 1)

    return {
        'CHAIRi': round(chair_i, 4),
        'CHAIRs': round(chair_s, 4),
        'Recall': round(recall, 4),
        'n_images': total_caps,
        'n_objects_mentioned': total_mentioned,
        'n_objects_hallucinated': total_halluc,
    }


def _generate_text(model, processor, img, prompt, model_type, device,
                   max_tokens):
    """Generate text for a single image (used by CHAIR)."""
    if model_type == 'llava-ov':
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=text,
                           return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False)
        plen = inputs['input_ids'].shape[1]
        return processor.decode(out[0][plen:], skip_special_tokens=True)

    elif model_type == 'qwen2vl':
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=text,
                           return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False)
        plen = inputs['input_ids'].shape[1]
        return processor.decode(out[0][plen:], skip_special_tokens=True)

    elif model_type == 'internvl':
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        tf = T.Compose([
            T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        pv = tf(img).unsqueeze(0).to(torch.bfloat16).to(device)
        return model.chat(processor, pv, f'<image>\n{prompt}',
                          dict(max_new_tokens=max_tokens, do_sample=False))

    elif model_type in ('llava-hf', 'llava-liuhaotian'):
        text = f'USER: <image>\n{prompt}\nASSISTANT:'
        inputs = processor(text=text, images=img,
                           return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False)
        return processor.decode(out[0], skip_special_tokens=True)

    elif model_type == 'idefics2':
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(
            msgs, add_generation_prompt=True)
        inputs = processor(images=img, text=text,
                           return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False)
        plen = inputs['input_ids'].shape[1]
        return processor.decode(out[0][plen:], skip_special_tokens=True)

    elif model_type == 'llava-llama3':
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=text,
                           return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False)
        plen = inputs['input_ids'].shape[1]
        return processor.decode(out[0][plen:], skip_special_tokens=True)

    raise ValueError(f'Unsupported model_type: {model_type}')

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"="*65}')
    print(f'  PMBT-Guided Neuron-Selective Weight Merging')
    print(f'  Method:  {args.method}')
    _formula_desc = ("(BRV paper: trade-off at target neurons)" if args.merge_formula == "brv"
                     else "(ours: additive at target neurons)")
    print(f'  Formula: {args.merge_formula}  {_formula_desc}')
    print(f'  VLM:     {args.vlm_path}')
    print(f'{"="*65}\n')

    # Validate required args for non-compose methods
    if args.method != 'compose':
        if args.method == 'visual_transplant' and args.base_llm_path is None:
            raise ValueError('--base_llm_path is required for visual_transplant')
        if args.label_dir is None:
            raise ValueError('--label_dir is required for text_inject and visual_transplant')

    # ── Compose branch: merge two saved step-16 / step-17 state dicts ──
    if args.method == 'compose':
        if args.step16_model_path is None or args.step17_model_path is None:
            raise ValueError(
                '--step16_model_path and --step17_model_path are required '
                'for the compose method')

        print('Step A: Loading original VLM weights...')
        from transformers import AutoModelForVision2Seq, AutoModel
        t0 = time.time()
        try:
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        except Exception:
            vlm_model = AutoModel.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16,
                trust_remote_code=True, low_cpu_mem_usage=True)
        vlm_state = vlm_model.state_dict()
        del vlm_model
        torch.cuda.empty_cache()
        print(f'  VLM loaded in {time.time()-t0:.1f}s ({len(vlm_state)} tensors)')

        print('Step B: Loading step-16 state dict (text_inject)...')
        s16_path = os.path.join(args.step16_model_path, 'merged_state_dict.pt')
        s16_state = torch.load(s16_path, map_location='cpu')
        print(f'  Step-16 loaded ({len(s16_state)} tensors)')

        print('Step C: Loading step-17 state dict (visual_transplant)...')
        s17_path = os.path.join(args.step17_model_path, 'merged_state_dict.pt')
        s17_state = torch.load(s17_path, map_location='cpu')
        print(f'  Step-17 loaded ({len(s17_state)} tensors)')

        print('Step D: Composing...')
        # Because step-16 only touches text neurons and step-17 only touches
        # visual neurons (disjoint masks), the composed state is simply:
        #   W_composed = W_vlm + (W_16 - W_vlm) + (W_17 - W_vlm)
        #              = W_16 + W_17 - W_vlm
        # No interference is possible: each key is modified by at most one edit.
        composed_state = {}
        n_text_edited = 0
        n_vis_edited = 0
        keys = list(vlm_state.keys())
        for key in keys:
            w_vlm = vlm_state[key].float()
            w_16  = s16_state[key].float() if key in s16_state else w_vlm
            w_17  = s17_state[key].float() if key in s17_state else w_vlm
            delta_16 = w_16 - w_vlm
            delta_17 = w_17 - w_vlm
            # Sanity check: deltas should be non-overlapping (both nonzero at
            # the same position would indicate a mask collision)
            both_nonzero = (delta_16.abs() > 0) & (delta_17.abs() > 0)
            if both_nonzero.any():
                n_collision = both_nonzero.sum().item()
                print(f'  WARNING: {n_collision} positions have non-zero deltas '
                      f'from BOTH step-16 and step-17 in {key} — masks may overlap')
            n_text_edited  += (delta_16.abs() > 0).sum().item()
            n_vis_edited   += (delta_17.abs() > 0).sum().item()
            composed_state[key] = (w_vlm + delta_16 + delta_17).half()
            # Free originals for this key immediately to keep peak RAM low
            del vlm_state[key], w_vlm, w_16, w_17, delta_16, delta_17
            if key in s16_state: del s16_state[key]
            if key in s17_state: del s17_state[key]
        print(f'  Text-inject positions modified:    {n_text_edited:,}')
        print(f'  Visual-transplant positions modified: {n_vis_edited:,}')

        del s16_state, s17_state, vlm_state
        import gc; gc.collect()

        compose_out = os.path.join(args.output_dir, 'model')
        os.makedirs(compose_out, exist_ok=True)

        # Reload base model, apply composed weights, save as HF checkpoint
        print('  Reloading base model for save_pretrained...')
        try:
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        except Exception:
            vlm_model = AutoModel.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16,
                trust_remote_code=True, low_cpu_mem_usage=True)
        vlm_model.load_state_dict(composed_state, strict=False)
        vlm_model.save_pretrained(compose_out)
        del vlm_model; gc.collect()

        # Also copy tokenizer / processor files so from_pretrained works fully
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.vlm_path)
        processor.save_pretrained(compose_out)
        del processor

        print(f'  Composed model saved → {compose_out}')

        # Save metadata
        meta = {
            'method': 'compose',
            'step16_model_path': args.step16_model_path,
            'step17_model_path': args.step17_model_path,
            'n_text_edited': n_text_edited,
            'n_vis_edited': n_vis_edited,
        }
        with open(os.path.join(args.output_dir, 'compose_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print('Done.')
        return 0

    # ── Step A: Load PMBT labels ──────────────────────────────────
    print('Step A: Loading PMBT neuron labels...')
    labels_by_layer, n_layers_labels, n_neurons_labels = \
        load_pmbt_labels(args.label_dir, args.taxonomy)

    n_layers  = args.n_layers  or n_layers_labels
    n_neurons = args.n_neurons or n_neurons_labels

    text_masks, visual_masks = build_neuron_masks(
        labels_by_layer, n_layers, n_neurons)

    # ── Step B+C: Model loading and merge setup ─────────────────────
    if args.method == 'text_inject':
        if args.math_llm_path is None:
            raise ValueError('--math_llm_path required for text_inject method')

        # BRV-style: extract text backbone, direct weighted average
        print('\nStep B: Extracting text backbone (BRV-style)...')
        print(f'  VLM:      {args.vlm_path}')
        print(f'  Math LLM: {args.math_llm_path}')

        vlm_text_sd, math_sd, brv_excluded_keys = \
            extract_text_backbone_state_dicts(
                args.vlm_path, args.math_llm_path, args.model_type)

        # Also load the full VLM state dict for saving the merged model
        print(f'  Loading full VLM for model saving...')
        t0 = time.time()
        if args.model_type == 'llava-ov':
            from transformers import LlavaOnevisionForConditionalGeneration
            vlm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'qwen2vl':
            from transformers import AutoModelForVision2Seq
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'internvl':
            from transformers import AutoModel
            vlm_model = AutoModel.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16,
                trust_remote_code=True, low_cpu_mem_usage=True)
        elif args.model_type in ('llava-hf', 'llava-liuhaotian'):
            from transformers import LlavaForConditionalGeneration
            vlm_model = LlavaForConditionalGeneration.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'idefics2':
            from transformers import AutoModelForVision2Seq
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'llava-llama3':
            from transformers import LlavaNextForConditionalGeneration
            vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            raise ValueError(f'Unsupported model_type: {args.model_type}')
        vlm_state = vlm_model.state_dict()
        del vlm_model
        torch.cuda.empty_cache()
        print(f'  Full VLM loaded in {time.time()-t0:.1f}s ({len(vlm_state)} tensors)')

        inject_masks = text_masks
        method_label = 'text_inject'
        _use_brv_direct = True

    else:  # visual_transplant
        if args.donor_vlm_path is None:
            raise ValueError('--donor_vlm_path required for visual_transplant method')
        if args.base_llm_path is None:
            raise ValueError('--base_llm_path required for visual_transplant method')

        # Load VLM + base LLM for task vector computation
        print('\nStep B: Loading model weights (CPU)...')
        print(f'  Loading VLM: {args.vlm_path}')
        t0 = time.time()
        if args.model_type == 'llava-ov':
            from transformers import LlavaOnevisionForConditionalGeneration
            vlm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'qwen2vl':
            from transformers import AutoModelForVision2Seq
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'internvl':
            from transformers import AutoModel
            vlm_model = AutoModel.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16,
                trust_remote_code=True, low_cpu_mem_usage=True)
        elif args.model_type in ('llava-hf', 'llava-liuhaotian'):
            from transformers import LlavaForConditionalGeneration
            vlm_model = LlavaForConditionalGeneration.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'idefics2':
            from transformers import AutoModelForVision2Seq
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif args.model_type == 'llava-llama3':
            from transformers import LlavaNextForConditionalGeneration
            vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
                args.vlm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            raise ValueError(f'Unsupported model_type: {args.model_type}')
        vlm_state = vlm_model.state_dict()
        del vlm_model
        torch.cuda.empty_cache()
        print(f'  VLM loaded in {time.time()-t0:.1f}s ({len(vlm_state)} tensors)')

        print(f'  Loading base LLM: {args.base_llm_path}')
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_llm_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=True)
        base_state = base_model.state_dict()
        del base_model
        torch.cuda.empty_cache()
        print(f'  Base LLM loaded ({len(base_state)} tensors)')

        print(f'\nStep C: Loading donor VLM and computing task vectors...')
        print(f'  Donor VLM: {args.donor_vlm_path}')

        # Use AutoModel generically — we only need the state dict weights,
        # not the full generation interface, so architecture doesn't matter here.
        # This handles LLaVA-OV, Qwen2.5-VL, InternVL etc. without class mismatches.
        from transformers import AutoModel
        try:
            donor_model = AutoModel.from_pretrained(
                args.donor_vlm_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True)
        except Exception:
            # Fallback: some VLMs require specific loaders
            from transformers import AutoModelForVision2Seq
            donor_model = AutoModelForVision2Seq.from_pretrained(
                args.donor_vlm_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True)

        donor_state = donor_model.state_dict()
        del donor_model
        torch.cuda.empty_cache()
        print(f'  Donor VLM loaded ({len(donor_state)} tensors)')

        # Task vector = W_donor_vlm - W_base (using VLM path naming)
        task_vectors = compute_task_vectors(
            vlm_state, base_state, donor_state,
            model_type=args.model_type,
            n_layers=n_layers,
            source='vlm')  # VLM-style path naming
        del donor_state, base_state
        import gc; gc.collect()  # aggressively free ~28GB before eval

        # For Method 5, use donor's visual mask if donor labels provided,
        # otherwise use target VLM's visual mask
        if args.donor_label_dir:
            print(f'  Loading donor VLM PMBT labels for visual mask...')
            donor_labels, _, _ = load_pmbt_labels(
                args.donor_label_dir, args.taxonomy)
            _, donor_visual_masks = build_neuron_masks(
                donor_labels, n_layers, n_neurons)
            # Intersection: only inject into neurons that are visual in BOTH
            inject_masks = {
                layer: target_m & donor_visual_masks.get(layer, torch.zeros(n_neurons, dtype=torch.bool))
                for layer, target_m in visual_masks.items()
            }
            n_intersect = sum(m.sum().item() for m in inject_masks.values())
            print(f'  Intersection visual mask: {n_intersect:,} neurons '
                  f'(visual in both target and donor)')
        else:
            inject_masks = visual_masks  # Method 5: inject into visual neurons

        method_label = 'visual_transplant'
        _use_brv_direct = False

    # Ensure variables defined for both paths
    if 'vlm_task_vectors' not in locals():
        vlm_task_vectors = None
    if 'remaining_tvs' not in locals():
        remaining_tvs = None
    if '_use_brv_direct' not in locals():
        _use_brv_direct = False

    # ── Step D: Lambda sweep ──────────────────────────────────────
    print(f'\nStep D: Lambda sweep: {args.lambda_sweep}')
    print(f'  Merge formula: {args.merge_formula}  {_formula_desc}')

    all_results = {}

    # Also run uniform (no mask) baseline if requested
    lambda_configs = [(lam, inject_masks, method_label)
                      for lam in args.lambda_sweep]
    if args.include_uniform_baseline:
        # Uniform mask = all neurons (no PMBT masking)
        all_mask = {layer: torch.ones(n_neurons, dtype=torch.bool)
                    for layer in range(n_layers)}
        for lam in args.lambda_sweep:
            lambda_configs.append((lam, all_mask, f'{method_label}_uniform'))

    if args.include_multimodal:
        # Text + multimodal neurons (broader text mask)
        text_multi_mask = {}
        for layer_idx in range(n_layers):
            mask = text_masks[layer_idx].clone()
            if layer_idx in labels_by_layer:
                for entry in labels_by_layer[layer_idx]:
                    if entry.get('label') == 'multimodal' and entry['neuron_idx'] < n_neurons:
                        mask[entry['neuron_idx']] = True
            text_multi_mask[layer_idx] = mask
        n_tm = sum(m.sum().item() for m in text_multi_mask.values())
        print(f'  Text+multi mask: {n_tm:,} neurons')
        for lam in args.lambda_sweep:
            lambda_configs.append((lam, text_multi_mask, f'{method_label}_text_multi'))

    if args.include_visual_only:
        # Visual neurons only (negative control — should hurt POPE)
        n_vo = sum(m.sum().item() for m in visual_masks.values())
        print(f'  Visual-only mask: {n_vo:,} neurons')
        for lam in args.lambda_sweep:
            lambda_configs.append((lam, visual_masks, f'{method_label}_visual_only'))

    if args.include_visual_multi:
        # Visual + multimodal neurons
        vis_multi_mask = {}
        for layer_idx in range(n_layers):
            mask = visual_masks[layer_idx].clone()
            if layer_idx in labels_by_layer:
                for entry in labels_by_layer[layer_idx]:
                    if entry.get('label') == 'multimodal' and entry['neuron_idx'] < n_neurons:
                        mask[entry['neuron_idx']] = True
            vis_multi_mask[layer_idx] = mask
        n_vm = sum(m.sum().item() for m in vis_multi_mask.values())
        print(f'  Visual+multi mask: {n_vm:,} neurons')
        for lam in args.lambda_sweep:
            lambda_configs.append((lam, vis_multi_mask, f'{method_label}_visual_multi'))

    if args.include_multimodal_only:
        # Multimodal neurons only
        multi_only_mask = {}
        for layer_idx in range(n_layers):
            mask = torch.zeros(n_neurons, dtype=torch.bool)
            if layer_idx in labels_by_layer:
                for entry in labels_by_layer[layer_idx]:
                    if entry.get('label') == 'multimodal' and entry['neuron_idx'] < n_neurons:
                        mask[entry['neuron_idx']] = True
            multi_only_mask[layer_idx] = mask
        n_mo = sum(m.sum().item() for m in multi_only_mask.values())
        print(f'  Multimodal-only mask: {n_mo:,} neurons')
        for lam in args.lambda_sweep:
            lambda_configs.append((lam, multi_only_mask, f'{method_label}_multimodal_only'))

    if args.include_random:
        # Random mask with same neuron count as text mask (sparsity control)
        import random as _rng
        _rng.seed(args.random_seed)
        n_text_total = sum(m.sum().item() for m in text_masks.values())
        random_mask = {layer_idx: torch.zeros(n_neurons, dtype=torch.bool)
                       for layer_idx in range(n_layers)}
        # Distribute n_text_total neurons randomly across layers
        all_positions = [(l, n) for l in range(n_layers) for n in range(n_neurons)]
        selected = _rng.sample(all_positions, min(n_text_total, len(all_positions)))
        for l, n in selected:
            random_mask[l][n] = True
        n_rm = sum(m.sum().item() for m in random_mask.values())
        print(f'  Random mask: {n_rm:,} neurons (matched to text count)')
        for lam in args.lambda_sweep:
            lambda_configs.append((lam, random_mask, f'{method_label}_random'))

    for lam, masks, run_label in lambda_configs:
        run_name = f'{run_label}_lambda{lam}'

        # Skip if merged model already exists on disk
        if args.save_model:
            model_save_dir = os.path.join(args.output_dir, run_name, 'model')
            save_path = os.path.join(model_save_dir, 'merged_state_dict.pt')
            if os.path.isfile(save_path):
                print(f'\n  ─── {run_name} — SKIP (already exists) ───')
                all_results[run_name] = {
                    'lambda': lam, 'method': run_label,
                    'skipped': True, 'save_path': save_path,
                }
                continue

        print(f'\n  ─── {run_name} ───')

        if _use_brv_direct:
            # ── BRV direct merge path (matches BRV merge.py exactly) ──
            vlm_text_run = {k: v.clone() for k, v in vlm_text_sd.items()}

            # Uniform = no mask; PMBT = selective MLP merge
            is_uniform = 'uniform' in run_label
            run_masks = None if is_uniform else masks

            n_merged, n_masked = apply_brv_direct_merge(
                vlm_text_run, math_sd, brv_excluded_keys, alpha=lam,
                masks=run_masks, n_layers=n_layers, n_neurons=n_neurons,
                model_type=args.model_type)

            print(f'  Merged {n_merged:,} keys at α={lam}'
                  + (f' ({n_masked:,} MLP neurons selectively)' if n_masked else ''))

            # Inject merged text backbone back into full VLM state dict
            merged_state = {k: v.clone() for k, v in vlm_state.items()}
            _inject_text_backbone_into_vlm(
                merged_state, vlm_text_run, args.model_type)
            n_updated = n_masked if n_masked else n_merged
            del vlm_text_run

        else:
            # ── Legacy task-vector path (visual_transplant method) ──
            merged_state = {k: v.clone() for k, v in vlm_state.items()}
            n_updated = apply_masked_merge(
                merged_state, task_vectors, masks,
                model_type=args.model_type,
                n_layers=n_layers,
                lam=lam,
                formula=args.merge_formula,
                vlm_task_vectors=vlm_task_vectors,
                remaining_tvs=remaining_tvs)

        print(f'  Updated {n_updated:,} neurons at λ={lam}')

        # Save merged model if requested
        if args.save_model:
            model_save_dir = os.path.join(args.output_dir, run_name, 'model')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(merged_state,
                       os.path.join(model_save_dir, 'merged_state_dict.pt'))
            print(f'  Model saved → {model_save_dir}')

        # Evaluate
        run_results = {
            'lambda': lam,
            'method': run_label,
            'n_neurons_updated': n_updated,
        }

        # ── A) POPE evaluation (inline, fast) ────────────────────
        if args.eval_pope and os.path.isfile(args.pope_path):
            print(f'  Evaluating on POPE ({args.n_pope_questions} questions)...')

            # Load VLM with merged weights for evaluation
            model, processor = load_model_for_eval(
                args.model_type, args.vlm_path,
                merged_state=merged_state,
                device=args.device)

            pope_results = eval_pope(
                model, processor,
                args.pope_path, args.pope_img_dir,
                args.n_pope_questions, args.model_type, args.device)

            run_results['pope'] = pope_results
            print(f'  POPE hallucination rate: '
                  f'{pope_results["hallucination_rate"]:.4f} '
                  f'(accuracy: {pope_results["accuracy"]:.4f})')

            # ── B) CHAIR evaluation (inline, reuses loaded model) ─
            if args.eval_chair:
                coco_img_dir = args.pope_img_dir  # same val2014 dir
                print(f'  Evaluating CHAIR ({args.chair_n_images} images)...')
                chair_results = eval_chair(
                    model, processor, args.model_type, args.device,
                    coco_img_dir, args.coco_ann_dir,
                    args.chair_n_images, args.chair_max_tokens)
                if chair_results:
                    run_results['chair'] = chair_results
                    print(f'  CHAIR-i: {chair_results["CHAIRi"]:.4f}  '
                          f'CHAIR-s: {chair_results["CHAIRs"]:.4f}  '
                          f'Recall: {chair_results["Recall"]:.4f}')
                else:
                    print(f'  [skip] CHAIR: could not run (missing COCO data?)')

            del model
            torch.cuda.empty_cache()
        else:
            if args.eval_pope:
                print(f'  [skip] POPE file not found: {args.pope_path}')

            # CHAIR without POPE — need to load model separately
            if args.eval_chair:
                coco_img_dir = args.pope_img_dir
                print(f'  Evaluating CHAIR ({args.chair_n_images} images)...')
                model, processor = load_model_for_eval(
                    args.model_type, args.vlm_path,
                    merged_state=merged_state,
                    device=args.device)
                chair_results = eval_chair(
                    model, processor, args.model_type, args.device,
                    coco_img_dir, args.coco_ann_dir,
                    args.chair_n_images, args.chair_max_tokens)
                if chair_results:
                    run_results['chair'] = chair_results
                    print(f'  CHAIR-i: {chair_results["CHAIRi"]:.4f}  '
                          f'CHAIR-s: {chair_results["CHAIRs"]:.4f}')
                del model
                torch.cuda.empty_cache()

        # ── C) VLMEvalKit evaluation (BRV benchmarks, subprocess) ─
        if args.eval_vlmevalkit and args.save_model:
            model_save_dir = os.path.join(args.output_dir, run_name, 'model')
            hf_dir = os.path.join(args.output_dir, run_name, 'hf_checkpoint')

            # Save as HF checkpoint (VLMEvalKit needs loadable model dir)
            save_as_hf_checkpoint(merged_state, args.model_type,
                                  args.vlm_path, hf_dir)

            vlmevalkit_work = os.path.join(args.output_dir, run_name,
                                           'vlmevalkit_results')
            print(f'  Evaluating BRV benchmarks via VLMEvalKit '
                  f'({len(args.vlmevalkit_benchmarks)} benchmarks)...')
            vlmeval_results = eval_vlmevalkit(
                hf_dir, args.vlmevalkit_benchmarks, args.model_type,
                args.vlmevalkit_n_gpus, args.vlmevalkit_dir,
                vlmevalkit_work)

            if vlmeval_results:
                run_results['vlmevalkit'] = vlmeval_results
                for bench, metrics in vlmeval_results.items():
                    overall = metrics.get('Overall',
                              metrics.get('Accuracy', ''))
                    if overall:
                        print(f'    {bench}: {overall}')

        elif args.eval_vlmevalkit and not args.save_model:
            print(f'  [skip] VLMEvalKit requires --save_model to create '
                  f'HF checkpoint')

        all_results[run_name] = run_results
        del merged_state

    # ── Step E: Save results ──────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'merge_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved → {results_path}')

    # Print summary table
    print(f'\n{"="*120}')
    print(f'  RESULTS SUMMARY')
    # Build header dynamically based on what was evaluated
    has_pope = any('pope' in r for r in all_results.values())
    has_chair = any('chair' in r for r in all_results.values())
    has_vlmeval = any('vlmevalkit' in r for r in all_results.values())

    header = f'  {"Run":<40}'
    if has_pope:
        header += f' {"POPE Acc":>9} {"POPE Hal":>9}'
    if has_chair:
        header += f' {"CHAIRi":>8} {"CHAIRs":>8} {"Recall":>8}'
    if has_vlmeval:
        # Show key BRV benchmarks in compact form
        header += f' {"MVista":>7} {"MVerse":>7} {"MVision":>8} {"DynaM":>7} {"MMStar":>7} {"MMMath":>7}'
    print(header)
    print(f'  {"─"*115}')

    for run_name, res in all_results.items():
        line = f'  {run_name:<40}'
        if has_pope:
            if 'pope' in res:
                line += (f' {res["pope"]["accuracy"]:>9.4f}'
                         f' {res["pope"]["hallucination_rate"]:>9.4f}')
            else:
                line += f' {"—":>9} {"—":>9}'
        if has_chair:
            if 'chair' in res:
                line += (f' {res["chair"]["CHAIRi"]:>8.4f}'
                         f' {res["chair"]["CHAIRs"]:>8.4f}'
                         f' {res["chair"]["Recall"]:>8.4f}')
            else:
                line += f' {"—":>8} {"—":>8} {"—":>8}'
        if has_vlmeval:
            vr = res.get('vlmevalkit', {})
            for bench in ['MathVista_MINI', 'MathVerse_MINI',
                          'MathVision_MINI', 'DynaMath', 'MMStar', 'MM-Math']:
                bdata = vr.get(bench, {})
                val = bdata.get('Overall', bdata.get('Accuracy', ''))
                if isinstance(val, (int, float)):
                    line += f' {val:>7.1f}'
                else:
                    line += f' {"—":>7}'
        print(line)

    print(f'{"="*120}')

    # If VLMEvalKit was run, also print detailed MathVerse splits
    if has_vlmeval:
        print(f'\n  MathVerse Detailed Splits:')
        print(f'  {"Run":<40} {"Overall":>8} {"T-D":>6} {"T-L":>6}'
              f' {"V-I":>6} {"V-D":>6} {"V-O":>6}')
        print(f'  {"─"*80}')
        mv_splits = ['MathVerse_MINI', 'MathVerse_MINI_Text_Dominant',
                      'MathVerse_MINI_Text_Lite',
                      'MathVerse_MINI_Vision_Intensive',
                      'MathVerse_MINI_Vision_Dominant',
                      'MathVerse_MINI_Vision_Only']
        for run_name, res in all_results.items():
            vr = res.get('vlmevalkit', {})
            if not vr:
                continue
            line = f'  {run_name:<40}'
            for bench in mv_splits:
                bdata = vr.get(bench, {})
                val = bdata.get('Overall', bdata.get('Accuracy', ''))
                if isinstance(val, (int, float)):
                    line += f' {val:>6.1f}'
                else:
                    line += f' {"—":>6}'
            print(line)
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())