#!/usr/bin/env python3
"""
compare_hook_points.py — Compare gate-only vs gate*up activations.

For each neuron, computes rank correlation between:
  - SiLU(gate_proj(x))           (what PMBT currently uses, following Xu et al.)
  - SiLU(gate_proj(x)) * up_proj(x)  (Geva's conceptual equivalent in SwiGLU)

If correlation is very high (>0.95), the choice barely matters for classification.
If correlation is low, PMBT labels may change significantly with the different hook point.

Usage:
    python compare_hook_points.py \
        --model_type llava-llama3 \
        --model_path llava-hf/llama3-llava-next-8b-hf \
        --n_layers 32 \
        --n_images 20 \
        --device 0

Outputs: per-layer statistics and overall summary.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats as scipy_stats

# ── Add LLaVA repo to path (needed for some model backends) ──
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')
if _LLAVA_PATH not in sys.path:
    sys.path.insert(0, _LLAVA_PATH)

from baukit import TraceDict


# ═══════════════════════════════════════════════════════════════
# Layer name helpers
# ═══════════════════════════════════════════════════════════════

def get_act_fn_names(model_type, n_layers, model=None):
    """Layer names for gate-only: SiLU(gate_proj(x))."""
    if model_type == 'llava-hf':
        prefix, suffix = 'model.language_model.layers', 'mlp.act_fn'
    elif model_type == 'internvl':
        prefix, suffix = 'language_model.model.layers', 'feed_forward.act_fn'
    elif model_type == 'qwen2vl':
        if model is not None and not hasattr(model.model, 'language_model'):
            prefix = 'model.layers'
        else:
            prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'llava-ov':
        prefix, suffix = 'model.language_model.layers', 'mlp.act_fn'
    elif model_type == 'llava-llama3':
        prefix, suffix = 'model.language_model.layers', 'mlp.act_fn'
    elif model_type == 'idefics2':
        prefix, suffix = 'model.text_model.layers', 'mlp.act_fn'
    else:  # llava-liuhaotian
        prefix, suffix = 'model.layers', 'mlp.act_fn'
    return [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]


def get_down_proj_names(model_type, n_layers, model=None):
    """Layer names for down_proj — we hook with retain_input=True to get gate*up."""
    if model_type == 'llava-hf':
        prefix, suffix = 'model.language_model.layers', 'mlp.down_proj'
    elif model_type == 'internvl':
        prefix, suffix = 'language_model.model.layers', 'feed_forward.w2'
    elif model_type == 'qwen2vl':
        if model is not None and not hasattr(model.model, 'language_model'):
            prefix = 'model.layers'
        else:
            prefix = 'model.language_model.layers'
        suffix = 'mlp.down_proj'
    elif model_type == 'llava-ov':
        prefix, suffix = 'model.language_model.layers', 'mlp.down_proj'
    elif model_type == 'llava-llama3':
        prefix, suffix = 'model.language_model.layers', 'mlp.down_proj'
    elif model_type == 'idefics2':
        prefix, suffix = 'model.text_model.layers', 'mlp.down_proj'
    else:  # llava-liuhaotian
        prefix, suffix = 'model.layers', 'mlp.down_proj'
    return [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]


# ═══════════════════════════════════════════════════════════════
# Model loading (simplified — supports main model types)
# ═══════════════════════════════════════════════════════════════

def load_model(model_type, model_path, device):
    """Load model and processor. Returns (model, processor, image_token_id)."""
    if model_type in ('llava-hf', 'llava-llama3'):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device).eval()
        image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
        return model, processor, image_token_id
    elif model_type == 'llava-ov':
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device).eval()
        image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
        return model, processor, image_token_id
    elif model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        return model, tokenizer, image_token_id
    elif model_type == 'qwen2vl':
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device).eval()
        image_token_id = 151655  # <|image_pad|>
        return model, processor, image_token_id
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _ensure_act_fn_is_module(model, model_type):
    """Patch plain-function act_fn to nn.Module for baukit hooking."""
    if model_type == 'llava-ov':
        lm = model.language_model
        layers = lm.model.layers if hasattr(lm, 'model') else lm.layers
    elif model_type == 'qwen2vl':
        lm = model.model
        layers = lm.language_model.layers if hasattr(lm, 'language_model') else lm.layers
    else:
        return
    for layer in layers:
        if not isinstance(layer.mlp.act_fn, torch.nn.Module):
            layer.mlp.act_fn = torch.nn.SiLU()


# ═══════════════════════════════════════════════════════════════
# Prepare a simple image+text input
# ═══════════════════════════════════════════════════════════════

def prepare_simple_input(model_type, processor, image, text, device):
    """Prepare model input from image + text. Returns dict of tensors."""
    if model_type in ('llava-hf', 'llava-llama3'):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors='pt')
        return {k: v.to(device) for k, v in inputs.items()}
    elif model_type == 'llava-ov':
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors='pt')
        return {k: v.to(device) for k, v in inputs.items()}
    elif model_type == 'qwen2vl':
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text}
        ]}]
        text_input = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs,
                          return_tensors='pt', padding=True)
        return {k: v.to(device) for k, v in inputs.items()}
    else:
        raise ValueError(f"prepare_simple_input not implemented for {model_type}")


# ═══════════════════════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Compare gate-only vs gate*up activations')
    parser.add_argument('--model_type', required=True,
                        choices=['llava-hf', 'llava-llama3', 'llava-ov', 'internvl',
                                 'qwen2vl', 'llava-liuhaotian', 'idefics2'])
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_layers', type=int, required=True)
    parser.add_argument('--n_images', type=int, default=20,
                        help='Number of COCO images to test (default: 20)')
    parser.add_argument('--coco_dir', default='data/coco/train2017',
                        help='Path to COCO train2017 images')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer indices to test (default: all)')
    parser.add_argument('--output', default=None,
                        help='Save results JSON to this path')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # ── Parse layer selection ──
    if args.layers:
        test_layers = [int(x) for x in args.layers.split(',')]
    else:
        # Default: sample ~8 layers across the model
        if args.n_layers <= 8:
            test_layers = list(range(args.n_layers))
        else:
            step = args.n_layers // 8
            test_layers = list(range(0, args.n_layers, step))[:8]
            if test_layers[-1] != args.n_layers - 1:
                test_layers.append(args.n_layers - 1)

    print(f"Testing layers: {test_layers}")

    # ── Load model ──
    print(f"Loading {args.model_type} from {args.model_path}...")
    t0 = time.time()
    model, processor, image_token_id = load_model(args.model_type, args.model_path, device)
    _ensure_act_fn_is_module(model, args.model_type)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Get layer names ──
    act_fn_names_all = get_act_fn_names(args.model_type, args.n_layers, model)
    down_proj_names_all = get_down_proj_names(args.model_type, args.n_layers, model)

    act_fn_names = [act_fn_names_all[i] for i in test_layers]
    down_proj_names = [down_proj_names_all[i] for i in test_layers]
    all_hook_names = act_fn_names + down_proj_names

    # Verify hooks work
    print(f"  Hooking {len(act_fn_names)} act_fn layers + {len(down_proj_names)} down_proj layers")

    # ── Collect COCO images ──
    from PIL import Image
    coco_images = sorted([f for f in os.listdir(args.coco_dir)
                          if f.endswith('.jpg')])[:args.n_images]
    print(f"  Using {len(coco_images)} images from {args.coco_dir}")

    prompt_text = "Could you describe the image?"

    # ── Storage for per-layer correlations ──
    # For each layer, we collect (gate_vector, gate_up_vector) across all tokens
    # from all images, then compute correlation
    layer_gate_acts = {li: [] for li in test_layers}      # gate only
    layer_gateup_acts = {li: [] for li in test_layers}    # gate * up

    # ── Run forward passes ──
    print(f"\nRunning {len(coco_images)} forward passes...")
    for img_idx, img_file in enumerate(coco_images):
        img_path = os.path.join(args.coco_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        try:
            inputs = prepare_simple_input(args.model_type, processor, img,
                                          prompt_text, device)
        except Exception as e:
            print(f"  Skip {img_file}: {e}")
            continue

        with torch.no_grad():
            # Hook both act_fn (output=gate) and down_proj (input=gate*up)
            with TraceDict(model, all_hook_names,
                          retain_input=True, retain_output=True) as td:
                model(**inputs)

        for idx, li in enumerate(test_layers):
            # Gate-only: output of act_fn = SiLU(gate_proj(x))
            gate_out = td[act_fn_names[idx]].output
            if isinstance(gate_out, tuple):
                gate_out = gate_out[0]
            gate_vals = gate_out[0].float().cpu()  # (seq_len, n_neurons)

            # Gate*up: input to down_proj = SiLU(gate_proj(x)) * up_proj(x)
            dp_input = td[down_proj_names[idx]].input
            if isinstance(dp_input, tuple):
                dp_input = dp_input[0]
            gateup_vals = dp_input[0].float().cpu()  # (seq_len, n_neurons)

            # Sanity check dimensions match
            assert gate_vals.shape == gateup_vals.shape, \
                f"Shape mismatch layer {li}: {gate_vals.shape} vs {gateup_vals.shape}"

            layer_gate_acts[li].append(gate_vals)
            layer_gateup_acts[li].append(gateup_vals)

        if (img_idx + 1) % 5 == 0:
            print(f"  {img_idx+1}/{len(coco_images)} done")

    # ── Compute correlations ──
    print(f"\n{'='*70}")
    print(f"{'Layer':>6} {'Neurons':>8} {'Tokens':>8} "
          f"{'Spearman':>10} {'Pearson':>10} "
          f"{'Sign_agree':>11} {'Gate>0%':>8} {'G*U>0%':>8}")
    print(f"{'-'*70}")

    results = {}
    all_spearman = []

    for li in test_layers:
        # Concatenate across images: (total_tokens, n_neurons)
        gate_cat = torch.cat(layer_gate_acts[li], dim=0).numpy()
        gateup_cat = torch.cat(layer_gateup_acts[li], dim=0).numpy()

        n_tokens, n_neurons = gate_cat.shape

        # Per-neuron Spearman correlation (across token positions)
        # Sample neurons if there are too many
        if n_neurons > 2000:
            neuron_sample = np.random.choice(n_neurons, 2000, replace=False)
        else:
            neuron_sample = np.arange(n_neurons)

        spearmans = []
        pearsons = []
        sign_agrees = []

        for ni in neuron_sample:
            g = gate_cat[:, ni]
            gu = gateup_cat[:, ni]

            # Skip constant neurons
            if g.std() < 1e-8 or gu.std() < 1e-8:
                continue

            rho, _ = scipy_stats.spearmanr(g, gu)
            r, _ = scipy_stats.pearsonr(g, gu)
            # Sign agreement: fraction of tokens where sign(gate) == sign(gate*up)
            sign_ag = np.mean(np.sign(g) == np.sign(gu))

            spearmans.append(rho)
            pearsons.append(r)
            sign_agrees.append(sign_ag)

        spearmans = np.array(spearmans)
        pearsons = np.array(pearsons)
        sign_agrees = np.array(sign_agrees)

        # What fraction of activations are positive?
        gate_pos_frac = (gate_cat > 0).mean()
        gateup_pos_frac = (gateup_cat > 0).mean()

        med_spearman = np.median(spearmans)
        med_pearson = np.median(pearsons)
        med_sign = np.median(sign_agrees)

        all_spearman.extend(spearmans.tolist())

        results[li] = {
            'n_tokens': int(n_tokens),
            'n_neurons_sampled': len(spearmans),
            'spearman_median': float(med_spearman),
            'spearman_mean': float(spearmans.mean()),
            'spearman_q25': float(np.percentile(spearmans, 25)),
            'spearman_q75': float(np.percentile(spearmans, 75)),
            'pearson_median': float(med_pearson),
            'pearson_mean': float(pearsons.mean()),
            'sign_agreement_median': float(med_sign),
            'gate_positive_frac': float(gate_pos_frac),
            'gateup_positive_frac': float(gateup_pos_frac),
        }

        print(f"{li:>6} {n_neurons:>8} {n_tokens:>8} "
              f"{med_spearman:>+10.4f} {med_pearson:>+10.4f} "
              f"{med_sign:>10.4f}  {gate_pos_frac:>7.1%} {gateup_pos_frac:>7.1%}")

    # ── Overall summary ──
    all_spearman = np.array(all_spearman)
    print(f"\n{'='*70}")
    print(f"OVERALL (across {len(test_layers)} layers, {len(all_spearman)} neuron samples):")
    print(f"  Spearman median: {np.median(all_spearman):.4f}")
    print(f"  Spearman mean:   {np.mean(all_spearman):.4f}")
    print(f"  Spearman Q25:    {np.percentile(all_spearman, 25):.4f}")
    print(f"  Spearman Q75:    {np.percentile(all_spearman, 75):.4f}")
    print(f"  Spearman < 0.5:  {(all_spearman < 0.5).mean():.1%} of neurons")
    print(f"  Spearman < 0.8:  {(all_spearman < 0.8).mean():.1%} of neurons")
    print(f"  Spearman > 0.95: {(all_spearman > 0.95).mean():.1%} of neurons")

    interpretation = ""
    if np.median(all_spearman) > 0.95:
        interpretation = "VERY HIGH — gate-only and gate*up are nearly identical. Hook choice barely matters."
    elif np.median(all_spearman) > 0.85:
        interpretation = "HIGH — mostly correlated but some neurons differ. Document in paper, likely minor impact on labels."
    elif np.median(all_spearman) > 0.70:
        interpretation = "MODERATE — meaningful differences exist. Consider re-running classification with gate*up."
    else:
        interpretation = "LOW — gate-only and gate*up give substantially different activation profiles. Gate*up is recommended."

    print(f"\n  Interpretation: {interpretation}")
    print(f"{'='*70}")

    # ── Save results ──
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        summary = {
            'model_type': args.model_type,
            'model_path': args.model_path,
            'n_images': len(coco_images),
            'test_layers': test_layers,
            'overall_spearman_median': float(np.median(all_spearman)),
            'overall_spearman_mean': float(np.mean(all_spearman)),
            'interpretation': interpretation,
            'per_layer': results,
        }
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
