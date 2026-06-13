#!/usr/bin/env python3
"""Create ablated VLM checkpoints by zeroing all neurons of a PMBT category.

For each modality category (visual, text, multimodal), this script:
  1. Loads the VLM
  2. Loads PMBT labels
  3. Zeros all down_proj columns for neurons in that category
  4. Saves the modified model as a full HF checkpoint

The saved checkpoints can be evaluated with VLMEvalKit directly,
matching SNRF's evaluation protocol.

Usage:
    # Create all 3 ablated checkpoints for one model
    python create_ablated_checkpoints.py \
        --model_type llava-llama3 \
        --model_path llava-hf/llama3-llava-next-8b-hf \
        --label_dir results/3-classify/full/llava-next-llama3-8b/llm_permutation \
        --n_layers 32 \
        --output_dir results/7-ablated-checkpoints/llava-next-llama3-8b \
        --categories visual text multimodal
"""

import argparse
import json
import os
import sys
import torch
import shutil
from collections import defaultdict


def load_labels(label_dir, n_layers):
    """Load PMBT neuron labels. Returns {layer_idx: {neuron_idx: label}}."""
    import glob

    # Try merged file first
    for name in ['neuron_labels_permutation_all.json',
                 'neuron_labels_permutation_all*.json']:
        if '*' in name:
            matches = glob.glob(os.path.join(label_dir, name))
            if matches:
                path = matches[0]
            else:
                continue
        else:
            path = os.path.join(label_dir, name)
            if not os.path.isfile(path):
                continue
        with open(path) as f:
            data = json.load(f)
        print(f'  Loaded labels from {path}')
        break
    else:
        raise FileNotFoundError(f'No merged label file found in {label_dir}')

    labels = {}
    for layer_idx in range(n_layers):
        key = str(layer_idx)
        if key not in data:
            continue
        labels[layer_idx] = {}
        for entry in data[key]:
            labels[layer_idx][entry['neuron_idx']] = entry['label']
    return labels


def get_neuron_map(labels, category, n_layers):
    """Get {layer_idx: [neuron_indices]} for all neurons of a category."""
    neuron_map = defaultdict(list)
    total = 0
    for layer_idx in range(n_layers):
        if layer_idx not in labels:
            continue
        for neuron_idx, label in labels[layer_idx].items():
            if label == category:
                neuron_map[layer_idx].append(neuron_idx)
                total += 1
    print(f'  Category "{category}": {total} neurons across '
          f'{len(neuron_map)} layers')
    return dict(neuron_map), total


def get_random_neuron_map(labels, target_count, n_layers, seed=42):
    """Get {layer_idx: [neuron_indices]} for randomly selected neurons.

    Samples target_count neurons uniformly from ALL labeled MLP neurons
    (regardless of category), matching the count of a specific category
    for fair comparison.
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    # Collect all (layer_idx, neuron_idx) pairs
    all_neurons = []
    for layer_idx in range(n_layers):
        if layer_idx not in labels:
            continue
        for neuron_idx in labels[layer_idx].keys():
            all_neurons.append((layer_idx, neuron_idx))

    # Sample
    if target_count >= len(all_neurons):
        selected = all_neurons
    else:
        indices = rng.choice(len(all_neurons), size=target_count, replace=False)
        selected = [all_neurons[i] for i in indices]

    # Build neuron_map
    neuron_map = defaultdict(list)
    for layer_idx, neuron_idx in selected:
        neuron_map[layer_idx].append(neuron_idx)

    print(f'  Random selection: {len(selected)} neurons across '
          f'{len(neuron_map)} layers (seed={seed})')
    return dict(neuron_map), len(selected)


def get_down_proj_key(model_type, layer_idx):
    """Get the dotted attribute path for down_proj weight."""
    prefix_map = {
        'llava-hf': 'model.language_model.layers',
        'llava-liuhaotian': 'model.layers',
        'llava-ov': 'model.language_model.layers',
        'internvl': 'language_model.model.layers',
        'qwen2vl': 'model.language_model.layers',
        'llava-llama3': 'model.language_model.layers',
        'idefics2': 'model.text_model.layers',
    }
    prefix = prefix_map.get(model_type, 'model.language_model.layers')
    if model_type == 'internvl':
        return f'{prefix}.{layer_idx}.feed_forward.w2'
    else:
        return f'{prefix}.{layer_idx}.mlp.down_proj'


def zero_neurons(model, model_type, neuron_map):
    """Zero down_proj columns for all neurons in neuron_map (permanent)."""
    zeroed = 0
    for layer_idx, neuron_indices in neuron_map.items():
        if not neuron_indices:
            continue
        dotted = get_down_proj_key(model_type, layer_idx)
        mod = model
        for p in dotted.split('.'):
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        w = mod.weight  # shape: (hidden_size, intermediate_size)
        idx = torch.tensor(neuron_indices, dtype=torch.long, device=w.device)
        w.data[:, idx] = 0
        zeroed += len(neuron_indices)
    print(f'  Zeroed {zeroed} neurons (permanent)')
    return zeroed


def load_model(model_type, model_path, device='cpu'):
    """Load VLM to CPU for weight modification.

    We load to CPU to avoid GPU memory pressure when saving checkpoints.
    """
    print(f'  Loading {model_type} from {model_path} to {device}...')

    if model_type in ('llava-liuhaotian', 'liuhaotian'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # For liuhaotian, load the full LLaVA model
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, _, _ = load_pretrained_model(
            model_path, None, model_name, device_map=device,
            torch_dtype=torch.float16)
        return model

    elif model_type == 'llava-llama3':
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
        return model

    elif model_type in ('llava-hf', 'hf'):
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
        return model

    elif model_type == 'llava-ov':
        from transformers import LlavaOnevisionForConditionalGeneration
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
        return model

    elif model_type == 'internvl':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True)
        return model

    elif model_type in ('qwen2vl', 'qwen25vl-7b', 'qwen25vl-3b'):
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
        return model

    elif model_type == 'idefics2':
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
        return model

    else:
        raise ValueError(f'Unknown model_type: {model_type}')


def save_checkpoint(model, model_path, output_dir, model_type):
    """Save modified model as HF checkpoint.

    Copies tokenizer/processor files from original, saves modified weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    print(f'  Saving model to {output_dir}...')
    model.save_pretrained(output_dir)

    # Copy tokenizer/processor files from original
    from huggingface_hub import snapshot_download
    try:
        # Resolve original path
        if os.path.isdir(model_path):
            src_dir = model_path
        else:
            src_dir = snapshot_download(model_path, local_files_only=True)
    except Exception:
        src_dir = model_path

    # Copy processor/tokenizer files that save_pretrained doesn't include
    for pattern in ['tokenizer*', 'processor*', 'chat_template*',
                    'preprocessor_config*', 'special_tokens*',
                    'added_tokens*', 'vocab*', 'merges*',
                    'sentencepiece*', 'generation_config*']:
        import glob
        for src_file in glob.glob(os.path.join(src_dir, pattern)):
            fname = os.path.basename(src_file)
            dst_file = os.path.join(output_dir, fname)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

    print(f'  Checkpoint saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Create ablated VLM checkpoints for VLMEvalKit evaluation')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Backend: llava-ov | internvl | qwen2vl | llava-llama3 | ...')
    parser.add_argument('--model_path', type=str, required=True,
                        help='HF model ID or local path')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Path to PMBT label directory')
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of transformer layers')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory for checkpoints')
    parser.add_argument('--categories', nargs='+',
                        default=['visual', 'text', 'multimodal'],
                        help='Categories to ablate (one checkpoint each)')
    parser.add_argument('--random', action='store_true', default=False,
                        help='Also create matched-count random ablation checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random ablation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to load model on (cpu recommended)')
    args = parser.parse_args()

    # Normalize model type
    effective_type = args.model_type
    if effective_type in ('qwen25vl-7b', 'qwen25vl-3b'):
        effective_type = 'qwen2vl'

    # Load labels
    print(f'\n{"="*60}')
    print(f'  Creating ablated checkpoints')
    print(f'  Model: {args.model_path} ({args.model_type})')
    print(f'  Labels: {args.label_dir}')
    print(f'  Categories: {args.categories}')
    print(f'{"="*60}\n')

    labels = load_labels(args.label_dir, args.n_layers)

    # Count neurons per category
    cat_counts = defaultdict(int)
    for layer_data in labels.values():
        for label in layer_data.values():
            cat_counts[label] += 1
    print(f'  Neuron counts: {dict(cat_counts)}\n')

    for category in args.categories:
        print(f'\n{"─"*60}')
        print(f'  Ablating: {category}')
        print(f'{"─"*60}')

        # Get neurons to zero
        neuron_map, total = get_neuron_map(labels, category, args.n_layers)
        if total == 0:
            print(f'  WARNING: No neurons found for category "{category}"')
            continue

        # Load fresh model
        model = load_model(effective_type, args.model_path, args.device)

        # Zero neurons (permanent modification)
        zero_neurons(model, effective_type, neuron_map)

        # Save checkpoint
        ckpt_dir = os.path.join(args.output_dir, f'ablated_{category}')
        save_checkpoint(model, args.model_path, ckpt_dir, effective_type)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f'  ✓ {category} checkpoint done: {ckpt_dir}')

        # ── Random matched-count checkpoint ──
        if args.random:
            print(f'\n{"─"*60}')
            print(f'  Random ablation (matched count={total}, for {category})')
            print(f'{"─"*60}')

            rand_map, rand_total = get_random_neuron_map(
                labels, total, args.n_layers, seed=args.seed)

            model = load_model(effective_type, args.model_path, args.device)
            zero_neurons(model, effective_type, rand_map)

            ckpt_dir = os.path.join(args.output_dir, f'random_{category}_count')
            save_checkpoint(model, args.model_path, ckpt_dir, effective_type)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f'  ✓ random_{category}_count checkpoint done: {ckpt_dir}')

    print(f'\n{"="*60}')
    print(f'  All checkpoints created in {args.output_dir}')
    print(f'{"="*60}')
    print(f'\n  Checkpoints:')
    for category in args.categories:
        print(f'    ablated_{category}/')
    if args.random:
        for category in args.categories:
            print(f'    random_{category}_count/')
    print(f'\n  Next: evaluate with VLMEvalKit')


if __name__ == '__main__':
    main()