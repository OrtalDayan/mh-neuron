#!/usr/bin/env python3
"""
Re-threshold existing classifier output to new p-value thresholds.
Extended to support llava-next-llama3-8b, qwen2-vl-7b, idefics2-8b.

Generates new label JSONs without re-running the classifier.
Only processes models whose classifier output exists.
"""
import json
from pathlib import Path
from collections import Counter


def rethreshold(input_path, output_path, new_threshold):
    with open(input_path) as f:
        data = json.load(f)

    out = {}
    stats = Counter()

    for layer_key, neurons in data.items():
        new_neurons = []
        for n in neurons:
            new_n = dict(n)
            p = n['p_value']
            cur_label = n['label']
            rate_diff = n['observed_rate_diff']

            if cur_label in ('text', 'visual'):
                if p < new_threshold:
                    new_label = cur_label
                else:
                    new_label = 'multimodal'
            else:  # multimodal or unknown
                if p < new_threshold:
                    new_label = 'text' if rate_diff < 0 else 'visual'
                else:
                    new_label = cur_label

            new_n['label'] = new_label
            new_neurons.append(new_n)
            stats[new_label] += 1

        out[layer_key] = new_neurons

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"      threshold p<{new_threshold}: {dict(stats)}")


def main():
    base_dir = Path('results/3-classify/full')

    models = [
        'llava-next-llama3-8b',
        'qwen2-vl-7b',
        'idefics2-8b',
    ]

    thresholds = {
        'p0.1':   0.1,
        'p0.01':  0.01,
        'p0.001': 0.001,
    }

    for model_name in models:
        model_dir = base_dir / model_name
        if not model_dir.exists():
            print(f"\n[SKIP] model dir not found: {model_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        for module_suffix in ['gate_up', 'attn']:
            src_dir = model_dir / f'llm_permutation_{module_suffix}_min100_max2048'
            src = src_dir / 'neuron_labels_permutation_all.json'

            if not src.exists():
                print(f"  [SKIP] {module_suffix} labels not found: {src}")
                continue

            print(f"\n  {module_suffix}:")
            for tag, thresh in thresholds.items():
                dst_dir = model_dir / f'llm_permutation_{module_suffix}_min100_max2048_{tag}'
                dst = dst_dir / 'neuron_labels_permutation_all.json'
                if dst.exists():
                    print(f"    [exists] {tag} — skip")
                    continue
                rethreshold(src, dst, thresh)


if __name__ == '__main__':
    main()
