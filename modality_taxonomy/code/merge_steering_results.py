"""
merge_steering_results.py — Step 12: Collect all step 11 steering results
into a single JSON file for plotting.

Walks the steering output directory tree:
    {steering_dir}/{taxonomy}/alpha_{value}/ablation_summary.json

Produces a unified JSON:
    {output_dir}/steering_merged.json

Structure:
    {
        "model_name": "llava-ov-7b",
        "taxonomy": "perm",
        "alphas": [0, 0.25, 0.5, ...],
        "conditions": ["baseline", "ablate_vis", "ablate_text", ...],
        "benchmarks": ["pope", "chair", "triviaqa", ...],
        "results": {
            "0.5": {
                "ablate_vis": {
                    "pope": {"accuracy": ..., "f1": ..., "yes_ratio": ...},
                    "chair": {"chair_i": ..., "chair_s": ...},
                    ...
                    "n_neurons_ablated": ...,
                    "top_n": ...,
                    "ranking_method": ...,
                    "halluc_scores_path": ...,
                    "deltas_vs_baseline": { ... }
                },
                ...
            }
        }
    }

Usage:
    python merge_steering_results.py \
        --steering_dir results/3-classify/full/llava-ov-7b/ablation/steering \
        --taxonomy perm \
        --model_name llava-ov-7b \
        --output_dir results/13-plots/llava-ov-7b
"""

import argparse                                                            # Line 1: parse command-line arguments
import json                                                                # Line 2: read/write JSON
import os                                                                  # Line 3: path manipulation
import glob                                                                # Line 4: file pattern matching


def parse_args():
    """Parse command-line arguments for steering result merging."""
    p = argparse.ArgumentParser(
        description='Merge step 11 steering results across all alpha '
                    'values into a single JSON for plotting.')
    p.add_argument('--steering_dir', required=True,                        # Line 5: root steering output dir
                   help='Root steering directory containing '
                        '{taxonomy}/alpha_{value}/ subdirs')
    p.add_argument('--taxonomy', default='perm',                           # Line 6: which taxonomy prefix to use
                   help='Taxonomy prefix used in step 11 (ft or perm)')
    p.add_argument('--model_name', default='model',                        # Line 7: model name for metadata
                   help='Model name for metadata in merged output')
    p.add_argument('--output_dir', required=True,                          # Line 8: output directory
                   help='Directory to write steering_merged.json')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)                            # Line 9: create output dir

    # Find all alpha directories
    tax_dir = os.path.join(args.steering_dir, args.taxonomy)               # Line 10: e.g. .../steering/perm/
    if not os.path.isdir(tax_dir):                                         # Line 11: check exists
        print(f'ERROR: Taxonomy directory not found: {tax_dir}')
        print(f'  Available: {os.listdir(args.steering_dir) if os.path.isdir(args.steering_dir) else "N/A"}')
        return

    alpha_dirs = sorted(glob.glob(os.path.join(tax_dir, 'alpha_*')))       # Line 12: find all alpha dirs
    print(f'Found {len(alpha_dirs)} alpha directories in {tax_dir}')

    if not alpha_dirs:                                                     # Line 13: nothing to merge
        print('ERROR: No alpha directories found.')
        return

    # Collect results
    all_alphas = []                                                        # Line 14: list of alpha values
    all_conditions = set()                                                 # Line 15: unique condition names
    all_benchmarks = set()                                                 # Line 16: unique benchmark names
    results = {}                                                           # Line 17: {alpha_str: {condition: data}}

    benchmark_keys = ['pope', 'chair', 'vqav2', 'vqa_perception',         # Line 18: known benchmark keys
                      'vqa_knowledge', 'triviaqa', 'mmlu', 'mme']

    for alpha_dir in alpha_dirs:                                           # Line 19: iterate alpha directories
        # Extract alpha value from directory name
        dir_name = os.path.basename(alpha_dir)                             # Line 20: e.g. "alpha_0.5"
        alpha_str = dir_name.replace('alpha_', '')                         # Line 21: e.g. "0.5"
        try:
            alpha_val = float(alpha_str)                                   # Line 22: parse float
        except ValueError:
            print(f'  WARNING: Cannot parse alpha from {dir_name}, skipping')
            continue

        # Load ablation_summary.json
        summary_path = os.path.join(alpha_dir, 'ablation_summary.json')    # Line 23: expected summary file
        if not os.path.isfile(summary_path):                               # Line 24: check exists
            print(f'  WARNING: No summary at {summary_path}, skipping')
            continue

        with open(summary_path) as f:                                      # Line 25: load summary
            summary = json.load(f)

        conditions = summary.get('conditions', {})                         # Line 26: condition results dict
        if not conditions:                                                 # Line 27: empty summary
            print(f'  WARNING: Empty conditions in {summary_path}, skipping')
            continue

        all_alphas.append(alpha_val)                                       # Line 28: record alpha value
        results[alpha_str] = {}                                            # Line 29: init alpha entry

        for cond_name, cond_data in conditions.items():                    # Line 30: iterate conditions
            all_conditions.add(cond_name)                                  # Line 31: record condition
            results[alpha_str][cond_name] = cond_data                      # Line 32: store full data

            # Track which benchmarks are present
            for bk in benchmark_keys:                                      # Line 33: check each benchmark
                if bk in cond_data:
                    all_benchmarks.add(bk)                                 # Line 34: record benchmark

        print(f'  alpha={alpha_str}: {len(conditions)} conditions, '
              f'benchmarks: {[b for b in benchmark_keys if b in list(conditions.values())[0]]}')

    # Sort
    all_alphas.sort()                                                      # Line 35: sort alpha values
    all_conditions = sorted(all_conditions)                                # Line 36: sort conditions
    all_benchmarks = sorted(all_benchmarks)                                # Line 37: sort benchmarks

    # Build merged output
    merged = {                                                             # Line 38: top-level structure
        'model_name': args.model_name,
        'taxonomy': args.taxonomy,
        'alphas': all_alphas,
        'conditions': all_conditions,
        'benchmarks': all_benchmarks,
        'results': results,
    }

    # Save
    merged_path = os.path.join(args.output_dir, 'steering_merged.json')    # Line 39: output path
    with open(merged_path, 'w') as f:                                      # Line 40: write JSON
        json.dump(merged, f, indent=2)

    print(f'\nMerged steering results:')
    print(f'  Alphas: {all_alphas}')
    print(f'  Conditions: {all_conditions}')
    print(f'  Benchmarks: {all_benchmarks}')
    print(f'  Saved → {merged_path}')


if __name__ == '__main__':
    main()
