#!/usr/bin/env python3
"""Print classification distribution summary across models and hooks.

Usage:
    python code/print_classification_summary.py
    python code/print_classification_summary.py --models llava-1.5-7b qwen2-vl-7b
    python code/print_classification_summary.py -o results/classification_summary.txt
"""

import argparse
import json
import os

ALL_MODELS = [
    'llava-1.5-7b', 'llava-next-llama3-8b', 'llava-onevision-7b',
    'internvl2.5-8b', 'qwen2-vl-7b', 'qwen2.5-vl-7b',
    'qwen2.5-vl-3b', 'idefics2-8b',
]

HOOKS = {
    'gate':    ('llm_permutation',                        'llm_fixed_threshold'),
    'gate_up': ('llm_permutation_gate_up_min100_max2048', 'llm_fixed_threshold_gate_up_min100_max2048'),
    'attn':    ('llm_permutation_attn_min100_max2048',    'llm_fixed_threshold_attn_min100_max2048'),
}


def build_table(base_dir, models, hooks_dict):
    lines = []
    for method in ['PMBT', 'FT']:
        idx = 0 if method == 'PMBT' else 1
        stats_file = ('permutation_stats_all.json' if method == 'PMBT'
                      else 'classification_stats_all.json')

        lines.append(f'\n{"="*100}')
        lines.append(f'  {method} Classification Distribution')
        lines.append(f'{"="*100}')
        lines.append(f'  {"Model":<25s} {"Hook":<10s} '
                     f'{"Visual":>12s} {"Text":>12s} '
                     f'{"Multi":>12s} {"Unknown":>12s} {"Total":>10s}')
        lines.append(f'  {"-"*25} {"-"*10} '
                     f'{"-"*12} {"-"*12} {"-"*12} {"-"*12} {"-"*10}')

        for mn in models:
            for hp in ['gate', 'gate_up', 'attn']:
                d_name = hooks_dict[hp][idx]
                f = os.path.join(base_dir, mn, d_name, stats_file)
                if not os.path.isfile(f):
                    lines.append(f'  {mn:<25s} {hp:<10s} {"MISSING":>12s}')
                    continue
                d = json.load(open(f))
                s = d['stats']
                t = sum(s.values())
                lines.append(
                    f'  {mn:<25s} {hp:<10s} '
                    f'{s["visual"]:>7,} ({100*s["visual"]/t:4.1f}%)'
                    f'{s["text"]:>7,} ({100*s["text"]/t:4.1f}%)'
                    f'{s["multimodal"]:>7,} ({100*s["multimodal"]/t:4.1f}%)'
                    f'{s["unknown"]:>7,} ({100*s["unknown"]/t:4.1f}%)'
                    f'{t:>9,}')
            lines.append('')
    return lines


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Print classification distribution summary')
    p.add_argument('--base_dir', default='results/3-classify/full',
                   help='Base directory for classification results')
    p.add_argument('--models', nargs='+', default=None,
                   help='Models to include (default: all 8)')
    p.add_argument('--output', '-o', default=None,
                   help='Save summary to file (also prints to stdout)')
    args = p.parse_args()

    models = args.models or ALL_MODELS
    lines = build_table(args.base_dir, models, HOOKS)
    text = '\n'.join(lines)

    print(text)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(text + '\n')
        print(f'\nSaved to {args.output}')