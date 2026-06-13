#!/usr/bin/env python3
"""Find the (text_bench, visual_bench) combinations with strongest dissociation.

A figure-design tool. The paper draft uses POPE as the visual benchmark; we
have TriviaQA and the two MathVerse splits as candidates for the text-bench
slot. This script scores all reasonable benchmark pairs to identify which
gives the cleanest visualisation of the double dissociation per model.

Terminology: filenames use 'text'/'visual'/'multimodal' for category; these
map to the paper's positional categories (text-position, etc.). The output
uses positional terminology consistent with the paper.

For each (model, ranking, fraction, text_bench, visual_bench), reports:

  text_specificity = drop(text-pos neurons -> text_bench)
                    - drop(text-pos neurons -> visual_bench)
  visual_specificity = drop(visual-pos neurons -> visual_bench)
                      - drop(visual-pos neurons -> text_bench)
  dissociation_strength = (text_specificity + visual_specificity) / 2

A high score here means: ablating text-position neurons damages the text
benchmark much more than the visual benchmark, AND ablating visual-position
neurons damages the visual benchmark much more than the text benchmark — the
classic double dissociation pattern.

Usage:
    python3 code/find_best_bench_pairs.py \\
        --models llava-next-llama3-8b idefics2-8b llava-onevision-7b \\
                 qwen2-vl-7b llava-1.5-7b internvl2.5-8b qwen2.5-vl-3b \\
        --ranking norm \\
        --output bench_pair_ranking.md
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


CELL_RE = re.compile(
    r'run1_(?P<ranking>D|norm|D_x_norm)_'
    r'gate_up_(?P<cat>text|visual|multimodal)_'
    r'(?P<bench>[A-Za-z_]+?)_'
    r'f(?P<frac>[\d.]+)_'
    r'(?P<trial>ranked|r\d)\.json$'
)

CANDIDATE_PAIRS = [
    ('TriviaQA', 'POPE'),
    ('TriviaQA', 'MathVerse_MINI_Vision_Only'),
    ('MathVerse_MINI_Text_Dominant', 'POPE'),
    ('MathVerse_MINI_Text_Dominant', 'MathVerse_MINI_Vision_Only'),
]

MODEL_LABELS = {
    'llava-next-llama3-8b': 'LLaVA-NeXT-LLaMA3-8B',
    'idefics2-8b': 'Idefics2-8B',
    'llava-onevision-7b': 'LLaVA-OneVision-7B',
    'qwen2-vl-7b': 'Qwen2-VL-7B',
    'llava-1.5-7b': 'LLaVA-1.5-7B',
    'internvl2.5-8b': 'InternVL2.5-8B',
    'qwen2.5-vl-3b': 'Qwen2.5-VL-3B',
}


def load_results(model_dir):
    out = []
    for fpath in glob.glob(os.path.join(model_dir, 'run1_*.json')):
        if os.path.getsize(fpath) < 200:
            continue
        m = CELL_RE.search(os.path.basename(fpath))
        if not m:
            continue
        try:
            data = json.load(open(fpath))
        except (json.JSONDecodeError, OSError):
            continue
        acc = data.get('accuracy_pct')
        if acc is None:
            continue
        out.append({
            'ranking': m['ranking'], 'cat': m['cat'], 'bench': m['bench'],
            'frac': float(m['frac']), 'trial': m['trial'], 'acc': float(acc),
        })
    return out


def get_cell(records, ranking, cat, bench, frac, trial='ranked'):
    for r in records:
        if (r['ranking'] == ranking and r['cat'] == cat and r['bench'] == bench
                and abs(r['frac'] - frac) < 1e-6 and r['trial'] == trial):
            return r['acc']
    return None


def baseline_acc(records, ranking, bench):
    candidates = [
        r for r in records
        if r['ranking'] == ranking and r['bench'] == bench and r['trial'] == 'ranked'
    ]
    if not candidates:
        candidates = [
            r for r in records
            if r['ranking'] == ranking and r['bench'] == bench
        ]
    if not candidates:
        return None
    smallest = min(r['frac'] for r in candidates)
    return float(np.mean([r['acc'] for r in candidates if r['frac'] == smallest]))


def score_pair(records, ranking, frac, text_bench, visual_bench):
    """Score how strongly a (text_bench, visual_bench) pair shows dissociation.

    Returns a dict with the four corner accuracies, the two specificities, and
    the overall dissociation strength. Returns None if any corner is missing.
    """
    base_t = baseline_acc(records, ranking, text_bench)
    base_v = baseline_acc(records, ranking, visual_bench)
    if base_t is None or base_v is None:
        return None

    cells = {
        ('text', text_bench):     get_cell(records, ranking, 'text', text_bench, frac),
        ('text', visual_bench):   get_cell(records, ranking, 'text', visual_bench, frac),
        ('visual', text_bench):   get_cell(records, ranking, 'visual', text_bench, frac),
        ('visual', visual_bench): get_cell(records, ranking, 'visual', visual_bench, frac),
    }
    if any(v is None for v in cells.values()):
        return None

    drop_t_on_t = base_t - cells[('text', text_bench)]
    drop_t_on_v = base_v - cells[('text', visual_bench)]
    drop_v_on_t = base_t - cells[('visual', text_bench)]
    drop_v_on_v = base_v - cells[('visual', visual_bench)]

    text_spec = drop_t_on_t - drop_t_on_v
    visual_spec = drop_v_on_v - drop_v_on_t
    strength = (text_spec + visual_spec) / 2

    return {
        'baseline_text': base_t,
        'baseline_visual': base_v,
        'drop_text_on_text': drop_t_on_t,
        'drop_text_on_visual': drop_t_on_v,
        'drop_visual_on_text': drop_v_on_t,
        'drop_visual_on_visual': drop_v_on_v,
        'text_specificity': text_spec,
        'visual_specificity': visual_spec,
        'strength': strength,
    }


def write_markdown(all_scores, path):
    """Group by model, show all bench-pair scores ranked."""
    with open(path, 'w') as f:
        f.write('# Best benchmark pairs for figure design\n\n')
        f.write('For each model, this table ranks candidate (text_bench, visual_bench)\n')
        f.write('pairs by dissociation strength. Pick the highest-scoring pair per model\n')
        f.write('for the main figure; relegate weaker pairs to the appendix.\n\n')
        f.write('Strength = (text_specificity + visual_specificity) / 2, in accuracy points.\n\n')

        by_model = defaultdict(list)
        for s in all_scores:
            by_model[s['model']].append(s)

        for model in sorted(by_model.keys()):
            label = MODEL_LABELS.get(model, model)
            f.write(f'## {label}\n\n')
            rows = sorted(
                by_model[model],
                key=lambda r: r['score']['strength'] if r['score'] else -1e9,
                reverse=True,
            )
            f.write('| frac | text_bench | visual_bench | text_spec | vis_spec | strength |\n')
            f.write('|------|------------|--------------|-----------|----------|----------|\n')
            for r in rows:
                if r['score'] is None:
                    f.write(f"| {r['frac']:.2f} | {r['text_bench']} | {r['visual_bench']} | — | — | — |\n")
                    continue
                s = r['score']
                f.write(
                    f"| {r['frac']:.2f} | {r['text_bench']} | {r['visual_bench']} | "
                    f"{s['text_specificity']:+.1f} | {s['visual_specificity']:+.1f} | "
                    f"**{s['strength']:+.1f}** |\n"
                )
            f.write('\n')

        # Cross-model summary: which pair is best on average?
        f.write('## Cross-model summary\n\n')
        f.write('Mean dissociation strength of each (text_bench, visual_bench) pair across all\n')
        f.write('models, fractions, where data was available. Use this to pick a single pair\n')
        f.write('for the main figure consistent across models.\n\n')

        pair_scores = defaultdict(list)
        for s in all_scores:
            if s['score'] is not None:
                key = (s['text_bench'], s['visual_bench'])
                pair_scores[key].append(s['score']['strength'])

        f.write('| text_bench | visual_bench | n | mean strength | min | max |\n')
        f.write('|------------|--------------|---|--------------|-----|-----|\n')
        for (tb, vb), vals in sorted(
            pair_scores.items(),
            key=lambda kv: -np.mean(kv[1]),
        ):
            f.write(
                f"| {tb} | {vb} | {len(vals)} | "
                f"{np.mean(vals):+.1f} | {min(vals):+.1f} | {max(vals):+.1f} |\n"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--ranking', default='norm', choices=['D', 'norm', 'D_x_norm'])
    p.add_argument('--fractions', nargs='+', type=float, default=[0.01, 0.05, 0.10])
    p.add_argument('--results-root', default='results/24-ranked-ablation/full')
    p.add_argument('--output', default='bench_pair_ranking.md')
    args = p.parse_args()

    print(f'Scoring all candidate benchmark pairs')
    print(f'  Ranking: {args.ranking}')
    print(f'  Fractions: {args.fractions}')
    print(f'  Models: {len(args.models)}')
    print()

    all_scores = []
    for model in args.models:
        model_dir = os.path.join(args.results_root, model, 'run1')
        if not os.path.isdir(model_dir):
            print(f'  ⚠ {model_dir} missing')
            continue
        records = load_results(model_dir)
        label = MODEL_LABELS.get(model, model)
        if not records:
            print(f'  ⚠ {label}: no records')
            continue
        print(f'  {label}: {len(records)} records')

        for text_bench, visual_bench in CANDIDATE_PAIRS:
            for frac in args.fractions:
                score = score_pair(records, args.ranking, frac, text_bench, visual_bench)
                all_scores.append({
                    'model': model, 'ranking': args.ranking, 'frac': frac,
                    'text_bench': text_bench, 'visual_bench': visual_bench,
                    'score': score,
                })

    write_markdown(all_scores, args.output)
    print(f'\n  ✓ Wrote {args.output}')

    # Print best pair per model to console
    print('\n── Best pair per model (across all fractions) ──')
    by_model = defaultdict(list)
    for s in all_scores:
        if s['score'] is not None:
            by_model[s['model']].append(s)
    for model in sorted(by_model.keys()):
        label = MODEL_LABELS.get(model, model)
        best = max(by_model[model], key=lambda r: r['score']['strength'])
        print(
            f"  {label:<28}  best={best['text_bench']:>30} vs {best['visual_bench']:<28} "
            f"f={best['frac']:.2f}  strength={best['score']['strength']:+.2f}"
        )


if __name__ == '__main__':
    main()
