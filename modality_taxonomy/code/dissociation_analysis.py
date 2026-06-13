#!/usr/bin/env python3
"""Compare double dissociation strength across rankings using ll3 results.

For each (ranking, category, benchmark, fraction) combination, computes:
  - ranked accuracy (the 'rk' trial — top-K neurons by ranking criterion)
  - random baseline accuracy (mean across r0..r4 trials)
  - drop = baseline_at_frac0 - ranked_at_frac
  - effect = ranked_drop - random_drop  (positive = ranking causally identifies neurons)

Then summarizes per ranking how strongly it produces double dissociation:
  - text neurons → big drop on TriviaQA (text bench), small drop on POPE/MV_Vision (visual benches)
  - visual neurons → big drop on POPE/MV_Vision, small drop on TriviaQA

Usage:
    python3 dissociation_analysis.py --model llava-next-llama3-8b
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


CELL_RE = re.compile(
    r'run1_(?P<ranking>D|norm|D_x_norm)_'
    r'gate_up_(?P<cat>text|visual|multimodal)_'
    r'(?P<bench>[A-Za-z_]+?)_'
    r'f(?P<frac>[\d.]+)_'
    r'(?P<trial>ranked|r\d)\.json$'
)


def load_results(results_dir):
    """Walk results dir, yield (ranking, cat, bench, frac, trial, accuracy) tuples."""
    for fpath in glob.glob(os.path.join(results_dir, 'run1_*.json')):
        if os.path.getsize(fpath) < 200:
            continue  # skip error placeholders
        m = CELL_RE.search(os.path.basename(fpath))
        if not m:
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            acc = data.get('accuracy_pct')
            if acc is None or not isinstance(acc, (int, float)):
                continue
            yield (
                m['ranking'], m['cat'], m['bench'],
                float(m['frac']), m['trial'], float(acc),
            )
        except (json.JSONDecodeError, OSError):
            continue


def aggregate(records):
    """Group records into (ranking, cat, bench, frac) -> {ranked: acc, random: [accs]}."""
    grouped = defaultdict(lambda: {'ranked': None, 'random': []})
    for ranking, cat, bench, frac, trial, acc in records:
        key = (ranking, cat, bench, frac)
        if trial == 'ranked':
            grouped[key]['ranked'] = acc
        else:
            grouped[key]['random'].append(acc)
    return grouped


def compute_baseline(grouped):
    """For each (ranking, cat, bench), use the smallest-fraction (rk or random mean) as baseline."""
    baselines = {}  # (ranking, cat, bench) -> baseline_acc
    by_rcb = defaultdict(list)
    for (rank, cat, bench, frac), vals in grouped.items():
        by_rcb[(rank, cat, bench)].append((frac, vals))
    for key, fracs_vals in by_rcb.items():
        # Use the smallest fraction as proxy for "approximately no ablation"
        fracs_vals.sort(key=lambda x: x[0])
        smallest_frac, smallest_vals = fracs_vals[0]
        if smallest_vals['ranked'] is not None:
            baselines[key] = smallest_vals['ranked']
        elif smallest_vals['random']:
            baselines[key] = sum(smallest_vals['random']) / len(smallest_vals['random'])
    return baselines


def assess_dissociation(grouped, baselines, fraction=0.10):
    """For each ranking, compute the dissociation strength at a given fraction.

    For a given ranking R:
      - effect_text = drop on TriviaQA from ablating TEXT neurons
      - effect_visual = drop on POPE+MV_Vision from ablating VISUAL neurons
      - leakage_text_to_visual = drop on POPE+MV_Vision from ablating TEXT neurons
      - leakage_visual_to_text = drop on TriviaQA from ablating VISUAL neurons

    Strong dissociation: high effect_text + effect_visual, low leakage.
    """
    # Map cat → expected benchmark family for "on-target" drops
    text_benches = {'TriviaQA'}
    visual_benches = {'POPE', 'MathVerse_MINI_Vision_Only'}

    summary = {}
    for ranking in ['D', 'norm', 'D_x_norm']:
        on_target_drops = []  # text neurons → text bench; visual neurons → visual bench
        off_target_drops = []  # text → visual bench; visual → text bench

        for (rank, cat, bench, frac), vals in grouped.items():
            if rank != ranking or abs(frac - fraction) > 1e-6 or vals['ranked'] is None:
                continue
            base = baselines.get((rank, cat, bench))
            if base is None:
                continue
            drop = base - vals['ranked']
            random_mean = (
                sum(vals['random']) / len(vals['random'])
                if vals['random'] else None
            )
            random_drop = (base - random_mean) if random_mean is not None else 0.0
            net_effect = drop - random_drop  # ranked drop minus random drop

            if cat == 'text' and bench in text_benches:
                on_target_drops.append((bench, net_effect))
            elif cat == 'visual' and bench in visual_benches:
                on_target_drops.append((bench, net_effect))
            elif cat == 'text' and bench in visual_benches:
                off_target_drops.append((bench, net_effect))
            elif cat == 'visual' and bench in text_benches:
                off_target_drops.append((bench, net_effect))

        on_mean = (
            sum(e for _, e in on_target_drops) / len(on_target_drops)
            if on_target_drops else 0.0
        )
        off_mean = (
            sum(e for _, e in off_target_drops) / len(off_target_drops)
            if off_target_drops else 0.0
        )
        summary[ranking] = {
            'on_target_mean_drop': on_mean,
            'off_target_mean_drop': off_mean,
            'dissociation_score': on_mean - off_mean,  # higher = better
            'n_on': len(on_target_drops),
            'n_off': len(off_target_drops),
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--model', default='llava-next-llama3-8b',
        help='Model directory name under results/24-ranked-ablation/full/',
    )
    p.add_argument(
        '--fraction', type=float, default=0.10,
        help='Ablation fraction to evaluate (default 0.10)',
    )
    p.add_argument(
        '--results-root', default='results/24-ranked-ablation/full',
    )
    args = p.parse_args()

    results_dir = os.path.join(args.results_root, args.model, 'run1')
    if not os.path.isdir(results_dir):
        raise SystemExit(f'  ✗ {results_dir} does not exist')

    print(f'  Loading from: {results_dir}')
    records = list(load_results(results_dir))
    print(f'  Loaded {len(records)} valid result records')

    grouped = aggregate(records)
    baselines = compute_baseline(grouped)
    summary = assess_dissociation(grouped, baselines, fraction=args.fraction)

    print()
    print(f'  Double dissociation strength @ fraction={args.fraction}')
    print(f'  (on_target = text→TriviaQA, visual→POPE+MVVisual)')
    print(f'  (off_target = text→POPE+MVVisual, visual→TriviaQA)')
    print(f'  (dissociation = on_target_drop - off_target_drop, higher is better)')
    print()
    print(f'  {"Ranking":<12} {"OnTgt drop":>12} {"OffTgt drop":>12} {"Dissoc":>10} {"n_on":>6} {"n_off":>6}')
    print(f'  {"-" * 65}')
    for ranking in ['D', 'norm', 'D_x_norm']:
        s = summary[ranking]
        print(
            f'  {ranking:<12} '
            f'{s["on_target_mean_drop"]:>12.2f} '
            f'{s["off_target_mean_drop"]:>12.2f} '
            f'{s["dissociation_score"]:>10.2f} '
            f'{s["n_on"]:>6d} {s["n_off"]:>6d}'
        )

    # Best ranking
    best = max(summary.items(), key=lambda kv: kv[1]['dissociation_score'])
    print()
    print(f'  >>> Best ranking by dissociation score: {best[0]} (score={best[1]["dissociation_score"]:.2f})')


if __name__ == '__main__':
    main()
