#!/usr/bin/env python3
"""Analyze neuron categorization vs global activation norm across models.

Reads per-layer neuron_labels.json files from the classify directory and tests
whether the categorical labels (text/visual/multimodal/unknown) actually
identify modality-distinct neurons, or whether one category is disproportionately
skewed toward globally-high-norm neurons. The latter would explain why one
category's ablation can damage off-category benchmarks (e.g. LL3 text → MV_VO).

Five analyses per model:
  1. Label distribution      — how many neurons in each cat
  2. Activation stats per cat — mean/median/max global_max_activation
  3. Cat composition of global top-5% — is one cat over-represented at high norms?
  4. Top-skew per cat         — what % of each cat's members land in global top-5%
  5. Ablation count at f=0.05 — how many neurons each cat actually loses

Usage:
    python3 code/neuron_categorization_analysis.py \\
        --models llava-next-llama3-8b idefics2-8b llava-onevision-7b qwen2-vl-7b \\
        --output neuron_categorization_analysis.csv
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def load_neuron_labels(model_dir):
    """Walk per-layer neuron_labels.json files and return a single DataFrame.

    Each row: one neuron with {layer, neuron_idx, label, pv, pt, pm, pu,
    global_max_activation, global_id}. global_id = 'L<layer>_N<neuron>' for
    cross-layer uniqueness.
    """
    rows = []
    layer_re = re.compile(r'layers\.(\d+)\.mlp')
    layer_dirs = sorted(glob.glob(
        os.path.join(model_dir, 'model.language_model.layers.*.mlp.down_proj')
    ))
    for layer_dir in layer_dirs:
        m = layer_re.search(layer_dir)
        if not m:
            continue
        layer = int(m.group(1))
        labels_path = os.path.join(layer_dir, 'neuron_labels.json')
        if not os.path.isfile(labels_path):
            continue
        try:
            data = json.load(open(labels_path))
        except (json.JSONDecodeError, OSError):
            continue
        for entry in data:
            rows.append({
                'layer': layer,
                'neuron_idx': entry['neuron_idx'],
                'label': entry['label'],
                'pv': entry.get('pv', 0.0),
                'pt': entry.get('pt', 0.0),
                'pm': entry.get('pm', 0.0),
                'pu': entry.get('pu', 0.0),
                'global_max_activation': entry.get('global_max_activation', 0.0),
                'global_id': f'L{layer}_N{entry["neuron_idx"]}',
            })
    return pd.DataFrame(rows)


def analyze_model(model, df, top_frac):
    """Run all 5 analyses for one model. Returns list of summary rows for the CSV."""
    cats_all = ['text', 'visual', 'multimodal', 'unknown']
    n_total = len(df)
    n_layers = df['layer'].nunique()

    print(f'\n  ═══ {model} ═══')
    print(f'    Loaded {n_total:,} neurons across {n_layers} layers')

    # ── 1. Label distribution ──
    print(f'\n    [1] Label distribution:')
    counts = df['label'].value_counts()
    label_pct = {}
    for label in cats_all:
        c = int(counts.get(label, 0))
        pct = c / n_total * 100 if n_total else 0
        label_pct[label] = pct
        print(f'      {label:12s}: {c:7,d} ({pct:5.1f}%)')

    # ── 2. Activation stats per cat ──
    print(f'\n    [2] global_max_activation per cat:')
    print(f'      {"cat":12s}  {"mean":>8s}  {"median":>8s}  {"p95":>8s}  {"max":>8s}  {"n":>7s}')
    act_stats = {}
    for cat in cats_all:
        sub = df[df.label == cat]['global_max_activation']
        if len(sub) == 0:
            continue
        s = {
            'mean':   float(sub.mean()),
            'median': float(sub.median()),
            'p95':    float(sub.quantile(0.95)),
            'max':    float(sub.max()),
            'n':      int(len(sub)),
        }
        act_stats[cat] = s
        print(f'      {cat:12s}  {s["mean"]:8.4f}  {s["median"]:8.4f}  {s["p95"]:8.4f}  {s["max"]:8.4f}  {s["n"]:7,d}')

    # ── 3. Cat composition of global top-K ──
    K_global = max(1, int(n_total * top_frac))
    df_sorted = df.sort_values('global_max_activation', ascending=False)
    top_global = df_sorted.head(K_global)
    print(f'\n    [3] Cat composition of GLOBAL top-{int(top_frac*100)}% by activation (K={K_global:,}):')
    print(f'        Shows which cat dominates the highest-norm neurons across the whole model.')
    print(f'        If a cat\'s % here >> its overall %, that cat is skewed toward high norms.')
    top_pct = {}
    for cat in cats_all:
        c = int((top_global['label'] == cat).sum())
        pct = c / K_global * 100 if K_global else 0
        top_pct[cat] = pct
        delta = pct - label_pct.get(cat, 0)
        print(f'      {cat:12s}: {c:6,d} ({pct:5.1f}%)  '
              f'overall={label_pct.get(cat, 0):5.1f}%  Δ={delta:+5.1f}pp')

    # ── 4. Per-cat: % of members in global top-K (norm-skew per cat) ──
    print(f'\n    [4] Per-cat: % of members landing in global top-{int(top_frac*100)}%:')
    print(f'        Higher = the cat\'s neurons concentrate at the top of the global norm distribution.')
    top_global_ids = set(top_global['global_id'])
    skew = {}
    for cat in cats_all:
        cat_df = df[df.label == cat]
        if len(cat_df) == 0:
            continue
        n_in_top = sum(gid in top_global_ids for gid in cat_df['global_id'])
        pct = n_in_top / len(cat_df) * 100
        skew[cat] = pct
        # Baseline: if labels were uniformly distributed across norms, this would equal top_frac*100
        baseline = top_frac * 100
        ratio = pct / baseline if baseline else 0
        print(f'      {cat:12s}: {n_in_top:6,d}/{len(cat_df):6,d} = {pct:5.2f}%  '
              f'(baseline {baseline:.2f}%, ratio={ratio:.2f}×)')

    # ── 5. Ablation count at f=0.05 — how many neurons does each cat lose? ──
    print(f'\n    [5] Ablation count at f={top_frac:g} per cat (top {int(top_frac*100)}% of cat members):')
    print(f'        Different cats have different sizes, so ablation footprints differ.')
    abl_count = {}
    for cat in cats_all:
        cat_df = df[df.label == cat]
        n_abl = max(1, int(len(cat_df) * top_frac)) if len(cat_df) else 0
        abl_count[cat] = n_abl
        print(f'      {cat:12s}: ablates {n_abl:6,d} neurons (out of {len(cat_df):6,d})')

    # ── Pack summary rows for the CSV ──
    summary = []
    for cat in cats_all:
        if cat not in act_stats:
            continue
        summary.append({
            'model': model,
            'cat': cat,
            'n_neurons': act_stats[cat]['n'],
            'pct_of_total': round(label_pct[cat], 2),
            'act_mean':   round(act_stats[cat]['mean'], 4),
            'act_median': round(act_stats[cat]['median'], 4),
            'act_p95':    round(act_stats[cat]['p95'], 4),
            'act_max':    round(act_stats[cat]['max'], 4),
            'pct_of_global_top_K':       round(top_pct.get(cat, 0), 2),
            'pct_of_cat_in_global_top':  round(skew.get(cat, 0), 2),
            'norm_skew_ratio':           round(skew.get(cat, 0) / (top_frac * 100), 2),
            f'ablation_count_at_f{top_frac:g}': abl_count.get(cat, 0),
        })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+',
                   default=['llava-next-llama3-8b', 'idefics2-8b',
                            'llava-onevision-7b', 'qwen2-vl-7b'])
    p.add_argument('--classify-root', default='results/3-classify/full')
    p.add_argument('--variant', default='llm_fixed_threshold_gate_up_min100_max2048')
    p.add_argument('--top-frac', type=float, default=0.05,
                   help='Fraction defining "top": 0.05 matches f=0.05 ablation cells')
    p.add_argument('--output', default='neuron_categorization_analysis.csv')
    args = p.parse_args()

    all_rows = []
    for model in args.models:
        model_dir = os.path.join(args.classify_root, model, args.variant)
        if not os.path.isdir(model_dir):
            print(f'  ⚠ {model_dir} does not exist, skipping')
            continue
        df = load_neuron_labels(model_dir)
        if df.empty:
            print(f'  ⚠ No neuron_labels.json files found under {model_dir}')
            continue
        all_rows.extend(analyze_model(model, df, args.top_frac))

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(args.output, index=False)
        print(f'\n  ✓ Saved summary CSV: {args.output}  ({len(out_df)} rows)')
    else:
        print('\n  No data collected — nothing to write.')


if __name__ == '__main__':
    main()
