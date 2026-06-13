#!/usr/bin/env python3
"""Plot ranked-vs-random ablation curves per modality, per benchmark, per model.

For each (model, benchmark) panel, two curves per modality category (text, visual,
multimodal):
  - Solid line: ranked-ablation accuracy (the targeted ablation)
  - Dashed line: mean accuracy over the 5 random-ablation trials at same fraction
  - Shaded band: [min, max] range across the 5 random trials

The visible gap (solid - dashed) shows the effect of targeted vs random ablation.
A long-format CSV stores per-cell numerical effects for downstream pivots/ranking.

Usage:
    python3 code/plot_ablation_effect.py \\
        --models llava-next-llama3-8b idefics2-8b llava-onevision-7b qwen2-vl-7b \\
        --benches TriviaQA MathVerse_MINI_Text_Dominant POPE MathVerse_MINI_Vision_Only \\
        --output ablation_effect_per_modality.pdf
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# Same regex as plot_ablation_curves.py — works for MathVerse_MINI_* benches because
# [A-Za-z_]+? is lazy and the trailing _f<frac> boundary anchors the bench span.
CELL_RE = re.compile(
    r'run1_(?P<ranking>D|norm|D_x_norm)_'
    r'gate_up_(?P<cat>text|visual|multimodal)_'
    r'(?P<bench>[A-Za-z_]+?)_'
    r'f(?P<frac>[\d.]+)_'
    r'(?P<trial>ranked|r\d)\.json$'
)

MODEL_LABELS = {
    'llava-next-llama3-8b': 'LLaVA-NeXT-LLaMA3-8B',
    'idefics2-8b':          'Idefics2-8B',
    'llava-onevision-7b':   'LLaVA-OneVision-7B',
    'qwen2-vl-7b':          'Qwen2-VL-7B',
    'llava-1.5-7b':         'LLaVA-1.5-7B',
    'internvl2.5-8b':       'InternVL2.5-8B',
    'qwen2.5-vl-3b':        'Qwen2.5-VL-3B',
}

# One color per modality (both ranked and random for a cat share the color)
CAT_COLORS = {
    'text':       '#d62728',  # red
    'visual':     '#1f77b4',  # blue
    'multimodal': '#9467bd',  # purple
}

FRACTIONS_DEFAULT = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def load_all_results(model_dir):
    """Load all run1_*.json cells from one model dir. Returns list of records."""
    out = []
    for fpath in glob.glob(os.path.join(model_dir, 'run1_*.json')):
        if os.path.getsize(fpath) < 200:                       # drop 35-byte error stubs
            continue
        m = CELL_RE.search(os.path.basename(fpath))
        if not m:                                              # filename doesn't match expected schema
            continue
        try:
            data = json.load(open(fpath))
        except (json.JSONDecodeError, OSError):
            continue
        acc = data.get('accuracy_pct')
        if acc is None or not isinstance(acc, (int, float)):
            continue
        out.append({
            'ranking': m['ranking'],
            'cat':     m['cat'],
            'bench':   m['bench'],
            'frac':    float(m['frac']),
            'trial':   m['trial'],
            'acc':     float(acc),
        })
    return out


def aggregate_per_cat(records, ranking, cat, bench):
    """For one (cat, bench), bucket records by frac and summarize random trials.

    Returns frac-sorted list of dicts with keys:
        frac, ranked, rand_mean, rand_min, rand_max, rand_std, n_rand
    NaN where data missing. ranked is single-trial; rand_* aggregate over r0..r4.
    """
    by_frac = defaultdict(lambda: {'ranked': None, 'random': []})
    for r in records:
        if r['ranking'] != ranking or r['cat'] != cat or r['bench'] != bench:
            continue
        if r['trial'] == 'ranked':
            by_frac[r['frac']]['ranked'] = r['acc']
        else:                                                  # r0, r1, r2, r3, r4
            by_frac[r['frac']]['random'].append(r['acc'])

    rows = []
    for f in sorted(by_frac.keys()):
        v = by_frac[f]
        rand_arr = np.array(v['random'], dtype=float) if v['random'] else np.array([])
        rows.append({
            'frac':      f,
            'ranked':    v['ranked'] if v['ranked'] is not None else np.nan,
            'rand_mean': float(np.mean(rand_arr))         if rand_arr.size      else np.nan,
            'rand_min':  float(np.min(rand_arr))          if rand_arr.size      else np.nan,
            'rand_max':  float(np.max(rand_arr))          if rand_arr.size      else np.nan,
            'rand_std':  float(np.std(rand_arr, ddof=1))  if rand_arr.size > 1  else 0.0,
            'n_rand':    int(rand_arr.size),
        })
    return rows


def plot_panel(ax, records, ranking, bench, cats, title,
               show_xlabel=True, show_ylabel=True, show_legend=False):
    """One (model, bench) panel: per cat, draw shaded [min,max] band, dashed mean, solid ranked."""
    for cat in cats:
        rows = aggregate_per_cat(records, ranking, cat, bench)
        if not rows:
            continue
        fracs  = np.array([r['frac']      for r in rows])
        ranked = np.array([r['ranked']    for r in rows])
        rmean  = np.array([r['rand_mean'] for r in rows])
        rmin   = np.array([r['rand_min']  for r in rows])
        rmax   = np.array([r['rand_max']  for r in rows])
        color  = CAT_COLORS.get(cat, '#888888')

        # Random [min, max] band — drawn first so curves render on top
        valid_band = ~np.isnan(rmin) & ~np.isnan(rmax)
        if valid_band.any():
            ax.fill_between(
                fracs[valid_band], rmin[valid_band], rmax[valid_band],
                color=color, alpha=0.12, zorder=1, linewidth=0,
            )

        # Random mean — dashed line
        valid_rand = ~np.isnan(rmean)
        if valid_rand.any():
            ax.plot(
                fracs[valid_rand], rmean[valid_rand], '--', color=color,
                linewidth=1.5, alpha=0.75, zorder=2,
                label=f'{cat} random (mean of 5)',
            )

        # Ranked — solid line with markers (the targeted ablation we care about)
        valid_rk = ~np.isnan(ranked)
        if valid_rk.any():
            ax.plot(
                fracs[valid_rk], ranked[valid_rk], '-o', color=color,
                linewidth=2, markersize=5, zorder=3,
                label=f'{cat} ranked',
            )

    ax.set_xscale('log')
    ax.set_xticks(FRACTIONS_DEFAULT)
    ax.set_xticklabels([f'{f:g}' for f in FRACTIONS_DEFAULT], fontsize=8)
    ax.set_xlim(0.008, 1.2)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel('Fraction of neurons ablated', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Accuracy (%)', fontsize=9)
    if show_legend:
        ax.legend(loc='lower left', fontsize=7, frameon=True, ncol=2)


def write_csv(model_data, ranking, benches, cats, csv_path):
    """Long-format CSV — one row per (model, bench, cat, frac)."""
    cols = ['model', 'bench', 'cat', 'frac',
            'ranked_acc', 'random_mean', 'random_min', 'random_max',
            'random_std', 'n_random', 'effect']

    def fmt(x):
        return f'{x:.4f}' if isinstance(x, (int, float)) and not np.isnan(x) else 'nan'

    with open(csv_path, 'w') as cf:
        cf.write(','.join(cols) + '\n')
        for m in model_data:
            for bench in benches:
                for cat in cats:
                    rows = aggregate_per_cat(m['records'], ranking, cat, bench)
                    for r in rows:
                        effect = r['ranked'] - r['rand_mean']  # NaN-safe via numpy semantics
                        cf.write(','.join([
                            m['full'], bench, cat, f'{r["frac"]:g}',
                            fmt(r['ranked']),    fmt(r['rand_mean']),
                            fmt(r['rand_min']),  fmt(r['rand_max']),
                            fmt(r['rand_std']),  str(r['n_rand']),
                            fmt(effect),
                        ]) + '\n')


def plot_figure(args):
    cats = ['text', 'visual', 'multimodal']

    # Load each model's records
    model_data = []
    for model in args.models:
        full = model
        if full not in MODEL_LABELS:                           # allow short aliases
            for mk in MODEL_LABELS:
                if model in mk:
                    full = mk
                    break
        label = MODEL_LABELS.get(full, full)
        model_dir = os.path.join(args.results_root, full, 'run1')
        if not os.path.isdir(model_dir):
            print(f'  ⚠ {model_dir} does not exist, skipping')
            continue
        records = load_all_results(model_dir)
        print(f'  {label}: {len(records)} records loaded')
        model_data.append({'full': full, 'label': label, 'records': records})

    if not model_data:
        print('  No model data — aborting.')
        return

    # CSV (long format) — written before plot so even if plotting fails the table survives
    out_pdf  = args.output if args.output.endswith('.pdf') else args.output + '.pdf'
    csv_path = out_pdf.replace('.pdf', '_effect_table.csv')
    write_csv(model_data, args.ranking, args.benches, cats, csv_path)
    print(f'\n  ✓ Saved CSV: {csv_path}')

    # Plot grid: rows=models, cols=benches
    n_models = len(model_data)
    n_cols   = len(args.benches)
    fig_w    = max(3.0 * n_cols, 8)                            # ~3 in/col, floor at 8
    fig_h    = max(2.6 * n_models, 4)
    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for row, m in enumerate(model_data):
        for col, bench in enumerate(args.benches):
            plot_panel(
                axes[row, col], m['records'], args.ranking, bench, cats,
                title=f'{m["label"]}\non {bench}',
                show_xlabel=(row == n_models - 1),
                show_ylabel=(col == 0),
                show_legend=(row == 0 and col == 0),           # legend in top-left only
            )

    fig.suptitle(
        f'Ranked vs random ablation curves per modality — ranking: {args.ranking}\n'
        f'Solid = ranked targeted ablation; dashed = mean of 5 random trials; band = [min, max]',
        fontsize=11, y=1.005,
    )

    out_png = out_pdf.replace('.pdf', '.png')
    fig.savefig(out_pdf, bbox_inches='tight', dpi=150)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f'  ✓ Saved: {out_pdf}')
    print(f'  ✓ Saved: {out_png}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['llava-next-llama3-8b'])
    p.add_argument('--ranking', default='norm', choices=['D', 'norm', 'D_x_norm'])
    p.add_argument('--benches', nargs='+',
                   default=['TriviaQA', 'MathVerse_MINI_Text_Dominant',
                            'POPE', 'MathVerse_MINI_Vision_Only'])
    p.add_argument('--results-root', default='results/24-ranked-ablation/full')
    p.add_argument('--output', default='ablation_effect_per_modality.pdf')
    args = p.parse_args()
    plot_figure(args)


if __name__ == '__main__':
    main()
