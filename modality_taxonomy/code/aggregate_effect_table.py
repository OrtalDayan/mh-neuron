#!/usr/bin/env python3
"""Aggregate the long-format effect CSV into a paper-ready ranking table.

Reads the long CSV produced by plot_ablation_effect.py and writes two outputs:

  1. <output>_wide.csv — one row per (model, bench, cat). Columns:
       - effect_f0.01, effect_f0.05, ..., effect_f1.0  (per-fraction effects)
       - mean_effect_low_frac : mean effect over fractions ≤ low_frac_threshold
       - max_neg_effect       : most negative effect across all fractions
       - auc_effect           : AUC of the effect curve under log10(frac) trapezoid

  2. <output>_ranked.csv — same rows, sorted within each (bench, cat) bucket
     by mean_effect_low_frac ascending (most negative = strongest gap = top of the list).

Also prints a top-3 console summary per (bench, cat) so you can eyeball the leaders.

Usage:
    python3 code/aggregate_effect_table.py \\
        --input paper_figure_2_effect_effect_table.csv \\
        --output paper_figure_2_effect_summary
"""

import argparse
import numpy as np
import pandas as pd


def auc_log_x(fracs, vals):
    """AUC of vals across log10(fracs), trapezoid rule. NaN-safe (drops NaN points)."""
    fracs = np.asarray(fracs, dtype=float)
    vals  = np.asarray(vals,  dtype=float)
    valid = ~np.isnan(vals) & (fracs > 0)                       # log10 needs frac > 0; NaN drops
    if valid.sum() < 2:                                         # trapezoid needs ≥2 points
        return np.nan
    log_f = np.log10(fracs[valid])
    trap = getattr(np, 'trapezoid', None) or np.trapz           # numpy 2.x renamed trapz → trapezoid
    return float(trap(vals[valid], log_f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='Long-format CSV from plot_ablation_effect.py')
    p.add_argument('--output', required=True,
                   help='Output prefix; writes <prefix>_wide.csv and <prefix>_ranked.csv')
    p.add_argument('--low-frac-threshold', type=float, default=0.10,
                   help='Fractions ≤ this go into mean_effect_low_frac (default 0.10)')
    args = p.parse_args()

    df = pd.read_csv(args.input)

    # ── Pivot effect by frac so each frac becomes a column ──
    wide = df.pivot_table(
        index=['model', 'bench', 'cat'], columns='frac', values='effect'
    ).round(4)
    wide.columns = [f'effect_f{c:g}' for c in wide.columns]    # rename frac numbers → effect_f<frac>
    wide = wide.reset_index()                                   # flatten the multiindex

    # ── Per-group summary metrics (mean over low fracs, max-neg, AUC) ──
    summaries = []
    for (model, bench, cat), grp in df.groupby(['model', 'bench', 'cat']):
        grp = grp.sort_values('frac')                           # AUC needs ordered fracs
        low = grp[grp.frac <= args.low_frac_threshold]          # the regime where targeted should win
        summaries.append({
            'model': model, 'bench': bench, 'cat': cat,
            'mean_effect_low_frac': low['effect'].mean(),       # NaN-safe via pandas
            'max_neg_effect':       grp['effect'].min(),        # most negative effect across fracs
            'auc_effect':           auc_log_x(grp['frac'].values, grp['effect'].values),
        })
    summary_df = pd.DataFrame(summaries).round(4)

    # ── Merge per-frac and summary into one wide table ──
    result = wide.merge(summary_df, on=['model', 'bench', 'cat'])

    out_prefix = args.output[:-4] if args.output.endswith('.csv') else args.output
    out_wide   = f'{out_prefix}_wide.csv'
    out_ranked = f'{out_prefix}_ranked.csv'

    result.to_csv(out_wide, index=False)
    print(f'  ✓ Wrote wide table:   {out_wide}  ({len(result)} rows)')

    # ── Ranked: sort by (bench, cat) buckets, then mean_effect_low_frac ascending ──
    # Most negative effects bubble to top within each (bench, cat) — clear paper ranking.
    ranked = result.sort_values(
        ['bench', 'cat', 'mean_effect_low_frac'],
        ascending=[True, True, True],
    )
    ranked.to_csv(out_ranked, index=False)
    print(f'  ✓ Wrote ranked table: {out_ranked}')

    # ── Console summary: top-3 strongest (most negative) effects per (bench, cat) ──
    print('\n  Top-3 most-negative mean_effect_low_frac per (bench × cat):')
    for (bench, cat), grp in result.groupby(['bench', 'cat']):
        top = grp.nsmallest(3, 'mean_effect_low_frac')[
            ['model', 'mean_effect_low_frac', 'max_neg_effect', 'auc_effect']
        ]
        print(f'    [{bench} × {cat}]')
        for _, row in top.iterrows():
            print(f'      {row["model"]:25s}  '
                  f'mean={row["mean_effect_low_frac"]:8.2f}  '
                  f'max_neg={row["max_neg_effect"]:8.2f}  '
                  f'auc={row["auc_effect"]:8.2f}')


if __name__ == '__main__':
    main()
