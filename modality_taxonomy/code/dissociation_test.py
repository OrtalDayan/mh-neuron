#!/usr/bin/env python3
"""Compute the double-dissociation verdict from a long-format effect CSV.

For each model, computes the mean gap-from-random (effect) at low fractions
(f ≤ low_frac_threshold) for the four cells of the dissociation matrix:
    text-cat ablation × text benches    (should be most negative)
    text-cat ablation × visual benches  (should be near zero)
    visual-cat ablation × visual benches (should be most negative)
    visual-cat ablation × text benches  (should be near zero)

Double dissociation requires (a) text→text more negative than text→visual AND
(b) visual→visual more negative than visual→text. Both halves must hold.

Reads the long-format CSV produced by plot_ablation_effect.py.

Usage:
    python3 code/dissociation_test.py \\
        --input paper_figure_2_effect_D_effect_table.csv \\
        --text-benches TriviaQA MathVerse_MINI_Text_Dominant \\
        --visual-benches POPE MathVerse_MINI_Vision_Only
"""

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='Long-format CSV from plot_ablation_effect.py')
    p.add_argument('--text-benches', nargs='+',
                   default=['TriviaQA', 'MathVerse_MINI_Text_Dominant'])
    p.add_argument('--visual-benches', nargs='+',
                   default=['POPE', 'MathVerse_MINI_Vision_Only'])
    p.add_argument('--low-frac-threshold', type=float, default=0.10,
                   help='Fractions ≤ this go into the per-cell mean (default 0.10)')
    p.add_argument('--max-frac', type=float, default=0.5,
                   help='Drop fracs > this from analysis to remove f=1.0 outliers (default 0.5)')
    p.add_argument('--output', default=None,
                   help='Optional CSV path to write the verdict table')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df = df[(df.frac >= 0) & (df.frac <= args.max_frac)]        # filter f range
    low = df[df.frac <= args.low_frac_threshold]                # low-frac regime only

    # Build per-(model, cat, bench-side) means by averaging effect across fracs and benches
    rows = []
    for (model, cat), grp in low.groupby(['model', 'cat']):
        if cat not in ('text', 'visual'):                       # skip multimodal/unknown for verdict
            continue
        eff_text   = grp[grp.bench.isin(args.text_benches)]['effect'].mean()
        eff_visual = grp[grp.bench.isin(args.visual_benches)]['effect'].mean()
        rows.append({
            'model': model, 'cat': cat,
            'eff_on_text_benches':   round(eff_text,   2),
            'eff_on_visual_benches': round(eff_visual, 2),
        })
    out = pd.DataFrame(rows)

    print(f'Mean effect (gap from random), fracs ≤ {args.low_frac_threshold:g}, by ablation cat × bench-side:')
    print(out.to_string(index=False))
    print()

    print('Double dissociation requires:')
    print(f'  (text-cat)   eff_on_text  more negative than eff_on_visual')
    print(f'  (visual-cat) eff_on_visual more negative than eff_on_text')
    print()

    print(f'{"Model":25s}  text-half  visual-half  Verdict')
    print('-' * 75)
    verdicts = []
    for model in out['model'].unique():
        t = out[(out.model == model) & (out.cat == 'text')]
        v = out[(out.model == model) & (out.cat == 'visual')]
        # Skip if we don't have both cats for this model
        if t.empty or v.empty:
            print(f'{model:25s}  insufficient data (need both text and visual cat rows)')
            continue
        t = t.iloc[0]; v = v.iloc[0]
        text_half_ok   = t['eff_on_text_benches']   < t['eff_on_visual_benches']
        visual_half_ok = v['eff_on_visual_benches'] < v['eff_on_text_benches']
        verdict = ('✓✓ DOUBLE DISSOCIATION' if text_half_ok and visual_half_ok
                   else '✓ text only'        if text_half_ok
                   else '✓ visual only'      if visual_half_ok
                   else '✗ neither')
        print(f'{model:25s}  '
              f'{("✓" if text_half_ok else "✗"):>9s}  '
              f'{("✓" if visual_half_ok else "✗"):>11s}  '
              f'{verdict}')
        verdicts.append({
            'model': model,
            'text_text':   t['eff_on_text_benches'],
            'text_visual': t['eff_on_visual_benches'],
            'visual_text': v['eff_on_text_benches'],
            'visual_visual': v['eff_on_visual_benches'],
            'text_half_ok':   text_half_ok,
            'visual_half_ok': visual_half_ok,
            'verdict': verdict,
        })

    if args.output and verdicts:
        pd.DataFrame(verdicts).to_csv(args.output, index=False)
        print(f'\n  ✓ Verdict CSV written: {args.output}')


if __name__ == '__main__':
    main()
