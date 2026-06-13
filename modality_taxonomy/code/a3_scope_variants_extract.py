#!/usr/bin/env python3
"""
Scope-variants extractor — tabulates the 24 LLaVA scope-variant cells.

Variants tested:
  V1 downproj    pmbt_tX_v1.0_m1.0_mlponly_a1.0_o1.0_downproj_pY/
  V2 kvmerge     pmbt_tX_v1.0_m1.0_kv_a1.0_o1.0_pY/
  V3 mlponly     pmbt_tX_v1.0_m1.0_mlponly_a1.0_o1.0_pY/
Plus reference cells for context:
  Full PMBT     pmbt_tX_v1.0_m1.0_a1.0_o1.0_pY/   (no scope suffix)

For α_text ∈ {0.4, 0.5, 0.6, 0.7}, p ∈ {p0.01, p0.001}.
"""
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path('results/25-merge/llava-next-llama3-8b/dart-prop')

ALPHAS = ['0.7', '0.6', '0.5', '0.4']
P_TAGS = ['_p0.01', '_p0.001']
P_LABELS = ['p<0.01', 'p<0.001']

VARIANTS = [
    ('Full PMBT (reference)',  '_v1.0_m1.0_a1.0_o1.0'),
    ('V3 MLP-only',            '_v1.0_m1.0_mlponly_a1.0_o1.0'),
    ('V1 downproj-only',       '_v1.0_m1.0_mlponly_a1.0_o1.0_downproj'),
    ('V2 KV merge',            '_v1.0_m1.0_kv_a1.0_o1.0'),
]


def find_csv(cell_dir, benchmark_substr):
    matches = []
    for p in cell_dir.rglob('*_score.csv'):
        if benchmark_substr in p.name:
            matches.append(p)
    if not matches:
        return None
    double_score = [m for m in matches if m.name.endswith('_score_score.csv')]
    if double_score:
        return max(double_score, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def read_score(csv_path, metric_hints):
    if csv_path is None or not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        cols_lower = {k.lower(): k for k in rows[0].keys() if k}
        for hint in metric_hints:
            for lkey, orig_key in cols_lower.items():
                if hint.lower() in lkey:
                    for row in rows:
                        v = row.get(orig_key, '')
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            continue
        for k, v in rows[0].items():
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    except Exception:
        pass
    return None


def collect_cell(variant_suffix, alpha, p_tag):
    dir_name = f"pmbt_t{alpha}{variant_suffix}{p_tag}"
    cell_dir = ROOT / dir_name
    if not cell_dir.exists():
        return None
    return {
        'mvtd': read_score(find_csv(cell_dir, 'MathVerse_MINI_Text_Dominant'),
                           ['Text Dominant', 'Overall', 'Acc', 'accuracy', 'score']),
        'pope': read_score(find_csv(cell_dir, 'POPE'),
                           ['Overall', 'F1', 'acc', 'score']),
        'mvi':  read_score(find_csv(cell_dir, 'MathVista_MINI'),
                           ['Overall', 'All', 'Acc', 'accuracy', 'score']),
        'dir':  str(cell_dir),
    }


def print_table(title, metric, unit=''):
    print(f"\n{'=' * 80}")
    print(f"  {title}{unit}")
    print(f"{'=' * 80}")
    # Header row
    print(f"  {'Variant':<22}", end='')
    for a in ALPHAS:
        for p_label in P_LABELS:
            print(f"  α={a} {p_label:<7}", end='')
    print()

    # Data rows
    for variant_name, variant_suffix in VARIANTS:
        print(f"  {variant_name:<22}", end='')
        for a in ALPHAS:
            for p_tag in P_TAGS:
                cell = collect_cell(variant_suffix, a, p_tag)
                if cell is None:
                    s = '--'
                elif cell[metric] is None:
                    s = 'pending'
                else:
                    s = f"{cell[metric]:.2f}"
                print(f"  {s:>11}", end='')
        print()


def main():
    print("=" * 80)
    print("  LLaVA Scope Variants at p<0.01 and p<0.001")
    print("=" * 80)

    print_table('MathVerse Text-Dominant (higher = better)', 'mvtd')
    print_table('POPE F1 (higher = better)', 'pope')
    print_table('MathVista_MINI (higher = better)', 'mvi')

    # Best per variant on MVTD
    print(f"\n{'=' * 80}")
    print("  BEST PER VARIANT (MVTD)")
    print(f"{'=' * 80}")
    for variant_name, variant_suffix in VARIANTS:
        best = None
        for a in ALPHAS:
            for p_tag, p_label in zip(P_TAGS, P_LABELS):
                cell = collect_cell(variant_suffix, a, p_tag)
                if cell is None or cell['mvtd'] is None:
                    continue
                if best is None or cell['mvtd'] > best['mvtd']:
                    best = {'mvtd': cell['mvtd'], 'pope': cell['pope'],
                            'mvi': cell['mvi'], 'a': a, 'p': p_label}
        if best:
            pope_s = f"{best['pope']:.2f}" if best['pope'] else '—'
            mvi_s  = f"{best['mvi']:.2f}"  if best['mvi']  else '—'
            print(f"  {variant_name:<22}  α={best['a']} {best['p']}:  "
                  f"MVTD={best['mvtd']:.2f}  POPE={pope_s}  MVI={mvi_s}")
        else:
            print(f"  {variant_name:<22}  pending")

    # Compared to known best (Full PMBT α=0.5 p<0.01 = 27.03)
    print(f"\n  Known best (Full PMBT α=0.5 p<0.01): MVTD=27.03, POPE=86.65")
    print(f"  Uniform baseline: MVTD=23.60")
    print(f"  VLM baseline:     MVTD=20.10")

    # Cell counts
    print(f"\n{'=' * 80}")
    print("  COMPLETION STATUS")
    print(f"{'=' * 80}")
    variant_counts = defaultdict(lambda: {'done': 0, 'pending': 0, 'missing': 0})
    for variant_name, variant_suffix in VARIANTS:
        for a in ALPHAS:
            for p_tag in P_TAGS:
                cell = collect_cell(variant_suffix, a, p_tag)
                if cell is None:
                    variant_counts[variant_name]['missing'] += 1
                elif cell['mvtd'] is None:
                    variant_counts[variant_name]['pending'] += 1
                else:
                    variant_counts[variant_name]['done'] += 1
    for variant_name, counts in variant_counts.items():
        total = counts['done'] + counts['pending'] + counts['missing']
        print(f"  {variant_name:<22}  done={counts['done']}/{total}  "
              f"pending={counts['pending']}  missing={counts['missing']}")


if __name__ == '__main__':
    main()
