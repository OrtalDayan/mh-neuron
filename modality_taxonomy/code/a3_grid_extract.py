#!/usr/bin/env python3
"""
A3 grid extractor — tabulate MVTD + POPE + MathVista across 4×3 grid.

Assumes merge output dirs follow:
    results/25-merge/llava-next-llama3-8b/dart-prop/pmbt_t{X}_v1.0_m1.0_a1.0_o1.0[_pY]/
"""
import csv
from pathlib import Path
import re

ROOT = Path('results/25-merge/llava-next-llama3-8b/dart-prop')

P_TAGS = ['', '_p0.1', '_p0.01', '_p0.001']          # '' = default p<0.05
P_LABELS = ['p<0.05', 'p<0.1', 'p<0.01', 'p<0.001']
ALPHAS = ['0.7', '0.5', '0.3']


def find_score(merge_dir: Path, benchmark_substr: str, column: str):
    """
    Search for a *_score.csv matching benchmark_substr under merge_dir,
    return the named-column value as float, or None if not found.
    """
    for csv_path in merge_dir.rglob('*_score.csv'):
        if benchmark_substr not in csv_path.name:
            continue
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if column in row and row[column]:
                        return float(row[column])
        except Exception:
            continue
    return None


def mvtd_score(merge_dir: Path):
    # MathVerse TD uses column 'Text Dominant' or 'Overall' — try both
    for col in ('Text Dominant', 'Overall'):
        v = find_score(merge_dir, 'MathVerse_MINI_Text_Dominant', col)
        if v is not None:
            return v
    return None


def pope_score(merge_dir: Path):
    # POPE F1 is stored in the 'Overall' column per journal
    return find_score(merge_dir, 'POPE', 'Overall')


def mvi_score(merge_dir: Path):
    # MathVista overall accuracy
    for col in ('Overall', 'Accuracy', 'accuracy', 'Math'):
        v = find_score(merge_dir, 'MathVista', col)
        if v is not None:
            return v
    return None


def collect_grid():
    grid = {}   # (p_label, alpha) -> {mvtd, pope, mvi}
    for p_tag, p_label in zip(P_TAGS, P_LABELS):
        for a in ALPHAS:
            dir_name = f'pmbt_t{a}_v1.0_m1.0_a1.0_o1.0{p_tag}'
            merge_dir = ROOT / dir_name
            if not merge_dir.exists():
                grid[(p_label, a)] = None
                continue
            grid[(p_label, a)] = {
                'mvtd': mvtd_score(merge_dir),
                'pope': pope_score(merge_dir),
                'mvi':  mvi_score(merge_dir),
                'dir':  str(merge_dir),
            }
    return grid


def print_benchmark(grid, metric_key, title, fmt='{:>6.2f}'):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    # Header
    print(f"  {'α_text':<8}", end='')
    for p_label in P_LABELS:
        print(f"  {p_label:>10}", end='')
    print()
    # Rows
    for a in ALPHAS:
        print(f"  α={a:<6}", end='')
        for p_label in P_LABELS:
            cell = grid.get((p_label, a))
            if cell is None:
                print(f"  {'--':>10}", end='')
            elif cell[metric_key] is None:
                print(f"  {'pending':>10}", end='')
            else:
                print(f"  {fmt.format(cell[metric_key]):>10}", end='')
        print()


def main():
    grid = collect_grid()

    print_benchmark(grid, 'mvtd', 'MathVerse Text-Dominant (higher = better)')
    print_benchmark(grid, 'pope', 'POPE F1 (higher = better)')
    print_benchmark(grid, 'mvi',  'MathVista_MINI (higher = better)')

    # Summary of best cell
    print(f"\n{'=' * 70}")
    print(f"  BEST MVTD CELL")
    print(f"{'=' * 70}")
    best = None
    for k, v in grid.items():
        if v is None or v['mvtd'] is None:
            continue
        if best is None or v['mvtd'] > best[1]['mvtd']:
            best = (k, v)
    if best:
        k, v = best
        print(f"  Config: {k[0]}, α_text={k[1]}")
        print(f"  MVTD={v['mvtd']:.2f}, POPE={v['pope']:.2f}, MVI={v['mvi']:.2f}" if v['pope'] is not None and v['mvi'] is not None else f"  MVTD={v['mvtd']:.2f}")
        print(f"  Dir: {v['dir']}")

    # Missing cells
    missing = [k for k, v in grid.items() if v is None]
    if missing:
        print(f"\n  [WARN] {len(missing)} cells have no merge dir yet:")
        for k in missing:
            print(f"    {k[0]}, α={k[1]}")


if __name__ == '__main__':
    main()
