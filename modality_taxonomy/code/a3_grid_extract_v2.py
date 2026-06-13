#!/usr/bin/env python3
"""
A3 grid extractor (v2) — auto-detects CSV column names.

Reads all score CSVs under a cell's directory and picks the best column
per benchmark based on known column-name patterns. Falls back gracefully.
"""
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path('results/25-merge/llava-next-llama3-8b/dart-prop')

P_LABELS = ['p<0.1', 'p<0.05 (main)', 'p<0.01', 'p<0.001']
P_TAGS   = ['_p0.1', '',              '_p0.01', '_p0.001']
ALPHAS   = ['0.7', '0.5', '0.3']


def find_csv(cell_dir: Path, benchmark_substr: str):
    """Find the most-recent score CSV matching benchmark_substr."""
    matches = []
    for p in cell_dir.rglob('*_score.csv'):
        if benchmark_substr in p.name:
            # Score files with double _score_score.csv are the final scored output from VLMEvalKit
            # Prefer them over the single _score.csv intermediate
            matches.append(p)
    if not matches:
        return None
    # Prefer double-score (final) over single-score (intermediate)
    double_score = [m for m in matches if m.name.endswith('_score_score.csv')]
    if double_score:
        return max(double_score, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def read_score(csv_path: Path, metric_hints):
    """
    Read a score value from CSV.
    metric_hints: list of column-name substrings to try in order (case-insensitive).
    Returns (value, column_name) or (None, None).
    """
    if csv_path is None or not csv_path.exists():
        return None, None
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None, None
        # Case-insensitive column match by substring
        cols_lower = {k.lower(): k for k in rows[0].keys() if k}
        for hint in metric_hints:
            for lkey, orig_key in cols_lower.items():
                if hint.lower() in lkey:
                    for row in rows:
                        v = row.get(orig_key, '')
                        try:
                            return float(v), orig_key
                        except (ValueError, TypeError):
                            continue
        # Fall back to first numeric value
        for k, v in rows[0].items():
            try:
                return float(v), k
            except (ValueError, TypeError):
                continue
    except Exception:
        pass
    return None, None


def collect_grid():
    """
    Returns: dict keyed by (p_label, alpha) -> {mvtd, pope, mvi, dir, columns}
    """
    grid = {}
    columns_seen = defaultdict(set)

    for p_tag, p_label in zip(P_TAGS, P_LABELS):
        for a in ALPHAS:
            dir_name = f'pmbt_t{a}_v1.0_m1.0_a1.0_o1.0{p_tag}'
            cell_dir = ROOT / dir_name
            if not cell_dir.exists():
                grid[(p_label, a)] = None
                continue

            mvtd_csv = find_csv(cell_dir, 'MathVerse_MINI_Text_Dominant')
            pope_csv = find_csv(cell_dir, 'POPE')
            mvi_csv  = find_csv(cell_dir, 'MathVista_MINI')

            mvtd, mvtd_col = read_score(mvtd_csv, ['Text Dominant', 'Overall', 'Acc', 'score'])
            pope, pope_col = read_score(pope_csv, ['Overall', 'F1', 'acc', 'score'])
            # MathVista: prefer 'All' overall, then 'Math', then 'Overall'
            mvi, mvi_col = read_score(mvi_csv, ['Overall', 'All', 'score', 'Acc'])

            columns_seen['mvtd'].add(mvtd_col)
            columns_seen['pope'].add(pope_col)
            columns_seen['mvi'].add(mvi_col)

            grid[(p_label, a)] = {
                'mvtd': mvtd, 'pope': pope, 'mvi': mvi,
                'mvtd_col': mvtd_col, 'pope_col': pope_col, 'mvi_col': mvi_col,
                'dir': str(cell_dir),
            }
    return grid, columns_seen


def print_table(grid, metric_key, title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'α_text':<8}", end='')
    for p_label in P_LABELS:
        print(f"  {p_label:>14}", end='')
    print()
    for a in ALPHAS:
        print(f"  α={a:<6}", end='')
        for p_label in P_LABELS:
            cell = grid.get((p_label, a))
            if cell is None:
                s = '--'
            elif cell[metric_key] is None:
                s = 'pending'
            else:
                s = f"{cell[metric_key]:.2f}"
            print(f"  {s:>14}", end='')
        print()


def main():
    grid, columns_seen = collect_grid()

    print_table(grid, 'mvtd', 'MathVerse Text-Dominant (MV-TD, higher = better)')
    print_table(grid, 'pope', 'POPE F1 (higher = better)')
    print_table(grid, 'mvi',  'MathVista_MINI (higher = better)')

    # Best MVTD
    print(f"\n{'=' * 70}\n  BEST MVTD CELL\n{'=' * 70}")
    best = None
    for k, v in grid.items():
        if v is None or v['mvtd'] is None:
            continue
        if best is None or v['mvtd'] > best[1]['mvtd']:
            best = (k, v)
    if best:
        k, v = best
        mvtd_str = f"{v['mvtd']:.2f}"
        pope_str = f"{v['pope']:.2f}" if v['pope'] is not None else 'pending'
        mvi_str  = f"{v['mvi']:.2f}"  if v['mvi']  is not None else 'pending'
        print(f"  Config: {k[0]}, α_text={k[1]}")
        print(f"  MVTD={mvtd_str}, POPE={pope_str}, MVI={mvi_str}")

    # Delta vs baseline (uniform α=0.9, MV-TD ≈ 23.60 from journal)
    BASELINE_MVTD = 23.60
    print(f"\n  Baseline (Uniform α=0.9): MV-TD = {BASELINE_MVTD}")
    print(f"  PMBT α=0.7 p<0.05 (original main): MV-TD = 26.02 (Δ = +2.42)")

    # Columns detected (diagnostic)
    print(f"\n{'=' * 70}\n  CSV COLUMNS DETECTED (diagnostic)\n{'=' * 70}")
    for metric, cols in columns_seen.items():
        cols_clean = sorted(c for c in cols if c is not None)
        print(f"  {metric}: {cols_clean}")

    # Status
    missing = [k for k, v in grid.items() if v is not None and v['mvtd'] is None]
    total = len(grid)
    complete = sum(1 for v in grid.values() if v is not None and v['mvtd'] is not None)
    print(f"\n  Status: {complete}/{total} cells have MVTD scores")


if __name__ == '__main__':
    main()
