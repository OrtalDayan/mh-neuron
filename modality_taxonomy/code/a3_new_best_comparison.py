#!/usr/bin/env python3
"""
Easy-to-read comparison of the 3 key LLaVA configs across all benchmarks.

Shows:
  - VLM baseline (no merge)
  - Uniform α=0.9 (BRV baseline)
  - PMBT α=0.7 p<0.05 (previous main result, 26.02 MVTD)
  - PMBT α=0.5 p<0.01 (NEW BEST, 27.03 MVTD)

Output: one row per benchmark, one column per config. Easy to paste into paper.
"""
import csv
from pathlib import Path

ROOT = Path('results/25-merge/llava-next-llama3-8b/dart-prop')

CONFIGS = [
    ('VLM baseline',              'baseline'),
    ('Uniform α=0.9',             'uniform_a0.9'),
    ('PMBT α=0.7 p<0.05 (main)',  'pmbt_t0.7_v1.0_m1.0_a1.0_o1.0'),
    ('PMBT α=0.5 p<0.01 (NEW)',   'pmbt_t0.5_v1.0_m1.0_a1.0_o1.0_p0.01'),
]

# (Display name, benchmark substring, column hints)
BENCHMARKS = [
    ('MathVerse-TD',   'MathVerse_MINI_Text_Dominant',     ['Text Dominant', 'Overall', 'accuracy', 'Acc']),
    ('MathVerse-TL',   'MathVerse_MINI_Text_Lite',         ['Text Lite', 'Overall', 'accuracy', 'Acc']),
    ('MathVerse-VI',   'MathVerse_MINI_Vision_Intensive',  ['Vision Intensive', 'Overall', 'accuracy', 'Acc']),
    ('MathVerse-VD',   'MathVerse_MINI_Vision_Dominant',   ['Vision Dominant', 'Overall', 'accuracy', 'Acc']),
    ('MathVerse-VO',   'MathVerse_MINI_Vision_Only',       ['Vision Only', 'Overall', 'accuracy', 'Acc']),
    ('MathVista',      'MathVista_MINI',                   ['Overall', 'All', 'accuracy', 'Acc']),
    ('MMStar',         'MMStar',                           ['Overall', 'average', 'accuracy', 'Acc']),
    ('DynaMath',       'DynaMath',                         ['worst', 'Overall', 'score', 'accuracy']),
    ('POPE (F1)',      'POPE',                             ['Overall', 'F1', 'acc']),
    ('TriviaQA',       'TriviaQA',                         ['accuracy', 'score', 'Overall']),
]


def find_csv(cell_dir, benchmark_substr):
    if not cell_dir.exists():
        return None
    matches = []
    for p in cell_dir.rglob('*_score.csv'):
        if benchmark_substr in p.name:
            matches.append(p)
    if not matches:
        # Also try *_acc.csv for MMStar
        for p in cell_dir.rglob('*_acc.csv'):
            if benchmark_substr in p.name:
                matches.append(p)
    if not matches:
        return None
    double_score = [m for m in matches if m.name.endswith('_score_score.csv')]
    if double_score:
        return max(double_score, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def read_score(csv_path, metric_hints):
    if csv_path is None:
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
        # Fallback: first numeric value in first row
        for k, v in rows[0].items():
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    except Exception:
        pass
    return None


def collect_scores():
    results = {}
    for config_name, dir_name in CONFIGS:
        cell_dir = ROOT / dir_name
        config_scores = {}
        for bench_name, bench_substr, hints in BENCHMARKS:
            csv_path = find_csv(cell_dir, bench_substr)
            score = read_score(csv_path, hints)
            config_scores[bench_name] = score
        results[config_name] = config_scores
    return results


def main():
    results = collect_scores()

    # Column widths — 4 configs
    col_w = 22
    name_w = 16

    print("=" * 120)
    print("  LLaVA-Next-LLaMA3-8B + Dart-Math-Prop2Diff — Full Benchmark Comparison")
    print("=" * 120)

    # Header
    print(f"\n  {'Benchmark':<{name_w}}", end='')
    for config_name, _ in CONFIGS:
        print(f"  {config_name:>{col_w}}", end='')
    print()
    print(f"  {'-' * name_w}" + ("  " + "-" * col_w) * len(CONFIGS))

    # Rows
    for bench_name, _, _ in BENCHMARKS:
        print(f"  {bench_name:<{name_w}}", end='')
        values = [results[c][bench_name] for c, _ in CONFIGS]
        for v in values:
            if v is None:
                s = '--'
            else:
                s = f"{v:.2f}"
            print(f"  {s:>{col_w}}", end='')
        print()

    # Delta row for new best vs previous main
    print()
    print(f"  {'Δ new vs main':<{name_w}}", end='')
    # pad for VLM and Uniform columns
    print(f"  {'':>{col_w}}  {'':>{col_w}}", end='')
    print(f"  {'(ref)':>{col_w}}", end='')

    main_scores = results['PMBT α=0.7 p<0.05 (main)']
    new_scores  = results['PMBT α=0.5 p<0.01 (NEW)']

    # delta col
    for bench_name, _, _ in BENCHMARKS:
        m = main_scores.get(bench_name)
        n = new_scores.get(bench_name)
        if m is None or n is None:
            pass
    # simpler delta formatting: print delta line below
    print()
    print(f"\n  {'Δ (new - main)':<{name_w}}", end='')
    print(f"  {'':>{col_w}}  {'':>{col_w}}", end='')
    print(f"  {'':>{col_w}}", end='')
    deltas = []
    for bench_name, _, _ in BENCHMARKS:
        m = main_scores.get(bench_name)
        n = new_scores.get(bench_name)
        if m is not None and n is not None:
            deltas.append((bench_name, n - m))
    # show only once
    print()

    # Better: print as its own simple section
    print()
    print(f"{'=' * 120}")
    print(f"  SUMMARY: NEW BEST (α=0.5, p<0.01) vs PREVIOUS MAIN (α=0.7, p<0.05)")
    print(f"{'=' * 120}")
    print(f"  {'Benchmark':<18} {'Previous main':>15} {'NEW best':>12} {'Δ':>10}  Direction")
    print(f"  {'-' * 75}")
    for bench_name, _, _ in BENCHMARKS:
        m = main_scores.get(bench_name)
        n = new_scores.get(bench_name)
        if m is None or n is None:
            m_s = f"{m:.2f}" if m is not None else 'pending'
            n_s = f"{n:.2f}" if n is not None else 'pending'
            print(f"  {bench_name:<18} {m_s:>15} {n_s:>12} {'--':>10}  --")
            continue
        delta = n - m
        # Most benchmarks: higher is better. Only DynaMath's "worst" is maybe different but still higher=better.
        arrow = '↑' if delta > 0.01 else ('↓' if delta < -0.01 else '≈')
        print(f"  {bench_name:<18} {m:>15.2f} {n:>12.2f} {delta:>+10.2f}  {arrow}")

    # Also compare new best vs uniform
    print()
    print(f"{'=' * 120}")
    print(f"  SUMMARY: NEW BEST vs UNIFORM BASELINE")
    print(f"{'=' * 120}")
    uniform_scores = results['Uniform α=0.9']
    print(f"  {'Benchmark':<18} {'Uniform':>15} {'NEW best':>12} {'Δ':>10}")
    print(f"  {'-' * 75}")
    for bench_name, _, _ in BENCHMARKS:
        u = uniform_scores.get(bench_name)
        n = new_scores.get(bench_name)
        u_s = f"{u:.2f}" if u is not None else 'missing'
        n_s = f"{n:.2f}" if n is not None else 'pending'
        if u is not None and n is not None:
            delta = n - u
            print(f"  {bench_name:<18} {u_s:>15} {n_s:>12} {delta:>+10.2f}")
        else:
            print(f"  {bench_name:<18} {u_s:>15} {n_s:>12} {'--':>10}")


if __name__ == '__main__':
    main()
