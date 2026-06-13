#!/usr/bin/env python3
"""
A3 cross-model extractor — shows LLaVA + Qwen + Idefics (2 conventions)
side-by-side across the α_text × p-threshold grid.

Tests whether the LLaVA finding (p<0.01, α=0.5 → MV-TD=27.03) generalizes.
"""
import csv
from pathlib import Path
from collections import defaultdict

P_LABELS = ['p<0.1', 'p<0.05 (main)', 'p<0.01', 'p<0.001']
P_TAGS   = ['_p0.1', '',              '_p0.01', '_p0.001']

# Per-model config: dir, donor, alphas to test, alpha_m convention, label
MODELS = [
    {
        'name':    'LLaVA-Next-LLaMA3-8B',
        'root':    Path('results/25-merge/llava-next-llama3-8b/dart-prop'),
        'alphas':  ['0.7', '0.6', '0.5', '0.4', '0.3'],
        'alpha_m': '1.0',
        'baseline_mvtd': 23.60,   # uniform α=0.9
    },
    {
        'name':    'Qwen2-VL-7B + Qwen2-Math',
        'root':    Path('results/25-merge/qwen2-vl-7b/qwen2-math'),
        'alphas':  ['0.9', '0.7', '0.5', '0.3'],
        'alpha_m': '1.0',
        'baseline_mvtd': 33.88,   # from A1 journal
    },
    {
        'name':    'Idefics2-8B + MAmmoTH-1 [Conv A: α_m=1.0]',
        'root':    Path('results/25-merge/idefics2-8b/mammoth1'),
        'alphas':  ['0.8', '0.7', '0.5', '0.3'],
        'alpha_m': '1.0',
        'baseline_mvtd': 22.59,   # from A1
    },
    {
        'name':    'Idefics2-8B + MAmmoTH-1 [Conv B: α_m=0.9, BRV-original]',
        'root':    Path('results/25-merge/idefics2-8b/mammoth1'),
        'alphas':  ['0.8', '0.7', '0.5', '0.3'],
        'alpha_m': '0.9',
        'baseline_mvtd': 22.59,
    },
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


def collect_grid(model_cfg):
    grid = {}
    for p_tag, p_label in zip(P_TAGS, P_LABELS):
        for a in model_cfg['alphas']:
            dir_name = f"pmbt_t{a}_v1.0_m{model_cfg['alpha_m']}_a1.0_o1.0{p_tag}"
            cell_dir = model_cfg['root'] / dir_name
            if not cell_dir.exists():
                grid[(p_label, a)] = None
                continue
            mvtd = read_score(find_csv(cell_dir, 'MathVerse_MINI_Text_Dominant'),
                              ['Text Dominant', 'Overall', 'Acc', 'accuracy', 'score'])
            pope = read_score(find_csv(cell_dir, 'POPE'),
                              ['Overall', 'F1', 'acc', 'score'])
            mvi  = read_score(find_csv(cell_dir, 'MathVista_MINI'),
                              ['Overall', 'All', 'Acc', 'accuracy', 'score'])
            grid[(p_label, a)] = {'mvtd': mvtd, 'pope': pope, 'mvi': mvi}
    return grid


def print_grid(model_cfg, grid, metric='mvtd'):
    print(f"\n{'─' * 90}")
    print(f"  {model_cfg['name']} — MV-TD")
    print(f"  Baseline (uniform): {model_cfg['baseline_mvtd']:.2f}")
    print(f"{'─' * 90}")
    # Header
    print(f"  {'α_text':<8}", end='')
    for p_label in P_LABELS:
        print(f"  {p_label:>14}", end='')
    print()
    # Rows
    best = None
    for a in model_cfg['alphas']:
        print(f"  α={a:<6}", end='')
        for p_label in P_LABELS:
            cell = grid.get((p_label, a))
            if cell is None:
                s = '--'
            elif cell[metric] is None:
                s = 'pending'
            else:
                val = cell[metric]
                s = f"{val:.2f}"
                if best is None or val > best['val']:
                    best = {'val': val, 'p': p_label, 'a': a, 'pope': cell['pope']}
            print(f"  {s:>14}", end='')
        print()
    if best:
        delta = best['val'] - model_cfg['baseline_mvtd']
        pope_s = f"POPE={best['pope']:.2f}" if best['pope'] is not None else "POPE=?"
        print(f"  ⭐ Best: ({best['p']}, α={best['a']}) → {best['val']:.2f} "
              f"(Δ={delta:+.2f} vs uniform, {pope_s})")


def main():
    print("=" * 90)
    print("  A3 Grid Results — all 3 models (+ Idefics dual convention)")
    print("=" * 90)

    for model_cfg in MODELS:
        grid = collect_grid(model_cfg)
        print_grid(model_cfg, grid, 'mvtd')

    # Cross-model summary
    print(f"\n{'=' * 90}")
    print("  GENERALIZATION CHECK")
    print(f"{'=' * 90}")
    print(f"  LLaVA finding: (p<0.01, α=0.5) was the optimum with +3.43 over uniform.")
    print(f"  Does this cell generalize?\n")

    check_cells = [('p<0.01', '0.5')]
    for model_cfg in MODELS:
        grid = collect_grid(model_cfg)
        for p_label, a in check_cells:
            if a not in model_cfg['alphas']:
                continue
            cell = grid.get((p_label, a))
            if cell is None:
                status = "dir missing"
                mvtd_str = "--"
            elif cell['mvtd'] is None:
                status = "pending"
                mvtd_str = "--"
            else:
                delta = cell['mvtd'] - model_cfg['baseline_mvtd']
                status = "✅" if delta > 0 else "❌"
                mvtd_str = f"{cell['mvtd']:.2f} (Δ={delta:+.2f})"
            print(f"  {status}  {model_cfg['name'][:60]:<60}  MVTD at (p<0.01, α=0.5): {mvtd_str}")


if __name__ == '__main__':
    main()
