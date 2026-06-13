"""Compute BRV-style evaluation splits from VLMEvalKit output.

Uses VLMEvalKit's own post_check() for exact per-sample correctness,
then groups by 'category' column for MathVista General/Math splits.

Usage:
    python code/compute_brv_splits.py --results-root results/25-merge/llava-next-llama3-8b/
    python code/compute_brv_splits.py --results-root results/25-merge/llava-next-llama3-8b/dart-prop/
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# Add VLMEvalKit to path so we can import their scoring functions
VLMEVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modern_vlms', 'VLMEvalKit_brv')
if not os.path.exists(VLMEVAL_DIR):
    VLMEVAL_DIR = os.path.join(os.getcwd(), 'modern_vlms', 'VLMEvalKit_brv')
sys.path.insert(0, VLMEVAL_DIR)

try:
    from vlmeval.dataset.utils.mathvista import post_check
    print('  [OK] Imported VLMEvalKit post_check()')
    USE_VLMEVAL = True
except ImportError as e:
    print(f'  [warn] Could not import VLMEvalKit post_check: {e}')
    USE_VLMEVAL = False


def find_files(eval_dir, pattern):
    """Find files matching pattern, preferring non-timestamp dirs."""
    files = glob.glob(os.path.join(eval_dir, '**', pattern), recursive=True)
    seen = {}
    for f in files:
        bench = extract_benchmark_name(f)
        if bench not in seen or '/T2' not in f:
            seen[bench] = f
    return seen


def extract_benchmark_name(filepath):
    """Extract benchmark name from VLMEvalKit filename."""
    base = os.path.basename(filepath)
    benchmarks = [
        'MathVista_MINI', 'MathVerse_MINI_Text_Dominant',
        'MathVerse_MINI_Text_Lite', 'MathVerse_MINI_Vision_Intensive',
        'MathVerse_MINI_Vision_Dominant', 'MathVerse_MINI_Vision_Only',
        'MMStar', 'DynaMath', 'MathVision_MINI', 'MM-Math',
    ]
    for b in benchmarks:
        if b in base:
            return b
    return base


def compute_mathvista_splits(xlsx_path):
    """Compute MathVista All/General/Math using VLMEvalKit's post_check."""
    df = pd.read_excel(xlsx_path)

    if not USE_VLMEVAL:
        print('    [error] Cannot compute splits without VLMEvalKit post_check')
        return {}

    # Per-sample correctness using VLMEvalKit's exact logic
    correct = []
    for i in range(len(df)):
        item = df.iloc[i]
        try:
            hit = post_check(item, prefetch=False)
            correct.append(bool(hit))
        except Exception:
            correct.append(False)

    df['correct'] = correct

    total = len(df)
    all_hit = int(df['correct'].sum())
    all_acc = all_hit / total * 100

    # General/Math split by category column
    math_mask = df['category'] == 'math-targeted-vqa'
    gen_mask = df['category'] == 'general-vqa'

    math_n = int(math_mask.sum())
    gen_n = int(gen_mask.sum())
    math_hit = int(df.loc[math_mask, 'correct'].sum())
    gen_hit = int(df.loc[gen_mask, 'correct'].sum())
    math_acc = math_hit / math_n * 100 if math_n > 0 else float('nan')
    gen_acc = gen_hit / gen_n * 100 if gen_n > 0 else float('nan')

    return {
        'All': round(all_acc, 1),
        'General': round(gen_acc, 1),
        'Math': round(math_acc, 1),
        'n_total': total, 'n_general': gen_n, 'n_math': math_n,
        'hit_total': all_hit, 'hit_general': gen_hit, 'hit_math': math_hit,
    }


def compute_mmstar_splits(xlsx_path):
    """Compute MMStar All/Math splits."""
    df = pd.read_excel(xlsx_path)

    df['correct'] = df.apply(
        lambda r: str(r.get('res', '')).strip().upper() == str(r['answer']).strip().upper(),
        axis=1
    )

    all_acc = df['correct'].mean() * 100

    # Find math subset
    math_acc = float('nan')
    for col in ['category', 'l2-category', 'l2_category', 'topic']:
        if col in df.columns:
            vals = df[col].astype(str).unique()
            math_vals = [v for v in vals if 'math' in v.lower()]
            if math_vals:
                math_mask = df[col].astype(str).isin(math_vals)
                if math_mask.sum() > 0:
                    math_acc = df.loc[math_mask, 'correct'].mean() * 100
                    break

    return {
        'All': round(all_acc, 1),
        'Math': round(math_acc, 1) if not np.isnan(math_acc) else 'N/A',
    }


def get_score_from_csv(csv_path):
    """Extract accuracy from VLMEvalKit score CSV."""
    try:
        score_df = pd.read_csv(csv_path)
        for _, row in score_df.iterrows():
            task = str(row.iloc[0]).strip().lower()
            if task == 'overall':
                return round(float(row['acc']), 1)
            if task == 'accuracy':
                for col in score_df.columns[1:]:
                    try:
                        val = float(row[col])
                        if pd.notna(val) and 0 <= val <= 100:
                            return round(val, 2)
                    except (ValueError, TypeError):
                        continue
        if 'acc' in score_df.columns:
            return round(float(score_df['acc'].iloc[0]), 2)
    except Exception as e:
        print(f'  [warn] Could not parse {csv_path}: {e}')
    return None


def process_eval_dir(eval_dir, label=''):
    """Process one eval directory, return BRV-style results dict."""
    print(f'\n  Processing: {label} ({eval_dir})')

    xlsx_files = find_files(eval_dir, '*_gpt-*.xlsx')
    csv_files = find_files(eval_dir, '*_score.csv')
    results = {}

    # ── MathVista: All, General, Math ──
    if 'MathVista_MINI' in xlsx_files:
        splits = compute_mathvista_splits(xlsx_files['MathVista_MINI'])
        if splits:
            results['MathVista_All'] = splits['All']
            results['MathVista_General'] = splits['General']
            results['MathVista_Math'] = splits['Math']
            if 'MathVista_MINI' in csv_files:
                results['MathVista_All_vlmeval'] = get_score_from_csv(csv_files['MathVista_MINI'])
            print(f'    MathVista: All={splits["All"]} (vlmeval={results.get("MathVista_All_vlmeval","?")})'
                  f'  General={splits["General"]}  Math={splits["Math"]}')
            print(f'      hits: {splits["hit_total"]}/{splits["n_total"]}'
                  f'  gen={splits["hit_general"]}/{splits["n_general"]}'
                  f'  math={splits["hit_math"]}/{splits["n_math"]}')

    # ── MathVerse: Overall + 5 subtasks ──
    mve = {}
    for subtask, short in [
        ('MathVerse_MINI_Text_Dominant', 'T-D'),
        ('MathVerse_MINI_Text_Lite', 'T-L'),
        ('MathVerse_MINI_Vision_Intensive', 'V-I'),
        ('MathVerse_MINI_Vision_Dominant', 'V-D'),
        ('MathVerse_MINI_Vision_Only', 'V-O'),
    ]:
        if subtask in csv_files:
            acc = get_score_from_csv(csv_files[subtask])
            if acc is not None:
                mve[short] = acc
                results[f'MathVerse_{short}'] = acc

    if mve:
        overall = round(sum(mve.values()) / len(mve), 1)
        results['MathVerse_Overall'] = overall
        parts = '  '.join(f'{k}={v}' for k, v in mve.items())
        print(f'    MathVerse: Overall={overall}  {parts}')

    # ── MMStar: All, Math ──
    if 'MMStar' in xlsx_files:
        splits = compute_mmstar_splits(xlsx_files['MMStar'])
        results['MMStar_All'] = splits['All']
        results['MMStar_Math'] = splits['Math']
        print(f'    MMStar: All={splits["All"]}  Math={splits["Math"]}')
    elif 'MMStar' in csv_files:
        acc = get_score_from_csv(csv_files['MMStar'])
        if acc is not None:
            results['MMStar_All'] = acc
            print(f'    MMStar: All={acc}')

    # ── Single-score benchmarks ──
    for bench, key in [('DynaMath', 'DynaMath'), ('MathVision_MINI', 'MathVision'), ('MM-Math', 'MM-Math')]:
        if bench in csv_files:
            acc = get_score_from_csv(csv_files[bench])
            if acc is not None:
                results[key] = acc
                print(f'    {key}: {acc}')

    return results


def print_brv_table(all_results, model_name):
    """Print results in BRV Table 2 format with deltas."""
    cols = [
        'MathVista_All', 'MathVista_General', 'MathVista_Math',
        'MathVerse_Overall', 'MathVerse_T-D', 'MathVerse_T-L',
        'MathVerse_V-I', 'MathVerse_V-D', 'MathVerse_V-O',
        'MMStar_All', 'MMStar_Math',
        'DynaMath', 'MathVision',
    ]
    labels = [
        'MV-All', 'MV-Gen', 'MV-Math',
        'MVe-All', 'MVe-TD', 'MVe-TL', 'MVe-VI', 'MVe-VD', 'MVe-VO',
        'MMS-All', 'MMS-Math',
        'DM', 'MVis',
    ]

    print(f'\n{"="*150}')
    print(f'  {model_name} — BRV Table 2 Format')
    print(f'{"="*150}')

    header = f'{"Method":<30s}'
    for l in labels:
        header += f'{l:>10s}'
    print(header)
    print('-' * len(header))

    result_keys = list(all_results.keys())
    for key in result_keys:
        res = all_results[key]
        row = f'{key:<30s}'
        for c in cols:
            val = res.get(c, '—')
            if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                row += f'{val:>10.1f}'
            else:
                row += f'{"—":>10s}'
        print(row)

    # Deltas
    if len(result_keys) >= 2:
        bl = all_results[result_keys[0]]
        print('-' * len(header))
        for key in result_keys[1:]:
            uni = all_results[key]
            row = f'{"Δ " + key:<30s}'
            for c in cols:
                v1 = bl.get(c)
                v2 = uni.get(c)
                try:
                    d = float(v2) - float(v1)
                    row += f'{d:>+10.1f}'
                except (TypeError, ValueError):
                    row += f'{"—":>10s}'
            print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description='Compute BRV-style splits from VLMEvalKit output')
    parser.add_argument('--eval-dir', type=str, action='append', default=[], help='Eval directory (can specify multiple)')
    parser.add_argument('--results-root', type=str, help='Root dir with eval_baseline/ and eval_uniform_*/')
    parser.add_argument('--model-name', type=str, default='', help='Model name for display')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    all_results = {}

    if args.eval_dir:
        for d in args.eval_dir:
            if os.path.isdir(d):
                label = os.path.basename(d)
                results = process_eval_dir(d, label)
                if results:
                    all_results[label] = results

    elif args.results_root:
        eval_dirs = sorted(glob.glob(os.path.join(args.results_root, 'eval_*')))
        for d in eval_dirs:
            if os.path.isdir(d):
                label = os.path.basename(d)
                results = process_eval_dir(d, label)
                if results:
                    all_results[label] = results
    else:
        parser.error('Provide --eval-dir or --results-root')

    model = args.model_name or os.path.basename(args.results_root or (args.eval_dir[0] if args.eval_dir else '') or 'model')
    print_brv_table(all_results, model)

    out_path = args.output
    if not out_path and args.results_root:
        out_path = os.path.join(args.results_root, 'brv_splits_summary.json')
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  Saved: {out_path}')


if __name__ == '__main__':
    main()