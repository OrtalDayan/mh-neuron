#!/usr/bin/env python3
"""
score_mathverse_local.py — Score MathVerse raw predictions locally without GPT API.

Extracts answer letters from model predictions and compares to ground truth.
Also scores POPE predictions. Produces a summary table.

Usage:
    python score_mathverse_local.py --base_dir results/mathverse_vlmekit_ablation/full/llava-next-llama3-8b
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd
import numpy as np


def extract_letter(pred):
    """Extract answer letter (A-D) from model prediction string."""
    s = str(pred).strip()
    if not s or s == 'nan':
        return None

    # Pattern: "The correct option letter is C."
    m = re.search(r'correct.*?(?:option|answer).*?letter.*?([A-D])', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern: "The answer is C" or "Answer: C"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-D])\b', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Starts with letter optionally followed by colon/period/space + content
    # e.g. "B", "B: 54°", "B. 25°", "A: 40°"
    m = re.match(r'^([A-D])(?:\s*[:.)\s]|$)', s)
    if m:
        return m.group(1).upper()

    # Pattern: "option C" or "choice B"
    m = re.search(r'(?:option|choice)\s+([A-D])\b', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Single letter on its own
    m = re.match(r'^([A-D])$', s.strip())
    if m:
        return m.group(1).upper()

    # Last resort: find any standalone A-D
    m = re.search(r'\b([A-D])\b', s)
    if m:
        return m.group(1).upper()

    return None


def extract_gt_letter(answer):
    """Extract ground truth letter from answer field."""
    s = str(answer).strip()
    if not s or s == 'nan':
        return None
    # Usually just a letter, or "A" etc.
    m = re.match(r'^([A-D])', s.upper())
    if m:
        return m.group(1)
    return None


def score_mathverse(xlsx_path):
    """Score a MathVerse raw prediction xlsx. Returns (accuracy, n_total, n_parsed)."""
    df = pd.read_excel(xlsx_path)
    if 'prediction' not in df.columns or 'answer' not in df.columns:
        return None, 0, 0

    pred_letters = df['prediction'].apply(extract_letter)
    gt_letters = df['answer'].apply(extract_gt_letter)

    valid = pred_letters.notna() & gt_letters.notna()
    n_total = len(df)
    n_parsed = valid.sum()

    if n_parsed == 0:
        return 0.0, n_total, 0

    acc = (pred_letters[valid] == gt_letters[valid]).mean() * 100
    return acc, n_total, int(n_parsed)


def score_pope(xlsx_path):
    """Score a POPE prediction xlsx. Returns (accuracy, n_total)."""
    df = pd.read_excel(xlsx_path)
    if 'prediction' not in df.columns or 'answer' not in df.columns:
        return None, 0

    pred_yes = df['prediction'].astype(str).str.lower().str.startswith('yes')
    gt_yes = df['answer'].astype(str).str.lower().str.strip() == 'yes'
    acc = (pred_yes == gt_yes).mean() * 100
    return acc, len(df)


def find_raw_xlsx(rdir, bench_name):
    """Find raw prediction xlsx for a benchmark (search all subdirs including bak)."""
    # First try non-backup locations
    pattern = os.path.join(rdir, '**', f'*{bench_name}.xlsx')
    files = [f for f in glob.glob(pattern, recursive=True)
             if 'score' not in os.path.basename(f)
             and 'extract' not in os.path.basename(f)
             and 'auxmatch' not in os.path.basename(f)]
    if files:
        return files[0]
    return None


def find_score_xlsx(rdir, bench_name):
    """Find GPT-scored xlsx if it exists."""
    pattern = os.path.join(rdir, '**', f'*{bench_name}_gpt-4o-mini_score.xlsx')
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True,
                        help='Base dir, e.g. results/mathverse_vlmekit_ablation/full/llava-next-llama3-8b')
    parser.add_argument('--output', default=None,
                        help='Save results to CSV')
    args = parser.parse_args()

    targets = ['baseline', 'visual', 'text', 'multimodal',
               'random_visual_count', 'random_text_count', 'random_multimodal_count']
    short_names = {
        'baseline': 'base', 'visual': 'vis', 'text': 'tex', 'multimodal': 'mul',
        'random_visual_count': 'rVis', 'random_text_count': 'rTex',
        'random_multimodal_count': 'rMul'
    }

    benchmarks = {
        'TD': 'MathVerse_MINI_Text_Dominant',
        'VO': 'MathVerse_MINI_Vision_Only',
        'VD': 'MathVerse_MINI_Vision_Dominant',
        'POPE': 'POPE'
    }

    # Collect results
    rows = []

    print(f"\n{'':>6} {'TD':>8} {'VO':>8} {'VD':>8} {'POPE':>8}   {'TD_src':>8} {'VO_src':>8} {'VD_src':>8}")
    print('-' * 72)

    for t in targets:
        rdir = os.path.join(args.base_dir, t, 'vlmevalkit_results')
        row_data = {'target': short_names[t]}
        row_str = f'{short_names[t]:>6}'
        src_str = '  '

        for bshort, bname in benchmarks.items():
            val = '—'
            source = ''

            if not os.path.isdir(rdir):
                row_data[bshort] = None
                row_str += f'{val:>8}'
                src_str += f'{"":>8}'
                continue

            try:
                if bname == 'POPE':
                    pope_files = [f for f in glob.glob(os.path.join(rdir, '**', '*_POPE.xlsx'), recursive=True)
                                  if 'auxmatch' not in f]
                    if pope_files:
                        acc, n = score_pope(pope_files[0])
                        val = f'{acc:.1f}'
                        source = 'pope'
                        row_data[bshort] = acc
                    else:
                        row_data[bshort] = None
                else:
                    # Try GPT-scored first
                    score_file = find_score_xlsx(rdir, bname)
                    if score_file:
                        df = pd.read_excel(score_file)
                        acc = df['score'].apply(lambda x: x == True or x == 'True').mean() * 100
                        val = f'{acc:.1f}'
                        source = 'gpt'
                        row_data[bshort] = acc
                    else:
                        # Fall back to local scoring
                        raw_file = find_raw_xlsx(rdir, bname)
                        if raw_file:
                            acc, n_total, n_parsed = score_mathverse(raw_file)
                            if acc is not None:
                                val = f'{acc:.1f}'
                                source = f'loc({n_parsed}/{n_total})'
                                row_data[bshort] = acc
                            else:
                                val = 'no_col'
                                row_data[bshort] = None
                        else:
                            row_data[bshort] = None
            except Exception as e:
                val = 'err'
                source = str(e)[:10]
                row_data[bshort] = None

            row_str += f'{val:>8}'
            src_str += f'{source:>8}'

        rows.append(row_data)
        print(row_str + src_str)

    # Print delta-vs-random table
    print(f"\n\n{'='*60}")
    print("DELTA vs RANDOM (category - matched random)")
    print(f"{'='*60}")
    print(f"{'':>6} {'TD':>8} {'VO':>8} {'VD':>8} {'POPE':>8}")
    print('-' * 40)

    pairs = [
        ('vis', 'rVis', 'visual'),
        ('tex', 'rTex', 'text'),
        ('mul', 'rMul', 'multimodal'),
    ]

    row_map = {r['target']: r for r in rows}
    for cat_short, rand_short, cat_name in pairs:
        cat_row = row_map.get(cat_short, {})
        rand_row = row_map.get(rand_short, {})
        row_str = f'{cat_name[:6]:>6}'
        for bshort in ['TD', 'VO', 'VD', 'POPE']:
            c = cat_row.get(bshort)
            r = rand_row.get(bshort)
            if c is not None and r is not None:
                delta = c - r
                row_str += f'{delta:>+8.1f}'
            else:
                row_str += f'{"—":>8}'
        print(row_str)

    # Save CSV
    if args.output:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()