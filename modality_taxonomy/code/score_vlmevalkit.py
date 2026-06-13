#!/usr/bin/env python3
"""
score_vlmevalkit.py — Score VLMEvalKit prediction xlsx files locally.

Reads the prediction files saved by VLMEvalKit --mode infer and computes
accuracy without needing OpenAI API or a local verifier model.

Supports: MMStar (MCQ), MathVista_MINI, MathVerse_MINI, MathVision, DynaMath

Usage:
    python score_vlmevalkit.py \
        --results_dir results/13-weight-merge/full/.../text_inject/
"""

import argparse
import glob
import json
import os
import re
import sys


def score_mcq(df):
    """Score multiple-choice questions (MMStar, etc).
    Compares 'prediction' to 'answer' column."""
    correct = 0
    total = 0
    for _, row in df.iterrows():
        pred = str(row.get('prediction', '')).strip()
        answer = str(row.get('answer', '')).strip()
        if not answer:
            continue
        total += 1
        # Extract letter from prediction
        pred_letter = extract_mcq_letter(pred)
        if pred_letter == answer.upper():
            correct += 1
    return {'accuracy': correct / total if total > 0 else 0,
            'correct': correct, 'total': total}


def extract_mcq_letter(text):
    """Extract MCQ answer letter (A/B/C/D) from model response."""
    text = text.strip()
    # Direct letter answer
    if text.upper() in ['A', 'B', 'C', 'D', 'E']:
        return text.upper()
    # "The answer is B" pattern
    m = re.search(r'answer\s+is\s+([A-E])', text, re.I)
    if m:
        return m.group(1).upper()
    # "(B)" pattern
    m = re.search(r'\(([A-E])\)', text)
    if m:
        return m.group(1).upper()
    # First letter if short
    if len(text) <= 3 and text[0].upper() in 'ABCDE':
        return text[0].upper()
    return text[:1].upper() if text else ''


def score_mathvista(df):
    """Score MathVista using rule-based matching."""
    correct = 0
    total = 0
    for _, row in df.iterrows():
        pred = str(row.get('prediction', '')).strip()
        answer = str(row.get('answer', '')).strip()
        qtype = str(row.get('question_type', '')).lower()
        if not answer:
            continue
        total += 1
        if 'multi' in qtype or 'choice' in qtype:
            if extract_mcq_letter(pred) == answer.upper():
                correct += 1
        else:
            # Free-form: extract number and compare
            if match_freeform(pred, answer):
                correct += 1
    return {'accuracy': correct / total if total > 0 else 0,
            'correct': correct, 'total': total}


def match_freeform(pred, answer):
    """Match free-form numerical/text answers."""
    pred = pred.strip().lower()
    answer = answer.strip().lower()
    # Exact match
    if pred == answer:
        return True
    # Try numeric comparison
    pred_num = extract_number(pred)
    ans_num = extract_number(answer)
    if pred_num is not None and ans_num is not None:
        return abs(pred_num - ans_num) < 1e-3 * max(abs(ans_num), 1)
    return False


def extract_number(text):
    """Extract the last number from text."""
    # Remove commas, dollar signs, percent
    text = text.replace(',', '').replace('$', '').replace('%', '')
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def score_file(xlsx_path, benchmark):
    """Score a single xlsx prediction file."""
    import pandas as pd
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f'  ERROR reading {xlsx_path}: {e}')
        return None

    if 'prediction' not in df.columns or 'answer' not in df.columns:
        print(f'  WARNING: missing prediction/answer columns in {xlsx_path}')
        return None

    correct = 0
    total = 0
    for _, row in df.iterrows():
        pred = str(row.get('prediction', '')).strip()
        answer = str(row.get('answer', '')).strip()
        qtype = str(row.get('question_type', '')).lower()
        atype = str(row.get('answer_type', '')).lower()
        if not answer:
            continue
        total += 1

        if 'multi_choice' in qtype or benchmark == 'MMStar':
            answer_option = str(row.get('answer_option', '')).strip().upper()
            pred_letter = extract_mcq_letter(pred)
            if answer_option and pred_letter == answer_option:
                correct += 1
            elif not answer_option and pred.lower() == answer.lower():
                correct += 1
        elif atype in ('integer', 'float'):
            pred_num = extract_number(pred)
            ans_num = extract_number(answer)
            if pred_num is not None and ans_num is not None:
                if ans_num == 0:
                    if abs(pred_num) < 1e-3:
                        correct += 1
                elif abs(pred_num - ans_num) / max(abs(ans_num), 1e-6) < 0.01:
                    correct += 1
            elif pred.strip() == answer.strip():
                correct += 1
        else:
            if pred.lower().strip() == answer.lower().strip():
                correct += 1

    return {'accuracy': correct / total if total > 0 else 0,
            'correct': correct, 'total': total}


def find_and_score(results_dir):
    """Find all VLMEvalKit xlsx files and score them."""
    all_scores = {}

    for variant_dir in sorted(glob.glob(os.path.join(results_dir, '*/'))):
        variant = os.path.basename(variant_dir.rstrip('/'))
        vlmeval_dir = os.path.join(variant_dir, 'vlmevalkit_results')
        if not os.path.isdir(vlmeval_dir):
            continue

        # Find xlsx files (nested under model_name/T.../model_bench.xlsx)
        xlsx_files = glob.glob(os.path.join(vlmeval_dir, '**', '*.xlsx'),
                               recursive=True)

        variant_scores = {}
        for xlsx in xlsx_files:
            fname = os.path.basename(xlsx)
            # Extract benchmark from filename: model_bench.xlsx
            for bench in ['MMStar', 'MathVista_MINI', 'MathVerse_MINI',
                          'MathVision', 'DynaMath']:
                if bench in fname:
                    score = score_file(xlsx, bench)
                    if score:
                        variant_scores[bench] = score
                        # Save as JSON in variant dir
                        result_path = os.path.join(
                            variant_dir, f'vlmeval_{bench}_results.json')
                        with open(result_path, 'w') as f:
                            json.dump(score, f, indent=2)
                        # Write done flag
                        flag_path = os.path.join(
                            variant_dir, f'vlmeval_{bench}_done.flag')
                        with open(flag_path, 'w') as f:
                            f.write('done\n')
                    break

        if variant_scores:
            all_scores[variant] = variant_scores

    return all_scores


def print_results_table(all_scores):
    """Print a formatted results table."""
    benchmarks = ['MMStar', 'MathVista_MINI', 'MathVerse_MINI',
                  'MathVision', 'DynaMath']
    header = f'{"Variant":<45}'
    for b in benchmarks:
        short = b.replace('_MINI', '').replace('Math', 'M')
        header += f' {short:>10}'
    print(header)
    print('-' * len(header))

    for variant in sorted(all_scores):
        row = f'{variant:<45}'
        for b in benchmarks:
            score = all_scores[variant].get(b, {})
            acc = score.get('accuracy', -1)
            if acc >= 0:
                row += f' {acc*100:>9.1f}%'
            else:
                row += f' {"—":>10}'
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', required=True,
                   help='Path to text_inject/ directory')
    args = p.parse_args()

    all_scores = find_and_score(args.results_dir)

    if all_scores:
        print(f'\n{"="*80}')
        print(f'  VLMEvalKit Scores (rule-based, {len(all_scores)} variants)')
        print(f'{"="*80}\n')
        print_results_table(all_scores)
        print()

        # Save summary
        summary_path = os.path.join(args.results_dir,
                                    'vlmevalkit_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_scores, f, indent=2)
        print(f'Summary saved → {summary_path}')
    else:
        print('No VLMEvalKit results found.')


if __name__ == '__main__':
    main()