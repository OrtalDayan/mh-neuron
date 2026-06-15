"""
summarize_eval_results.py — Step 20: Summarize VLMEvalKit benchmark results

Scans step 19 evaluation output directories, extracts the primary metric
from each benchmark's score CSV, and produces a single comparison CSV:

    benchmark, baseline, composed, delta

Also extracts MathVista subcategory scores (math-related vs perception)
to enable direct comparison with BRV (Chen et al., 2025) Table 2.

Usage:
    python summarize_eval_results.py \
        --eval_dir results/19-evaluate/full/qwen2.5-vl-7b \
        --output_dir results/20-summary/full/qwen2.5-vl-7b \
        --model_name qwen2.5-vl-7b
"""

import argparse
import csv
import glob
import os

# ── Benchmark → primary metric extraction ────────────────────────
# Each benchmark's score CSV has a different format.
# We define how to extract the single headline number from each.

BENCHMARK_METRICS = {
    'MathVista_MINI':              {'metric': 'acc',     'label': 'MathVista'},
    'MathVerse_MINI_Vision_Only':  {'metric': 'Overall', 'label': 'MathVerse (Vision Only)'},
    'MathVision':                  {'metric': 'acc',     'label': 'MathVision'},
    'DynaMath':                    {'metric': 'acc',     'label': 'DynaMath'},
    'DynaMath_WorstCase':          {'metric': 'WorstCase', 'label': 'DynaMath (Worst Case)'},
    'MMStar':                      {'metric': 'acc',     'label': 'MMStar'},
    'POPE':                        {'metric': 'acc',     'label': 'POPE'},
    'HallusionBench':              {'metric': 'aAcc',    'label': 'HallusionBench'},
}

# ── MathVista subcategories for BRV comparison ───────────────────
# BRV paper reports "Math VQA" vs "General VQA" splits.
# VLMEvalKit reports per-skill scores; we group them to match BRV.
#
# Math-related skills (reasoning-heavy, where merging should help):
MATHVISTA_MATH_SKILLS = [
    'geometry reasoning',
    'geometry problem solving',
    'algebraic reasoning',
    'arithmetic reasoning',
    'math word problem',
    'logical reasoning',
]

# Perception-heavy skills (where merging may slightly hurt):
MATHVISTA_PERCEPTION_SKILLS = [
    'visual question answering',
    'figure question answering',
    'scientific reasoning',
    'textbook question answering',
    'numeric commonsense',
    'statistical reasoning',
]


def extract_score(score_csv_path, benchmark_name):
    """Extract the primary metric from a VLMEvalKit score CSV.

    Args:
        score_csv_path: path to the *_score.csv file
        benchmark_name: which benchmark this is (key in BENCHMARK_METRICS)

    Returns:
        float or None — the extracted score, or None if parsing failed
    """
    rows = _read_csv(score_csv_path)
    if rows is None:
        return None

    metric_info = BENCHMARK_METRICS.get(benchmark_name, {})
    target_metric = metric_info.get('metric', 'acc')

    # Strategy 0: DynaMath worst-case — find the row where the first column ('Setting') == 'Worst Case'
    if target_metric == 'WorstCase':
        for row in rows:
            first_val = list(row.values())[0] if row else ''
            if first_val.strip().lower() == 'worst case':
                for col in ['Overall', 'acc', 'Acc']:
                    if col in row:
                        try:
                            return float(row[col])
                        except (ValueError, TypeError):
                            continue
        return None

    # Strategy 1: Look for 'Overall' row with target metric column
    for row in rows:
        first_val = list(row.values())[0] if row else ''
        if first_val.strip().lower() == 'overall':
            if target_metric in row:
                try:
                    return float(row[target_metric])
                except (ValueError, TypeError):
                    pass

    # Strategy 2: MathVerse — look for 'Vision Only' row, 'Overall' column
    if 'MathVerse' in benchmark_name:
        for row in rows:
            first_val = list(row.values())[0] if row else ''
            if 'vision only' in first_val.strip().lower():
                if 'Overall' in row:
                    try:
                        return float(row['Overall'])
                    except (ValueError, TypeError):
                        pass

    # Strategy 3: First row, look for common metric columns
    row = rows[0]
    for col_name in [target_metric, 'acc', 'Overall', 'Acc', 'aAcc', 'accuracy']:
        if col_name in row:
            try:
                return float(row[col_name])
            except (ValueError, TypeError):
                continue

    # Strategy 4: Last numeric column of first row
    for val in reversed(list(row.values())):
        try:
            return float(val)
        except (ValueError, TypeError):
            continue

    return None


def extract_mathvista_subcategories(score_csv_path):
    """Extract MathVista per-skill and grouped scores.

    Reads the score CSV and returns a dict with:
      - each individual skill → (acc, n_total)
      - 'Math-related' → weighted average of math skills
      - 'Perception' → weighted average of perception skills

    Args:
        score_csv_path: path to the MathVista *_score.csv file

    Returns:
        dict {skill_name: (acc, n_total)} or empty dict on failure
    """
    rows = _read_csv(score_csv_path)
    if rows is None:
        return {}

    # First column name varies ("Task&Skill", etc.)
    first_col = list(rows[0].keys())[0]

    # Parse each row into {skill_lower: (acc, tot)}
    skill_scores = {}
    for row in rows:
        skill = row[first_col].strip().lower()
        try:
            acc = float(row.get('acc', 0))
            tot = int(float(row.get('tot', 0)))
        except (ValueError, TypeError):
            continue
        skill_scores[skill] = (acc, tot)

    # Build result with original-case skill names
    result = {}
    for row in rows:
        skill = row[first_col].strip()
        if skill.lower() == 'overall':
            continue
        try:
            acc = float(row.get('acc', 0))
            tot = int(float(row.get('tot', 0)))
            result[skill.lower()] = (acc, tot)
        except (ValueError, TypeError):
            continue

    # Math-related group (weighted average)
    math_correct = 0
    math_total = 0
    for skill in MATHVISTA_MATH_SKILLS:
        if skill in skill_scores:
            acc, tot = skill_scores[skill]
            math_correct += acc / 100.0 * tot
            math_total += tot
    if math_total > 0:
        result['Math-related'] = (math_correct / math_total * 100.0, math_total)

    # Perception group (weighted average)
    perc_correct = 0
    perc_total = 0
    for skill in MATHVISTA_PERCEPTION_SKILLS:
        if skill in skill_scores:
            acc, tot = skill_scores[skill]
            perc_correct += acc / 100.0 * tot
            perc_total += tot
    if perc_total > 0:
        result['Perception'] = (perc_correct / perc_total * 100.0, perc_total)

    return result


def _read_csv(path):
    """Read a CSV file and return list of dicts, or None on error."""
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows if rows else None
    except Exception as e:
        print(f'  WARNING: Could not read {path}: {e}')
        return None


def find_score_files(model_dir):
    """Find all *_score.csv files in a model's result directory.

    VLMEvalKit nests results deeply:
      model_dir/<model_name>/T<timestamp>/*_score.csv
      model_dir/<model_name>/T<timestamp>/bak_*/*_score.csv

    We search recursively and pick the most recently modified file
    per benchmark.

    Args:
        model_dir: e.g. results/19-evaluate/full/qwen2.5-vl-7b/baseline/

    Returns:
        dict {benchmark_name: score_csv_path}
    """
    results = {}

    # Recursive search for all score CSVs
    score_files = glob.glob(os.path.join(model_dir, '**', '*_score.csv'),
                            recursive=True)

    for fpath in score_files:
        if not os.path.isfile(fpath):
            continue
        fname = os.path.basename(fpath)
        for bench in BENCHMARK_METRICS:
            if bench in fname:
                # Keep the most recently modified file per benchmark
                try:
                    if bench not in results or \
                            os.path.getmtime(fpath) > os.path.getmtime(results[bench]):
                        results[bench] = fpath
                except OSError:
                    pass
                break

    # DynaMath worst-case lives in the same score CSV as the Average row
    if 'DynaMath' in results:
        results['DynaMath_WorstCase'] = results['DynaMath']

    return results


def main():
    p = argparse.ArgumentParser(
        description='Step 20: Summarize VLMEvalKit benchmark results')
    p.add_argument('--eval_dir', required=True,
                   help='Step 19 eval output dir '
                        '(e.g. results/19-evaluate/full/qwen2.5-vl-7b)')
    p.add_argument('--output_dir', required=True,
                   help='Output directory for summary CSV')
    p.add_argument('--model_name', default='',
                   help='Model name for display in summary')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover model tags (subdirectories in eval_dir) ──────────
    model_tags = {}
    for d in sorted(os.listdir(args.eval_dir)):
        full = os.path.join(args.eval_dir, d)
        if os.path.isdir(full):
            model_tags[d] = full

    if not model_tags:
        print(f'  ERROR: No model directories found in {args.eval_dir}')
        return

    print(f'\nStep 20: Summarizing evaluation results')
    print(f'  Eval dir: {args.eval_dir}')
    print(f'  Models found: {list(model_tags.keys())}')

    # ── Extract scores per model × benchmark ──────────────────────
    all_scores = {}        # {model_tag: {benchmark: score}}
    all_mathvista = {}     # {model_tag: {skill: (acc, tot)}}
    for tag, path in model_tags.items():
        score_files = find_score_files(path)
        all_scores[tag] = {}
        for bench, fpath in score_files.items():
            score = extract_score(fpath, bench)
            all_scores[tag][bench] = score
            status = f'{score:.2f}' if score is not None else 'MISSING'
            print(f'  {tag} / {bench}: {status}')

            # Extract MathVista subcategories
            if bench == 'MathVista_MINI':
                all_mathvista[tag] = extract_mathvista_subcategories(fpath)

    # ── Identify baseline vs composed tags ────────────────────────
    baseline_tag = None
    composed_tags = []
    for tag in model_tags:
        if 'baseline' in tag:
            baseline_tag = tag
        else:
            composed_tags.append(tag)

    if baseline_tag is None:
        print('  WARNING: No baseline found, using first model as reference')
        baseline_tag = list(model_tags.keys())[0]
        composed_tags = list(model_tags.keys())[1:]

    # ── Helper: format a delta value ──────────────────────────────
    def fmt_delta(b, c):
        if b is not None and c is not None:
            d = c - b
            return f'{"+" if d >= 0 else ""}{d:.2f}'
        return '—'

    def fmt_score(s):
        return f'{s:.2f}' if s is not None else '—'

    # ── Helper: human-readable label for model tag ─────────────────
    def tag_label(tag):
        """Convert model tag to a short readable column header."""
        if 'baseline' in tag:
            return 'Baseline'
        elif '17_uniform' in tag:
            return f'Vis-transplant uniform ({tag})'
        elif '17_text_only' in tag:
            return f'Vis-transplant text_only ({tag})'
        elif 'vis_multi' in tag:
            return f'Vis-transplant vis+multi ({tag})'
        elif 'uniform' in tag:
            return f'Uniform/BRV ({tag})'
        elif 'text_multi' in tag:
            return f'PMBT text+multi ({tag})'
        elif 'visual_only' in tag:
            return f'PMBT visual_only ({tag})'
        elif 'step16' in tag:
            return f'PMBT text_inject ({tag})'
        elif 'step17' in tag:
            return f'Vis-transplant visual ({tag})'
        elif 'step18' in tag:
            return f'PMBT composed ({tag})'
        return tag

    # ── Write main summary CSV ────────────────────────────────────
    benchmarks = list(BENCHMARK_METRICS.keys())
    csv_path = os.path.join(args.output_dir, 'eval_summary.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Benchmark']
        header.append(tag_label(baseline_tag))
        for ct in composed_tags:
            header.append(tag_label(ct))
            header.append('Delta')
        writer.writerow(header)

        # Benchmark rows
        for bench in benchmarks:
            label = BENCHMARK_METRICS[bench]['label']
            row = [label]
            b_score = all_scores.get(baseline_tag, {}).get(bench)
            row.append(fmt_score(b_score))
            for ct in composed_tags:
                c_score = all_scores.get(ct, {}).get(bench)
                row.append(fmt_score(c_score))
                row.append(fmt_delta(b_score, c_score))
            writer.writerow(row)

    print(f'\n  Summary saved → {csv_path}')

    # ── Write MathVista subcategory CSV ───────────────────────────
    if all_mathvista:
        sub_csv_path = os.path.join(args.output_dir,
                                    'mathvista_subcategories.csv')

        # Ordered rows: grouped aggregates first, then individual skills
        sub_rows_order = [
            'general-vqa',         # BRV Table 2 "General" column (n=460); requires patched MathVista_acc
            'math-targeted-vqa',   # BRV Table 2 "Math" column (n=540); requires patched MathVista_acc
            '---',
            'Math-related',
            'Perception',
            '---',  # separator
            'geometry reasoning',
            'geometry problem solving',
            'algebraic reasoning',
            'arithmetic reasoning',
            'math word problem',
            'logical reasoning',
            '---',
            'visual question answering',
            'figure question answering',
            'scientific reasoning',
            'textbook question answering',
            'numeric commonsense',
            'statistical reasoning',
        ]

        with open(sub_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['MathVista Subcategory', 'N']
            header.append(tag_label(baseline_tag))
            for ct in composed_tags:
                header.append(tag_label(ct))
                header.append('Delta')
            writer.writerow(header)

            for skill in sub_rows_order:
                if skill == '---':
                    writer.writerow([])
                    continue

                # Get N from baseline (same dataset for all models)
                b_data = all_mathvista.get(baseline_tag, {})
                b_acc, b_tot = b_data.get(skill, (None, 0))

                row = [skill, str(b_tot)]
                row.append(fmt_score(b_acc))

                for ct in composed_tags:
                    c_data = all_mathvista.get(ct, {})
                    c_acc, _ = c_data.get(skill, (None, 0))
                    row.append(fmt_score(c_acc))
                    row.append(fmt_delta(b_acc, c_acc))

                writer.writerow(row)

        print(f'  MathVista subcategories saved → {sub_csv_path}')

    # ── Print main summary to stdout ──────────────────────────────
    print(f'\n{"="*80}')
    print(f'  EVALUATION SUMMARY: {args.model_name}')
    print(f'{"="*80}')

    _print_csv(csv_path)

    # ── Print MathVista subcategories to stdout ───────────────────
    if all_mathvista:
        print(f'\n{"="*80}')
        print(f'  MATHVISTA SUBCATEGORIES (for BRV comparison)')
        print(f'{"="*80}')

        _print_csv(sub_csv_path)

    print(f'{"="*80}\n')


def _print_csv(path):
    """Pretty-print a CSV file as an aligned table."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        rows = [r for r in reader]

    if not rows:
        return

    # Filter out empty separator rows for width calculation
    data_rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not data_rows:
        return

    ncols = max(len(r) for r in data_rows)
    # Pad short rows
    for r in rows:
        while len(r) < ncols:
            r.append('')

    widths = [max(len(r[i]) for r in data_rows if i < len(r))
              for i in range(ncols)]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths)

    for i, row in enumerate(rows):
        if not any(cell.strip() for cell in row):
            print()  # empty separator
            continue
        print(fmt.format(*row))
        if i == 0:
            print('─' * (sum(widths) + 2 * (ncols - 1)))


if __name__ == '__main__':
    main()