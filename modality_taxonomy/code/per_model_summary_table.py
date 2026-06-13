#!/usr/bin/env python3
"""Per-model summary table: dissociation gap with bootstrap CIs.

For the paper's Section 4.3 (Modality Taxonomy Ablation): we compute, for each
model, the *dissociation gap* — the differential damage from ablating
text-position vs. visual-position neurons on a text benchmark vs. a visual
benchmark. The gap is positive iff the taxonomy correctly predicts where the
damage will land.

Terminology: filenames use 'text', 'visual', 'multimodal' for the category
field. These correspond to the paper's positional categories (text-position,
visual-position, multimodal-position). Outputs use the positional terminology.

Definition of dissociation gap at fraction f:
    drop(cat, bench) = baseline_acc(bench) - ablated_acc(cat, bench, f)
                       (baseline_acc taken from the smallest available fraction
                        in the ranked or random data)

    text_specificity = drop(text, text_bench) - drop(text, visual_bench)
    visual_specificity = drop(visual, visual_bench) - drop(visual, text_bench)
    gap = (text_specificity + visual_specificity) / 2

Higher gap = stronger double dissociation evidence.

We bootstrap CIs by resampling the random trials per cell (n=5 at each frac).
The 'ranked' cell is fixed (it's the canonical confidence-ranked top-K), so we
compute the CI for the *random baseline* and propagate it through the gap
calculation. This gives a confidence interval on how much MORE damage the
ranked ablation does than what we'd expect by chance.

Usage:
    python3 code/per_model_summary_table.py \\
        --models llava-next-llama3-8b idefics2-8b llava-onevision-7b \\
                 qwen2-vl-7b llava-1.5-7b internvl2.5-8b qwen2.5-vl-3b \\
        --ranking norm \\
        --fractions 0.01 0.05 0.10 \\
        --output table_dissociation_summary.md

Outputs both a markdown table and a LaTeX booktabs table.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


CELL_RE = re.compile(
    r'run1_(?P<ranking>D|norm|D_x_norm)_'
    r'gate_up_(?P<cat>text|visual|multimodal)_'
    r'(?P<bench>[A-Za-z_]+?)_'
    r'f(?P<frac>[\d.]+)_'
    r'(?P<trial>ranked|r\d)\.json$'
)

MODEL_LABELS = {
    'llava-next-llama3-8b': 'LLaVA-NeXT-LLaMA3-8B',
    'idefics2-8b': 'Idefics2-8B',
    'llava-onevision-7b': 'LLaVA-OneVision-7B',
    'qwen2-vl-7b': 'Qwen2-VL-7B',
    'llava-1.5-7b': 'LLaVA-1.5-7B',
    'internvl2.5-8b': 'InternVL2.5-8B',
    'qwen2.5-vl-3b': 'Qwen2.5-VL-3B',
}


def load_results(model_dir):
    """Return list of records: (ranking, cat, bench, frac, trial, acc)."""
    out = []
    for fpath in glob.glob(os.path.join(model_dir, 'run1_*.json')):
        if os.path.getsize(fpath) < 200:
            continue
        m = CELL_RE.search(os.path.basename(fpath))
        if not m:
            continue
        try:
            data = json.load(open(fpath))
        except (json.JSONDecodeError, OSError):
            continue
        acc = data.get('accuracy_pct')
        if acc is None:
            continue
        out.append({
            'ranking': m['ranking'],
            'cat': m['cat'],
            'bench': m['bench'],
            'frac': float(m['frac']),
            'trial': m['trial'],
            'acc': float(acc),
        })
    return out


def baseline_acc(records, ranking, bench):
    """Baseline = ranked acc at the smallest available fraction.

    Rationale: at very small fractions ablation should be near-zero impact,
    approximating unablated performance.
    """
    candidates = [
        r for r in records
        if r['ranking'] == ranking and r['bench'] == bench and r['trial'] == 'ranked'
    ]
    if not candidates:
        # fall back to mean of random at smallest fraction
        candidates = [
            r for r in records
            if r['ranking'] == ranking and r['bench'] == bench
        ]
    if not candidates:
        return None
    smallest = min(r['frac'] for r in candidates)
    smallest_records = [r for r in candidates if r['frac'] == smallest]
    return float(np.mean([r['acc'] for r in smallest_records]))


def get_cell(records, ranking, cat, bench, frac, trial):
    """Get accuracy for one cell, or None if not present."""
    for r in records:
        if (r['ranking'] == ranking and r['cat'] == cat and r['bench'] == bench
                and abs(r['frac'] - frac) < 1e-6 and r['trial'] == trial):
            return r['acc']
    return None


def get_random_accs(records, ranking, cat, bench, frac):
    """Get all random-trial accuracies for one cell."""
    return [
        r['acc'] for r in records
        if r['ranking'] == ranking and r['cat'] == cat and r['bench'] == bench
        and abs(r['frac'] - frac) < 1e-6 and r['trial'] != 'ranked'
    ]


def dissociation_gap(records, ranking, frac, text_bench, visual_bench):
    """Compute the dissociation gap and its bootstrap CI at one fraction.

    Returns (gap, ci_low, ci_high, components_dict).
    """
    base_text = baseline_acc(records, ranking, text_bench)
    base_vis = baseline_acc(records, ranking, visual_bench)
    if base_text is None or base_vis is None:
        return None, None, None, None

    # Ranked drops: how much does ranked ablation drop the bench?
    ranked_text_on_text = get_cell(records, ranking, 'text', text_bench, frac, 'ranked')
    ranked_text_on_vis = get_cell(records, ranking, 'text', visual_bench, frac, 'ranked')
    ranked_vis_on_text = get_cell(records, ranking, 'visual', text_bench, frac, 'ranked')
    ranked_vis_on_vis = get_cell(records, ranking, 'visual', visual_bench, frac, 'ranked')

    if any(x is None for x in [
        ranked_text_on_text, ranked_text_on_vis, ranked_vis_on_text, ranked_vis_on_vis
    ]):
        return None, None, None, None

    # Net drop relative to random baseline drop at the same fraction.
    rand_text_on_text = get_random_accs(records, ranking, 'text', text_bench, frac)
    rand_text_on_vis = get_random_accs(records, ranking, 'text', visual_bench, frac)
    rand_vis_on_text = get_random_accs(records, ranking, 'visual', text_bench, frac)
    rand_vis_on_vis = get_random_accs(records, ranking, 'visual', visual_bench, frac)

    # If we have random data, compute net effect (ranked drop - random drop).
    # Otherwise just use raw drop.
    def net_drop(ranked, base, rand_list):
        ranked_drop = base - ranked
        if rand_list:
            rand_drop = base - np.mean(rand_list)
            return ranked_drop - rand_drop
        return ranked_drop

    # Specificity of text-position neurons: damage on text - damage on visual.
    text_spec = net_drop(ranked_text_on_text, base_text, rand_text_on_text) \
              - net_drop(ranked_text_on_vis, base_vis, rand_text_on_vis)
    # Specificity of visual-position neurons: damage on visual - damage on text.
    vis_spec = net_drop(ranked_vis_on_vis, base_vis, rand_vis_on_vis) \
             - net_drop(ranked_vis_on_text, base_text, rand_vis_on_text)
    gap = (text_spec + vis_spec) / 2

    # Bootstrap CI: resample the random-trial pools 1000 times, recompute gap.
    rng = np.random.default_rng(42)
    bootstrap_gaps = []
    n_boot = 1000
    for _ in range(n_boot):
        def _resampled_net_drop(ranked, base, rand_list):
            ranked_drop = base - ranked
            if rand_list:
                resampled = rng.choice(rand_list, size=len(rand_list), replace=True)
                rand_drop = base - np.mean(resampled)
                return ranked_drop - rand_drop
            return ranked_drop

        text_spec_b = _resampled_net_drop(ranked_text_on_text, base_text, rand_text_on_text) \
                    - _resampled_net_drop(ranked_text_on_vis, base_vis, rand_text_on_vis)
        vis_spec_b = _resampled_net_drop(ranked_vis_on_vis, base_vis, rand_vis_on_vis) \
                   - _resampled_net_drop(ranked_vis_on_text, base_text, rand_vis_on_text)
        bootstrap_gaps.append((text_spec_b + vis_spec_b) / 2)

    ci_low, ci_high = np.percentile(bootstrap_gaps, [2.5, 97.5])

    components = {
        'baseline_text': base_text,
        'baseline_visual': base_vis,
        'text_neurons_on_text': ranked_text_on_text,
        'text_neurons_on_visual': ranked_text_on_vis,
        'visual_neurons_on_text': ranked_vis_on_text,
        'visual_neurons_on_visual': ranked_vis_on_vis,
        'text_specificity': text_spec,
        'visual_specificity': vis_spec,
    }
    return gap, ci_low, ci_high, components


def format_table_rows(args, results):
    """Format results into rows ready for tabulation."""
    rows = []
    for model, model_data in results.items():
        for frac in args.fractions:
            cell = model_data.get(frac)
            if cell is None:
                rows.append({
                    'model': model, 'frac': frac, 'gap': None,
                    'ci_low': None, 'ci_high': None,
                })
                continue
            gap, lo, hi, _ = cell
            rows.append({
                'model': model, 'frac': frac,
                'gap': gap, 'ci_low': lo, 'ci_high': hi,
            })
    return rows


def write_markdown(rows, fractions, path):
    """Write a markdown table grouped by model, columns are fractions."""
    with open(path, 'w') as f:
        f.write('# Dissociation gap by model and ablation fraction\n\n')
        f.write('Each cell shows the dissociation gap (in accuracy points) with bootstrap 95% CI.\n')
        f.write('Higher gap = stronger evidence that the position-based taxonomy identifies\n')
        f.write('functionally separable neurons. Gap > 0 indicates the predicted double dissociation.\n\n')

        header = '| Model | ' + ' | '.join(f'f={x:.2f}' for x in fractions) + ' |'
        sep = '|' + '|'.join(['---'] * (len(fractions) + 1)) + '|'
        f.write(header + '\n')
        f.write(sep + '\n')

        # Group by model
        by_model = defaultdict(dict)
        for r in rows:
            by_model[r['model']][r['frac']] = r

        for model in sorted(by_model.keys()):
            label = MODEL_LABELS.get(model, model)
            cells = []
            for frac in fractions:
                r = by_model[model].get(frac)
                if r is None or r['gap'] is None:
                    cells.append('—')
                else:
                    cells.append(
                        f"{r['gap']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
                    )
            f.write(f'| {label} | ' + ' | '.join(cells) + ' |\n')

        f.write('\n')
        f.write('Format: `gap [CI_low, CI_high]` in accuracy-point units.\n')
        f.write('CI is the 2.5–97.5 percentile of 1000 bootstrap resamples\n')
        f.write('over the 5 random trials at each fraction.\n')


def write_latex(rows, fractions, path):
    """Write a LaTeX booktabs table for direct paper inclusion."""
    with open(path, 'w') as f:
        f.write('% Dissociation gap by model and ablation fraction.\n')
        f.write('% Generated by per_model_summary_table.py.\n')
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Dissociation gap (in accuracy points) by model and ablation fraction. ')
        f.write('Higher gap indicates stronger evidence that text-position neurons preferentially ')
        f.write('damage text benchmarks while visual-position neurons preferentially damage visual ')
        f.write('benchmarks. Square brackets show bootstrap 95\\% CIs over the random-baseline trials.}\n')
        f.write('\\label{tab:dissociation_gap}\n')
        f.write('\\begin{tabular}{l' + 'c' * len(fractions) + '}\n')
        f.write('\\toprule\n')
        cols = 'Model & ' + ' & '.join(f'$f={x:g}$' for x in fractions) + ' \\\\\n'
        f.write(cols)
        f.write('\\midrule\n')

        by_model = defaultdict(dict)
        for r in rows:
            by_model[r['model']][r['frac']] = r

        for model in sorted(by_model.keys()):
            label = MODEL_LABELS.get(model, model)
            cells = []
            for frac in fractions:
                r = by_model[model].get(frac)
                if r is None or r['gap'] is None:
                    cells.append('---')
                else:
                    cells.append(
                        f"${r['gap']:.1f}_{{[{r['ci_low']:.1f}, {r['ci_high']:.1f}]}}$"
                    )
            f.write(f'{label} & ' + ' & '.join(cells) + ' \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True,
                   help='Model directory names (full names, not aliases)')
    p.add_argument('--ranking', default='norm', choices=['D', 'norm', 'D_x_norm'])
    p.add_argument('--fractions', nargs='+', type=float, default=[0.01, 0.05, 0.10],
                   help='Fractions to evaluate at (paper uses small ones)')
    p.add_argument('--text-bench', default='TriviaQA')
    p.add_argument('--visual-bench', default='POPE')
    p.add_argument('--results-root', default='results/24-ranked-ablation/full')
    p.add_argument('--output', default='table_dissociation_summary.md')
    args = p.parse_args()

    print(f'Computing dissociation gap with bootstrap CIs')
    print(f'  Ranking: {args.ranking}')
    print(f'  Text benchmark: {args.text_bench}')
    print(f'  Visual benchmark: {args.visual_bench}')
    print(f'  Fractions: {args.fractions}')
    print()

    results = {}
    for model in args.models:
        model_dir = os.path.join(args.results_root, model, 'run1')
        if not os.path.isdir(model_dir):
            print(f'  ⚠ {model_dir} missing, skipping')
            continue
        records = load_results(model_dir)
        if not records:
            print(f'  ⚠ {model}: no records')
            continue

        results[model] = {}
        for frac in args.fractions:
            cell = dissociation_gap(
                records, args.ranking, frac,
                args.text_bench, args.visual_bench,
            )
            results[model][frac] = cell
            label = MODEL_LABELS.get(model, model)
            if cell[0] is None:
                print(f'  {label} f={frac:.2f}: insufficient data')
            else:
                gap, lo, hi, _ = cell
                print(f'  {label} f={frac:.2f}: gap={gap:+.2f} CI=[{lo:+.2f}, {hi:+.2f}]')

    rows = format_table_rows(args, results)
    write_markdown(rows, args.fractions, args.output)
    latex_path = args.output.replace('.md', '.tex') if args.output.endswith('.md') \
                 else args.output + '.tex'
    write_latex(rows, args.fractions, latex_path)

    print()
    print(f'  ✓ Markdown: {args.output}')
    print(f'  ✓ LaTeX:    {latex_path}')


if __name__ == '__main__':
    main()
