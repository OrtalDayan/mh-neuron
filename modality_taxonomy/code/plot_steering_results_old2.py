"""
plot_steering_results.py — Step 13: Generate publication-quality figures
for ECCV from merged step 11 steering results.

Reads steering_merged.json (from step 12) and enrichment results
(from step 10) and produces:

  Figure A — Dose-response curve (alpha sweep)
             x = alpha, y = metric per condition
             Shows monotonic hallucination reduction with visual suppression

  Figure B — Ablation curve (top-N at best alpha)
             x = neurons suppressed, y = metric per modality
             Shows visual neurons give steepest reduction per neuron

  Figure C — Multi-benchmark bar chart (best alpha)
             Grouped bars: benchmark x condition
             Shows visual suppression helps halluc without hurting knowledge

  Figure D — Three-method enrichment comparison
             DH vs CETT-diff vs combined fold-enrichment per modality

  Figure E — Per-layer x modality enrichment heatmap
             Re-styled from step 10 Phase 3

  Table F  — LaTeX-ready results table with deltas vs baseline

Usage:
    python plot_steering_results.py \
        --merged_json results/13-plots/llava-ov-7b/steering_merged.json \
        --enrichment_dir results/10-halluc_scores/full/llava-ov-7b \
        --model_name llava-ov-7b \
        --output_dir results/13-plots/llava-ov-7b
"""

import argparse                                                            # Line 1: parse command-line arguments
import json                                                                # Line 2: read/write JSON
import os                                                                  # Line 3: path manipulation
import numpy as np                                                         # Line 4: numerical operations


# =========================================================================
# ECCV style constants
# =========================================================================

COLORS = {                                                                 # Line 5: per-condition colours
    'ablate_vis':   '#D62728',   # red — visual neurons
    'ablate_text':  '#1F77B4',   # blue — text neurons
    'ablate_multi': '#2CA02C',   # green — multimodal neurons
    'ablate_multimodal': '#2CA02C',
    'ablate_unknown': '#9467BD', # purple — unknown neurons
    'ablate_encoder': '#FF7F0E', # orange — vision encoder
    'ablate_projector': '#8C564B', # brown — projector
    'random':       '#7F7F7F',   # grey — random baseline
    'baseline':     '#000000',   # black — unmodified model
}

LABELS = {                                                                 # Line 6: display labels
    'ablate_vis':   'Visual',
    'ablate_text':  'Text',
    'ablate_multi': 'Multimodal',
    'ablate_multimodal': 'Multimodal',
    'ablate_unknown': 'Unknown',
    'ablate_encoder': 'Encoder',
    'ablate_projector': 'Projector',
    'random':       'Random',
    'baseline':     'Baseline',
}

MARKERS = {                                                                # Line 7: marker shapes
    'ablate_vis':   'o',
    'ablate_text':  's',
    'ablate_multi': '^',
    'ablate_multimodal': '^',
    'ablate_unknown': 'D',
    'ablate_encoder': 'P',
    'ablate_projector': 'H',
    'random':       'x',
    'baseline':     '*',
}

BENCH_CONFIG = {                                                           # Line 8: (display, metric, higher_is_better)
    'pope':            ('POPE',         'accuracy', True),
    'chair':           ('CHAIR$_i$',    'chair_i',  False),
    'triviaqa':        ('TriviaQA',     'accuracy', True),
    'mmlu':            ('MMLU',         'accuracy', True),
    'vqav2':           ('VQAv2',        'accuracy', True),
    'vqa_perception':  ('VQA-Percept.', 'accuracy', True),
    'vqa_knowledge':   ('VQA-Know.',    'accuracy', True),
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate ECCV figures from steering + enrichment results.')
    p.add_argument('--merged_json', required=True,
                   help='Path to steering_merged.json from step 12')
    p.add_argument('--enrichment_dir', default=None,
                   help='Step 10 output dir with enrichment_results*.json '
                        'and per_layer_enrichment.npy')
    p.add_argument('--model_name', default='Model',
                   help='Model name for figure titles')
    p.add_argument('--output_dir', required=True,
                   help='Directory to save figures')
    p.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'],
                   help='Figure format (default: pdf for LaTeX)')
    p.add_argument('--dpi', type=int, default=300,
                   help='DPI for png output')
    p.add_argument('--best_alpha', type=float, default=None,
                   help='Override best alpha. Auto-selects if not set.')
    return p.parse_args()


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })
    return plt


def load_merged(path):
    with open(path) as f:
        return json.load(f)


def _resolve_alpha_key(data, alpha_val):
    """Find the string key in data['results'] closest to alpha_val."""
    alpha_str = str(alpha_val)
    if alpha_str in data['results']:
        return alpha_str
    for k in data['results']:
        if abs(float(k) - alpha_val) < 1e-6:
            return k
    return None


def find_best_alpha(data, condition='ablate_vis', benchmark='pope',
                    metric='accuracy'):
    """Find alpha giving best metric for a condition."""
    higher_is_better = BENCH_CONFIG.get(benchmark, ('', '', True))[2]
    best_alpha = 1.0
    best_val = None

    for alpha_str, conditions_dict in data['results'].items():
        if condition not in conditions_dict:
            continue
        cond_data = conditions_dict[condition]
        if benchmark not in cond_data:
            continue
        val = cond_data[benchmark].get(metric)
        if val is None:
            continue
        if best_val is None:
            best_val = val
            best_alpha = float(alpha_str)
        elif higher_is_better and val > best_val:
            best_val = val
            best_alpha = float(alpha_str)
        elif not higher_is_better and val < best_val:
            best_val = val
            best_alpha = float(alpha_str)

    return best_alpha


# =========================================================================
# Figure A — Dose-response curve
# =========================================================================

def plot_dose_response(data, output_dir, model_name, fmt='pdf', dpi=300):
    """POPE accuracy and CHAIR_i vs alpha for each modality condition."""
    plt = setup_matplotlib()

    benchmarks_to_plot = [b for b in ['pope', 'chair'] if b in data['benchmarks']]
    if not benchmarks_to_plot:
        print('  WARNING: Neither POPE nor CHAIR in data. Skipping Figure A.')
        return

    fig, axes = plt.subplots(1, len(benchmarks_to_plot),
                             figsize=(5.5 * len(benchmarks_to_plot), 4.5))
    if len(benchmarks_to_plot) == 1:
        axes = [axes]

    conditions_order = ['ablate_vis', 'ablate_text', 'ablate_multimodal',
                        'ablate_unknown', 'ablate_encoder', 'ablate_projector',
                        'random', 'baseline']

    for ax_idx, bench_key in enumerate(benchmarks_to_plot):
        ax = axes[ax_idx]
        bench_name, metric_key, _ = BENCH_CONFIG[bench_key]

        for condition in conditions_order:
            alphas_plot = []
            values_plot = []

            for alpha_str, conds in sorted(data['results'].items(),
                                           key=lambda x: float(x[0])):
                if condition not in conds:
                    continue
                if bench_key not in conds[condition]:
                    continue
                val = conds[condition][bench_key].get(metric_key)
                if val is not None:
                    alphas_plot.append(float(alpha_str))
                    values_plot.append(val)

            if alphas_plot:
                ax.plot(alphas_plot, values_plot,
                        color=COLORS.get(condition, '#333'),
                        marker=MARKERS.get(condition, 'o'),
                        label=LABELS.get(condition, condition),
                        markersize=5)

        ax.axvline(x=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'Scaling factor $\alpha$')
        ylabel = bench_name + (' Accuracy' if metric_key == 'accuracy' else '')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{model_name}')
        ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fig_A_dose_response.{fmt}')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'  Figure A (dose-response) -> {out_path}')


# =========================================================================
# Figure B — Ablation curve (top-N at best alpha)
# =========================================================================

def plot_ablation_curve(data, best_alpha, output_dir, model_name,
                        fmt='pdf', dpi=300):
    """POPE accuracy vs number of neurons suppressed at best alpha."""
    plt = setup_matplotlib()

    alpha_str = _resolve_alpha_key(data, best_alpha)
    if alpha_str is None:
        print(f'  WARNING: Alpha {best_alpha} not found. Skipping Figure B.')
        return

    conditions_dict = data['results'][alpha_str]

    # Group by base_condition, collect (top_n, metric)
    curve_data = {}
    for cond_name, cond_data in conditions_dict.items():
        base = cond_data.get('base_condition', cond_name)
        top_n = cond_data.get('top_n')
        if top_n is None or 'pope' not in cond_data:
            continue
        acc = cond_data['pope']['accuracy']
        curve_data.setdefault(base, []).append((top_n, acc))

    if not curve_data:
        print(f'  WARNING: No top_n curve data at alpha={best_alpha}. '
              f'Skipping Figure B. (Was --curve used in step 11?)')
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for condition in ['ablate_vis', 'ablate_text', 'ablate_multimodal',
                      'ablate_unknown', 'random']:
        if condition not in curve_data:
            continue
        points = sorted(curve_data[condition])
        ns = [p[0] for p in points]
        accs = [p[1] for p in points]
        ax.plot(ns, accs, color=COLORS.get(condition, '#333'),
                marker=MARKERS.get(condition, 'o'),
                label=LABELS.get(condition, condition), markersize=5)

    baseline_data = conditions_dict.get('baseline', {})
    if 'pope' in baseline_data:
        ax.axhline(y=baseline_data['pope']['accuracy'],
                   color='black', linestyle='--', linewidth=0.8,
                   alpha=0.5, label='Baseline')

    ax.set_xlabel('Top-N neurons suppressed')
    ax.set_ylabel('POPE Accuracy')
    ax.set_xscale('log')
    ax.set_title(f'{model_name} ($\\alpha$={best_alpha})')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fig_B_ablation_curve.{fmt}')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'  Figure B (ablation curve) -> {out_path}')


# =========================================================================
# Figure C — Multi-benchmark bar chart
# =========================================================================

def plot_multi_benchmark(data, best_alpha, output_dir, model_name,
                         fmt='pdf', dpi=300):
    """Grouped bar chart: benchmark groups x condition bars at best alpha."""
    plt = setup_matplotlib()

    alpha_str = _resolve_alpha_key(data, best_alpha)
    if alpha_str is None:
        print(f'  WARNING: Alpha {best_alpha} not found. Skipping Figure C.')
        return

    conditions_dict = data['results'][alpha_str]
    conditions_to_plot = ['baseline', 'ablate_vis', 'ablate_text',
                          'ablate_multimodal', 'ablate_encoder',
                          'ablate_projector', 'random']

    available_benchmarks = []
    for bench_key in BENCH_CONFIG:
        for cond in conditions_to_plot:
            if cond in conditions_dict and bench_key in conditions_dict[cond]:
                available_benchmarks.append(bench_key)
                break

    if not available_benchmarks:
        print(f'  WARNING: No benchmark data at alpha={best_alpha}. '
              f'Skipping Figure C.')
        return

    n_bench = len(available_benchmarks)
    n_cond = len(conditions_to_plot)
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(max(8, n_bench * 1.8), 5))
    x = np.arange(n_bench)

    for i, condition in enumerate(conditions_to_plot):
        values = []
        for bench_key in available_benchmarks:
            _, metric_key, _ = BENCH_CONFIG[bench_key]
            val = None
            if condition in conditions_dict:
                cond_data = conditions_dict[condition]
                if bench_key in cond_data:
                    val = cond_data[bench_key].get(metric_key)
            values.append(val if val is not None else 0)

        offset = (i - n_cond / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               color=COLORS.get(condition, '#333'),
               label=LABELS.get(condition, condition),
               alpha=0.85, edgecolor='white', linewidth=0.5)

    bench_labels = [BENCH_CONFIG[b][0] for b in available_benchmarks]
    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Benchmark Performance ($\\alpha$={best_alpha})')
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fig_C_multi_benchmark.{fmt}')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'  Figure C (multi-benchmark) -> {out_path}')


# =========================================================================
# Figure D — Three-method enrichment comparison
# =========================================================================

def plot_enrichment_comparison(enrichment_dir, output_dir, model_name,
                               fmt='pdf', dpi=300):
    """Side-by-side bars: DH vs CETT-diff vs combined fold-enrichment."""
    plt = setup_matplotlib()

    methods = {
        r'$\Delta H$ (ablation)': 'enrichment_results.json',
        'CETT-diff': 'enrichment_results_cett_diff.json',
        r'Combined ($\Delta H \times$ CETT)': 'enrichment_results_combined.json',
    }

    method_data = {}
    for label, filename in methods.items():
        path = os.path.join(enrichment_dir, filename)
        if os.path.isfile(path):
            with open(path) as f:
                method_data[label] = json.load(f)
        else:
            print(f'  Note: {filename} not found, omitting from Figure D')

    if not method_data:
        print(f'  WARNING: No enrichment results in {enrichment_dir}. '
              f'Skipping Figure D.')
        return

    categories = ['visual', 'text', 'multimodal', 'unknown']
    cat_labels = ['Visual', 'Text', 'Multimodal', 'Unknown']
    n_methods = len(method_data)
    n_cats = len(categories)
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_cats)

    for i, (method_label, enrich_data) in enumerate(method_data.items()):
        folds = []
        for cat in categories:
            cat_data = enrich_data.get('categories', {}).get(cat, {})
            folds.append(cat_data.get('fold_enrichment', 1.0))

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, folds, bar_width, label=method_label,
               alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel('Fold Enrichment')
    ax.set_title(f'{model_name} — Enrichment by Identification Method')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fig_D_enrichment_comparison.{fmt}')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'  Figure D (enrichment comparison) -> {out_path}')


# =========================================================================
# Figure E — Per-layer enrichment heatmap
# =========================================================================

def plot_layer_heatmap(enrichment_dir, output_dir, model_name,
                       fmt='pdf', dpi=300):
    """Per-layer x modality fold-enrichment heatmap, ECCV-styled."""
    plt = setup_matplotlib()
    import matplotlib.colors as mcolors

    heatmap_path = os.path.join(enrichment_dir, 'per_layer_enrichment.npy')
    if not os.path.isfile(heatmap_path):
        print(f'  WARNING: {heatmap_path} not found. Skipping Figure E.')
        return

    heatmap = np.load(heatmap_path)
    n_layers = heatmap.shape[0]

    fig, ax = plt.subplots(figsize=(5, max(8, n_layers * 0.3)))

    finite = heatmap[np.isfinite(heatmap)]
    vmin = max(0, float(np.nanmin(finite)))
    vmax = min(5, float(np.nanmax(finite)))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', norm=norm,
                   interpolation='nearest')

    ax.set_xticks(range(4))
    ax.set_xticklabels(['Visual', 'Text', 'Multi.', 'Unk.'])
    ax.set_ylabel('Layer')
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))

    plt.colorbar(im, ax=ax, label='Fold Enrichment', shrink=0.8)
    ax.set_title(f'{model_name} — Per-Layer Enrichment')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fig_E_layer_heatmap.{fmt}')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'  Figure E (layer heatmap) -> {out_path}')


# =========================================================================
# Table F — LaTeX-ready results table
# =========================================================================

def save_results_table(data, best_alpha, output_dir, model_name):
    """Save human-readable and LaTeX results table with deltas vs baseline."""
    alpha_str = _resolve_alpha_key(data, best_alpha)
    if alpha_str is None:
        print(f'  WARNING: Alpha {best_alpha} not found. Skipping table.')
        return

    conditions_dict = data['results'][alpha_str]
    baseline = conditions_dict.get('baseline', {})
    conditions_order = ['ablate_vis', 'ablate_text', 'ablate_multimodal',
                        'ablate_unknown', 'ablate_encoder', 'ablate_projector',
                        'random']

    available_benchmarks = [b for b in BENCH_CONFIG if b in baseline]
    if not available_benchmarks:
        print(f'  WARNING: No benchmarks in baseline. Skipping table.')
        return

    # ── Plain-text table ──
    lines = []
    header = f'{"Condition":<16}'
    for bk in available_benchmarks:
        bname = BENCH_CONFIG[bk][0].replace('$_i$', '_i')
        header += f'  {bname:>12}'
    lines.append(header)
    lines.append('-' * len(header))

    row = f'{"Baseline":<16}'
    for bk in available_benchmarks:
        _, mk, _ = BENCH_CONFIG[bk]
        val = baseline.get(bk, {}).get(mk, 0)
        row += f'  {val:>12.4f}'
    lines.append(row)
    lines.append('-' * len(header))

    for condition in conditions_order:
        if condition not in conditions_dict:
            continue
        cond_data = conditions_dict[condition]
        label = LABELS.get(condition, condition)
        row = f'{label:<16}'
        for bk in available_benchmarks:
            _, mk, _ = BENCH_CONFIG[bk]
            bval = baseline.get(bk, {}).get(mk, 0)
            cval = cond_data.get(bk, {}).get(mk, 0)
            delta = cval - bval if cval and bval else 0
            sign = '+' if delta > 0 else ''
            row += f'  {sign}{delta:>11.4f}'
        lines.append(row)
    lines.append('-' * len(header))

    table_text = '\n'.join(lines)
    table_path = os.path.join(output_dir, 'results_table.txt')
    with open(table_path, 'w') as f:
        f.write(table_text)

    # ── LaTeX table ──
    latex = []
    latex.append('\\begin{tabular}{l' + 'r' * len(available_benchmarks) + '}')
    latex.append('\\toprule')

    lh = 'Condition'
    for bk in available_benchmarks:
        lh += f' & {BENCH_CONFIG[bk][0]}'
    lh += ' \\\\'
    latex.append(lh)
    latex.append('\\midrule')

    lr = 'Baseline'
    for bk in available_benchmarks:
        _, mk, _ = BENCH_CONFIG[bk]
        val = baseline.get(bk, {}).get(mk, 0)
        lr += f' & {val:.4f}'
    lr += ' \\\\'
    latex.append(lr)
    latex.append('\\midrule')

    for condition in conditions_order:
        if condition not in conditions_dict:
            continue
        cond_data = conditions_dict[condition]
        label = LABELS.get(condition, condition)
        lr = label
        for bk in available_benchmarks:
            _, mk, _ = BENCH_CONFIG[bk]
            bval = baseline.get(bk, {}).get(mk, 0)
            cval = cond_data.get(bk, {}).get(mk, 0)
            delta = cval - bval if cval and bval else 0
            sign = '+' if delta > 0 else ''
            lr += f' & {sign}{delta:.4f}'
        lr += ' \\\\'
        latex.append(lr)

    latex.append('\\bottomrule')
    latex.append('\\end{tabular}')

    latex_path = os.path.join(output_dir, 'results_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f'  Table (text)  -> {table_path}')
    print(f'  Table (LaTeX) -> {latex_path}')
    print(f'\n{table_text}')


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'STEP 13: Generating ECCV figures')
    print(f'  Model: {args.model_name}')
    print(f'  Format: {args.format} (dpi={args.dpi})')
    print(f'{"="*60}\n')

    data = load_merged(args.merged_json)
    print(f'Loaded {len(data["alphas"])} alphas, '
          f'{len(data["conditions"])} conditions, '
          f'{len(data["benchmarks"])} benchmarks')

    if args.best_alpha is not None:
        best_alpha = args.best_alpha
    else:
        best_alpha = find_best_alpha(data, condition='ablate_vis',
                                     benchmark='pope', metric='accuracy')
    print(f'Best alpha: {best_alpha}')
    print(f'\nGenerating figures...\n')

    # Figure A: Dose-response
    plot_dose_response(data, args.output_dir, args.model_name,
                       fmt=args.format, dpi=args.dpi)

    # Figure B: Ablation curve
    plot_ablation_curve(data, best_alpha, args.output_dir, args.model_name,
                        fmt=args.format, dpi=args.dpi)

    # Figure C: Multi-benchmark
    plot_multi_benchmark(data, best_alpha, args.output_dir, args.model_name,
                         fmt=args.format, dpi=args.dpi)

    # Figures D + E: Enrichment (need step 10 dir)
    if args.enrichment_dir:
        plot_enrichment_comparison(args.enrichment_dir, args.output_dir,
                                  args.model_name, fmt=args.format,
                                  dpi=args.dpi)
        plot_layer_heatmap(args.enrichment_dir, args.output_dir,
                          args.model_name, fmt=args.format, dpi=args.dpi)

    # Table F
    save_results_table(data, best_alpha, args.output_dir, args.model_name)

    print(f'\n{"="*60}')
    print(f'STEP 13 COMPLETE')
    print(f'  Figures: {args.output_dir}/fig_*.{args.format}')
    print(f'  Table:   {args.output_dir}/results_table.tex')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
