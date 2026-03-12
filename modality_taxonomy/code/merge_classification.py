#!/usr/bin/env python3
"""
merge_classification.py — Merge per-shard classification results.

Lightweight: only uses json, os, glob — no torch, no numpy, no PIL.
Runs in <1 second.

Called by run_pipeline.sh step 4 instead of neuron_modality_statistical.py --merge.
"""

import argparse         # CLI parsing
import glob             # file pattern matching
import json             # JSON I/O
import os               # path operations
import sys              # sys.exit

# ── Expected architecture per model ──────────────────────────────
# model_name: (n_layers, neurons_per_layer)
MODEL_SPECS = {
    'llava-1.5-7b':       (32, 11008),    # LLaMA-2 7B MLP intermediate
    'internvl2.5-8b':     (32, 14336),    # InternLM2 8B MLP intermediate
    'qwen2.5-vl-7b':      (28, 18944),    # Qwen2.5 7B MLP intermediate
    'llava-onevision-7b':  (28, 3584),    # Qwen2 7B (smaller MLP)
}


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Layer names per model (copied from neuron_modality_statistical.py)
# ═══════════════════════════════════════════════════════════════════

def get_layer_names(model_type, n_layers):
    """Return act_fn layer names for n_layers layers.

    Line 1: select prefix based on model_type
    Line 2: select suffix (always act_fn)
    Line 3: return list of full layer path strings
    """
    if model_type == 'llava-hf':
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'internvl':
        prefix = 'language_model.model.layers'
        suffix = 'feed_forward.act_fn'
    elif model_type == 'qwen2vl':
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    elif model_type == 'llava-ov':
        prefix = 'model.language_model.layers'
        suffix = 'mlp.act_fn'
    else:  # llava-liuhaotian
        prefix = 'model.layers'
        suffix = 'mlp.act_fn'
    return [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]     # list of layer path strings


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Merge logic (copied from neuron_modality_statistical.py)
# ═══════════════════════════════════════════════════════════════════

def merge_dir(out_base, stats_glob, label_filename, layer_names,
              layer_start, layer_end, model_name=None):
    """Merge per-shard stats and neuron labels in one output directory.

    Line 1: find all classification_stats_layers*.json files
    Line 2: combine their stats dicts and per-layer breakdowns
    Line 3: save merged classification_stats_all.json
    Line 4: collect per-layer neuron_labels.json into one file
    Line 5: verify against expected architecture if model_name is known

    Returns total classified count (0 if nothing found).
    """
    if not os.path.isdir(out_base):                                  # directory doesn't exist
        print(f'  SKIP: directory not found: {out_base}')
        return 0

    n_layers = layer_end - layer_start                               # total expected layers

    # Find and sort shard stats files by layer start number
    stats_files = sorted(
        glob.glob(os.path.join(out_base, stats_glob)),               # match pattern
        key=lambda f: int(                                           # sort by layer number
            os.path.basename(f).split('layers')[1].split('-')[0]))

    if not stats_files:                                              # no files found
        print(f'  No {stats_glob} files found in {out_base}')
        return 0

    # Accumulate stats across shards
    total_stats = {'visual': 0, 'text': 0, 'multimodal': 0, 'unknown': 0}
    per_layer_stats = {}                                             # {layer_idx: stats_dict}
    meta = None                                                      # metadata from first file

    for f in stats_files:                                            # iterate shard files
        with open(f) as fp:
            data = json.load(fp)
        if meta is None:                                             # capture metadata from first file
            meta = {k: data[k] for k in data if k not in (
                'stats', 'layer_start', 'layer_end',
                'topn_heap_time_sec', 'act_pattern_time_sec', 'time_sec')}
        for layer_idx in range(data['layer_start'], data['layer_end']):
            per_layer_stats[layer_idx] = data['stats']               # store per-layer stats
        for k in total_stats:                                        # accumulate totals
            total_stats[k] += data['stats'].get(k, 0)

    found_layers = sorted(per_layer_stats.keys())                    # which layers we found
    missing = sorted(set(range(layer_start, layer_end)) - set(found_layers))
    print(f'  Found stats for {len(found_layers)}/{n_layers} layers '
          f'from {len(stats_files)} shard files')
    if missing:                                                      # warn about missing layers
        print(f'  WARNING: missing layers: {missing}')
        print(f'  Proceeding with {len(found_layers)}/{n_layers} available layers.')

    # Effective range from what was found
    effective_start = min(found_layers)                               # earliest layer found
    effective_end = max(found_layers) + 1                            # one past latest layer

    # Save merged stats
    summary = dict(meta) if meta else {}                             # start with metadata
    summary['layer_start'] = effective_start
    summary['layer_end'] = effective_end
    summary['n_layers_found'] = len(found_layers)
    summary['stats'] = total_stats
    summary['per_layer_stats'] = {str(l): per_layer_stats[l]         # per-layer breakdown
                                  for l in sorted(per_layer_stats)}

    # Determine output filename
    if 'classification' in stats_glob:
        summary_name = 'classification_stats_all.json'
    else:
        summary_name = 'permutation_stats_all.json'
    summary_path = os.path.join(out_base, summary_name)

    with open(summary_path, 'w') as fp:                              # write merged stats
        json.dump(summary, fp, indent=2)

    total = sum(total_stats.values())                                # total neuron count
    print(f'  Overall ({total:,} neurons across {len(found_layers)} layers):')
    for k in ['visual', 'text', 'multimodal', 'unknown']:           # print breakdown
        print(f'    {k:12s}: {total_stats[k]:6,} ({100*total_stats[k]/total:.1f}%)')
    print(f'  Saved → {summary_path}')

    # Merge per-layer neuron labels into one file
    merged_labels = {}                                               # {layer_idx_str: labels_list}
    for l in sorted(per_layer_stats.keys()):                         # iterate found layers
        layer_name = layer_names[l]                                  # e.g. model.layers.5.mlp.act_fn
        label_path = os.path.join(out_base, layer_name, label_filename)
        if os.path.isfile(label_path):                               # file exists
            with open(label_path) as fp:
                merged_labels[str(l)] = json.load(fp)
        else:
            print(f'    WARNING: missing {label_path}')

    labels_all = label_filename.replace('.json', '_all.json')        # output filename
    labels_path = os.path.join(out_base, labels_all)
    with open(labels_path, 'w') as fp:                               # write merged labels
        json.dump(merged_labels, fp, indent=2)
    print(f'  Merged neuron labels for {len(merged_labels)} layers → {labels_path}')

    # ── Verify against expected architecture ─────────────────────
    if model_name and model_name in MODEL_SPECS:
        exp_layers, exp_neurons = MODEL_SPECS[model_name]            # expected architecture
        exp_total = exp_layers * exp_neurons                         # expected total neurons

        print(f'\n  ── Verification ({model_name}) ──')

        # Check layer count
        if len(found_layers) == exp_layers:                          # all layers present
            print(f'    Layers:        {len(found_layers)}/{exp_layers}  ✓')
        else:                                                        # missing layers
            print(f'    Layers:        {len(found_layers)}/{exp_layers}  ✗ INCOMPLETE')

        # Check per-layer neuron count from labels
        layer_ok = True                                              # assume all OK
        for layer_str in sorted(merged_labels.keys(), key=int):      # iterate layers
            n = len(merged_labels[layer_str])                        # neurons in this layer
            if n != exp_neurons:                                     # mismatch
                print(f'    Layer {layer_str:>2}:      {n:,} neurons (expected {exp_neurons:,})  ✗')
                layer_ok = False
        if layer_ok and merged_labels:                               # all layers correct
            print(f'    Neurons/layer: {exp_neurons:,} each  ✓')

        # Check total
        if total == exp_total:                                       # exact match
            print(f'    Total:         {total:,} / {exp_total:,}  ✓')
        else:                                                        # mismatch
            diff = exp_total - total                                 # how many missing
            print(f'    Total:         {total:,} / {exp_total:,}  ✗ ({diff:+,} neurons)')

    return total                                                     # return for caller


# ═══════════════════════════════════════════════════════════════════
# Section 3 — CLI and main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Merge per-shard classification results (lightweight, no torch)')
    p.add_argument('--model_type', required=True,                    # model backend name
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl',
                            'qwen2vl', 'llava-ov'])
    p.add_argument('--model', required=True,                         # model output dir name
                   help='Model name (e.g. llava-1.5-7b, internvl2.5-8b)')
    p.add_argument('--output_dir', required=True,                    # base output dir
                   help='Output directory (e.g. results/3-classify/full)')
    p.add_argument('--layer_start', type=int, default=0,             # first layer
                   help='First layer index (default: 0)')
    p.add_argument('--layer_end', type=int, default=32,              # last layer
                   help='One past last layer (default: 32)')
    p.add_argument('--plot', action='store_true',                    # generate plots
                   help='Generate taxonomy plots after merging')

    args = p.parse_args()

    layer_names = get_layer_names(args.model_type, args.layer_end)   # build layer name list

    # Fixed-threshold (Xu-style)
    out_base_llm = os.path.join(args.output_dir, args.model,        # e.g. results/.../llava-1.5-7b/llm_fixed_threshold
                                'llm_fixed_threshold')
    print(f'\nMerging fixed-threshold (Xu-style) results from {out_base_llm}')
    merge_dir(out_base_llm, 'classification_stats_layers*.json',
              'neuron_labels.json', layer_names,
              args.layer_start, args.layer_end, model_name=args.model)

    # Permutation-test
    out_base_perm = os.path.join(args.output_dir, args.model,       # e.g. results/.../llava-1.5-7b/llm_permutation
                                 'llm_permutation')
    print(f'\nMerging permutation-test results from {out_base_perm}')
    merge_dir(out_base_perm, 'permutation_stats_layers*.json',
              'neuron_labels_permutation.json', layer_names,
              args.layer_start, args.layer_end, model_name=args.model)

    # ── Optional: generate plots ─────────────────────────────────
    if args.plot:
        generate_plots(out_base_llm, args.model, 'fixed_threshold')
        generate_plots(out_base_perm, args.model, 'permutation')

        # Combined 2×3 grids (all 6 plots in one figure)
        th_suffix = f'th_{args.model}' if args.model else 'th'
        pmbt_suffix = f'pmbt_{args.model}' if args.model else 'pmbt'
        generate_combined_grid(out_base_llm, args.model, th_suffix)
        generate_combined_grid(out_base_perm, args.model, pmbt_suffix)

        # Side-by-side th vs pmbt comparisons
        generate_side_by_side(out_base_llm, out_base_perm,
                              args.model, args.output_dir)

    # ── Find Figure 3 candidate neurons ──────────────────────────
    find_fig3_script = os.path.join(os.path.dirname(__file__),       # same dir as this script
                                     'find_fig3_neurons.py')
    if os.path.isfile(find_fig3_script):
        fig3_data_dir = os.path.join(args.output_dir, args.model,
                                      'llm_fixed_threshold')
        topn_dir = os.path.join(fig3_data_dir, 'topn_heap')
        if os.path.isdir(topn_dir):                                  # topn_heap exists
            fig3_out = os.path.join(args.output_dir, args.model,
                                     'fig3_candidates.txt')
            print(f'\n  ── Finding Figure 3 candidate neurons ──')
            import subprocess                                        # run find_fig3
            result = subprocess.run(
                [sys.executable, find_fig3_script,
                 '--data_dir', fig3_data_dir,
                 '--model_type', args.model_type],
                capture_output=True, text=True)
            print(result.stdout)                                     # show candidates
            if result.stderr:
                print(result.stderr)
            # Save to file
            with open(fig3_out, 'w') as f:
                f.write(result.stdout)
            print(f'  Saved candidates → {fig3_out}')
        else:
            print(f'\n  SKIP find_fig3: {topn_dir} not found')
    else:
        print(f'\n  SKIP find_fig3: {find_fig3_script} not found')


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Plotting (only imported if --plot is passed)
# ═══════════════════════════════════════════════════════════════════

# Color palette for neuron types — consistent across all plots
TYPE_COLORS = {
    'visual':      '#e74c3c',   # red
    'text':        '#3498db',   # blue
    'multimodal':  '#9b59b6',   # purple
    'unknown':     '#bdc3c7',   # light grey
}
TYPE_ORDER = ['visual', 'text', 'multimodal', 'unknown']             # consistent ordering


def generate_plots(out_base, model_name, method_label):
    """Generate all taxonomy plots for one classification method.

    Line 1: load merged stats and neuron labels
    Line 2: create per-layer stacked bar chart
    Line 3: create overall pie chart
    Line 4: create confidence histograms
    Line 5: create per-layer percentage line chart
    Line 6: save all plots to plots/ subdirectory
    """
    import matplotlib                                                # import only when needed
    matplotlib.use('Agg')                                            # non-interactive backend
    import matplotlib.pyplot as plt                                  # plotting
    import matplotlib.ticker as mticker                              # axis formatting

    # ── Load data ────────────────────────────────────────────────
    stats_name = ('classification_stats_all.json'
                  if 'fixed' in method_label else 'permutation_stats_all.json')
    stats_path = os.path.join(out_base, stats_name)
    if not os.path.isfile(stats_path):                               # merged stats not found
        print(f'  SKIP plots: {stats_path} not found')
        return

    with open(stats_path) as f:
        stats_data = json.load(f)

    labels_name = ('neuron_labels_all.json'
                   if 'fixed' in method_label else 'neuron_labels_permutation_all.json')
    labels_path = os.path.join(out_base, labels_name)
    has_labels = os.path.isfile(labels_path)                         # labels may not exist
    all_labels = {}
    if has_labels:
        with open(labels_path) as f:
            all_labels = json.load(f)

    # ── Create output directory ──────────────────────────────────
    plot_dir = os.path.join(out_base, 'plots')                       # e.g. .../llm_fixed_threshold/plots/
    os.makedirs(plot_dir, exist_ok=True)

    # Clean stale unsuffixed plots from previous runs
    stale_names = [
        'per_layer_counts.png', 'per_layer_percentages.png',
        'per_layer_trend.png', 'overall_pie.png',
        'probability_distributions.png', 'confidence_by_label.png',
        'top3pct_per_layer.png', 'top3pct_trend.png',
    ]
    for stale in stale_names:                                        # remove old unsuffixed files
        stale_path = os.path.join(plot_dir, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)
            print(f'    Removed stale {stale}')

    per_layer = stats_data.get('per_layer_stats', {})                # {layer_str: {visual:N, ...}}
    overall = stats_data.get('stats', {})                            # {visual:N, text:N, ...}
    title_prefix = f'{model_name} ({method_label})'                  # e.g. llava-1.5-7b (fixed_threshold)

    # Short suffix for filenames: "th" for fixed_threshold, "pmbt" for permutation
    base_suffix = 'th' if 'fixed' in method_label else 'pmbt'          # method tag
    suffix = f'{base_suffix}_{model_name}' if model_name else base_suffix  # e.g. "th_internvl2.5-8b"

    print(f'\n  ── Generating plots → {plot_dir} ──')

    # ═════════════════════════════════════════════════════════════
    # Plot 1: Per-layer stacked bar chart (counts)
    # ═════════════════════════════════════════════════════════════
    layers = sorted(per_layer.keys(), key=int)                       # sorted layer indices
    n = len(layers)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 5))           # scale width with layers
    bottom = [0] * n                                                 # stacking baseline

    for cat in TYPE_ORDER:                                           # stack each category
        vals = [per_layer[l].get(cat, 0) for l in layers]            # counts per layer
        ax.bar(range(n), vals, bottom=bottom,                        # stacked bar
               color=TYPE_COLORS[cat], label=cat, width=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]              # update baseline

    ax.set_xlabel('Layer', fontsize=11)                              # x label
    ax.set_ylabel('Neuron Count', fontsize=11)                       # y label
    ax.set_title(f'{title_prefix}\nPer-Layer Neuron Classification', fontsize=13)
    ax.set_xticks(range(n))                                          # one tick per layer
    ax.set_xticklabels(layers, fontsize=8)                           # layer index labels
    ax.legend(loc='upper right', fontsize=9)                         # legend
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(              # comma-separated thousands
        lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    path1 = os.path.join(plot_dir, f'per_layer_counts_{suffix}.png')
    fig.savefig(path1, dpi=150)                                      # save
    plt.close(fig)
    print(f'    Saved {path1}')

    # ═════════════════════════════════════════════════════════════
    # Plot 2: Per-layer stacked bar chart (percentages)
    # ═════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 5))
    bottom = [0.0] * n                                               # percentage baseline

    for cat in TYPE_ORDER:                                           # stack each category
        vals = [per_layer[l].get(cat, 0) for l in layers]            # raw counts
        totals = [sum(per_layer[l].get(c, 0)                         # layer totals
                      for c in TYPE_ORDER) for l in layers]
        pcts = [100 * v / t if t > 0 else 0                         # convert to percentage
                for v, t in zip(vals, totals)]
        ax.bar(range(n), pcts, bottom=bottom,
               color=TYPE_COLORS[cat], label=cat, width=0.8)
        bottom = [b + p for b, p in zip(bottom, pcts)]

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'{title_prefix}\nPer-Layer Neuron Classification (%)', fontsize=13)
    ax.set_xticks(range(n))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylim(0, 100)                                              # 0-100%
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    path2 = os.path.join(plot_dir, f'per_layer_percentages_{suffix}.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f'    Saved {path2}')

    # ═════════════════════════════════════════════════════════════
    # Plot 3: Overall pie chart (with legend to avoid label overlap)
    # ═════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7, 6))
    sizes = [overall.get(cat, 0) for cat in TYPE_ORDER]              # counts per category
    colors = [TYPE_COLORS[cat] for cat in TYPE_ORDER]                # matching colors
    total = sum(sizes)                                               # total neurons

    # Only include non-zero slices
    pie_labels, pie_sizes, pie_colors = [], [], []
    for cat, sz, col in zip(TYPE_ORDER, sizes, colors):
        if sz > 0:
            pie_labels.append(f'{cat}: {sz:,} ({100*sz/total:.1f}%)')
            pie_sizes.append(sz)
            pie_colors.append(col)

    wedges, _ = ax.pie(pie_sizes, colors=pie_colors, startangle=90, # pie without inline labels
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.legend(wedges, pie_labels, loc='lower center',                # legend below pie
              bbox_to_anchor=(0.5, -0.12), fontsize=10, ncol=2,
              frameon=False)
    ax.set_title(f'{title_prefix}\nOverall ({total:,} neurons)', fontsize=13)
    plt.tight_layout()
    path3 = os.path.join(plot_dir, f'overall_pie_{suffix}.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path3}')

    # ═════════════════════════════════════════════════════════════
    # Plot 4: Confidence / probability (format-aware: th vs pmbt)
    # ═════════════════════════════════════════════════════════════
    if not has_labels or not all_labels:
        print(f'    SKIP confidence plots: no neuron labels available')
        return

    # Detect format: fixed-threshold has 'pv', permutation has 'p_value'
    sample_neuron = next(iter(all_labels.values()))[0]               # first neuron of first layer
    is_permutation = 'p_value' in sample_neuron and 'pv' not in sample_neuron

    if is_permutation:
        _plot_confidence_permutation(all_labels, title_prefix,
                                     plot_dir, suffix, bins=50)
    else:
        _plot_confidence_fixed_threshold(all_labels, title_prefix,
                                         plot_dir, suffix, bins=50)

    # ═════════════════════════════════════════════════════════════
    # Plot 5: Per-layer percentage line chart (trend view)
    # ═════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 5))

    for cat in TYPE_ORDER:                                           # one line per type
        vals = [per_layer[l].get(cat, 0) for l in layers]            # counts
        totals = [sum(per_layer[l].get(c, 0)                         # totals
                      for c in TYPE_ORDER) for l in layers]
        pcts = [100 * v / t if t > 0 else 0                         # percentages
                for v, t in zip(vals, totals)]
        ax.plot(range(n), pcts, color=TYPE_COLORS[cat],              # line
                label=cat, linewidth=2, marker='o', markersize=3)

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'{title_prefix}\nNeuron Type Distribution Across Layers', fontsize=13)
    ax.set_xticks(range(n))
    ax.set_xticklabels(layers, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)                                         # light grid
    plt.tight_layout()
    path5 = os.path.join(plot_dir, f'per_layer_trend_{suffix}.png')
    fig.savefig(path5, dpi=150)
    plt.close(fig)
    print(f'    Saved {path5}')

    # ═════════════════════════════════════════════════════════════
    # Plot 6: Top-3% specialist neuron distribution (Xu Figure 7)
    # ═════════════════════════════════════════════════════════════
    if has_labels and all_labels:
        _plot_top3_layer_distribution(all_labels, title_prefix,
                                      plot_dir, suffix, method_label,
                                      layers)

    print(f'  Done — plots saved to {plot_dir}')


def _plot_top3_layer_distribution(all_labels, title_prefix,
                                   plot_dir, suffix, method_label,
                                   layers):
    """Layer distribution of top-3% most confident neurons per category.

    Reproduces Xu et al. Figure 7: for each category, take the top 3%
    neurons ranked by their probability for that category, then plot
    which layers they fall in.

    Line 1: detect label format (fixed-threshold pv/pt/pm/pu vs permutation)
    Line 2: for each category, rank all neurons by confidence
    Line 3: take top 3%, count per layer
    Line 4: plot 4 subplots (one per category) + combined overlay
    Line 5: also print >80% confidence stats (Xu Figure 5 comparison)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Detect format
    sample_neuron = next(iter(all_labels.values()))[0]
    is_permutation = 'p_value' in sample_neuron and 'pv' not in sample_neuron

    # Collect (layer_idx, confidence_for_category) for every neuron
    # For fixed-threshold: pv, pt, pm, pu directly
    # For permutation: derive from p_value and observed_rate_diff
    cat_neurons = {cat: [] for cat in TYPE_ORDER}                    # {cat: [(layer, conf), ...]}

    for layer_str, neurons in all_labels.items():                    # iterate layers
        layer_idx = int(layer_str)
        for neuron in neurons:                                       # iterate neurons
            if is_permutation:
                pval = neuron.get('p_value', 1.0)
                rdiff = neuron.get('observed_rate_diff', 0.0)
                # Derive category-specific confidence for permutation
                # visual: high confidence = low p-value + positive rate_diff
                # text: high confidence = low p-value + negative rate_diff
                # multimodal: high confidence = high p-value (no significant diff)
                # unknown: only if no valid data
                significance = 1.0 - pval                           # higher = more significant
                cat_neurons['visual'].append(
                    (layer_idx, significance * max(0, rdiff)))       # positive rdiff × significance
                cat_neurons['text'].append(
                    (layer_idx, significance * max(0, -rdiff)))      # negative rdiff × significance
                cat_neurons['multimodal'].append(
                    (layer_idx, pval))                               # high p = multimodal
                cat_neurons['unknown'].append(
                    (layer_idx, 0.0))                                # no meaningful ranking
            else:
                pv = neuron.get('pv', 0)                             # P(visual)
                pt = neuron.get('pt', 0)                             # P(text)
                pm = neuron.get('pm', 0)                             # P(multimodal)
                pu = neuron.get('pu', 0)                             # P(unknown)
                cat_neurons['visual'].append((layer_idx, pv))
                cat_neurons['text'].append((layer_idx, pt))
                cat_neurons['multimodal'].append((layer_idx, pm))
                cat_neurons['unknown'].append((layer_idx, pu))

    total_neurons = len(cat_neurons['visual'])                       # same for all cats
    top_k = max(1, int(total_neurons * 0.03))                       # top 3%
    n_layers = len(layers)
    layer_set = sorted(set(int(l) for l in layers))                  # all layer indices

    # ── Print >80% confidence stats (compare to Xu Figure 5) ────
    print(f'\n  ── High-confidence neuron stats (>80% probability) ──')
    for cat in TYPE_ORDER:
        if is_permutation and cat == 'unknown':
            continue                                                 # skip for permutation
        above_80 = sum(1 for _, conf in cat_neurons[cat] if conf > 0.8)
        pct = 100 * above_80 / total_neurons if total_neurons > 0 else 0
        print(f'    {cat:12s}: {above_80:6,} ({pct:.1f}%) neurons with confidence > 80%')

    # ── Top 3% per category: count per layer ─────────────────────
    top3_per_layer = {}                                              # {cat: {layer: count}}
    for cat in TYPE_ORDER:
        # Sort by confidence descending, take top 3%
        ranked = sorted(cat_neurons[cat],                            # sort by confidence
                        key=lambda x: x[1], reverse=True)
        top_neurons = ranked[:top_k]                                 # top 3%

        layer_counts = {l: 0 for l in layer_set}                    # init counts
        for layer_idx, _ in top_neurons:
            layer_counts[layer_idx] += 1
        top3_per_layer[cat] = layer_counts

    # ── Plot 6a: 4 subplots, one per category (like Xu Figure 7) ─
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flat

    for ax, cat in zip(axes, TYPE_ORDER):
        counts = top3_per_layer[cat]
        x = range(n_layers)
        vals = [counts.get(l, 0) for l in layer_set]

        ax.bar(x, vals, color=TYPE_COLORS[cat], alpha=0.8,          # bar chart
               edgecolor='white', linewidth=0.5)
        ax.set_title(f'{cat} (top {top_k:,} neurons, 3%)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_set, fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'{title_prefix}\nTop 3% Neuron Distribution Across Layers (Xu Fig.7 style)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path6a = os.path.join(plot_dir, f'top3pct_per_layer_{suffix}.png')
    fig.savefig(path6a, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path6a}')

    # ── Plot 6b: overlay line chart (all 4 types on one axis) ────
    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.4), 5))

    for cat in TYPE_ORDER:
        counts = top3_per_layer[cat]
        vals = [counts.get(l, 0) for l in layer_set]
        ax.plot(range(n_layers), vals, color=TYPE_COLORS[cat],       # line per type
                label=f'{cat} (top 3%)', linewidth=2,
                marker='o', markersize=4)

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Neuron Count (top 3%)', fontsize=11)
    ax.set_title(f'{title_prefix}\nTop 3% Specialist Distribution (Xu Fig.7 style)',
                 fontsize=13)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_set, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path6b = os.path.join(plot_dir, f'top3pct_trend_{suffix}.png')
    fig.savefig(path6b, dpi=150)
    plt.close(fig)
    print(f'    Saved {path6b}')


def _plot_confidence_fixed_threshold(all_labels, title_prefix,
                                      plot_dir, suffix, bins=50):
    """Confidence plots for fixed-threshold labels (pv/pt/pm/pu fields).

    Line 1: collect pv, pt, pm, pu from every neuron
    Line 2: plot 4a — overlapping histograms of all probabilities
    Line 3: plot 4b — per-label confidence (prob of assigned label)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conf_by_label = {cat: [] for cat in TYPE_ORDER}                  # {label: [confidence, ...]}
    pv_all, pt_all, pm_all = [], [], []                              # raw probability lists

    for layer_str, neurons in all_labels.items():                    # iterate all layers
        for neuron in neurons:                                       # iterate neurons
            label = neuron.get('label', 'unknown')                   # assigned label
            pv = neuron.get('pv', 0)                                 # P(visual)
            pt = neuron.get('pt', 0)                                 # P(text)
            pm = neuron.get('pm', 0)                                 # P(multimodal)
            pu = neuron.get('pu', 0)                                 # P(unknown)

            pv_all.append(pv)                                        # collect for histogram
            pt_all.append(pt)
            pm_all.append(pm)

            conf_map = {'visual': pv, 'text': pt,                    # map label → its prob
                        'multimodal': pm, 'unknown': pu}
            if label in conf_map:
                conf_by_label[label].append(conf_map[label])

    # 4a: Overlapping histograms of pv, pt, pm
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pv_all, bins=bins, alpha=0.5,                            # visual probabilities
            color=TYPE_COLORS['visual'], label='P(visual)', density=True)
    ax.hist(pt_all, bins=bins, alpha=0.5,                            # text probabilities
            color=TYPE_COLORS['text'], label='P(text)', density=True)
    ax.hist(pm_all, bins=bins, alpha=0.5,                            # multimodal probabilities
            color=TYPE_COLORS['multimodal'], label='P(multimodal)', density=True)
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{title_prefix}\nClassification Probability Distributions', fontsize=13)
    ax.set_xlim(0, 1)                                                # 0–1 range
    ax.legend(fontsize=10)
    plt.tight_layout()
    path4a = os.path.join(plot_dir, f'probability_distributions_{suffix}.png')
    fig.savefig(path4a, dpi=150)
    plt.close(fig)
    print(f'    Saved {path4a}')

    # 4b: Confidence of assigned label — one histogram per type
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)    # 4 subplots
    for ax, cat in zip(axes, TYPE_ORDER):                            # one per type
        vals = conf_by_label.get(cat, [])
        if vals:
            ax.hist(vals, bins=bins, color=TYPE_COLORS[cat],
                    alpha=0.8, edgecolor='white', linewidth=0.3)
            median = sorted(vals)[len(vals) // 2]                    # median
            ax.axvline(median, color='black', linestyle='--',
                       linewidth=1, label=f'median={median:.2f}')
            ax.legend(fontsize=8)
        ax.set_title(f'{cat}\n({len(vals):,} neurons)', fontsize=10)
        ax.set_xlabel('Confidence (P)', fontsize=9)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel('Count', fontsize=10)
    fig.suptitle(f'{title_prefix} — Confidence of Assigned Label', fontsize=13, y=1.02)
    plt.tight_layout()
    path4b = os.path.join(plot_dir, f'confidence_by_label_{suffix}.png')
    fig.savefig(path4b, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path4b}')


def _plot_confidence_permutation(all_labels, title_prefix,
                                  plot_dir, suffix, bins=50):
    """Confidence plots for permutation-test labels (p_value / observed_rate_diff).

    Line 1: collect p_value and observed_rate_diff from every neuron
    Line 2: plot 4a — p-value histogram + rate_diff histogram (2 subplots)
    Line 3: plot 4b — per-label confidence (1 - p_value)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_vals_all = []                                                  # all p-values
    rate_diffs_all = []                                              # all observed_rate_diff
    conf_by_label = {cat: [] for cat in TYPE_ORDER}                  # {label: [1-pval, ...]}
    p_by_label = {cat: [] for cat in TYPE_ORDER}                     # {label: [pval, ...]}

    for layer_str, neurons in all_labels.items():                    # iterate all layers
        for neuron in neurons:                                       # iterate neurons
            label = neuron.get('label', 'unknown')                   # assigned label
            pval = neuron.get('p_value', 1.0)                       # permutation p-value
            rdiff = neuron.get('observed_rate_diff', 0.0)            # vis_rate - txt_rate

            p_vals_all.append(pval)
            rate_diffs_all.append(rdiff)

            # Confidence = 1 - p_value
            # visual/text: low p → high confidence (significant difference)
            # multimodal: high p → classified as multimodal (no significant diff)
            #   so confidence for multimodal = 1 - p still makes sense
            #   (lower is "less confident" it's truly multimodal)
            confidence = 1.0 - pval
            if label in conf_by_label:
                conf_by_label[label].append(confidence)
                p_by_label[label].append(pval)

    # 4a: Two subplots — p-value distribution + rate_diff by label
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: p-value histogram colored by label
    for cat in ['visual', 'text', 'multimodal']:                     # skip unknown (tiny)
        vals = p_by_label.get(cat, [])
        if vals:
            ax1.hist(vals, bins=bins, alpha=0.5,                     # overlapping histograms
                     color=TYPE_COLORS[cat], label=f'{cat} ({len(vals):,})',
                     density=True)
    ax1.set_xlabel('p-value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('p-value Distribution by Label', fontsize=12)
    ax1.axvline(0.05, color='black', linestyle='--', linewidth=1,   # significance threshold
                label='α=0.05')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=9)

    # Right: observed_rate_diff histogram (positive = visual, negative = text)
    ax2.hist(rate_diffs_all, bins=bins, alpha=0.7,
             color='#2c3e50', edgecolor='white', linewidth=0.3)
    ax2.axvline(0, color='red', linestyle='-', linewidth=1.5,       # zero line
                label='D=0 (equal)')
    ax2.set_xlabel('Rate Difference (vis_rate − txt_rate)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Observed Rate Difference', fontsize=12)
    ax2.legend(fontsize=9)

    fig.suptitle(f'{title_prefix}\nPermutation Test Statistics', fontsize=13, y=1.02)
    plt.tight_layout()
    path4a = os.path.join(plot_dir, f'probability_distributions_{suffix}.png')
    fig.savefig(path4a, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path4a}')

    # 4b: Confidence (1 - p_value) per label
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)    # 4 subplots
    for ax, cat in zip(axes, TYPE_ORDER):
        vals = conf_by_label.get(cat, [])
        if vals:
            ax.hist(vals, bins=bins, color=TYPE_COLORS[cat],
                    alpha=0.8, edgecolor='white', linewidth=0.3)
            median = sorted(vals)[len(vals) // 2]                    # median
            ax.axvline(median, color='black', linestyle='--',
                       linewidth=1, label=f'median={median:.2f}')
            ax.legend(fontsize=8)
        ax.set_title(f'{cat}\n({len(vals):,} neurons)', fontsize=10)
        ax.set_xlabel('Confidence (1 − p)', fontsize=9)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel('Count', fontsize=10)
    fig.suptitle(f'{title_prefix} — Confidence (1 − p-value)', fontsize=13, y=1.02)
    plt.tight_layout()
    path4b = os.path.join(plot_dir, f'confidence_by_label_{suffix}.png')
    fig.savefig(path4b, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path4b}')


def generate_combined_grid(out_base, model_name, suffix):
    """Combine all 6 plots into a single 2x3 grid figure.

    Line 1: load all 6 saved PNGs from plots/ directory
    Line 2: arrange in 2 rows x 3 columns
    Line 3: save as combined_all_{suffix}.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    plot_dir = os.path.join(out_base, 'plots')
    if not os.path.isdir(plot_dir):
        return

    # The 8 plots in display order (top row, bottom row)
    plot_names = [
        f'per_layer_counts_{suffix}.png',          # row 1, col 1
        f'per_layer_percentages_{suffix}.png',      # row 1, col 2
        f'per_layer_trend_{suffix}.png',            # row 1, col 3
        f'top3pct_trend_{suffix}.png',              # row 1, col 4
        f'overall_pie_{suffix}.png',                # row 2, col 1
        f'probability_distributions_{suffix}.png',  # row 2, col 2
        f'confidence_by_label_{suffix}.png',        # row 2, col 3
        f'top3pct_per_layer_{suffix}.png',          # row 2, col 4
    ]

    # Check all plots exist (skip missing gracefully)
    paths = [os.path.join(plot_dir, name) for name in plot_names]
    available = [(p, name) for p, name in zip(paths, plot_names)
                 if os.path.isfile(p)]
    if len(available) < 6:
        print(f'  SKIP combined grid ({suffix}): only {len(available)} plots found')
        return

    method_label = 'Fixed Threshold' if suffix == 'th' else 'Permutation Test'

    # Subplot titles for each plot
    subplot_titles = [
        'Per-Layer Counts', 'Per-Layer Percentages',
        'Per-Layer Trend (All)', 'Top 3% Trend (Xu Fig.7)',
        'Overall Distribution', 'Probability Distributions',
        'Confidence by Label', 'Top 3% Per-Layer (Xu Fig.7)',
    ]

    n_plots = len(available)
    n_cols = 4                                                       # 4 columns
    n_rows = (n_plots + n_cols - 1) // n_cols                        # ceil division
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(36, 8 * n_rows))              # wide figure
    fig.suptitle(f'{model_name} — {method_label}',
                 fontsize=18, fontweight='bold', y=0.99)

    for idx, (path, _) in enumerate(available):                      # fill grid
        ax = axes.flat[idx]
        img = mpimg.imread(path)                                     # load PNG
        ax.imshow(img)                                               # display
        if idx < len(subplot_titles):
            ax.set_title(subplot_titles[idx], fontsize=12,           # subplot label
                         fontweight='bold', pad=8, color='#34495e')
        ax.axis('off')                                               # hide axes

    # Hide unused axes
    for idx in range(len(available), len(axes.flat)):
        axes.flat[idx].axis('off')

    plt.subplots_adjust(hspace=0.12, wspace=0.05)                   # tight spacing
    out_path = os.path.join(plot_dir, f'combined_all_{suffix}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {out_path}')


def generate_side_by_side(out_base_th, out_base_pmbt,
                          model_name, output_dir):
    """Generate side-by-side th vs pmbt comparison for each plot type.

    Line 1: for each of the 6 plot types, load th and pmbt versions
    Line 2: place them side by side (th left, pmbt right)
    Line 3: save to a shared comparison directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Build suffixes matching generate_plots
    th_suffix = f'th_{model_name}' if model_name else 'th'
    pmbt_suffix = f'pmbt_{model_name}' if model_name else 'pmbt'

    plot_dir_th = os.path.join(out_base_th, 'plots')
    plot_dir_pmbt = os.path.join(out_base_pmbt, 'plots')

    if not os.path.isdir(plot_dir_th) or not os.path.isdir(plot_dir_pmbt):
        print(f'  SKIP side-by-side: one or both plot dirs missing')
        return

    # Output directory for comparison plots
    compare_dir = os.path.join(output_dir, model_name, 'plots_comparison')
    os.makedirs(compare_dir, exist_ok=True)

    print(f'\n  ── Generating side-by-side comparisons → {compare_dir} ──')

    # Plot types to compare
    plot_types = [
        ('per_layer_counts',          'Per-Layer Neuron Counts'),
        ('per_layer_percentages',     'Per-Layer Percentages'),
        ('per_layer_trend',           'Per-Layer Trend (All Neurons)'),
        ('top3pct_trend',             'Top 3% Trend (Xu Fig.7 style)'),
        ('overall_pie',               'Overall Distribution'),
        ('probability_distributions', 'Probability Distributions'),
        ('confidence_by_label',       'Confidence by Label'),
        ('top3pct_per_layer',         'Top 3% Per-Layer (Xu Fig.7 style)'),
    ]

    for base_name, title in plot_types:                              # iterate plot types
        th_path = os.path.join(plot_dir_th, f'{base_name}_{th_suffix}.png')
        pmbt_path = os.path.join(plot_dir_pmbt, f'{base_name}_{pmbt_suffix}.png')

        if not os.path.isfile(th_path) or not os.path.isfile(pmbt_path):
            continue                                                 # skip if either missing

        img_th = mpimg.imread(th_path)                               # load th plot
        img_pmbt = mpimg.imread(pmbt_path)                           # load pmbt plot

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))      # side by side
        fig.suptitle(f'{model_name} — {title}',
                     fontsize=14, fontweight='bold', y=1.0)

        ax1.imshow(img_th)                                           # th on left
        ax1.set_title('Fixed Threshold', fontsize=12, color='#2c3e50')
        ax1.axis('off')

        ax2.imshow(img_pmbt)                                         # pmbt on right
        ax2.set_title('Permutation Test', fontsize=12, color='#2c3e50')
        ax2.axis('off')

        plt.tight_layout()
        cmp_suffix = f'_{model_name}' if model_name else ''
        out_path = os.path.join(compare_dir, f'{base_name}_comparison{cmp_suffix}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved {out_path}')

    # Also create a master grid directly from source images (not nested composites)
    # Layout: 6 rows (one per plot type) × 2 columns (th left, pmbt right)
    row_images = []                                                  # [(th_img, pmbt_img, title), ...]
    for base_name, title in plot_types:
        th_path = os.path.join(plot_dir_th, f'{base_name}_{th_suffix}.png')
        pmbt_path = os.path.join(plot_dir_pmbt, f'{base_name}_{pmbt_suffix}.png')
        if os.path.isfile(th_path) and os.path.isfile(pmbt_path):
            row_images.append((th_path, pmbt_path, title))

    if len(row_images) >= 4:                                         # at least 4 rows to be useful
        n_rows = len(row_images)
        fig, axes = plt.subplots(n_rows, 2,                         # 6 rows × 2 cols
                                  figsize=(28, 7 * n_rows))          # tall figure, generous height

        # Column headers
        axes[0, 0].set_title('Fixed Threshold', fontsize=16,
                             fontweight='bold', color='#2c3e50', pad=20)
        axes[0, 1].set_title('Permutation Test', fontsize=16,
                             fontweight='bold', color='#2c3e50', pad=20)

        for row, (th_path, pmbt_path, title) in enumerate(row_images):
            img_th = mpimg.imread(th_path)                           # load th source image
            img_pmbt = mpimg.imread(pmbt_path)                       # load pmbt source image

            axes[row, 0].imshow(img_th)                              # th in left column
            axes[row, 0].axis('off')

            axes[row, 1].imshow(img_pmbt)                            # pmbt in right column
            axes[row, 1].axis('off')

        # Adjust layout first so positions are final
        fig.suptitle(f'{model_name} — Fixed Threshold vs Permutation Test',
                     fontsize=20, fontweight='bold', y=1.0)
        plt.subplots_adjust(hspace=0.15, wspace=0.05,               # tight spacing
                            left=0.08)                               # room for row labels

        # Now add row labels using final axes positions
        for row, (_, _, title) in enumerate(row_images):
            pos = axes[row, 0].get_position()
            y_mid = pos.y0 + pos.height / 2                         # vertical center of row
            fig.text(0.02, y_mid, title, fontsize=13,                # row label on far left
                     fontweight='bold', ha='center', va='center',
                     rotation=90, color='#34495e')

        master_path = os.path.join(compare_dir, f'all_comparisons_{model_name}.png')
        fig.savefig(master_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved {master_path}')

    print(f'  Done — comparisons saved to {compare_dir}')



if __name__ == "__main__":
    main()