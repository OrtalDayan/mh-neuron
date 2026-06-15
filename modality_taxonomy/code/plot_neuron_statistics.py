#!/usr/bin/env python3
"""
plot_neuron_statistics.py — Generate Xu et al. Figures 5, 6, 7

Reads Phase 3 (neuron_labels.json) and/or Phase 4 (neuron_labels_permutation.json)
output from neuron_modality_statistical.py and produces publication-quality charts.

Figures:
    --fig5   Stacked bar chart: proportion of visual/text/multimodal/unknown
             neurons per layer.  Xu Figure 5 (Section 4.3).
    --fig6   High-confidence distribution: only neurons with max(pv,pt,pm,pu) > 0.8.
             Xu Figure 6 (Section 4.3).
    --fig7   Cross-model comparison: same as fig5 but overlays multiple models
             side-by-side.  Xu Figure 7 (Section 4.6).
    --all    Generate all three figures (default).

Data sources:
    Phase 3 (Xu-style argmax):
        {data_dir}/{layer_name}/neuron_labels.json
    Phase 4 (Permutation test):
        {data_dir_perm}/{layer_name}/neuron_labels_permutation.json

Both contain per-neuron dicts with keys:
    neuron_idx, label, pv, pt, pm, pu (Phase 3)
    neuron_idx, label, p_value, otsu_threshold, ... (Phase 4)

No GPU needed.

Usage:
    # All figures from Phase 3 data
    python plot_neuron_statistics.py \\
        --data_dir classification_xu/llava-1.5-7b/llm \\
        --all

    # Figure 5 only, using permutation test labels
    python plot_neuron_statistics.py \\
        --data_dir classification_xu/llava-1.5-7b/llm_permutation \\
        --label_file neuron_labels_permutation.json \\
        --fig5

    # Figure 6 from Phase 3 (high-confidence, needs pv/pt/pm/pu)
    python plot_neuron_statistics.py \\
        --data_dir classification_xu/llava-1.5-7b/llm \\
        --fig6

    # Figure 7: compare LLaVA-1.5 vs another model
    python plot_neuron_statistics.py \\
        --data_dirs classification_xu/llava-1.5-7b/llm \\
                    classification_xu/internvl-2.5-8b/llm \\
        --model_names "LLaVA-1.5 7B" "InternVL 2.5 8B" \\
        --fig7
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')                                               # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Data loading
# ═══════════════════════════════════════════════════════════════════

# Colour scheme matching Xu paper conventions
COLORS = {
    'visual':     '#E74C3C',                                        # red
    'text':       '#3498DB',                                        # blue
    'multimodal': '#2ECC71',                                        # green
    'unknown':    '#95A5A6',                                        # grey
}

TYPE_ORDER = ['visual', 'text', 'multimodal', 'unknown']            # consistent legend order


def get_layer_dir_name(layer, model_type='llava-hf'):
    """Build the directory name for a layer.

    neuron_modality_statistical.py saves each layer's labels in a directory
    named after the full hook target, e.g.:
        model.language_model.model.layers.0.mlp.act_fn   (HF)
        model.layers.0.mlp.act_fn                        (original)

    Args:
        layer:      int — layer index (0-31)
        model_type: str — 'llava-hf' or 'llava-liuhaotian'

    Returns:
        str — directory name for this layer
    """
    if model_type == 'llava-hf':
        return f'model.language_model.model.layers.{layer}.mlp.act_fn'
    elif model_type == 'internvl':
        return f'language_model.model.layers.{layer}.feed_forward.act_fn'
    elif model_type in ('qwen2vl', 'llava-ov'):
        return f'model.language_model.layers.{layer}.mlp.act_fn'
    else:
        return f'model.layers.{layer}.mlp.act_fn'


def load_layer_labels(data_dir, layer, model_type='llava-hf',
                      label_file='neuron_labels.json'):
    """Load neuron labels for one layer.

    Args:
        data_dir:   str — base directory (e.g. classification_xu/llava-1.5-7b/llm)
        layer:      int — layer index
        model_type: str — 'llava-hf' or 'llava-liuhaotian'
        label_file: str — filename ('neuron_labels.json' or
                          'neuron_labels_permutation.json')

    Returns:
        list[dict] — one dict per neuron with at minimum 'label' key,
                     and 'pv','pt','pm','pu' for Phase 3 data
    """
    layer_dir = get_layer_dir_name(layer, model_type)
    path = os.path.join(data_dir, layer_dir, label_file)
    with open(path) as f:
        return json.load(f)


def load_all_layers(data_dir, layer_start=0, layer_end=31,
                    model_type='llava-hf', label_file='neuron_labels.json'):
    """Load labels for all layers in range.

    Returns:
        dict — {layer_idx: list[dict]} mapping each layer to its neuron labels
    """
    all_labels = {}
    for layer in range(layer_start, layer_end + 1):
        try:
            labels = load_layer_labels(
                data_dir, layer, model_type, label_file)
            all_labels[layer] = labels
        except FileNotFoundError:
            print(f'  WARNING: No data for layer {layer}, skipping')
    return all_labels


def compute_layer_counts(all_labels):
    """Count neurons per type per layer.

    Args:
        all_labels: dict — {layer: list[dict]} from load_all_layers

    Returns:
        layers:  sorted list of layer indices
        counts:  dict — {type: np.array of counts per layer}
        totals:  np.array — total neurons per layer
    """
    layers = sorted(all_labels.keys())
    counts = {t: np.zeros(len(layers), dtype=int) for t in TYPE_ORDER}

    for i, layer in enumerate(layers):
        for neuron in all_labels[layer]:
            lbl = neuron['label']
            if lbl in counts:
                counts[lbl][i] += 1
            else:
                counts['unknown'][i] += 1                           # catch unexpected labels

    totals = sum(counts[t] for t in TYPE_ORDER)
    return layers, counts, totals


def compute_high_confidence_counts(all_labels, threshold=0.8):
    """Count high-confidence neurons per type per layer.

    A neuron is "high-confidence" if its dominant probability exceeds
    the threshold (Xu uses 0.8).  Phase 3 labels include pv, pt, pm, pu.
    Phase 4 labels don't have individual probabilities, so this only
    works with Phase 3 data.

    Args:
        all_labels: dict — {layer: list[dict]} from load_all_layers
        threshold:  float — minimum probability to count as high-confidence

    Returns:
        layers:  sorted list of layer indices
        counts:  dict — {type: np.array of high-confidence counts}
        totals:  np.array — total high-confidence neurons per layer
        n_neurons_per_layer: np.array — total neurons per layer (for %)
    """
    layers = sorted(all_labels.keys())
    counts = {t: np.zeros(len(layers), dtype=int) for t in TYPE_ORDER}
    n_neurons = np.zeros(len(layers), dtype=int)

    for i, layer in enumerate(layers):
        for neuron in all_labels[layer]:
            n_neurons[i] += 1
            # Get the dominant probability
            pv = neuron.get('pv', 0)
            pt = neuron.get('pt', 0)
            pm = neuron.get('pm', 0)
            pu = neuron.get('pu', 0)
            max_p = max(pv, pt, pm, pu)

            if max_p > threshold:
                lbl = neuron['label']
                if lbl in counts:
                    counts[lbl][i] += 1
                else:
                    counts['unknown'][i] += 1

    totals = sum(counts[t] for t in TYPE_ORDER)
    return layers, counts, totals, n_neurons


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Figure 5: Stacked bar chart (neuron type proportions)
# ═══════════════════════════════════════════════════════════════════

def plot_fig5(all_labels, output_path, title='', dpi=200, fmt='png'):
    """Generate Figure 5: stacked bar chart of neuron type proportions.

    Each layer is one bar, stacked to 100%. Four segments correspond to
    visual (red), text (blue), multimodal (green), unknown (grey).

    Args:
        all_labels: dict — {layer: list[dict]} from load_all_layers
        output_path: str — file to save
        title: str — chart title (default auto-generated)
        dpi: int — resolution
        fmt: str — file format
    """
    layers, counts, totals = compute_layer_counts(all_labels)

    # Convert to percentages
    pcts = {}
    for t in TYPE_ORDER:
        pcts[t] = np.where(totals > 0,
                           counts[t] / totals * 100, 0)            # avoid division by zero

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(layers))
    bar_width = 0.8
    bottom = np.zeros(len(layers))

    for t in TYPE_ORDER:
        ax.bar(x, pcts[t], bar_width, bottom=bottom,
               color=COLORS[t], label=t.capitalize(),
               edgecolor='white', linewidth=0.3)
        bottom += pcts[t]

    # Formatting
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Proportion (%)', fontsize=12)
    ax.set_title(title or 'Neuron Type Distribution Across Layers',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add neuron count annotation at bottom
    n_total = int(totals.sum())
    n_per = int(totals[0]) if len(totals) > 0 else 0
    ax.text(0.01, -0.12,
            f'Total: {n_total:,} neurons ({n_per:,} per layer)',
            fontsize=8, color='gray',
            transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  Saved Figure 5: {output_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Figure 6: High-confidence neuron distribution
# ═══════════════════════════════════════════════════════════════════

def plot_fig6(all_labels, output_path, threshold=0.8,
              title='', dpi=200, fmt='png'):
    """Generate Figure 6: grouped bar chart of high-confidence neurons.

    Shows the count (or percentage) of neurons per layer whose dominant
    probability exceeds the threshold.  Bars grouped by type, side-by-side.

    Requires Phase 3 data (neuron_labels.json with pv/pt/pm/pu).

    Args:
        all_labels: dict — {layer: list[dict]}
        output_path: str — output file path
        threshold: float — confidence threshold (default 0.8)
        title: str — chart title
        dpi: int — figure resolution
        fmt: str — file format
    """
    # Verify Phase 3 data (has pv/pt/pm/pu)
    sample_layer = next(iter(all_labels.values()))
    if 'pv' not in sample_layer[0]:
        print('  ERROR: Figure 6 requires Phase 3 data with pv/pt/pm/pu. '
              'Use --label_file neuron_labels.json (not permutation).')
        return

    layers, counts, totals, n_neurons = compute_high_confidence_counts(
        all_labels, threshold)

    # Convert to percentages of total neurons per layer
    pcts = {}
    for t in TYPE_ORDER:
        pcts[t] = np.where(n_neurons > 0,
                           counts[t] / n_neurons * 100, 0)

    # ── Plot: grouped bar chart ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    height_ratios=[1, 1],
                                    sharex=True)

    x = np.arange(len(layers))
    n_types = len(TYPE_ORDER)
    group_width = 0.8
    bar_w = group_width / n_types

    # Top panel: counts
    for i, t in enumerate(TYPE_ORDER):
        offset = (i - n_types / 2 + 0.5) * bar_w
        ax1.bar(x + offset, counts[t], bar_w,
                color=COLORS[t], label=t.capitalize(),
                edgecolor='white', linewidth=0.3)

    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(
        title or f'High-Confidence Neurons (p > {threshold:.0%}) Per Layer',
        fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotate total high-confidence %
    total_hc = int(totals.sum())
    total_all = int(n_neurons.sum())
    hc_pct = total_hc / total_all * 100 if total_all > 0 else 0
    ax1.text(0.01, 0.95,
             f'Total: {total_hc:,} / {total_all:,} '
             f'({hc_pct:.1f}%) neurons above {threshold:.0%}',
             fontsize=9, color='gray',
             transform=ax1.transAxes, va='top')

    # Bottom panel: percentages
    for i, t in enumerate(TYPE_ORDER):
        offset = (i - n_types / 2 + 0.5) * bar_w
        ax2.bar(x + offset, pcts[t], bar_w,
                color=COLORS[t],
                edgecolor='white', linewidth=0.3)

    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Proportion (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, fontsize=8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Per-type summary annotation
    type_totals = {t: int(counts[t].sum()) for t in TYPE_ORDER}
    summary_parts = [f'{t.capitalize()}: {type_totals[t]:,} '
                     f'({type_totals[t]/total_all*100:.1f}%)'
                     for t in TYPE_ORDER if type_totals[t] > 0]
    ax2.text(0.01, -0.15,
             '  |  '.join(summary_parts),
             fontsize=8, color='gray',
             transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  Saved Figure 6: {output_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Figure 7: Cross-model comparison
# ═══════════════════════════════════════════════════════════════════

def plot_fig7(model_data, model_names, output_path,
              title='', dpi=200, fmt='png'):
    """Generate Figure 7: cross-model comparison of neuron type distributions.

    Plots one subplot per model, each showing the stacked area/line chart
    of neuron type proportions across layers.  Allows side-by-side visual
    comparison of how different VLMs distribute neuron types.

    Args:
        model_data:  list[dict] — each is {layer: list[dict]} from load_all_layers
        model_names: list[str] — display names for each model
        output_path: str — output file path
        title: str — overall figure title
        dpi: int — figure resolution
        fmt: str — file format
    """
    n_models = len(model_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]                                               # ensure iterable

    for idx, (all_labels, name) in enumerate(zip(model_data, model_names)):
        ax = axes[idx]
        layers, counts, totals = compute_layer_counts(all_labels)
        x = np.arange(len(layers))

        # Compute percentages
        pcts = {}
        for t in TYPE_ORDER:
            pcts[t] = np.where(totals > 0,
                               counts[t] / totals * 100, 0)

        # Stacked area (filled line chart) — matches Xu Figure 5/7 style
        bottom = np.zeros(len(layers))
        for t in TYPE_ORDER:
            ax.fill_between(x, bottom, bottom + pcts[t],
                            color=COLORS[t], alpha=0.7,
                            label=t.capitalize())
            ax.plot(x, bottom + pcts[t], color=COLORS[t],
                    linewidth=0.8, alpha=0.9)
            bottom += pcts[t]

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks(x[::2])                                      # every 2nd layer for readability
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)],
                           fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

        # Add total neuron count
        n_total = int(totals.sum())
        ax.text(0.02, 0.02, f'{n_total:,} neurons',
                fontsize=8, color='gray',
                transform=ax.transAxes)

    axes[0].set_ylabel('Proportion (%)', fontsize=11)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(TYPE_ORDER), fontsize=10,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    fig.suptitle(
        title or 'Neuron Type Distribution Across Models',
        fontsize=14, fontweight='bold', y=1.06)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  Saved Figure 7: {output_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 4b — Figure 8: Combined FT (top) + PMBT (bottom) grid
# ═══════════════════════════════════════════════════════════════════

def plot_fig8_combined(ft_model_data, pmbt_model_data, model_names,
                       output_path, title='', dpi=200, fmt='png'):
    """Generate Figure 8: 2-row × N-column combined stacked bar grid.

    Layout::

        ┌─────────────────────────────────────────────┐
        │  Model A          Model B          Model C   │  ← column titles
        │  [FT stacked bar] [FT stacked bar] [...]     │  ← top row: FT
        │  [PM stacked bar] [PM stacked bar] [...]     │  ← bottom row: PMBT
        └─────────────────────────────────────────────┘

    Each cell is a 100 % stacked bar chart (one bar per layer).
    The left y-axis label carries the row tag ("FT" / "PMBT").

    Args:
        ft_model_data:   list[dict] — one {layer: list[neuron_dict]} per model,
                         loaded with FT (fixed-threshold) label files.
        pmbt_model_data: list[dict] — same shape, loaded with PMBT
                         (permutation-test) label files.
        model_names:     list[str]  — display name for each model (column header).
        output_path:     str — file path to save the figure.
        title:           str — overall suptitle (auto-generated if empty).
        dpi:             int — raster resolution.
        fmt:             str — file format passed to savefig.
    """
    n_models = len(model_names)                                      # number of columns

    # ── Build 2 × N subplot grid ──────────────────────────────
    fig, axes = plt.subplots(
        2, n_models,
        figsize=(6 * n_models, 8),                                   # 6 in wide per model, 4 in per row
        sharey='row',                                                 # share y-axis within each row
        sharex='col',                                                 # share x-axis within each column
    )

    # Guarantee axes is always a 2-D array even with a single model
    if n_models == 1:
        axes = axes.reshape(2, 1)                                    # (2,) → (2, 1)

    ROW_TAGS   = ['FT', 'PMBT']                                      # row labels shown on y-axis
    ROW_DATA   = [ft_model_data, pmbt_model_data]                    # parallel data lists

    for row_idx, (row_tag, model_data_list) in enumerate(
            zip(ROW_TAGS, ROW_DATA)):

        for col_idx, (all_labels, name) in enumerate(
                zip(model_data_list, model_names)):

            ax = axes[row_idx, col_idx]

            # ── Compute per-layer percentages ──────────────
            layers, counts, totals = compute_layer_counts(all_labels)
            x      = np.arange(len(layers))                          # integer x positions
            bottom = np.zeros(len(layers))                           # running stack bottom

            pcts = {}
            for t in TYPE_ORDER:
                pcts[t] = np.where(totals > 0,
                                   counts[t] / totals * 100, 0)     # avoid /0; result in [0,100]

            # ── Draw stacked bars ──────────────────────────
            for t in TYPE_ORDER:
                ax.bar(x, pcts[t], 0.8,
                       bottom=bottom,
                       color=COLORS[t],
                       label=t.capitalize(),
                       edgecolor='white', linewidth=0.3)
                bottom += pcts[t]                                    # advance stack

            # ── Column title (top row only) ────────────────
            if row_idx == 0:
                ax.set_title(name, fontsize=12, fontweight='bold', pad=8)

            # ── Y-axis label with row tag (left column only) ──
            if col_idx == 0:
                ax.set_ylabel(
                    f'{row_tag}\nProportion (%)',
                    fontsize=11, labelpad=6)

            # ── X-axis labels (bottom row only) ───────────
            if row_idx == 1:
                ax.set_xlabel('Layer', fontsize=10)
                step = max(1, len(layers) // 16)                     # show at most 16 ticks
                ax.set_xticks(x[::step])
                ax.set_xticklabels(
                    [layers[i] for i in range(0, len(layers), step)],
                    fontsize=7)
            else:
                ax.tick_params(axis='x', labelbottom=False)          # hide x-tick labels on top row

            # ── Shared formatting ──────────────────────────
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
            ax.grid(axis='y', alpha=0.2, linestyle='--')

            # Neuron-count annotation (bottom-left corner)
            n_total = int(totals.sum())
            ax.text(0.02, 0.02,
                    f'{n_total:,} neurons',
                    fontsize=7, color='gray',
                    transform=ax.transAxes, va='bottom')

    # ── Shared legend centred above all subplots ───────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()         # pull from first cell
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(TYPE_ORDER),
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
        framealpha=0.9)

    # ── Overall title ──────────────────────────────────────────
    fig.suptitle(
        title or 'Neuron Type Distribution — FT (top) vs PMBT (bottom)',
        fontsize=14, fontweight='bold', y=1.06)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  Saved Figure 8 (combined FT+PMBT): {output_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Bonus: Line chart (matches your existing plot style)
# ═══════════════════════════════════════════════════════════════════

def plot_line_chart(all_labels, output_path, title='', dpi=200, fmt='png'):
    """Generate a line chart of neuron type proportions per layer.

    This matches the style of your existing neuron_modality_per_layer.png.
    One coloured line per type, x-axis = layer, y-axis = percentage.

    Args:
        all_labels: dict — {layer: list[dict]}
        output_path: str — output file path
        title: str — chart title
        dpi: int — figure resolution
        fmt: str — file format
    """
    layers, counts, totals = compute_layer_counts(all_labels)
    x = np.array(layers)

    # Compute percentages
    pcts = {}
    for t in TYPE_ORDER:
        pcts[t] = np.where(totals > 0,
                           counts[t] / totals * 100, 0)

    fig, ax = plt.subplots(figsize=(12, 5))

    for t in TYPE_ORDER:
        ax.plot(x, pcts[t], '-o', color=COLORS[t],
                label=t.capitalize(), markersize=4, linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Neurons (%)', fontsize=12)
    ax.set_title(title or 'Neuron Type Distribution Across Layers',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  Saved line chart: {output_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Bonus: Side-by-side Phase 3 vs Phase 4 comparison
# ═══════════════════════════════════════════════════════════════════

def plot_phase_comparison(phase3_labels, phase4_labels, output_path,
                          title='', dpi=200, fmt='png'):
    """Generate a side-by-side comparison of Phase 3 (Xu) vs Phase 4 (Permutation).

    Two subplots: left = Phase 3 stacked bars, right = Phase 4 stacked bars.
    Allows direct visual comparison of the two classification methods.

    Args:
        phase3_labels: dict — {layer: list[dict]} Phase 3 data
        phase4_labels: dict — {layer: list[dict]} Phase 4 data
        output_path: str — output file path
        title: str — overall title
        dpi: int — figure resolution
        fmt: str — file format
    """
    plot_fig7(
        model_data=[phase3_labels, phase4_labels],
        model_names=['Phase 3 — Xu-style Classification',
                     'Phase 4 — Permutation Test'],
        output_path=output_path,
        title=title or 'Classification Method Comparison',
        dpi=dpi, fmt=fmt,
    )


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Argument parsing and main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate Xu et al. Figures 5, 6, 7 from classification data')

    # Data paths
    p.add_argument('--data_dir',
                   help='Classification output dir '
                        '(e.g. classification_xu/llava-1.5-7b/llm)')
    p.add_argument('--data_dirs', nargs='+',
                   help='Multiple data dirs for --fig7 cross-model comparison')
    p.add_argument('--model_names', nargs='+',
                   help='Display names for each model in --fig7')
    p.add_argument('--perm_dir',
                   help='Phase 4 permutation dir (for --compare). '
                        'Default: replaces /llm with /llm_permutation')

    # File names
    p.add_argument('--label_file', default='neuron_labels.json',
                   help='Label filename per layer '
                        '(neuron_labels.json or neuron_labels_permutation.json)')
    p.add_argument('--model_type', default='llava-hf',
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
                   help='Model type (determines layer directory naming)')
    p.add_argument('--model_types', nargs='+', default=None,
                   help='Per-model type list for --fig7 cross-model comparison. '
                        'Must match length of --data_dirs. Falls back to '
                        '--model_type for all models if not provided.')
    p.add_argument('--label_files', nargs='+', default=None,
                   help='Per-model label filenames for --fig7 cross-model '
                        'comparison. Must match length of --data_dirs. '
                        'Falls back to --label_file for all models if not provided.')

    # Layer range
    p.add_argument('--layer_start', type=int, default=0)
    p.add_argument('--layer_end', type=int, default=31)

    # Figure selection
    p.add_argument('--fig5', action='store_true',
                   help='Stacked bar chart: neuron type proportions per layer')
    p.add_argument('--fig6', action='store_true',
                   help='High-confidence distribution (>80%% threshold)')
    p.add_argument('--fig7', action='store_true',
                   help='Cross-model comparison (requires --data_dirs)')
    p.add_argument('--fig8', action='store_true',
                   help='Combined FT (top row) + PMBT (bottom row) cross-model '
                        'stacked bar grid. Requires --ft_dirs, --pmbt_dirs, '
                        'and --model_names.')
    p.add_argument('--line', action='store_true',
                   help='Line chart (matches existing plot style)')
    p.add_argument('--compare', action='store_true',
                   help='Side-by-side Phase 3 vs Phase 4 comparison')
    p.add_argument('--all', action='store_true',
                   help='Generate all figures (default if no fig specified)')

    # Figure 8 — per-row data directories and metadata
    p.add_argument('--ft_dirs', nargs='+', default=None,
                   help='FT classification dirs, one per model (top row of --fig8). '
                        'Must align with --model_names.')
    p.add_argument('--pmbt_dirs', nargs='+', default=None,
                   help='PMBT classification dirs, one per model (bottom row of --fig8). '
                        'Must align with --model_names.')
    p.add_argument('--ft_model_types', nargs='+', default=None,
                   help='Per-model model_type for --ft_dirs. '
                        'Falls back to --model_type for all models if omitted.')
    p.add_argument('--pmbt_model_types', nargs='+', default=None,
                   help='Per-model model_type for --pmbt_dirs. '
                        'Falls back to --model_type for all models if omitted.')
    p.add_argument('--ft_label_files', nargs='+', default=None,
                   help='Per-model label filename for --ft_dirs (default: neuron_labels.json).')
    p.add_argument('--pmbt_label_files', nargs='+', default=None,
                   help='Per-model label filename for --pmbt_dirs '
                        '(default: neuron_labels_permutation.json).')

    # Figure 6 options
    p.add_argument('--threshold', type=float, default=0.8,
                   help='Confidence threshold for Figure 6 (default 0.8)')

    # Output
    p.add_argument('--output_dir', default='figure_outputs',
                   help='Directory for output figures')
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--format', default='png',
                   choices=['png', 'pdf', 'svg'])
    p.add_argument('--title_prefix', default='',
                   help='Prefix for figure titles (e.g. model name)')
    p.add_argument('--model_name', default='',
                   help='Human-readable model name appended to filenames '
                        '(e.g. "llava-1.5-7b")')

    return p.parse_args()


def main():
    args = parse_args()

    # Default to --all if no specific figure requested
    if not any([args.fig5, args.fig6, args.fig7, args.fig8,
                args.line, args.compare]):
        args.all = True

    os.makedirs(args.output_dir, exist_ok=True)

    prefix = args.title_prefix
    # Auto-set title prefix from model_name if not explicitly provided
    if not prefix and args.model_name:
        prefix = args.model_name
    if prefix and not prefix.endswith(' '):
        prefix += ' '

    # Filename suffix from model_name (e.g. "_llava-1.5-7b")
    mn_suffix = f'_{args.model_name}' if args.model_name else ''

    # ── Load primary data ─────────────────────────────────────
    if args.data_dir:
        print(f'Loading data from {args.data_dir}')
        print(f'  Label file:  {args.label_file}')
        print(f'  Model type:  {args.model_type}')
        print(f'  Layers:      {args.layer_start}-{args.layer_end}')

        all_labels = load_all_layers(
            args.data_dir, args.layer_start, args.layer_end,
            args.model_type, args.label_file)

        if not all_labels:
            print('ERROR: No data loaded. Check --data_dir and --model_type.')
            return

        n_layers = len(all_labels)
        n_neurons = sum(len(v) for v in all_labels.values())
        print(f'  Loaded: {n_layers} layers, {n_neurons:,} neurons')

    # ── Figure 5: Stacked bar chart ───────────────────────────
    if args.fig5 or args.all:
        out = os.path.join(args.output_dir,
                           f'fig5_stacked_bar{mn_suffix}.{args.format}')
        plot_fig5(all_labels, out,
                  title=f'{prefix}Neuron Type Distribution Across Layers',
                  dpi=args.dpi, fmt=args.format)

    # ── Figure 6: High-confidence distribution ────────────────
    if args.fig6 or args.all:
        out = os.path.join(args.output_dir,
                           f'fig6_high_confidence{mn_suffix}.{args.format}')
        plot_fig6(all_labels, out, threshold=args.threshold,
                  title=f'{prefix}High-Confidence Neurons '
                        f'(p > {args.threshold:.0%}) Per Layer',
                  dpi=args.dpi, fmt=args.format)

    # ── Figure 7: Cross-model comparison ──────────────────────
    if args.fig7 or args.all and args.data_dirs:
        if not args.data_dirs:
            if args.data_dir:
                # Single model — produce figure 7 with just one panel
                args.data_dirs = [args.data_dir]
                args.model_names = args.model_names or ['LLaVA-1.5 7B']
            else:
                print('  SKIP fig7: no --data_dirs provided')

        if args.data_dirs:
            model_data = []
            names = args.model_names or [f'Model {i+1}'
                                         for i in range(len(args.data_dirs))]
            # Per-model types and label files (fall back to singular args)
            mtypes = args.model_types or [args.model_type] * len(args.data_dirs)
            lfiles = args.label_files or [args.label_file] * len(args.data_dirs)
            for dd, mt, lf in zip(args.data_dirs, mtypes, lfiles):
                print(f'  Loading {dd} for fig7... (type={mt}, labels={lf})')
                md = load_all_layers(
                    dd, args.layer_start, args.layer_end,
                    mt, lf)
                model_data.append(md)

            out = os.path.join(args.output_dir,
                               f'fig7_cross_model{mn_suffix}.{args.format}')
            plot_fig7(model_data, names, out,
                      title=f'{prefix}Neuron Type Distribution '
                            f'Across Models',
                      dpi=args.dpi, fmt=args.format)

    # ── Figure 8: Combined FT + PMBT cross-model grid ────────
    if args.fig8:
        if not args.ft_dirs or not args.pmbt_dirs:
            print('  SKIP fig8: --ft_dirs and --pmbt_dirs are both required')
        elif not args.model_names:
            print('  SKIP fig8: --model_names is required')
        else:
            names = args.model_names

            # Resolve per-model types — fall back to single --model_type
            ft_mtypes   = args.ft_model_types   or [args.model_type] * len(args.ft_dirs)
            pmbt_mtypes = args.pmbt_model_types  or [args.model_type] * len(args.pmbt_dirs)

            # Resolve per-model label files — fall back to sensible defaults
            ft_lfiles   = args.ft_label_files   or ['neuron_labels.json']            * len(args.ft_dirs)
            pmbt_lfiles = args.pmbt_label_files  or ['neuron_labels_permutation.json'] * len(args.pmbt_dirs)

            # ── Load FT data (one dict per model) ─────────
            ft_data = []
            for dd, mt, lf in zip(args.ft_dirs, ft_mtypes, ft_lfiles):
                print(f'  [fig8 FT]   Loading {dd}  (type={mt}, labels={lf})')
                ft_data.append(
                    load_all_layers(dd, args.layer_start, args.layer_end, mt, lf))

            # ── Load PMBT data (one dict per model) ────────
            pmbt_data = []
            for dd, mt, lf in zip(args.pmbt_dirs, pmbt_mtypes, pmbt_lfiles):
                print(f'  [fig8 PMBT] Loading {dd}  (type={mt}, labels={lf})')
                pmbt_data.append(
                    load_all_layers(dd, args.layer_start, args.layer_end, mt, lf))

            out = os.path.join(args.output_dir,
                               f'fig8_combined_ft_pmbt{mn_suffix}.{args.format}')
            plot_fig8_combined(
                ft_data, pmbt_data, names, out,
                title=f'{prefix}Neuron Type Distribution — FT vs PMBT',
                dpi=args.dpi, fmt=args.format)

    # ── Line chart (bonus, matches existing style) ────────────
    if args.line or args.all:
        out = os.path.join(args.output_dir,
                           f'line_chart{mn_suffix}.{args.format}')
        plot_line_chart(all_labels, out,
                        title=f'{prefix}Neuron Type Distribution '
                              f'Across Layers',
                        dpi=args.dpi, fmt=args.format)

    # ── Phase 3 vs Phase 4 comparison ─────────────────────────
    if args.compare or args.all:
        # Determine permutation dir
        if args.perm_dir:
            perm_dir = args.perm_dir
        elif args.data_dir:
            # Default: replace /llm with /llm_permutation
            perm_dir = args.data_dir.rstrip('/').replace(
                '/llm', '/llm_permutation')
        else:
            perm_dir = None

        if perm_dir and os.path.isdir(perm_dir):
            print(f'\n  Loading Phase 4 data from {perm_dir}')
            phase4_labels = load_all_layers(
                perm_dir, args.layer_start, args.layer_end,
                args.model_type,
                'neuron_labels_permutation.json')

            if phase4_labels:
                out = os.path.join(args.output_dir,
                                   f'phase3_vs_phase4{mn_suffix}.{args.format}')
                plot_phase_comparison(all_labels, phase4_labels, out,
                                     title=f'{prefix}Phase 3 (Xu) vs '
                                           f'Phase 4 (Permutation Test)',
                                     dpi=args.dpi, fmt=args.format)
            else:
                print('  WARNING: No Phase 4 data loaded')
        else:
            if args.compare:
                print(f'  SKIP compare: permutation dir not found '
                      f'({perm_dir})')

    print(f'\n{"═"*60}')
    print(f'Done. Figures saved to {args.output_dir}/')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# """
# plot_neuron_statistics.py — Generate Xu et al. Figures 5, 6, 7

# Reads Phase 3 (neuron_labels.json) and/or Phase 4 (neuron_labels_permutation.json)
# output from neuron_modality_statistical.py and produces publication-quality charts.

# Figures:
#     --fig5   Stacked bar chart: proportion of visual/text/multimodal/unknown
#              neurons per layer.  Xu Figure 5 (Section 4.3).
#     --fig6   High-confidence distribution: only neurons with max(pv,pt,pm,pu) > 0.8.
#              Xu Figure 6 (Section 4.3).
#     --fig7   Cross-model comparison: same as fig5 but overlays multiple models
#              side-by-side.  Xu Figure 7 (Section 4.6).
#     --all    Generate all three figures (default).

# Data sources:
#     Phase 3 (Xu-style argmax):
#         {data_dir}/{layer_name}/neuron_labels.json
#     Phase 4 (Permutation test):
#         {data_dir_perm}/{layer_name}/neuron_labels_permutation.json

# Both contain per-neuron dicts with keys:
#     neuron_idx, label, pv, pt, pm, pu (Phase 3)
#     neuron_idx, label, p_value, otsu_threshold, ... (Phase 4)

# No GPU needed.

# Usage:
#     # All figures from Phase 3 data
#     python plot_neuron_statistics.py \\
#         --data_dir classification_xu/llava-1.5-7b/llm \\
#         --all

#     # Figure 5 only, using permutation test labels
#     python plot_neuron_statistics.py \\
#         --data_dir classification_xu/llava-1.5-7b/llm_permutation \\
#         --label_file neuron_labels_permutation.json \\
#         --fig5

#     # Figure 6 from Phase 3 (high-confidence, needs pv/pt/pm/pu)
#     python plot_neuron_statistics.py \\
#         --data_dir classification_xu/llava-1.5-7b/llm \\
#         --fig6

#     # Figure 7: compare LLaVA-1.5 vs another model
#     python plot_neuron_statistics.py \\
#         --data_dirs classification_xu/llava-1.5-7b/llm \\
#                     classification_xu/internvl-2.5-8b/llm \\
#         --model_names "LLaVA-1.5 7B" "InternVL 2.5 8B" \\
#         --fig7
# """

# import argparse
# import json
# import os

# import matplotlib
# matplotlib.use('Agg')                                               # non-interactive backend
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np


# # ═══════════════════════════════════════════════════════════════════
# # Section 1 — Data loading
# # ═══════════════════════════════════════════════════════════════════

# # Colour scheme matching Xu paper conventions
# COLORS = {
#     'visual':     '#E74C3C',                                        # red
#     'text':       '#3498DB',                                        # blue
#     'multimodal': '#2ECC71',                                        # green
#     'unknown':    '#95A5A6',                                        # grey
# }

# TYPE_ORDER = ['visual', 'text', 'multimodal', 'unknown']            # consistent legend order


# def get_layer_dir_name(layer, model_type='llava-hf'):
#     """Build the directory name for a layer.

#     neuron_modality_statistical.py saves each layer's labels in a directory
#     named after the full hook target, e.g.:
#         model.language_model.model.layers.0.mlp.act_fn   (HF)
#         model.layers.0.mlp.act_fn                        (original)

#     Args:
#         layer:      int — layer index (0-31)
#         model_type: str — 'llava-hf' or 'llava-liuhaotian'

#     Returns:
#         str — directory name for this layer
#     """
#     if model_type == 'llava-hf':
#         return f'model.language_model.model.layers.{layer}.mlp.act_fn'
#     elif model_type == 'internvl':
#         return f'language_model.model.layers.{layer}.feed_forward.act_fn'
#     elif model_type in ('qwen2vl', 'llava-ov'):
#         return f'model.language_model.layers.{layer}.mlp.act_fn'
#     else:
#         return f'model.layers.{layer}.mlp.act_fn'


# def load_layer_labels(data_dir, layer, model_type='llava-hf',
#                       label_file='neuron_labels.json'):
#     """Load neuron labels for one layer.

#     Args:
#         data_dir:   str — base directory (e.g. classification_xu/llava-1.5-7b/llm)
#         layer:      int — layer index
#         model_type: str — 'llava-hf' or 'llava-liuhaotian'
#         label_file: str — filename ('neuron_labels.json' or
#                           'neuron_labels_permutation.json')

#     Returns:
#         list[dict] — one dict per neuron with at minimum 'label' key,
#                      and 'pv','pt','pm','pu' for Phase 3 data
#     """
#     layer_dir = get_layer_dir_name(layer, model_type)
#     path = os.path.join(data_dir, layer_dir, label_file)
#     with open(path) as f:
#         return json.load(f)


# def load_all_layers(data_dir, layer_start=0, layer_end=31,
#                     model_type='llava-hf', label_file='neuron_labels.json'):
#     """Load labels for all layers in range.

#     Returns:
#         dict — {layer_idx: list[dict]} mapping each layer to its neuron labels
#     """
#     all_labels = {}
#     for layer in range(layer_start, layer_end + 1):
#         try:
#             labels = load_layer_labels(
#                 data_dir, layer, model_type, label_file)
#             all_labels[layer] = labels
#         except FileNotFoundError:
#             print(f'  WARNING: No data for layer {layer}, skipping')
#     return all_labels


# def compute_layer_counts(all_labels):
#     """Count neurons per type per layer.

#     Args:
#         all_labels: dict — {layer: list[dict]} from load_all_layers

#     Returns:
#         layers:  sorted list of layer indices
#         counts:  dict — {type: np.array of counts per layer}
#         totals:  np.array — total neurons per layer
#     """
#     layers = sorted(all_labels.keys())
#     counts = {t: np.zeros(len(layers), dtype=int) for t in TYPE_ORDER}

#     for i, layer in enumerate(layers):
#         for neuron in all_labels[layer]:
#             lbl = neuron['label']
#             if lbl in counts:
#                 counts[lbl][i] += 1
#             else:
#                 counts['unknown'][i] += 1                           # catch unexpected labels

#     totals = sum(counts[t] for t in TYPE_ORDER)
#     return layers, counts, totals


# def compute_high_confidence_counts(all_labels, threshold=0.8):
#     """Count high-confidence neurons per type per layer.

#     A neuron is "high-confidence" if its dominant probability exceeds
#     the threshold (Xu uses 0.8).  Phase 3 labels include pv, pt, pm, pu.
#     Phase 4 labels don't have individual probabilities, so this only
#     works with Phase 3 data.

#     Args:
#         all_labels: dict — {layer: list[dict]} from load_all_layers
#         threshold:  float — minimum probability to count as high-confidence

#     Returns:
#         layers:  sorted list of layer indices
#         counts:  dict — {type: np.array of high-confidence counts}
#         totals:  np.array — total high-confidence neurons per layer
#         n_neurons_per_layer: np.array — total neurons per layer (for %)
#     """
#     layers = sorted(all_labels.keys())
#     counts = {t: np.zeros(len(layers), dtype=int) for t in TYPE_ORDER}
#     n_neurons = np.zeros(len(layers), dtype=int)

#     for i, layer in enumerate(layers):
#         for neuron in all_labels[layer]:
#             n_neurons[i] += 1
#             # Get the dominant probability
#             pv = neuron.get('pv', 0)
#             pt = neuron.get('pt', 0)
#             pm = neuron.get('pm', 0)
#             pu = neuron.get('pu', 0)
#             max_p = max(pv, pt, pm, pu)

#             if max_p > threshold:
#                 lbl = neuron['label']
#                 if lbl in counts:
#                     counts[lbl][i] += 1
#                 else:
#                     counts['unknown'][i] += 1

#     totals = sum(counts[t] for t in TYPE_ORDER)
#     return layers, counts, totals, n_neurons


# # ═══════════════════════════════════════════════════════════════════
# # Section 2 — Figure 5: Stacked bar chart (neuron type proportions)
# # ═══════════════════════════════════════════════════════════════════

# def plot_fig5(all_labels, output_path, title='', dpi=200, fmt='png'):
#     """Generate Figure 5: stacked bar chart of neuron type proportions.

#     Each layer is one bar, stacked to 100%. Four segments correspond to
#     visual (red), text (blue), multimodal (green), unknown (grey).

#     Args:
#         all_labels: dict — {layer: list[dict]} from load_all_layers
#         output_path: str — file to save
#         title: str — chart title (default auto-generated)
#         dpi: int — resolution
#         fmt: str — file format
#     """
#     layers, counts, totals = compute_layer_counts(all_labels)

#     # Convert to percentages
#     pcts = {}
#     for t in TYPE_ORDER:
#         pcts[t] = np.where(totals > 0,
#                            counts[t] / totals * 100, 0)            # avoid division by zero

#     # ── Plot ──────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(14, 5))

#     x = np.arange(len(layers))
#     bar_width = 0.8
#     bottom = np.zeros(len(layers))

#     for t in TYPE_ORDER:
#         ax.bar(x, pcts[t], bar_width, bottom=bottom,
#                color=COLORS[t], label=t.capitalize(),
#                edgecolor='white', linewidth=0.3)
#         bottom += pcts[t]

#     # Formatting
#     ax.set_xlabel('Layer', fontsize=12)
#     ax.set_ylabel('Proportion (%)', fontsize=12)
#     ax.set_title(title or 'Neuron Type Distribution Across Layers',
#                  fontsize=14, fontweight='bold', pad=10)
#     ax.set_xticks(x)
#     ax.set_xticklabels(layers, fontsize=8)
#     ax.set_ylim(0, 100)
#     ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
#     ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
#     ax.grid(axis='y', alpha=0.3, linestyle='--')

#     # Add neuron count annotation at bottom
#     n_total = int(totals.sum())
#     n_per = int(totals[0]) if len(totals) > 0 else 0
#     ax.text(0.01, -0.12,
#             f'Total: {n_total:,} neurons ({n_per:,} per layer)',
#             fontsize=8, color='gray',
#             transform=ax.transAxes)

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
#                 facecolor='white')
#     plt.close(fig)
#     print(f'  Saved Figure 5: {output_path}')


# # ═══════════════════════════════════════════════════════════════════
# # Section 3 — Figure 6: High-confidence neuron distribution
# # ═══════════════════════════════════════════════════════════════════

# def plot_fig6(all_labels, output_path, threshold=0.8,
#               title='', dpi=200, fmt='png'):
#     """Generate Figure 6: grouped bar chart of high-confidence neurons.

#     Shows the count (or percentage) of neurons per layer whose dominant
#     probability exceeds the threshold.  Bars grouped by type, side-by-side.

#     Requires Phase 3 data (neuron_labels.json with pv/pt/pm/pu).

#     Args:
#         all_labels: dict — {layer: list[dict]}
#         output_path: str — output file path
#         threshold: float — confidence threshold (default 0.8)
#         title: str — chart title
#         dpi: int — figure resolution
#         fmt: str — file format
#     """
#     # Verify Phase 3 data (has pv/pt/pm/pu)
#     sample_layer = next(iter(all_labels.values()))
#     if 'pv' not in sample_layer[0]:
#         print('  ERROR: Figure 6 requires Phase 3 data with pv/pt/pm/pu. '
#               'Use --label_file neuron_labels.json (not permutation).')
#         return

#     layers, counts, totals, n_neurons = compute_high_confidence_counts(
#         all_labels, threshold)

#     # Convert to percentages of total neurons per layer
#     pcts = {}
#     for t in TYPE_ORDER:
#         pcts[t] = np.where(n_neurons > 0,
#                            counts[t] / n_neurons * 100, 0)

#     # ── Plot: grouped bar chart ───────────────────────────────
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
#                                     height_ratios=[1, 1],
#                                     sharex=True)

#     x = np.arange(len(layers))
#     n_types = len(TYPE_ORDER)
#     group_width = 0.8
#     bar_w = group_width / n_types

#     # Top panel: counts
#     for i, t in enumerate(TYPE_ORDER):
#         offset = (i - n_types / 2 + 0.5) * bar_w
#         ax1.bar(x + offset, counts[t], bar_w,
#                 color=COLORS[t], label=t.capitalize(),
#                 edgecolor='white', linewidth=0.3)

#     ax1.set_ylabel('Count', fontsize=11)
#     ax1.set_title(
#         title or f'High-Confidence Neurons (p > {threshold:.0%}) Per Layer',
#         fontsize=14, fontweight='bold', pad=10)
#     ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
#     ax1.grid(axis='y', alpha=0.3, linestyle='--')

#     # Annotate total high-confidence %
#     total_hc = int(totals.sum())
#     total_all = int(n_neurons.sum())
#     hc_pct = total_hc / total_all * 100 if total_all > 0 else 0
#     ax1.text(0.01, 0.95,
#              f'Total: {total_hc:,} / {total_all:,} '
#              f'({hc_pct:.1f}%) neurons above {threshold:.0%}',
#              fontsize=9, color='gray',
#              transform=ax1.transAxes, va='top')

#     # Bottom panel: percentages
#     for i, t in enumerate(TYPE_ORDER):
#         offset = (i - n_types / 2 + 0.5) * bar_w
#         ax2.bar(x + offset, pcts[t], bar_w,
#                 color=COLORS[t],
#                 edgecolor='white', linewidth=0.3)

#     ax2.set_xlabel('Layer', fontsize=12)
#     ax2.set_ylabel('Proportion (%)', fontsize=11)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(layers, fontsize=8)
#     ax2.grid(axis='y', alpha=0.3, linestyle='--')

#     # Per-type summary annotation
#     type_totals = {t: int(counts[t].sum()) for t in TYPE_ORDER}
#     summary_parts = [f'{t.capitalize()}: {type_totals[t]:,} '
#                      f'({type_totals[t]/total_all*100:.1f}%)'
#                      for t in TYPE_ORDER if type_totals[t] > 0]
#     ax2.text(0.01, -0.15,
#              '  |  '.join(summary_parts),
#              fontsize=8, color='gray',
#              transform=ax2.transAxes)

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
#                 facecolor='white')
#     plt.close(fig)
#     print(f'  Saved Figure 6: {output_path}')


# # ═══════════════════════════════════════════════════════════════════
# # Section 4 — Figure 7: Cross-model comparison
# # ═══════════════════════════════════════════════════════════════════

# def plot_fig7(model_data, model_names, output_path,
#               title='', dpi=200, fmt='png'):
#     """Generate Figure 7: cross-model comparison of neuron type distributions.

#     Plots one subplot per model, each showing the stacked area/line chart
#     of neuron type proportions across layers.  Allows side-by-side visual
#     comparison of how different VLMs distribute neuron types.

#     Args:
#         model_data:  list[dict] — each is {layer: list[dict]} from load_all_layers
#         model_names: list[str] — display names for each model
#         output_path: str — output file path
#         title: str — overall figure title
#         dpi: int — figure resolution
#         fmt: str — file format
#     """
#     n_models = len(model_data)
#     fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5),
#                              sharey=True)
#     if n_models == 1:
#         axes = [axes]                                               # ensure iterable

#     for idx, (all_labels, name) in enumerate(zip(model_data, model_names)):
#         ax = axes[idx]
#         layers, counts, totals = compute_layer_counts(all_labels)
#         x = np.arange(len(layers))

#         # Compute percentages
#         pcts = {}
#         for t in TYPE_ORDER:
#             pcts[t] = np.where(totals > 0,
#                                counts[t] / totals * 100, 0)

#         # Stacked area (filled line chart) — matches Xu Figure 5/7 style
#         bottom = np.zeros(len(layers))
#         for t in TYPE_ORDER:
#             ax.fill_between(x, bottom, bottom + pcts[t],
#                             color=COLORS[t], alpha=0.7,
#                             label=t.capitalize())
#             ax.plot(x, bottom + pcts[t], color=COLORS[t],
#                     linewidth=0.8, alpha=0.9)
#             bottom += pcts[t]

#         ax.set_xlabel('Layer', fontsize=11)
#         ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
#         ax.set_xticks(x[::2])                                      # every 2nd layer for readability
#         ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)],
#                            fontsize=8)
#         ax.set_ylim(0, 100)
#         ax.grid(axis='y', alpha=0.2, linestyle='--')

#         # Add total neuron count
#         n_total = int(totals.sum())
#         ax.text(0.02, 0.02, f'{n_total:,} neurons',
#                 fontsize=8, color='gray',
#                 transform=ax.transAxes)

#     axes[0].set_ylabel('Proportion (%)', fontsize=11)

#     # Single legend for all subplots
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center',
#                ncol=len(TYPE_ORDER), fontsize=10,
#                bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

#     fig.suptitle(
#         title or 'Neuron Type Distribution Across Models',
#         fontsize=14, fontweight='bold', y=1.06)

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
#                 facecolor='white')
#     plt.close(fig)
#     print(f'  Saved Figure 7: {output_path}')


# # ═══════════════════════════════════════════════════════════════════
# # Section 5 — Bonus: Line chart (matches your existing plot style)
# # ═══════════════════════════════════════════════════════════════════

# def plot_line_chart(all_labels, output_path, title='', dpi=200, fmt='png'):
#     """Generate a line chart of neuron type proportions per layer.

#     This matches the style of your existing neuron_modality_per_layer.png.
#     One coloured line per type, x-axis = layer, y-axis = percentage.

#     Args:
#         all_labels: dict — {layer: list[dict]}
#         output_path: str — output file path
#         title: str — chart title
#         dpi: int — figure resolution
#         fmt: str — file format
#     """
#     layers, counts, totals = compute_layer_counts(all_labels)
#     x = np.array(layers)

#     # Compute percentages
#     pcts = {}
#     for t in TYPE_ORDER:
#         pcts[t] = np.where(totals > 0,
#                            counts[t] / totals * 100, 0)

#     fig, ax = plt.subplots(figsize=(12, 5))

#     for t in TYPE_ORDER:
#         ax.plot(x, pcts[t], '-o', color=COLORS[t],
#                 label=t.capitalize(), markersize=4, linewidth=2)

#     ax.set_xlabel('Layer', fontsize=12)
#     ax.set_ylabel('Neurons (%)', fontsize=12)
#     ax.set_title(title or 'Neuron Type Distribution Across Layers',
#                  fontsize=14, fontweight='bold', pad=10)
#     ax.set_xticks(x)
#     ax.legend(fontsize=10, framealpha=0.9)
#     ax.grid(alpha=0.3, linestyle='--')
#     ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
#                 facecolor='white')
#     plt.close(fig)
#     print(f'  Saved line chart: {output_path}')


# # ═══════════════════════════════════════════════════════════════════
# # Section 6 — Bonus: Side-by-side Phase 3 vs Phase 4 comparison
# # ═══════════════════════════════════════════════════════════════════

# def plot_phase_comparison(phase3_labels, phase4_labels, output_path,
#                           title='', dpi=200, fmt='png'):
#     """Generate a side-by-side comparison of Phase 3 (Xu) vs Phase 4 (Permutation).

#     Two subplots: left = Phase 3 stacked bars, right = Phase 4 stacked bars.
#     Allows direct visual comparison of the two classification methods.

#     Args:
#         phase3_labels: dict — {layer: list[dict]} Phase 3 data
#         phase4_labels: dict — {layer: list[dict]} Phase 4 data
#         output_path: str — output file path
#         title: str — overall title
#         dpi: int — figure resolution
#         fmt: str — file format
#     """
#     plot_fig7(
#         model_data=[phase3_labels, phase4_labels],
#         model_names=['Phase 3 — Xu-style Classification',
#                      'Phase 4 — Permutation Test'],
#         output_path=output_path,
#         title=title or 'Classification Method Comparison',
#         dpi=dpi, fmt=fmt,
#     )


# # ═══════════════════════════════════════════════════════════════════
# # Section 7 — Argument parsing and main
# # ═══════════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser(
#         description='Generate Xu et al. Figures 5, 6, 7 from classification data')

#     # Data paths
#     p.add_argument('--data_dir',
#                    help='Classification output dir '
#                         '(e.g. classification_xu/llava-1.5-7b/llm)')
#     p.add_argument('--data_dirs', nargs='+',
#                    help='Multiple data dirs for --fig7 cross-model comparison')
#     p.add_argument('--model_names', nargs='+',
#                    help='Display names for each model in --fig7')
#     p.add_argument('--perm_dir',
#                    help='Phase 4 permutation dir (for --compare). '
#                         'Default: replaces /llm with /llm_permutation')

#     # File names
#     p.add_argument('--label_file', default='neuron_labels.json',
#                    help='Label filename per layer '
#                         '(neuron_labels.json or neuron_labels_permutation.json)')
#     p.add_argument('--model_type', default='llava-hf',
#                    choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
#                    help='Model type (determines layer directory naming)')
#     p.add_argument('--model_types', nargs='+', default=None,
#                    help='Per-model type list for --fig7 cross-model comparison. '
#                         'Must match length of --data_dirs. Falls back to '
#                         '--model_type for all models if not provided.')
#     p.add_argument('--label_files', nargs='+', default=None,
#                    help='Per-model label filenames for --fig7 cross-model '
#                         'comparison. Must match length of --data_dirs. '
#                         'Falls back to --label_file for all models if not provided.')

#     # Layer range
#     p.add_argument('--layer_start', type=int, default=0)
#     p.add_argument('--layer_end', type=int, default=31)

#     # Figure selection
#     p.add_argument('--fig5', action='store_true',
#                    help='Stacked bar chart: neuron type proportions per layer')
#     p.add_argument('--fig6', action='store_true',
#                    help='High-confidence distribution (>80%% threshold)')
#     p.add_argument('--fig7', action='store_true',
#                    help='Cross-model comparison (requires --data_dirs)')
#     p.add_argument('--line', action='store_true',
#                    help='Line chart (matches existing plot style)')
#     p.add_argument('--compare', action='store_true',
#                    help='Side-by-side Phase 3 vs Phase 4 comparison')
#     p.add_argument('--all', action='store_true',
#                    help='Generate all figures (default if no fig specified)')

#     # Figure 6 options
#     p.add_argument('--threshold', type=float, default=0.8,
#                    help='Confidence threshold for Figure 6 (default 0.8)')

#     # Output
#     p.add_argument('--output_dir', default='figure_outputs',
#                    help='Directory for output figures')
#     p.add_argument('--dpi', type=int, default=200)
#     p.add_argument('--format', default='png',
#                    choices=['png', 'pdf', 'svg'])
#     p.add_argument('--title_prefix', default='',
#                    help='Prefix for figure titles (e.g. model name)')
#     p.add_argument('--model_name', default='',
#                    help='Human-readable model name appended to filenames '
#                         '(e.g. "llava-1.5-7b")')

#     return p.parse_args()


# def main():
#     args = parse_args()

#     # Default to --all if no specific figure requested
#     if not any([args.fig5, args.fig6, args.fig7,
#                 args.line, args.compare]):
#         args.all = True

#     os.makedirs(args.output_dir, exist_ok=True)

#     prefix = args.title_prefix
#     # Auto-set title prefix from model_name if not explicitly provided
#     if not prefix and args.model_name:
#         prefix = args.model_name
#     if prefix and not prefix.endswith(' '):
#         prefix += ' '

#     # Filename suffix from model_name (e.g. "_llava-1.5-7b")
#     mn_suffix = f'_{args.model_name}' if args.model_name else ''

#     # ── Load primary data ─────────────────────────────────────
#     if args.data_dir:
#         print(f'Loading data from {args.data_dir}')
#         print(f'  Label file:  {args.label_file}')
#         print(f'  Model type:  {args.model_type}')
#         print(f'  Layers:      {args.layer_start}-{args.layer_end}')

#         all_labels = load_all_layers(
#             args.data_dir, args.layer_start, args.layer_end,
#             args.model_type, args.label_file)

#         if not all_labels:
#             print('ERROR: No data loaded. Check --data_dir and --model_type.')
#             return

#         n_layers = len(all_labels)
#         n_neurons = sum(len(v) for v in all_labels.values())
#         print(f'  Loaded: {n_layers} layers, {n_neurons:,} neurons')

#     # ── Figure 5: Stacked bar chart ───────────────────────────
#     if args.fig5 or args.all:
#         out = os.path.join(args.output_dir,
#                            f'fig5_stacked_bar{mn_suffix}.{args.format}')
#         plot_fig5(all_labels, out,
#                   title=f'{prefix}Neuron Type Distribution Across Layers',
#                   dpi=args.dpi, fmt=args.format)

#     # ── Figure 6: High-confidence distribution ────────────────
#     if args.fig6 or args.all:
#         out = os.path.join(args.output_dir,
#                            f'fig6_high_confidence{mn_suffix}.{args.format}')
#         plot_fig6(all_labels, out, threshold=args.threshold,
#                   title=f'{prefix}High-Confidence Neurons '
#                         f'(p > {args.threshold:.0%}) Per Layer',
#                   dpi=args.dpi, fmt=args.format)

#     # ── Figure 7: Cross-model comparison ──────────────────────
#     if args.fig7 or args.all and args.data_dirs:
#         if not args.data_dirs:
#             if args.data_dir:
#                 # Single model — produce figure 7 with just one panel
#                 args.data_dirs = [args.data_dir]
#                 args.model_names = args.model_names or ['LLaVA-1.5 7B']
#             else:
#                 print('  SKIP fig7: no --data_dirs provided')

#         if args.data_dirs:
#             model_data = []
#             names = args.model_names or [f'Model {i+1}'
#                                          for i in range(len(args.data_dirs))]
#             # Per-model types and label files (fall back to singular args)
#             mtypes = args.model_types or [args.model_type] * len(args.data_dirs)
#             lfiles = args.label_files or [args.label_file] * len(args.data_dirs)
#             for dd, mt, lf in zip(args.data_dirs, mtypes, lfiles):
#                 print(f'  Loading {dd} for fig7... (type={mt}, labels={lf})')
#                 md = load_all_layers(
#                     dd, args.layer_start, args.layer_end,
#                     mt, lf)
#                 model_data.append(md)

#             out = os.path.join(args.output_dir,
#                                f'fig7_cross_model{mn_suffix}.{args.format}')
#             plot_fig7(model_data, names, out,
#                       title=f'{prefix}Neuron Type Distribution '
#                             f'Across Models',
#                       dpi=args.dpi, fmt=args.format)

#     # ── Line chart (bonus, matches existing style) ────────────
#     if args.line or args.all:
#         out = os.path.join(args.output_dir,
#                            f'line_chart{mn_suffix}.{args.format}')
#         plot_line_chart(all_labels, out,
#                         title=f'{prefix}Neuron Type Distribution '
#                               f'Across Layers',
#                         dpi=args.dpi, fmt=args.format)

#     # ── Phase 3 vs Phase 4 comparison ─────────────────────────
#     if args.compare or args.all:
#         # Determine permutation dir
#         if args.perm_dir:
#             perm_dir = args.perm_dir
#         elif args.data_dir:
#             # Default: replace /llm with /llm_permutation
#             perm_dir = args.data_dir.rstrip('/').replace(
#                 '/llm', '/llm_permutation')
#         else:
#             perm_dir = None

#         if perm_dir and os.path.isdir(perm_dir):
#             print(f'\n  Loading Phase 4 data from {perm_dir}')
#             phase4_labels = load_all_layers(
#                 perm_dir, args.layer_start, args.layer_end,
#                 args.model_type,
#                 'neuron_labels_permutation.json')

#             if phase4_labels:
#                 out = os.path.join(args.output_dir,
#                                    f'phase3_vs_phase4{mn_suffix}.{args.format}')
#                 plot_phase_comparison(all_labels, phase4_labels, out,
#                                      title=f'{prefix}Phase 3 (Xu) vs '
#                                            f'Phase 4 (Permutation Test)',
#                                      dpi=args.dpi, fmt=args.format)
#             else:
#                 print('  WARNING: No Phase 4 data loaded')
#         else:
#             if args.compare:
#                 print(f'  SKIP compare: permutation dir not found '
#                       f'({perm_dir})')

#     print(f'\n{"═"*60}')
#     print(f'Done. Figures saved to {args.output_dir}/')
#     print(f'{"═"*60}')


# if __name__ == '__main__':
#     main()


# # #!/usr/bin/env python3
# # """
# # plot_neuron_statistics.py — Generate Xu et al. Figures 5, 6, 7

# # Reads Phase 3 (neuron_labels.json) and/or Phase 4 (neuron_labels_permutation.json)
# # output from neuron_modality_statistical.py and produces publication-quality charts.

# # Figures:
# #     --fig5   Stacked bar chart: proportion of visual/text/multimodal/unknown
# #              neurons per layer.  Xu Figure 5 (Section 4.3).
# #     --fig6   High-confidence distribution: only neurons with max(pv,pt,pm,pu) > 0.8.
# #              Xu Figure 6 (Section 4.3).
# #     --fig7   Cross-model comparison: same as fig5 but overlays multiple models
# #              side-by-side.  Xu Figure 7 (Section 4.6).
# #     --all    Generate all three figures (default).

# # Data sources:
# #     Phase 3 (Xu-style argmax):
# #         {data_dir}/{layer_name}/neuron_labels.json
# #     Phase 4 (Permutation test):
# #         {data_dir_perm}/{layer_name}/neuron_labels_permutation.json

# # Both contain per-neuron dicts with keys:
# #     neuron_idx, label, pv, pt, pm, pu (Phase 3)
# #     neuron_idx, label, p_value, otsu_threshold, ... (Phase 4)

# # No GPU needed.

# # Usage:
# #     # All figures from Phase 3 data
# #     python plot_neuron_statistics.py \\
# #         --data_dir classification_xu/llava-1.5-7b/llm \\
# #         --all

# #     # Figure 5 only, using permutation test labels
# #     python plot_neuron_statistics.py \\
# #         --data_dir classification_xu/llava-1.5-7b/llm_permutation \\
# #         --label_file neuron_labels_permutation.json \\
# #         --fig5

# #     # Figure 6 from Phase 3 (high-confidence, needs pv/pt/pm/pu)
# #     python plot_neuron_statistics.py \\
# #         --data_dir classification_xu/llava-1.5-7b/llm \\
# #         --fig6

# #     # Figure 7: compare LLaVA-1.5 vs another model
# #     python plot_neuron_statistics.py \\
# #         --data_dirs classification_xu/llava-1.5-7b/llm \\
# #                     classification_xu/internvl-2.5-8b/llm \\
# #         --model_names "LLaVA-1.5 7B" "InternVL 2.5 8B" \\
# #         --fig7
# # """

# # import argparse
# # import json
# # import os

# # import matplotlib
# # matplotlib.use('Agg')                                               # non-interactive backend
# # import matplotlib.pyplot as plt
# # import matplotlib.ticker as mticker
# # import numpy as np


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 1 — Data loading
# # # ═══════════════════════════════════════════════════════════════════

# # # Colour scheme matching Xu paper conventions
# # COLORS = {
# #     'visual':     '#E74C3C',                                        # red
# #     'text':       '#3498DB',                                        # blue
# #     'multimodal': '#2ECC71',                                        # green
# #     'unknown':    '#95A5A6',                                        # grey
# # }

# # TYPE_ORDER = ['visual', 'text', 'multimodal', 'unknown']            # consistent legend order


# # def get_layer_dir_name(layer, model_type='llava-hf'):
# #     """Build the directory name for a layer.

# #     neuron_modality_statistical.py saves each layer's labels in a directory
# #     named after the full hook target, e.g.:
# #         model.language_model.model.layers.0.mlp.act_fn   (HF)
# #         model.layers.0.mlp.act_fn                        (original)

# #     Args:
# #         layer:      int — layer index (0-31)
# #         model_type: str — 'llava-hf' or 'llava-liuhaotian'

# #     Returns:
# #         str — directory name for this layer
# #     """
# #     if model_type == 'llava-hf':
# #         return f'model.language_model.model.layers.{layer}.mlp.act_fn'
# #     elif model_type == 'internvl':
# #         return f'language_model.model.layers.{layer}.feed_forward.act_fn'
# #     elif model_type in ('qwen2vl', 'llava-ov'):
# #         return f'model.language_model.layers.{layer}.mlp.act_fn'
# #     else:
# #         return f'model.layers.{layer}.mlp.act_fn'


# # def load_layer_labels(data_dir, layer, model_type='llava-hf',
# #                       label_file='neuron_labels.json'):
# #     """Load neuron labels for one layer.

# #     Args:
# #         data_dir:   str — base directory (e.g. classification_xu/llava-1.5-7b/llm)
# #         layer:      int — layer index
# #         model_type: str — 'llava-hf' or 'llava-liuhaotian'
# #         label_file: str — filename ('neuron_labels.json' or
# #                           'neuron_labels_permutation.json')

# #     Returns:
# #         list[dict] — one dict per neuron with at minimum 'label' key,
# #                      and 'pv','pt','pm','pu' for Phase 3 data
# #     """
# #     layer_dir = get_layer_dir_name(layer, model_type)
# #     path = os.path.join(data_dir, layer_dir, label_file)
# #     with open(path) as f:
# #         return json.load(f)


# # def load_all_layers(data_dir, layer_start=0, layer_end=31,
# #                     model_type='llava-hf', label_file='neuron_labels.json'):
# #     """Load labels for all layers in range.

# #     Returns:
# #         dict — {layer_idx: list[dict]} mapping each layer to its neuron labels
# #     """
# #     all_labels = {}
# #     for layer in range(layer_start, layer_end + 1):
# #         try:
# #             labels = load_layer_labels(
# #                 data_dir, layer, model_type, label_file)
# #             all_labels[layer] = labels
# #         except FileNotFoundError:
# #             print(f'  WARNING: No data for layer {layer}, skipping')
# #     return all_labels


# # def compute_layer_counts(all_labels):
# #     """Count neurons per type per layer.

# #     Args:
# #         all_labels: dict — {layer: list[dict]} from load_all_layers

# #     Returns:
# #         layers:  sorted list of layer indices
# #         counts:  dict — {type: np.array of counts per layer}
# #         totals:  np.array — total neurons per layer
# #     """
# #     layers = sorted(all_labels.keys())
# #     counts = {t: np.zeros(len(layers), dtype=int) for t in TYPE_ORDER}

# #     for i, layer in enumerate(layers):
# #         for neuron in all_labels[layer]:
# #             lbl = neuron['label']
# #             if lbl in counts:
# #                 counts[lbl][i] += 1
# #             else:
# #                 counts['unknown'][i] += 1                           # catch unexpected labels

# #     totals = sum(counts[t] for t in TYPE_ORDER)
# #     return layers, counts, totals


# # def compute_high_confidence_counts(all_labels, threshold=0.8):
# #     """Count high-confidence neurons per type per layer.

# #     A neuron is "high-confidence" if its dominant probability exceeds
# #     the threshold (Xu uses 0.8).  Phase 3 labels include pv, pt, pm, pu.
# #     Phase 4 labels don't have individual probabilities, so this only
# #     works with Phase 3 data.

# #     Args:
# #         all_labels: dict — {layer: list[dict]} from load_all_layers
# #         threshold:  float — minimum probability to count as high-confidence

# #     Returns:
# #         layers:  sorted list of layer indices
# #         counts:  dict — {type: np.array of high-confidence counts}
# #         totals:  np.array — total high-confidence neurons per layer
# #         n_neurons_per_layer: np.array — total neurons per layer (for %)
# #     """
# #     layers = sorted(all_labels.keys())
# #     counts = {t: np.zeros(len(layers), dtype=int) for t in TYPE_ORDER}
# #     n_neurons = np.zeros(len(layers), dtype=int)

# #     for i, layer in enumerate(layers):
# #         for neuron in all_labels[layer]:
# #             n_neurons[i] += 1
# #             # Get the dominant probability
# #             pv = neuron.get('pv', 0)
# #             pt = neuron.get('pt', 0)
# #             pm = neuron.get('pm', 0)
# #             pu = neuron.get('pu', 0)
# #             max_p = max(pv, pt, pm, pu)

# #             if max_p > threshold:
# #                 lbl = neuron['label']
# #                 if lbl in counts:
# #                     counts[lbl][i] += 1
# #                 else:
# #                     counts['unknown'][i] += 1

# #     totals = sum(counts[t] for t in TYPE_ORDER)
# #     return layers, counts, totals, n_neurons


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 2 — Figure 5: Stacked bar chart (neuron type proportions)
# # # ═══════════════════════════════════════════════════════════════════

# # def plot_fig5(all_labels, output_path, title='', dpi=200, fmt='png'):
# #     """Generate Figure 5: stacked bar chart of neuron type proportions.

# #     Each layer is one bar, stacked to 100%. Four segments correspond to
# #     visual (red), text (blue), multimodal (green), unknown (grey).

# #     Args:
# #         all_labels: dict — {layer: list[dict]} from load_all_layers
# #         output_path: str — file to save
# #         title: str — chart title (default auto-generated)
# #         dpi: int — resolution
# #         fmt: str — file format
# #     """
# #     layers, counts, totals = compute_layer_counts(all_labels)

# #     # Convert to percentages
# #     pcts = {}
# #     for t in TYPE_ORDER:
# #         pcts[t] = np.where(totals > 0,
# #                            counts[t] / totals * 100, 0)            # avoid division by zero

# #     # ── Plot ──────────────────────────────────────────────────
# #     fig, ax = plt.subplots(figsize=(14, 5))

# #     x = np.arange(len(layers))
# #     bar_width = 0.8
# #     bottom = np.zeros(len(layers))

# #     for t in TYPE_ORDER:
# #         ax.bar(x, pcts[t], bar_width, bottom=bottom,
# #                color=COLORS[t], label=t.capitalize(),
# #                edgecolor='white', linewidth=0.3)
# #         bottom += pcts[t]

# #     # Formatting
# #     ax.set_xlabel('Layer', fontsize=12)
# #     ax.set_ylabel('Proportion (%)', fontsize=12)
# #     ax.set_title(title or 'Neuron Type Distribution Across Layers',
# #                  fontsize=14, fontweight='bold', pad=10)
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(layers, fontsize=8)
# #     ax.set_ylim(0, 100)
# #     ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
# #     ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
# #     ax.grid(axis='y', alpha=0.3, linestyle='--')

# #     # Add neuron count annotation at bottom
# #     n_total = int(totals.sum())
# #     n_per = int(totals[0]) if len(totals) > 0 else 0
# #     ax.text(0.01, -0.12,
# #             f'Total: {n_total:,} neurons ({n_per:,} per layer)',
# #             fontsize=8, color='gray',
# #             transform=ax.transAxes)

# #     fig.tight_layout()
# #     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
# #                 facecolor='white')
# #     plt.close(fig)
# #     print(f'  Saved Figure 5: {output_path}')


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 3 — Figure 6: High-confidence neuron distribution
# # # ═══════════════════════════════════════════════════════════════════

# # def plot_fig6(all_labels, output_path, threshold=0.8,
# #               title='', dpi=200, fmt='png'):
# #     """Generate Figure 6: grouped bar chart of high-confidence neurons.

# #     Shows the count (or percentage) of neurons per layer whose dominant
# #     probability exceeds the threshold.  Bars grouped by type, side-by-side.

# #     Requires Phase 3 data (neuron_labels.json with pv/pt/pm/pu).

# #     Args:
# #         all_labels: dict — {layer: list[dict]}
# #         output_path: str — output file path
# #         threshold: float — confidence threshold (default 0.8)
# #         title: str — chart title
# #         dpi: int — figure resolution
# #         fmt: str — file format
# #     """
# #     # Verify Phase 3 data (has pv/pt/pm/pu)
# #     sample_layer = next(iter(all_labels.values()))
# #     if 'pv' not in sample_layer[0]:
# #         print('  ERROR: Figure 6 requires Phase 3 data with pv/pt/pm/pu. '
# #               'Use --label_file neuron_labels.json (not permutation).')
# #         return

# #     layers, counts, totals, n_neurons = compute_high_confidence_counts(
# #         all_labels, threshold)

# #     # Convert to percentages of total neurons per layer
# #     pcts = {}
# #     for t in TYPE_ORDER:
# #         pcts[t] = np.where(n_neurons > 0,
# #                            counts[t] / n_neurons * 100, 0)

# #     # ── Plot: grouped bar chart ───────────────────────────────
# #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
# #                                     height_ratios=[1, 1],
# #                                     sharex=True)

# #     x = np.arange(len(layers))
# #     n_types = len(TYPE_ORDER)
# #     group_width = 0.8
# #     bar_w = group_width / n_types

# #     # Top panel: counts
# #     for i, t in enumerate(TYPE_ORDER):
# #         offset = (i - n_types / 2 + 0.5) * bar_w
# #         ax1.bar(x + offset, counts[t], bar_w,
# #                 color=COLORS[t], label=t.capitalize(),
# #                 edgecolor='white', linewidth=0.3)

# #     ax1.set_ylabel('Count', fontsize=11)
# #     ax1.set_title(
# #         title or f'High-Confidence Neurons (p > {threshold:.0%}) Per Layer',
# #         fontsize=14, fontweight='bold', pad=10)
# #     ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
# #     ax1.grid(axis='y', alpha=0.3, linestyle='--')

# #     # Annotate total high-confidence %
# #     total_hc = int(totals.sum())
# #     total_all = int(n_neurons.sum())
# #     hc_pct = total_hc / total_all * 100 if total_all > 0 else 0
# #     ax1.text(0.01, 0.95,
# #              f'Total: {total_hc:,} / {total_all:,} '
# #              f'({hc_pct:.1f}%) neurons above {threshold:.0%}',
# #              fontsize=9, color='gray',
# #              transform=ax1.transAxes, va='top')

# #     # Bottom panel: percentages
# #     for i, t in enumerate(TYPE_ORDER):
# #         offset = (i - n_types / 2 + 0.5) * bar_w
# #         ax2.bar(x + offset, pcts[t], bar_w,
# #                 color=COLORS[t],
# #                 edgecolor='white', linewidth=0.3)

# #     ax2.set_xlabel('Layer', fontsize=12)
# #     ax2.set_ylabel('Proportion (%)', fontsize=11)
# #     ax2.set_xticks(x)
# #     ax2.set_xticklabels(layers, fontsize=8)
# #     ax2.grid(axis='y', alpha=0.3, linestyle='--')

# #     # Per-type summary annotation
# #     type_totals = {t: int(counts[t].sum()) for t in TYPE_ORDER}
# #     summary_parts = [f'{t.capitalize()}: {type_totals[t]:,} '
# #                      f'({type_totals[t]/total_all*100:.1f}%)'
# #                      for t in TYPE_ORDER if type_totals[t] > 0]
# #     ax2.text(0.01, -0.15,
# #              '  |  '.join(summary_parts),
# #              fontsize=8, color='gray',
# #              transform=ax2.transAxes)

# #     fig.tight_layout()
# #     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
# #                 facecolor='white')
# #     plt.close(fig)
# #     print(f'  Saved Figure 6: {output_path}')


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 4 — Figure 7: Cross-model comparison
# # # ═══════════════════════════════════════════════════════════════════

# # def plot_fig7(model_data, model_names, output_path,
# #               title='', dpi=200, fmt='png'):
# #     """Generate Figure 7: cross-model comparison of neuron type distributions.

# #     Plots one subplot per model, each showing the stacked area/line chart
# #     of neuron type proportions across layers.  Allows side-by-side visual
# #     comparison of how different VLMs distribute neuron types.

# #     Args:
# #         model_data:  list[dict] — each is {layer: list[dict]} from load_all_layers
# #         model_names: list[str] — display names for each model
# #         output_path: str — output file path
# #         title: str — overall figure title
# #         dpi: int — figure resolution
# #         fmt: str — file format
# #     """
# #     n_models = len(model_data)
# #     fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5),
# #                              sharey=True)
# #     if n_models == 1:
# #         axes = [axes]                                               # ensure iterable

# #     for idx, (all_labels, name) in enumerate(zip(model_data, model_names)):
# #         ax = axes[idx]
# #         layers, counts, totals = compute_layer_counts(all_labels)
# #         x = np.arange(len(layers))

# #         # Compute percentages
# #         pcts = {}
# #         for t in TYPE_ORDER:
# #             pcts[t] = np.where(totals > 0,
# #                                counts[t] / totals * 100, 0)

# #         # Stacked area (filled line chart) — matches Xu Figure 5/7 style
# #         bottom = np.zeros(len(layers))
# #         for t in TYPE_ORDER:
# #             ax.fill_between(x, bottom, bottom + pcts[t],
# #                             color=COLORS[t], alpha=0.7,
# #                             label=t.capitalize())
# #             ax.plot(x, bottom + pcts[t], color=COLORS[t],
# #                     linewidth=0.8, alpha=0.9)
# #             bottom += pcts[t]

# #         ax.set_xlabel('Layer', fontsize=11)
# #         ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
# #         ax.set_xticks(x[::2])                                      # every 2nd layer for readability
# #         ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)],
# #                            fontsize=8)
# #         ax.set_ylim(0, 100)
# #         ax.grid(axis='y', alpha=0.2, linestyle='--')

# #         # Add total neuron count
# #         n_total = int(totals.sum())
# #         ax.text(0.02, 0.02, f'{n_total:,} neurons',
# #                 fontsize=8, color='gray',
# #                 transform=ax.transAxes)

# #     axes[0].set_ylabel('Proportion (%)', fontsize=11)

# #     # Single legend for all subplots
# #     handles, labels = axes[0].get_legend_handles_labels()
# #     fig.legend(handles, labels, loc='upper center',
# #                ncol=len(TYPE_ORDER), fontsize=10,
# #                bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

# #     fig.suptitle(
# #         title or 'Neuron Type Distribution Across Models',
# #         fontsize=14, fontweight='bold', y=1.06)

# #     fig.tight_layout()
# #     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
# #                 facecolor='white')
# #     plt.close(fig)
# #     print(f'  Saved Figure 7: {output_path}')


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 5 — Bonus: Line chart (matches your existing plot style)
# # # ═══════════════════════════════════════════════════════════════════

# # def plot_line_chart(all_labels, output_path, title='', dpi=200, fmt='png'):
# #     """Generate a line chart of neuron type proportions per layer.

# #     This matches the style of your existing neuron_modality_per_layer.png.
# #     One coloured line per type, x-axis = layer, y-axis = percentage.

# #     Args:
# #         all_labels: dict — {layer: list[dict]}
# #         output_path: str — output file path
# #         title: str — chart title
# #         dpi: int — figure resolution
# #         fmt: str — file format
# #     """
# #     layers, counts, totals = compute_layer_counts(all_labels)
# #     x = np.array(layers)

# #     # Compute percentages
# #     pcts = {}
# #     for t in TYPE_ORDER:
# #         pcts[t] = np.where(totals > 0,
# #                            counts[t] / totals * 100, 0)

# #     fig, ax = plt.subplots(figsize=(12, 5))

# #     for t in TYPE_ORDER:
# #         ax.plot(x, pcts[t], '-o', color=COLORS[t],
# #                 label=t.capitalize(), markersize=4, linewidth=2)

# #     ax.set_xlabel('Layer', fontsize=12)
# #     ax.set_ylabel('Neurons (%)', fontsize=12)
# #     ax.set_title(title or 'Neuron Type Distribution Across Layers',
# #                  fontsize=14, fontweight='bold', pad=10)
# #     ax.set_xticks(x)
# #     ax.legend(fontsize=10, framealpha=0.9)
# #     ax.grid(alpha=0.3, linestyle='--')
# #     ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

# #     fig.tight_layout()
# #     fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
# #                 facecolor='white')
# #     plt.close(fig)
# #     print(f'  Saved line chart: {output_path}')


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 6 — Bonus: Side-by-side Phase 3 vs Phase 4 comparison
# # # ═══════════════════════════════════════════════════════════════════

# # def plot_phase_comparison(phase3_labels, phase4_labels, output_path,
# #                           title='', dpi=200, fmt='png'):
# #     """Generate a side-by-side comparison of Phase 3 (Xu) vs Phase 4 (Permutation).

# #     Two subplots: left = Phase 3 stacked bars, right = Phase 4 stacked bars.
# #     Allows direct visual comparison of the two classification methods.

# #     Args:
# #         phase3_labels: dict — {layer: list[dict]} Phase 3 data
# #         phase4_labels: dict — {layer: list[dict]} Phase 4 data
# #         output_path: str — output file path
# #         title: str — overall title
# #         dpi: int — figure resolution
# #         fmt: str — file format
# #     """
# #     plot_fig7(
# #         model_data=[phase3_labels, phase4_labels],
# #         model_names=['Phase 3 — Xu-style Classification',
# #                      'Phase 4 — Permutation Test'],
# #         output_path=output_path,
# #         title=title or 'Classification Method Comparison',
# #         dpi=dpi, fmt=fmt,
# #     )


# # # ═══════════════════════════════════════════════════════════════════
# # # Section 7 — Argument parsing and main
# # # ═══════════════════════════════════════════════════════════════════

# # def parse_args():
# #     p = argparse.ArgumentParser(
# #         description='Generate Xu et al. Figures 5, 6, 7 from classification data')

# #     # Data paths
# #     p.add_argument('--data_dir',
# #                    help='Classification output dir '
# #                         '(e.g. classification_xu/llava-1.5-7b/llm)')
# #     p.add_argument('--data_dirs', nargs='+',
# #                    help='Multiple data dirs for --fig7 cross-model comparison')
# #     p.add_argument('--model_names', nargs='+',
# #                    help='Display names for each model in --fig7')
# #     p.add_argument('--perm_dir',
# #                    help='Phase 4 permutation dir (for --compare). '
# #                         'Default: replaces /llm with /llm_permutation')

# #     # File names
# #     p.add_argument('--label_file', default='neuron_labels.json',
# #                    help='Label filename per layer '
# #                         '(neuron_labels.json or neuron_labels_permutation.json)')
# #     p.add_argument('--model_type', default='llava-hf',
# #                    choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
# #                    help='Model type (determines layer directory naming)')

# #     # Layer range
# #     p.add_argument('--layer_start', type=int, default=0)
# #     p.add_argument('--layer_end', type=int, default=31)

# #     # Figure selection
# #     p.add_argument('--fig5', action='store_true',
# #                    help='Stacked bar chart: neuron type proportions per layer')
# #     p.add_argument('--fig6', action='store_true',
# #                    help='High-confidence distribution (>80%% threshold)')
# #     p.add_argument('--fig7', action='store_true',
# #                    help='Cross-model comparison (requires --data_dirs)')
# #     p.add_argument('--line', action='store_true',
# #                    help='Line chart (matches existing plot style)')
# #     p.add_argument('--compare', action='store_true',
# #                    help='Side-by-side Phase 3 vs Phase 4 comparison')
# #     p.add_argument('--all', action='store_true',
# #                    help='Generate all figures (default if no fig specified)')

# #     # Figure 6 options
# #     p.add_argument('--threshold', type=float, default=0.8,
# #                    help='Confidence threshold for Figure 6 (default 0.8)')

# #     # Output
# #     p.add_argument('--output_dir', default='figure_outputs',
# #                    help='Directory for output figures')
# #     p.add_argument('--dpi', type=int, default=200)
# #     p.add_argument('--format', default='png',
# #                    choices=['png', 'pdf', 'svg'])
# #     p.add_argument('--title_prefix', default='',
# #                    help='Prefix for figure titles (e.g. model name)')
# #     p.add_argument('--model_name', default='',
# #                    help='Human-readable model name appended to filenames '
# #                         '(e.g. "llava-1.5-7b")')

# #     return p.parse_args()


# # def main():
# #     args = parse_args()

# #     # Default to --all if no specific figure requested
# #     if not any([args.fig5, args.fig6, args.fig7,
# #                 args.line, args.compare]):
# #         args.all = True

# #     os.makedirs(args.output_dir, exist_ok=True)

# #     prefix = args.title_prefix
# #     # Auto-set title prefix from model_name if not explicitly provided
# #     if not prefix and args.model_name:
# #         prefix = args.model_name
# #     if prefix and not prefix.endswith(' '):
# #         prefix += ' '

# #     # Filename suffix from model_name (e.g. "_llava-1.5-7b")
# #     mn_suffix = f'_{args.model_name}' if args.model_name else ''

# #     # ── Load primary data ─────────────────────────────────────
# #     if args.data_dir:
# #         print(f'Loading data from {args.data_dir}')
# #         print(f'  Label file:  {args.label_file}')
# #         print(f'  Model type:  {args.model_type}')
# #         print(f'  Layers:      {args.layer_start}-{args.layer_end}')

# #         all_labels = load_all_layers(
# #             args.data_dir, args.layer_start, args.layer_end,
# #             args.model_type, args.label_file)

# #         if not all_labels:
# #             print('ERROR: No data loaded. Check --data_dir and --model_type.')
# #             return

# #         n_layers = len(all_labels)
# #         n_neurons = sum(len(v) for v in all_labels.values())
# #         print(f'  Loaded: {n_layers} layers, {n_neurons:,} neurons')

# #     # ── Figure 5: Stacked bar chart ───────────────────────────
# #     if args.fig5 or args.all:
# #         out = os.path.join(args.output_dir,
# #                            f'fig5_stacked_bar{mn_suffix}.{args.format}')
# #         plot_fig5(all_labels, out,
# #                   title=f'{prefix}Neuron Type Distribution Across Layers',
# #                   dpi=args.dpi, fmt=args.format)

# #     # ── Figure 6: High-confidence distribution ────────────────
# #     if args.fig6 or args.all:
# #         out = os.path.join(args.output_dir,
# #                            f'fig6_high_confidence{mn_suffix}.{args.format}')
# #         plot_fig6(all_labels, out, threshold=args.threshold,
# #                   title=f'{prefix}High-Confidence Neurons '
# #                         f'(p > {args.threshold:.0%}) Per Layer',
# #                   dpi=args.dpi, fmt=args.format)

# #     # ── Figure 7: Cross-model comparison ──────────────────────
# #     if args.fig7 or args.all and args.data_dirs:
# #         if not args.data_dirs:
# #             if args.data_dir:
# #                 # Single model — produce figure 7 with just one panel
# #                 args.data_dirs = [args.data_dir]
# #                 args.model_names = args.model_names or ['LLaVA-1.5 7B']
# #             else:
# #                 print('  SKIP fig7: no --data_dirs provided')

# #         if args.data_dirs:
# #             model_data = []
# #             names = args.model_names or [f'Model {i+1}'
# #                                          for i in range(len(args.data_dirs))]
# #             for dd in args.data_dirs:
# #                 print(f'  Loading {dd} for fig7...')
# #                 md = load_all_layers(
# #                     dd, args.layer_start, args.layer_end,
# #                     args.model_type, args.label_file)
# #                 model_data.append(md)

# #             out = os.path.join(args.output_dir,
# #                                f'fig7_cross_model{mn_suffix}.{args.format}')
# #             plot_fig7(model_data, names, out,
# #                       title=f'{prefix}Neuron Type Distribution '
# #                             f'Across Models',
# #                       dpi=args.dpi, fmt=args.format)

# #     # ── Line chart (bonus, matches existing style) ────────────
# #     if args.line or args.all:
# #         out = os.path.join(args.output_dir,
# #                            f'line_chart{mn_suffix}.{args.format}')
# #         plot_line_chart(all_labels, out,
# #                         title=f'{prefix}Neuron Type Distribution '
# #                               f'Across Layers',
# #                         dpi=args.dpi, fmt=args.format)

# #     # ── Phase 3 vs Phase 4 comparison ─────────────────────────
# #     if args.compare or args.all:
# #         # Determine permutation dir
# #         if args.perm_dir:
# #             perm_dir = args.perm_dir
# #         elif args.data_dir:
# #             # Default: replace /llm with /llm_permutation
# #             perm_dir = args.data_dir.rstrip('/').replace(
# #                 '/llm', '/llm_permutation')
# #         else:
# #             perm_dir = None

# #         if perm_dir and os.path.isdir(perm_dir):
# #             print(f'\n  Loading Phase 4 data from {perm_dir}')
# #             phase4_labels = load_all_layers(
# #                 perm_dir, args.layer_start, args.layer_end,
# #                 args.model_type,
# #                 'neuron_labels_permutation.json')

# #             if phase4_labels:
# #                 out = os.path.join(args.output_dir,
# #                                    f'phase3_vs_phase4{mn_suffix}.{args.format}')
# #                 plot_phase_comparison(all_labels, phase4_labels, out,
# #                                      title=f'{prefix}Phase 3 (Xu) vs '
# #                                            f'Phase 4 (Permutation Test)',
# #                                      dpi=args.dpi, fmt=args.format)
# #             else:
# #                 print('  WARNING: No Phase 4 data loaded')
# #         else:
# #             if args.compare:
# #                 print(f'  SKIP compare: permutation dir not found '
# #                       f'({perm_dir})')

# #     print(f'\n{"═"*60}')
# #     print(f'Done. Figures saved to {args.output_dir}/')
# #     print(f'{"═"*60}')


# # if __name__ == '__main__':
# #     main()