"""
generate_specialist_figure.py

Reads per_layer_stats from classification_stats_all.json (FT) and
permutation_stats_all.json (PMBT) for each model, then produces a
4x2 grid of line plots showing per-layer specialist neuron counts
for visual, text, and multimodal neurons.

A dotted horizontal baseline marks 3% of total neurons divided
equally across layers, making it easy to see which layers are
above or below the uniform-distribution expectation.

Usage (standalone, pointing at local flat-dir JSON copies):
    python generate_specialist_figure.py --stats-dir /path/to/stats --output-dir /path/to/out

Usage (via run_pipeline.sh / classify-dir layout):
    python generate_specialist_figure.py \
        --classify-dir results/3-classify/full \
        --output-dir results/14-layer-plots/full

JSON layout expected:
  classify-dir mode:  <classify-dir>/<model>/llm_fixed_threshold/classification_stats_all.json
                      <classify-dir>/<model>/llm_permutation/permutation_stats_all.json
  stats-dir mode:     <stats-dir>/classification_stats_all_<slug>.json
                      <stats-dir>/permutation_stats_all_<slug>.json

Outputs:
    fig_specialist_layers.pdf
    fig_specialist_layers.png
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate per-layer specialist neuron count figure (FT vs PMBT)."
)
group = parser.add_mutually_exclusive_group()
# Pipeline layout: JSONs live inside the classify results tree
group.add_argument(
    "--classify-dir", default=None,
    help="Path to results/3-classify/<mode>. "
         "Looks for <model>/llm_fixed_threshold/classification_stats_all.json "
         "and <model>/llm_permutation/permutation_stats_all.json"
)
# Standalone / legacy layout: flat directory with slugged filenames
group.add_argument(
    "--stats-dir", default=None,
    help="Flat directory containing classification_stats_all_<slug>.json "
         "and permutation_stats_all_<slug>.json files."
)
parser.add_argument(
    "--output-dir", default="outputs",
    help="Directory to write fig_specialist_layers.{pdf,png}. Created if absent."
)
parser.add_argument(
    "--dpi", type=int, default=200,
    help="Output DPI for PNG (default: 200)."
)
parser.add_argument(
    "--pct", type=float, default=3.0,
    help="Specialist threshold as percentage of total neurons (default: 3.0)."
)
args = parser.parse_args()

# ── model registry ──────────────────────────────────────────────────────────
# Order matches Fig. 8 left-to-right column order:
#   LLaVA-1.5 → LLaVA-OV → Qwen → InternVL
MODEL_REGISTRY = [
    # (model_id,             display_label,    flat-dir slug)
    ("llava-1.5-7b",        "LLaVA-1.5-7b",   "llava-1_5-7b"),
    ("llava-onevision-7b",  "LLaVA-OV-7B",    "llava-onevision-7b"),
    ("qwen2.5-vl-7b",       "Qwen2.5-VL-7B",  "qwen2_5-vl-7b"),
    ("internvl2.5-8b",      "InternVL2.5-8B", "internvl2_5-8b"),
]

# Neuron types to plot (unknown excluded — near-zero under PMBT, clutters plot)
TYPES  = ["visual", "text", "multimodal"]
LABELS = ["Visual", "Text", "Multimodal"]
# Colours consistent with Fig. 8 stacked bars and trend plots
COLORS = ["#d62728", "#1f77b4", "#2ca02c"]
MARKERS = ["o", "s", "^"]


# ── path resolution ─────────────────────────────────────────────────────────
def resolve_paths(classify_dir, stats_dir):
    """
    For each model in MODEL_REGISTRY, locate the FT and PMBT JSON files.
    Returns list of (model_id, display_label, ft_path, pmbt_path) for
    models where both files exist. Prints a status line for each model.
    """
    found = []
    for model_id, label, slug in MODEL_REGISTRY:
        if classify_dir:
            # Pipeline tree layout
            ft_path   = os.path.join(
                classify_dir, model_id,
                "llm_fixed_threshold", "classification_stats_all.json"
            )
            pmbt_path = os.path.join(
                classify_dir, model_id,
                "llm_permutation", "permutation_stats_all.json"
            )
        else:
            # Flat directory layout
            ft_path   = os.path.join(
                stats_dir, f"classification_stats_all_{slug}.json"
            )
            pmbt_path = os.path.join(
                stats_dir, f"permutation_stats_all_{slug}.json"
            )

        ft_ok   = os.path.isfile(ft_path)
        pmbt_ok = os.path.isfile(pmbt_path)

        if ft_ok and pmbt_ok:
            found.append((model_id, label, ft_path, pmbt_path))
            print(f"  ✓ {model_id}")
        else:
            missing = []
            if not ft_ok:   missing.append(f"FT={ft_path}")
            if not pmbt_ok: missing.append(f"PMBT={pmbt_path}")
            print(f"  ✗ {model_id} — skipping ({', '.join(missing)})",
                  file=sys.stderr)

    if not found:
        print(
            "ERROR: no models found. "
            "Pass --classify-dir or --stats-dir with valid paths.",
            file=sys.stderr
        )
        sys.exit(1)
    return found


# ── default path resolution ─────────────────────────────────────────────────
if args.classify_dir is None and args.stats_dir is None:
    default = "results/3-classify/full"
    if os.path.isdir(default):
        args.classify_dir = default
    else:
        print(
            "ERROR: no --classify-dir or --stats-dir provided, "
            "and default path 'results/3-classify/full' not found.",
            file=sys.stderr
        )
        sys.exit(1)

print("Resolving model data paths...")
MODELS = resolve_paths(args.classify_dir, args.stats_dir)
os.makedirs(args.output_dir, exist_ok=True)


# ── helpers ─────────────────────────────────────────────────────────────────
def load_stats(filepath):
    """
    Load a classification or permutation stats JSON.
    Returns (per_layer dict, total_neurons int, neurons_per_layer int).
    per_layer keys are integers (layer index), values are {type: count}.
    """
    with open(filepath) as f:
        d = json.load(f)
    # per_layer_stats keys are strings; convert to int
    per_layer = {int(k): v for k, v in d["per_layer_stats"].items()}
    total     = sum(d["stats"].values())          # total neurons across all types
    npl       = d.get("neurons_per_layer", None)  # neurons per layer
    return per_layer, total, npl


# ── plotting ─────────────────────────────────────────────────────────────────
n_models = len(MODELS)

fig, axes = plt.subplots(
    nrows=n_models,
    ncols=2,
    figsize=(12, 3.5 * n_models),
    sharex=False,   # models have different layer counts
    sharey=False,   # neuron counts differ across models (different sizes)
    constrained_layout=True,
)

# Ensure axes is always 2-D even for a single model
if n_models == 1:
    axes = axes[np.newaxis, :]

for row, (model_id, model_label, ft_path, pmbt_path) in enumerate(MODELS):

    # Load both methods
    pl_ft,   total_ft,   npl_ft   = load_stats(ft_path)
    pl_pmbt, total_pmbt, npl_pmbt = load_stats(pmbt_path)

    # Use FT total for the baseline (both methods classify the same neurons)
    total    = total_ft
    n_layers = len(pl_ft)
    # Baseline: if specialist neurons were uniformly spread, each layer would
    # contain (pct% of total) / n_layers specialists
    baseline = (args.pct / 100.0 * total) / n_layers

    for col, (pl, method_label) in enumerate([
        (pl_ft,   "FT"),
        (pl_pmbt, "PMBT"),
    ]):
        ax     = axes[row, col]
        layers = sorted(pl.keys())        # sorted layer indices

        # Plot one line per neuron type
        for t, lbl, color, mrk in zip(TYPES, LABELS, COLORS, MARKERS):
            counts = np.array([pl[l][t] for l in layers])  # count per layer
            ax.plot(
                layers, counts,
                color=color, marker=mrk,
                markersize=3.5, linewidth=1.5,
                label=lbl
            )

        # Dotted baseline: equal share of the top-pct% threshold
        ax.axhline(
            baseline,
            color="black", linestyle=":", linewidth=0.9,
            label=f"{args.pct:.0f}%/layer ≈ {int(baseline):,}"
        )

        # ── axis formatting ────────────────────────────────────────────────
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
        ax.set_xticks(layers[::4])                          # label every 4th layer
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        # Format y-axis as integer thousands (e.g. 10,000)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3)

        # Column header on top row only
        if row == 0:
            ax.set_title(method_label, fontsize=13, fontweight="bold", pad=6)

        # Model label as y-axis label on left column only
        if col == 0:
            ax.set_ylabel(model_label, fontsize=10, fontweight="bold")

        # x-axis label on bottom row only
        if row == n_models - 1:
            ax.set_xlabel("Layer", fontsize=9)

# Shared legend below the entire grid
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, lbls,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    fontsize=10,
    frameon=True,
    edgecolor="grey"
)

fig.suptitle(
    f"Per-Layer Specialist Neuron Counts — FT vs PMBT  "
    f"(dotted line = {args.pct:.0f}%/layer baseline)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

# ── save ─────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out_path = os.path.join(args.output_dir, f"fig_specialist_layers.{ext}")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
print("Done.")
