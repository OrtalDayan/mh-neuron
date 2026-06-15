"""
generate_layer_tables_plots.py

Reads per_layer_stats from classification_stats_all_*.json (FT) and
permutation_stats_all_*.json (PMBT), then produces:
  1. A 4x2 grid of line plots (trend figure) saved as PDF + PNG
  2. LaTeX longtable source for each model x method combination
     (all tables written to a single .tex snippet file)

Usage:
    python generate_layer_tables_plots.py

Outputs (written to /home/claude/outputs/):
    fig_layer_trends.pdf
    fig_layer_trends.png
    supp_layer_tables.tex   <- paste into supplementary.tex
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "results/3-classify/full"
OUT_DIR  = "results/14-layer-plots/full"
os.makedirs(OUT_DIR, exist_ok=True)

# Model order: LLaVA-1.5 → LLaVA-OV → Qwen → InternVL
MODELS = [
    ("llava-1.5-7b",      "LLaVA-1.5-7b",       "classification_stats_all_llava-1_5-7b.json",      "permutation_stats_all_llava-1_5-7b.json"),
    ("llava-onevision-7b","LLaVA-OV-7B",         "classification_stats_all_llava-onevision-7b.json", "permutation_stats_all_llava-onevision-7b.json"),
    ("qwen2.5-vl-7b",     "Qwen2.5-VL-7B",       "classification_stats_all_qwen2_5-vl-7b.json",     "permutation_stats_all_qwen2_5-vl-7b.json"),
    ("internvl2.5-8b",    "InternVL2.5-8B",      "classification_stats_all_internvl2_5-8b.json",     "permutation_stats_all_internvl2_5-8b.json"),
]

TYPES   = ["visual", "text", "multimodal", "unknown"]
LABELS  = ["Visual", "Text", "Multimodal", "Unknown"]
# Colours consistent with Fig. 8 stacked bars: red, blue, green, grey
COLORS  = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]
LSTYLES = ["-", "-", "-", "--"]   # unknown dashed to de-emphasise


# ── helpers ────────────────────────────────────────────────────────────────
def load_per_layer(filepath):
    """Return dict {layer_int: {type: count}} sorted by layer."""
    with open(filepath) as f:
        d = json.load(f)
    raw = d["per_layer_stats"]          # keys are strings "0","1",...
    return {int(k): v for k, v in raw.items()}


def to_pct(per_layer):
    """Convert counts to proportions (%) per layer. Returns {layer: {type: pct}}."""
    out = {}
    for layer, counts in sorted(per_layer.items()):
        total = sum(counts[t] for t in TYPES)
        out[layer] = {t: 100.0 * counts[t] / total if total > 0 else 0.0
                      for t in TYPES}
    return out


# ── 1. TREND PLOTS ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=4, ncols=2,
    figsize=(12, 14),
    sharex=False,       # models have different layer counts
    sharey=True,        # all y-axes 0-100%
    constrained_layout=True,
)

for row, (model_id, model_label, ft_file, pmbt_file) in enumerate(MODELS):
    for col, (fname, method_label) in enumerate([
        (ft_file,   "FT"),
        (pmbt_file, "PMBT"),
    ]):
        ax = axes[row, col]
        per_layer = load_per_layer(os.path.join(DATA_DIR, fname))
        pct       = to_pct(per_layer)

        layers = sorted(pct.keys())
        x      = np.array(layers)

        for t, label, color, ls in zip(TYPES, LABELS, COLORS, LSTYLES):
            y = np.array([pct[l][t] for l in layers])
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.6,
                    label=label, marker="o", markersize=2.5, markeredgewidth=0)

        # ── axis formatting ────────────────────────────────────────────────
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(layers[::4])              # every 4th layer label
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3)

        # column title on top row only
        if row == 0:
            ax.set_title(method_label, fontsize=13, fontweight="bold", pad=6)

        # model label on left column only (as y-label)
        if col == 0:
            ax.set_ylabel(model_label, fontsize=10, fontweight="bold")

        if row == len(MODELS) - 1:
            ax.set_xlabel("Layer", fontsize=9)

# shared legend below the grid
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.02),
           ncol=4,
           fontsize=10,
           frameon=True,
           edgecolor="grey")

fig.suptitle("Per-Layer Neuron Type Proportions — FT vs PMBT",
             fontsize=13, fontweight="bold", y=1.01)

for ext in ["pdf", "png"]:
    out_path = os.path.join(OUT_DIR, f"fig_layer_trends.{ext}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)


# ── 2. LaTeX TABLES ────────────────────────────────────────────────────────
def fmt(v):
    """Format percentage to 1 decimal place."""
    return f"{v:.1f}"


def make_longtable(model_label, per_layer_ft, per_layer_pmbt):
    """Return a LaTeX longtable string for one model (FT + PMBT side by side)."""
    pct_ft   = to_pct(per_layer_ft)
    pct_pmbt = to_pct(per_layer_pmbt)
    all_layers = sorted(set(pct_ft) | set(pct_pmbt))

    lines = []
    lines.append(r"\begin{longtable}{@{}r rrrr c rrrr@{}}")
    lines.append(r"  \caption{Per-layer neuron type proportions (\%) for " + model_label + r".}")
    lines.append(r"  \label{tab:layer-" + model_label.lower().replace(" ", "-").replace(".", "") + r"} \\")
    lines.append(r"  \toprule")
    lines.append(r"  & \multicolumn{4}{c}{\textbf{FT}} && \multicolumn{4}{c}{\textbf{PMBT}} \\")
    lines.append(r"  \cmidrule(lr){2-5}\cmidrule(lr){7-10}")
    lines.append(r"  \textbf{Layer} & Vis. & Text & Multi. & Unk. && Vis. & Text & Multi. & Unk. \\")
    lines.append(r"  \midrule")
    lines.append(r"  \endfirsthead")
    # repeated header on subsequent pages
    lines.append(r"  \toprule")
    lines.append(r"  & \multicolumn{4}{c}{\textbf{FT}} && \multicolumn{4}{c}{\textbf{PMBT}} \\")
    lines.append(r"  \cmidrule(lr){2-5}\cmidrule(lr){7-10}")
    lines.append(r"  \textbf{Layer} & Vis. & Text & Multi. & Unk. && Vis. & Text & Multi. & Unk. \\")
    lines.append(r"  \midrule")
    lines.append(r"  \endhead")
    lines.append(r"  \midrule \multicolumn{10}{r}{\textit{continued on next page}} \\")
    lines.append(r"  \endfoot")
    lines.append(r"  \bottomrule")
    lines.append(r"  \endlastfoot")

    for l in all_layers:
        ft   = pct_ft.get(l,   {t: 0.0 for t in TYPES})
        pmbt = pct_pmbt.get(l, {t: 0.0 for t in TYPES})
        lines.append(
            f"  {l} & "
            f"{fmt(ft['visual'])} & {fmt(ft['text'])} & "
            f"{fmt(ft['multimodal'])} & {fmt(ft['unknown'])} && "
            f"{fmt(pmbt['visual'])} & {fmt(pmbt['text'])} & "
            f"{fmt(pmbt['multimodal'])} & {fmt(pmbt['unknown'])} \\\\"
        )

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ── write all four tables to one .tex snippet ──────────────────────────────
tex_out = os.path.join(OUT_DIR, "supp_layer_tables.tex")
with open(tex_out, "w") as f:
    # preamble comment
    f.write("% ============================================================\n")
    f.write("% Auto-generated by generate_layer_tables_plots.py\n")
    f.write("% Paste into supplementary.tex inside Sec. B (layer-wise section)\n")
    f.write("% Requires: \\usepackage{longtable, booktabs} in preamble\n")
    f.write("% ============================================================\n\n")

    for model_id, model_label, ft_file, pmbt_file in MODELS:
        pl_ft   = load_per_layer(os.path.join(DATA_DIR, ft_file))
        pl_pmbt = load_per_layer(os.path.join(DATA_DIR, pmbt_file))
        table   = make_longtable(model_label, pl_ft, pl_pmbt)
        f.write(f"% ── {model_label} ──\n")
        f.write(table + "\n\n")

print(f"Saved: {tex_out}")
print("Done.")
