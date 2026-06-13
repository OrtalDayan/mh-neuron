"""
generate_confidence_figures.py

Generates the classification confidence and statistical validity figures
for Supplementary Section D. Reads per-neuron label files directly:

  FT   : <classify-dir>/<model>/llm_fixed_threshold/<layer>/neuron_labels.json
           fields per neuron: neuron_idx, label, pv, pt, pm, pu
  PMBT : <classify-dir>/<model>/llm_permutation/<layer>/neuron_labels_permutation.json
           fields per neuron: neuron_idx, label, p_value, observed_rate_diff

Produces three multi-panel figures, one PDF + PNG each:

  fig_D1_ft_probs.pdf/.png
      4-model × 4-type grid of FT probability (pv/pt/pm/pu) histograms.
      Shows rightward-skew of visual/text vs. broad multimodal distributions.

  fig_D2_pmbt_pvalues.pdf/.png
      4-model × 2-panel grid: p-value histogram + observed_rate_diff histogram.
      Shows bimodal p-value distribution and near-zero D concentration.

  fig_D3_confidence_by_label.pdf/.png
      4-model strip: per-label confidence (1-p for PMBT, max(pv,pt,pm) for FT)
      shown as overlaid KDE curves — visual/text high, multimodal broad.

Usage (pipeline layout):
    python generate_confidence_figures.py \\
        --classify-dir results/3-classify/full \\
        --output-dir   results/14-layer-plots/full

Usage (standalone with flat stats dir, not applicable — this script needs
individual neuron_labels files, not the aggregated stats JSONs):
    python generate_confidence_figures.py \\
        --classify-dir /path/to/results/3-classify/full \\
        --output-dir   /path/to/output
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate Sec D confidence distribution figures for all models."
)
parser.add_argument(
    "--classify-dir", required=True,
    help="Path to results/3-classify/<mode>. "
         "Expects <model>/llm_fixed_threshold/<layer>/neuron_labels.json "
         "and <model>/llm_permutation/<layer>/neuron_labels_permutation.json"
)
parser.add_argument(
    "--output-dir", default="outputs",
    help="Directory to write figures. Created if absent."
)
parser.add_argument(
    "--dpi", type=int, default=200,
    help="Output DPI for PNG (default: 200)."
)
parser.add_argument(
    "--max-neurons", type=int, default=None,
    help="Cap neurons loaded per model (for fast testing, e.g. --max-neurons 50000)."
)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ── model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    ("llava-1.5-7b",        "LLaVA-1.5-7b"),
    ("llava-onevision-7b",  "LLaVA-OV-7B"),
    ("qwen2.5-vl-7b",       "Qwen2.5-VL-7B"),
    ("internvl2.5-8b",      "InternVL2.5-8B"),
]

# ── colour palette consistent with other figures ────────────────────────────
TYPE_COLORS = {
    "visual":     "#d62728",
    "text":       "#1f77b4",
    "multimodal": "#2ca02c",
    "unknown":    "#7f7f7f",
}

# ── helpers ─────────────────────────────────────────────────────────────────
def iter_layer_dirs(base_dir):
    """
    Yield (layer_int, layer_dir) for all numeric subdirectories of base_dir,
    sorted by layer index.
    """
    if not os.path.isdir(base_dir):
        return
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            # layer dirs are named like "layer_0", "layer_00", "0", etc.
            # Extract trailing integer
            num_str = name.split("_")[-1].lstrip("0") or "0"
            try:
                yield int(num_str), full
            except ValueError:
                continue


def load_ft_neurons(ft_base, max_neurons=None):
    """
    Load all FT neuron dicts from neuron_labels.json files under ft_base.
    Returns list of dicts with keys: label, pv, pt, pm, pu.
    Stops early if max_neurons is set.
    """
    neurons = []
    for _, layer_dir in iter_layer_dirs(ft_base):
        fpath = os.path.join(layer_dir, "neuron_labels.json")
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        # data is either a list of dicts or a dict keyed by neuron_idx
        items = data if isinstance(data, list) else list(data.values())
        for n in items:
            neurons.append({
                "label": n.get("label", "unknown"),
                "pv":    float(n.get("pv", 0.0)),
                "pt":    float(n.get("pt", 0.0)),
                "pm":    float(n.get("pm", 0.0)),
                "pu":    float(n.get("pu", 0.0)),
            })
            if max_neurons and len(neurons) >= max_neurons:
                return neurons
    return neurons


def load_pmbt_neurons(pmbt_base, max_neurons=None):
    """
    Load all PMBT neuron dicts from neuron_labels_permutation.json files.
    Returns list of dicts with keys: label, p_value, observed_rate_diff.
    """
    neurons = []
    for _, layer_dir in iter_layer_dirs(pmbt_base):
        fpath = os.path.join(layer_dir, "neuron_labels_permutation.json")
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        for n in items:
            neurons.append({
                "label":               n.get("label", "unknown"),
                "p_value":             float(n.get("p_value", 1.0)),
                "observed_rate_diff":  float(n.get("observed_rate_diff", 0.0)),
            })
            if max_neurons and len(neurons) >= max_neurons:
                return neurons
    return neurons


def split_by_label(neurons, key):
    """
    Split a list of neuron dicts by 'label' field.
    Returns dict {label: np.array of key values}.
    """
    out = {"visual": [], "text": [], "multimodal": [], "unknown": []}
    for n in neurons:
        lbl = n["label"]
        if lbl in out:
            out[lbl].append(n[key])
    return {k: np.array(v) for k, v in out.items()}


# ── load all models ─────────────────────────────────────────────────────────
print("Loading neuron label files...")
model_data = []   # list of (label, ft_neurons, pmbt_neurons)

for model_id, model_label in MODEL_REGISTRY:
    ft_base   = os.path.join(args.classify_dir, model_id, "llm_fixed_threshold")
    pmbt_base = os.path.join(args.classify_dir, model_id, "llm_permutation")

    if not os.path.isdir(ft_base) or not os.path.isdir(pmbt_base):
        print(f"  ✗ {model_id} — missing dirs, skipping", file=sys.stderr)
        continue

    ft_neurons   = load_ft_neurons(ft_base,   max_neurons=args.max_neurons)
    pmbt_neurons = load_pmbt_neurons(pmbt_base, max_neurons=args.max_neurons)

    if not ft_neurons or not pmbt_neurons:
        print(f"  ✗ {model_id} — no neuron files found, skipping", file=sys.stderr)
        continue

    print(f"  ✓ {model_id}: FT={len(ft_neurons):,}  PMBT={len(pmbt_neurons):,}")
    model_data.append((model_label, ft_neurons, pmbt_neurons))

if not model_data:
    print("ERROR: no models loaded. Check --classify-dir path.", file=sys.stderr)
    sys.exit(1)

n_models = len(model_data)
BINS = 50   # histogram bins


# ════════════════════════════════════════════════════════════════════════════
# Figure D1: FT probability distributions (pv, pt, pm, pu) per model
# 4-row × 4-col grid;  rows = models,  cols = probability type
# ════════════════════════════════════════════════════════════════════════════
FT_PROB_TYPES = [
    ("pv", "Visual ($p_v$)",     TYPE_COLORS["visual"]),
    ("pt", "Text ($p_t$)",       TYPE_COLORS["text"]),
    ("pm", "Multimodal ($p_m$)", TYPE_COLORS["multimodal"]),
    ("pu", "Unknown ($p_u$)",    TYPE_COLORS["unknown"]),
]

fig1, axes1 = plt.subplots(
    n_models, 4,
    figsize=(14, 3.2 * n_models),
    constrained_layout=True,
)
if n_models == 1:
    axes1 = axes1[np.newaxis, :]

for row, (model_label, ft_neurons, _) in enumerate(model_data):
    for col, (field, col_title, color) in enumerate(FT_PROB_TYPES):
        ax   = axes1[row, col]
        vals = np.array([n[field] for n in ft_neurons])

        ax.hist(vals, bins=BINS, range=(0, 1), color=color, alpha=0.75,
                edgecolor="white", linewidth=0.3)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=8)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

        if row == 0:
            ax.set_title(col_title, fontsize=11, fontweight="bold", pad=5)
        if col == 0:
            ax.set_ylabel(model_label, fontsize=9, fontweight="bold")

fig1.suptitle(
    "Fixed-Threshold (FT) Probability Distributions per Neuron Type",
    fontsize=12, fontweight="bold", y=1.01
)

for ext in ["pdf", "png"]:
    p = os.path.join(args.output_dir, f"fig_D1_ft_probs.{ext}")
    fig1.savefig(p, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {p}")
plt.close(fig1)


# ════════════════════════════════════════════════════════════════════════════
# Figure D2: PMBT p-value + observed_rate_diff histograms
# 4-row × 2-col grid;  col 0 = p-value,  col 1 = rate diff D
# ════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(
    n_models, 2,
    figsize=(10, 3.2 * n_models),
    constrained_layout=True,
)
if n_models == 1:
    axes2 = axes2[np.newaxis, :]

for row, (model_label, _, pmbt_neurons) in enumerate(model_data):
    # ── col 0: p-value histogram ────────────────────────────────────────────
    ax = axes2[row, 0]
    pvals = np.array([n["p_value"] for n in pmbt_neurons])
    ax.hist(pvals, bins=BINS, range=(0, 1), color="#5c4e8a", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    # mark significance threshold
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1.2,
               label=r"$\alpha = 0.05$")
    ax.set_xlim(0, 1)
    ax.set_xlabel("$p$-value", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    if row == 0:
        ax.set_title("PMBT $p$-value Distribution", fontsize=11,
                     fontweight="bold", pad=5)
    if col == 0:
        ax.set_ylabel(model_label, fontsize=9, fontweight="bold")
    ax.set_ylabel(model_label, fontsize=9, fontweight="bold")

    # ── col 1: observed_rate_diff (D) histogram ─────────────────────────────
    ax = axes2[row, 1]
    dvals = np.array([n["observed_rate_diff"] for n in pmbt_neurons])
    # clip to [-1, 1] for display (theoretical range)
    dvals_c = np.clip(dvals, -1, 1)
    # colour bars by sign: positive D → visual (red), negative → text (blue)
    pos = dvals_c[dvals_c >= 0]
    neg = dvals_c[dvals_c < 0]
    bins_d = np.linspace(-1, 1, BINS + 1)
    ax.hist(dvals_c, bins=bins_d, color="#888888", alpha=0.5,
            edgecolor="white", linewidth=0.3, label="All")
    ax.hist(pos, bins=bins_d[bins_d >= 0], color=TYPE_COLORS["visual"],
            alpha=0.7, edgecolor="white", linewidth=0.3, label="Visual ($D>0$)")
    ax.hist(neg, bins=bins_d[bins_d <= 0], color=TYPE_COLORS["text"],
            alpha=0.7, edgecolor="white", linewidth=0.3, label="Text ($D<0$)")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Observed rate difference $D$", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    if row == 0:
        ax.set_title("PMBT Rate Difference $D$ Distribution",
                     fontsize=11, fontweight="bold", pad=5)

fig2.suptitle(
    "PMBT Statistical Distributions: $p$-values and Rate Differences",
    fontsize=12, fontweight="bold", y=1.01
)

for ext in ["pdf", "png"]:
    p = os.path.join(args.output_dir, f"fig_D2_pmbt_stats.{ext}")
    fig2.savefig(p, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {p}")
plt.close(fig2)


# ════════════════════════════════════════════════════════════════════════════
# Figure D3: Confidence score by label — FT and PMBT side by side
# Confidence: FT  → max(pv, pt, pm)   (excludes unknown)
#             PMBT → 1 - p_value
# 4-row × 2-col grid: col 0 = FT confidence, col 1 = PMBT confidence
# Uses overlaid KDE curves per label (visual/text/multimodal)
# ════════════════════════════════════════════════════════════════════════════
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CONF_LABELS = ["visual", "text", "multimodal"]

fig3, axes3 = plt.subplots(
    n_models, 2,
    figsize=(10, 3.2 * n_models),
    constrained_layout=True,
    sharey=False,
)
if n_models == 1:
    axes3 = axes3[np.newaxis, :]

for row, (model_label, ft_neurons, pmbt_neurons) in enumerate(model_data):

    # ── col 0: FT confidence = max(pv, pt, pm) by label ────────────────────
    ax = axes3[row, 0]
    ft_by_label = split_by_label(ft_neurons, "pv")   # placeholder — override below
    # Rebuild with max(pv,pt,pm) as confidence
    ft_conf_by_label = {lbl: [] for lbl in CONF_LABELS}
    for n in ft_neurons:
        lbl = n["label"]
        if lbl in ft_conf_by_label:
            ft_conf_by_label[lbl].append(max(n["pv"], n["pt"], n["pm"]))
    ft_conf_by_label = {k: np.array(v) for k, v in ft_conf_by_label.items()}

    for lbl in CONF_LABELS:
        vals = ft_conf_by_label[lbl]
        if len(vals) < 10:
            continue
        color = TYPE_COLORS[lbl]
        if HAS_SCIPY and len(vals) > 20:
            # KDE curve
            kde = gaussian_kde(vals, bw_method=0.08)
            xs  = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=1.8,
                    label=lbl.capitalize())
            ax.fill_between(xs, kde(xs), alpha=0.12, color=color)
        else:
            # fallback: histogram
            ax.hist(vals, bins=30, range=(0,1), color=color, alpha=0.4,
                    density=True, label=lbl.capitalize())

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence $\\max(p_v, p_t, p_m)$", fontsize=9)
    ax.set_ylabel(model_label, fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    if row == 0:
        ax.set_title("FT Classification Confidence by Label",
                     fontsize=11, fontweight="bold", pad=5)
        ax.legend(fontsize=8, frameon=False)

    # ── col 1: PMBT confidence = 1 - p_value by label ──────────────────────
    ax = axes3[row, 1]
    pmbt_conf_by_label = {lbl: [] for lbl in CONF_LABELS}
    for n in pmbt_neurons:
        lbl = n["label"]
        if lbl in pmbt_conf_by_label:
            pmbt_conf_by_label[lbl].append(1.0 - n["p_value"])
    pmbt_conf_by_label = {k: np.array(v) for k, v in pmbt_conf_by_label.items()}

    for lbl in CONF_LABELS:
        vals = pmbt_conf_by_label[lbl]
        if len(vals) < 10:
            continue
        color = TYPE_COLORS[lbl]
        if HAS_SCIPY and len(vals) > 20:
            kde = gaussian_kde(vals, bw_method=0.08)
            xs  = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=1.8,
                    label=lbl.capitalize())
            ax.fill_between(xs, kde(xs), alpha=0.12, color=color)
        else:
            ax.hist(vals, bins=30, range=(0,1), color=color, alpha=0.4,
                    density=True, label=lbl.capitalize())

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence $1 - p$-value", fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    if row == 0:
        ax.set_title("PMBT Classification Confidence by Label",
                     fontsize=11, fontweight="bold", pad=5)
        ax.legend(fontsize=8, frameon=False)

fig3.suptitle(
    "Classification Confidence by Neuron Label — FT vs PMBT",
    fontsize=12, fontweight="bold", y=1.01
)

for ext in ["pdf", "png"]:
    p = os.path.join(args.output_dir, f"fig_D3_confidence_by_label.{ext}")
    fig3.savefig(p, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {p}")
plt.close(fig3)

print("Done.")
