"""halluc_taxonomy.py — Hallucination Taxonomy Classification

Assigns each neuron a hallucination type from the 2×2 matrix:

    ┌──────────────────────────────┬────────────────────────┬──────────────────────┐
    │                              │  High CETT-diff        │  Low CETT-diff       │
    │                              │  (model relies on it)  │  (model ignores it)  │
    ├──────────────────────────────┼────────────────────────┼──────────────────────┤
    │  High Cohen's d              │  Type A — Real driver  │  Type B — Epiphenom. │
    │  (fires differently)         │  ★ ABLATE THESE        │                      │
    ├──────────────────────────────┼────────────────────────┼──────────────────────┤
    │  Low Cohen's d               │  Type C — General cap. │  Type D — Irrelevant │
    │  (fires similarly)           │  Don't ablate          │                      │
    └──────────────────────────────┴────────────────────────┴──────────────────────┘

Inputs (produced by halluc_score_neurons.py):
    results/halluc_scores/<comp>/<layer>/scores.npz
        Arrays: cohen_d [N], cett_diff [N]

Inputs (produced by neuron_modality_statistical.py):
    results/classification/<model>/<comp>/<layer>/neuron_labels.json
        List of dicts with keys: neuron_idx, label (visual/text/multimodal/unknown)

Outputs:
    results/halluc_taxonomy/<model>/
        summary.json                — per-component and grand-total counts
        per_layer_summary.json      — per-(comp, layer) counts
        enrichment.json             — Fisher's exact p-values per modality type
        <comp>/<layer>/halluc_labels.json — per-neuron hallucination type labels

Usage:
    python halluc_taxonomy.py
    python halluc_taxonomy.py \\
        --scores_dir results/halluc_scores \\
        --labels_dir results/classification/llava-1.5-7b \\
        --out_dir    results/halluc_taxonomy/llava-1.5-7b \\
        --cohen_threshold_pct  50 \\
        --cett_threshold_pct   50
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════
# Section 1 — CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    # Build the argument parser with all configurable paths and thresholds
    p = argparse.ArgumentParser(
        description='Classify neurons into hallucination taxonomy (Type A/B/C/D)'
    )
    p.add_argument(
        '--scores_dir', default='results/halluc_scores',
        help='Root dir produced by halluc_score_neurons.py. '
             'Expected layout: <scores_dir>/<comp>/<layer>/scores.npz'
    )
    p.add_argument(
        '--labels_dir', default='results/classification/llava-1.5-7b',
        help='Root dir produced by neuron_modality_statistical.py. '
             'Expected layout: <labels_dir>/<comp>/<layer>/neuron_labels.json'
    )
    p.add_argument(
        '--out_dir', default='results/halluc_taxonomy/llava-1.5-7b',
        help='Output root. Per-neuron halluc_labels.json files land here.'
    )
    p.add_argument(
        '--cohen_threshold_pct', type=float, default=50,
        help='Percentile of |cohen_d| used as the high/low boundary (default: 50 = median).'
    )
    p.add_argument(
        '--cett_threshold_pct', type=float, default=50,
        help='Percentile of cett_diff used as the high/low boundary (default: 50 = median).'
    )
    p.add_argument(
        '--components', nargs='+',
        default=['vision', 'projector', 'llm_fixed_threshold'],
        help='Component sub-directories to process.'
    )
    p.add_argument(
        '--abs_cohen', action='store_true', default=True,
        help='Use |cohen_d| for thresholding (direction-agnostic). Default: True.'
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Data loading helpers
# ═══════════════════════════════════════════════════════════════════

def load_scores(scores_dir, comp, layer):
    """Load cohen_d and cett_diff arrays from scores.npz for one layer.

    Returns (cohen_d, cett_diff) as float32 numpy arrays, or (None, None)
    if the file does not exist.
    """
    # Construct path matching halluc_score_neurons.py output convention
    path = os.path.join(scores_dir, comp, layer, 'scores.npz')
    if not os.path.isfile(path):
        return None, None
    data = np.load(path)
    # Both arrays have shape [num_neurons_in_layer]
    cohen_d  = data['cohen_d'].astype(np.float32)
    cett_diff = data['cett_diff'].astype(np.float32)
    return cohen_d, cett_diff


def load_modality_labels(labels_dir, comp, layer):
    """Load modality labels from neuron_labels.json for one layer.

    Returns a dict {neuron_idx: label_str} where label_str ∈
    {visual, text, multimodal, unknown}.
    Returns None if the file does not exist.
    """
    path = os.path.join(labels_dir, comp, layer, 'neuron_labels.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        records = json.load(f)
    # Build neuron_idx → label mapping; fall back to list position if key absent
    label_map = {}
    for i, rec in enumerate(records):
        idx = rec.get('neuron_idx', i)  # neuron_modality_statistical.py uses 'neuron_idx'
        label_map[idx] = rec['label']
    return label_map


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Hallucination type assignment
# ═══════════════════════════════════════════════════════════════════

HALLUC_TYPES = ['A', 'B', 'C', 'D']  # ordered for display

# Human-readable names used in summary JSON
HALLUC_TYPE_NAMES = {
    'A': 'Real driver (High Cohen_d + High CETT)',
    'B': 'Epiphenomenon (High Cohen_d + Low CETT)',
    'C': 'General capability (Low Cohen_d + High CETT)',
    'D': 'Irrelevant (Low Cohen_d + Low CETT)',
}


def assign_halluc_types(cohen_d, cett_diff, cohen_thr, cett_thr, abs_cohen=True):
    """Vectorised assignment of hallucination type labels.

    Args:
        cohen_d:    np.ndarray [N] — Cohen's d per neuron
        cett_diff:  np.ndarray [N] — CETT-diff per neuron
        cohen_thr:  float — threshold separating high/low Cohen's d
        cett_thr:   float — threshold separating high/low CETT-diff
        abs_cohen:  bool  — if True, use |cohen_d| for comparison

    Returns:
        labels: list[str] of length N, each ∈ {'A', 'B', 'C', 'D'}
    """
    # Use absolute value of cohen_d so we capture neurons that differ
    # in either direction (hallucinating higher or lower than correct)
    cd = np.abs(cohen_d) if abs_cohen else cohen_d

    # Boolean masks — True = "high" on that metric
    high_cohen = cd >= cohen_thr          # shape [N]
    high_cett  = cett_diff >= cett_thr   # shape [N]

    # Map 2×2 boolean combinations to type labels
    # Type A: high Cohen's d AND high CETT-diff
    # Type B: high Cohen's d AND low  CETT-diff
    # Type C: low  Cohen's d AND high CETT-diff
    # Type D: low  Cohen's d AND low  CETT-diff
    labels = np.where(
        high_cohen & high_cett, 'A',
        np.where(high_cohen & ~high_cett, 'B',
        np.where(~high_cohen & high_cett, 'C', 'D'))
    )
    return labels.tolist()


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Fisher's exact test for enrichment
# ═══════════════════════════════════════════════════════════════════

def fisher_enrichment(halluc_labels, modality_labels, halluc_type, modality_type):
    """Compute a 2×2 Fisher's exact test:

    Rows = hallucination type (target vs rest)
    Cols = modality type (target vs rest)

    Returns (odds_ratio, p_value).  Uses two-sided alternative.
    """
    # Count the four cells of the contingency table
    a = sum(1 for h, m in zip(halluc_labels, modality_labels)
            if h == halluc_type and m == modality_type)          # target × target
    b = sum(1 for h, m in zip(halluc_labels, modality_labels)
            if h == halluc_type and m != modality_type)          # target × other
    c = sum(1 for h, m in zip(halluc_labels, modality_labels)
            if h != halluc_type and m == modality_type)          # other × target
    d = sum(1 for h, m in zip(halluc_labels, modality_labels)
            if h != halluc_type and m != modality_type)          # other × other

    # scipy.stats.fisher_exact expects [[a, b], [c, d]]
    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
    return float(odds_ratio), float(p_value)


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Discover layers present in both scores and labels
# ═══════════════════════════════════════════════════════════════════

def discover_layers(scores_dir, labels_dir, comp):
    """Return sorted list of layer subdirectory names that have both
    a scores.npz file and a neuron_labels.json file for the given component.
    """
    scores_comp = os.path.join(scores_dir, comp)
    labels_comp = os.path.join(labels_dir, comp)

    # Collect layers that exist in both directories
    scores_layers = set()
    if os.path.isdir(scores_comp):
        for entry in os.scandir(scores_comp):
            if entry.is_dir() and os.path.isfile(
                    os.path.join(entry.path, 'scores.npz')):
                scores_layers.add(entry.name)

    labels_layers = set()
    if os.path.isdir(labels_comp):
        for entry in os.scandir(labels_comp):
            if entry.is_dir() and os.path.isfile(
                    os.path.join(entry.path, 'neuron_labels.json')):
                labels_layers.add(entry.name)

    # Intersection — only layers where both inputs exist
    common = sorted(scores_layers & labels_layers)
    return common


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Global threshold computation
# ═══════════════════════════════════════════════════════════════════

def compute_global_thresholds(scores_dir, components, layers_by_comp,
                               cohen_pct, cett_pct, abs_cohen):
    """Pool all cohen_d and cett_diff values across all components and layers,
    then return the percentile-based thresholds.

    Computing thresholds globally (rather than per-layer) ensures that the
    binary "high/low" cut means the same thing across layers and components.
    """
    all_cohen  = []
    all_cett   = []

    for comp in components:
        for layer in layers_by_comp.get(comp, []):
            cd, ct = load_scores(scores_dir, comp, layer)
            if cd is None:
                continue
            # Accumulate; use |cohen_d| for threshold if abs_cohen=True
            all_cohen.append(np.abs(cd) if abs_cohen else cd)
            all_cett.append(ct)

    if not all_cohen:
        raise RuntimeError('No scores.npz files found. Check --scores_dir.')

    all_cohen = np.concatenate(all_cohen)   # shape [total_neurons]
    all_cett  = np.concatenate(all_cett)

    cohen_thr = float(np.percentile(all_cohen, cohen_pct))  # e.g. 50 → median
    cett_thr  = float(np.percentile(all_cett,  cett_pct))

    print(f'Global thresholds  |cohen_d| > {cohen_thr:.4f} (p{cohen_pct}), '
          f'cett_diff > {cett_thr:.4f} (p{cett_pct})')
    return cohen_thr, cett_thr


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 7a. Discover layers available for each component ---
    layers_by_comp = {}
    for comp in args.components:
        layers = discover_layers(args.scores_dir, args.labels_dir, comp)
        layers_by_comp[comp] = layers
        print(f'{comp}: {len(layers)} layers with both scores and modality labels')

    # --- 7b. Compute global percentile thresholds ---
    cohen_thr, cett_thr = compute_global_thresholds(
        args.scores_dir, args.components, layers_by_comp,
        args.cohen_threshold_pct, args.cett_threshold_pct, args.abs_cohen
    )

    # --- 7c. Accumulators for summary statistics ---
    # comp_counts[comp][halluc_type] = count
    comp_counts = defaultdict(lambda: defaultdict(int))

    # cross_tab[comp][modality][halluc_type] = count — for enrichment
    cross_tab = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # per_layer_counts[(comp, layer)][halluc_type] = count
    per_layer_counts = {}

    # Lists for global enrichment test (all neurons pooled)
    all_halluc_labels    = []
    all_modality_labels  = []

    # --- 7d. Process each (component, layer) pair ---
    for comp in args.components:
        for layer in layers_by_comp.get(comp, []):

            # Load inputs
            cohen_d, cett_diff = load_scores(args.scores_dir, comp, layer)
            modality_map = load_modality_labels(args.labels_dir, comp, layer)

            if cohen_d is None or modality_map is None:
                continue  # should not happen after discover_layers, but be safe

            # Assign hallucination types to all neurons in this layer
            halluc_type_arr = assign_halluc_types(
                cohen_d, cett_diff, cohen_thr, cett_thr, args.abs_cohen
            )

            # Build per-neuron records for output JSON
            records = []
            layer_counts = defaultdict(int)

            for neuron_idx, halluc_type in enumerate(halluc_type_arr):
                modality = modality_map.get(neuron_idx, 'unknown')

                # Collect record for saving
                records.append({
                    'neuron_idx':   neuron_idx,
                    'halluc_type':  halluc_type,                  # 'A'/'B'/'C'/'D'
                    'modality':     modality,                      # 'visual'/'text'/...
                    'cohen_d':      round(float(cohen_d[neuron_idx]), 6),
                    'cett_diff':    round(float(cett_diff[neuron_idx]), 6),
                })

                # Accumulate counts
                layer_counts[halluc_type]          += 1
                comp_counts[comp][halluc_type]     += 1
                cross_tab[comp][modality][halluc_type] += 1

                all_halluc_labels.append(halluc_type)
                all_modality_labels.append(modality)

            per_layer_counts[(comp, layer)] = dict(layer_counts)

            # Save per-neuron halluc_labels.json
            out_layer_dir = os.path.join(args.out_dir, comp, layer)
            os.makedirs(out_layer_dir, exist_ok=True)
            out_path = os.path.join(out_layer_dir, 'halluc_labels.json')
            with open(out_path, 'w') as f:
                json.dump(records, f)

    # --- 7e. Compute enrichment (Fisher's exact) per modality × halluc_type ---
    modality_types = ['visual', 'text', 'multimodal', 'unknown']
    enrichment = {}

    for mod in modality_types:
        enrichment[mod] = {}
        for ht in HALLUC_TYPES:
            # Pool all neurons globally for the enrichment test
            or_, p_ = fisher_enrichment(
                all_halluc_labels, all_modality_labels, ht, mod
            )
            enrichment[mod][ht] = {
                'odds_ratio': round(or_, 4),
                'p_value':    round(p_, 6),
            }

    # Save enrichment JSON
    with open(os.path.join(args.out_dir, 'enrichment.json'), 'w') as f:
        json.dump(enrichment, f, indent=2)

    # --- 7f. Build and save per-layer summary JSON ---
    per_layer_json = {}
    for (comp, layer), counts in per_layer_counts.items():
        key = f'{comp}/{layer}'
        total = sum(counts.values())
        per_layer_json[key] = {
            'total': total,
            'counts': counts,
            'fractions': {
                ht: round(counts.get(ht, 0) / total, 4) if total else 0
                for ht in HALLUC_TYPES
            }
        }
    with open(os.path.join(args.out_dir, 'per_layer_summary.json'), 'w') as f:
        json.dump(per_layer_json, f, indent=2)

    # --- 7g. Build and save top-level summary JSON ---
    summary = {
        'thresholds': {
            'cohen_d_abs_percentile': args.cohen_threshold_pct,
            'cett_diff_percentile':   args.cett_threshold_pct,
            'cohen_d_threshold':      round(cohen_thr, 6),
            'cett_diff_threshold':    round(cett_thr, 6),
            'abs_cohen':              args.abs_cohen,
        },
        'halluc_type_names': HALLUC_TYPE_NAMES,
        'per_component': {},
        'grand_total': {},
    }

    # Per-component subtotals
    for comp in args.components:
        counts = comp_counts[comp]
        total = sum(counts.values())
        if total == 0:
            continue
        summary['per_component'][comp] = {
            'total': total,
            'counts': dict(counts),
            'fractions': {
                ht: round(counts.get(ht, 0) / total, 4)
                for ht in HALLUC_TYPES
            },
            'cross_tab_modality': {
                mod: dict(cross_tab[comp][mod])
                for mod in modality_types
            },
        }

    # Grand total across all components
    gt_counts = defaultdict(int)
    for counts in comp_counts.values():
        for ht, n in counts.items():
            gt_counts[ht] += n
    gt_total = sum(gt_counts.values())
    summary['grand_total'] = {
        'total': gt_total,
        'counts': dict(gt_counts),
        'fractions': {
            ht: round(gt_counts.get(ht, 0) / gt_total, 4) if gt_total else 0
            for ht in HALLUC_TYPES
        },
    }

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # --- 7h. Print human-readable report ---
    _print_report(summary, enrichment, per_layer_counts, args.components)


# ═══════════════════════════════════════════════════════════════════
# Section 8 — Console report
# ═══════════════════════════════════════════════════════════════════

def _print_report(summary, enrichment, per_layer_counts, components):
    """Print a classify_stats.py-style console report."""

    sep = '=' * 80
    print(f'\n{sep}')
    print('HALLUCINATION TAXONOMY — PER-COMPONENT SUMMARY')
    print(sep)
    thr = summary['thresholds']
    print(f"Thresholds: |cohen_d| ≥ {thr['cohen_d_threshold']:.4f} "
          f"(p{thr['cohen_d_abs_percentile']}),  "
          f"cett_diff ≥ {thr['cett_diff_threshold']:.4f} "
          f"(p{thr['cett_diff_percentile']})")

    for comp in components:
        comp_data = summary['per_component'].get(comp)
        if comp_data is None:
            continue
        total = comp_data['total']
        print(f'\n{comp.upper()} ({total:,} neurons):')
        for ht in HALLUC_TYPES:
            n   = comp_data['counts'].get(ht, 0)
            frc = comp_data['fractions'].get(ht, 0)
            name = HALLUC_TYPE_NAMES[ht]
            print(f'  Type {ht} ({name[:30]:30s}): {n:7,}  ({100*frc:5.1f}%)')

    print(f'\n{sep}')
    print('MODALITY × HALLUCINATION TYPE ENRICHMENT (Fisher\'s exact, two-sided)')
    print(sep)
    hdr = f'{"Modality":<14s}'
    for ht in HALLUC_TYPES:
        hdr += f'  Type {ht} OR      p'
    print(hdr)
    for mod in ['visual', 'text', 'multimodal', 'unknown']:
        row = f'{mod:<14s}'
        for ht in HALLUC_TYPES:
            entry = enrichment.get(mod, {}).get(ht, {})
            or_ = entry.get('odds_ratio', float('nan'))
            p_  = entry.get('p_value',    float('nan'))
            row += f'  {or_:6.3f}  {p_:.4f}'
        print(row)

    print(f'\n{sep}')
    gt = summary['grand_total']
    print(f'GRAND TOTAL ({gt["total"]:,} neurons):')
    for ht in HALLUC_TYPES:
        n   = gt['counts'].get(ht, 0)
        frc = gt['fractions'].get(ht, 0)
        print(f'  Type {ht}: {n:7,}  ({100*frc:5.1f}%)  — {HALLUC_TYPE_NAMES[ht]}')
    print()


if __name__ == '__main__':
    main()
