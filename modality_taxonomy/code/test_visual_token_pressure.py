#!/usr/bin/env python3
"""
test_visual_token_pressure.py — Visual Token Pressure Hypothesis Analysis

Tests two predictions from the VTP hypothesis:
  P1: Visual neuron % correlates with training-time visual token ratio
  P2: Models with sophisticated projectors show less backbone visual adaptation

Reads permutation_stats_all.json (or classification_stats_all.json) from
each model's output directory and combines with known architecture specs.

Usage:
  python code/test_visual_token_pressure.py \
      --output_dir results/3-classify/full \
      --out results/9-vtp
  
  # Specify models explicitly (otherwise auto-discovers all in output_dir):
  python code/test_visual_token_pressure.py \
      --output_dir results/3-classify/full \
      --models llava-1.5-7b internvl2.5-8b qwen2.5-vl-7b llava-onevision-7b \
      --out results/9-vtp

  # Include Stage 2 checkpoint for P3 comparison:
  python code/test_visual_token_pressure.py \
      --output_dir results/3-classify/full \
      --out results/9-vtp --include_si
"""

import argparse    # CLI argument parsing
import json        # load/save JSON stats files
import os          # path operations, directory listing
import sys         # sys.exit for error handling

# ═══════════════════════════════════════════════════════════════════
# Section 1 — Architecture Specifications (known from model cards)
# ═══════════════════════════════════════════════════════════════════

# Each model's training-time visual token properties.
# These are estimated from published model cards, papers, and code.
#
# Fields:
#   display_name:         human-readable name for plots
#   vision_encoder:       name of vision backbone
#   projector_type:       'mlp' | 'pixel_shuffle' | '2d_pooling'
#   projector_complexity: 1 (simple MLP) to 3 (sophisticated compression)
#   visual_tokens_per_image: estimated visual tokens injected per image during training
#   text_tokens_per_sample:  estimated text tokens per training sample (instruction + response)
#   training_stages:      number of fine-tuning stages with backbone unfrozen
#   has_video_stage:      whether training includes video/multi-image data
#   visual_token_ratio:   estimated ratio of visual:text tokens during training
#   n_layers:             number of transformer layers
#   ffn_dim:              FFN intermediate dimension
#   total_neurons:        n_layers × ffn_dim
#   backbone_family:      which pretrained LLM family

MODEL_SPECS = {                                                        # architecture specs per model
    'llava-1.5-7b': {
        'display_name':           'LLaVA-1.5-7B',
        'vision_encoder':         'CLIP ViT-L/14 (304M)',
        'projector_type':         'mlp',
        'projector_complexity':   1,                                   # simple 2-layer MLP
        'visual_tokens_per_image': 576,                                # 1 × 336×336 crop → 576 CLIP tokens
        'text_tokens_per_sample':  50,                                 # avg instruction + response length
        'training_stages':         2,                                  # align + SFT
        'has_video_stage':         False,
        'visual_token_ratio':      11.5,                               # 576 / 50
        'n_layers':                32,
        'ffn_dim':                 11008,
        'total_neurons':           352256,
        'backbone_family':         'LLaMA-2',
    },
    'internvl2.5-8b': {
        'display_name':           'InternVL2.5-8B',
        'vision_encoder':         'InternViT-6B (6B)',
        'projector_type':         'pixel_shuffle',
        'projector_complexity':   3,                                   # pixel-shuffle 4× reduction + MLP
        'visual_tokens_per_image': 256,                                # 448×448 → 1024 tokens → 4× shuffle → 256
        'text_tokens_per_sample':  60,                                 # slightly longer responses
        'training_stages':         3,                                  # progressive multi-stage
        'has_video_stage':         False,
        'visual_token_ratio':      4.3,                                # 256 / 60
        'n_layers':                32,
        'ffn_dim':                 14336,
        'total_neurons':           458752,
        'backbone_family':         'InternLM2',
    },
    'qwen2.5-vl-7b': {
        'display_name':           'Qwen2.5-VL-7B',
        'vision_encoder':         'ViT (native, dynamic resolution)',
        'projector_type':         '2d_pooling',
        'projector_complexity':   3,                                   # 2D spatial pooling compression
        'visual_tokens_per_image': 320,                                # dynamic res, 2D pooling compresses heavily
        'text_tokens_per_sample':  55,
        'training_stages':         3,                                  # multi-stage integrated
        'has_video_stage':         True,                               # native video support
        'visual_token_ratio':      5.8,                                # 320 / 55
        'n_layers':                28,
        'ffn_dim':                 18944,
        'total_neurons':           530432,
        'backbone_family':         'Qwen2.5',
    },
    'llava-onevision-7b': {
        'display_name':           'LLaVA-OneVision-7B',
        'vision_encoder':         'SigLIP-SO400M',
        'projector_type':         'mlp',
        'projector_complexity':   1,                                   # simple 2-layer MLP (same as LLaVA-1.5)
        'visual_tokens_per_image': 5832,                               # AnyRes: avg ~8 tiles × 729 tokens
        'text_tokens_per_sample':  55,
        'training_stages':         3,                                  # align + SFT + OneVision
        'has_video_stage':         True,                               # Stage 3 = video + multi-image
        'visual_token_ratio':      106.0,                              # 5832 / 55 (includes video amplification)
        'n_layers':                28,
        'ffn_dim':                 18944,
        'total_neurons':           530432,
        'backbone_family':         'Qwen2',
    },
    # P3: Stage 2 checkpoint (before OneVision training)
    'llava-onevision-7b-si': {
        'display_name':           'LLaVA-OV-7B (Stage 2 only)',
        'vision_encoder':         'SigLIP-SO400M',
        'projector_type':         'mlp',
        'projector_complexity':   1,
        'visual_tokens_per_image': 2916,                               # AnyRes but single-image only: ~4 tiles × 729
        'text_tokens_per_sample':  55,
        'training_stages':         2,                                  # align + SFT only (no OneVision stage)
        'has_video_stage':         False,
        'visual_token_ratio':      53.0,                               # 2916 / 55
        'n_layers':                28,
        'ffn_dim':                 18944,
        'total_neurons':           530432,
        'backbone_family':         'Qwen2',
    },
}


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Load classification results
# ═══════════════════════════════════════════════════════════════════

def load_model_stats(output_dir, model_name, taxonomy='pmbt'):
    """Load classification stats for one model.

    Line 1: determine which stats file to load (PMBT or FT)
    Line 2: load JSON and extract per-category totals
    Line 3: compute percentages
    Line 4: extract per-layer breakdown for layer-level analysis
    Line 5: return dict with counts, percentages, and per-layer data

    Args:
        output_dir: base classification output dir (e.g. results/3-classify/full)
        model_name: model directory name (e.g. 'llava-1.5-7b')
        taxonomy:   'pmbt' for permutation test, 'ft' for fixed threshold

    Returns:
        dict with keys: counts, percentages, per_layer, total, n_layers
        None if stats file not found
    """
    if taxonomy == 'pmbt':                                             # permutation test results
        stats_dir = os.path.join(output_dir, model_name, 'llm_permutation')
        stats_file = os.path.join(stats_dir, 'permutation_stats_all.json')
    else:                                                              # fixed threshold results
        stats_dir = os.path.join(output_dir, model_name, 'llm_fixed_threshold')
        stats_file = os.path.join(stats_dir, 'classification_stats_all.json')

    if not os.path.isfile(stats_file):                                 # file not found
        print(f'  WARNING: {stats_file} not found')
        return None

    with open(stats_file) as f:                                        # load JSON
        data = json.load(f)

    stats = data['stats']                                              # overall counts dict
    total = sum(stats.values())                                        # total neuron count
    if total == 0:                                                     # avoid division by zero
        print(f'  WARNING: {model_name} has 0 total neurons')
        return None

    percentages = {k: 100.0 * v / total for k, v in stats.items()}    # compute percentages

    # Extract per-layer stats if available
    per_layer = {}                                                     # {layer_idx: {cat: count}}
    if 'per_layer_stats' in data:                                      # per-layer data present
        for layer_str, layer_stats in data['per_layer_stats'].items():
            layer_total = sum(layer_stats.values())                    # total neurons in this layer
            if layer_total > 0:
                per_layer[int(layer_str)] = {                          # store counts and percentages
                    'counts': layer_stats,
                    'total': layer_total,
                    'pct': {k: 100.0 * v / layer_total
                            for k, v in layer_stats.items()},
                }

    n_layers = data.get('n_layers_found', len(per_layer))             # number of layers classified

    return {                                                           # return structured result
        'counts': stats,
        'percentages': percentages,
        'per_layer': per_layer,
        'total': total,
        'n_layers': n_layers,
    }


def auto_discover_models(output_dir):
    """Find all models with classification results in output_dir.

    Line 1: list subdirectories of output_dir
    Line 2: check each for permutation_stats_all.json or classification_stats_all.json
    Line 3: return list of model names that have results

    Returns:
        list of model name strings
    """
    models = []                                                        # found model names
    if not os.path.isdir(output_dir):                                  # output dir doesn't exist
        return models

    for entry in sorted(os.listdir(output_dir)):                       # iterate subdirectories
        model_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(model_dir):                               # skip non-directories
            continue
        # Check for either PMBT or FT stats
        pmbt_stats = os.path.join(model_dir, 'llm_permutation',
                                   'permutation_stats_all.json')
        ft_stats = os.path.join(model_dir, 'llm_fixed_threshold',
                                 'classification_stats_all.json')
        if os.path.isfile(pmbt_stats) or os.path.isfile(ft_stats):    # has classification results
            models.append(entry)

    return models


# ═══════════════════════════════════════════════════════════════════
# Section 3 — P1: Visual Token Ratio Correlation
# ═══════════════════════════════════════════════════════════════════

def test_p1_correlation(model_data, model_specs):
    """Test P1: visual token ratio correlates with visual neuron %.

    Line 1: extract visual_token_ratio from specs and visual % from results
    Line 2: compute Spearman and Pearson correlations
    Line 3: report results and significance

    Args:
        model_data: dict {model_name: stats_dict} from load_model_stats
        model_specs: dict {model_name: spec_dict} from MODEL_SPECS

    Returns:
        dict with correlation results, data points, and interpretation
    """
    # Build paired data points
    points = []                                                        # (model, ratio, visual_pct)
    for model_name, stats in model_data.items():                       # iterate models
        if model_name not in model_specs:                              # no spec for this model
            print(f'  WARNING: no spec for {model_name}, skipping P1')
            continue
        spec = model_specs[model_name]
        ratio = spec['visual_token_ratio']                             # training-time visual:text ratio
        vis_pct = stats['percentages']['visual']                       # PMBT visual neuron %
        points.append({
            'model': spec['display_name'],
            'visual_token_ratio': ratio,
            'visual_neuron_pct': vis_pct,
            'total_neurons': stats['total'],
        })

    if len(points) < 3:                                                # need at least 3 for correlation
        print(f'  P1: only {len(points)} models — insufficient for correlation')
        return {'status': 'insufficient_data', 'n_models': len(points)}

    # Sort by ratio for display
    points.sort(key=lambda p: p['visual_token_ratio'])

    # Compute correlations
    ratios = [p['visual_token_ratio'] for p in points]                 # x values
    vis_pcts = [p['visual_neuron_pct'] for p in points]                # y values

    # Spearman rank correlation (non-parametric, robust to outliers)
    n = len(ratios)
    rank_x = _ranks(ratios)                                            # rank transform x
    rank_y = _ranks(vis_pcts)                                          # rank transform y
    spearman_r = _pearson(rank_x, rank_y)                              # Pearson on ranks = Spearman

    # Pearson correlation (linear)
    pearson_r = _pearson(ratios, vis_pcts)

    # Log-scale Pearson (visual token ratio spans orders of magnitude)
    import math
    log_ratios = [math.log10(r) for r in ratios]                       # log10 transform
    log_pearson_r = _pearson(log_ratios, vis_pcts)

    # Interpretation
    if abs(spearman_r) > 0.8:                                          # strong correlation
        interpretation = 'STRONG support for P1'
    elif abs(spearman_r) > 0.5:                                        # moderate correlation
        interpretation = 'MODERATE support for P1'
    else:                                                              # weak correlation
        interpretation = 'WEAK support for P1'

    result = {                                                         # package results
        'status': 'computed',
        'n_models': n,
        'data_points': points,
        'spearman_r': round(spearman_r, 4),
        'pearson_r': round(pearson_r, 4),
        'log_pearson_r': round(log_pearson_r, 4),
        'interpretation': interpretation,
    }

    # Print summary
    print(f'\n  ═══ P1: Visual Token Ratio → Visual Neuron % ═══')
    print(f'  {"Model":<28s} {"Token Ratio":>12s} {"Visual %":>10s}')
    print(f'  {"─"*28} {"─"*12} {"─"*10}')
    for p in points:                                                   # print each data point
        print(f'  {p["model"]:<28s} {p["visual_token_ratio"]:>12.1f} '
              f'{p["visual_neuron_pct"]:>9.1f}%')
    print(f'\n  Spearman ρ = {spearman_r:.4f}')
    print(f'  Pearson  r = {pearson_r:.4f}')
    print(f'  Log-Pearson r = {log_pearson_r:.4f}  (log₁₀ ratio vs %)')
    print(f'  → {interpretation}')

    return result


# ═══════════════════════════════════════════════════════════════════
# Section 4 — P2: Projector Sophistication Analysis
# ═══════════════════════════════════════════════════════════════════

def test_p2_projector(model_data, model_specs):
    """Test P2: sophisticated projectors → less backbone visual adaptation.

    Line 1: group models by projector complexity
    Line 2: compare visual neuron % between simple (MLP) and complex projectors
    Line 3: control for training intensity by comparing same-backbone models

    Args:
        model_data: dict {model_name: stats_dict}
        model_specs: dict {model_name: spec_dict}

    Returns:
        dict with comparison results and interpretation
    """
    points = []                                                        # (model, complexity, vis_pct, ratio)
    for model_name, stats in model_data.items():
        if model_name not in model_specs:
            continue
        spec = model_specs[model_name]
        points.append({
            'model': spec['display_name'],
            'projector_type': spec['projector_type'],
            'projector_complexity': spec['projector_complexity'],
            'visual_neuron_pct': stats['percentages']['visual'],
            'visual_token_ratio': spec['visual_token_ratio'],
            'backbone_family': spec['backbone_family'],
        })

    # Group by complexity
    simple = [p for p in points if p['projector_complexity'] == 1]     # simple MLP projectors
    complex_proj = [p for p in points if p['projector_complexity'] >= 2]  # sophisticated projectors

    # Key comparison: LLaVA-OV vs Qwen (same backbone family, different projectors)
    qwen2_models = [p for p in points
                     if p['backbone_family'] in ('Qwen2', 'Qwen2.5')]
    controlled_comparison = None
    if len(qwen2_models) >= 2:                                         # have both Qwen2 variants
        simple_qwen = [p for p in qwen2_models
                        if p['projector_complexity'] == 1]
        complex_qwen = [p for p in qwen2_models
                         if p['projector_complexity'] >= 2]
        if simple_qwen and complex_qwen:
            controlled_comparison = {
                'simple': simple_qwen[0],
                'complex': complex_qwen[0],
                'visual_pct_diff': (simple_qwen[0]['visual_neuron_pct'] -
                                     complex_qwen[0]['visual_neuron_pct']),
            }

    # Residual analysis: visual neuron % after controlling for token ratio
    # If projector doesn't matter, visual% should depend only on token ratio
    # If projector matters, complex projectors should have lower visual% than
    # predicted by their token ratio alone
    residuals = []
    if len(points) >= 3:
        # Simple linear fit: vis_pct = a * log(ratio) + b
        import math
        log_ratios = [math.log10(p['visual_token_ratio']) for p in points]
        vis_pcts = [p['visual_neuron_pct'] for p in points]
        a, b = _linear_fit(log_ratios, vis_pcts)                      # least-squares fit

        for i, p in enumerate(points):                                 # compute residuals
            predicted = a * log_ratios[i] + b
            residual = vis_pcts[i] - predicted                         # positive = more visual than expected
            residuals.append({
                'model': p['model'],
                'projector_type': p['projector_type'],
                'complexity': p['projector_complexity'],
                'actual_visual_pct': vis_pcts[i],
                'predicted_visual_pct': round(predicted, 1),
                'residual': round(residual, 1),
            })

    # Interpretation
    if controlled_comparison:
        diff = controlled_comparison['visual_pct_diff']
        if diff > 10:                                                  # >10pp difference
            interpretation = (f'STRONG support for P2: simple MLP projector → '
                            f'{diff:.1f}pp more visual neurons than complex projector '
                            f'(same Qwen2 backbone)')
        elif diff > 5:
            interpretation = (f'MODERATE support for P2: {diff:.1f}pp difference '
                            f'between simple and complex projectors (Qwen2 backbone)')
        else:
            interpretation = f'WEAK support for P2: only {diff:.1f}pp difference'
    else:
        interpretation = 'P2: no controlled comparison available (need Qwen2 variants)'

    result = {
        'status': 'computed',
        'points': points,
        'simple_projector_models': simple,
        'complex_projector_models': complex_proj,
        'controlled_comparison': controlled_comparison,
        'residuals': residuals,
        'interpretation': interpretation,
    }

    # Print summary
    print(f'\n  ═══ P2: Projector Sophistication → Backbone Visual Adaptation ═══')
    print(f'  {"Model":<28s} {"Projector":>14s} {"Complexity":>10s} {"Visual %":>10s}')
    print(f'  {"─"*28} {"─"*14} {"─"*10} {"─"*10}')
    for p in sorted(points, key=lambda x: x['projector_complexity']):
        print(f'  {p["model"]:<28s} {p["projector_type"]:>14s} '
              f'{p["projector_complexity"]:>10d} {p["visual_neuron_pct"]:>9.1f}%')

    if controlled_comparison:
        print(f'\n  Controlled comparison (same Qwen2 backbone):')
        s = controlled_comparison['simple']
        c = controlled_comparison['complex']
        print(f'    Simple MLP ({s["model"]}): {s["visual_neuron_pct"]:.1f}%')
        print(f'    Complex ({c["model"]}):    {c["visual_neuron_pct"]:.1f}%')
        print(f'    Difference: {controlled_comparison["visual_pct_diff"]:.1f}pp')

    if residuals:
        print(f'\n  Residuals (actual − predicted from token ratio):')
        for r in residuals:
            sign = '+' if r['residual'] >= 0 else ''
            print(f'    {r["model"]:<28s} {r["projector_type"]:>14s}  '
                  f'actual={r["actual_visual_pct"]:.1f}%  '
                  f'predicted={r["predicted_visual_pct"]:.1f}%  '
                  f'residual={sign}{r["residual"]:.1f}pp')

    print(f'\n  → {interpretation}')

    return result


# ═══════════════════════════════════════════════════════════════════
# Section 5 — P3: Stage 2 vs Stage 3 Comparison
# ═══════════════════════════════════════════════════════════════════

def test_p3_stage_comparison(model_data, model_specs):
    """Test P3: Stage 2 checkpoint should be less visual-dominant.

    Line 1: check if both llava-onevision-7b and llava-onevision-7b-si results exist
    Line 2: compare visual neuron % between Stage 2 (SI) and Stage 3 (OV)
    Line 3: compute per-layer differences

    Args:
        model_data: dict {model_name: stats_dict}
        model_specs: dict {model_name: spec_dict}

    Returns:
        dict with comparison results, or status='missing_data'
    """
    ov_key = 'llava-onevision-7b'                                      # Stage 3 (full OneVision)
    si_key = 'llava-onevision-7b-si'                                   # Stage 2 (single-image only)

    if ov_key not in model_data:                                       # Stage 3 results missing
        print(f'\n  P3: {ov_key} results not found — skipping')
        return {'status': 'missing_ov'}
    if si_key not in model_data:                                       # Stage 2 results missing
        print(f'\n  P3: {si_key} results not found — run pipeline on SI checkpoint')
        print(f'       bash code/run_pipeline.sh --model-type llava-ov-si --step all')
        return {'status': 'missing_si',
                'instruction': 'Run: bash code/run_pipeline.sh --model-type llava-ov-si --step all'}

    ov = model_data[ov_key]                                            # Stage 3 results
    si = model_data[si_key]                                            # Stage 2 results

    diff = {cat: ov['percentages'][cat] - si['percentages'][cat]       # per-category difference
            for cat in ['visual', 'text', 'multimodal', 'unknown']}

    # Per-layer comparison (if both have per-layer data)
    layer_diffs = {}                                                   # {layer: {cat: diff}}
    common_layers = sorted(set(ov['per_layer'].keys()) &
                            set(si['per_layer'].keys()))
    for l in common_layers:
        ov_pct = ov['per_layer'][l]['pct']
        si_pct = si['per_layer'][l]['pct']
        layer_diffs[l] = {cat: ov_pct.get(cat, 0) - si_pct.get(cat, 0)
                           for cat in ['visual', 'text', 'multimodal', 'unknown']}

    # Interpretation
    vis_diff = diff['visual']
    if vis_diff > 5:                                                   # Stage 3 adds >5pp visual
        interpretation = (f'STRONG support for P3: OneVision stage adds {vis_diff:.1f}pp '
                        f'visual neurons. Stage 2 checkpoint is less visual-dominant.')
    elif vis_diff > 2:
        interpretation = (f'MODERATE support for P3: OneVision stage adds {vis_diff:.1f}pp '
                        f'visual neurons.')
    elif vis_diff > 0:
        interpretation = (f'WEAK support for P3: only {vis_diff:.1f}pp increase from '
                        f'OneVision stage.')
    else:
        interpretation = (f'P3 NOT SUPPORTED: Stage 3 has {vis_diff:.1f}pp fewer visual '
                        f'neurons than Stage 2.')

    result = {
        'status': 'computed',
        'stage3_visual_pct': ov['percentages']['visual'],
        'stage2_visual_pct': si['percentages']['visual'],
        'overall_diff': diff,
        'layer_diffs': layer_diffs,
        'interpretation': interpretation,
    }

    # Print summary
    print(f'\n  ═══ P3: Stage 2 (SI) vs Stage 3 (OneVision) ═══')
    print(f'  {"Category":<14s} {"Stage 2 (SI)":>14s} {"Stage 3 (OV)":>14s} {"Δ":>8s}')
    print(f'  {"─"*14} {"─"*14} {"─"*14} {"─"*8}')
    for cat in ['visual', 'text', 'multimodal', 'unknown']:
        si_v = si['percentages'][cat]
        ov_v = ov['percentages'][cat]
        d = diff[cat]
        sign = '+' if d >= 0 else ''
        print(f'  {cat:<14s} {si_v:>13.1f}% {ov_v:>13.1f}% {sign}{d:>7.1f}pp')

    print(f'\n  → {interpretation}')

    return result


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Publication-Ready Figures
# ═══════════════════════════════════════════════════════════════════

def generate_figures(p1_result, p2_result, p3_result, out_dir):
    """Generate publication-ready figures for the VTP hypothesis.

    Line 1: import matplotlib, set publication style
    Line 2: Figure A — scatter plot of token ratio vs visual neuron %
    Line 3: Figure B — projector complexity bar chart with controlled comparison
    Line 4: Figure C — P3 stage comparison (if data available)
    Line 5: Figure D — combined summary panel
    """
    import matplotlib                                                  # matplotlib backend
    matplotlib.use('Agg')                                              # non-interactive backend
    import matplotlib.pyplot as plt                                    # plotting API
    import math                                                        # log transform

    os.makedirs(out_dir, exist_ok=True)                                # create output directory

    # Publication style
    plt.rcParams.update({                                              # set plot style
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # Color scheme
    COLORS = {                                                         # consistent color palette
        'llava-1.5': '#e74c3c',
        'internvl':  '#3498db',
        'qwen':      '#2ecc71',
        'llava-ov':  '#9b59b6',
        'llava-si':  '#e67e22',
    }

    def _model_color(name):
        """Map model display name to color."""
        name_lower = name.lower()
        if 'stage 2' in name_lower or '-si' in name_lower:
            return COLORS['llava-si']
        if 'onevision' in name_lower or 'llava-ov' in name_lower:
            return COLORS['llava-ov']
        if 'internvl' in name_lower:
            return COLORS['internvl']
        if 'qwen' in name_lower:
            return COLORS['qwen']
        return COLORS['llava-1.5']

    # ── Figure A: P1 scatter plot ──────────────────────────────
    if p1_result.get('status') == 'computed':
        pts = p1_result['data_points']
        fig, ax = plt.subplots(figsize=(7, 5))

        ratios = [p['visual_token_ratio'] for p in pts]                # x values
        vis_pcts = [p['visual_neuron_pct'] for p in pts]               # y values

        for p in pts:                                                  # plot each model
            c = _model_color(p['model'])
            ax.scatter(p['visual_token_ratio'], p['visual_neuron_pct'],
                      s=120, c=c, edgecolors='black', linewidths=0.5,
                      zorder=3)
            # Label with model name (offset to avoid overlap)
            offset_x = 0.05                                            # log-scale offset
            ax.annotate(p['model'],
                       (p['visual_token_ratio'], p['visual_neuron_pct']),
                       textcoords='offset points', xytext=(8, 5),
                       fontsize=9, color=c, fontweight='bold')

        # Log-linear fit line
        log_ratios = [math.log10(r) for r in ratios]
        a, b = _linear_fit(log_ratios, vis_pcts)
        x_fit = [min(ratios) * 0.8, max(ratios) * 1.2]               # extend fit range
        y_fit = [a * math.log10(x) + b for x in x_fit]
        ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.5, linewidth=1.5)

        ax.set_xscale('log')                                           # log scale for ratio
        ax.set_xlabel('Visual Token Ratio (visual:text tokens during training)')
        ax.set_ylabel('Visual Neuron % (PMBT)')
        ax.set_title(f'P1: Visual Token Pressure Hypothesis\n'
                    f'Spearman ρ = {p1_result["spearman_r"]:.3f}, '
                    f'Log-Pearson r = {p1_result["log_pearson_r"]:.3f}')
        ax.grid(True, alpha=0.3)

        fig.savefig(os.path.join(out_dir, 'P1_token_ratio_vs_visual_pct.png'))
        fig.savefig(os.path.join(out_dir, 'P1_token_ratio_vs_visual_pct.pdf'))
        plt.close(fig)
        print(f'  Saved P1 figure → {out_dir}/')

    # ── Figure B: P2 projector comparison ──────────────────────
    if p2_result.get('status') == 'computed' and p2_result.get('residuals'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Left: visual % by projector type
        pts = p2_result['points']
        pts_sorted = sorted(pts, key=lambda x: x['visual_neuron_pct'])
        names = [p['model'] for p in pts_sorted]
        vis_pcts = [p['visual_neuron_pct'] for p in pts_sorted]
        bar_colors = [_model_color(p['model']) for p in pts_sorted]

        bars = ax1.barh(names, vis_pcts, color=bar_colors, edgecolor='black',
                        linewidth=0.5)
        for bar, p in zip(bars, pts_sorted):                           # annotate projector type
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{p["projector_type"]} (C={p["projector_complexity"]})',
                    va='center', fontsize=9, color='gray')

        ax1.set_xlabel('Visual Neuron % (PMBT)')
        ax1.set_title('Visual Neuron % by Model')
        ax1.set_xlim(0, max(vis_pcts) * 1.25)

        # Right: residuals from token-ratio prediction
        residuals = p2_result['residuals']
        res_sorted = sorted(residuals, key=lambda x: x['residual'])
        r_names = [r['model'] for r in res_sorted]
        r_vals = [r['residual'] for r in res_sorted]
        r_colors = ['#e74c3c' if r > 0 else '#3498db' for r in r_vals]

        ax2.barh(r_names, r_vals, color=r_colors, edgecolor='black',
                linewidth=0.5, alpha=0.7)
        ax2.axvline(x=0, color='black', linewidth=0.8)
        ax2.set_xlabel('Residual (actual − predicted from token ratio, pp)')
        ax2.set_title('P2: Projector Effect\n'
                      '(positive = more visual than token ratio predicts)')

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'P2_projector_comparison.png'))
        fig.savefig(os.path.join(out_dir, 'P2_projector_comparison.pdf'))
        plt.close(fig)
        print(f'  Saved P2 figure → {out_dir}/')

    # ── Figure C: P3 stage comparison ──────────────────────────
    if p3_result.get('status') == 'computed':
        categories = ['visual', 'text', 'multimodal', 'unknown']
        si_vals = [p3_result['stage2_visual_pct']]                     # placeholder — need full data
        # Build full bar chart from overall_diff
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(categories))
        width = 0.35
        # We need the actual percentages; reconstruct from diff
        ov_pcts = []
        si_pcts = []
        for cat in categories:
            ov_val = p3_result['overall_diff'][cat]                    # this is the difference
            # We stored stage3 and stage2 visual_pct at top level but only for visual
            # For full data we need to load again — use a simpler approach
            pass

        # Simpler: just show the visual% comparison as a focused bar chart
        fig, ax = plt.subplots(figsize=(5, 5))
        labels = ['Stage 2\n(Single-Image)', 'Stage 3\n(OneVision)']
        vals = [p3_result['stage2_visual_pct'], p3_result['stage3_visual_pct']]
        colors = [COLORS['llava-si'], COLORS['llava-ov']]

        bars = ax.bar(labels, vals, color=colors, edgecolor='black',
                      linewidth=0.5, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

        diff = p3_result['overall_diff']['visual']
        ax.set_ylabel('Visual Neuron % (PMBT)')
        ax.set_title(f'P3: Effect of OneVision Training Stage\n'
                    f'Δ = {diff:+.1f}pp visual neurons')
        ax.set_ylim(0, max(vals) * 1.15)

        fig.savefig(os.path.join(out_dir, 'P3_stage_comparison.png'))
        fig.savefig(os.path.join(out_dir, 'P3_stage_comparison.pdf'))
        plt.close(fig)
        print(f'  Saved P3 figure → {out_dir}/')

    # ── Figure D: Combined inflation spectrum ──────────────────
    if p1_result.get('status') == 'computed':
        pts = p1_result['data_points']
        fig, ax = plt.subplots(figsize=(7, 5))

        # Grouped bar: visual%, text%, multimodal%, unknown%
        # Need full percentages — P1 only stores visual%
        # Use visual% as primary and derive the rest would need full data
        # For now just do the key P1 scatter on log scale with linear annotation

        # Enhanced P1: add token count annotations
        for p in pts:
            c = _model_color(p['model'])
            ax.scatter(p['visual_token_ratio'], p['visual_neuron_pct'],
                      s=p['total_neurons'] / 3000,                    # size ~ total neurons
                      c=c, edgecolors='black', linewidths=0.5,
                      alpha=0.8, zorder=3)
            ax.annotate(f'{p["model"]}\n({p["total_neurons"]/1000:.0f}K neurons)',
                       (p['visual_token_ratio'], p['visual_neuron_pct']),
                       textcoords='offset points', xytext=(10, -5),
                       fontsize=8, color=c)

        ax.set_xscale('log')
        ax.set_xlabel('Visual Token Ratio (training)')
        ax.set_ylabel('Visual Neuron % (PMBT)')
        ax.set_title('Visual Token Pressure: All Models\n'
                    '(bubble size ∝ total FFN neurons)')
        ax.grid(True, alpha=0.3)

        fig.savefig(os.path.join(out_dir, 'P1_bubble_chart.png'))
        fig.savefig(os.path.join(out_dir, 'P1_bubble_chart.pdf'))
        plt.close(fig)
        print(f'  Saved bubble chart → {out_dir}/')

    print(f'\n  All figures saved to {out_dir}/')


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Statistical Helpers (no numpy dependency)
# ═══════════════════════════════════════════════════════════════════

def _ranks(values):
    """Compute ranks for a list of values (1-based, handles ties by averaging).

    Line 1: sort values with original indices
    Line 2: assign ranks, averaging ties
    Line 3: return ranks in original order
    """
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])           # sort by value
    ranks = [0.0] * n
    i = 0
    while i < n:                                                       # iterate through sorted values
        j = i
        while j < n - 1 and indexed[j+1][1] == indexed[i][1]:         # find ties
            j += 1
        avg_rank = (i + j) / 2.0 + 1                                  # average rank for ties
        for k in range(i, j + 1):                                     # assign ranks
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x, y):
    """Compute Pearson correlation coefficient.

    Line 1: compute means
    Line 2: compute covariance and standard deviations
    Line 3: return correlation coefficient
    """
    n = len(x)
    mx = sum(x) / n                                                    # mean x
    my = sum(y) / n                                                    # mean y
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n     # covariance
    sx = (sum((xi - mx)**2 for xi in x) / n) ** 0.5                   # std x
    sy = (sum((yi - my)**2 for yi in y) / n) ** 0.5                   # std y
    if sx == 0 or sy == 0:                                             # degenerate case
        return 0.0
    return cov / (sx * sy)                                             # correlation


def _linear_fit(x, y):
    """Simple least-squares linear fit: y = a*x + b.

    Line 1: compute means and sums
    Line 2: solve for slope and intercept
    Line 3: return (slope, intercept)
    """
    n = len(x)
    mx = sum(x) / n                                                    # mean x
    my = sum(y) / n                                                    # mean y
    ss_xx = sum((xi - mx)**2 for xi in x)                              # sum of squared deviations
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))       # sum of cross deviations
    if ss_xx == 0:                                                     # degenerate case
        return 0.0, my
    a = ss_xy / ss_xx                                                  # slope
    b = my - a * mx                                                    # intercept
    return a, b


# ═══════════════════════════════════════════════════════════════════
# Section 8 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Visual Token Pressure Hypothesis — tests P1, P2, P3')
    p.add_argument('--output_dir', required=True,                      # base classification output dir
                   help='Classification output dir (e.g. results/3-classify/full)')
    p.add_argument('--models', nargs='*', default=None,                # explicit model list
                   help='Model names to include (default: auto-discover)')
    p.add_argument('--taxonomy', default='pmbt',                       # which taxonomy to use
                   choices=['pmbt', 'ft'],
                   help='Taxonomy: pmbt (permutation) or ft (fixed threshold)')
    p.add_argument('--out', default='results/9-vtp',                   # output directory for figures
                   help='Output directory for figures and summary')
    p.add_argument('--include_si', action='store_true',                # include Stage 2 checkpoint
                   help='Include llava-onevision-7b-si (Stage 2) for P3')
    p.add_argument('--no_figures', action='store_true',                # skip figure generation
                   help='Skip figure generation (text analysis only)')

    args = p.parse_args()

    print('═══════════════════════════════════════════════════════════')
    print('  Visual Token Pressure Hypothesis — Analysis')
    print('═══════════════════════════════════════════════════════════')
    print(f'  Output dir:  {args.output_dir}')
    print(f'  Taxonomy:    {args.taxonomy}')
    print(f'  Figure dir:  {args.out}')

    # Discover or validate models
    if args.models:                                                    # explicit model list
        model_names = args.models
    else:                                                              # auto-discover
        model_names = auto_discover_models(args.output_dir)
        if not model_names:
            print(f'\nERROR: no models found in {args.output_dir}')
            sys.exit(1)

    print(f'  Models:      {", ".join(model_names)}')

    # Load stats for each model
    model_data = {}                                                    # {model_name: stats}
    for name in model_names:
        stats = load_model_stats(args.output_dir, name, args.taxonomy)
        if stats:                                                      # loaded successfully
            model_data[name] = stats
            print(f'    ✓ {name}: {stats["total"]:,} neurons, '
                  f'{stats["n_layers"]} layers, '
                  f'visual={stats["percentages"]["visual"]:.1f}%')
        else:
            print(f'    ✗ {name}: not found or empty')

    if len(model_data) < 2:                                            # need at least 2 models
        print(f'\nERROR: need at least 2 models, found {len(model_data)}')
        sys.exit(1)

    # Run tests
    p1_result = test_p1_correlation(model_data, MODEL_SPECS)
    p2_result = test_p2_projector(model_data, MODEL_SPECS)
    p3_result = test_p3_stage_comparison(model_data, MODEL_SPECS)

    # Generate figures
    if not args.no_figures:
        generate_figures(p1_result, p2_result, p3_result, args.out)

    # Save JSON summary
    os.makedirs(args.out, exist_ok=True)
    summary = {
        'models': list(model_data.keys()),
        'taxonomy': args.taxonomy,
        'P1_visual_token_correlation': p1_result,
        'P2_projector_sophistication': p2_result,
        'P3_stage_comparison': p3_result,
    }
    summary_path = os.path.join(args.out, 'vtp_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n  Summary saved → {summary_path}')

    print('\n═══════════════════════════════════════════════════════════')
    print('  VTP Analysis Complete')
    print('═══════════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()