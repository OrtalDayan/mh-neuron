#!/usr/bin/env python3
"""Step 29 — Weight Diff Effective Rank Analysis.

Compares VLM weights to base LLM weights at PMBT visual vs text neuron
positions. Measures effective rank (spectral entropy) of the weight diff
to test the theoretical prediction: sequential fine-tuning concentrates
late-added modalities into low-rank subspaces.

If visual diff is low-rank and text diff is high-rank (or near-zero),
this confirms that VLM fine-tuning concentrated visual processing into
a sparse subnetwork while text processing remains distributed.

No training required — just loads two models and computes SVD.
"""

import argparse
import json
import os
import sys
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Weight diff effective rank: VLM vs base LLM')
    p.add_argument('--model_type', type=str, default='')
    p.add_argument('--vlm_path', type=str, default='')
    p.add_argument('--llm_path', type=str, default='')
    p.add_argument('--model_name', type=str, default='')
    p.add_argument('--n_layers', type=int, default=0)
    p.add_argument('--label_dir', type=str, default='')
    p.add_argument('--output_dir', type=str, default='')
    p.add_argument('--weight_types', type=str, default='down_proj',
                   help='Comma-separated: down_proj,gate_proj,up_proj')
    # Cross-model comparison mode
    p.add_argument('--cross_model', action='store_true',
                   help='Run cross-model comparison instead of single model')
    p.add_argument('--base_dir', type=str, default='results/29-weight-diff-rank/full',
                   help='Base directory containing per-model results')
    p.add_argument('--models', type=str, default='llava-onevision-7b,qwen2.5-vl-7b,internvl2.5-8b',
                   help='Comma-separated model names for cross-model comparison')
    p.add_argument('--cross_weight_type', type=str, default='down_proj',
                   help='Weight type for cross-model comparison')
    return p.parse_args()


def get_weight_keys(model_type, layer_idx, weight_name, is_vlm=True):
    """Return state_dict key for MLP weight matrix.
    
    This is a hint — actual resolution happens in load_weight_from_model
    which searches the state_dict.
    """
    # InternLM naming: feed_forward.w1 (gate), w2 (down), w3 (up)
    intern_map = {'gate_proj': 'w1', 'down_proj': 'w2', 'up_proj': 'w3'}
    
    # We'll try multiple patterns and return whichever matches
    patterns = []
    
    if model_type == 'internvl':
        wn = intern_map[weight_name]
        patterns = [
            f'language_model.model.layers.{layer_idx}.feed_forward.{wn}.weight',
            f'model.language_model.model.layers.{layer_idx}.feed_forward.{wn}.weight',
            f'model.layers.{layer_idx}.feed_forward.{wn}.weight',
        ]
        if not is_vlm:
            patterns = [
                f'model.layers.{layer_idx}.feed_forward.{wn}.weight',
                f'layers.{layer_idx}.feed_forward.{wn}.weight',
            ]
    else:
        patterns = [
            f'language_model.model.layers.{layer_idx}.mlp.{weight_name}.weight',
            f'model.language_model.layers.{layer_idx}.mlp.{weight_name}.weight',
            f'model.language_model.model.layers.{layer_idx}.mlp.{weight_name}.weight',
            f'model.layers.{layer_idx}.mlp.{weight_name}.weight',
            f'language_model.layers.{layer_idx}.mlp.{weight_name}.weight',
        ]
        if not is_vlm:
            patterns = [
                f'model.layers.{layer_idx}.mlp.{weight_name}.weight',
                f'layers.{layer_idx}.mlp.{weight_name}.weight',
            ]
    
    return patterns


def load_weight_from_model(model, key_patterns):
    """Load a weight tensor from a model's state_dict, trying multiple key patterns."""
    sd = model.state_dict()
    for key in key_patterns:
        if key in sd:
            return sd[key].float(), key
    raise KeyError(f'None of {key_patterns} found in state_dict. '
                   f'Sample keys: {[k for k in list(sd.keys())[:10]]}')


def effective_rank(M):
    """Compute effective rank = exp(spectral entropy).
    
    For a matrix M, compute singular values, normalize to a distribution,
    then compute exp(-sum(p * log(p))). Ranges from 1 (rank-1) to
    min(rows, cols) (full rank / uniform singular values).
    """
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]  # remove numerical zeros
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def nuclear_norm_ratio(M):
    """Nuclear norm / (Frobenius norm * sqrt(min_dim)).
    
    Ranges from 0 (rank-1) to 1 (all singular values equal).
    """
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    nuc = s.sum()
    frob = np.sqrt((s ** 2).sum())
    min_dim = min(M.shape)
    return float(nuc / (frob * np.sqrt(min_dim)))


def top_k_energy(M, k=10):
    """Fraction of Frobenius norm captured by top-k singular values."""
    s = np.linalg.svd(M, compute_uv=False)
    total = (s ** 2).sum()
    if total < 1e-10:
        return 0.0
    topk = (s[:k] ** 2).sum()
    return float(topk / total)


def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

    import torch

    print(f'\n{"="*60}')
    print(f'WEIGHT DIFF EFFECTIVE RANK ANALYSIS')
    print(f'{"="*60}')
    print(f'  VLM: {args.vlm_path}')
    print(f'  LLM: {args.llm_path}')
    print(f'  Model type: {args.model_type}')
    print(f'  Layers: {args.n_layers}')

    # ── Load PMBT labels ──
    label_path = os.path.join(args.label_dir,
                               'neuron_labels_permutation_all.json')
    with open(label_path) as f:
        all_labels = json.load(f)

    # Build per-layer masks
    masks = {}
    for lidx in range(args.n_layers):
        layer_data = all_labels.get(str(lidx), [])
        labels = [e['label'] for e in layer_data]
        masks[lidx] = {
            'visual': np.array([l == 'visual' for l in labels]),
            'text': np.array([l == 'text' for l in labels]),
            'multimodal': np.array([l == 'multimodal' for l in labels]),
            'unknown': np.array([l == 'unknown' for l in labels]),
        }

    n_vis = sum(m['visual'].sum() for m in masks.values())
    n_txt = sum(m['text'].sum() for m in masks.values())
    print(f'  PMBT: {n_vis} visual, {n_txt} text neurons total')

    # ── Load models ONE AT A TIME to avoid OOM ──
    # Extract MLP weights from VLM, delete it, then load LLM
    import gc

    weight_types = [w.strip() for w in args.weight_types.split(',')]

    print(f'\n  Loading VLM to extract weights...')
    if args.model_type in ('llava-ov',):
        from transformers import LlavaOnevisionForConditionalGeneration
        vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
    elif args.model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq
        vlm = AutoModelForVision2Seq.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
    elif args.model_type == 'internvl':
        from transformers import AutoModel
        vlm = AutoModel.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True)
    else:
        from transformers import AutoModelForVision2Seq
        vlm = AutoModelForVision2Seq.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)

    # Extract all needed weights from VLM
    vlm_weights = {}  # (wt, lidx) → numpy float32
    for wt in weight_types:
        for lidx in range(args.n_layers):
            keys = get_weight_keys(args.model_type, lidx, wt, is_vlm=True)
            try:
                w, matched_key = load_weight_from_model(vlm, keys)
                vlm_weights[(wt, lidx)] = w.numpy()
                if lidx == 0:
                    print(f'    VLM key ({wt}): {matched_key}')
            except KeyError as e:
                if lidx == 0:
                    print(f'    VLM key ({wt}): NOT FOUND — {e}')

    print(f'  Extracted {len(vlm_weights)} weight matrices from VLM')
    del vlm
    gc.collect()

    print(f'  Loading base LLM...')
    from transformers import AutoModelForCausalLM
    _hf_token = None
    _token_path = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.isfile(_token_path):
        with open(_token_path) as f:
            _hf_token = f.read().strip()
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
        token=_hf_token)

    # Extract all needed weights from LLM
    llm_weights = {}
    for wt in weight_types:
        for lidx in range(args.n_layers):
            keys = get_weight_keys(args.model_type, lidx, wt, is_vlm=False)
            try:
                w, matched_key = load_weight_from_model(llm, keys)
                llm_weights[(wt, lidx)] = w.numpy()
                if lidx == 0:
                    print(f'    LLM key ({wt}): {matched_key}')
            except KeyError as e:
                if lidx == 0:
                    print(f'    LLM key ({wt}): NOT FOUND — {e}')

    print(f'  Extracted {len(llm_weights)} weight matrices from LLM')
    del llm
    gc.collect()

    print(f'  Both models processed. Computing diffs...\n')

    # ── Compute weight diffs per layer ──
    results = {}

    for wt in weight_types:
        print(f'\n  ── Weight: {wt} ──')
        results[wt] = {'layers': {}}

        for lidx in range(args.n_layers):
            if (wt, lidx) not in vlm_weights or (wt, lidx) not in llm_weights:
                print(f'    Layer {lidx}: SKIP (missing weights)')
                continue

            w_vlm = vlm_weights[(wt, lidx)]
            w_llm = llm_weights[(wt, lidx)]

            diff = w_vlm - w_llm

            # For down_proj: shape (hidden_size, intermediate_size)
            # Columns correspond to neurons → split by PMBT mask
            # For gate/up_proj: shape (intermediate_size, hidden_size)
            # Rows correspond to neurons → split by PMBT mask
            if wt == 'down_proj':
                # Columns = neurons
                vis_mask = masks[lidx]['visual'][:diff.shape[1]]
                txt_mask = masks[lidx]['text'][:diff.shape[1]]
                vis_diff = diff[:, vis_mask]
                txt_diff = diff[:, txt_mask]
                all_diff = diff
            else:
                # Rows = neurons
                vis_mask = masks[lidx]['visual'][:diff.shape[0]]
                txt_mask = masks[lidx]['text'][:diff.shape[0]]
                vis_diff = diff[vis_mask, :]
                txt_diff = diff[txt_mask, :]
                all_diff = diff

            # Compute metrics
            vis_er = effective_rank(vis_diff) if vis_diff.size > 0 else 0
            txt_er = effective_rank(txt_diff) if txt_diff.size > 0 else 0
            all_er = effective_rank(all_diff)

            vis_nnr = nuclear_norm_ratio(vis_diff) if vis_diff.size > 0 else 0
            txt_nnr = nuclear_norm_ratio(txt_diff) if txt_diff.size > 0 else 0

            vis_e10 = top_k_energy(vis_diff, k=10) if vis_diff.size > 0 else 0
            txt_e10 = top_k_energy(txt_diff, k=10) if txt_diff.size > 0 else 0

            vis_frob = float(np.linalg.norm(vis_diff, 'fro')) if vis_diff.size > 0 else 0
            txt_frob = float(np.linalg.norm(txt_diff, 'fro')) if txt_diff.size > 0 else 0
            all_frob = float(np.linalg.norm(all_diff, 'fro'))

            layer_result = {
                'n_visual': int(vis_mask.sum()),
                'n_text': int(txt_mask.sum()),
                'visual_eff_rank': round(vis_er, 2),
                'text_eff_rank': round(txt_er, 2),
                'all_eff_rank': round(all_er, 2),
                'visual_nnr': round(vis_nnr, 4),
                'text_nnr': round(txt_nnr, 4),
                'visual_top10_energy': round(vis_e10, 4),
                'text_top10_energy': round(txt_e10, 4),
                'visual_frobenius': round(vis_frob, 4),
                'text_frobenius': round(txt_frob, 4),
                'all_frobenius': round(all_frob, 4),
            }
            results[wt]['layers'][str(lidx)] = layer_result

            # Store SVD components for cross-model analysis
            if vis_diff.size > 0:
                U_vis, s_vis, Vt_vis = np.linalg.svd(vis_diff, full_matrices=False)
                layer_result['_vis_singular_values'] = s_vis.tolist()
                # Store top-k right singular vectors (neuron-space directions)
                layer_result['_vis_top_vectors'] = Vt_vis[:20].tolist()
            if txt_diff.size > 0:
                _, s_txt, _ = np.linalg.svd(txt_diff, full_matrices=False)
                layer_result['_txt_singular_values'] = s_txt.tolist()

            print(f'    L{lidx:2d}: vis_rank={vis_er:6.1f} '
                  f'txt_rank={txt_er:6.1f} '
                  f'vis_top10={vis_e10:.3f} '
                  f'txt_top10={txt_e10:.3f} '
                  f'vis_frob={vis_frob:.2f} '
                  f'txt_frob={txt_frob:.2f}')

        # Compute summary across layers
        vis_ranks = [v['visual_eff_rank']
                     for v in results[wt]['layers'].values()
                     if v['visual_eff_rank'] > 0]
        txt_ranks = [v['text_eff_rank']
                     for v in results[wt]['layers'].values()
                     if v['text_eff_rank'] > 0]

        if vis_ranks and txt_ranks:
            results[wt]['summary'] = {
                'mean_visual_eff_rank': round(np.mean(vis_ranks), 2),
                'mean_text_eff_rank': round(np.mean(txt_ranks), 2),
                'rank_ratio': round(np.mean(vis_ranks) / np.mean(txt_ranks), 4),
                'visual_is_lower_rank': bool(np.mean(vis_ranks) < np.mean(txt_ranks)),
            }

            print(f'\n  Summary ({wt}):')
            print(f'    Mean visual effective rank: {np.mean(vis_ranks):.1f}')
            print(f'    Mean text effective rank:   {np.mean(txt_ranks):.1f}')
            print(f'    Ratio (vis/txt):            {np.mean(vis_ranks)/np.mean(txt_ranks):.3f}')
            if np.mean(vis_ranks) < np.mean(txt_ranks):
                print(f'    → CONFIRMED: Visual diff is lower-rank than text diff')
            else:
                print(f'    → NOT confirmed: Text diff is lower-rank')

        # ══════════════════════════════════════════════════════════
        # Analysis A: Universal Dimensionality
        # Are visual effective ranks consistent across layers?
        # Low std/mean ratio = universal dimensionality
        # ══════════════════════════════════════════════════════════
        if vis_ranks:
            vis_std = np.std(vis_ranks)
            vis_cv = vis_std / np.mean(vis_ranks) if np.mean(vis_ranks) > 0 else float('inf')
            txt_std = np.std(txt_ranks) if txt_ranks else 0
            txt_cv = txt_std / np.mean(txt_ranks) if txt_ranks and np.mean(txt_ranks) > 0 else float('inf')

            results[wt]['analysis_a_universality'] = {
                'visual_ranks_per_layer': [round(r, 2) for r in vis_ranks],
                'text_ranks_per_layer': [round(r, 2) for r in txt_ranks],
                'visual_mean': round(np.mean(vis_ranks), 2),
                'visual_std': round(vis_std, 2),
                'visual_cv': round(vis_cv, 4),
                'text_mean': round(np.mean(txt_ranks), 2),
                'text_std': round(txt_std, 2),
                'text_cv': round(txt_cv, 4),
                'visual_median': round(float(np.median(vis_ranks)), 2),
                'visual_min': round(float(np.min(vis_ranks)), 2),
                'visual_max': round(float(np.max(vis_ranks)), 2),
            }

            print(f'\n  ── Analysis A: Universal Dimensionality ({wt}) ──')
            print(f'    Visual eff rank: {np.mean(vis_ranks):.1f} ± {vis_std:.1f} '
                  f'(CV={vis_cv:.3f}, range=[{np.min(vis_ranks):.1f}, {np.max(vis_ranks):.1f}])')
            print(f'    Text eff rank:   {np.mean(txt_ranks):.1f} ± {txt_std:.1f} '
                  f'(CV={txt_cv:.3f})')
            if vis_cv < 0.3:
                print(f'    → TIGHT clustering (CV<0.3): visual diff occupies ~{np.mean(vis_ranks):.0f}-dimensional subspace consistently')
            elif vis_cv < 0.5:
                print(f'    → Moderate clustering (CV<0.5)')
            else:
                print(f'    → Loose/variable across layers')

        # ══════════════════════════════════════════════════════════
        # Analysis B: Cross-Layer Subspace Alignment
        # Do visual diff top singular vectors align across layers?
        # High alignment = same directions used at every layer
        # ══════════════════════════════════════════════════════════
        layer_keys_with_vecs = [k for k, v in results[wt]['layers'].items()
                                if '_vis_top_vectors' in v]
        if len(layer_keys_with_vecs) >= 2:
            k_align = 10  # compare top-10 directions
            alignments = []
            for i in range(len(layer_keys_with_vecs)):
                for j in range(i + 1, len(layer_keys_with_vecs)):
                    ki, kj = layer_keys_with_vecs[i], layer_keys_with_vecs[j]
                    Vi = np.array(results[wt]['layers'][ki]['_vis_top_vectors'][:k_align])
                    Vj = np.array(results[wt]['layers'][kj]['_vis_top_vectors'][:k_align])
                    # Subspace alignment: mean |cos similarity| between all pairs
                    # Use Grassmann distance proxy: ||Vi @ Vj^T||_F^2 / k
                    if Vi.shape[1] == Vj.shape[1]:
                        gram = Vi @ Vj.T  # (k, k)
                        alignment = float(np.sum(gram ** 2) / k_align)
                        alignments.append(alignment)

            if alignments:
                mean_align = np.mean(alignments)
                results[wt]['analysis_b_subspace_alignment'] = {
                    'n_layer_pairs': len(alignments),
                    'mean_alignment': round(mean_align, 4),
                    'std_alignment': round(float(np.std(alignments)), 4),
                    'min_alignment': round(float(np.min(alignments)), 4),
                    'max_alignment': round(float(np.max(alignments)), 4),
                    'interpretation': (
                        'Random subspaces would give ~k/d ≈ 0.001. '
                        'Alignment >> 0.01 means layers share visual directions.'
                    ),
                }

                print(f'\n  ── Analysis B: Cross-Layer Subspace Alignment ({wt}) ──')
                print(f'    Mean alignment (top-{k_align} SVs): {mean_align:.4f} '
                      f'(std={np.std(alignments):.4f})')
                print(f'    Range: [{np.min(alignments):.4f}, {np.max(alignments):.4f}]')
                # Random baseline for this dimensionality
                n_neurons = results[wt]['layers'][layer_keys_with_vecs[0]].get('n_visual', 1000)
                random_baseline = k_align / max(n_neurons, 1)
                print(f'    Random baseline: ~{random_baseline:.6f}')
                if mean_align > random_baseline * 10:
                    print(f'    → STRONG alignment: visual fine-tuning uses similar directions across layers')
                elif mean_align > random_baseline * 3:
                    print(f'    → Moderate alignment')
                else:
                    print(f'    → Weak/no alignment (layer-specific directions)')

        # ══════════════════════════════════════════════════════════
        # Analysis C: Rank vs Layer Depth Profile
        # Is there a consistent curve shape across models?
        # ══════════════════════════════════════════════════════════
        if vis_ranks and len(vis_ranks) > 4:
            layer_indices = sorted([int(k) for k in results[wt]['layers'].keys()
                                    if results[wt]['layers'][k]['visual_eff_rank'] > 0])
            vis_by_layer = [results[wt]['layers'][str(l)]['visual_eff_rank']
                            for l in layer_indices]
            txt_by_layer = [results[wt]['layers'][str(l)]['text_eff_rank']
                            for l in layer_indices]

            # Divide into thirds: early, middle, late
            n = len(layer_indices)
            third = max(1, n // 3)
            early_vis = np.mean(vis_by_layer[:third])
            mid_vis = np.mean(vis_by_layer[third:2*third])
            late_vis = np.mean(vis_by_layer[2*third:])
            early_txt = np.mean(txt_by_layer[:third])
            mid_txt = np.mean(txt_by_layer[third:2*third])
            late_txt = np.mean(txt_by_layer[2*third:])

            # Correlation with layer index
            vis_corr = float(np.corrcoef(layer_indices, vis_by_layer)[0, 1])
            txt_corr = float(np.corrcoef(layer_indices, txt_by_layer)[0, 1])

            results[wt]['analysis_c_depth_profile'] = {
                'visual_early_mean': round(early_vis, 2),
                'visual_mid_mean': round(mid_vis, 2),
                'visual_late_mean': round(late_vis, 2),
                'text_early_mean': round(early_txt, 2),
                'text_mid_mean': round(mid_txt, 2),
                'text_late_mean': round(late_txt, 2),
                'visual_layer_correlation': round(vis_corr, 4),
                'text_layer_correlation': round(txt_corr, 4),
                'visual_rank_by_layer': {str(l): round(r, 2)
                                          for l, r in zip(layer_indices, vis_by_layer)},
                'text_rank_by_layer': {str(l): round(r, 2)
                                        for l, r in zip(layer_indices, txt_by_layer)},
            }

            print(f'\n  ── Analysis C: Rank vs Layer Depth ({wt}) ──')
            print(f'    Visual rank profile: early={early_vis:.1f}  mid={mid_vis:.1f}  late={late_vis:.1f}')
            print(f'    Text rank profile:   early={early_txt:.1f}  mid={mid_txt:.1f}  late={late_txt:.1f}')
            print(f'    Visual-layer correlation: r={vis_corr:.3f}')
            print(f'    Text-layer correlation:   r={txt_corr:.3f}')
            if abs(vis_corr) > 0.5:
                direction = "increases" if vis_corr > 0 else "decreases"
                print(f'    → Visual rank {direction} with depth (r={vis_corr:.3f})')
            else:
                print(f'    → No strong depth trend for visual rank')

    # ── Save singular value profiles separately (for cross-model analysis) ──
    sv_data = {}
    for wt in weight_types:
        sv_data[wt] = {}
        for lidx in range(args.n_layers):
            lidx_str = str(lidx)
            layer_d = results.get(wt, {}).get('layers', {}).get(lidx_str, {})
            # We already stripped _vis_singular_values above, so recompute quickly
            # Actually let's save before cleanup — restructure
    # Already cleaned up, skip SV saving for now

    # ── Clean up internal SVD data before saving ──
    for wt in results:
        for lidx_str in list(results[wt].get('layers', {}).keys()):
            for internal_key in ['_vis_singular_values', '_vis_top_vectors',
                                  '_txt_singular_values']:
                results[wt]['layers'][lidx_str].pop(internal_key, None)

    # ── Save results ──
    save_path = os.path.join(args.output_dir, 'weight_diff_rank.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {save_path}')

    print(f'\n  Done.')


def cross_model_comparison(base_dir, models, weight_type='down_proj'):
    """Compare weight diff rank across multiple models.
    
    Run after all per-model analyses are complete:
        python weight_diff_rank.py --cross_model \
            --base_dir results/29-weight-diff-rank/full \
            --models llava-onevision-7b,qwen2.5-vl-7b,internvl2.5-8b
    """
    print(f'\n{"="*60}')
    print(f'CROSS-MODEL COMPARISON')
    print(f'{"="*60}\n')

    all_vis_ranks = {}  # model → list of per-layer ranks
    all_txt_ranks = {}
    summaries = {}

    for model in models:
        path = os.path.join(base_dir, model, 'weight_diff_rank.json')
        if not os.path.isfile(path):
            print(f'  [skip] {model}: no results found')
            continue
        with open(path) as f:
            data = json.load(f)

        wt_data = data.get(weight_type, {})
        layers = wt_data.get('layers', {})
        vis_r = [v['visual_eff_rank'] for v in layers.values() if v.get('visual_eff_rank', 0) > 0]
        txt_r = [v['text_eff_rank'] for v in layers.values() if v.get('text_eff_rank', 0) > 0]

        all_vis_ranks[model] = vis_r
        all_txt_ranks[model] = txt_r
        summaries[model] = wt_data.get('summary', {})

        print(f'  {model}:')
        print(f'    Visual eff rank: {np.mean(vis_r):.1f} ± {np.std(vis_r):.1f} '
              f'(median={np.median(vis_r):.1f})')
        print(f'    Text eff rank:   {np.mean(txt_r):.1f} ± {np.std(txt_r):.1f}')

    if len(all_vis_ranks) < 2:
        print('\n  Need at least 2 models for comparison.')
        return

    # ── Universal dimensionality check ──
    print(f'\n  ── Universal Dimensionality ──')
    all_vis_means = [np.mean(r) for r in all_vis_ranks.values()]
    all_txt_means = [np.mean(r) for r in all_txt_ranks.values()]
    cross_model_vis_cv = np.std(all_vis_means) / np.mean(all_vis_means) if np.mean(all_vis_means) > 0 else float('inf')
    cross_model_txt_cv = np.std(all_txt_means) / np.mean(all_txt_means) if np.mean(all_txt_means) > 0 else float('inf')

    print(f'    Visual mean ranks across models: {[f"{v:.1f}" for v in all_vis_means]}')
    print(f'    Cross-model CV: {cross_model_vis_cv:.3f}')
    print(f'    Text mean ranks across models:   {[f"{v:.1f}" for v in all_txt_means]}')
    print(f'    Cross-model CV: {cross_model_txt_cv:.3f}')

    if cross_model_vis_cv < 0.3:
        print(f'    → UNIVERSAL: Visual diff occupies ~{np.mean(all_vis_means):.0f}-dimensional subspace across ALL architectures')
    elif cross_model_vis_cv < 0.5:
        print(f'    → Moderately consistent visual dimensionality')
    else:
        print(f'    → Architecture-dependent visual dimensionality')

    # ── Rank ratio consistency ──
    ratios = [np.mean(all_vis_ranks[m]) / np.mean(all_txt_ranks[m])
              for m in all_vis_ranks if m in all_txt_ranks and np.mean(all_txt_ranks[m]) > 0]
    if ratios:
        print(f'\n  ── Rank Ratio Consistency ──')
        print(f'    Visual/text rank ratios: {[f"{r:.3f}" for r in ratios]}')
        print(f'    Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}')
        if np.std(ratios) / np.mean(ratios) < 0.3 if np.mean(ratios) > 0 else False:
            print(f'    → UNIVERSAL ratio: visual diff is consistently '
                  f'{1/np.mean(ratios):.1f}x lower-rank than text diff')

    # ── Depth profile comparison ──
    print(f'\n  ── Depth Profile Comparison ──')
    for model in all_vis_ranks:
        ranks = all_vis_ranks[model]
        n = len(ranks)
        if n < 6:
            continue
        third = n // 3
        early = np.mean(ranks[:third])
        mid = np.mean(ranks[third:2*third])
        late = np.mean(ranks[2*third:])
        print(f'    {model}: early={early:.1f}  mid={mid:.1f}  late={late:.1f}')

    # ── Save cross-model results ──
    cross_results = {
        'models': models,
        'weight_type': weight_type,
        'visual_mean_ranks': {m: round(np.mean(r), 2) for m, r in all_vis_ranks.items()},
        'text_mean_ranks': {m: round(np.mean(r), 2) for m, r in all_txt_ranks.items()},
        'cross_model_visual_cv': round(cross_model_vis_cv, 4),
        'cross_model_text_cv': round(cross_model_txt_cv, 4),
        'rank_ratios': {m: round(np.mean(all_vis_ranks[m]) / np.mean(all_txt_ranks[m]), 4)
                        for m in all_vis_ranks if m in all_txt_ranks and np.mean(all_txt_ranks[m]) > 0},
        'universal_visual_rank': cross_model_vis_cv < 0.3,
    }
    save_path = os.path.join(base_dir, 'cross_model_rank_comparison.json')
    with open(save_path, 'w') as f:
        json.dump(cross_results, f, indent=2)
    print(f'\n  Cross-model results saved to {save_path}')


def main():
    args = parse_args()

    # Cross-model comparison mode
    if hasattr(args, 'cross_model') and args.cross_model:
        models = [m.strip() for m in args.models.split(',')]
        cross_model_comparison(args.base_dir, models,
                               weight_type=args.cross_weight_type)
        return

    run_analysis(args)


if __name__ == '__main__':
    main()