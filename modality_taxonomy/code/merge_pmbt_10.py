"""PMBT-guided selective weight merging for VLMs.

Based on BRV's merge.py (linear interpolation), extended with per-neuron
alpha masks derived from PMBT (Permutation-Based Modality Taxonomy) labels.

SIGNAL MODES (--signal_mode):
  label          : Categorical PMBT (original behavior). Each neuron gets α_text,
                   α_visual, α_multimodal, or α_other based on its categorical label.
  confw_pvalue   : A2 confidence-weighted soft α using (1 - p_value) as confidence.
                   α_j = 1 - conf_j * (1 - α_label). [Back-compat: same as --confidence_weighted.]
  A_rd           : Path A with rate_diff as confidence source.
                   conf_j = min(|rate_diff_j| / C, 1.0);  α_j = 1 - conf_j * (1 - α_label).
                   Graded within category, but structurally limited when α_label=1.0.
  B_pure         : Path B continuous α from signed rate_diff.
                   r_j = clip(rate_diff_j / C, -1, 1)
                   α_j = interp between α_text (r=-1) / α_multi (r=0) / α_visual (r=+1).
                   Abandons categorical labels — every neuron is on a 1D modality axis.
  B_rdxnorm      : Path B × norm. Uses (rate_diff_j * norm_j) as the modality signal.
                   Weights modality preference by functional importance.
  B_gated        : Path B norm-gated. rate_diff sets direction, norm sets strength.
                   α_j = α_multi + w_j * (α_continuous_j - α_multi), where
                         w_j = min(norm_j / N_scale, 1.0).

The --signal_scale_C controls the rate_diff scale. Default depends on the model
   and can be tuned via the probe script (see sweep_rd_probe.sh).
The --signal_norm_scale controls the norm scale (only used by B_rdxnorm / B_gated).

Typical usage:
    # Categorical PMBT (original)
    python merge_pmbt.py --model1_path ... --model2_path ... \\
        --output_dir ... --alpha 1.0 --mode pmbt \\
        --pmbt_labels ... --alpha_text 0.7 --alpha_visual 1.0 --alpha_multimodal 1.0 \\
        --signal_mode label

    # Path A with rate_diff confidence, scale C=0.1
    python merge_pmbt.py ... --signal_mode A_rd --signal_scale_C 0.1

    # Path B continuous, scale C=0.1
    python merge_pmbt.py ... --signal_mode B_pure --signal_scale_C 0.1

    # Path B × norm
    python merge_pmbt.py ... --signal_mode B_rdxnorm --signal_scale_C 0.1 --signal_norm_scale 1.0
"""

import torch
import os
import json
import argparse
import re
from tqdm import tqdm
from transformers import (
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    Qwen2VLForConditionalGeneration,
)


def extract_layer_number(key):
    """Extract layer number from a state_dict key (BRV-compatible)."""
    match = re.search(r'\d+', key)
    return int(match.group()) if match else None


def load_pmbt_labels(label_path):
    """Load PMBT neuron labels from JSON.

    Returns:
        dict: {layer_int: {neuron_idx_int: (category_str, confidence_float, rate_diff_float)}}

    Where:
        - category_str is the classifier label ('text'/'visual'/'multimodal'/'unknown')
        - confidence_float = 1 - p_value  (clamped to [0, 1])
        - rate_diff_float = observed_rate_diff (signed; >0 means visual-leaning,
                                                      <0 means text-leaning)
    """
    with open(label_path) as f:
        raw = json.load(f)

    labels = {}
    for layer_key, layer_data in raw.items():
        layer_num = int(layer_key)
        neuron_map = {}

        if isinstance(layer_data, list):
            for info in layer_data:
                idx = info['neuron_idx']
                cat = info.get('label', 'unknown')
                p_value = info.get('p_value', 1.0)
                rate_diff = info.get('observed_rate_diff', 0.0)
                conf = max(0.0, min(1.0, 1.0 - p_value))
                neuron_map[idx] = (cat, conf, rate_diff)
        elif isinstance(layer_data, dict):
            for idx_str, info in layer_data.items():
                if isinstance(info, dict):
                    cat = info.get('label', info.get('category', 'unknown'))
                    p_value = info.get('p_value', 1.0)
                    rate_diff = info.get('observed_rate_diff', 0.0)
                    conf = max(0.0, min(1.0, 1.0 - p_value))
                    neuron_map[int(idx_str)] = (cat, conf, rate_diff)
                elif isinstance(info, str):
                    # Plain string label — no p_value or rate_diff available
                    neuron_map[int(idx_str)] = (info, 1.0, 0.0)

        labels[layer_num] = neuron_map

    # Stats
    total = sum(len(v) for v in labels.values())
    cats = {}
    conf_sum, rd_absum = 0.0, 0.0
    for lm in labels.values():
        for val in lm.values():
            cat, conf, rd = val
            cats[cat] = cats.get(cat, 0) + 1
            conf_sum += conf
            rd_absum += abs(rd)
    print(f'  PMBT labels: {total} neurons across {len(labels)} layers')
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f'    {c}: {n} ({100*n/total:.1f}%)')
    if total:
        print(f'    mean confidence (1-p): {conf_sum/total:.3f}')
        print(f'    mean |rate_diff|:      {rd_absum/total:.4f}')
    return labels


def _alpha_from_signal(cat, conf, rate_diff, norm_val,
                      alpha_text, alpha_visual, alpha_multimodal, alpha_default,
                      signal_mode, scale_C, norm_scale):
    """Compute α for a single neuron given all signals and the mode.

    Returns:
        float α_j in [0, 1] (usually)

    This is the central formula dispatcher. All per-neuron α computations flow through here.
    """
    # ── Modes that ignore rate_diff and norm ──
    if signal_mode == 'label':
        # Categorical PMBT
        if cat == 'text':        return alpha_text
        if cat == 'visual':      return alpha_visual
        if cat == 'multimodal':  return alpha_multimodal
        return alpha_default  # unknown

    if signal_mode == 'confw_pvalue':
        # A2: α = 1 - conf * (1 - α_label),  conf = 1 - p_value
        if cat == 'text':        alpha_label = alpha_text
        elif cat == 'visual':    alpha_label = alpha_visual
        elif cat == 'multimodal': alpha_label = alpha_multimodal
        else:                    alpha_label = alpha_default
        return 1.0 - conf * (1.0 - alpha_label)

    # ── Modes using rate_diff ──
    # For A_rd: unsigned |rate_diff| / C → confidence scalar
    # For B_*:  signed   rate_diff / C  → position on modality axis

    if signal_mode == 'A_rd':
        # Path A: same formula as confw_pvalue but conf from |rate_diff|
        conf_from_rd = min(abs(rate_diff) / max(scale_C, 1e-8), 1.0)
        if cat == 'text':        alpha_label = alpha_text
        elif cat == 'visual':    alpha_label = alpha_visual
        elif cat == 'multimodal': alpha_label = alpha_multimodal
        else:                    alpha_label = alpha_default
        return 1.0 - conf_from_rd * (1.0 - alpha_label)

    # ── Path B variants: continuous α on [α_text, α_multi, α_visual] axis ──

    if cat == 'unknown':
        # Unknown neurons (low activation) bypass continuous formula
        return alpha_default

    # Build the rate_diff-based modality signal per variant
    if signal_mode == 'B_pure':
        signal = rate_diff
    elif signal_mode == 'B_rdxnorm':
        signal = rate_diff * norm_val  # weighted by functional importance
    elif signal_mode == 'B_gated':
        # direction comes from rate_diff; strength gates by norm
        signal = rate_diff  # direction only; gating applied below
    else:
        raise ValueError(f'Unknown signal_mode: {signal_mode}')

    r = max(-1.0, min(1.0, signal / max(scale_C, 1e-8)))

    # Interpolate between α_text (r=-1), α_multi (r=0), α_visual (r=+1)
    if r >= 0:
        alpha_continuous = alpha_multimodal * (1.0 - r) + alpha_visual * r
    else:
        alpha_continuous = alpha_multimodal * (1.0 + r) + alpha_text * (-r)

    if signal_mode == 'B_gated':
        # Gate: low-norm neurons stay at α_multi; high-norm get α_continuous
        w = min(norm_val / max(norm_scale, 1e-8), 1.0)
        return alpha_multimodal * (1.0 - w) + alpha_continuous * w

    return alpha_continuous


def build_alpha_mask_mlp(layer_num, weight_key, weight_shape, pmbt_labels,
                         alpha_text, alpha_visual, alpha_multimodal, alpha_default,
                         signal_mode='label', scale_C=0.1, norm_scale=1.0,
                         layer_norms=None):
    """Build a per-neuron alpha tensor for an MLP weight matrix.

    For gate_proj/up_proj: shape (intermediate_dim, hidden_dim)
        → each ROW corresponds to a neuron → mask along dim=0
    For down_proj: shape (hidden_dim, intermediate_dim)
        → each COLUMN corresponds to a neuron → mask along dim=1

    Args:
        layer_num, weight_key, weight_shape: weight metadata
        pmbt_labels: {layer: {neuron_idx: (cat, conf, rate_diff)}}
        alpha_text/visual/multimodal: per-category alphas
        alpha_default: fallback alpha for unlabeled neurons
        signal_mode: see _alpha_from_signal
        scale_C, norm_scale: signal mode scale parameters
        layer_norms: optional {neuron_idx: norm_float} for this layer (down_proj norms)

    Returns:
        torch.Tensor broadcastable to weight_shape, containing per-neuron alphas
    """
    layer_labels = pmbt_labels.get(layer_num, {})
    if not layer_labels:
        return torch.full(weight_shape, alpha_default)

    # Determine neuron dimension
    if 'down_proj' in weight_key:
        n_neurons = weight_shape[1]
        neuron_dim = 1
    elif 'gate_proj' in weight_key or 'up_proj' in weight_key:
        n_neurons = weight_shape[0]
        neuron_dim = 0
    else:
        return torch.full(weight_shape, alpha_default)

    # Build 1D alpha vector, one value per neuron
    alphas = torch.full((n_neurons,), alpha_default)
    layer_norms = layer_norms or {}

    for idx, val in layer_labels.items():
        if idx >= n_neurons:
            continue
        # Backward compatibility: handle 2-tuple (cat, conf) or plain string
        if isinstance(val, tuple):
            if len(val) == 3:
                cat, conf, rate_diff = val
            elif len(val) == 2:
                cat, conf = val
                rate_diff = 0.0
            else:
                cat, conf, rate_diff = val[0], 1.0, 0.0
        else:
            cat, conf, rate_diff = val, 1.0, 0.0

        norm_val = layer_norms.get(idx, 0.0)
        alphas[idx] = _alpha_from_signal(
            cat, conf, rate_diff, norm_val,
            alpha_text, alpha_visual, alpha_multimodal, alpha_default,
            signal_mode, scale_C, norm_scale,
        )

    # Reshape to broadcast
    if neuron_dim == 1:
        return alphas.unsqueeze(0)   # (1, N) broadcasts to (H, N)
    else:
        return alphas.unsqueeze(1)   # (N, 1) broadcasts to (N, H)


def is_mlp_weight(key):
    return any(s in key for s in ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight'])


def is_attn_weight(key):
    return any(s in key for s in ['q_proj.weight', 'o_proj.weight'])


def is_kv_weight(key):
    return any(s in key for s in ['k_proj.weight', 'v_proj.weight'])


def build_alpha_mask_attn(layer_num, weight_key, weight_shape, pmbt_labels_attn,
                          alpha_text, alpha_visual, alpha_multimodal, alpha_default,
                          signal_mode='label', scale_C=0.1, norm_scale=1.0):
    """Per-head attention alpha mask. Norm-based modes fall back to label for attn heads."""
    layer_labels = pmbt_labels_attn.get(layer_num, {})
    if not layer_labels:
        return torch.full(weight_shape, alpha_default)

    hidden_dim = weight_shape[0]
    n_heads = len(layer_labels)
    if n_heads == 0:
        return torch.full(weight_shape, alpha_default)
    head_dim = hidden_dim // n_heads

    # Attention heads don't have a comparable 'norm' concept — use label-level fallback
    # for B_rdxnorm / B_gated (they behave like B_pure without norm gating).
    attn_signal_mode = signal_mode
    if signal_mode in ('B_rdxnorm', 'B_gated'):
        attn_signal_mode = 'B_pure'

    def head_alpha(val):
        if isinstance(val, tuple):
            if len(val) == 3:
                cat, conf, rate_diff = val
            elif len(val) == 2:
                cat, conf = val
                rate_diff = 0.0
            else:
                cat, conf, rate_diff = val[0], 1.0, 0.0
        else:
            cat, conf, rate_diff = val, 1.0, 0.0
        return _alpha_from_signal(
            cat, conf, rate_diff, 0.0,
            alpha_text, alpha_visual, alpha_multimodal, alpha_default,
            attn_signal_mode, scale_C, norm_scale,
        )

    if 'q_proj' in weight_key:
        alphas = torch.full((hidden_dim,), alpha_default)
        for idx, val in layer_labels.items():
            start = idx * head_dim
            end = start + head_dim
            if end <= hidden_dim:
                alphas[start:end] = head_alpha(val)
        return alphas.unsqueeze(1)

    elif 'o_proj' in weight_key:
        alphas = torch.full((hidden_dim,), alpha_default)
        for idx, val in layer_labels.items():
            start = idx * head_dim
            end = start + head_dim
            if end <= hidden_dim:
                alphas[start:end] = head_alpha(val)
        return alphas.unsqueeze(0)

    else:
        return torch.full(weight_shape, alpha_default)


def build_alpha_mask_kv(layer_num, weight_key, weight_shape, pmbt_labels_attn,
                        alpha_text, alpha_visual, alpha_multimodal, alpha_default,
                        num_q_heads,
                        signal_mode='label', scale_C=0.1, norm_scale=1.0):
    """Per-KV-head alpha via majority vote from Q head labels. Uses head_alpha helper."""
    layer_labels = pmbt_labels_attn.get(layer_num, {})
    if not layer_labels:
        return torch.full(weight_shape, alpha_default)

    kv_dim = weight_shape[0]
    hidden_dim = weight_shape[1]
    head_dim = hidden_dim // num_q_heads
    num_kv_heads = kv_dim // head_dim
    group_size = num_q_heads // num_kv_heads

    attn_signal_mode = signal_mode
    if signal_mode in ('B_rdxnorm', 'B_gated'):
        attn_signal_mode = 'B_pure'

    def head_alpha(val):
        if isinstance(val, tuple):
            if len(val) == 3:
                cat, conf, rate_diff = val
            elif len(val) == 2:
                cat, conf = val
                rate_diff = 0.0
            else:
                cat, conf, rate_diff = val[0], 1.0, 0.0
        else:
            cat, conf, rate_diff = val, 1.0, 0.0
        return _alpha_from_signal(
            cat, conf, rate_diff, 0.0,
            alpha_text, alpha_visual, alpha_multimodal, alpha_default,
            attn_signal_mode, scale_C, norm_scale,
        )

    alphas = torch.full((kv_dim,), alpha_default)
    from collections import Counter
    for h_kv in range(num_kv_heads):
        q_start = h_kv * group_size
        q_end = q_start + group_size
        # Collect labels and find majority
        cat_counts = Counter()
        vals_in_group = []
        for q_h in range(q_start, q_end):
            if q_h in layer_labels:
                v = layer_labels[q_h]
                cat_only = v[0] if isinstance(v, tuple) else v
                cat_counts[cat_only] += 1
                vals_in_group.append(v)

        if cat_counts:
            majority_cat = cat_counts.most_common(1)[0][0]
            # Use the first Q head with majority label as the representative
            rep = next((v for v in vals_in_group
                       if (v[0] if isinstance(v, tuple) else v) == majority_cat),
                      vals_in_group[0])
            h_alpha = head_alpha(rep)
        else:
            h_alpha = alpha_default

        start = h_kv * head_dim
        end = start + head_dim
        if end <= kv_dim:
            alphas[start:end] = h_alpha

    return alphas.unsqueeze(1)


def precompute_down_proj_norms(state_dict, pmbt_labels):
    """For each (layer, neuron) in pmbt_labels, compute ||W_down_proj[:, neuron]||.

    Returns:
        {layer: {neuron_idx: norm_float}}
    """
    norms = {}
    down_proj_keys = [k for k in state_dict.keys() if 'down_proj.weight' in k]
    print(f'  [precompute norms] Found {len(down_proj_keys)} down_proj weights')

    for k in down_proj_keys:
        layer_num = extract_layer_number(k)
        if layer_num is None or layer_num not in pmbt_labels:
            continue
        W = state_dict[k]  # shape (hidden_dim, intermediate_dim)
        # Column norms (one per neuron)
        col_norms = W.to(dtype=torch.float32).norm(dim=0)  # (intermediate_dim,)
        layer_norms = {}
        for idx in pmbt_labels[layer_num].keys():
            if idx < col_norms.shape[0]:
                layer_norms[idx] = float(col_norms[idx])
        norms[layer_num] = layer_norms

    # Stats
    if norms:
        all_vals = [v for lm in norms.values() for v in lm.values()]
        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            mean = sum(all_vals) / len(all_vals)
            srt = sorted(all_vals)
            p50 = srt[len(srt) // 2]
            p95 = srt[min(int(0.95 * len(srt)), len(srt) - 1)]
            print(f'  [norm stats] n={len(all_vals):,}  min={mn:.4f}  mean={mean:.4f}  '
                  f'p50={p50:.4f}  p95={p95:.4f}  max={mx:.4f}')
    return norms


def merge_models(model1_path, model2_path, output_dir, alpha, mode='base',
                 base_layer_num=-1, basemodel_path='base', density=0.2, alpha2=0.2,
                 pmbt_labels_path=None, pmbt_labels_gate_path=None,
                 pmbt_labels_attn_path=None,
                 alpha_text=None, alpha_visual=None, alpha_multimodal=None,
                 alpha_other=None, kv_merge=False,
                 mlp_projs=None, merge_scope='both',
                 signal_mode='label', scale_C=0.1, norm_scale=1.0,
                 layer_start=None, layer_end=None):
    """Merge two models, optionally with PMBT-guided per-neuron alphas.

    layer_start / layer_end: if set, only merge layers in [layer_start, layer_end).
    Layers outside this range keep their VLM baseline weights (no blending).
    None means no restriction on that end. layer_end=None → merge up to last layer.
    """

    def _layer_in_range(layer_num):
        """Is this layer eligible for merging? True if in range or not a layer weight."""
        if layer_num is None:
            return True  # non-layer weights (embeddings, layernorms) — default policy applies
        if layer_start is not None and layer_num < layer_start:
            return False
        if layer_end is not None and layer_num >= layer_end:
            return False
        return True

    # ── Load models (unchanged from previous versions) ──
    print(f'\n  Loading VLM: {model1_path}')
    _lm_head = None
    if 'llava' in model1_path:
        _full_model = LlavaNextForConditionalGeneration.from_pretrained(
            model1_path, dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation="eager",
            trust_remote_code=True,
        )
        if hasattr(_full_model.language_model, 'lm_head'):
            _lm_head = _full_model.language_model.lm_head.weight.data.clone()
            print(f'  [lm_head] Found lm_head in language_model')
        elif hasattr(_full_model, 'lm_head'):
            _lm_head = _full_model.lm_head.weight.data.clone()
            print(f'  [lm_head] Found lm_head in top-level model')
        else:
            print(f'  [warn] Could not find lm_head — will use embed_tokens as tie')
            _lm_head = _full_model.language_model.get_input_embeddings().weight.data.clone()
        model1 = _full_model.language_model
        del _full_model
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path, dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation="eager",
        )
        excluded_keys = {'model.embed_tokens.weight', 'lm_head.weight'}

    elif 'idefics' in model1_path:
        model1 = AutoModelForVision2Seq.from_pretrained(
            model1_path, dtype=torch.float16,
        ).model.text_model
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path, dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation="eager",
        ).model
        excluded_keys = {'embed_tokens.weight', 'lm_head.weight'}

    elif 'Qwen' in model1_path:
        _full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model1_path, dtype='auto',
        )
        if hasattr(_full_model.model, 'language_model'):
            model1 = _full_model.model.language_model
            print(f'  [Qwen2-VL] Extracted inner .language_model')
        else:
            model1 = _full_model.model
        del _full_model
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path, dtype='auto',
        ).model
        excluded_keys = {'embed_tokens.weight'}

    else:
        raise ValueError(f'Unsupported model: {model1_path}')

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    del model1, model2
    torch.cuda.empty_cache()

    # ── Align key namespaces (unchanged) ──
    sample1 = next(iter(state_dict1))
    sample2 = next(iter(state_dict2))
    print(f'  [diag] sample VLM key: {sample1}')
    print(f'  [diag] sample LLM key: {sample2}')

    def _find_prefix(keys):
        for k in keys:
            idx = k.find('layers.')
            if idx >= 0:
                return k[:idx]
        return ''

    prefix1 = _find_prefix(state_dict1.keys())
    prefix2 = _find_prefix(state_dict2.keys())
    print(f'  [diag] VLM prefix: "{prefix1}"  LLM prefix: "{prefix2}"')

    prefix_stripped = False
    if prefix1 != prefix2:
        if prefix2 and not prefix1:
            print(f'  [key align] Stripping "{prefix2}" prefix from math LLM keys')
            state_dict2 = {k[len(prefix2):] if k.startswith(prefix2) else k: v
                           for k, v in state_dict2.items()}
            excluded_keys = {k[len(prefix2):] if k.startswith(prefix2) else k
                             for k in excluded_keys}
            prefix_stripped = prefix2
        elif prefix1 and not prefix2:
            print(f'  [key align] Adding "{prefix1}" prefix to math LLM keys')
            state_dict2 = {prefix1 + k if not k.startswith(prefix1) else k: v
                           for k, v in state_dict2.items()}
        else:
            print(f'  [key align] Replacing "{prefix2}" with "{prefix1}" in math LLM keys')
            state_dict2 = {prefix1 + k[len(prefix2):] if k.startswith(prefix2) else k: v
                           for k, v in state_dict2.items()}

    shared = set(state_dict1.keys()) & set(state_dict2.keys())
    only1 = set(state_dict1.keys()) - set(state_dict2.keys())
    only2 = set(state_dict2.keys()) - set(state_dict1.keys())
    print(f'  Keys — shared: {len(shared)}, VLM-only: {len(only1)}, LLM-only: {len(only2)}')

    # ── Load PMBT labels ──
    pmbt_labels = None
    pmbt_labels_gate = None
    pmbt_labels_attn = None
    precomputed_norms = None
    precomputed_norms_gate = None
    num_q_heads = None
    if mode == 'pmbt':
        if merge_scope != 'attn':
            assert pmbt_labels_path is not None, '--pmbt_labels required for mode=pmbt when scope includes MLP'
        assert alpha_text is not None, '--alpha_text required for mode=pmbt'
        assert alpha_visual is not None, '--alpha_visual required for mode=pmbt'
        assert alpha_multimodal is not None, '--alpha_multimodal required for mode=pmbt'
        if pmbt_labels_path is not None:
            print(f'\n  Loading PMBT MLP labels (gate_up) from {pmbt_labels_path}')
            pmbt_labels = load_pmbt_labels(pmbt_labels_path)
        if pmbt_labels_gate_path is not None:
            print(f'  Loading PMBT gate-only labels from {pmbt_labels_gate_path}')
            pmbt_labels_gate = load_pmbt_labels(pmbt_labels_gate_path)
            print(f'  Mixed-hook: gate_proj → gate labels, up_proj/down_proj → gate_up labels')
        if pmbt_labels_attn_path is not None:
            print(f'  Loading PMBT attention labels from {pmbt_labels_attn_path}')
            pmbt_labels_attn = load_pmbt_labels(pmbt_labels_attn_path)
        print(f'  Alpha values: text={alpha_text}, visual={alpha_visual}, '
              f'multimodal={alpha_multimodal}, default={alpha}')
        print(f'  Signal mode: {signal_mode}')
        if signal_mode != 'label':
            print(f'  Signal scale_C: {scale_C}')
        if signal_mode in ('B_rdxnorm', 'B_gated'):
            print(f'  Signal norm_scale: {norm_scale}')
            # Precompute down_proj norms — needed for B_rdxnorm and B_gated
            print(f'  [precompute norms] Computing down_proj column norms...')
            if pmbt_labels is not None:
                precomputed_norms = precompute_down_proj_norms(state_dict1, pmbt_labels)
            if pmbt_labels_gate is not None:
                precomputed_norms_gate = precompute_down_proj_norms(state_dict1, pmbt_labels_gate)

        if kv_merge and pmbt_labels_attn is not None:
            _max_head = max(max(int(h) for h in layer.keys()) for layer in pmbt_labels_attn.values() if layer)
            num_q_heads = _max_head + 1
            print(f'  KV merge: enabled (num_q_heads={num_q_heads}, majority vote from Q head labels)')
        elif kv_merge:
            print(f'  KV merge: requested but no attn labels — k/v will use uniform alpha')
            kv_merge = False

    # ── Merge ──
    if mode in ['ties', 'dareties', 'darelinear']:
        import merge_utils
        basemodel = AutoModelForCausalLM.from_pretrained(
            basemodel_path, dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation="eager",
        )
        if 'llava' not in model1_path:
            basemodel = basemodel.model
        state_dict_base = basemodel.state_dict()
        del basemodel

        taskvec1 = {k: state_dict1[k] - state_dict_base[k]
                    for k in state_dict1.keys() if k not in excluded_keys}
        taskvec2 = {k: state_dict2[k] - state_dict_base[k]
                    for k in state_dict2.keys() if k not in excluded_keys}
        del state_dict2

        weights = torch.tensor([alpha, alpha2 if alpha2 is not None else 1 - alpha])
        if mode == 'ties':
            mixvec = {k: merge_utils.ties([taskvec1[k], taskvec2[k]], weights, density)
                      for k in taskvec1.keys()}
        elif mode == 'dareties':
            mixvec = {k: merge_utils.dare_ties([taskvec1[k], taskvec2[k]], weights, density)
                      for k in taskvec1.keys()}
        elif mode == 'darelinear':
            mixvec = {k: merge_utils.dare_linear([taskvec1[k], taskvec2[k]], weights, density)
                      for k in taskvec1.keys()}

        state_dict1 = {k: state_dict_base[k] + mixvec[k] if k not in excluded_keys
                       else state_dict1[k] for k in state_dict_base.keys()}

    elif mode == 'pmbt':
        n_uniform = 0
        n_selective_mlp = 0
        n_selective_attn = 0
        n_selective_kv = 0
        n_skipped_mlp = 0
        n_skipped_attn = 0
        n_skipped_layer_range = 0

        for key in tqdm(list(state_dict2.keys()), desc='PMBT merge'):
            if key in excluded_keys:
                del state_dict2[key]
                continue

            layer_num = extract_layer_number(key)

            # Layer-range restriction: out-of-range layers keep VLM baseline untouched
            if layer_num is not None and not _layer_in_range(layer_num):
                n_skipped_layer_range += 1
                del state_dict2[key]
                continue

            if is_mlp_weight(key) and layer_num is not None:
                if merge_scope == 'attn':
                    n_skipped_mlp += 1
                    del state_dict2[key]
                    continue
                proj_type = None
                if 'gate_proj' in key: proj_type = 'gate'
                elif 'up_proj' in key: proj_type = 'up'
                elif 'down_proj' in key: proj_type = 'down'
                if mlp_projs is not None and proj_type not in mlp_projs:
                    n_skipped_mlp += 1
                    del state_dict2[key]
                    continue
                # Choose labels source (mixed-hook: gate_proj → gate labels)
                if pmbt_labels_gate is not None and 'gate_proj' in key:
                    _labels = pmbt_labels_gate
                    _norms = precomputed_norms_gate
                else:
                    _labels = pmbt_labels
                    _norms = precomputed_norms
                # Per-layer norms (pass only norms for this layer)
                layer_norms = _norms.get(layer_num, {}) if _norms else {}
                alpha_mask = build_alpha_mask_mlp(
                    layer_num, key, state_dict1[key].shape,
                    _labels, alpha_text, alpha_visual,
                    alpha_multimodal, alpha,
                    signal_mode=signal_mode,
                    scale_C=scale_C, norm_scale=norm_scale,
                    layer_norms=layer_norms)
                alpha_mask = alpha_mask.to(dtype=state_dict1[key].dtype,
                                          device=state_dict1[key].device)
                state_dict1[key].copy_(
                    alpha_mask * state_dict1[key] + (1 - alpha_mask) * state_dict2[key])
                n_selective_mlp += 1

            elif is_attn_weight(key) and layer_num is not None and pmbt_labels_attn is not None:
                if merge_scope == 'mlp':
                    n_skipped_attn += 1
                    del state_dict2[key]
                    continue
                alpha_mask = build_alpha_mask_attn(
                    layer_num, key, state_dict1[key].shape,
                    pmbt_labels_attn, alpha_text, alpha_visual,
                    alpha_multimodal, alpha,
                    signal_mode=signal_mode,
                    scale_C=scale_C, norm_scale=norm_scale)
                alpha_mask = alpha_mask.to(dtype=state_dict1[key].dtype,
                                          device=state_dict1[key].device)
                state_dict1[key].copy_(
                    alpha_mask * state_dict1[key] + (1 - alpha_mask) * state_dict2[key])
                n_selective_attn += 1

            elif is_kv_weight(key) and layer_num is not None and kv_merge and pmbt_labels_attn is not None:
                if merge_scope == 'mlp':
                    n_skipped_attn += 1
                    del state_dict2[key]
                    continue
                alpha_mask = build_alpha_mask_kv(
                    layer_num, key, state_dict1[key].shape,
                    pmbt_labels_attn, alpha_text, alpha_visual,
                    alpha_multimodal, alpha, num_q_heads,
                    signal_mode=signal_mode,
                    scale_C=scale_C, norm_scale=norm_scale)
                alpha_mask = alpha_mask.to(dtype=state_dict1[key].dtype,
                                          device=state_dict1[key].device)
                state_dict1[key].copy_(
                    alpha_mask * state_dict1[key] + (1 - alpha_mask) * state_dict2[key])
                n_selective_kv += 1

            else:
                _other_alpha = alpha_other if alpha_other is not None else alpha
                state_dict1[key].copy_(
                    _other_alpha * state_dict1[key] + (1 - _other_alpha) * state_dict2[key])
                n_uniform += 1

            del state_dict2[key]
        del state_dict2
        torch.cuda.empty_cache()

        _other_alpha = alpha_other if alpha_other is not None else alpha
        print(f'\n  Merged: {n_selective_mlp} MLP weights (selective), '
              f'{n_selective_attn} attention Q/O weights (selective), '
              f'{n_selective_kv} attention K/V weights (GQA majority vote), '
              f'{n_uniform} other weights (α_other={_other_alpha})')
        if n_skipped_mlp or n_skipped_attn:
            print(f'  Skipped (kept VLM baseline): {n_skipped_mlp} MLP, {n_skipped_attn} attention')
        if n_skipped_layer_range:
            _range_str = f'[{layer_start if layer_start is not None else 0}, {layer_end if layer_end is not None else "end"})'
            print(f'  Layer-range skipped (kept VLM baseline, outside {_range_str}): {n_skipped_layer_range} weights')

    else:
        # Uniform merging (base / layerswap)
        n_uni_merged = 0
        n_uni_skipped_range = 0
        for key in tqdm(list(state_dict2.keys()), desc='Uniform merge'):
            layer_number = extract_layer_number(key)
            if key not in excluded_keys:
                # Layer-range restriction: skip merging, keep VLM baseline
                if layer_number is not None and not _layer_in_range(layer_number):
                    n_uni_skipped_range += 1
                elif mode == 'layerswap':
                    if layer_number is not None and layer_number <= base_layer_num:
                        pass
                    else:
                        state_dict1[key].copy_(
                            alpha * state_dict1[key] + (1 - alpha) * state_dict2[key])
                        n_uni_merged += 1
                elif mode == 'base':
                    state_dict1[key].copy_(
                        alpha * state_dict1[key] + (1 - alpha) * state_dict2[key])
                    n_uni_merged += 1
            del state_dict2[key]
        del state_dict2
        torch.cuda.empty_cache()
        if n_uni_skipped_range:
            _range_str = f'[{layer_start if layer_start is not None else 0}, {layer_end if layer_end is not None else "end"})'
            print(f'  Uniform merge: {n_uni_merged} merged, {n_uni_skipped_range} skipped (outside {_range_str})')

    # ── Save ──
    if prefix_stripped:
        print(f'  [key restore] Adding "{prefix_stripped}" prefix back for VLMEvalKit compatibility')
        state_dict1 = {prefix_stripped + k: v for k, v in state_dict1.items()}

    if 'llava' in model1_path and _lm_head is not None:
        state_dict1['lm_head.weight'] = _lm_head
        print(f'  [lm_head] Added lm_head.weight to saved state dict')

    os.makedirs(output_dir, exist_ok=True)
    if mode == 'pmbt':
        save_name = f'merged_model_pmbt_t{alpha_text}_v{alpha_visual}_m{alpha_multimodal}.pth'
    else:
        save_name = f'merged_model_{alpha}.pth'
    save_path = os.path.join(output_dir, save_name)
    torch.save(state_dict1, save_path)
    print(f'\n  Saved: {save_path} ({os.path.getsize(save_path) / 1e9:.1f} GB)')


def main():
    parser = argparse.ArgumentParser(
        description='Merge VLM + text-FT models (BRV-compatible + PMBT-guided + signal modes)')

    parser.add_argument('--model1_path', type=str, required=True)
    parser.add_argument('--model2_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--mode', type=str, default='base',
                        choices=['base', 'pmbt', 'layerswap', 'ties', 'dareties', 'darelinear'])
    parser.add_argument('--base_layer_num', type=int, default=-1)
    parser.add_argument('--basemodel_path', type=str, default='base')
    parser.add_argument('--density', type=float, default=0.2)
    parser.add_argument('--alpha2', type=float, default=0.2)

    parser.add_argument('--pmbt_labels', type=str, default=None)
    parser.add_argument('--pmbt_labels_gate', type=str, default=None)
    parser.add_argument('--pmbt_labels_attn', type=str, default=None)
    parser.add_argument('--alpha_text', type=float, default=None)
    parser.add_argument('--alpha_visual', type=float, default=None)
    parser.add_argument('--alpha_multimodal', type=float, default=None)
    parser.add_argument('--alpha_other', type=float, default=None)
    parser.add_argument('--kv_merge', action='store_true', default=False)
    parser.add_argument('--mlp_projs', type=str, default='gate,up,down')
    parser.add_argument('--merge_scope', type=str, default='both',
                        choices=['both', 'mlp', 'attn'])

    # ── NEW: signal mode ──
    parser.add_argument('--signal_mode', type=str, default='label',
                        choices=['label', 'confw_pvalue', 'A_rd',
                                 'B_pure', 'B_rdxnorm', 'B_gated'],
                        help='Per-neuron α signal. See module docstring for details.')
    parser.add_argument('--signal_scale_C', type=float, default=0.1,
                        help='Scale parameter C for rate_diff-based signals (A_rd, B_*).')
    parser.add_argument('--signal_norm_scale', type=float, default=1.0,
                        help='Scale parameter for down_proj norm in B_rdxnorm / B_gated.')

    # ── Back-compat ──
    parser.add_argument('--confidence_weighted', action='store_true', default=False,
                        help='[deprecated] Same as --signal_mode confw_pvalue.')

    # ── Layer-range restriction ──
    parser.add_argument('--layer_start', type=int, default=None,
                        help='First layer idx (inclusive) to merge. '
                             'Layers below are kept at VLM baseline. None = no restriction.')
    parser.add_argument('--layer_end', type=int, default=None,
                        help='End layer idx (exclusive) for merging. '
                             'Layers at or above are kept at VLM baseline. None = no restriction.')

    args = parser.parse_args()

    # Handle --confidence_weighted back-compat
    if args.confidence_weighted and args.signal_mode == 'label':
        args.signal_mode = 'confw_pvalue'
    elif args.confidence_weighted and args.signal_mode != 'confw_pvalue':
        parser.error('--confidence_weighted conflicts with --signal_mode='
                     f'{args.signal_mode}. Use --signal_mode directly.')

    mlp_projs = set(p.strip() for p in args.mlp_projs.split(',') if p.strip())
    valid_projs = {'gate', 'up', 'down'}
    if not mlp_projs.issubset(valid_projs):
        parser.error(f'--mlp_projs must be comma-separated subset of {valid_projs}, '
                     f'got {mlp_projs}')

    merge_models(
        args.model1_path, args.model2_path, args.output_dir, args.alpha,
        mode=args.mode, base_layer_num=args.base_layer_num,
        basemodel_path=args.basemodel_path, density=args.density,
        alpha2=args.alpha2, pmbt_labels_path=args.pmbt_labels,
        pmbt_labels_gate_path=args.pmbt_labels_gate,
        pmbt_labels_attn_path=args.pmbt_labels_attn,
        alpha_text=args.alpha_text, alpha_visual=args.alpha_visual,
        alpha_multimodal=args.alpha_multimodal,
        alpha_other=args.alpha_other, kv_merge=args.kv_merge,
        mlp_projs=mlp_projs, merge_scope=args.merge_scope,
        signal_mode=args.signal_mode,
        scale_C=args.signal_scale_C,
        norm_scale=args.signal_norm_scale,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )


if __name__ == '__main__':
    main()