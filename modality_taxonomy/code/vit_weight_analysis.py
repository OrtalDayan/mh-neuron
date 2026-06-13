"""
vit_weight_analysis.py — Visual Instruction Tuning weight change analysis

Compares per-neuron weight changes between a base LLM (before VIT) and
the corresponding VLM (after VIT), then correlates with PMBT taxonomy labels.

Hypothesis: "visually-responsive" neurons (PMBT label = visual) are the
neurons most modified during visual instruction tuning — i.e., VIT
preferentially recruited the model's most capable neurons for visual
processing rather than developing specialised visual circuitry.

For each MLP neuron (layer l, index j), we compute:
    delta_j = || W_vlm[j, :] - W_base[j, :] ||_2

where W is the gate_proj weight matrix (shape: d_ffn × d_model).
Each row j corresponds to one neuron's input weights.

We also compute the same for down_proj (shape: d_model × d_ffn),
where each column j corresponds to one neuron's output weights.

Then we test whether visually-responsive neurons have significantly
larger delta than text-responsive neurons.

Supported model pairs:
    llava-ov:   Qwen2-7B-Instruct          → llava-onevision-qwen2-7b-ov-hf
    internvl:   InternLM2-7B               → InternVL2.5-8B
    qwen2vl:    Qwen2.5-7B-Instruct        → Qwen2.5-VL-7B-Instruct
    llava:      lmsys/vicuna-7b-v1.5       → liuhaotian/llava-v1.5-7b

Usage:
    python vit_weight_analysis.py \
        --vlm_path <path_to_vlm> \
        --base_path <path_to_base_llm> \
        --model_type llava-ov \
        --label_dir results/3-classify/full/llava-onevision-7b/llm_permutation \
        --output_dir results/26-vit-analysis/llava-onevision-7b
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Argument parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='VIT weight change analysis: correlate neuron weight '
                    'changes with PMBT taxonomy labels')

    # Model paths
    p.add_argument('--vlm_path', required=True,
                   help='Path to VLM (after visual instruction tuning)')
    p.add_argument('--base_path', required=True,
                   help='Path to base LLM (before visual instruction tuning)')
    p.add_argument('--model_type', required=True,
                   choices=['llava-ov', 'internvl', 'qwen2vl',
                            'llava-liuhaotian', 'llava-hf',
                            'llava-llama3', 'llava-mistral'],
                   help='Model type for weight key mapping')

    # PMBT labels
    p.add_argument('--label_dir', required=True,
                   help='Directory containing PMBT permutation labels '
                        '(neuron_labels_permutation_all.json or per-layer)')
    p.add_argument('--n_layers', type=int, default=None,
                   help='Number of LLM layers (auto-detected if None)')

    # Output
    p.add_argument('--output_dir', required=True,
                   help='Output directory for analysis results')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Weight key mapping per model type
# ═══════════════════════════════════════════════════════════════════

def get_weight_key_templates(model_type):
    """Return (gate_proj_template, down_proj_template, up_proj_template)
    with {layer} placeholder for each model type.

    These are the MLP weight matrices in the LLM backbone.
    For the VLM, these may be nested under a language_model prefix.
    We try both with and without the prefix.
    """
    # Line 1: define the LLM-internal key patterns (no language_model prefix)
    #         These match the base LLM's state dict keys directly
    if model_type in ('llava-ov', 'llava-hf', 'llava-liuhaotian',
                      'llava-llama3', 'llava-mistral'):
        # LLaMA / Qwen2 / Vicuna architecture
        base = 'model.layers.{layer}.mlp'
        gate = f'{base}.gate_proj.weight'
        down = f'{base}.down_proj.weight'
        up = f'{base}.up_proj.weight'
    elif model_type == 'qwen2vl':
        # Qwen2.5 architecture (same MLP structure as LLaMA)
        base = 'model.layers.{layer}.mlp'
        gate = f'{base}.gate_proj.weight'
        down = f'{base}.down_proj.weight'
        up = f'{base}.up_proj.weight'
    elif model_type == 'internvl':
        # InternLM2 architecture
        base = 'model.layers.{layer}.feed_forward'
        gate = f'{base}.w1.weight'       # gate_proj equivalent
        down = f'{base}.w2.weight'       # down_proj equivalent
        up = f'{base}.w3.weight'         # up_proj equivalent
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    return gate, down, up


# Line 2: VLM state dicts often have a language_model prefix
#          e.g. "language_model.model.layers.0.mlp.gate_proj.weight"
#          This function tries both with and without the prefix
VLM_PREFIXES = [
    '',                          # base LLM or VLM without prefix
    'language_model.',           # most HF VLMs (LLaVA, Qwen2-VL)
    'model.',                    # some VLMs nest under model.
    'language_model.model.',     # InternVL nests deeper
]


def find_weight_in_state_dict(state_dict, key_template, layer_idx):
    """Try to find a weight key in the state dict, with various prefixes.

    Args:
        state_dict: model state dict (or set of keys for efficiency)
        key_template: template string with {layer} placeholder
        layer_idx: integer layer index

    Returns:
        (actual_key, tensor) if found, else (None, None)
    """
    # Line 1: format the base key with the layer index
    base_key = key_template.format(layer=layer_idx)

    # Line 2: try each prefix until we find a match
    keys = state_dict if isinstance(state_dict, set) else state_dict.keys()
    for prefix in VLM_PREFIXES:
        candidate = prefix + base_key
        if candidate in keys:
            if isinstance(state_dict, set):
                return candidate, None   # key-only mode
            return candidate, state_dict[candidate]

    return None, None


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Load PMBT labels
# ═══════════════════════════════════════════════════════════════════

def load_pmbt_labels(label_dir, n_layers):
    """Load PMBT permutation test labels for all layers.

    Returns:
        dict {layer_idx: list of {'label': str, 'neuron_idx': int, ...}}
    """
    # Line 1: try the merged all-layers file first
    all_path = os.path.join(label_dir, 'neuron_labels_permutation_all.json')
    if os.path.isfile(all_path):
        print(f'  Loading merged labels from {all_path}')
        with open(all_path) as f:
            data = json.load(f)
        # Keys may be string layer indices ("0", "1", ...) or layer names
        labels = {}
        for key, neurons in data.items():
            # Line 2: extract integer layer index from key
            try:
                layer_idx = int(key)
            except ValueError:
                # Key is a layer name like "model.layers.0.mlp.act_fn"
                import re
                m = re.search(r'\.(\d+)\.', key)
                if m:
                    layer_idx = int(m.group(1))
                else:
                    continue
            labels[layer_idx] = neurons
        return labels

    # Line 3: fallback — load per-layer files
    print(f'  Loading per-layer labels from {label_dir}')
    labels = {}
    for l in range(n_layers):
        # Try various directory structures
        for pattern in [
            f'layer_{l}/neuron_labels_permutation.json',
            f'*layers.{l}*/neuron_labels_permutation.json',
        ]:
            import glob
            matches = glob.glob(os.path.join(label_dir, pattern))
            if matches:
                with open(matches[0]) as f:
                    labels[l] = json.load(f)
                break
        # Line 4: also try scanning subdirectories for layer index
        if l not in labels:
            for entry in os.listdir(label_dir):
                subdir = os.path.join(label_dir, entry)
                if os.path.isdir(subdir) and f'.{l}.' in entry:
                    label_file = os.path.join(
                        subdir, 'neuron_labels_permutation.json')
                    if os.path.isfile(label_file):
                        with open(label_file) as f:
                            labels[l] = json.load(f)
                        break

    print(f'  Loaded labels for {len(labels)} layers')
    return labels


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Per-neuron weight change computation
# ═══════════════════════════════════════════════════════════════════

def compute_per_neuron_delta(vlm_state, base_state, model_type, n_layers):
    """Compute per-neuron L2 weight change between VLM and base LLM.

    For each neuron j in each layer l, computes:
        delta_gate[l][j] = || W_gate_vlm[j,:] - W_gate_base[j,:] ||_2
        delta_down[l][j] = || W_down_vlm[:,j] - W_down_base[:,j] ||_2
        delta_up[l][j]   = || W_up_vlm[j,:]   - W_up_base[j,:]   ||_2

    gate_proj and up_proj have shape (d_ffn, d_model) — row j = neuron j
    down_proj has shape (d_model, d_ffn) — column j = neuron j

    Args:
        vlm_state: VLM state dict
        base_state: base LLM state dict
        model_type: model type string
        n_layers: number of LLM layers

    Returns:
        dict with keys 'gate', 'down', 'up', 'combined', each mapping to
        {layer_idx: np.array of shape (n_neurons,)}
    """
    gate_tmpl, down_tmpl, up_tmpl = get_weight_key_templates(model_type)

    # Line 1: detect key prefixes for VLM and base state dicts
    vlm_keys = set(vlm_state.keys())
    base_keys = set(base_state.keys())

    results = {
        'gate': {},     # per-neuron delta for gate_proj (input weights)
        'down': {},     # per-neuron delta for down_proj (output weights)
        'up': {},       # per-neuron delta for up_proj (parallel branch)
        'combined': {}, # sqrt(gate^2 + down^2 + up^2) per neuron
    }

    for l in range(n_layers):
        # Line 2: find matching keys in both state dicts
        deltas = {}
        for name, tmpl in [('gate', gate_tmpl), ('down', down_tmpl),
                           ('up', up_tmpl)]:
            # Find the key in VLM state dict
            vlm_key, vlm_w = find_weight_in_state_dict(vlm_state, tmpl, l)
            base_key, base_w = find_weight_in_state_dict(base_state, tmpl, l)

            if vlm_key is None or base_key is None:
                if l == 0:
                    print(f'    WARNING: could not find {name} weight for '
                          f'layer {l}')
                    print(f'      VLM key tried: {tmpl.format(layer=l)}')
                    # Line 3: show sample keys for debugging
                    sample_vlm = [k for k in list(vlm_keys)[:20]
                                  if 'mlp' in k or 'feed' in k]
                    sample_base = [k for k in list(base_keys)[:20]
                                   if 'mlp' in k or 'feed' in k]
                    print(f'      VLM sample MLP keys: {sample_vlm[:5]}')
                    print(f'      Base sample MLP keys: {sample_base[:5]}')
                deltas[name] = None
                continue

            # Line 4: compute per-neuron L2 delta
            diff = vlm_w.float() - base_w.float()

            if name == 'down':
                # down_proj: (d_model, d_ffn) — column j = neuron j
                # L2 norm per column
                delta = torch.norm(diff, dim=0).numpy()  # (d_ffn,)
            else:
                # gate_proj, up_proj: (d_ffn, d_model) — row j = neuron j
                # L2 norm per row
                delta = torch.norm(diff, dim=1).numpy()  # (d_ffn,)

            deltas[name] = delta
            results[name][l] = delta

        # Line 5: compute combined delta (RMS of all three)
        available = [deltas[k] for k in ['gate', 'down', 'up']
                     if deltas.get(k) is not None]
        if available:
            stacked = np.stack(available, axis=0)  # (n_components, d_ffn)
            combined = np.sqrt((stacked ** 2).mean(axis=0))  # (d_ffn,)
            results['combined'][l] = combined

        if l % 8 == 0:
            n_neurons = len(available[0]) if available else 0
            mean_delta = combined.mean() if available else 0
            print(f'    Layer {l:2d}: {n_neurons} neurons, '
                  f'mean combined delta = {mean_delta:.6f}')

    return results


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Statistical analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_correlation(deltas, labels, n_layers, delta_type='combined'):
    """Correlate per-neuron weight changes with PMBT labels.

    Tests:
    1. Mann-Whitney U: visual_delta > text_delta (one-sided)
    2. Effect size: Cohen's d between visual and text groups
    3. Per-layer breakdown

    Args:
        deltas: dict from compute_per_neuron_delta
        labels: dict from load_pmbt_labels
        n_layers: number of layers
        delta_type: which delta to use ('gate', 'down', 'up', 'combined')

    Returns:
        dict with analysis results
    """
    delta_dict = deltas[delta_type]

    # Line 1: collect deltas grouped by PMBT label across all layers
    all_by_label = {'visual': [], 'text': [], 'multimodal': [], 'unknown': []}
    per_layer_results = {}

    for l in range(n_layers):
        if l not in delta_dict or l not in labels:
            continue

        layer_delta = delta_dict[l]
        layer_labels = labels[l]

        # Line 2: group deltas by label for this layer
        layer_by_label = {'visual': [], 'text': [],
                          'multimodal': [], 'unknown': []}
        for neuron in layer_labels:
            idx = neuron['neuron_idx']
            label = neuron['label']
            if idx < len(layer_delta) and label in layer_by_label:
                layer_by_label[label].append(float(layer_delta[idx]))

        # Line 3: aggregate into global arrays
        for cat in all_by_label:
            all_by_label[cat].extend(layer_by_label[cat])

        # Line 4: per-layer Mann-Whitney test (visual > text)
        vis = np.array(layer_by_label['visual'])
        txt = np.array(layer_by_label['text'])
        if len(vis) >= 10 and len(txt) >= 10:
            u_stat, p_two = sp_stats.mannwhitneyu(
                vis, txt, alternative='greater')
            per_layer_results[l] = {
                'visual_mean': float(vis.mean()),
                'visual_median': float(np.median(vis)),
                'text_mean': float(txt.mean()),
                'text_median': float(np.median(txt)),
                'multimodal_mean': float(np.mean(layer_by_label['multimodal']))
                    if layer_by_label['multimodal'] else None,
                'n_visual': len(vis),
                'n_text': len(txt),
                'mann_whitney_U': float(u_stat),
                'p_value': float(p_two),
                'ratio_mean': float(vis.mean() / max(txt.mean(), 1e-12)),
            }

    # Line 5: global analysis across all layers
    vis_all = np.array(all_by_label['visual'])
    txt_all = np.array(all_by_label['text'])
    multi_all = np.array(all_by_label['multimodal'])
    unk_all = np.array(all_by_label['unknown'])

    global_results = {}
    for cat, arr in [('visual', vis_all), ('text', txt_all),
                     ('multimodal', multi_all), ('unknown', unk_all)]:
        if len(arr) > 0:
            global_results[cat] = {
                'n': len(arr),
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'std': float(arr.std()),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
            }

    # Line 6: global Mann-Whitney U test (visual > text, one-sided)
    if len(vis_all) >= 10 and len(txt_all) >= 10:
        u_stat, p_val = sp_stats.mannwhitneyu(
            vis_all, txt_all, alternative='greater')
        cohens_d = ((vis_all.mean() - txt_all.mean()) /
                    np.sqrt((vis_all.var() + txt_all.var()) / 2))
        global_results['visual_vs_text'] = {
            'mann_whitney_U': float(u_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'ratio_mean': float(vis_all.mean() / max(txt_all.mean(), 1e-12)),
            'ratio_median': float(
                np.median(vis_all) / max(np.median(txt_all), 1e-12)),
        }

    # Line 7: additional pairwise comparisons
    pairwise = []
    categories = [('visual', vis_all), ('text', txt_all),
                  ('multimodal', multi_all)]
    for i, (name_a, arr_a) in enumerate(categories):
        for name_b, arr_b in categories[i+1:]:
            if len(arr_a) >= 10 and len(arr_b) >= 10:
                u, p = sp_stats.mannwhitneyu(
                    arr_a, arr_b, alternative='greater')
                pairwise.append({
                    'comparison': f'{name_a} > {name_b}',
                    'U': float(u),
                    'p_value': float(p),
                    'mean_a': float(arr_a.mean()),
                    'mean_b': float(arr_b.mean()),
                    'ratio': float(arr_a.mean() / max(arr_b.mean(), 1e-12)),
                })

    # Line 8: point-biserial correlation (is_visual ~ delta)
    #          binary: 1 = visual, 0 = text (excluding multimodal/unknown)
    if len(vis_all) >= 10 and len(txt_all) >= 10:
        combined_delta = np.concatenate([vis_all, txt_all])
        is_visual_binary = np.concatenate([
            np.ones(len(vis_all)), np.zeros(len(txt_all))])
        r, p_corr = sp_stats.pointbiserialr(is_visual_binary, combined_delta)
        global_results['point_biserial'] = {
            'r': float(r),
            'p_value': float(p_corr),
        }

    return {
        'delta_type': delta_type,
        'global': global_results,
        'pairwise': pairwise,
        'per_layer': {str(l): v for l, v in sorted(per_layer_results.items())},
    }


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print('\n' + '═' * 60)
    print('  VIT Weight Change Analysis')
    print('═' * 60)
    print(f'  VLM:       {args.vlm_path}')
    print(f'  Base LLM:  {args.base_path}')
    print(f'  Model:     {args.model_type}')
    print(f'  Labels:    {args.label_dir}')
    print(f'  Output:    {args.output_dir}')

    # ── Step 1: Load state dicts (CPU only, no GPU needed) ────────
    print(f'\n  Loading VLM state dict...')
    vlm_state = torch.load(
        _find_state_dict(args.vlm_path), map_location='cpu')
    # Line 1: handle nested state dicts (some checkpoints wrap in 'model')
    if 'model' in vlm_state and not any(
            'layers' in k for k in vlm_state.keys()):
        vlm_state = vlm_state['model']
    print(f'    {len(vlm_state)} keys loaded')

    print(f'  Loading base LLM state dict...')
    base_state = torch.load(
        _find_state_dict(args.base_path), map_location='cpu')
    if 'model' in base_state and not any(
            'layers' in k for k in base_state.keys()):
        base_state = base_state['model']
    print(f'    {len(base_state)} keys loaded')

    # ── Step 2: Detect n_layers ──────────────────────────────────
    n_layers = args.n_layers
    if n_layers is None:
        # Line 2: count layers by finding max layer index in state dict
        import re
        max_layer = -1
        for key in list(vlm_state.keys()) + list(base_state.keys()):
            m = re.search(r'layers\.(\d+)\.', key)
            if m:
                max_layer = max(max_layer, int(m.group(1)))
        n_layers = max_layer + 1
    print(f'  Detected {n_layers} layers')

    # ── Step 3: Load PMBT labels ─────────────────────────────────
    print(f'\n  Loading PMBT labels...')
    labels = load_pmbt_labels(args.label_dir, n_layers)
    print(f'    Loaded labels for {len(labels)} layers')

    # ── Step 4: Compute per-neuron weight deltas ─────────────────
    print(f'\n  Computing per-neuron weight changes...')
    deltas = compute_per_neuron_delta(
        vlm_state, base_state, args.model_type, n_layers)

    # Line 3: free memory after computing deltas
    del vlm_state, base_state

    # ── Step 5: Statistical analysis ─────────────────────────────
    print(f'\n  Running statistical analysis...')
    all_results = {}
    for delta_type in ['gate', 'down', 'up', 'combined']:
        if deltas[delta_type]:
            print(f'\n    --- {delta_type} ---')
            result = analyze_correlation(
                deltas, labels, n_layers, delta_type)
            all_results[delta_type] = result

            # Line 4: print key results
            g = result['global']
            if 'visual_vs_text' in g:
                vt = g['visual_vs_text']
                sig = ('***' if vt['p_value'] < 0.001 else
                       '**' if vt['p_value'] < 0.01 else
                       '*' if vt['p_value'] < 0.05 else 'ns')
                print(f'    visual vs text: p={vt["p_value"]:.2e} {sig}  '
                      f'd={vt["cohens_d"]:.3f}  '
                      f'ratio={vt["ratio_mean"]:.3f}')
            if 'point_biserial' in g:
                pb = g['point_biserial']
                print(f'    point-biserial r={pb["r"]:.4f}  '
                      f'p={pb["p_value"]:.2e}')
            for cat in ['visual', 'text', 'multimodal', 'unknown']:
                if cat in g:
                    c = g[cat]
                    print(f'    {cat:12s}: mean={c["mean"]:.6f}  '
                          f'median={c["median"]:.6f}  (n={c["n"]:,})')

            # Line 5: print per-layer summary (layers where visual > text)
            layers_sig = sum(
                1 for v in result['per_layer'].values()
                if v['p_value'] < 0.05)
            layers_total = len(result['per_layer'])
            print(f'    Per-layer: {layers_sig}/{layers_total} layers '
                  f'significant (p<0.05)')

    # ── Step 6: Save results ─────────────────────────────────────
    output_path = os.path.join(args.output_dir, 'vit_weight_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Results saved to {output_path}')

    # ── Step 7: Print summary table ──────────────────────────────
    print(f'\n  {"═" * 60}')
    print(f'  SUMMARY: VIT Weight Change × PMBT Label')
    print(f'  {"═" * 60}')
    if 'combined' in all_results:
        r = all_results['combined']
        g = r['global']
        print(f'  Delta type: combined (gate + down + up RMS)')
        print()
        for cat in ['visual', 'text', 'multimodal', 'unknown']:
            if cat in g:
                c = g[cat]
                print(f'    {cat:12s}: Δ = {c["mean"]:.6f} ± {c["std"]:.6f}'
                      f'  (n={c["n"]:,})')
        print()
        if 'visual_vs_text' in g:
            vt = g['visual_vs_text']
            sig = ('***' if vt['p_value'] < 0.001 else
                   '**' if vt['p_value'] < 0.01 else
                   '*' if vt['p_value'] < 0.05 else 'ns')
            print(f'  Visual > Text: p = {vt["p_value"]:.2e} {sig}')
            print(f'  Cohen\'s d:     {vt["cohens_d"]:.3f}')
            print(f'  Mean ratio:    {vt["ratio_mean"]:.3f}x')
        if 'point_biserial' in g:
            pb = g['point_biserial']
            print(f'  Point-biserial r = {pb["r"]:.4f}  (p={pb["p_value"]:.2e})')

    elapsed = time.time() - t0
    print(f'\n  Total time: {elapsed / 60:.1f} min')


def _find_state_dict(model_path):
    """Find the actual state dict file given a model directory, file, or HF ID.

    Handles:
      - Direct file paths (pytorch_model.bin, model.safetensors)
      - Local directories with state dict files
      - HuggingFace model IDs (e.g. "Qwen/Qwen2-7B-Instruct")
      - HF cache directories
    """
    # Line 1: if it's a direct file path, return it
    if os.path.isfile(model_path):
        return model_path

    # Line 2: if it's an existing directory, look for state dict files inside
    if os.path.isdir(model_path):
        return _find_state_dict_in_dir(model_path)

    # Line 3: not a local path — treat as HuggingFace model ID
    #          First check if it's already cached locally
    try:
        from huggingface_hub import snapshot_download
        print(f'    Resolving HF model: {model_path}')
        local_dir = snapshot_download(
            model_path,
            local_files_only=False,   # download if not cached
            ignore_patterns=['*.msgpack', '*.h5', '*.ot',
                             'tokenizer*', '*.md', '*.txt'],
        )
        print(f'    Resolved to: {local_dir}')
        return _find_state_dict_in_dir(local_dir)
    except ImportError:
        pass
    except Exception as e:
        print(f'    WARNING: HF download failed: {e}')

    # Line 4: try to find in local HF cache manually
    hf_cache = os.environ.get('HF_HOME',
        os.environ.get('HUGGINGFACE_HUB_CACHE',
            os.path.expanduser('~/.cache/huggingface/hub')))
    # Also check project-local cache
    project_cache = os.path.join(os.getcwd(), '.cache/huggingface/hub')
    for cache_dir in [project_cache, hf_cache]:
        if not os.path.isdir(cache_dir):
            continue
        # HF cache uses models--org--name format
        model_dir_name = 'models--' + model_path.replace('/', '--')
        cached = os.path.join(cache_dir, model_dir_name)
        if os.path.isdir(cached):
            # Find the latest snapshot
            snapshots = os.path.join(cached, 'snapshots')
            if os.path.isdir(snapshots):
                entries = sorted(os.listdir(snapshots))
                if entries:
                    snapshot_dir = os.path.join(snapshots, entries[-1])
                    print(f'    Found in cache: {snapshot_dir}')
                    return _find_state_dict_in_dir(snapshot_dir)

    raise FileNotFoundError(
        f'Could not find model: {model_path}. '
        f'Not a local path and HF resolution failed. '
        f'Try downloading first: '
        f'huggingface-cli download {model_path}')


def _find_state_dict_in_dir(directory):
    """Find state dict file(s) inside a directory."""
    # Line 1: check for common single-file state dicts
    candidates = [
        'pytorch_model.bin',
        'model.safetensors',
    ]
    for c in candidates:
        full = os.path.join(directory, c)
        if os.path.isfile(full):
            return full

    # Line 2: for sharded safetensors, return sentinel for multi-file load
    safetensor_files = sorted([
        f for f in os.listdir(directory) if f.endswith('.safetensors')])
    if safetensor_files:
        return ('safetensors', directory, safetensor_files)

    # Line 3: for sharded pytorch models, return sentinel
    bin_files = sorted([
        f for f in os.listdir(directory)
        if f.startswith('pytorch_model') and f.endswith('.bin')])
    if bin_files:
        return ('sharded_bin', directory, bin_files)

    raise FileNotFoundError(
        f'No state dict found in {directory}. '
        f'Contents: {os.listdir(directory)[:15]}')


# Line 5: override torch.load to handle safetensors and sharded formats
_original_torch_load = torch.load
def _smart_load(path_or_sentinel, **kwargs):
    """Load state dict handling safetensors and sharded formats."""
    if isinstance(path_or_sentinel, tuple):
        fmt, directory, files = path_or_sentinel

        if fmt == 'safetensors':
            try:
                from safetensors.torch import load_file
                state_dict = {}
                for f in files:
                    shard = load_file(os.path.join(directory, f))
                    state_dict.update(shard)
                return state_dict
            except ImportError:
                raise ImportError(
                    'safetensors package required. '
                    'pip install safetensors')

        elif fmt == 'sharded_bin':
            state_dict = {}
            for f in files:
                shard = _original_torch_load(
                    os.path.join(directory, f), **kwargs)
                state_dict.update(shard)
            return state_dict

    return _original_torch_load(path_or_sentinel, **kwargs)

# Monkey-patch torch.load for this script
torch.load = _smart_load


if __name__ == '__main__':
    main()