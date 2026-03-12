#!/usr/bin/env python3
"""Find Figure 3-equivalent neurons for any VLM model.

Selects 6 neurons (1 visual, 1 text, 4 multimodal) that activate on
the SAME COCO images used in the original LLaVA Figure 3, enabling
direct visual comparison across models.

Panel targets (matching LLaVA Figure 3):
  (a) visual      — image 000000403170  (airplanes/motorcycles)
  (b) text        — image 000000065793  (teddy bears)
  (c) multimodal  — image 000000156852  (kitchen)
  (d) multimodal  — image 000000323964  (doughnuts)
  (e) multimodal  — image 000000276332  (zebras)        ← same neuron as (f)
  (f) multimodal  — image 000000060034  (fire hydrant)  ← same neuron as (e)

For panels (e) and (f), the script tries to find a SINGLE neuron that
has BOTH images in its top-N (like the original LLaVA figure).

Usage:
    python find_fig3_neurons.py \
        --data_dir results/3-classify/full/internvl2.5-8b/llm_fixed_threshold \
        --model_type internvl \
        --layer_start 0 --layer_end 31 \
        --top_n 50

    python find_fig3_neurons.py \
        --data_dir results/3-classify/full/qwen2.5-vl-7b/llm_fixed_threshold \
        --model_type qwen2vl \
        --layer_start 0 --layer_end 27 \
        --top_n 50
"""

import argparse
import json
import os
import sys
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Panel specifications — same COCO images as LLaVA Figure 3
# ═══════════════════════════════════════════════════════════════════

PANEL_TARGETS = [
    # (panel, required_type, coco_id, description)
    ('(a)', 'visual',     '000000403170', 'Visual neuron — airplanes/motorcycles'),
    ('(b)', 'text',       '000000065793', 'Text neuron — teddy bears/stuffed animals'),
    ('(c)', 'multimodal', '000000156852', 'Multi-modal neuron — kitchen/thumbs up/tie'),
    ('(d)', 'multimodal', '000000323964', 'Multi-modal neuron — doughnuts'),
]

# Panels (e) and (f) share the SAME neuron, different images
PANEL_EF_IMAGES = [
    ('(e)', '000000276332', 'Multi-modal neuron — zebras (same neuron as f)'),
    ('(f)', '000000060034', 'Multi-modal neuron — fire hydrant/pigeons (same neuron as e)'),
]


# ═══════════════════════════════════════════════════════════════════
# Layer naming per model type
# ═══════════════════════════════════════════════════════════════════

def get_layer_name(model_type, layer):
    """Return the full layer directory name for a given model type and index."""
    if model_type == 'llava-hf':
        return f'model.language_model.model.layers.{layer}.mlp.act_fn'
    elif model_type == 'internvl':
        return f'language_model.model.layers.{layer}.feed_forward.act_fn'
    elif model_type in ('qwen2vl', 'llava-ov'):
        return f'model.language_model.layers.{layer}.mlp.act_fn'
    else:  # llava-liuhaotian
        return f'model.layers.{layer}.mlp.act_fn'


# ═══════════════════════════════════════════════════════════════════
# Core search logic
# ═══════════════════════════════════════════════════════════════════

def load_sampled_ids(topn_heap_dir):
    """Load sample_idx → COCO image ID mapping."""
    path = os.path.join(topn_heap_dir, 'sampled_ids.json')
    with open(path) as f:
        return json.load(f)


def build_coco_to_sample(sampled_ids):
    """Build reverse mapping: coco_id → sample_idx."""
    return {cid: idx for idx, cid in enumerate(sampled_ids)}


def load_topn_heap_layer(topn_heap_dir, layer):
    """Load top-N sample IDs and activations for one layer.

    Returns:
        sids: (n_neurons, top_n) int32  — sample indices
        acts: (n_neurons, top_n) float32 — activation values
    """
    sids = np.load(os.path.join(topn_heap_dir, f'top_n_sids_layer{layer}.npy'))
    acts = np.load(os.path.join(topn_heap_dir, f'top_n_acts_layer{layer}.npy'))
    return sids, acts


def load_neuron_labels(data_dir, layer_name):
    """Load neuron labels from layer directory.

    Tries neuron_labels.json first; returns empty list if not found.
    """
    label_path = os.path.join(data_dir, layer_name, 'neuron_labels.json')
    if not os.path.isfile(label_path):
        return []
    with open(label_path) as f:
        return json.load(f)


def find_neurons_with_image(sids, target_sample_idx, top_n):
    """Find all neurons that have target_sample_idx in their top-N.

    Args:
        sids:              (n_neurons, heap_size) — sample indices per neuron
        target_sample_idx: int — the sample index to search for
        top_n:             int — only search within each neuron's top-N slots

    Returns:
        list[int] — neuron indices that contain the target image
    """
    # Limit search to top_n columns
    search_cols = min(top_n, sids.shape[1])
    sids_topn = sids[:, :search_cols]

    # Boolean mask: (n_neurons,) — True if target appears in this neuron's top-N
    mask = np.any(sids_topn == target_sample_idx, axis=1)
    return np.where(mask)[0].tolist()


def get_confidence(label_entry):
    """Extract the dominant confidence value for a neuron label entry."""
    lbl = label_entry.get('label', 'unknown')
    if lbl == 'visual':
        return label_entry.get('pv', 0.0)
    elif lbl == 'text':
        return label_entry.get('pt', 0.0)
    elif lbl == 'multimodal':
        return label_entry.get('pm', 0.0)
    return 0.0


def find_rank_for_image(sids, neuron_idx, target_sample_idx, top_n):
    """Find at which rank the target image appears for a specific neuron.

    Returns:
        int or None — rank index (0-based), or None if not found
    """
    search_cols = min(top_n, sids.shape[1])
    row = sids[neuron_idx, :search_cols]
    matches = np.where(row == target_sample_idx)[0]
    return int(matches[0]) if len(matches) > 0 else None


# ═══════════════════════════════════════════════════════════════════
# Main search
# ═══════════════════════════════════════════════════════════════════

def search_panels(data_dir, model_type, layer_start, layer_end, top_n):
    """Search all layers for Figure 3 candidate neurons.

    For each panel target, scans every layer to find neurons of the
    correct type that have the target COCO image in their top-N.
    Picks the highest-confidence match.

    Returns:
        list[dict] — 6 entries matching FIG3_NEURONS format
    """
    topn_heap_dir = os.path.join(data_dir, 'topn_heap')

    # Load sampled_ids → coco_id mapping
    print('Loading sampled_ids...')
    sampled_ids = load_sampled_ids(topn_heap_dir)
    coco_to_sample = build_coco_to_sample(sampled_ids)
    print(f'  {len(sampled_ids)} samples in dataset')

    # Check all target images exist in the dataset
    all_target_ids = (
        [t[2] for t in PANEL_TARGETS] +
        [t[1] for t in PANEL_EF_IMAGES]
    )
    for cid in all_target_ids:
        if cid not in coco_to_sample:
            print(f'  WARNING: COCO image {cid} not found in sampled_ids!')

    results = {}  # panel → best match dict

    # ── Search panels (a)-(d): independent neurons ─────────────
    for panel, req_type, coco_id, description in PANEL_TARGETS:
        print(f'\n{"─"*60}')
        print(f'Searching for {panel}: {req_type} neuron with image {coco_id}')

        target_sid = coco_to_sample.get(coco_id)
        if target_sid is None:
            print(f'  SKIP: image {coco_id} not in dataset')
            continue

        best = None  # (confidence, layer, neuron_idx, rank)

        for layer in range(layer_start, layer_end + 1):
            # Load top-N heap
            sids_path = os.path.join(topn_heap_dir, f'top_n_sids_layer{layer}.npy')
            if not os.path.isfile(sids_path):
                continue
            sids, acts = load_topn_heap_layer(topn_heap_dir, layer)

            # Find neurons with this image in their top-N
            candidates = find_neurons_with_image(sids, target_sid, top_n)
            if not candidates:
                continue

            # Load labels to filter by type and get confidence
            layer_name = get_layer_name(model_type, layer)
            labels = load_neuron_labels(data_dir, layer_name)
            if not labels:
                continue

            for nidx in candidates:
                if nidx >= len(labels):
                    continue
                entry = labels[nidx]
                if entry.get('label') != req_type:
                    continue

                conf = get_confidence(entry)
                rank = find_rank_for_image(sids, nidx, target_sid, top_n)

                if best is None or conf > best[0]:
                    best = (conf, layer, nidx, rank)

        if best:
            conf, layer, nidx, rank = best
            print(f'  ✓ FOUND: layer={layer}, neuron={nidx}, '
                  f'conf={conf:.3f}, rank={rank}')
            results[panel] = {
                'panel': panel,
                'layer': layer,
                'neuron_idx': nidx,
                'coco_id': coco_id,
                'label': req_type,
                'description': description,
                '_confidence': conf,
                '_rank': rank,
            }
        else:
            print(f'  ✗ NOT FOUND — no {req_type} neuron with image '
                  f'{coco_id} in top-{top_n}')

    # ── Search panels (e)+(f): SAME neuron, TWO images ─────────
    print(f'\n{"─"*60}')
    print(f'Searching for (e)+(f): SINGLE multimodal neuron with BOTH images')

    ef_sample_idxs = []
    for _, coco_id, _ in PANEL_EF_IMAGES:
        sid = coco_to_sample.get(coco_id)
        if sid is None:
            print(f'  SKIP: image {coco_id} not in dataset')
        ef_sample_idxs.append(sid)

    if all(sid is not None for sid in ef_sample_idxs):
        best_ef = None  # (confidence, layer, neuron_idx, ranks)

        for layer in range(layer_start, layer_end + 1):
            sids_path = os.path.join(topn_heap_dir, f'top_n_sids_layer{layer}.npy')
            if not os.path.isfile(sids_path):
                continue
            sids, acts = load_topn_heap_layer(topn_heap_dir, layer)

            # Find neurons that have BOTH images in top-N
            cands_e = set(find_neurons_with_image(sids, ef_sample_idxs[0], top_n))
            cands_f = set(find_neurons_with_image(sids, ef_sample_idxs[1], top_n))
            both = cands_e & cands_f

            if not both:
                continue

            layer_name = get_layer_name(model_type, layer)
            labels = load_neuron_labels(data_dir, layer_name)
            if not labels:
                continue

            for nidx in both:
                if nidx >= len(labels):
                    continue
                entry = labels[nidx]
                if entry.get('label') != 'multimodal':
                    continue

                conf = get_confidence(entry)
                rank_e = find_rank_for_image(sids, nidx, ef_sample_idxs[0], top_n)
                rank_f = find_rank_for_image(sids, nidx, ef_sample_idxs[1], top_n)

                if best_ef is None or conf > best_ef[0]:
                    best_ef = (conf, layer, nidx, rank_e, rank_f)

        if best_ef:
            conf, layer, nidx, rank_e, rank_f = best_ef
            print(f'  ✓ FOUND shared neuron: layer={layer}, neuron={nidx}, '
                  f'conf={conf:.3f}, rank_e={rank_e}, rank_f={rank_f}')
            for (panel, coco_id, desc), rank in zip(
                    PANEL_EF_IMAGES, [rank_e, rank_f]):
                results[panel] = {
                    'panel': panel,
                    'layer': layer,
                    'neuron_idx': nidx,
                    'coco_id': coco_id,
                    'label': 'multimodal',
                    'description': desc,
                    '_confidence': conf,
                    '_rank': rank,
                }
        else:
            # Fallback: find separate neurons for (e) and (f)
            print(f'  ✗ No single neuron found with BOTH images in top-{top_n}')
            print(f'  Falling back to separate neurons for (e) and (f)...')

            for (panel, coco_id, desc), target_sid in zip(
                    PANEL_EF_IMAGES, ef_sample_idxs):
                best = None
                for layer in range(layer_start, layer_end + 1):
                    sids_path = os.path.join(
                        topn_heap_dir, f'top_n_sids_layer{layer}.npy')
                    if not os.path.isfile(sids_path):
                        continue
                    sids, acts = load_topn_heap_layer(topn_heap_dir, layer)
                    candidates = find_neurons_with_image(sids, target_sid, top_n)
                    if not candidates:
                        continue

                    layer_name = get_layer_name(model_type, layer)
                    labels = load_neuron_labels(data_dir, layer_name)
                    if not labels:
                        continue

                    for nidx in candidates:
                        if nidx >= len(labels):
                            continue
                        entry = labels[nidx]
                        if entry.get('label') != 'multimodal':
                            continue
                        conf = get_confidence(entry)
                        rank = find_rank_for_image(sids, nidx, target_sid, top_n)
                        if best is None or conf > best[0]:
                            best = (conf, layer, nidx, rank)

                if best:
                    conf, layer, nidx, rank = best
                    print(f'  ✓ {panel}: layer={layer}, neuron={nidx}, '
                          f'conf={conf:.3f}, rank={rank}')
                    results[panel] = {
                        'panel': panel,
                        'layer': layer,
                        'neuron_idx': nidx,
                        'coco_id': coco_id,
                        'label': 'multimodal',
                        'description': desc + ' (separate neuron)',
                        '_confidence': conf,
                        '_rank': rank,
                    }
                else:
                    print(f'  ✗ {panel}: NOT FOUND')

    return results


# ═══════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════

def format_python_list(results, model_type, var_name):
    """Format results as a Python list ready to paste into
    visualize_neuron_activations.py.
    """
    lines = [f'{var_name} = [']

    for panel in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        if panel not in results:
            lines.append(f'    # {panel}: NOT FOUND — expand top_n or check '
                         f'classification results')
            continue

        r = results[panel]
        lines.append('    {')
        lines.append(f"        'panel': '{r['panel']}',")
        lines.append(f"        'layer': {r['layer']},")
        lines.append(f"        'neuron_idx': {r['neuron_idx']},")
        lines.append(f"        'coco_id': '{r['coco_id']}',")
        lines.append(f"        'label': '{r['label']}',")
        lines.append(f"        'description': '{r['description']}',")
        lines.append(f"        # confidence={r['_confidence']:.3f}, "
                     f"rank={r['_rank']}")
        lines.append('    },')

    lines.append(']')
    return '\n'.join(lines)


def format_json(results):
    """Format results as JSON for programmatic use."""
    out = []
    for panel in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        if panel in results:
            r = dict(results[panel])
            r.pop('_confidence', None)
            r.pop('_rank', None)
            out.append(r)
    return json.dumps(out, indent=2)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

MODEL_VAR_NAMES = {
    'internvl':         'FIG3_NEURONS_INTERNVL',
    'qwen2vl':          'FIG3_NEURONS_QWEN2VL',
    'llava-ov':         'FIG3_NEURONS_LLAVA_OV',
    'llava-hf':         'FIG3_NEURONS_LLAVA',
    'llava-liuhaotian': 'FIG3_NEURONS_LLAVA',
}

MODEL_LAYERS = {
    'llava-hf':         (0, 31),
    'llava-liuhaotian': (0, 31),
    'internvl':         (0, 31),
    'qwen2vl':          (0, 27),
    'llava-ov':         (0, 27),
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Find Figure 3-equivalent neurons for any VLM model.')

    p.add_argument('--data_dir', required=True,
                   help='FT classification directory '
                        '(e.g. results/3-classify/full/internvl2.5-8b/'
                        'llm_fixed_threshold)')
    p.add_argument('--model_type', required=True,
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl',
                            'qwen2vl', 'llava-ov'],
                   help='Model type (determines layer directory naming)')
    p.add_argument('--layer_start', type=int, default=None,
                   help='First layer to search (default: auto per model)')
    p.add_argument('--layer_end', type=int, default=None,
                   help='Last layer to search inclusive (default: auto per model)')
    p.add_argument('--top_n', type=int, default=50,
                   help='Search within each neuron\'s top-N activating images '
                        '(default: 50)')
    p.add_argument('--output_json', default=None,
                   help='Save results as JSON file. Default: '
                        '<data_dir>/../fig3_neurons.json')
    p.add_argument('--output_py', default=None,
                   help='Save results as Python snippet file (optional)')

    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect layer range from model type
    default_start, default_end = MODEL_LAYERS.get(
        args.model_type, (0, 31))
    layer_start = args.layer_start if args.layer_start is not None else default_start
    layer_end = args.layer_end if args.layer_end is not None else default_end

    # Auto-set output JSON path: <data_dir>/../fig3_neurons.json
    if args.output_json is None:
        args.output_json = os.path.join(
            os.path.dirname(args.data_dir.rstrip('/')),
            'fig3_neurons.json')

    print(f'{"═"*60}')
    print(f'  Find Figure 3 Neurons')
    print(f'{"═"*60}')
    print(f'  Model type:  {args.model_type}')
    print(f'  Data dir:    {args.data_dir}')
    print(f'  Layer range: {layer_start}–{layer_end}')
    print(f'  Top-N:       {args.top_n}')
    print(f'{"═"*60}')

    # Run search
    results = search_panels(
        args.data_dir, args.model_type,
        layer_start, layer_end, args.top_n)

    # ── Summary ───────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'  RESULTS: {len(results)}/6 panels found')
    print(f'{"═"*60}')

    for panel in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        if panel in results:
            r = results[panel]
            print(f'  {panel} {r["label"]:11s}  layer={r["layer"]:2d}  '
                  f'neuron={r["neuron_idx"]:5d}  '
                  f'conf={r["_confidence"]:.3f}  '
                  f'rank={r["_rank"]:2d}  '
                  f'{r["coco_id"]}')
        else:
            print(f'  {panel} ✗ NOT FOUND')

    # ── Output Python list ────────────────────────────────────
    var_name = MODEL_VAR_NAMES.get(args.model_type, 'FIG3_NEURONS')
    py_snippet = format_python_list(results, args.model_type, var_name)

    print(f'\n{"─"*60}')
    print(f'Paste this into visualize_neuron_activations.py:')
    print(f'{"─"*60}\n')
    print(py_snippet)

    # ── Save outputs ──────────────────────────────────────────
    with open(args.output_json, 'w') as f:
        f.write(format_json(results) + '\n')
    print(f'\n  Saved JSON → {args.output_json}')

    if args.output_py:
        with open(args.output_py, 'w') as f:
            f.write(py_snippet + '\n')
        print(f'  Saved Python snippet → {args.output_py}')

    if len(results) < 6:
        print(f'\n  TIP: {6 - len(results)} panels missing. Try:')
        print(f'    --top_n 100  (expand search to top-100)')
        print(f'    Or check that classification labels exist for all layers.')

    return 0 if len(results) == 6 else 1


if __name__ == '__main__':
    sys.exit(main())
