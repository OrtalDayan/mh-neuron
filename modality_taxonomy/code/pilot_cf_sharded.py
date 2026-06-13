#!/usr/bin/env python3
"""
Counterfactual Perturbation Classifier Pilot — SHARDED MULTI-GPU VERSION.

Parallelizes the CF pilot across N GPUs by sharding samples.
With 4 A100s: ~22 minutes wall time.
With 8 A100s: ~11 minutes wall time.
(vs ~90 minutes for single-GPU sequential.)

Two modes:
  shard:  Process this shard's samples, save activations to .npz files.
  merge:  Load all shards, concatenate, run classifier, compare to PMBT.

Usage:
  Phase 1 (run N shards in parallel via bsub):
    for s in 0 1 2 3; do
      bsub -gpu "num=1" -- "python pilot_cf_sharded.py shard \\
          --shard_idx $s --n_shards 4 [other args]"
    done

  Phase 2 (after all shards finish, merge):
    bsub -w "done(shard_0) && ..." -- "python pilot_cf_sharded.py merge \\
        --n_shards 4 [other args]"

  Or use dispatch_cf_sharded.sh which handles the orchestration.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Argument parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='CF perturbation classifier pilot — sharded multi-GPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument('mode', choices=['shard', 'merge', 'all_local'],
                   help='shard = run one shard; merge = combine + classify; '
                        'all_local = single-GPU sequential (no sharding)')

    # Sharding
    p.add_argument('--shard_idx', type=int, default=0)
    p.add_argument('--n_shards', type=int, default=1)
    p.add_argument('--shard_dir', default='results/cf_pilot_shards',
                   help='Per-shard intermediate file directory')

    # Model
    p.add_argument('--model_type', default='llava-llama3',
                   choices=['llava-llama3'])
    p.add_argument('--model_path',
                   default='llava-hf/llama3-llava-next-8b-hf')

    # Paired data
    p.add_argument('--paired_data_path', default='data/cf_paired_500.json')
    p.add_argument('--coco_captions_path', default='data/coco_captions.json')
    p.add_argument('--coco_img_dir', default='data/coco/train2017')
    p.add_argument('--n_pilot_samples', type=int, default=500)
    p.add_argument('--K_text', type=int, default=5)
    p.add_argument('--K_image', type=int, default=5)

    # Layers
    p.add_argument('--layers', default='14,18,22')
    p.add_argument('--hook_point', default='gate_up',
                   choices=['gate', 'gate_up', 'down'])

    # CF classifier
    p.add_argument('--n_permutations', type=int, default=1000)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n_noise_pairs', type=int, default=500)
    p.add_argument('--noise_percentile', type=float, default=95.0)
    p.add_argument('--noise_K', type=int, default=5)

    # Comparison + output
    p.add_argument('--pmbt_label_dir',
                   default='results/3-classify/full/llava-next-llama3-8b/'
                           'llm_permutation_gate_up_min100_max2048')
    p.add_argument('--output_dir',
                   default='results/3-classify/full/llava-next-llama3-8b/'
                           'llm_cf_permutation_gate_up_pilot')

    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip_paired_build', action='store_true')
    p.add_argument('--skip_noise_pass', action='store_true')
    p.add_argument('--build_paired_only', action='store_true',
                   help='Build paired data and exit (does not need GPU)')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Sharding utilities
# ═══════════════════════════════════════════════════════════════════

def shard_samples(sample_ids, shard_idx, n_shards):
    """Round-robin sharding: sample i goes to shard (i % n_shards)."""
    return [sid for i, sid in enumerate(sample_ids)
            if i % n_shards == shard_idx]


def shard_path(shard_dir, kind, shard_idx, layer_idx):
    """Path to shard intermediate file. kind ∈ {image_var, text_var, noise_var}."""
    return os.path.join(
        shard_dir, f'{kind}_layer{layer_idx}_shard{shard_idx}.npz'
    )


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Build paired data, model loading, etc.
# ═══════════════════════════════════════════════════════════════════

def build_paired_data(coco_captions_path, n_samples, K_image, K_text,
                       out_path, seed=42):
    """Build paired-data JSON via caption-embedding kNN."""
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors

    print(f'Loading COCO captions from {coco_captions_path}...')
    with open(coco_captions_path) as f:
        coco = json.load(f)

    valid_ids = [iid for iid, caps in coco.items() if len(caps) >= K_text]
    print(f'Found {len(valid_ids)} images with >= {K_text} captions')

    if len(valid_ids) < n_samples + K_image:
        raise ValueError(
            f'Not enough images: need {n_samples + K_image}, have {len(valid_ids)}'
        )

    print(f'Embedding {len(valid_ids)} canonical captions...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    canonical_caps = [coco[iid][0] for iid in valid_ids]
    embeddings = model.encode(canonical_caps, show_progress_bar=True,
                               convert_to_numpy=True)

    print(f'Building kNN index (K = {K_image + 1})...')
    nn = NearestNeighbors(n_neighbors=K_image + 1, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    rng = np.random.RandomState(seed)
    sampled_idxs = rng.choice(len(valid_ids), size=n_samples, replace=False)

    paired = {}
    for i in sampled_idxs:
        img_id = valid_ids[i]
        similar_idxs = indices[i, 1:K_image + 1]
        paired[img_id] = {
            'canonical_caption': coco[img_id][0],
            'captions': coco[img_id][:K_text],
            'similar_images': [valid_ids[j] for j in similar_idxs],
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(paired, f)
    print(f'Saved paired data to {out_path}: {len(paired)} samples')
    return paired


def load_model(model_type, model_path, device):
    from transformers import LlavaNextForConditionalGeneration, AutoProcessor

    if model_type != 'llava-llama3':
        raise NotImplementedError(f'Pilot supports llava-llama3 only')

    print(f'Loading {model_type}: {model_path}...')
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).eval().to(device)

    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    if image_token_id is None or image_token_id < 0:
        image_token_id = 128256

    print(f'  Loaded. Image token ID: {image_token_id}')
    return model, processor, image_token_id


def get_layer_names(n_layers, hook_point='gate_up'):
    if hook_point == 'gate':
        suffix, retain_input = 'mlp.act_fn', False
    elif hook_point == 'gate_up':
        suffix, retain_input = 'mlp.down_proj', True
    elif hook_point == 'down':
        suffix, retain_input = 'mlp.down_proj', False
    else:
        raise ValueError(f'Unknown hook_point: {hook_point}')

    prefix = 'model.language_model.layers'
    names = [f'{prefix}.{i}.{suffix}' for i in range(n_layers)]
    return names, retain_input


def prepare_input(processor, img, caption_text, device):
    """Build LLaMA3-LLaVA-Next teacher-forced input."""
    question = 'Could you describe the image?'
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': question},
        ]},
        {'role': 'assistant', 'content': [
            {'type': 'text', 'text': caption_text},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(images=img, text=text, return_tensors='pt').to(device)
    return inputs


def record_activations(model, processor, img, caption_text,
                        layer_names, retain_input, layer_indices,
                        device, image_token_id):
    """Run one forward pass, return max activation per neuron."""
    from baukit import TraceDict

    inputs = prepare_input(processor, img, caption_text, device)
    input_ids = inputs['input_ids'][0].cpu().numpy()
    visual_mask_full = (input_ids == image_token_id)

    last_vis = np.where(visual_mask_full)[0]
    if len(last_vis) == 0:
        text_mask_full = np.zeros_like(visual_mask_full)
        text_mask_full[-50:] = True
    else:
        text_start = last_vis[-1] + 5
        text_mask_full = np.zeros_like(visual_mask_full)
        text_mask_full[text_start:] = True

    content_mask = visual_mask_full | text_mask_full

    layer_acts = {}
    with torch.no_grad():
        with TraceDict(model, layer_names, retain_input=retain_input) as td:
            model(**inputs)

    for li, layer_name in enumerate(layer_names):
        if li not in layer_indices:
            continue
        if retain_input:
            inp = td[layer_name].input
            out = inp[0] if isinstance(inp, tuple) else inp
        else:
            out = td[layer_name].output
            if isinstance(out, tuple):
                out = out[0]
        acts = out[0].float()
        acts_content = acts[content_mask]
        if acts_content.shape[0] == 0:
            n_neurons = acts.shape[-1]
            layer_acts[li] = np.zeros(n_neurons, dtype=np.float32)
        else:
            max_per_neuron = acts_content.max(dim=0).values.cpu().numpy()
            layer_acts[li] = max_per_neuron.astype(np.float32)

    return layer_acts


def build_random_pairs(paired, all_img_ids, n_pairs, K_per_group, seed):
    """Build random (image, caption) pairs for the noise pass."""
    rng = np.random.RandomState(seed)
    all_captions = []
    for iid, data in paired.items():
        all_captions.extend([(iid, cap) for cap in data['captions']])

    available_imgs = list(set(all_img_ids))
    noise_groups = []
    for g in range(n_pairs):
        group = []
        for _ in range(K_per_group):
            rand_img = available_imgs[rng.randint(len(available_imgs))]
            _, rand_cap = all_captions[rng.randint(len(all_captions))]
            group.append((rand_img, rand_cap))
        noise_groups.append(group)
    return noise_groups


def _resolve_img_path(img_dir, img_id):
    candidates = [
        os.path.join(img_dir, f'{int(img_id):012d}.jpg'),
        os.path.join(img_dir, f'COCO_train2017_{int(img_id):012d}.jpg'),
        os.path.join(img_dir, f'COCO_val2017_{int(img_id):012d}.jpg'),
        os.path.join(img_dir, f'{img_id}.jpg'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Classifier (Option 2: noise-floor calibration)
# ═══════════════════════════════════════════════════════════════════

def classify_neurons_cf_option2_gpu_batched(
    image_var_acts, text_var_acts, noise_var_acts,
    n_permutations=1000, alpha=0.05, noise_percentile=95.0,
    device='cuda:0', seed=42,
):
    """Vectorized GPU classifier with noise-floor threshold (Option 2)."""
    image_var = torch.tensor(image_var_acts, device=device, dtype=torch.float32)
    text_var = torch.tensor(text_var_acts, device=device, dtype=torch.float32)
    noise_var = torch.tensor(noise_var_acts, device=device, dtype=torch.float32)

    n_neurons = image_var.shape[0]

    var_image = image_var.var(dim=2).mean(dim=1)
    var_text = text_var.var(dim=2).mean(dim=1)
    var_noise = noise_var.var(dim=2).mean(dim=1)
    observed_D = var_image - var_text
    max_signal = torch.maximum(var_image, var_text)

    var_noise_np = var_noise.cpu().numpy()
    engaged_threshold = float(np.percentile(var_noise_np, noise_percentile))
    print(f'  Noise percentile {noise_percentile}: threshold = {engaged_threshold:.6f}')
    print(f'  Noise variance: median={np.median(var_noise_np):.6f}, '
          f'p95={np.percentile(var_noise_np, 95):.6f}, '
          f'max={var_noise_np.max():.6f}')

    n_image_total = image_var.shape[1] * image_var.shape[2]
    n_text_total = text_var.shape[1] * text_var.shape[2]
    pooled = torch.cat(
        [image_var.reshape(n_neurons, -1),
         text_var.reshape(n_neurons, -1)],
        dim=1
    )

    rng = torch.Generator(device=device).manual_seed(seed)
    null_Ds = torch.zeros(n_neurons, n_permutations, device=device)
    for p in range(n_permutations):
        perm = torch.randperm(n_image_total + n_text_total,
                                generator=rng, device=device)
        shuf = pooled[:, perm]
        i_shuf = shuf[:, :n_image_total].reshape(image_var.shape)
        t_shuf = shuf[:, n_image_total:].reshape(text_var.shape)
        null_Ds[:, p] = i_shuf.var(dim=2).mean(dim=1) - \
                       t_shuf.var(dim=2).mean(dim=1)

    p_values = (null_Ds.abs() >= observed_D.abs().unsqueeze(1)).float().mean(dim=1)

    var_image_np = var_image.cpu().numpy()
    var_text_np = var_text.cpu().numpy()
    observed_D_np = observed_D.cpu().numpy()
    p_values_np = p_values.cpu().numpy()
    max_signal_np = max_signal.cpu().numpy()

    results = []
    for n in range(n_neurons):
        max_sig = max_signal_np[n]
        p = p_values_np[n]
        D = observed_D_np[n]

        if max_sig <= engaged_threshold:
            label = 'unknown'
        elif p < alpha:
            label = 'visual' if D > 0 else 'text'
        else:
            label = 'multimodal'

        results.append({
            'neuron_idx': n,
            'label': label,
            'p_value': float(p),
            'observed_D': float(D),
            'var_image': float(var_image_np[n]),
            'var_text': float(var_text_np[n]),
            'var_noise': float(var_noise_np[n]),
            'max_signal': float(max_sig),
        })

    return results, engaged_threshold


# ═══════════════════════════════════════════════════════════════════
# Section 5 — PMBT comparison
# ═══════════════════════════════════════════════════════════════════

def load_pmbt_labels(pmbt_label_dir, layer_idx, layer_name):
    candidates = [
        os.path.join(pmbt_label_dir, layer_name, 'neuron_labels_permutation.json'),
        os.path.join(pmbt_label_dir, f'layer_{layer_idx}',
                      'neuron_labels_permutation.json'),
        os.path.join(pmbt_label_dir, f'layer{layer_idx}',
                      'neuron_labels_permutation.json'),
        os.path.join(pmbt_label_dir,
                      f'neuron_labels_permutation_layer{layer_idx}.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def compare_labels(cf_results, pmbt_labels):
    cf_map = {r['neuron_idx']: r['label'] for r in cf_results}
    pmbt_map = {r['neuron_idx']: r['label'] for r in pmbt_labels}
    common = sorted(set(cf_map.keys()) & set(pmbt_map.keys()))
    if len(common) == 0:
        return None

    cats = ['visual', 'text', 'multimodal', 'unknown']
    confusion = {p: {c: 0 for c in cats} for p in cats}
    agreements = 0
    for n in common:
        pl, cl = pmbt_map[n], cf_map[n]
        if pl in cats and cl in cats:
            confusion[pl][cl] += 1
        if pl == cl:
            agreements += 1

    return {
        'n_compared': len(common),
        'agreement_rate': agreements / len(common),
        'confusion': confusion,
        'categories': cats,
    }


def print_comparison(comparison, layer_idx):
    if comparison is None:
        print(f'  Layer {layer_idx}: No common neurons')
        return
    print(f'\n  ── Layer {layer_idx} ──')
    print(f'  Compared {comparison["n_compared"]} neurons')
    print(f'  Agreement rate: {comparison["agreement_rate"]:.1%}')
    print(f'\n  Confusion matrix (rows = PMBT, cols = CF):')
    cats = comparison['categories']
    header = "PMBT \\ CF"
    print(f'  {header:<14}', end='')
    for c in cats:
        print(f'{c:>14}', end='')
    print(f'{"row total":>11}')
    for p in cats:
        rt = sum(comparison['confusion'][p].values())
        print(f'  {p:<14}', end='')
        for c in cats:
            print(f'{comparison["confusion"][p][c]:>14}', end='')
        print(f'{rt:>11}')


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Shard mode: process this shard's samples
# ═══════════════════════════════════════════════════════════════════

def run_shard(args):
    print(f'═' * 70)
    print(f'CF Pilot — Shard {args.shard_idx} / {args.n_shards}')
    print(f'═' * 70)

    layer_indices = [int(x) for x in args.layers.split(',')]
    device = args.device

    if not os.path.exists(args.paired_data_path):
        sys.exit(f'Paired data not found: {args.paired_data_path}')

    with open(args.paired_data_path) as f:
        paired = json.load(f)

    all_sample_ids = list(paired.keys())[:args.n_pilot_samples]
    my_samples = shard_samples(all_sample_ids, args.shard_idx, args.n_shards)
    print(f'Shard {args.shard_idx}: {len(my_samples)} samples '
          f'(of {len(all_sample_ids)} total)')

    model, processor, image_token_id = load_model(
        args.model_type, args.model_path, device)

    n_layers = model.config.text_config.num_hidden_layers \
        if hasattr(model.config, 'text_config') \
        else model.config.num_hidden_layers
    layer_names, retain_input = get_layer_names(n_layers, args.hook_point)

    # Probe for n_neurons
    probe_id = my_samples[0]
    probe_path = _resolve_img_path(args.coco_img_dir, probe_id)
    if probe_path is None:
        sys.exit(f'Could not find probe image: {probe_id}')
    probe_img = Image.open(probe_path).convert('RGB')
    probe_caption = paired[probe_id]['canonical_caption']
    probe_acts = record_activations(
        model, processor, probe_img, probe_caption,
        layer_names, retain_input, layer_indices, device, image_token_id)
    n_neurons = probe_acts[layer_indices[0]].shape[0]
    print(f'  n_neurons per layer: {n_neurons}')

    K_v, K_t, K_n = args.K_image, args.K_text, args.noise_K
    n_my = len(my_samples)

    image_var_acts = {li: np.zeros((n_my, K_v, n_neurons), dtype=np.float32)
                       for li in layer_indices}
    text_var_acts = {li: np.zeros((n_my, K_t, n_neurons), dtype=np.float32)
                      for li in layer_indices}

    # CF Pass
    print(f'\nShard {args.shard_idx}: CF Pass...')
    t0 = time.time()
    skipped = 0
    for s_idx, img_id in enumerate(tqdm(my_samples, desc=f'CF[{args.shard_idx}]')):
        sd = paired[img_id]
        canonical_caption = sd['canonical_caption']
        captions = sd['captions'][:K_t]
        similar_imgs = sd['similar_images'][:K_v]

        cp = _resolve_img_path(args.coco_img_dir, img_id)
        if cp is None:
            skipped += 1
            continue
        canonical_img = Image.open(cp).convert('RGB')

        for k, cap in enumerate(captions):
            try:
                acts = record_activations(
                    model, processor, canonical_img, cap,
                    layer_names, retain_input, layer_indices,
                    device, image_token_id)
                for li in layer_indices:
                    text_var_acts[li][s_idx, k] = acts[li]
            except Exception as e:
                print(f'  [skip text {k} of {img_id}]: {e}')

        for k, sid in enumerate(similar_imgs):
            sp = _resolve_img_path(args.coco_img_dir, sid)
            if sp is None:
                continue
            try:
                sim_img = Image.open(sp).convert('RGB')
                acts = record_activations(
                    model, processor, sim_img, canonical_caption,
                    layer_names, retain_input, layer_indices,
                    device, image_token_id)
                for li in layer_indices:
                    image_var_acts[li][s_idx, k] = acts[li]
            except Exception as e:
                print(f'  [skip image {k} of {img_id}]: {e}')

    print(f'CF Pass done in {(time.time()-t0)/60:.1f} min ({skipped} skipped)')

    # Noise Pass (sharded)
    if not args.skip_noise_pass:
        my_n_noise = args.n_noise_pairs // args.n_shards
        if args.shard_idx < args.n_noise_pairs % args.n_shards:
            my_n_noise += 1

        noise_var_acts = {li: np.zeros((my_n_noise, K_n, n_neurons), dtype=np.float32)
                           for li in layer_indices}

        print(f'\nShard {args.shard_idx}: Noise Pass ({my_n_noise} groups)...')
        all_img_ids = list(paired.keys())
        noise_groups = build_random_pairs(
            paired, all_img_ids, my_n_noise, K_n,
            seed=args.seed + 1000 + args.shard_idx)

        t0 = time.time()
        for g_idx, group in enumerate(tqdm(noise_groups,
                                              desc=f'Noise[{args.shard_idx}]')):
            for k, (img_id, cap) in enumerate(group):
                ip = _resolve_img_path(args.coco_img_dir, img_id)
                if ip is None:
                    continue
                try:
                    img = Image.open(ip).convert('RGB')
                    acts = record_activations(
                        model, processor, img, cap,
                        layer_names, retain_input, layer_indices,
                        device, image_token_id)
                    for li in layer_indices:
                        noise_var_acts[li][g_idx, k] = acts[li]
                except Exception as e:
                    print(f'  [skip noise {k} of {g_idx}]: {e}')
        print(f'Noise Pass done in {(time.time()-t0)/60:.1f} min')

    # Save shard files
    os.makedirs(args.shard_dir, exist_ok=True)
    for li in layer_indices:
        np.savez_compressed(
            shard_path(args.shard_dir, 'image_var', args.shard_idx, li),
            data=image_var_acts[li])
        np.savez_compressed(
            shard_path(args.shard_dir, 'text_var', args.shard_idx, li),
            data=text_var_acts[li])
        if not args.skip_noise_pass:
            np.savez_compressed(
                shard_path(args.shard_dir, 'noise_var', args.shard_idx, li),
                data=noise_var_acts[li])
    print(f'\nShard {args.shard_idx} saved to {args.shard_dir}')


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Merge mode: combine shards and run classifier
# ═══════════════════════════════════════════════════════════════════

def run_merge(args):
    print(f'═' * 70)
    print(f'CF Pilot — Merge ({args.n_shards} shards)')
    print(f'═' * 70)

    layer_indices = [int(x) for x in args.layers.split(',')]
    device = args.device

    # Verify all shard files exist
    missing = []
    for li in layer_indices:
        for s in range(args.n_shards):
            for kind in ['image_var', 'text_var']:
                f = shard_path(args.shard_dir, kind, s, li)
                if not os.path.exists(f):
                    missing.append(f)
            if not args.skip_noise_pass:
                f = shard_path(args.shard_dir, 'noise_var', s, li)
                if not os.path.exists(f):
                    missing.append(f)
    if missing:
        print(f'Missing {len(missing)} shard files. First few:')
        for f in missing[:5]:
            print(f'  {f}')
        sys.exit(1)

    # Layer names (no model load needed — config only)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path)
    n_layers = config.text_config.num_hidden_layers \
        if hasattr(config, 'text_config') \
        else config.num_hidden_layers
    layer_names, _ = get_layer_names(n_layers, args.hook_point)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    for li in layer_indices:
        layer_name = layer_names[li]
        print(f'\n  Layer {li} ({layer_name})')

        # Concatenate shards
        image_var_shards = [np.load(shard_path(args.shard_dir, 'image_var', s, li))['data']
                             for s in range(args.n_shards)]
        text_var_shards = [np.load(shard_path(args.shard_dir, 'text_var', s, li))['data']
                            for s in range(args.n_shards)]
        image_var = np.concatenate(image_var_shards, axis=0)
        text_var = np.concatenate(text_var_shards, axis=0)

        # (n_neurons, n_samples, K)
        img_acts = image_var.transpose(2, 0, 1)
        txt_acts = text_var.transpose(2, 0, 1)

        if not args.skip_noise_pass:
            noise_shards = [np.load(shard_path(args.shard_dir, 'noise_var', s, li))['data']
                             for s in range(args.n_shards)]
            noise_var = np.concatenate(noise_shards, axis=0)
            noi_acts = noise_var.transpose(2, 0, 1)

            print(f'  image_var {image_var.shape}, text_var {text_var.shape}, '
                  f'noise_var {noise_var.shape}')

            results, threshold = classify_neurons_cf_option2_gpu_batched(
                img_acts, txt_acts, noi_acts,
                n_permutations=args.n_permutations,
                alpha=args.alpha,
                noise_percentile=args.noise_percentile,
                device=device,
                seed=args.seed + li,
            )
        else:
            print(f'  image_var {image_var.shape}, text_var {text_var.shape}')
            sys.exit('Option 1 (skip_noise_pass) not implemented in merge mode. '
                      'Run with full noise pass.')

        # Save labels
        layer_dir = os.path.join(args.output_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        out_path = os.path.join(layer_dir, 'neuron_labels_cf_permutation.json')
        with open(out_path, 'w') as f:
            json.dump({
                'method': 'cf_option2',
                'engaged_threshold': threshold,
                'noise_percentile': args.noise_percentile,
                'alpha': args.alpha,
                'n_permutations': args.n_permutations,
                'n_samples': int(image_var.shape[0]),
                'n_shards': args.n_shards,
                'results': results,
            }, f, indent=2)
        print(f'  Saved to {out_path}')

        # Distribution
        label_counts = defaultdict(int)
        for r in results:
            label_counts[r['label']] += 1
        total = sum(label_counts.values())
        print(f'  Label distribution:')
        for lbl in ['visual', 'text', 'multimodal', 'unknown']:
            n = label_counts[lbl]
            print(f'    {lbl:<12} {n:>6} ({100*n/total:>5.1f}%)')

        # Compare to PMBT
        pmbt = load_pmbt_labels(args.pmbt_label_dir, li, layer_name)
        if pmbt is not None:
            comparison = compare_labels(results, pmbt)
            if comparison is not None:
                print_comparison(comparison, li)
                summary[li] = {
                    'cf_labels': dict(label_counts),
                    'engaged_threshold': threshold,
                    'comparison': comparison,
                }
        else:
            print(f'  No PMBT labels found at {args.pmbt_label_dir}')
            summary[li] = {'cf_labels': dict(label_counts),
                            'engaged_threshold': threshold}

    # Summary
    summary_path = os.path.join(args.output_dir, 'pilot_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to {summary_path}')

    print('\n' + '═' * 70)
    print('Pilot complete')
    print('═' * 70)


# ═══════════════════════════════════════════════════════════════════
# Section 8 — Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Build-paired-only mode: build paired data and exit (no GPU needed)
    if args.build_paired_only:
        if os.path.exists(args.paired_data_path):
            print(f'Paired data already exists at {args.paired_data_path}; '
                  f'skipping build')
            return
        print(f'Building paired data ({args.n_pilot_samples} samples)...')
        build_paired_data(
            args.coco_captions_path,
            args.n_pilot_samples,
            args.K_image,
            args.K_text,
            args.paired_data_path,
            seed=args.seed,
        )
        return

    if args.mode == 'shard':
        run_shard(args)
    elif args.mode == 'merge':
        run_merge(args)
    elif args.mode == 'all_local':
        # Single-GPU sequential — for testing or no-cluster scenarios
        # Run one "shard" with shard_idx=0, n_shards=1, then merge
        run_shard(args)
        run_merge(args)


if __name__ == '__main__':
    main()