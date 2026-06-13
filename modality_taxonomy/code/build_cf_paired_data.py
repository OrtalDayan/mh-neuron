#!/usr/bin/env python3
"""
build_cf_paired_data.py
=======================

Build paired data for the counterfactual perturbation classifier.

Uses the same COCO conventions as run_pipeline.sh:
  - COCO captions: /home/projects/bagon/shared/coco2017/annotations/captions_train2017.json
  - COCO images:   /home/projects/bagon/shared/coco2017/images/train2017/
  - detail_23k.json subset (matches the existing PMBT pipeline's image scope)

Output structure:
  {
      "<img_id>": {
          "canonical_caption": "...",
          "captions": ["cap1", "cap2", "cap3", "cap4", "cap5"],
          "similar_images": ["<id1>", "<id2>", "<id3>", "<id4>", "<id5>"]
      },
      ...
  }

Method for image variants:
  Caption-embedding kNN with sentence-transformers (all-MiniLM-L6-v2).
  For each canonical image, finds K_image nearest neighbors by cosine similarity
  over the canonical captions' embeddings.

  This is sufficient for an initial pilot. For the NeurIPS-quality version,
  upgrade to Crisscrossed Captions (CxC) image-image similarity ratings or
  CLIP image-embedding kNN — see the PDF at cf_neurips_paper_plan.pdf.

Usage:
  # Default: 500 samples, K_image=K_text=5
  python build_cf_paired_data.py \\
      --coco_ann_path /home/projects/bagon/shared/coco2017/annotations/captions_train2017.json \\
      --detail_23k_path data/detail_23k.json \\
      --output data/cf_paired_500.json

  # Smoke test: 10 samples, K=3
  python build_cf_paired_data.py \\
      --coco_ann_path /home/projects/bagon/shared/coco2017/annotations/captions_train2017.json \\
      --detail_23k_path data/detail_23k.json \\
      --output data/cf_paired_smoketest.json \\
      --n_samples 10 --K_image 3 --K_text 3
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Build paired data for CF classifier')

    p.add_argument('--coco_ann_path',
                   default='/home/projects/bagon/shared/coco2017/'
                           'annotations/captions_train2017.json',
                   help='COCO annotations JSON (captions_train2017.json)')
    p.add_argument('--detail_23k_path',
                   default='data/detail_23k.json',
                   help='Path to detail_23k.json (defines image subset; '
                        'matches PMBT pipeline scope)')
    p.add_argument('--output', required=True,
                   help='Output paired-data JSON')

    p.add_argument('--n_samples', type=int, default=500,
                   help='Number of canonical samples to build')
    p.add_argument('--K_image', type=int, default=5,
                   help='Number of similar-image neighbors per sample')
    p.add_argument('--K_text', type=int, default=5,
                   help='Number of text variants (uses COCO native captions)')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto',
                   help='Device for sentence-transformers embedding: '
                        '"auto" (use GPU if available, else CPU), '
                        '"cuda:0" (force GPU), "cpu" (force CPU). '
                        'GPU embedding is ~10-100x faster on 23K captions.')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for embedding (larger = faster on GPU)')
    p.add_argument('--use_detail_23k', action='store_true', default=True,
                   help='Restrict samples to detail_23k subset (default: True)')
    p.add_argument('--no_detail_23k', dest='use_detail_23k', action='store_false',
                   help='Use full COCO instead of detail_23k subset')

    return p.parse_args()


def load_coco_captions_grouped(ann_path):
    """Load COCO captions, grouped {img_id_str: [caps...]}.

    Standard COCO annotation format:
        {"annotations": [{"image_id": 123, "caption": "..."}, ...]}
    """
    print(f'Loading COCO captions from {ann_path} …')
    with open(ann_path) as f:
        coco = json.load(f)

    if 'annotations' not in coco:
        sys.exit(f'Expected "annotations" key in {ann_path}; '
                  f'got: {list(coco.keys())[:5]}')

    captions = defaultdict(list)
    for ann in coco['annotations']:
        # COCO image_id is int; pad to 12-digit string to match detail_23k
        img_id_str = f"{ann['image_id']:012d}"
        captions[img_id_str].append(ann['caption'])

    print(f'  {len(coco["annotations"])} captions across {len(captions)} images')
    cap_counts = [len(c) for c in captions.values()]
    print(f'  Captions per image: min={min(cap_counts)}, '
          f'max={max(cap_counts)}, mean={sum(cap_counts)/len(cap_counts):.2f}')

    return dict(captions)


def load_detail_23k_ids(detail_23k_path):
    """Load detail_23k.json image IDs."""
    print(f'Loading detail_23k from {detail_23k_path} …')
    if not os.path.exists(detail_23k_path):
        sys.exit(f'detail_23k.json not found at {detail_23k_path}.\n'
                  f'This file defines the image subset used by the PMBT pipeline.\n'
                  f'It should be at modality_taxonomy/data/detail_23k.json '
                  f'(typically downloaded with the original Xu et al. release).')

    with open(detail_23k_path) as f:
        data = json.load(f)

    # detail_23k entries are {"id": "000000xxxxxx", "image": "000000xxxxxx.jpg", ...}
    ids = [item['id'] for item in data]
    print(f'  Loaded {len(ids)} image IDs from detail_23k')
    return ids


def main():
    args = parse_args()

    print('═' * 70)
    print('  CF Paired Data Builder')
    print('═' * 70)
    print(f'  COCO ann path:     {args.coco_ann_path}')
    print(f'  detail_23k path:   {args.detail_23k_path}')
    print(f'  Output:            {args.output}')
    print(f'  N samples:         {args.n_samples}')
    print(f'  K_image (similar): {args.K_image}')
    print(f'  K_text (captions): {args.K_text}')
    print(f'  Use detail_23k:    {args.use_detail_23k}')
    print()

    # Load COCO captions
    captions = load_coco_captions_grouped(args.coco_ann_path)

    # Restrict to detail_23k if requested
    if args.use_detail_23k:
        detail_ids = set(load_detail_23k_ids(args.detail_23k_path))
        valid_ids = [iid for iid in captions if iid in detail_ids
                      and len(captions[iid]) >= args.K_text]
        print(f'  Filtered to {len(valid_ids)} images in detail_23k '
              f'with >= {args.K_text} captions')
    else:
        valid_ids = [iid for iid, caps in captions.items()
                      if len(caps) >= args.K_text]
        print(f'  {len(valid_ids)} images with >= {args.K_text} captions')

    if len(valid_ids) < args.n_samples + args.K_image:
        sys.exit(f'Not enough valid images: need {args.n_samples + args.K_image}, '
                  f'have {len(valid_ids)}')

    # Embed canonical captions
    print(f'\nEmbedding {len(valid_ids)} canonical captions '
          f'(sentence-transformers all-MiniLM-L6-v2)…')
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import NearestNeighbors
        import torch
    except ImportError as e:
        sys.exit(f'Missing dependency: {e}\n'
                  f'Install with: uv add sentence-transformers scikit-learn\n'
                  f'(or: pip install --user sentence-transformers scikit-learn)')

    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'  Using device: {device}')
    if device == 'cpu':
        print(f'  WARNING: CPU embedding ~{len(valid_ids)/200:.0f}s on 200 samples/s. '
              f'GPU is ~10-100x faster. Re-run with --device cuda:0 if available, '
              f'or submit as a small bsub job.')

    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    canonical_caps = [captions[iid][0] for iid in valid_ids]
    embeddings = model.encode(canonical_caps,
                                show_progress_bar=True,
                                convert_to_numpy=True,
                                batch_size=args.batch_size)

    # Build kNN
    print(f'\nBuilding kNN index (K = {args.K_image + 1})…')
    nn = NearestNeighbors(n_neighbors=args.K_image + 1, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Sample canonical images
    rng = np.random.RandomState(args.seed)
    sampled_idxs = rng.choice(len(valid_ids), size=args.n_samples, replace=False)

    paired = {}
    for i in sampled_idxs:
        img_id = valid_ids[i]
        similar_idxs = indices[i, 1:args.K_image + 1]  # skip self
        paired[img_id] = {
            'canonical_caption': captions[img_id][0],
            'captions': captions[img_id][:args.K_text],
            'similar_images': [valid_ids[j] for j in similar_idxs],
        }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(paired, f, indent=2)

    print(f'\nSaved {len(paired)} samples to {args.output}')
    print(f'Output size: {os.path.getsize(args.output) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
