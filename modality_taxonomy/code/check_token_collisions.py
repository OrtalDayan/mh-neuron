#!/usr/bin/env python3
"""
check_token_collisions.py — Check if generated descriptions contain tokens
that collide with the model's image placeholder token ID.

Such collisions cause tensor size mismatches during teacher-forcing in
InternVL (and potentially other models) because model.forward() selects
ALL positions matching the image token ID, not just the ones we injected.

Usage (called by run_pipeline.sh --step check_collisions):
    python code/check_token_collisions.py \
        --model_type internvl \
        --model_path modern_vlms/pretrained/InternVL2_5-8B \
        --desc_path results/1-describe/full/generated_descriptions_internvl2.5-8b.json
"""

import argparse                 # CLI argument parsing
import json                     # load description JSON
import sys                      # sys.exit


def main():
    p = argparse.ArgumentParser(
        description='Check for image token collisions in descriptions')
    p.add_argument('--model_type', required=True,                    # model backend name
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl',
                            'qwen2vl', 'llava-ov'])
    p.add_argument('--model_path', required=True,                    # local path to tokenizer/model
                   help='Path to pretrained model (used for tokenizer)')
    p.add_argument('--desc_path', required=True,                     # path to generated descriptions JSON
                   help='Path to generated_descriptions*.json')
    args = p.parse_args()

    # ── Load tokenizer + resolve image token ID ──────────────────
    print(f'Model type:  {args.model_type}')                         # show which model
    print(f'Model path:  {args.model_path}')                         # show tokenizer source
    print(f'Desc file:   {args.desc_path}')                          # show description file

    from transformers import AutoTokenizer, AutoProcessor             # import here to avoid slow startup if args are wrong

    if args.model_type == 'internvl':
        tokenizer = AutoTokenizer.from_pretrained(                   # InternVL: tokenizer is the processor
            args.model_path, trust_remote_code=True)
        image_token = '<IMG_CONTEXT>'                                # InternVL's image placeholder
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    elif args.model_type == 'qwen2vl':
        processor = AutoProcessor.from_pretrained(                   # Qwen2VL: use processor.tokenizer
            args.model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        image_token = '<|image_pad|>'                                # Qwen2VL's image placeholder
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    elif args.model_type == 'llava-ov':
        processor = AutoProcessor.from_pretrained(                   # LLaVA-OV: use processor.tokenizer
            args.model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        image_token = '<image>'                                      # LLaVA-OV image token
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or image_token_id == tokenizer.unk_token_id:
            image_token_id = 151655                                  # fallback known ID

    elif args.model_type == 'llava-hf':
        processor = AutoProcessor.from_pretrained(                   # HF LLaVA: use processor.tokenizer
            args.model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        image_token = '<image>'                                      # LLaVA-HF image token
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    else:  # llava-liuhaotian
        from transformers import AutoTokenizer as AT                 # original LLaVA uses plain tokenizer
        tokenizer = AT.from_pretrained(args.model_path)
        image_token = '<image>'
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    print(f'Image token: {image_token} → id={image_token_id}')      # show resolved token

    if image_token_id is None or image_token_id == getattr(tokenizer, 'unk_token_id', -1):
        print(f'WARNING: {image_token} not in vocab or maps to UNK — no collisions possible')
        return

    # ── Load descriptions ────────────────────────────────────────
    with open(args.desc_path) as f:                                  # load JSON
        raw = json.load(f)

    # Handle nested format: {id: {text: "..."}} or flat: {id: "..."}
    descriptions = {}                                                # normalised {id: text}
    for k, v in raw.items():                                         # iterate entries
        if isinstance(v, dict):                                      # nested format
            descriptions[k] = v.get('text', str(v))
        else:                                                        # flat format
            descriptions[k] = str(v)

    print(f'Descriptions loaded: {len(descriptions)}')

    # ── Check each description for collisions ────────────────────
    collisions = {}                                                  # {img_id: n_stray_tokens}
    total_stray = 0                                                  # total across all descriptions

    for img_id, desc in descriptions.items():                        # iterate descriptions
        ids = tokenizer.encode(desc, add_special_tokens=False)       # tokenize description only
        n_stray = sum(1 for t in ids if t == image_token_id)         # count collisions
        if n_stray > 0:                                              # found collision
            collisions[img_id] = n_stray
            total_stray += n_stray

    # ── Report ───────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  COLLISION REPORT: {args.model_type}')
    print(f'{"="*60}')
    print(f'  Total descriptions:   {len(descriptions)}')
    print(f'  With collisions:      {len(collisions)}')
    print(f'  Total stray tokens:   {total_stray}')

    if collisions:
        print(f'\n  Affected descriptions (showing first 20):')
        for i, (img_id, n) in enumerate(sorted(                     # sort by count descending
                collisions.items(), key=lambda x: -x[1])):
            if i >= 20:                                              # limit output
                print(f'  ... and {len(collisions) - 20} more')
                break
            desc_preview = descriptions[img_id][:80]                 # first 80 chars
            print(f'    {img_id}: {n:3d} stray tokens — {desc_preview}...')

        print(f'\n  ⚠ These descriptions will cause crashes without the')
        print(f'    IMG_CONTEXT filtering fix in prepare_inputs_internvl.')
        print(f'    Make sure neuron_modality_statistical.py has the fix.')
    else:
        print(f'\n  ✓ No collisions found — safe to run without filtering fix.')


if __name__ == '__main__':
    main()
