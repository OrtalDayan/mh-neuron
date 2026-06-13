#!/usr/bin/env python3
"""Combine Figure 3 panels from three hooks side-by-side (1×6 grid).

Layout per row:
    (a) Layer 27, Neuron 3900
    ┌──────────────────┬──────────────────┬──────────────────┐
    │ gate (Xu et al.) │ gate_up (ours)   │ attn (heads)     │
    │ FT + PMBT gate   │ PMBT gate_up     │ PMBT attn        │
    │ [img][overlay]   │ [img][overlay]   │ [img][overlay]   │
    │ [text tokens]    │ [text tokens]    │ [text tokens]    │
    └──────────────────┴──────────────────┴──────────────────┘

Usage:
    python combine_fig3_hooks.py \
        --dir1 .../fig3_gate_.../panels \
        --dir2 .../fig3_gate_up_.../panels \
        --dir3 .../fig3_attn_.../panels \
        --model_name llava-1.5-7b \
        --output fig3_combined_hooks.png
"""

import argparse
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont


def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()



def autocrop_whitespace(img, threshold=250, margin=4):
    """Crop white space from bottom of image."""
    import numpy as np
    arr = np.array(img)
    # Find last row that isn't all-white
    non_white = arr.min(axis=(1, 2)) < threshold
    if not non_white.any():
        return img
    last_row = non_white.nonzero()[0][-1]
    crop_h = min(last_row + margin, img.height)
    return img.crop((0, 0, img.width, crop_h))


def load_panel(panel_dir, panel_letter, model_name):
    if panel_dir is None:
        return None
    candidates = [
        f'fig3_panel_({panel_letter})_{model_name}.png',
        f'fig3_panel_{panel_letter}_{model_name}.png',
    ]
    for name in candidates:
        path = os.path.join(panel_dir, name)
        if os.path.isfile(path):
            return autocrop_whitespace(Image.open(path).convert('RGB'))
    available = os.listdir(panel_dir) if os.path.isdir(panel_dir) else []
    print(f'  WARNING: Panel ({panel_letter}) not found in {panel_dir}')
    print(f'  Available: {available[:10]}')
    return None


def load_metadata(panel_dir, panel_letter, model_name):
    if panel_dir is None:
        return None
    candidates = [
        f'fig3_panel_({panel_letter})_{model_name}_meta.json',
        f'fig3_panel_{panel_letter}_{model_name}_meta.json',
    ]
    for name in candidates:
        path = os.path.join(panel_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return None


def add_hook_label(img, label, font_size=34):
    """Add a small hook label banner at the top."""
    banner_h = font_size + 12
    new_img = Image.new('RGB', (img.width, img.height + banner_h), 'white')
    draw = ImageDraw.Draw(new_img)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (img.width - text_w) // 2
    draw.text((x, 4), label, fill='#333333', font=font)
    new_img.paste(img, (0, banner_h))
    return new_img


def combine_row(images, labels, suptitle=None, suptitle_font_size=44, gap=4):
    """Place 2-3 panel images side-by-side with suptitle, no dividers."""
    # Add hook labels
    labeled = []
    for img, label in zip(images, labels):
        if img is not None:
            labeled.append(add_hook_label(img, label))
        else:
            labeled.append(None)

    # Filter out None
    valid = [img for img in labeled if img is not None]
    if not valid:
        return None

    # Match heights
    max_h = max(img.height for img in valid)
    matched = []
    for img in labeled:
        if img is None:
            # Placeholder
            w = valid[0].width
            placeholder = Image.new('RGB', (w, max_h), (245, 245, 245))
            matched.append(placeholder)
        elif img.height < max_h:
            padded = Image.new('RGB', (img.width, max_h), 'white')
            padded.paste(img, (0, 0))
            matched.append(padded)
        else:
            matched.append(img)

    # Total width with small gaps (no divider lines)
    total_w = sum(img.width for img in matched) + gap * (len(matched) - 1)

    # Suptitle height
    suptitle_h = 0
    if suptitle:
        suptitle_h = suptitle_font_size + 14

    combined = Image.new('RGB', (total_w, max_h + suptitle_h), 'white')

    # Draw suptitle
    if suptitle:
        draw = ImageDraw.Draw(combined)
        font = get_font(suptitle_font_size)
        bbox = draw.textbbox((0, 0), suptitle, font=font)
        text_w = bbox[2] - bbox[0]
        x = (total_w - text_w) // 2
        draw.text((x, 4), suptitle, fill='black', font=font)

    # Paste panels
    x = 0
    for i, img in enumerate(matched):
        combined.paste(img, (x, suptitle_h))
        x += img.width + gap


    # Auto-crop bottom whitespace from combined row
    import numpy as np
    arr = np.array(combined)
    # Find last non-white row (threshold 250 to catch near-white)
    non_white = np.where(arr.min(axis=2) < 250)[0]
    if len(non_white) > 0:
        bottom = min(non_white[-1] + 8, combined.height)  # 8px margin
        combined = combined.crop((0, 0, combined.width, bottom))

    return combined


def make_grid(panels, padding=6):
    """Stack panels vertically with minimal padding."""
    valid = [p for p in panels if p is not None]
    if not valid:
        return None

    max_w = max(img.width for img in valid)
    total_h = sum(img.height for img in valid) + padding * (len(valid) + 1)

    grid = Image.new('RGB', (max_w + padding * 2, total_h), 'white')
    y = padding
    for img in valid:
        # Center horizontally
        x = (max_w + padding * 2 - img.width) // 2
        grid.paste(img, (x, y))
        y += img.height + padding

    return grid


# Default Xu et al. Figure 3 neurons
FIG3_DEFAULTS = {
    'a': {'layer': 27, 'neuron_idx': 3900},
    'b': {'layer': 2,  'neuron_idx': 4450},
    'c': {'layer': 29, 'neuron_idx': 600},
    'd': {'layer': 31, 'neuron_idx': 1800},
    'e': {'layer': 21, 'neuron_idx': 6100},
    'f': {'layer': 21, 'neuron_idx': 6100},
}


def main():
    p = argparse.ArgumentParser(
        description='Combine Figure 3 panels from 2-3 hooks side-by-side')
    p.add_argument('--dir1', required=True,
                   help='Panel dir for column 1 (gate)')
    p.add_argument('--dir2', required=True,
                   help='Panel dir for column 2 (gate_up)')
    p.add_argument('--dir3', default=None,
                   help='Panel dir for column 3 (attn, optional)')
    p.add_argument('--model_name', required=True)
    p.add_argument('--output', default='fig3_combined_hooks.png')
    p.add_argument('--label1', default='gate (Xu et al.)')
    p.add_argument('--label2', default='gate_up (ours)')
    p.add_argument('--label3', default='attn (heads)')
    p.add_argument('--panels', default='a,b,c,d,e,f')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    panel_letters = [x.strip() for x in args.panels.split(',')]
    dirs = [args.dir1, args.dir2]
    labels = [args.label1, args.label2]
    if args.dir3:
        dirs.append(args.dir3)
        labels.append(args.label3)

    n_cols = len(dirs)
    print(f'Combining {len(panel_letters)} panels × {n_cols} hooks:')
    for i, (d, l) in enumerate(zip(dirs, labels)):
        print(f'  Col {i+1}: {d} — {l}')

    rows = []
    for letter in panel_letters:
        print(f'\nPanel ({letter}):')
        images = [load_panel(d, letter, args.model_name) for d in dirs]

        # Get metadata for suptitle
        meta = None
        for d in dirs:
            meta = load_metadata(d, letter, args.model_name)
            if meta:
                break
        if meta is None:
            meta = FIG3_DEFAULTS.get(letter, {})

        layer = meta.get('layer', '?')
        nidx = meta.get('neuron_idx', '?')
        suptitle = f'({letter})  Layer {layer}, Neuron {nidx}'

        row = combine_row(images, labels, suptitle=suptitle)
        rows.append(row)
        if row:
            print(f'  {suptitle} → {row.size}')

    n_valid = sum(1 for r in rows if r is not None)
    if n_valid == 0:
        print('\nERROR: No panels to combine')
        sys.exit(1)

    print(f'\nStacking {n_valid} rows...')
    grid = make_grid(rows, padding=2)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    grid.save(args.output, dpi=(args.dpi, args.dpi))
    print(f'Saved: {args.output} ({grid.size[0]}x{grid.size[1]} px)')


if __name__ == '__main__':
    main()