#!/usr/bin/env python3
"""Step 5b — Ranked Fraction Ablation: Dose-response taxonomy validation.

Ranks neurons within each modality category by a combined score
(CETT-diff × classification confidence) and ablates top-N% fractions
by zeroing their down_proj weight columns, measuring the effect on
POPE, CHAIR, TriviaQA and MMLU benchmarks. Weight zeroing is
mathematically equivalent to activation zeroing but operates directly
on model weights, connecting the diagnostic experiment to permanent
weight editing interventions.

Equal-count design (phases 3-4): all conditions ablate the same
absolute number of neurons, determined by min(visual, text, multimodal)
count × fraction. This removes the confound that larger categories
contribute more network capacity. Unknown neurons are ablated at 100%
(single run) to confirm they are functionally inert.

Four phases:
  Phase 0 (1 GPU):  Contrastive POPE filtering + CETT-diff scoring → neuron ranking
  Phase 1 (N GPUs): One job per (category, fraction) → benchmark evaluation
  Phase 2 (CPU):    Merge results → dose-response summary + optional enrichment JSON
  Phase 3 (N GPUs): Random-sample equal-count trials (30 seeds per condition × fraction)
  Phase 4 (CPU):    Merge random trials → statistics + Mann-Whitney pairwise tests

Usage:
  # Phase 0: compute ranking
  python ranked_fraction_ablation.py --phase 0 --model_type llava-ov ...

  # Phase 1: evaluate one condition
  python ranked_fraction_ablation.py --phase 1 --condition visual --fraction 0.05 ...

  # Phase 2: merge all results
  python ranked_fraction_ablation.py --phase 2 ...

  # Phase 3: one random trial (equal-count)
  python ranked_fraction_ablation.py --phase 3 --condition visual --fraction 0.10 --trial 0 ...

  # Phase 4: merge trials + statistics
  python ranked_fraction_ablation.py --phase 4 ...
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# Inlined evaluation functions (from neuron_ablation_validate.py)
# ═══════════════════════════════════════════════════════════════════════

# ImageNet normalisation constants for InternVL preprocessing
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12,
                                  image_size=448, use_thumbnail=True):
    """Tile a PIL image into sub-patches for InternVL2.5."""
    orig_w, orig_h = image.size
    aspect_ratio   = orig_w / orig_h

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = min(
        target_ratios,
        key=lambda r: abs(aspect_ratio - r[0] / r[1])
    )
    rows, cols = best_ratio

    resized = image.resize(
        (image_size * rows, image_size * cols), resample=Image.BICUBIC)

    patches = [
        resized.crop((
            col * image_size, row * image_size,
            (col + 1) * image_size, (row + 1) * image_size,
        ))
        for row in range(rows)
        for col in range(cols)
    ]

    if use_thumbnail and len(patches) > 1:
        patches.append(image.resize((image_size, image_size),
                                     resample=Image.BICUBIC))
    return patches


def _internvl_preprocess_image(img, max_num=12):
    """Convert a PIL image to InternVL pixel_values tensor."""
    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(
            (448, 448),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    patches      = _internvl_dynamic_preprocess(img, max_num=max_num)
    pixel_values = torch.stack([transform(p) for p in patches])
    return pixel_values.to(torch.bfloat16)


def generate_answer(model, model_type, tokenizer_or_processor, image_processor,
                    image_token_id, img, question, device, max_new_tokens=512):
    """Generate a text answer for an image + question.

    Handles HF LLaVA, original LLaVA (liuhaotian), LLaVA-Next-LLaMA3,
    InternVL2.5, LLaVA-OneVision, and Qwen2.5-VL backends.
    """
    import torch
    with torch.no_grad():
        if model_type in ('hf', 'llava-hf'):
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = tokenizer_or_processor(
                images=img, text=prompt, return_tensors='pt'
            ).to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        elif model_type == 'llava-llama3':
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            prompt = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                images=img, text=prompt, return_tensors='pt'
            ).to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        elif model_type in ('liuhaotian', 'llava-liuhaotian'):
            from llava.conversation import conv_templates
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token

            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], f"<image>\n{question}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer_or_processor, IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(device)

            image_tensor = image_processor.preprocess(
                img, return_tensors='pt'
            )['pixel_values'].half().to(device)

            output_ids = model.generate(
                input_ids, images=image_tensor,
                max_new_tokens=max_new_tokens, do_sample=False)
            answer = tokenizer_or_processor.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        elif model_type == 'internvl':
            pixel_values = _internvl_preprocess_image(img).to(device)
            generation_config = dict(
                max_new_tokens=max_new_tokens, do_sample=False)
            answer = model.chat(
                tokenizer_or_processor, pixel_values,
                question, generation_config)

        elif model_type == 'llava-ov':
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            prompt_text = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                images=img, text=prompt_text,
                return_tensors='pt').to(device)

            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        elif model_type == 'idefics2':
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            prompt_text = tokenizer_or_processor.apply_chat_template(
                messages, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                text=prompt_text, images=[img],
                return_tensors='pt').to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        else:  # qwen2vl
            from qwen_vl_utils import process_vision_info

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": question},
            ]}]
            prompt_text = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = tokenizer_or_processor(
                text=[prompt_text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors='pt').to(device)

            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

    return answer.strip()


def generate_answer_text_only(model, model_type, tokenizer_or_processor,
                              question, device, max_new_tokens=512):
    """Generate a text answer WITHOUT any image input.

    Removes the vision encoder overhead entirely — no dummy image,
    no image tokens.  Currently supports llava-llama3, llava-ov,
    hf/llava-hf, and qwen2vl.
    """
    import torch
    with torch.no_grad():
        if model_type in ('hf', 'llava-hf'):
            prompt = f"USER: {question}\nASSISTANT:"
            inputs = tokenizer_or_processor(
                text=prompt, return_tensors='pt'
            ).to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        elif model_type == 'llava-llama3':
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]
            prompt = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                text=prompt, return_tensors='pt'
            ).to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        elif model_type in ('liuhaotian', 'llava-liuhaotian'):
            from llava.conversation import conv_templates
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token

            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_or_processor.encode(
                prompt, return_tensors='pt'
            ).to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens, do_sample=False)
            answer = tokenizer_or_processor.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        elif model_type == 'internvl':
            generation_config = dict(
                max_new_tokens=max_new_tokens, do_sample=False)
            # InternVL chat() can accept pixel_values=None for text-only
            answer = model.chat(
                tokenizer_or_processor, None,
                question, generation_config)

        elif model_type == 'llava-ov':
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]
            prompt_text = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                text=prompt_text,
                return_tensors='pt').to(device)

            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        elif model_type == 'idefics2':
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]
            prompt_text = tokenizer_or_processor.apply_chat_template(
                messages, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                text=prompt_text,
                return_tensors='pt').to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

        else:  # qwen2vl
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]
            prompt_text = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_or_processor(
                text=[prompt_text],
                padding=True, return_tensors='pt').to(device)

            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = inputs['input_ids'].shape[1]
            answer = tokenizer_or_processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True)

    return answer.strip()


# Blank white image used as dummy input for text-only benchmarks.
DUMMY_IMAGE = Image.new('RGB', (224, 224), (255, 255, 255))


def load_pope_questions(pope_path, num_questions=None):
    """Load POPE questions from JSONL file.

    Returns list of dicts with keys: question_id, image, text, label
    """
    questions = []
    with open(pope_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            questions.append(q)
    if num_questions is not None and num_questions < len(questions):
        questions = questions[:num_questions]
    return questions


def evaluate_pope(model, model_type, tokenizer_or_processor, image_processor,
                  image_token_id, questions, img_dir, device):
    """Run POPE evaluation: ask yes/no questions, measure accuracy + F1.

    Returns dict with accuracy, precision, recall, f1, yes_ratio, n_questions.
    """
    correct = 0
    tp = fp = fn = tn = 0
    total = 0
    n_yes_pred = 0

    for q in tqdm(questions, desc='POPE eval'):
        img_path = os.path.join(img_dir, q['image'])
        if not os.path.isfile(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        question_text = q['text']
        gt_label = q['label'].strip().lower()

        # Append short-answer suffix (official LLaVA-1.5 POPE eval format).
        # Without this, models like LLaVA-1.5 generate verbose responses
        # that bias toward "No" answers.
        pope_question = question_text + "\nAnswer the question using a single word or phrase."

        answer = generate_answer(model, model_type, tokenizer_or_processor,
                                 image_processor, image_token_id,
                                 img, pope_question, device,
                                 max_new_tokens=10)
        answer_lower = answer.strip().lower()

        pred_yes = answer_lower.startswith('yes')
        gt_yes = gt_label == 'yes'

        if pred_yes == gt_yes:
            correct += 1
        if pred_yes and gt_yes:
            tp += 1
        elif pred_yes and not gt_yes:
            fp += 1
        elif not pred_yes and gt_yes:
            fn += 1
        else:
            tn += 1

        if pred_yes:
            n_yes_pred += 1
        total += 1

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    yes_ratio = n_yes_pred / max(total, 1)

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'yes_ratio': round(yes_ratio, 4),
        'n_questions': total,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def load_coco_objects(ann_path):
    """Load COCO instance annotations → {image_id: set of category names}."""
    with open(ann_path) as f:
        data = json.load(f)
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    image_objects = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_name = cat_id_to_name[ann['category_id']]
        image_objects.setdefault(img_id, set()).add(cat_name.lower())
    return image_objects, cat_id_to_name


# COCO synonym mapping (from CHAIR-metric-standalone, Rohrbach et al.)
# Maps common alternate names → canonical COCO category name.
CHAIR_SYNONYMS = {
    'motorbike': 'motorcycle', 'aeroplane': 'airplane',
    'sofa': 'couch', 'tv': 'television',
    'cell phone': 'cell phone', 'mobile phone': 'cell phone',
    'laptop computer': 'laptop', 'hot dog': 'hot dog',
    'teddy bear': 'teddy bear', 'fire hydrant': 'fire hydrant',
    'stop sign': 'stop sign', 'parking meter': 'parking meter',
    'wine glass': 'wine glass', 'baseball bat': 'baseball bat',
    'baseball glove': 'baseball glove', 'tennis racket': 'tennis racket',
    'sports ball': 'sports ball', 'potted plant': 'potted plant',
    'hair drier': 'hair drier', 'motor bike': 'motorcycle',
    'motor cycle': 'motorcycle', 'air plane': 'airplane',
    'suit case': 'suitcase', 'traffic light': 'traffic light',
    'street light': 'traffic light', 'bow tie': 'tie',
    'stove top oven': 'oven', 'dining table': 'dining table',
    'bath tub': 'bathtub', 'hair dryer': 'hair drier',
}


def _build_chair_matchers(all_cat_names):
    """Build sorted lists of multi-word and single-word COCO object names.

    Returns (multi_word_names, single_word_names) both as sorted lists
    (longest first for multi-word) with their canonical forms.
    Each entry is (pattern, canonical_name).
    """
    import re as _re
    # Build canonical lookup: alternate name → COCO canonical name
    canon = {}
    for name in all_cat_names:
        canon[name] = name
    for alt, can in CHAIR_SYNONYMS.items():
        canon[alt] = can

    # Separate single-word and multi-word patterns
    multi = []   # (pattern_str, canonical)
    single = []  # (pattern_str, canonical)
    for pattern, canonical in canon.items():
        if ' ' in pattern:
            multi.append((pattern, canonical))
        else:
            single.append((pattern, canonical))

    # Sort multi-word longest first to match "traffic light" before "light"
    multi.sort(key=lambda x: len(x[0]), reverse=True)
    return multi, single


def evaluate_chair(model, model_type, tokenizer_or_processor, image_processor,
                   image_token_id, ann_path, img_dir, device,
                   num_images=500, seed=42):
    """Run CHAIR evaluation: generate captions, measure object hallucination.

    Follows SRF (Ali et al.) and CHAIR-metric-standalone methodology:
    - Select images with ≥3 annotated objects
    - Use synonym mapping for COCO category matching
    - Word-boundary matching to avoid substring false positives
    - Caption prompt: "Please describe this image in detail."
    - Max 64 tokens (matching SRF paper)

    CHAIR_i = hallucinated objects / total mentioned objects
    CHAIR_s = captions with ≥1 hallucinated object / total captions
    """
    import re

    image_objects, cat_id_to_name = load_coco_objects(ann_path)
    all_cat_names = set(n.lower() for n in cat_id_to_name.values())

    # Build matchers with synonym support
    multi_patterns, single_patterns = _build_chair_matchers(all_cat_names)

    with open(ann_path) as f:
        data = json.load(f)
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    # Filter: only images with ≥3 annotated objects (standard practice)
    valid_ids = [iid for iid, objs in image_objects.items()
                 if len(objs) >= 3 and iid in id_to_filename]

    rng = np.random.RandomState(seed)
    if num_images < len(valid_ids):
        chosen_ids = rng.choice(valid_ids, size=num_images, replace=False).tolist()
    else:
        chosen_ids = valid_ids

    total_objects_mentioned = 0
    total_hallucinated = 0
    total_caps = 0
    caps_with_hallucination = 0
    total_caption_len = 0
    total_gt_objects = 0
    total_recalled = 0

    caption_prompt = "Please describe this image in detail."

    for img_id in tqdm(chosen_ids, desc='CHAIR eval'):
        filename = id_to_filename[img_id]
        img_path = os.path.join(img_dir, filename)
        if not os.path.isfile(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        caption = generate_answer(model, model_type, tokenizer_or_processor,
                                  image_processor, image_token_id,
                                  img, caption_prompt, device,
                                  max_new_tokens=64)

        gt_objects = image_objects.get(img_id, set())
        caption_lower = caption.lower()
        total_caption_len += len(caption.split())
        total_caps += 1

        # Extract mentioned objects using word-boundary matching
        mentioned = set()    # canonical names mentioned in caption
        hallucinated = set() # canonical names that are hallucinated

        # 1. Multi-word patterns first (e.g. "traffic light", "hot dog")
        for pattern, canonical in multi_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', caption_lower):
                mentioned.add(canonical)
                if canonical not in gt_objects:
                    hallucinated.add(canonical)

        # 2. Single-word patterns with word-boundary matching
        words = re.findall(r'\b\w+\b', caption_lower)
        for word in words:
            for pattern, canonical in single_patterns:
                if word == pattern:
                    mentioned.add(canonical)
                    if canonical not in gt_objects:
                        hallucinated.add(canonical)

        total_objects_mentioned += len(mentioned)
        total_hallucinated += len(hallucinated)
        if hallucinated:
            caps_with_hallucination += 1
        total_gt_objects += len(gt_objects)
        total_recalled += len(mentioned & gt_objects)

    chair_i = total_hallucinated / max(total_objects_mentioned, 1)
    chair_s = caps_with_hallucination / max(total_caps, 1)
    recall = total_recalled / max(total_gt_objects, 1)
    avg_len = total_caption_len / max(total_caps, 1)

    return {
        'chair_i': round(chair_i, 4),
        'chair_s': round(chair_s, 4),
        'recall': round(recall, 4),
        'n_images': total_caps,
        'total_objects_mentioned': total_objects_mentioned,
        'total_hallucinated': total_hallucinated,
        'total_captions': total_caps,
        'captions_with_hallucination': caps_with_hallucination,
        'avg_caption_len': round(avg_len, 1),
    }


def load_triviaqa(path, num_questions=None, seed=42):
    """Load TriviaQA questions from verified-web-dev.json."""
    with open(path) as f:
        data = json.load(f)

    items = []
    for entry in data['Data']:
        answer_obj = entry['Answer']
        aliases = set()
        aliases.add(answer_obj['Value'].strip().lower())
        for alias in answer_obj.get('Aliases', []):
            aliases.add(alias.strip().lower())
        items.append({
            'question': entry['Question'],
            'answer': answer_obj['Value'],
            'aliases': aliases,
        })

    if num_questions is not None and num_questions < len(items):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(items), size=num_questions, replace=False)
        items = [items[i] for i in indices]

    return items


def evaluate_triviaqa(model, model_type, tokenizer_or_processor, image_processor,
                      image_token_id, items, device, text_only=False):
    """Run TriviaQA evaluation.  When text_only=True, no image is fed."""
    correct = 0
    total = 0

    for item in tqdm(items, desc='TriviaQA eval'):
        prompt = (f"Answer the following question briefly.\n"
                  f"Question: {item['question']}\nAnswer:")

        if text_only:
            answer = generate_answer_text_only(
                model, model_type, tokenizer_or_processor,
                prompt, device, max_new_tokens=30)
        else:
            answer = generate_answer(
                model, model_type, tokenizer_or_processor,
                image_processor, image_token_id,
                DUMMY_IMAGE, prompt, device, max_new_tokens=30)

        answer_lower = answer.strip().lower()
        if any(alias in answer_lower for alias in item['aliases']):
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {
        'accuracy': round(accuracy, 4),
        'n_questions': total,
    }


def load_mmlu(mmlu_dir, num_questions=None, seed=42):
    """Load MMLU multiple-choice questions from test/*.csv files."""
    import csv

    test_dir = os.path.join(mmlu_dir, 'test')
    if not os.path.isdir(test_dir):
        test_dir = mmlu_dir

    items = []
    csv_files = sorted(glob.glob(os.path.join(test_dir, '*.csv')))

    for csv_path in csv_files:
        subject = os.path.basename(csv_path).replace('_test.csv', '')
        subject = subject.replace('_', ' ')
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = row[0], row[1], row[2], row[3], row[4], row[5]
                items.append({
                    'subject': subject,
                    'question': question,
                    'choices': {'A': a, 'B': b, 'C': c, 'D': d},
                    'answer': answer.strip().upper(),
                })

    if num_questions is not None and num_questions < len(items):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(items), size=num_questions, replace=False)
        items = [items[i] for i in indices]

    return items


def evaluate_mmlu(model, model_type, tokenizer_or_processor, image_processor,
                  image_token_id, items, device, text_only=False):
    """Run MMLU evaluation.  When text_only=True, no image is fed."""
    correct = 0
    total = 0

    for item in tqdm(items, desc='MMLU eval'):
        choices = item['choices']
        prompt = (f"Answer the following multiple choice question. "
                  f"Reply with just the letter (A, B, C, or D).\n\n"
                  f"Question: {item['question']}\n"
                  f"A. {choices['A']}\n"
                  f"B. {choices['B']}\n"
                  f"C. {choices['C']}\n"
                  f"D. {choices['D']}\n"
                  f"Answer:")

        if text_only:
            answer = generate_answer_text_only(
                model, model_type, tokenizer_or_processor,
                prompt, device, max_new_tokens=5)
        else:
            answer = generate_answer(
                model, model_type, tokenizer_or_processor,
                image_processor, image_token_id,
                DUMMY_IMAGE, prompt, device, max_new_tokens=5)

        pred_letter = answer.strip().upper()
        if pred_letter and pred_letter[0] in 'ABCD':
            pred_letter = pred_letter[0]
        else:
            pred_letter = ''

        if pred_letter == item['answer']:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {
        'accuracy': round(accuracy, 4),
        'n_questions': total,
    }


# ═══════════════════════════════════════════════════════════════════════
# MathVerse evaluation (option A — in-memory, no VLMEvalKit)
# ═══════════════════════════════════════════════════════════════════════

def load_mathverse(data_dir, subtask='Text_Dominant'):
    """Load MathVerse questions from prepare_mathverse_td.py output.

    Args:
        data_dir: directory containing questions.json and images/
        subtask: 'Text_Dominant', 'Vision_Only', or 'Vision_Dominant'

    Returns: list of dicts with keys: question, answer, image_path, choices
    """
    questions_path = os.path.join(data_dir, f'questions_{subtask}.json')
    if not os.path.exists(questions_path):
        # Fallback: single questions.json with subtask field
        questions_path = os.path.join(data_dir, 'questions.json')
    if not os.path.exists(questions_path):
        print(f'  WARNING: MathVerse data not found at {questions_path}')
        return []

    with open(questions_path) as f:
        items = json.load(f)

    img_dir = os.path.join(data_dir, 'images')
    result = []
    for item in items:
        # Filter by subtask if single file
        if 'problem_version' in item:
            version = item['problem_version'].replace(' ', '_')
            if version != subtask:
                continue

        img_path = os.path.join(img_dir, item.get('image', ''))
        if not os.path.isfile(img_path):
            continue

        result.append({
            'question': item.get('question', ''),
            'answer': str(item.get('answer', '')).strip(),
            'image_path': img_path,
            'question_type': item.get('question_type', 'multi-choice'),
        })
    return result


def _extract_answer_letter(pred):
    """Extract answer letter (A-D) from model prediction."""
    import re
    s = str(pred).strip()
    if not s:
        return None
    # "The correct option letter is C."
    m = re.search(r'correct.*?(?:option|answer).*?letter.*?([A-D])', s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # "The answer is C" or "Answer: C"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-D])\b', s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # Starts with letter: "B", "B: 54°", "B. 25°"
    m = re.match(r'^([A-D])(?:\s*[:.)\s]|$)', s)
    if m: return m.group(1).upper()
    # "option C"
    m = re.search(r'(?:option|choice)\s+([A-D])\b', s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # Any standalone A-D
    m = re.search(r'\b([A-D])\b', s)
    if m: return m.group(1).upper()
    return None


def evaluate_mathverse(model, model_type, tokenizer_or_processor, image_processor,
                       image_token_id, questions, device, max_new_tokens=256):
    """Run MathVerse evaluation: multi-choice math, measure accuracy.

    Returns dict with accuracy, n_questions, n_parsed.
    """
    correct = 0
    total = 0
    n_parsed = 0

    for q in tqdm(questions, desc='MathVerse eval'):
        try:
            img = Image.open(q['image_path']).convert('RGB')
        except Exception:
            continue

        answer = generate_answer(model, model_type, tokenizer_or_processor,
                                 image_processor, image_token_id,
                                 img, q['question'], device,
                                 max_new_tokens=max_new_tokens)

        pred_letter = _extract_answer_letter(answer)
        gt_letter = _extract_answer_letter(q['answer'])

        total += 1
        if pred_letter is not None and gt_letter is not None:
            n_parsed += 1
            if pred_letter == gt_letter:
                correct += 1

    accuracy = correct / max(total, 1)
    return {
        'accuracy': round(accuracy, 4),
        'n_questions': total,
        'n_parsed': n_parsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Ranked fraction ablation')

    # Phase selection
    p.add_argument('--phase', type=int, required=True, choices=[0, 1, 2, 3, 4, 5],
                   help='0=ranking, 1=evaluate condition, 2=merge results, '
                        '3=random-sample trial, 4=merge trials + statistics, '
                        '5=SNRF-style 100%% ablation (in-memory, one GPU job)')

    # Model
    p.add_argument('--model_type', type=str, required=True,
                   help='Backend: llava-ov | internvl | qwen2vl | qwen25vl-7b | qwen25vl-3b | idefics2 | llava-hf | llava-llama3')
    p.add_argument('--model_path', type=str, default=None,
                   help='Path to pretrained model (required for phase 0 and 1)')
    p.add_argument('--model_name', type=str, default=None,
                   help='Human-readable model name for outputs')
    p.add_argument('--n_layers', type=int, required=True)

    # Labels
    p.add_argument('--label_dir', type=str, required=True,
                   help='Directory with neuron_labels_permutation_all.json')
    p.add_argument('--taxonomy', type=str, default='pmbt')

    # Data paths
    p.add_argument('--pope_path', type=str, default=None)
    p.add_argument('--pope_img_dir', type=str, default=None)
    p.add_argument('--pope_splits_dir', type=str, default=None)
    p.add_argument('--triviaqa_path', type=str, default=None)
    p.add_argument('--triviaqa_num', type=int, default=2000)
    p.add_argument('--chair_ann_path', type=str, default=None,
                   help='Path to COCO instances_val2014.json for CHAIR eval')
    p.add_argument('--chair_img_dir', type=str, default=None,
                   help='Path to COCO val2014 images for CHAIR eval')
    p.add_argument('--chair_num_images', type=int, default=500,
                   help='Number of images for CHAIR evaluation')
    p.add_argument('--mmlu_dir', type=str, default=None,
                   help='Path to MMLU data directory')
    p.add_argument('--mmlu_num', type=int, default=2000,
                   help='Number of MMLU questions')

    # MathVerse data (phase 5)
    p.add_argument('--mathverse_dir', type=str, default=None,
                   help='Directory with MathVerse questions.json + images/ (from prepare_mathverse_td.py)')
    p.add_argument('--mathverse_subtasks', type=str,
                   default='Text_Dominant,Vision_Only',
                   help='Comma-separated MathVerse subtasks to evaluate')
    p.add_argument('--phase5_condition', type=str, default='all',
                   help='Phase 5: run single condition (visual|text|multimodal|'
                        'random_visual_count|random_text_count|random_multimodal_count|'
                        'baseline) or "all" for sequential')
    p.add_argument('--phase5_benchmark', type=str, default='all',
                   help='Phase 5: run single benchmark (POPE|MV_Text_Dominant|'
                        'MV_Vision_Only|TriviaQA) or "all"')
    p.add_argument('--phase5_limit', type=int, default=0,
                   help='Phase 5: limit samples per benchmark (0=no limit, e.g. 20 for smoke test)')

    # Contrastive settings (phase 0)
    p.add_argument('--contrastive_start_per_split', type=int, default=1250)
    p.add_argument('--contrastive_samples', type=int, default=10)
    p.add_argument('--contrastive_cap_per_split', type=int, default=333)
    p.add_argument('--triviaqa_cap', type=int, default=1000)

    # Phase 1: condition to evaluate
    p.add_argument('--condition', type=str, default=None,
                   help='Category to ablate: visual|text|multimodal|unknown|random')
    p.add_argument('--fraction', type=float, default=None,
                   help='Fraction of neurons in category to ablate: 0.01, 0.05, ..., 1.0')

    # Phase 3: random trial index
    p.add_argument('--trial', type=int, default=None,
                   help='Trial index for random-sample ablation (phase 3)')
    p.add_argument('--n_trials', type=int, default=30,
                   help='Total number of trials per condition (phase 4 merge)')
    p.add_argument('--text_only_benchmarks', action='store_true',
                   help='Run TriviaQA/MMLU without dummy image (pure text input)')

    # Ranking method
    p.add_argument('--ranking', type=str, default='cett',
                   choices=['cett', 'selectivity', 'combined', 'random_trials'],
                   help='Neuron ranking method: '
                        'cett = CETT-diff × (1-p), '
                        'selectivity = |observed_rate_diff| from PMBT, '
                        'combined = CETT-diff × |observed_rate_diff|, '
                        'random_trials = random sampling (phases 3-4 only)')

    # Output
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Label loading with p-values
# ═══════════════════════════════════════════════════════════════════════

def load_labels_with_pvalues(label_dir, taxonomy, n_layers):
    """Load neuron labels and return per-neuron (label, p_value) tuples.

    Returns:
        neurons: list of dicts, each with keys:
            layer, neuron_idx, label, p_value, key (str 'layer_neuron')
    """
    label_filename = ('neuron_labels.json' if taxonomy == 'ft'
                      else 'neuron_labels_permutation.json')
    merged_name = label_filename.replace('.json', '_all.json')

    # Try merged file first, then try with model name suffix
    labels_raw = None
    for candidate in [merged_name]:
        merged_path = os.path.join(label_dir, candidate)
        if os.path.isfile(merged_path):
            with open(merged_path) as f:
                labels_raw = json.load(f)
            print(f'  Loaded labels from {merged_path}')
            break

    # Try glob for model-suffixed files
    if labels_raw is None:
        import glob
        pattern = os.path.join(label_dir,
                               label_filename.replace('.json', '_all*.json'))
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0]) as f:
                labels_raw = json.load(f)
            print(f'  Loaded labels from {matches[0]}')

    if labels_raw is None:
        raise FileNotFoundError(
            f'No merged label file found in {label_dir}')

    neurons = []
    for layer_idx in range(n_layers):
        key = str(layer_idx)
        if key not in labels_raw:
            continue
        layer_data = sorted(labels_raw[key],
                            key=lambda x: x['neuron_idx'])
        for entry in layer_data:
            neurons.append({
                'layer': layer_idx,
                'neuron_idx': entry['neuron_idx'],
                'label': entry['label'],
                'p_value': entry.get('p_value', 0.0),
                'rate_diff': entry.get('observed_rate_diff', 0.0),
                'key': f"{layer_idx}_{entry['neuron_idx']}",
            })

    print(f'  {len(neurons)} neurons loaded across {n_layers} layers')
    return neurons


# ═══════════════════════════════════════════════════════════════════════
# Phase 0: Compute neuron ranking
# ═══════════════════════════════════════════════════════════════════════

def run_phase0(args):
    """Contrastive POPE + CETT-diff → combined ranking per neuron."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'PHASE 0: Compute neuron ranking')
    print(f'{"="*60}\n')

    # ── Load labels with p-values ──
    neurons = load_labels_with_pvalues(
        args.label_dir, args.taxonomy, args.n_layers)

    # ── For random_trials: no-op (phase 3 uses raw POPE/TriviaQA directly) ──
    if args.ranking == 'random_trials':
        print(f'  [random_trials] Equal-count mode — no contrastive data needed')
        print(f'  Phase 3 will evaluate raw POPE/TriviaQA/CHAIR/MMLU directly.')
        print(f'\n  Phase 0 (random_trials) complete.')
        return

    # ── Non-random-trials: ranked ablation (requires halluc_score_neurons) ──
    try:
        from halluc_score_neurons import (
            get_layer_names, build_contrastive_pope_set,
        )
    except ImportError:
        print('  ERROR: Ranked ablation (cett/selectivity/combined) requires')
        print('         halluc_score_neurons.py in the code/ directory.')
        print('         For equal-count validation, use --ranking random_trials')
        return

    # ── Load raw CETT scores (skip if selectivity-only ranking) ──
    cett_scores = None
    if args.ranking in ('cett', 'combined'):
        cached_cett_mean = os.path.join(args.output_dir, 'cett_mean_scores.json')

        if os.path.isfile(cached_cett_mean):
            with open(cached_cett_mean) as f:
                cett_ser = json.load(f)
            cett_scores = {
                (int(k.split('_')[0]), int(k.split('_')[1])): v
                for k, v in cett_ser.items()
            }
            print(f'  [CACHE] Loaded raw mean CETT: {len(cett_scores)} neurons')
        else:
            # Compute CETT scores. If contrastive_pope.jsonl already exists
            # (e.g. symlinked from step 10), skip sampling and only run CETT.
            cached_pope = os.path.join(args.output_dir, 'contrastive_pope.jsonl')
            skip = os.path.isfile(cached_pope)

            # Break any symlinks for CETT output files so we don't
            # overwrite step 10's data
            for _cett_f in ['cett_diff_scores.json', 'cett_mean_scores.json']:
                _cett_p = os.path.join(args.output_dir, _cett_f)
                if os.path.islink(_cett_p):
                    os.unlink(_cett_p)

            if skip:
                print(f'  Computing CETT scores (using cached contrastive set)...')
            else:
                print(f'  Computing contrastive POPE + CETT scores...')
            _, n_clean, _ = build_contrastive_pope_set(args, skip_sampling=skip)

            if os.path.isfile(cached_cett_mean):
                with open(cached_cett_mean) as f:
                    cett_ser = json.load(f)
                cett_scores = {
                    (int(k.split('_')[0]), int(k.split('_')[1])): v
                    for k, v in cett_ser.items()
                }
                print(f'  Loaded raw mean CETT: {len(cett_scores)} neurons')
            else:
                print('  ERROR: CETT computation did not produce cett_mean_scores.json')
                print('  Make sure halluc_score_neurons.py is updated to save mean CETT.')
                return
    else:
        print(f'  [selectivity ranking] Skipping CETT — using rate_diff from PMBT labels only')

    # ── Store raw values per neuron ──
    ranking_method = args.ranking
    print(f'\n  Building neuron data (ranking method: {ranking_method})...')

    ranking = {}  # key → {raw values + score (set per-category below)}
    n_missing_cett = 0

    for n in neurons:
        k = (n['layer'], n['neuron_idx'])
        cett_val = cett_scores.get(k, 0.0) if cett_scores else 0.0
        if cett_scores and k not in cett_scores:
            n_missing_cett += 1

        ranking[n['key']] = {
            'score': 0.0,  # will be set per-category below
            'cett': cett_val,
            'p_value': n['p_value'],
            'rate_diff': n['rate_diff'],
            'label': n['label'],
            'layer': n['layer'],
            'neuron_idx': n['neuron_idx'],
        }

    if n_missing_cett > 0:
        print(f'  WARNING: {n_missing_cett} neurons missing CETT '
              f'(set to 0)')

    # ── Rank within each category and set score = actual sort key ──
    categories = ['visual', 'text', 'multimodal', 'unknown']
    ranked_by_cat = {}

    def _sort_key(cat, v):
        """Return the sort key for a neuron in the given category.
        Higher = ranked first (all sort descending)."""
        if ranking_method == 'selectivity':
            if cat == 'visual':
                return v['rate_diff']           # highest D first
            elif cat == 'text':
                return -v['rate_diff']          # most negative D first (flip sign)
            else:                               # multimodal, unknown
                return -abs(v['rate_diff'])      # lowest |D| first (flip sign)
        elif ranking_method == 'cett':
            return v['cett']                     # highest CETT first, all categories
        elif ranking_method == 'combined':
            if cat == 'visual':
                return v['cett'] * v['rate_diff']        # highest CETT×D
            elif cat == 'text':
                return v['cett'] * (-v['rate_diff'])     # highest CETT×(−D)
            else:                                         # multimodal, unknown
                return v['cett']                          # highest CETT
        return 0.0

    for cat in categories:
        cat_neurons = [(k, v) for k, v in ranking.items()
                       if v['label'] == cat]

        # Sort descending by sort key
        cat_neurons.sort(key=lambda x: _sort_key(cat, x[1]), reverse=True)

        # Store the actual sort key as 'score' so it matches the ranking
        for k, v in cat_neurons:
            ranking[k]['score'] = _sort_key(cat, v)

        ranked_by_cat[cat] = [k for k, v in cat_neurons]
        print(f'  {cat:14s}: {len(ranked_by_cat[cat]):>7,} neurons ranked')

        # Print top 5 for sanity check
        if cat_neurons:
            print(f'    Top 5: ', end='')
            for k, v in cat_neurons[:5]:
                print(f'L{v["layer"]}n{v["neuron_idx"]}'
                      f'(score={v["score"]:.4f} D={v["rate_diff"]:+.4f} CETT={v["cett"]:.3f}) ',
                      end='')
            print()

    # ── Save ranking ──
    ranking_path = os.path.join(args.output_dir, 'neuron_ranking.json')
    with open(ranking_path, 'w') as f:
        json.dump(ranking, f, indent=2)

    ranked_cats_path = os.path.join(args.output_dir,
                                     'ranked_neurons_by_category.json')
    with open(ranked_cats_path, 'w') as f:
        json.dump(ranked_by_cat, f, indent=2)

    print(f'\n  Ranking saved to {ranking_path}')
    print(f'  Per-category ranked lists saved to {ranked_cats_path}')
    print(f'\n  Phase 0 complete. Submit Phase 1 jobs next.')


# ═══════════════════════════════════════════════════════════════════════
# Weight zeroing — zero down_proj columns for selected neurons
# ═══════════════════════════════════════════════════════════════════════

class WeightZeroing:
    """Context manager that zeros down_proj weight columns for selected neurons.

    Zeroing down_proj[:, neuron_idx] makes that neuron's contribution to the
    layer output exactly zero, regardless of activation value. This is
    mathematically equivalent to activation zeroing but operates on weights
    directly — connecting the diagnostic experiment to permanent weight edits.

    Args:
        model: loaded VLM
        model_type: backend identifier
        neuron_map: dict {layer_idx: list of neuron indices to ablate}
    """

    def __init__(self, model, model_type, neuron_map):
        self.model = model
        self.model_type = model_type
        self.neuron_map = neuron_map  # {layer_idx: [neuron_indices]}
        self._backups = {}  # {layer_idx: tensor}

    def _get_down_proj(self, layer_idx):
        """Get the down_proj weight tensor for a layer."""
        prefix_map = {
            'llava-hf': 'model.language_model.layers',
            'llava-liuhaotian': 'model.layers',
            'llava-ov': 'model.language_model.layers',
            'internvl': 'language_model.model.layers',
            'qwen2vl': 'model.language_model.layers',
            'llava-llama3': 'model.language_model.layers',
            'idefics2': 'model.text_model.layers',
        }
        prefix = prefix_map.get(self.model_type,
                                'model.language_model.layers')
        # InternVL uses feed_forward.w2 instead of mlp.down_proj
        if self.model_type == 'internvl':
            dotted = f'{prefix}.{layer_idx}.feed_forward.w2'
        else:
            dotted = f'{prefix}.{layer_idx}.mlp.down_proj'
        mod = self.model
        for p in dotted.split('.'):
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        return mod.weight  # shape: (hidden_size, intermediate_size)

    def __enter__(self):
        import torch
        for layer_idx, neuron_indices in self.neuron_map.items():
            if not neuron_indices:
                continue
            w = self._get_down_proj(layer_idx)
            idx = torch.tensor(neuron_indices, dtype=torch.long,
                               device=w.device)
            # Backup the columns we're about to zero
            self._backups[layer_idx] = (idx, w.data[:, idx].clone())
            # Zero the columns — neuron contributes nothing to output
            w.data[:, idx] = 0
        return self

    def __exit__(self, *exc):
        # Restore original weights
        for layer_idx, (idx, backup) in self._backups.items():
            w = self._get_down_proj(layer_idx)
            w.data[:, idx] = backup
        self._backups.clear()


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Evaluate one condition
# ═══════════════════════════════════════════════════════════════════════

def load_model_and_processor(args, device='cuda:0'):
    """Load VLM model and processor onto the given device.

    Returns:
        (model, processor, image_processor)
        image_processor is only non-None for llava-liuhaotian (original LLaVA).
    """
    import torch

    print(f'  Loading model on {device} (type={args.model_type})...')

    if args.model_type in ('llava-liuhaotian', 'liuhaotian'):
        # Original LLaVA — uses llava repo's own loader
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name, device_map=device,
            torch_dtype=torch.float16)
        model.eval()
        print(f'  Model loaded (liuhaotian).')
        return model, tokenizer, image_processor

    elif args.model_type in ('llava-llama3',):
        # LLaVA-Next-LLaMA3 — HF format (LlavaNextForConditionalGeneration)
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        print(f'  Model loaded (llava-llama3 / LlavaNext HF).')
        return model, processor, None

    elif args.model_type in ('llava-hf', 'hf'):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        print(f'  Model loaded (llava-hf).')
        return model, processor, None

    elif args.model_type == 'llava-ov':
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                  AutoProcessor)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
        print(f'  Model loaded (llava-ov).')
        return model, processor, None

    elif args.model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True).to(device).eval()
        processor = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
        print(f'  Model loaded (internvl).')
        return model, processor, None

    elif args.model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
        print(f'  Model loaded (qwen2vl).')
        return model, processor, None

    elif args.model_type == 'idefics2':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        print(f'  Model loaded (idefics2).')
        return model, processor, None

    else:
        raise ValueError(f'Unknown model_type: {args.model_type}')


def run_phase1(args):
    """Evaluate one (category, fraction) condition (ranked ablation)."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from halluc_score_neurons import (
            compute_hallucination_rate, compute_triviaqa_error_rate,
            load_pope_data, check_triviaqa_answer,
        )
    except ImportError:
        print('  ERROR: Phase 1 (ranked ablation) requires halluc_score_neurons.py')
        print('         For equal-count validation, use --ranking random_trials (phases 3-4)')
        return

    condition = args.condition
    fraction = args.fraction
    assert condition is not None, '--condition required for phase 1'
    assert fraction is not None, '--fraction required for phase 1'

    print(f'\n{"="*60}')
    print(f'PHASE 1: {condition} @ {fraction*100:.0f}%')
    print(f'{"="*60}\n')

    # ── Load ranking ──
    ranked_cats_path = os.path.join(args.output_dir,
                                     'ranked_neurons_by_category.json')
    ranking_path = os.path.join(args.output_dir, 'neuron_ranking.json')

    with open(ranked_cats_path) as f:
        ranked_by_cat = json.load(f)
    with open(ranking_path) as f:
        ranking = json.load(f)

    # ── Select neurons to ablate ──
    if condition == 'random':
        # Count-matched random baseline: same number as visual at this fraction
        rng = np.random.RandomState(args.seed)
        n_visual = len(ranked_by_cat.get('visual', []))
        n_to_ablate = max(1, int(n_visual * fraction))
        all_keys = list(ranking.keys())
        selected_keys = list(rng.choice(all_keys, size=n_to_ablate,
                                         replace=False))
    else:
        cat_keys = ranked_by_cat.get(condition, [])
        n_to_ablate = max(1, int(len(cat_keys) * fraction))
        selected_keys = cat_keys[:n_to_ablate]

    print(f'  Ablating {len(selected_keys)} neurons '
          f'({condition} top {fraction*100:.0f}%)')

    # Build neuron_map: {layer_idx: [neuron_indices]}
    neuron_map = defaultdict(list)
    for key in selected_keys:
        info = ranking[key]
        neuron_map[info['layer']].append(info['neuron_idx'])

    layers_affected = len(neuron_map)
    print(f'  Across {layers_affected} layers')

    # ── Load model ──
    device = 'cuda:0'
    model, processor, image_processor = load_model_and_processor(args, device)

    # ── Prepare weight zeroing ──
    ablation = WeightZeroing(
        model, args.model_type, dict(neuron_map))

    # ── Load evaluation data ──
    # Use contrastive POPE if available, else raw POPE
    contrastive_pope = os.path.join(args.output_dir,
                                     'contrastive_pope.jsonl')
    if os.path.isfile(contrastive_pope):
        pope_questions = load_pope_data(contrastive_pope)
    else:
        pope_questions = load_pope_data(args.pope_path)

    # ── Baseline (no ablation) ──
    print(f'\n  Computing baseline...')
    baseline_pope = compute_hallucination_rate(
        model, processor, pope_questions,
        args.pope_img_dir, device, args.model_type)
    print(f'  Baseline POPE: accuracy={baseline_pope["accuracy"]:.4f}, '
          f'halluc_rate={baseline_pope["hallucination_rate"]:.4f}')

    baseline_tqa = None
    tqa_questions = None
    if args.triviaqa_path:
        tqa_contrastive = os.path.join(args.output_dir,
                                        'contrastive_triviaqa.jsonl')
        if os.path.isfile(tqa_contrastive):
            with open(tqa_contrastive) as f:
                tqa_questions = [json.loads(l) for l in f if l.strip()]
            print(f'  Loaded {len(tqa_questions)} contrastive TriviaQA Qs')
        else:
            from halluc_score_neurons import load_triviaqa_data
            tqa_questions = load_triviaqa_data(
                args.triviaqa_path, args.triviaqa_num, seed=args.seed)

        baseline_tqa = compute_triviaqa_error_rate(
            model, processor, tqa_questions, device, args.model_type)
        print(f'  Baseline TriviaQA: error_rate='
              f'{baseline_tqa["error_rate"]:.4f}')

    # ── Ablated evaluation (zero weights, evaluate, restore) ──
    print(f'\n  Zeroing down_proj weights for {len(selected_keys)} neurons...')

    with ablation:
        ablated_pope = compute_hallucination_rate(
            model, processor, pope_questions,
            args.pope_img_dir, device, args.model_type)
        print(f'  Ablated POPE: accuracy={ablated_pope["accuracy"]:.4f}, '
              f'halluc_rate={ablated_pope["hallucination_rate"]:.4f}')

        ablated_tqa = None
        if tqa_questions is not None:
            ablated_tqa = compute_triviaqa_error_rate(
                model, processor, tqa_questions, device, args.model_type)

    # Weights are now restored
    print(f'  Weights restored.')

    delta_pope_acc = ablated_pope['accuracy'] - baseline_pope['accuracy']
    delta_pope_hr = (ablated_pope['hallucination_rate']
                     - baseline_pope['hallucination_rate'])
    print(f'  Δ accuracy={delta_pope_acc:+.4f}, '
          f'Δ halluc_rate={delta_pope_hr:+.4f}')

    if ablated_tqa is not None:
        delta_tqa = (ablated_tqa['error_rate']
                     - baseline_tqa['error_rate'])
        print(f'  Ablated TriviaQA: error_rate='
              f'{ablated_tqa["error_rate"]:.4f} (Δ={delta_tqa:+.4f})')

    # ── Save result ──
    result = {
        'condition': condition,
        'fraction': fraction,
        'n_neurons_ablated': len(selected_keys),
        'n_layers_affected': layers_affected,
        'baseline_pope': baseline_pope,
        'ablated_pope': ablated_pope,
        'delta_pope_accuracy': delta_pope_acc,
        'delta_pope_halluc_rate': delta_pope_hr,
    }
    if baseline_tqa is not None:
        result['baseline_tqa'] = baseline_tqa
    if ablated_tqa is not None:
        result['ablated_tqa'] = ablated_tqa
        result['delta_tqa_error_rate'] = (
            ablated_tqa['error_rate'] - baseline_tqa['error_rate'])

    frac_str = f'{fraction:.2f}'.replace('.', 'p')
    result_path = os.path.join(
        args.output_dir,
        f'ablation_result_{condition}_{frac_str}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Result saved to {result_path}')

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Merge results
# ═══════════════════════════════════════════════════════════════════════

def run_phase2(args):
    """Merge per-condition results into summary + enrichment-compatible output."""
    import glob

    print(f'\n{"="*60}')
    print(f'PHASE 2: Merge results')
    print(f'{"="*60}\n')

    result_files = sorted(glob.glob(
        os.path.join(args.output_dir, 'ablation_result_*.json')))

    if not result_files:
        print('  ERROR: No result files found')
        return

    # ── Parse all results ──
    results = []
    for f_path in result_files:
        with open(f_path) as f:
            results.append(json.load(f))
        print(f'  Loaded {os.path.basename(f_path)}')

    # ── Build summary table ──
    summary = defaultdict(dict)
    for r in results:
        cond = r['condition']
        frac = r['fraction']
        summary[cond][frac] = {
            'n_neurons': r['n_neurons_ablated'],
            'pope_accuracy': r['ablated_pope']['accuracy'],
            'pope_halluc_rate': r['ablated_pope']['hallucination_rate'],
            'delta_pope_accuracy': r['delta_pope_accuracy'],
            'delta_pope_halluc_rate': r['delta_pope_halluc_rate'],
        }
        if 'ablated_tqa' in r:
            summary[cond][frac]['tqa_error_rate'] = (
                r['ablated_tqa']['error_rate'])
            summary[cond][frac]['delta_tqa_error_rate'] = (
                r['delta_tqa_error_rate'])

    # Add baseline from any result file
    baseline = results[0]
    summary['baseline'] = {
        'pope_accuracy': baseline['baseline_pope']['accuracy'],
        'pope_halluc_rate': (
            baseline['baseline_pope']['hallucination_rate']),
    }
    if 'baseline_tqa' in baseline:
        summary['baseline']['tqa_error_rate'] = (
            baseline['baseline_tqa']['error_rate'])

    # ── Save summary ──
    summary_path = os.path.join(args.output_dir,
                                 'ranked_ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(dict(summary), f, indent=2, default=str)
    print(f'\n  Summary saved to {summary_path}')

    # ── Print dose-response table ──
    categories = ['visual', 'text', 'multimodal', 'unknown', 'random']
    fractions = sorted(set(r['fraction'] for r in results))

    print(f'\n{"─"*90}')
    print(f'{"Condition":<14} {"Frac":>6} {"N":>8} '
          f'{"POPE Acc":>9} {"Δ Acc":>8} '
          f'{"TQA Err":>9} {"Δ Err":>8}')
    print(f'{"─"*90}')

    base = summary['baseline']
    print(f'{"baseline":<14} {"":>6} {"":>8} '
          f'{base["pope_accuracy"]:>9.4f} {"":>8}', end='')
    if 'tqa_error_rate' in base:
        print(f' {base["tqa_error_rate"]:>9.4f}', end='')
    print()

    for cond in categories:
        if cond not in summary:
            continue
        for frac in fractions:
            if frac not in summary[cond]:
                continue
            s = summary[cond][frac]
            line = (f'{cond:<14} {frac:>6.2f} {s["n_neurons"]:>8,} '
                    f'{s["pope_accuracy"]:>9.4f} '
                    f'{s["delta_pope_accuracy"]:>+8.4f}')
            if 'tqa_error_rate' in s:
                line += (f' {s["tqa_error_rate"]:>9.4f} '
                         f'{s["delta_tqa_error_rate"]:>+8.4f}')
            print(line)
    print(f'{"─"*90}')

    # ── Generate enrichment-compatible output for step 15 ──
    # For the 100% fraction, compute enrichment-like statistics
    # that step 15 can read
    enrich_path = os.path.join(args.output_dir,
                                'dose_response_for_plots.json')
    with open(enrich_path, 'w') as f:
        json.dump({
            'summary': dict(summary),
            'fractions': fractions,
            'categories': categories,
            'model_name': args.model_name,
        }, f, indent=2, default=str)
    print(f'  Plot data saved to {enrich_path}')

    print(f'\n  Phase 2 complete.')


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Random-sample category ablation (one trial)
# ═══════════════════════════════════════════════════════════════════════

def run_phase3(args):
    """Randomly sample neurons from a category and evaluate ablation effect.

    Unlike phase 1 which uses ranked top-N%, this phase randomly samples
    N neurons from the category. Repeated with different seeds, this tests
    whether the category label alone predicts behavioral impact — no
    ranking, no hallucination signal, purely the taxonomy label.

    Equal-count design: all conditions ablate the same absolute number of
    neurons (based on the smallest of visual/text/multimodal), removing
    the confound that larger categories contribute more network capacity.
    """
    import torch

    condition = args.condition
    fraction = args.fraction
    trial = args.trial
    assert condition is not None, '--condition required for phase 3'
    assert fraction is not None, '--fraction required for phase 3'
    assert trial is not None, '--trial required for phase 3'

    print(f'\n{"="*60}')
    print(f'PHASE 3: Random-sample trial {trial}')
    print(f'  condition={condition}, fraction={fraction*100:.0f}%')
    print(f'{"="*60}\n')

    # ── Load labels to get neurons per category ──
    neurons = load_labels_with_pvalues(
        args.label_dir, args.taxonomy, args.n_layers)

    # ── Group neurons by category ──
    by_cat = defaultdict(list)
    for n in neurons:
        by_cat[n['label']].append(n)

    # ── Compute equal count from min(visual, text, multimodal) ──
    equal_count_base = min(
        len(by_cat.get('visual', [])),
        len(by_cat.get('text', [])),
        len(by_cat.get('multimodal', [])),
    )
    print(f'  Equal-count base (min of vis/text/multi): {equal_count_base}')

    # ── Select neurons to ablate ──
    rng = np.random.RandomState(args.seed + trial)

    if condition == 'unknown':
        # Unknown: ablate ALL unknown neurons (single run, no fraction sweep)
        pool = by_cat.get('unknown', [])
        n_to_ablate = len(pool)
    elif condition == 'random':
        # Count-matched random baseline: same equal count as other categories
        n_to_ablate = max(1, int(equal_count_base * fraction))
        pool = neurons  # all neurons
    else:
        # visual, text, multimodal: equal count based on smallest category
        pool = by_cat.get(condition, [])
        n_to_ablate = max(1, int(equal_count_base * fraction))

    if n_to_ablate > len(pool):
        n_to_ablate = len(pool)

    sampled = list(rng.choice(len(pool), size=n_to_ablate, replace=False))
    selected = [pool[i] for i in sampled]

    print(f'  Sampled {len(selected)} neurons from {condition} '
          f'(pool={len(pool)}, frac={fraction})')

    # ── Build neuron_map ──
    neuron_map = defaultdict(list)
    for n in selected:
        neuron_map[n['layer']].append(n['neuron_idx'])

    print(f'  Across {len(neuron_map)} layers')

    # ── Load model ──
    device = 'cuda:0'
    model, processor, image_processor = load_model_and_processor(args, device)

    # ── Prepare ablation ──
    ablation = WeightZeroing(model, args.model_type, dict(neuron_map))

    # ── Load evaluation data (raw, no contrastive filtering) ──
    pope_questions = load_pope_questions(args.pope_path)
    print(f'  Loaded {len(pope_questions)} POPE questions')

    # Resolve image_token_id for eval interface
    image_token_id = None
    if hasattr(model, 'config') and hasattr(model.config, 'image_token_index'):
        image_token_id = model.config.image_token_index
    elif hasattr(processor, 'tokenizer'):
        _img_tok = getattr(processor.tokenizer, 'convert_tokens_to_ids', None)
        if _img_tok:
            try:
                image_token_id = _img_tok('<image>')
            except Exception:
                pass

    # Helper: compute hallucination rate from POPE results
    def _pope_halluc_rate(pope_result):
        fp = pope_result.get('fp', 0)
        tn = pope_result.get('tn', 0)
        return fp / max(fp + tn, 1)

    # ── Baseline (no ablation) ──
    print(f'\n  Computing baseline...')
    baseline_pope = evaluate_pope(
        model, args.model_type, processor, image_processor, image_token_id,
        pope_questions, args.pope_img_dir, device)
    baseline_pope['hallucination_rate'] = _pope_halluc_rate(baseline_pope)
    print(f'  Baseline POPE: acc={baseline_pope["accuracy"]:.4f}, '
          f'halluc={baseline_pope["hallucination_rate"]:.4f}')

    baseline_tqa = None
    tqa_items = None
    if args.triviaqa_path:
        tqa_items = load_triviaqa(
            args.triviaqa_path, num_questions=args.triviaqa_num, seed=args.seed)
        print(f'  Loaded {len(tqa_items)} TriviaQA questions')
        baseline_tqa = evaluate_triviaqa(
            model, args.model_type, processor, image_processor, image_token_id,
            tqa_items, device, text_only=args.text_only_benchmarks)
        baseline_tqa['error_rate'] = 1.0 - baseline_tqa['accuracy']
        print(f'  Baseline TriviaQA: err={baseline_tqa["error_rate"]:.4f}')

    baseline_chair = None
    if args.chair_ann_path and args.chair_img_dir:
        baseline_chair = evaluate_chair(
            model, args.model_type, processor, image_processor, image_token_id,
            args.chair_ann_path, args.chair_img_dir, device,
            num_images=args.chair_num_images, seed=args.seed)
        print(f'  Baseline CHAIR: chair_i={baseline_chair["chair_i"]:.4f}, '
              f'chair_s={baseline_chair["chair_s"]:.4f}')

    baseline_mmlu = None
    mmlu_items = None
    if args.mmlu_dir:
        mmlu_items = load_mmlu(args.mmlu_dir, num_questions=args.mmlu_num,
                               seed=args.seed)
        print(f'  Loaded {len(mmlu_items)} MMLU questions')
        baseline_mmlu = evaluate_mmlu(
            model, args.model_type, processor, image_processor, image_token_id,
            mmlu_items, device, text_only=args.text_only_benchmarks)
        print(f'  Baseline MMLU: accuracy={baseline_mmlu["accuracy"]:.4f}')

    # ── Ablated evaluation ──
    print(f'\n  Ablating {len(selected)} neurons (trial {trial})...')
    with ablation:
        ablated_pope = evaluate_pope(
            model, args.model_type, processor, image_processor, image_token_id,
            pope_questions, args.pope_img_dir, device)
        ablated_pope['hallucination_rate'] = _pope_halluc_rate(ablated_pope)
        print(f'  Ablated POPE: acc={ablated_pope["accuracy"]:.4f}, '
              f'halluc={ablated_pope["hallucination_rate"]:.4f}')

        ablated_tqa = None
        if tqa_items is not None:
            ablated_tqa = evaluate_triviaqa(
                model, args.model_type, processor, image_processor, image_token_id,
                tqa_items, device, text_only=args.text_only_benchmarks)
            ablated_tqa['error_rate'] = 1.0 - ablated_tqa['accuracy']

        ablated_chair = None
        if baseline_chair is not None:
            ablated_chair = evaluate_chair(
                model, args.model_type, processor, image_processor, image_token_id,
                args.chair_ann_path, args.chair_img_dir, device,
                num_images=args.chair_num_images, seed=args.seed)
            print(f'  Ablated CHAIR: chair_i={ablated_chair["chair_i"]:.4f}')

        ablated_mmlu = None
        if mmlu_items is not None:
            ablated_mmlu = evaluate_mmlu(
                model, args.model_type, processor, image_processor, image_token_id,
                mmlu_items, device, text_only=args.text_only_benchmarks)
            print(f'  Ablated MMLU: accuracy={ablated_mmlu["accuracy"]:.4f}')

    # Weights restored
    delta_pope_hr = (ablated_pope['hallucination_rate']
                     - baseline_pope['hallucination_rate'])
    delta_pope_acc = ablated_pope['accuracy'] - baseline_pope['accuracy']
    print(f'  ΔH={delta_pope_hr:+.4f}, ΔAcc={delta_pope_acc:+.4f}')

    if ablated_tqa is not None:
        delta_tqa = ablated_tqa['error_rate'] - baseline_tqa['error_rate']
        print(f'  ΔTQA={delta_tqa:+.4f}')

    if ablated_chair is not None:
        delta_chair_i = ablated_chair['chair_i'] - baseline_chair['chair_i']
        print(f'  ΔCHAIR_i={delta_chair_i:+.4f}')

    if ablated_mmlu is not None:
        delta_mmlu = ablated_mmlu['accuracy'] - baseline_mmlu['accuracy']
        print(f'  ΔMMLU={delta_mmlu:+.4f}')

    # ── Save result ──
    frac_str = f'{fraction:.2f}'.replace('.', 'p')
    result = {
        'condition': condition,
        'fraction': fraction,
        'trial': trial,
        'seed_used': args.seed + trial,
        'text_only_benchmarks': args.text_only_benchmarks,
        'n_neurons_ablated': len(selected),
        'n_layers_affected': len(neuron_map),
        'baseline_pope': baseline_pope,
        'ablated_pope': ablated_pope,
        'delta_pope_accuracy': delta_pope_acc,
        'delta_pope_halluc_rate': delta_pope_hr,
    }
    if baseline_tqa is not None:
        result['baseline_tqa'] = baseline_tqa
    if ablated_tqa is not None:
        result['ablated_tqa'] = ablated_tqa
        result['delta_tqa_error_rate'] = delta_tqa
    if baseline_chair is not None:
        result['baseline_chair'] = baseline_chair
    if ablated_chair is not None:
        result['ablated_chair'] = ablated_chair
        result['delta_chair_i'] = delta_chair_i
    if baseline_mmlu is not None:
        result['baseline_mmlu'] = baseline_mmlu
    if ablated_mmlu is not None:
        result['ablated_mmlu'] = ablated_mmlu
        result['delta_mmlu_accuracy'] = delta_mmlu

    result_path = os.path.join(
        args.output_dir,
        f'random_trial_{condition}_{frac_str}_t{trial:03d}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Saved to {result_path}')

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Merge random trials + statistical tests
# ═══════════════════════════════════════════════════════════════════════

def run_phase4(args):
    """Merge all random-sample trial results and compute statistics.

    For each (condition, fraction), computes mean ± std of ΔH and ΔTQA,
    then runs Mann-Whitney U tests comparing each category vs random.
    """
    import glob
    from scipy import stats as sp_stats

    print(f'\n{"="*60}')
    print(f'PHASE 4: Merge random trials + statistics')
    print(f'{"="*60}\n')

    trial_files = sorted(glob.glob(
        os.path.join(args.output_dir, 'random_trial_*.json')))

    if not trial_files:
        print('  ERROR: No trial result files found')
        return

    # ── Load all trials ──
    trials = []
    for fp in trial_files:
        with open(fp) as f:
            trials.append(json.load(f))
    print(f'  Loaded {len(trials)} trial results')

    # ── Group by (condition, fraction) ──
    grouped = defaultdict(list)
    for t in trials:
        key = (t['condition'], t['fraction'])
        grouped[key].append(t)

    # ── Compute statistics ──
    categories = ['visual', 'text', 'multimodal', 'unknown', 'random']
    fractions = sorted(set(t['fraction'] for t in trials))

    summary = {}
    for (cond, frac), trial_list in sorted(grouped.items()):
        dh_values = [t['delta_pope_halluc_rate'] for t in trial_list]
        dtqa_values = [t.get('delta_tqa_error_rate', None) for t in trial_list]
        dtqa_values = [v for v in dtqa_values if v is not None]
        dchair_values = [t.get('delta_chair_i', None) for t in trial_list]
        dchair_values = [v for v in dchair_values if v is not None]
        dmmlu_values = [t.get('delta_mmlu_accuracy', None) for t in trial_list]
        dmmlu_values = [v for v in dmmlu_values if v is not None]

        entry = {
            'condition': cond,
            'fraction': frac,
            'n_trials': len(trial_list),
            'n_neurons_ablated': trial_list[0]['n_neurons_ablated'],
            'pope_dH_mean': float(np.mean(dh_values)),
            'pope_dH_std': float(np.std(dh_values, ddof=1)) if len(dh_values) > 1 else 0.0,
            'pope_dH_values': dh_values,
        }
        if dtqa_values:
            entry['tqa_dErr_mean'] = float(np.mean(dtqa_values))
            entry['tqa_dErr_std'] = float(np.std(dtqa_values, ddof=1)) if len(dtqa_values) > 1 else 0.0
            entry['tqa_dErr_values'] = dtqa_values
        if dchair_values:
            entry['chair_dI_mean'] = float(np.mean(dchair_values))
            entry['chair_dI_std'] = float(np.std(dchair_values, ddof=1)) if len(dchair_values) > 1 else 0.0
            entry['chair_dI_values'] = dchair_values
        if dmmlu_values:
            entry['mmlu_dAcc_mean'] = float(np.mean(dmmlu_values))
            entry['mmlu_dAcc_std'] = float(np.std(dmmlu_values, ddof=1)) if len(dmmlu_values) > 1 else 0.0
            entry['mmlu_dAcc_values'] = dmmlu_values

        summary[(cond, frac)] = entry

    # ── Pairwise Mann-Whitney U tests ──
    # For POPE: test if visual ΔH > each other category
    # For CHAIR: test if visual ΔCHAIR > each other category
    # For TriviaQA: test if text ΔTQA > each other category
    # For MMLU: test if text ΔMMLU < each other category (accuracy drops more)
    print(f'\n  ── Summary statistics ──')
    print(f'\n  {"cond":<14} {"frac":>5}  {"n":>3}  '
          f'{"dH_mean":>8} {"dH_std":>7}  '
          f'{"dCHR_mean":>9} {"dCHR_std":>8}  '
          f'{"dTQA_mean":>9} {"dTQA_std":>8}  '
          f'{"dMMLU_mean":>10} {"dMMLU_std":>9}')
    print(f'  {"-"*110}')

    stat_results = []
    for frac in fractions:
        for cond in categories:
            key = (cond, frac)
            if key not in summary:
                continue
            s = summary[key]
            dh_m = s['pope_dH_mean']
            dh_s = s['pope_dH_std']
            dchr_m = s.get('chair_dI_mean', float('nan'))
            dchr_s = s.get('chair_dI_std', float('nan'))
            dtqa_m = s.get('tqa_dErr_mean', float('nan'))
            dtqa_s = s.get('tqa_dErr_std', float('nan'))
            dmmlu_m = s.get('mmlu_dAcc_mean', float('nan'))
            dmmlu_s = s.get('mmlu_dAcc_std', float('nan'))

            print(f'  {cond:<14} {frac:>5.2f}  {s["n_trials"]:>3}  '
                  f'{dh_m:>+8.4f} {dh_s:>7.4f}  '
                  f'{dchr_m:>+9.4f} {dchr_s:>8.4f}  '
                  f'{dtqa_m:>+9.4f} {dtqa_s:>8.4f}  '
                  f'{dmmlu_m:>+10.4f} {dmmlu_s:>9.4f}')

            stat_results.append({
                'condition': cond,
                'fraction': frac,
                'n_trials': s['n_trials'],
                'n_neurons_ablated': s['n_neurons_ablated'],
                'pope_dH_mean': dh_m,
                'pope_dH_std': dh_s,
                'chair_dI_mean': dchr_m,
                'chair_dI_std': dchr_s,
                'tqa_dErr_mean': dtqa_m,
                'tqa_dErr_std': dtqa_s,
                'mmlu_dAcc_mean': dmmlu_m,
                'mmlu_dAcc_std': dmmlu_s,
            })
        print()

    # ── All pairwise comparisons ──
    print(f'\n  ── Pairwise Mann-Whitney U tests ──')
    pairwise_results = []

    for frac in fractions:
        print(f'\n  Fraction: {frac:.0%}')
        print(f'  {"comparison":<30} {"task":<8} {"U":>10} {"p":>12} {"sig":>5} '
              f'{"mean_A":>8} {"mean_B":>8} {"diff":>8}')
        print(f'  {"-"*95}')

        for cond_a in categories:
            for cond_b in categories:
                if cond_a == cond_b:
                    continue
                key_a = (cond_a, frac)
                key_b = (cond_b, frac)
                if key_a not in summary or key_b not in summary:
                    continue

                sa = summary[key_a]
                sb = summary[key_b]

                # POPE ΔH comparison
                dh_a = sa['pope_dH_values']
                dh_b = sb['pope_dH_values']
                if len(dh_a) >= 3 and len(dh_b) >= 3:
                    u_pope, p_pope = sp_stats.mannwhitneyu(
                        dh_a, dh_b, alternative='greater')
                    sig = '***' if p_pope < 0.001 else '**' if p_pope < 0.01 else '*' if p_pope < 0.05 else 'ns'
                    label = f'{cond_a} > {cond_b}'
                    diff = sa['pope_dH_mean'] - sb['pope_dH_mean']
                    print(f'  {label:<30} {"POPE":<8} {u_pope:>10.0f} {p_pope:>12.6f} {sig:>5} '
                          f'{sa["pope_dH_mean"]:>+8.4f} {sb["pope_dH_mean"]:>+8.4f} {diff:>+8.4f}')
                    pairwise_results.append({
                        'fraction': frac,
                        'cond_a': cond_a, 'cond_b': cond_b,
                        'task': 'pope',
                        'U': float(u_pope), 'p': float(p_pope),
                        'mean_a': sa['pope_dH_mean'],
                        'mean_b': sb['pope_dH_mean'],
                    })

                # TriviaQA ΔTQA comparison
                dtqa_a = sa.get('tqa_dErr_values', [])
                dtqa_b = sb.get('tqa_dErr_values', [])
                if len(dtqa_a) >= 3 and len(dtqa_b) >= 3:
                    u_tqa, p_tqa = sp_stats.mannwhitneyu(
                        dtqa_a, dtqa_b, alternative='greater')
                    sig = '***' if p_tqa < 0.001 else '**' if p_tqa < 0.01 else '*' if p_tqa < 0.05 else 'ns'
                    label = f'{cond_a} > {cond_b}'
                    diff = sa.get('tqa_dErr_mean', 0) - sb.get('tqa_dErr_mean', 0)
                    print(f'  {label:<30} {"TQA":<8} {u_tqa:>10.0f} {p_tqa:>12.6f} {sig:>5} '
                          f'{sa.get("tqa_dErr_mean",0):>+8.4f} {sb.get("tqa_dErr_mean",0):>+8.4f} {diff:>+8.4f}')
                    pairwise_results.append({
                        'fraction': frac,
                        'cond_a': cond_a, 'cond_b': cond_b,
                        'task': 'tqa',
                        'U': float(u_tqa), 'p': float(p_tqa),
                        'mean_a': sa.get('tqa_dErr_mean', 0),
                        'mean_b': sb.get('tqa_dErr_mean', 0),
                    })

                # CHAIR ΔCHAIR_i comparison
                dchr_a = sa.get('chair_dI_values', [])
                dchr_b = sb.get('chair_dI_values', [])
                if len(dchr_a) >= 3 and len(dchr_b) >= 3:
                    u_chr, p_chr = sp_stats.mannwhitneyu(
                        dchr_a, dchr_b, alternative='greater')
                    sig = '***' if p_chr < 0.001 else '**' if p_chr < 0.01 else '*' if p_chr < 0.05 else 'ns'
                    label = f'{cond_a} > {cond_b}'
                    diff = sa.get('chair_dI_mean', 0) - sb.get('chair_dI_mean', 0)
                    print(f'  {label:<30} {"CHAIR":<8} {u_chr:>10.0f} {p_chr:>12.6f} {sig:>5} '
                          f'{sa.get("chair_dI_mean",0):>+8.4f} {sb.get("chair_dI_mean",0):>+8.4f} {diff:>+8.4f}')
                    pairwise_results.append({
                        'fraction': frac,
                        'cond_a': cond_a, 'cond_b': cond_b,
                        'task': 'chair',
                        'U': float(u_chr), 'p': float(p_chr),
                        'mean_a': sa.get('chair_dI_mean', 0),
                        'mean_b': sb.get('chair_dI_mean', 0),
                    })

                # MMLU ΔMMLU accuracy comparison (more negative = worse)
                dmmlu_a = sa.get('mmlu_dAcc_values', [])
                dmmlu_b = sb.get('mmlu_dAcc_values', [])
                if len(dmmlu_a) >= 3 and len(dmmlu_b) >= 3:
                    # Test if cond_a drops accuracy MORE (i.e. more negative)
                    u_mmlu, p_mmlu = sp_stats.mannwhitneyu(
                        [-v for v in dmmlu_a], [-v for v in dmmlu_b],
                        alternative='greater')
                    sig = '***' if p_mmlu < 0.001 else '**' if p_mmlu < 0.01 else '*' if p_mmlu < 0.05 else 'ns'
                    label = f'{cond_a} > {cond_b}'
                    diff = sa.get('mmlu_dAcc_mean', 0) - sb.get('mmlu_dAcc_mean', 0)
                    print(f'  {label:<30} {"MMLU":<8} {u_mmlu:>10.0f} {p_mmlu:>12.6f} {sig:>5} '
                          f'{sa.get("mmlu_dAcc_mean",0):>+8.4f} {sb.get("mmlu_dAcc_mean",0):>+8.4f} {diff:>+8.4f}')
                    pairwise_results.append({
                        'fraction': frac,
                        'cond_a': cond_a, 'cond_b': cond_b,
                        'task': 'mmlu',
                        'U': float(u_mmlu), 'p': float(p_mmlu),
                        'mean_a': sa.get('mmlu_dAcc_mean', 0),
                        'mean_b': sb.get('mmlu_dAcc_mean', 0),
                    })

    # ── Key hypothesis tests summary ──
    print(f'\n  ── Key hypothesis tests ──')
    print(f'  {"hypothesis":<45} {"frac":>5} {"p":>12} {"sig":>5}')
    print(f'  {"-"*72}')
    for pw in pairwise_results:
        # Show only the key predictions
        is_key = False
        # Hallucination axis: visual ablation should increase hallucination most
        if pw['task'] == 'pope' and pw['cond_a'] == 'visual':
            is_key = True
        if pw['task'] == 'chair' and pw['cond_a'] == 'visual':
            is_key = True
        # Reasoning axis: text ablation should hurt reasoning most
        if pw['task'] == 'tqa' and pw['cond_a'] == 'text':
            is_key = True
        if pw['task'] == 'mmlu' and pw['cond_a'] == 'text':
            is_key = True
        if is_key:
            hyp = f'{pw["cond_a"]} > {pw["cond_b"]} on {pw["task"].upper()}'
            sig = '***' if pw['p'] < 0.001 else '**' if pw['p'] < 0.01 else '*' if pw['p'] < 0.05 else 'ns'
            print(f'  {hyp:<45} {pw["fraction"]:>5.2f} {pw["p"]:>12.6f} {sig:>5}')

    # ── Save ──
    out_path = os.path.join(args.output_dir, 'random_trial_summary.json')
    with open(out_path, 'w') as f:
        json.dump(stat_results, f, indent=2, default=str)
    print(f'\n  Summary saved to {out_path}')

    # ── Save pairwise comparisons ──
    pw_path = os.path.join(args.output_dir, 'random_trial_pairwise.json')
    with open(pw_path, 'w') as f:
        json.dump(pairwise_results, f, indent=2, default=str)
    print(f'  Pairwise tests saved to {pw_path}')

    # ── Save raw distributions for plotting ──
    dist_path = os.path.join(args.output_dir, 'random_trial_distributions.json')
    dist_data = {}
    for (cond, frac), s in sorted(summary.items()):
        frac_str = f'{frac:.2f}'.replace('.', 'p')
        dist_data[f'{cond}_{frac_str}'] = {
            'pope_dH': s['pope_dH_values'],
            'chair_dI': s.get('chair_dI_values', []),
            'tqa_dErr': s.get('tqa_dErr_values', []),
            'mmlu_dAcc': s.get('mmlu_dAcc_values', []),
        }
    with open(dist_path, 'w') as f:
        json.dump(dist_data, f, indent=2)
    print(f'  Distributions saved to {dist_path}')

    print(f'\n  Phase 4 complete.')


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: SNRF-style 100% ablation with in-memory evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_phase5(args):
    """SNRF-style ablation: zero ALL neurons of each PMBT category, evaluate.

    Supports parallel execution via --phase5_condition:
      - 'all': run baseline + all 6 conditions sequentially (one job)
      - 'baseline': run baseline only
      - 'visual', 'text', etc.: run baseline + that one condition (one job)
      - 'merge': don't run inference, just merge per-condition JSONs

    This is option A for step 22.
    """
    import torch

    condition_arg = getattr(args, 'phase5_condition', 'all')

    # ── Merge mode: combine per-condition results into summary ──
    if condition_arg == 'merge':
        _phase5_merge(args)
        return

    print(f'\n{"="*60}')
    print(f'PHASE 5: SNRF-style 100% ablation (in-memory)')
    print(f'  Condition: {condition_arg}')
    print(f'{"="*60}\n')

    # ── Load labels ──
    neurons = load_labels_with_pvalues(
        args.label_dir, args.taxonomy, args.n_layers)

    by_cat = defaultdict(list)
    for n in neurons:
        by_cat[n['label']].append(n)

    categories = ['visual', 'text', 'multimodal']
    cat_counts = {c: len(by_cat.get(c, [])) for c in categories}
    print(f'  Neuron counts: {cat_counts}')

    # ── Build neuron maps for each condition ──
    def _build_map(neuron_list):
        m = defaultdict(list)
        for n in neuron_list:
            m[n['layer']].append(n['neuron_idx'])
        return dict(m)

    conditions = {}
    for cat in categories:
        conditions[cat] = _build_map(by_cat[cat])

    # Random baselines: matched count, random selection
    rng = np.random.RandomState(42)
    for cat in categories:
        count = len(by_cat[cat])
        sampled_indices = rng.choice(len(neurons), size=min(count, len(neurons)),
                                     replace=False)
        random_neurons = [neurons[i] for i in sampled_indices]
        conditions[f'random_{cat}_count'] = _build_map(random_neurons)
        print(f'  random_{cat}_count: {count} neurons (matched)')

    # ── Load model ──
    device = 'cuda:0'
    model, processor, image_processor = load_model_and_processor(args, device)

    image_token_id = None
    if hasattr(model, 'config') and hasattr(model.config, 'image_token_index'):
        image_token_id = model.config.image_token_index
    elif hasattr(processor, 'tokenizer'):
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
        except Exception:
            pass

    # ── Load evaluation data ──
    benchmarks = {}
    benchmark_filter = getattr(args, 'phase5_benchmark', 'all')

    # POPE
    if args.pope_path and args.pope_img_dir:
        if benchmark_filter in ('all', 'POPE'):
            pope_questions = load_pope_questions(args.pope_path)
            benchmarks['POPE'] = ('pope', pope_questions)
            print(f'  Loaded {len(pope_questions)} POPE questions')

    # MathVerse subtasks
    if args.mathverse_dir:
        subtasks = [s.strip() for s in args.mathverse_subtasks.split(',')]
        for st in subtasks:
            bench_key = f'MV_{st}'
            if benchmark_filter not in ('all', bench_key):
                continue
            items = load_mathverse(args.mathverse_dir, subtask=st)
            if items:
                benchmarks[bench_key] = ('mathverse', items)
                print(f'  Loaded {len(items)} MathVerse {st} questions')
            else:
                print(f'  WARNING: No MathVerse {st} questions found')

    # TriviaQA (text-only — tests pure text reasoning)
    if args.triviaqa_path:
        if benchmark_filter in ('all', 'TriviaQA'):
            tqa_items = load_triviaqa(args.triviaqa_path,
                                      num_questions=args.triviaqa_num,
                                      seed=getattr(args, 'seed', 42))
            if tqa_items:
                benchmarks['TriviaQA'] = ('triviaqa', tqa_items)
                print(f'  Loaded {len(tqa_items)} TriviaQA questions (text-only)')

    if not benchmarks:
        print('  ERROR: No benchmarks loaded. Provide --pope_path, --mathverse_dir, or --triviaqa_path')
        return

    # Apply sample limit for smoke testing
    _limit = getattr(args, 'phase5_limit', 0)
    if _limit > 0:
        print(f'  [limit] Capping each benchmark to {_limit} samples (smoke test)')
        for bname, (btype, bdata) in list(benchmarks.items()):
            if len(bdata) > _limit:
                benchmarks[bname] = (btype, bdata[:_limit])
                print(f'    {bname}: {len(bdata)} → {_limit}')

    # Benchmark suffix for output filenames
    if benchmark_filter != 'all':
        _bench_suffix = f'_{benchmark_filter}'
    else:
        _bench_suffix = ''

    # ── Helper to evaluate all benchmarks ──
    def _evaluate_all():
        results = {}
        for bname, (btype, bdata) in benchmarks.items():
            if btype == 'pope':
                r = evaluate_pope(model, args.model_type, processor,
                                  image_processor, image_token_id,
                                  bdata, args.pope_img_dir, device)
                results[bname] = r
            elif btype == 'mathverse':
                r = evaluate_mathverse(model, args.model_type, processor,
                                       image_processor, image_token_id,
                                       bdata, device)
                results[bname] = r
            elif btype == 'triviaqa':
                r = evaluate_triviaqa(model, args.model_type, processor,
                                      image_processor, image_token_id,
                                      bdata, device, text_only=True)
                results[bname] = r
        return results

    # ── Baseline (always runs) ──
    print(f'\n  ── Baseline (no ablation) ──')
    baseline = _evaluate_all()
    for bname, r in baseline.items():
        print(f'    {bname}: acc={r["accuracy"]:.4f}')

    # ── Determine which conditions to run ──
    all_cond_names = list(categories) + [f'random_{c}_count' for c in categories]
    if condition_arg == 'all':
        targets = all_cond_names
    elif condition_arg == 'baseline':
        targets = []
    else:
        if condition_arg not in conditions:
            print(f'  ERROR: Unknown condition "{condition_arg}"')
            print(f'  Valid: {list(conditions.keys())} or "all" or "baseline" or "merge"')
            return
        targets = [condition_arg]

    # ── Run ablated conditions ──
    all_results = {'baseline': baseline}
    for cond in targets:
        nmap = conditions[cond]
        n_neurons = sum(len(v) for v in nmap.values())
        print(f'\n  ── {cond} ({n_neurons} neurons) ──')

        ablation = WeightZeroing(model, args.model_type, nmap)
        with ablation:
            cond_results = _evaluate_all()

        all_results[cond] = cond_results
        for bname, r in cond_results.items():
            base_acc = baseline[bname]['accuracy']
            delta = r['accuracy'] - base_acc
            print(f'    {bname}: acc={r["accuracy"]:.4f} (Δ={delta:+.4f})')

    # ── Save per-condition result ──
    os.makedirs(args.output_dir, exist_ok=True)
    bench_names = list(benchmarks.keys())

    if condition_arg == 'all':
        save_name = f'phase5_snrf_ablation{_bench_suffix}.json'
    elif condition_arg == 'baseline':
        save_name = f'phase5_baseline{_bench_suffix}.json'
    else:
        save_name = f'phase5_{condition_arg}{_bench_suffix}.json'

    save_data = {
        'model_type': args.model_type,
        'model_name': args.model_name,
        'neuron_counts': cat_counts,
        'benchmarks': bench_names,
        'condition': condition_arg,
    }
    for t in ['baseline'] + targets:
        save_data[t] = {}
        for bn in bench_names:
            if bn in all_results[t]:
                save_data[t][bn] = all_results[t][bn]

    out_path = os.path.join(args.output_dir, save_name)
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\n  Saved to {out_path}')

    # ── Print + save summary if running all ──
    if condition_arg == 'all':
        _phase5_print_and_save(args, all_results, categories, bench_names)

    # ── Try to merge if all per-condition files exist ──
    _phase5_try_merge(args, categories, bench_names)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'\n  Phase 5 complete.')


def _phase5_try_merge(args, categories=None, bench_names=None):
    """If all per-condition JSONs exist, merge into summary."""
    if categories is None:
        categories = ['visual', 'text', 'multimodal']
    all_conds = list(categories) + [f'random_{c}_count' for c in categories]
    all_files = ['phase5_baseline.json'] + [f'phase5_{c}.json' for c in all_conds]

    missing = [f for f in all_files
               if not os.path.exists(os.path.join(args.output_dir, f))]
    if missing:
        print(f'\n  [{len(missing)} per-condition files still missing — skipping merge]')
        return

    print(f'\n  All per-condition files found — merging...')
    _phase5_merge(args)


def _phase5_merge(args):
    """Merge per-condition (and optionally per-benchmark) JSON files into combined summary."""
    categories = ['visual', 'text', 'multimodal']
    all_conds = list(categories) + [f'random_{c}_count' for c in categories]
    all_targets = ['baseline'] + all_conds

    all_results = {}
    bench_names_set = set()

    for cond in all_targets:
        if cond == 'baseline':
            base_fname = 'phase5_baseline'
        else:
            base_fname = f'phase5_{cond}'

        # Try non-suffixed file first
        fpath = os.path.join(args.output_dir, f'{base_fname}.json')
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            bench_names_set.update(data.get('benchmarks', []))
            if 'baseline' in data and 'baseline' not in all_results:
                all_results['baseline'] = data['baseline']
            if cond in data:
                all_results[cond] = data[cond]
            continue

        # Try benchmark-suffixed files (e.g. phase5_baseline_POPE.json)
        import glob
        pattern = os.path.join(args.output_dir, f'{base_fname}_*.json')
        bench_files = glob.glob(pattern)
        if bench_files:
            cond_data = {}
            for bf in bench_files:
                with open(bf) as f:
                    data = json.load(f)
                bench_names_set.update(data.get('benchmarks', []))
                # Merge benchmark results from this file
                if 'baseline' in data:
                    if 'baseline' not in all_results:
                        all_results['baseline'] = {}
                    all_results['baseline'].update(data['baseline'])
                if cond in data:
                    cond_data.update(data[cond])
            if cond_data:
                all_results[cond] = cond_data
        else:
            print(f'  WARNING: No files found for {cond}')

    bench_names = sorted(bench_names_set)
    if not bench_names:
        print('  ERROR: No benchmark data found in any per-condition file')
        return

    _phase5_print_and_save(args, all_results, categories, bench_names)

    # Also save combined JSON
    combined = {
        'model_type': getattr(args, 'model_type', ''),
        'model_name': getattr(args, 'model_name', ''),
        'benchmarks': bench_names,
    }
    for t in all_targets:
        if t in all_results:
            combined[t] = all_results[t]

    out_path = os.path.join(args.output_dir, 'phase5_snrf_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'\n  Combined results saved to {out_path}')


def _phase5_print_and_save(args, all_results, categories, bench_names):
    """Print summary tables and save CSV."""
    short_names = {
        'baseline': 'base', 'visual': 'vis', 'text': 'tex', 'multimodal': 'mul',
        'random_visual_count': 'rVis', 'random_text_count': 'rTex',
        'random_multimodal_count': 'rMul'
    }
    all_targets = ['baseline'] + list(categories) + [f'random_{c}_count' for c in categories]

    print(f'\n\n{"="*60}')
    print(f'RESULTS TABLE (accuracy %)')
    print(f'{"="*60}')
    header = f'{"":>6}'
    for bn in bench_names:
        header += f'{bn:>12}'
    print(header)
    print('-' * (6 + 12 * len(bench_names)))

    for t in all_targets:
        if t not in all_results:
            continue
        row = f'{short_names.get(t, t):>6}'
        for bn in bench_names:
            if bn in all_results[t]:
                row += f'{all_results[t][bn]["accuracy"]*100:>12.1f}'
            else:
                row += f'{"—":>12}'
        print(row)

    # Delta vs random
    print(f'\n{"="*60}')
    print(f'DELTA vs RANDOM (category - matched random) in accuracy %')
    print(f'{"="*60}')
    print(f'{"":>12}' + ''.join(f'{bn:>12}' for bn in bench_names))
    print('-' * (12 + 12 * len(bench_names)))

    for cat in categories:
        rand_key = f'random_{cat}_count'
        row = f'{cat:>12}'
        for bn in bench_names:
            cat_acc = all_results.get(cat, {}).get(bn, {}).get('accuracy')
            rand_acc = all_results.get(rand_key, {}).get(bn, {}).get('accuracy')
            if cat_acc is not None and rand_acc is not None:
                delta = (cat_acc - rand_acc) * 100
                row += f'{delta:>+12.1f}'
            else:
                row += f'{"—":>12}'
        print(row)

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'phase5_results.csv')
    rows = []
    for t in all_targets:
        if t not in all_results:
            continue
        row_data = {'target': short_names.get(t, t)}
        for bn in bench_names:
            if bn in all_results[t]:
                row_data[bn] = round(all_results[t][bn]['accuracy'] * 100, 1)
            else:
                row_data[bn] = None
        rows.append(row_data)
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'  CSV saved to {csv_path}')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Normalize qwen25vl-7b / qwen25vl-3b → qwen2vl: same architecture
    if args.model_type in ('qwen25vl-7b', 'qwen25vl-3b'):
        args.model_type = 'qwen2vl'

    if args.phase == 0:
        run_phase0(args)
    elif args.phase == 1:
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)
    elif args.phase == 3:
        run_phase3(args)
    elif args.phase == 4:
        run_phase4(args)
    elif args.phase == 5:
        run_phase5(args)


if __name__ == '__main__':
    main()