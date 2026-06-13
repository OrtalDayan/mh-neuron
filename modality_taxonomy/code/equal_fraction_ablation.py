#!/usr/bin/env python3
"""Step 5b — Ranked Fraction Ablation: Dose-response taxonomy validation.

Ranks neurons within each modality category by a combined score
(CETT-diff × classification confidence) and ablates top-N% fractions
by zeroing their down_proj weight columns, measuring the effect on
POPE, CHAIR, TriviaQA and MMLU benchmarks. Weight zeroing is
mathematically equivalent to activation zeroing but operates directly
on model weights, connecting the diagnostic experiment to permanent
weight editing interventions.

Equal-fraction design (phases 3-4): each category ablates f% of its
own neurons.  Because category sizes differ (e.g., visual ~200 vs
text ~3000), the absolute neuron count differs per condition.  Each
condition gets its own matched-count random baseline (random_visual,
random_text, random_multimodal), sampled from ALL neurons.  The
comparison is always "category ablation vs same-count random".
Validation signal: the *pattern* of delta-vs-random across benchmarks
(crossover interaction), not the absolute magnitude.

Three phases:
  Phase 1 (N GPUs): Random-sample equal-fraction trials (30 seeds per condition × fraction)
  Phase 2 (CPU):    Merge random trials → statistics + Mann-Whitney pairwise tests
  Phase 3 (1 GPU):  SNRF-style 100% ablation — zero ALL neurons per PMBT category

Usage:
  # Phase 1: one random trial (equal-fraction)
  python equal_fraction_ablation.py --phase 1 --condition visual --fraction 0.10 --trial 0 ...

  # Phase 2: merge trials + statistics
  python equal_fraction_ablation.py --phase 2 ...

  # Phase 3: SNRF-style 100% ablation
  python equal_fraction_ablation.py --phase 3 --phase3_condition visual ...
"""

import argparse
import json
import os
import re
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


def load_pope_all_protocols(pope_dir):
    """Load POPE questions from all 3 protocol files in a directory.

    Looks for coco_pope_random.json, coco_pope_popular.json,
    coco_pope_adversarial.json. Tags each question with 'protocol'.

    Returns dict: {protocol_name: [questions]}
    """
    protocols = {}
    for protocol in ['random', 'popular', 'adversarial']:
        fname = f'coco_pope_{protocol}.json'
        fpath = os.path.join(pope_dir, fname)
        if os.path.isfile(fpath):
            qs = load_pope_questions(fpath)
            for q in qs:
                q['protocol'] = protocol
            protocols[protocol] = qs
            print(f'  Loaded {len(qs)} POPE {protocol} questions')
        else:
            print(f'  WARNING: {fpath} not found — skipping {protocol}')
    return protocols


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

        # VLMEvalKit sends the bare POPE question without any suffix.
        # Do NOT append "Answer the question using a single word or phrase."
        # as that is a LLaVA-specific convention not used by VLMEvalKit/SNRF.
        pope_question = question_text

        answer = generate_answer(model, model_type, tokenizer_or_processor,
                                 image_processor, image_token_id,
                                 img, pope_question, device,
                                 max_new_tokens=10)
        answer_lower = answer.strip().lower()
        gt_yes = gt_label == 'yes'

        # VLMEvalKit YOrN_Extraction: remove punctuation, split into words,
        # check word-level membership (not substring).
        # If both 'yes' and 'no' appear, or neither appears → Unknown (wrong).
        words = re.sub(r'[^\w\s]', ' ', answer_lower).split()
        has_yes = 'yes' in words
        has_no = 'no' in words

        if has_yes and not has_no:
            pred_yes = True
        elif has_no and not has_yes:
            pred_yes = False
        else:
            # Both present or neither → Unknown → count as wrong
            pred_yes = not gt_yes  # ensures it's marked incorrect

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


def evaluate_pope_all_protocols(model, model_type, tokenizer_or_processor,
                                 image_processor, image_token_id,
                                 pope_protocols, img_dir, device):
    """Evaluate POPE across all protocols, return per-protocol + average.

    Args:
        pope_protocols: dict {protocol_name: [questions]} from load_pope_all_protocols

    Returns dict with keys: per_protocol (dict of results), average (averaged metrics).
    """
    per_protocol = {}
    accuracies = []

    for protocol, questions in pope_protocols.items():
        result = evaluate_pope(
            model, model_type, tokenizer_or_processor, image_processor,
            image_token_id, questions, img_dir, device)
        per_protocol[protocol] = result
        accuracies.append(result['accuracy'])
        print(f'    POPE {protocol}: acc={result["accuracy"]:.4f}, '
              f'f1={result["f1"]:.4f}, yes_ratio={result["yes_ratio"]:.4f}')

    avg_accuracy = round(sum(accuracies) / max(len(accuracies), 1), 4)
    print(f'    POPE average: acc={avg_accuracy:.4f}')

    return {
        'per_protocol': per_protocol,
        'average_accuracy': avg_accuracy,
        'accuracy': avg_accuracy,  # for backward compatibility
        'n_protocols': len(per_protocol),
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

def load_mathverse(data_dir, subtask='Text_Dominant', mcq_only=True):
    """Load MathVerse questions from prepare_mathverse_td.py output.

    Args:
        data_dir: directory containing questions.json and images/
        subtask: 'Text_Dominant', 'Vision_Only', or 'Vision_Dominant'
        mcq_only: If True, skip questions with non-letter answers (17/436).
            These require GPT-based extraction which we can't do in ablation.

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
    skipped_non_mcq = 0
    for item in items:
        # Filter by subtask if single file
        if 'problem_version' in item:
            version = item['problem_version'].replace(' ', '_')
            if version != subtask:
                continue

        img_path = os.path.join(img_dir, item.get('image', ''))
        if not os.path.isfile(img_path):
            continue

        ans = str(item.get('answer', '')).strip()

        # Skip non-letter answers if mcq_only (e.g. "22.3", "f(x)=...")
        if mcq_only and ans not in ('A', 'B', 'C', 'D'):
            skipped_non_mcq += 1
            continue

        result.append({
            'question': item.get('question', ''),
            'answer': ans,
            'image_path': img_path,
            'question_type': item.get('question_type', 'multi-choice'),
        })
    if skipped_non_mcq > 0:
        print(f'  MathVerse {subtask}: skipped {skipped_non_mcq} non-MCQ questions, kept {len(result)}')
    return result


def _extract_answer_letter(pred):
    """Extract answer letter (A-D) from model prediction.

    Uses explicit patterns first, then falls back to VLMEvalKit's
    can_infer_option logic: accept only if exactly ONE choice letter
    appears as a standalone word in the last 5 words of the response.
    """
    import re
    s = str(pred).strip()
    if not s:
        return None

    # ── Explicit patterns (high confidence) ──

    # "The correct option letter is C."
    m = re.search(r'correct.*?(?:option|answer).*?letter.*?([A-D])', s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # "The answer is C" or "Answer: C"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-D])\b', s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # Starts with letter followed by punctuation or end: "B", "B: 54°", "B. 25°"
    m = re.match(r'^([A-D])(?:\s*[:.)]|$)', s)
    if m and len(s) < 20: return m.group(1).upper()
    # "option C" / "choice C"
    m = re.search(r'(?:option|choice)\s+([A-D])\b', s, re.IGNORECASE)
    if m: return m.group(1).upper()

    # ── VLMEvalKit can_infer_option fallback ──
    # Replace punctuation with spaces, split into words
    answer_mod = s
    for c in '.()[],:;!*#{}':
        answer_mod = answer_mod.replace(c, ' ')
    splits = [x.strip() for x in answer_mod.split() if x.strip()]

    # Count how many choice letters (A-D) appear as standalone words
    choices = ('A', 'B', 'C', 'D')
    found = [ch for ch in choices if ch in splits]

    if len(found) == 1:
        ch = found[0]
        idx = splits.index(ch)
        # Accept only if the letter is in the last 5 words
        if idx >= len(splits) - 5:
            return ch

    return None


def evaluate_mathverse(model, model_type, tokenizer_or_processor, image_processor,
                       image_token_id, questions, device, max_new_tokens=256,
                       mcq_suffix=True):
    """Run MathVerse evaluation: multi-choice math, measure accuracy.

    Args:
        mcq_suffix: If True, append VLMEvalKit-style MCQ suffix to prompt
            ("Please select the correct answer from the options above.")
            to match BRV evaluation. If False, send raw question only
            (used for ablation evaluation).

    Returns dict with accuracy, n_questions, n_parsed, predictions.
        predictions: list of {question, answer, prediction, pred_letter, gt_letter}
        for downstream GPT scoring (Run 1 evaluation).
    """
    correct = 0
    total = 0
    n_parsed = 0
    predictions = []

    _suffix = '\nPlease select the correct answer from the options above.' if mcq_suffix else ''

    for q in tqdm(questions, desc='MathVerse eval'):
        try:
            img = Image.open(q['image_path']).convert('RGB')
        except Exception:
            continue

        answer = generate_answer(model, model_type, tokenizer_or_processor,
                                 image_processor, image_token_id,
                                 img, q['question'] + _suffix,
                                 device, max_new_tokens=max_new_tokens)

        pred_letter = _extract_answer_letter(answer)
        gt_letter = _extract_answer_letter(q['answer'])

        total += 1
        if pred_letter is not None and gt_letter is not None:
            n_parsed += 1
            if pred_letter == gt_letter:
                correct += 1

        # Save raw prediction for GPT scoring later
        predictions.append({
            'question': q['question'],
            'answer': q['answer'],
            'prediction': answer,
            'pred_letter': pred_letter,
            'gt_letter': gt_letter,
        })

    accuracy = correct / max(total, 1)
    return {
        'accuracy': round(accuracy, 4),
        'n_questions': total,
        'n_parsed': n_parsed,
        'predictions': predictions,
    }


# ═══════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='D-ranked ablation sweep with fraction-based neuron selection')

    # Phase selection
    p.add_argument('--phase', type=int, required=True, choices=[1, 2],
                   help='1=D-ranked sweep ablation (1 GPU), '
                        '2=merge results + statistics (CPU only)')

    # Model
    p.add_argument('--model_type', type=str, required=True,
                   help='Backend: llava-ov | internvl | qwen2vl | qwen25vl-7b | qwen25vl-3b | idefics2 | llava-hf | llava-llama3')
    p.add_argument('--model_path', type=str, default=None,
                   help='Path to pretrained model (required for phase 1)')
    p.add_argument('--model_name', type=str, default=None,
                   help='Human-readable model name for outputs')
    p.add_argument('--n_layers', type=int, required=True)

    # Labels
    p.add_argument('--label_dir', type=str, default=None,
                   help='Directory with PMBT neuron labels')
    p.add_argument('--taxonomy', type=str, default='pmbt')
    p.add_argument('--hook_point', type=str, default='gate',
                   choices=['gate', 'gate_up', 'attn'],
                   help='Hook point: gate (FFN gate), gate_up (FFN gate*up), attn (o_proj input)')

    # Data paths
    p.add_argument('--pope_path', type=str, default=None)
    p.add_argument('--pope_img_dir', type=str, default=None)
    p.add_argument('--pope_dir', type=str, default=None,
                   help='Directory with all 3 POPE protocol files')
    p.add_argument('--triviaqa_path', type=str, default=None)
    p.add_argument('--triviaqa_num', type=int, default=2000)
    p.add_argument('--mathverse_dir', type=str, default=None,
                   help='Directory with MathVerse questions.json + images/')
    p.add_argument('--mathverse_subtasks', type=str,
                   default='Text_Dominant,Vision_Only',
                   help='Comma-separated MathVerse subtasks to evaluate')

    # Ranking method
    p.add_argument('--ranking', type=str, default='D_x_norm',
                   choices=['D', 'norm', 'D_x_norm', 'D_then_norm'],
                   help='Neuron ranking: '
                        'D = abs(rate_diff) only (strongest modality bias first), '
                        'norm = output projection L2 norm only (largest residual impact first), '
                        'D_x_norm = abs(rate_diff) × norm (combined), '
                        'D_then_norm = abs(rate_diff) primary, norm tiebreaker')

    # Sweep settings
    p.add_argument('--sweep_fracs', type=str,
                   default='0.01,0.05,0.1,0.25,0.5,1.0',
                   help='Comma-separated fractions of each category to ablate')
    p.add_argument('--n_random_trials', type=int, default=5,
                   help='Number of random baseline trials per fraction')
    p.add_argument('--categories', type=str, default='visual,text,multimodal',
                   help='Comma-separated categories to ablate')
    p.add_argument('--category', type=str, default=None,
                   choices=['visual', 'text', 'multimodal'],
                   help='Run a single category (for parallel job submission). '
                        'Overrides --categories when set.')
    p.add_argument('--benchmark', type=str, default=None,
                   help='Run a single benchmark (for parallel job submission). '
                        'E.g. POPE, MV_Text_Dominant, MV_Vision_Only, TriviaQA. '
                        'When set, only that benchmark is loaded and evaluated.')
    p.add_argument('--fraction', type=float, default=None,
                   help='Run a single fraction (for parallel job submission). '
                        'E.g. 0.05. When set, only that fraction is evaluated.')
    p.add_argument('--trial_idx', type=str, default=None,
                   help='Run a single trial (for parallel job submission). '
                        '"ranked" = deterministic ranked ablation, '
                        '"0","1",... = random baseline trial index. '
                        'When set, runs exactly one evaluation per job.')
    p.add_argument('--sample_limit', type=int, default=0,
                   help='Cap samples per benchmark (0=no limit, e.g. 10 for smoke test)')
    p.add_argument('--baseline_only', action='store_true',
                   help='Run baseline evaluation only (no ablation). '
                        'Useful for comparing against SNRF/BRV paper baselines.')

    # Output
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()

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
# Weight zeroing context manager
# ═══════════════════════════════════════════════════════════════════════

class WeightZeroing:
    """Context manager that zeros specific neuron weights and restores on exit.

    For FFN neurons (gate/gate_up hook): zeros columns in down_proj.
      - Neuron k contributes via down_proj[:, k]; zeroing this column
        removes that neuron's contribution to the residual stream.

    For attention neurons (attn hook): zeros columns in o_proj.
      - The o_proj input dimension corresponds to concatenated head outputs.
        Zeroing o_proj[:, k] removes that attention dimension's contribution.
    """

    def __init__(self, model, model_type, neuron_map, hook_point='gate'):
        """
        Args:
            model: The VLM model
            model_type: Model type string
            neuron_map: Dict {layer_idx: [neuron_indices]}
            hook_point: 'gate', 'gate_up', or 'attn'
        """
        import torch
        self.model = model
        self.neuron_map = neuron_map
        self.hook_point = hook_point
        self.saved_weights = {}
        self.weight_params = {}

        # For attn hook: determine num_heads from max neuron_idx across all layers
        self.num_heads = None
        self.head_dim = None
        if hook_point == 'attn':
            max_idx = max(max(idxs) for idxs in neuron_map.values() if idxs)
            self.num_heads = max_idx + 1

        # Target weight based on hook point
        if hook_point in ('gate', 'gate_up'):
            target_suffix = '.mlp.down_proj.weight'
        elif hook_point == 'attn':
            target_suffix = '.self_attn.o_proj.weight'
        else:
            raise ValueError(f'Unknown hook_point: {hook_point}')

        # Build mapping: layer_idx -> parameter
        for name, param in model.named_parameters():
            for layer_idx in neuron_map:
                pattern = f'layers.{layer_idx}{target_suffix}'
                if pattern in name:
                    self.weight_params[layer_idx] = param
                    # Compute head_dim from first found param
                    if hook_point == 'attn' and self.head_dim is None:
                        self.head_dim = param.shape[1] // self.num_heads
                        print(f'  [WeightZeroing] attn: {self.num_heads} heads, '
                              f'head_dim={self.head_dim}, o_proj shape={list(param.shape)}')
                    break

        missing = set(neuron_map.keys()) - set(self.weight_params.keys())
        if missing:
            # Try to diagnose
            all_names = [n for n, _ in model.named_parameters()
                         if target_suffix.lstrip('.') in n]
            raise RuntimeError(
                f'Could not find weight params for layers: {sorted(missing)}. '
                f'Found {len(all_names)} matching params: {all_names[:5]}...')

    def _expand_head_indices(self, idx):
        """Expand head indices [0,5,12] to column ranges in o_proj."""
        expanded = []
        for head_idx in idx:
            start = head_idx * self.head_dim
            expanded.extend(range(start, start + self.head_dim))
        return expanded

    def __enter__(self):
        for layer_idx, indices in self.neuron_map.items():
            param = self.weight_params[layer_idx]
            idx = indices if isinstance(indices, list) else list(indices)

            # For attention hook: expand head indices to o_proj column ranges
            if self.hook_point == 'attn':
                idx = self._expand_head_indices(idx)

            # Save original column values
            self.saved_weights[layer_idx] = param.data[:, idx].clone()
            # Zero the columns — removes these neurons' contribution
            param.data[:, idx] = 0
        return self

    def __exit__(self, *args):
        for layer_idx, indices in self.neuron_map.items():
            param = self.weight_params[layer_idx]
            idx = indices if isinstance(indices, list) else list(indices)

            if self.hook_point == 'attn':
                idx = self._expand_head_indices(idx)

            # Restore original weights
            param.data[:, idx] = self.saved_weights[layer_idx]
        self.saved_weights.clear()


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# Model loading helper
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
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name, device_map=device,
            torch_dtype=torch.float16)
        model.eval()
        return model, tokenizer, image_processor

    elif args.model_type in ('llava-llama3',):
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        return model, processor, None

    elif args.model_type in ('llava-hf', 'hf'):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        return model, processor, None

    elif args.model_type == 'llava-ov':
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                  AutoProcessor)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
        return model, processor, None

    elif args.model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True).to(device).eval()
        processor = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
        return model, processor, None

    elif args.model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).to(device).eval()
        return model, processor, None

    elif args.model_type == 'idefics2':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        return model, processor, None

    else:
        raise ValueError(f'Unknown model_type: {args.model_type}')

# ═══════════════════════════════════════════════════════════════════════
# Phase 4: D-ranked sweep ablation
# ═══════════════════════════════════════════════════════════════════════

def precompute_output_norms(model, model_type, hook_point, n_layers):
    """Precompute per-unit output-projection L2 norms.

    For gate/gate_up: L2 norm of each down_proj column → shape (n_neurons,).
        Measures how much neuron n's activation moves the residual stream.
    For attn: Frobenius norm of each head's block in o_proj → shape (n_heads,).

    Returns:
        dict: layer_index → 1-D numpy array of norms.
    """
    import torch

    # Determine target weight suffix
    if hook_point in ('gate', 'gate_up'):
        target_suffix = '.mlp.down_proj.weight'
    elif hook_point == 'attn':
        target_suffix = '.self_attn.o_proj.weight'
    else:
        raise ValueError(f'Unknown hook_point: {hook_point}')

    # Build layer_idx → parameter mapping
    layer_params = {}
    for name, param in model.named_parameters():
        for l in range(n_layers):
            if f'layers.{l}{target_suffix}' in name:
                layer_params[l] = param
                break

    norms_by_layer = {}
    with torch.no_grad():
        for l in range(n_layers):
            if l not in layer_params:
                continue
            w = layer_params[l].float()

            if hook_point in ('gate', 'gate_up'):
                # w shape: (hidden_dim, intermediate_dim)
                # Column n = neuron n → L2 norm per column
                norms_by_layer[l] = torch.norm(w, dim=0).cpu().numpy()
            elif hook_point == 'attn':
                # w shape: (hidden_dim, hidden_dim)
                # Need to infer n_heads from the config
                hidden_dim = w.shape[0]
                # Try common head counts
                for nh in [32, 28, 16, 40, 64]:
                    if hidden_dim % nh == 0:
                        n_heads = nh
                        break
                else:
                    n_heads = 32  # fallback
                head_dim = hidden_dim // n_heads
                w_heads = w.reshape(hidden_dim, n_heads, head_dim)
                norms_by_layer[l] = torch.norm(w_heads, dim=(0, 2)).cpu().numpy()

    sample_l = next(iter(norms_by_layer))
    sample_n = norms_by_layer[sample_l]
    print(f'  Output norms: layer {sample_l} range [{sample_n.min():.4f}, '
          f'{sample_n.max():.4f}], mean {sample_n.mean():.4f}, '
          f'shape {sample_n.shape}')
    return norms_by_layer


def rank_neurons_for_ablation(neurons, ranking, norms_by_layer, category):
    """Rank neurons within a category by the chosen strategy.

    For visual neurons: D > 0, rank by largest D first.
    For text neurons: D < 0, rank by most negative D first (largest |D|).
    For multimodal neurons: D ≈ 0, rank by smallest p-value first
        (most statistically significant multimodal pattern).

    Args:
        neurons: list of dicts with 'layer', 'neuron_idx', 'label',
                 'rate_diff' (D), 'p_value'
        ranking: one of 'D', 'norm', 'D_x_norm', 'D_then_norm'
        norms_by_layer: dict from precompute_output_norms()
        category: 'visual', 'text', or 'multimodal'

    Returns:
        list of neuron dicts, sorted best-first for ablation.
    """
    # Attach norm to each neuron
    for n in neurons:
        layer_norms = norms_by_layer.get(n['layer'])
        if layer_norms is not None and n['neuron_idx'] < len(layer_norms):
            n['norm'] = float(layer_norms[n['neuron_idx']])
        else:
            n['norm'] = 0.0

    if ranking == 'D':
        # Strongest modality bias first
        # Visual: largest D. Text: largest |D| (most negative). Multi: smallest p.
        if category == 'multimodal':
            return sorted(neurons, key=lambda n: n['p_value'])
        else:
            return sorted(neurons, key=lambda n: abs(n['rate_diff']), reverse=True)

    elif ranking == 'norm':
        # Largest residual-stream impact first
        return sorted(neurons, key=lambda n: n['norm'], reverse=True)

    elif ranking == 'D_x_norm':
        # Combined: modality bias × residual impact
        if category == 'multimodal':
            # For multimodal: use (1 - p_value) × norm (most significant + highest impact)
            return sorted(neurons,
                          key=lambda n: (1.0 - n['p_value']) * n['norm'],
                          reverse=True)
        else:
            return sorted(neurons,
                          key=lambda n: abs(n['rate_diff']) * n['norm'],
                          reverse=True)

    elif ranking == 'D_then_norm':
        # D primary, norm tiebreaker
        if category == 'multimodal':
            # p_value primary (ascending), norm tiebreaker (descending)
            return sorted(neurons,
                          key=lambda n: (n['p_value'], -n['norm']))
        else:
            # abs(D) primary (descending), norm tiebreaker (descending)
            return sorted(neurons,
                          key=lambda n: (-abs(n['rate_diff']), -n['norm']))

    else:
        raise ValueError(f'Unknown ranking: {ranking}')




# ═══════════════════════════════════════════════════════════════════════
# Phase 1: D-ranked sweep ablation (1 GPU)
# ═══════════════════════════════════════════════════════════════════════

def run_phase1(args):
    """D-ranked sweep ablation with fraction-based neuron selection.

    For each modality category, ranks neurons by the chosen strategy
    (D, norm, D×norm, or D-then-norm), then ablates top-f% for each
    fraction in the sweep. For each fraction, also runs n_random_trials
    random baselines (same count drawn from ALL neurons) for comparison.

    Equal-fraction design: each category ablates f% of its OWN neurons.
    Because category sizes differ (e.g., visual 151K vs multimodal 31K),
    the absolute count differs per condition. Each condition gets its own
    matched-count random baseline drawn from ALL neurons.

    Evaluates on MathVerse Text_Dominant + Vision_Only (default).
    """
    import torch

    fracs = [float(f) for f in args.sweep_fracs.split(',')]
    if args.category:
        categories = [args.category]
    else:
        categories = [c.strip() for c in args.categories.split(',')]
    n_random = args.n_random_trials

    print(f'\n{"="*60}')
    print(f'PHASE 1: D-ranked sweep ablation')
    print(f'  Ranking:        {args.ranking}')
    print(f'  Categories:     {categories}')
    print(f'  Fractions:      {fracs}')
    print(f'  Random trials:  {n_random} per fraction')
    print(f'{"="*60}\n')

    # ── Load labels (skip for baseline-only) ──
    neurons = []
    by_cat = defaultdict(list)
    if not args.baseline_only:
        neurons = load_labels_with_pvalues(
            args.label_dir, args.taxonomy, args.n_layers)

        for n in neurons:
            by_cat[n['label']].append(n)

        for cat in categories:
            print(f'  {cat}: {len(by_cat.get(cat, []))} neurons')
    else:
        print(f'  [baseline-only] Skipping label loading')

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

    # ── Precompute output projection norms (skip for baseline-only) ──
    norms_by_layer = {}
    if not args.baseline_only:
        print(f'\n  Precomputing output projection norms...')
        norms_by_layer = precompute_output_norms(
            model, args.model_type, args.hook_point, args.n_layers)

    # ── Load benchmarks ──
    _bench_filter = args.benchmark  # None = all, or 'POPE', 'MV_Text_Dominant', etc.
    benchmarks = {}
    if args.mathverse_dir:
        subtasks = [s.strip() for s in args.mathverse_subtasks.split(',')]
        for st in subtasks:
            bench_key = f'MV_{st}'
            if _bench_filter and _bench_filter != bench_key:
                continue
            items = load_mathverse(args.mathverse_dir, subtask=st)
            if items:
                benchmarks[bench_key] = ('mathverse', items)
                print(f'  Loaded {len(items)} MathVerse {st} questions')

    if args.pope_img_dir and args.pope_dir and os.path.isdir(args.pope_dir):
        if not _bench_filter or _bench_filter == 'POPE':
            pope_protocols = load_pope_all_protocols(args.pope_dir)
            if pope_protocols:
                benchmarks['POPE'] = ('pope_all', pope_protocols)

    if args.triviaqa_path:
        if not _bench_filter or _bench_filter == 'TriviaQA':
            tqa_items = load_triviaqa(args.triviaqa_path,
                                       num_questions=args.triviaqa_num,
                                       seed=args.seed)
            if tqa_items:
                benchmarks['TriviaQA'] = ('triviaqa', tqa_items)

    if not benchmarks:
        print('  ERROR: No benchmarks loaded. Provide --mathverse_dir')
        return

    # Apply sample limit for smoke testing
    if args.sample_limit > 0:
        print(f'  [limit] Capping each benchmark to {args.sample_limit} samples')
        for bname, (btype, bdata) in list(benchmarks.items()):
            if isinstance(bdata, list) and len(bdata) > args.sample_limit:
                benchmarks[bname] = (btype, bdata[:args.sample_limit])
                print(f'    {bname}: → {args.sample_limit}')
            elif isinstance(bdata, dict):
                # POPE: dict {protocol: [questions]} — cap each protocol
                capped = {}
                for proto, qs in bdata.items():
                    capped[proto] = qs[:args.sample_limit]
                    print(f'    {bname}/{proto}: {len(qs)} → {len(capped[proto])}')
                benchmarks[bname] = (btype, capped)

    bench_names = list(benchmarks.keys())

    # ── Evaluation helper ──
    def _evaluate_all(mcq_suffix=True):
        results = {}
        for bname, (btype, bdata) in benchmarks.items():
            if btype == 'pope_all':
                r = evaluate_pope_all_protocols(
                    model, args.model_type, processor,
                    image_processor, image_token_id,
                    bdata, args.pope_img_dir, device)
                results[bname] = r
            elif btype == 'mathverse':
                r = evaluate_mathverse(model, args.model_type, processor,
                                       image_processor, image_token_id,
                                       bdata, device, mcq_suffix=mcq_suffix)
                results[bname] = r
            elif btype == 'triviaqa':
                r = evaluate_triviaqa(model, args.model_type, processor,
                                      image_processor, image_token_id,
                                      bdata, device, text_only=True)
                results[bname] = r
        return results

    # ── Build neuron map helper ──
    def _build_map(neuron_list):
        m = defaultdict(list)
        for n in neuron_list:
            m[n['layer']].append(n['neuron_idx'])
        return dict(m)

    # ── Baseline: load from file or run fresh ──
    def _load_saved_baseline():
        """Try to load baseline from a previously saved baseline_{bench}.json."""
        if not args.benchmark:
            return None
        bl_path = os.path.join(args.output_dir, f'baseline_{args.benchmark}.json')
        if os.path.isfile(bl_path):
            with open(bl_path) as f:
                data = json.load(f)
            if 'baseline' in data:
                print(f'  Loaded saved baseline from {bl_path}')
                return data['baseline']
        return None

    # ── Baseline-only mode: ablation-style eval, save and exit ──
    if args.baseline_only:
        os.makedirs(args.output_dir, exist_ok=True)
        _bench_suffix = f'_{args.benchmark}' if args.benchmark else ''

        print(f'\n  ── Baseline (no ablation, with MCQ suffix) ──')
        baseline = _evaluate_all(mcq_suffix=True)
        for bname, r in baseline.items():
            print(f'    {bname}: acc={r["accuracy"]:.4f}  n_parsed={r.get("n_parsed","N/A")}')

        # Save predictions separately for GPT scoring
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        for bname, r in baseline.items():
            if r.get('predictions'):
                pred_path = os.path.join(pred_dir, f'pred_baseline_{bname}.json')
                with open(pred_path, 'w') as pf:
                    json.dump(r['predictions'], pf)
                print(f'    Predictions saved: {pred_path}')

        # Strip predictions from baseline before saving (keep JSON small)
        baseline_clean = {}
        for bn in bench_names:
            baseline_clean[bn] = {k: v for k, v in baseline[bn].items() if k != 'predictions'}

        save_data = {
            'model_type': args.model_type,
            'model_name': args.model_name,
            'hook_point': args.hook_point,
            'eval_mode': 'ablation',
            'mcq_suffix': True,
            'benchmarks': bench_names,
            'baseline': baseline_clean,
        }
        out_path = os.path.join(args.output_dir, f'baseline{_bench_suffix}.json')
        with open(out_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f'\n  Baseline saved to {out_path}')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'\n  Baseline-only complete.')
        return

    # ── Ablation mode: load or run ablation-style baseline ──
    saved_baseline = _load_saved_baseline()
    if saved_baseline is not None:
        baseline = saved_baseline
        for bname, r in baseline.items():
            print(f'    {bname}: acc={r["accuracy"]:.4f} (from file)')
    else:
        print(f'\n  ── Baseline (no ablation, with MCQ suffix) ──')
        baseline = _evaluate_all(mcq_suffix=True)
        for bname, r in baseline.items():
            print(f'    {bname}: acc={r["accuracy"]:.4f}')

    # ── Determine fractions to run ──
    if args.fraction is not None:
        fracs_to_run = [args.fraction]
    else:
        fracs_to_run = fracs

    # ── Sweep each category ──
    os.makedirs(args.output_dir, exist_ok=True)
    all_curves = {}

    for cat in categories:
        cat_neurons = by_cat.get(cat, [])
        if not cat_neurons:
            print(f'\n  [skip] {cat}: no neurons')
            continue

        # Rank neurons
        ranked = rank_neurons_for_ablation(
            list(cat_neurons), args.ranking, norms_by_layer, cat)

        print(f'\n  ══ {cat} ({len(ranked)} neurons, ranked by {args.ranking}) ══')
        if ranked:
            top = ranked[0]
            print(f'    Top neuron: layer={top["layer"]} idx={top["neuron_idx"]} '
                  f'D={top["rate_diff"]:.4f} norm={top.get("norm", 0):.4f}')

        cat_curve = {'ranked': {}, 'random': {}}

        for frac in fracs_to_run:
            K = max(1, int(len(ranked) * frac))
            if K > len(ranked):
                K = len(ranked)

            frac_key = f'{frac:.2f}'

            # ── Determine what to run at this fraction ──
            run_ranked = (args.trial_idx is None or args.trial_idx == 'ranked')
            if args.trial_idx is None:
                # Full sweep: run ranked + all random trials
                random_seeds_to_run = list(range(n_random))
            elif args.trial_idx == 'ranked':
                random_seeds_to_run = []
            else:
                # Single random trial
                run_ranked = False
                random_seeds_to_run = [int(args.trial_idx)]

            # ── Ranked ablation (deterministic) ──
            if run_ranked:
                top_k = ranked[:K]
                nmap = _build_map(top_k)
                n_actual = sum(len(v) for v in nmap.values())
                print(f'\n    frac={frac:.0%}: ablating top-{K} ranked {cat} neurons '
                      f'({n_actual} across {len(nmap)} layers)')

                ablation = WeightZeroing(model, args.model_type, nmap,
                                         hook_point=args.hook_point)
                with ablation:
                    ranked_results = _evaluate_all()

                cat_curve['ranked'][frac_key] = {'K': K}
                for bname, r in ranked_results.items():
                    delta = r['accuracy'] - baseline[bname]['accuracy']
                    cat_curve['ranked'][frac_key][bname] = {
                        'accuracy': r['accuracy'],
                        'delta': delta,
                    }
                    print(f'      {bname}: acc={r["accuracy"]:.4f} '
                          f'(Δ={delta:+.4f} vs baseline)')

                    # Save predictions for GPT scoring (Run 1)
                    if r.get('predictions'):
                        pred_dir = os.path.join(args.output_dir, 'predictions')
                        os.makedirs(pred_dir, exist_ok=True)
                        pred_path = os.path.join(pred_dir,
                            f'pred_{args.hook_point}_{cat}_{bname}_f{frac:.2f}_ranked.json')
                        with open(pred_path, 'w') as pf:
                            json.dump(r['predictions'], pf)

            # ── Random baselines ──
            if random_seeds_to_run:
                random_accs = {bn: [] for bn in bench_names}
                for seed_i in random_seeds_to_run:
                    rng = np.random.RandomState(args.seed + seed_i * 1000)
                    random_indices = rng.choice(len(neurons), size=K, replace=False)
                    random_neurons = [neurons[i] for i in random_indices]
                    rmap = _build_map(random_neurons)

                    print(f'\n    frac={frac:.0%}: random trial {seed_i} '
                          f'(K={K}, seed={args.seed + seed_i * 1000})')

                    ablation = WeightZeroing(model, args.model_type, rmap,
                                             hook_point=args.hook_point)
                    with ablation:
                        rand_results = _evaluate_all()

                    for bname, r in rand_results.items():
                        random_accs[bname].append(r['accuracy'])
                        print(f'      {bname}: acc={r["accuracy"]:.4f}')

                        # Save predictions for GPT scoring (Run 1)
                        if r.get('predictions'):
                            pred_dir = os.path.join(args.output_dir, 'predictions')
                            os.makedirs(pred_dir, exist_ok=True)
                            pred_path = os.path.join(pred_dir,
                                f'pred_{args.hook_point}_{cat}_{bname}_f{frac:.2f}_r{seed_i}.json')
                            with open(pred_path, 'w') as pf:
                                json.dump(r['predictions'], pf)

                cat_curve['random'][frac_key] = {'K': K}
                for bname in bench_names:
                    accs = random_accs[bname]
                    cat_curve['random'][frac_key][bname] = {
                        'mean': float(np.mean(accs)),
                        'std': float(np.std(accs)),
                        'seeds': accs,
                        'seed_indices': random_seeds_to_run,
                    }

            # ── Compute delta_vs_random if both ranked and random exist ──
            if (frac_key in cat_curve['ranked'] and
                    frac_key in cat_curve['random']):
                for bname in bench_names:
                    if (bname in cat_curve['ranked'][frac_key] and
                            bname in cat_curve['random'][frac_key]):
                        rand_mean = cat_curve['random'][frac_key][bname]['mean']
                        ranked_acc = cat_curve['ranked'][frac_key][bname]['accuracy']
                        cat_curve['ranked'][frac_key][bname]['delta_vs_random'] = \
                            ranked_acc - rand_mean
                        print(f'      {bname} ranked Δ vs random: '
                              f'{ranked_acc - rand_mean:+.4f}')

        all_curves[cat] = cat_curve

    # ── Save results ──
    save_data = {
        'model_type': args.model_type,
        'model_name': args.model_name,
        'hook_point': args.hook_point,
        'ranking': args.ranking,
        'sweep_fracs': [float(f) for f in fracs_to_run],
        'n_random_trials': n_random,
        'benchmarks': bench_names,
        'baseline': {bn: baseline[bn] for bn in bench_names},
        'neuron_counts': {cat: len(by_cat.get(cat, [])) for cat in categories},
        'curves': all_curves,
    }

    _cat_suffix = f'_{args.category}' if args.category else ''
    _bench_suffix = f'_{args.benchmark}' if args.benchmark else ''
    _frac_suffix = f'_f{args.fraction:.2f}' if args.fraction is not None else ''
    if args.trial_idx is not None:
        _trial_suffix = f'_{args.trial_idx}' if args.trial_idx == 'ranked' \
            else f'_rand{args.trial_idx}'
    else:
        _trial_suffix = ''
    out_path = os.path.join(
        args.output_dir,
        f'sweep_{args.ranking}_{args.hook_point}'
        f'{_cat_suffix}{_bench_suffix}{_frac_suffix}{_trial_suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\n  Saved to {out_path}')

    # ── Print summary table ──
    _print_sweep_summary(args, all_curves, fracs, bench_names, categories, baseline)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'\n  Phase 1 complete.')


def _print_sweep_summary(args, all_curves, fracs, bench_names, categories, baseline):
    """Print summary tables for the sweep."""
    print(f'\n{"="*60}')
    print(f'SWEEP SUMMARY: {args.ranking} ranking, {args.hook_point} hook')
    print(f'{"="*60}')

    print(f'\n  Baseline:')
    for bn in bench_names:
        print(f'    {bn}: {baseline[bn]["accuracy"]:.4f}')

    for cat in categories:
        if cat not in all_curves:
            continue
        curve = all_curves[cat]
        print(f'\n  {cat.upper()} ({args.ranking}):')
        header = f'    {"frac":>6} {"K":>7}'
        for bn in bench_names:
            header += f'  {"acc":>7} {"Δbase":>7} {"Δrand":>7}'
        print(header)
        print(f'    {"-"*6} {"-"*7}' + (f'  {"-"*7} {"-"*7} {"-"*7}') * len(bench_names))
        for frac in fracs:
            fk = f'{frac:.2f}'
            if fk not in curve['ranked']:
                continue
            K = curve['ranked'][fk].get('K', '?')
            row = f'    {frac:>6.0%} {K:>7}'
            for bn in bench_names:
                r = curve['ranked'][fk].get(bn, {})
                acc = r.get('accuracy', 0)
                delta_base = r.get('delta', 0)
                delta_rand = r.get('delta_vs_random', 0)
                row += f'  {acc:>7.4f} {delta_base:>+7.4f} {delta_rand:>+7.4f}'
            print(row)


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Merge + statistics (CPU only)
# ═══════════════════════════════════════════════════════════════════════

def run_phase2(args):
    """Merge sweep results across rankings and compute summary statistics.

    Scans output_dir for sweep_*.json files, compiles a comparison table
    across rankings, and computes:
      - Z-score per fraction: (ranked_acc - random_mean) / random_std
      - Sign test across fractions: how many fractions show ranked < random
        (binomial test, H0: p=0.5)
      - Mann-Whitney U test per fraction (only when n_random >= 30)
    """
    try:
        from scipy.stats import binomtest
        def _sign_test(k, n):
            return binomtest(k, n, 0.5).pvalue
    except ImportError:
        from scipy.stats import binom_test
        def _sign_test(k, n):
            return binom_test(k, n, 0.5)

    print(f'\n{"="*60}')
    print(f'PHASE 2: Merge sweep results + statistics')
    print(f'{"="*60}\n')

    # ── Scan for sweep files ──
    sweep_files = sorted(glob.glob(
        os.path.join(args.output_dir, 'sweep_*.json')))
    if not sweep_files:
        print(f'  No sweep files found in {args.output_dir}')
        return

    print(f'  Found {len(sweep_files)} sweep files')

    # ── Load all sweep results ──
    all_sweeps = {}
    for fpath in sweep_files:
        with open(fpath) as f:
            data = json.load(f)
        ranking = data.get('ranking', 'unknown')
        hook = data.get('hook_point', 'unknown')
        key = f'{ranking}_{hook}'
        all_sweeps[key] = data
        print(f'    {key}: {fpath}')

    # ── Compile comparison with statistics ──
    categories = [c.strip() for c in args.categories.split(',')]
    bench_names = None
    all_stats = {}  # key → {cat → {bench → stats}}

    for key, data in sorted(all_sweeps.items()):
        if bench_names is None:
            bench_names = data.get('benchmarks', [])
        n_random = data.get('n_random_trials', 5)

        print(f'\n{"="*60}')
        print(f'  {key}  (n_random={n_random})')
        print(f'{"="*60}')

        key_stats = {}

        for cat in categories:
            curves = data.get('curves', {}).get(cat, {})
            ranked = curves.get('ranked', {})
            random_data = curves.get('random', {})
            if not ranked:
                continue

            cat_stats = {}

            # ── Per-fraction z-scores ──
            print(f'\n    {cat.upper()}:')
            header = f'      {"frac":>6} {"K":>6}'
            for bn in bench_names:
                header += f'  {"ranked":>7} {"Δrand":>7} {"z":>7}'
                if n_random >= 30:
                    header += f' {"MW-p":>7}'
            print(header)
            print(f'      {"-"*6} {"-"*6}' +
                  (f'  {"-"*7} {"-"*7} {"-"*7}' + (f' {"-"*7}' if n_random >= 30 else '')) * len(bench_names))

            # Track sign direction per benchmark across fractions
            sign_counts = {bn: {'below': 0, 'total': 0} for bn in bench_names}

            for fk in sorted(ranked.keys()):
                K = ranked[fk].get('K', '?')
                row = f'      {fk:>6} {K:>6}'
                frac_stats = {'K': K}

                for bn in bench_names:
                    r = ranked[fk].get(bn, {})
                    rd = random_data.get(fk, {}).get(bn, {})
                    ranked_acc = r.get('accuracy', 0)
                    seeds = rd.get('seeds', [])
                    rand_mean = rd.get('mean', 0)
                    rand_std = rd.get('std', 0)
                    dvr = r.get('delta_vs_random', 0)

                    # Z-score: how many std below random mean
                    if rand_std > 1e-8:
                        z = (ranked_acc - rand_mean) / rand_std
                    elif abs(ranked_acc - rand_mean) > 1e-8:
                        z = float('-inf') if ranked_acc < rand_mean else float('inf')
                    else:
                        z = 0.0

                    bn_stats = {
                        'ranked_acc': ranked_acc,
                        'random_mean': rand_mean,
                        'random_std': rand_std,
                        'delta_vs_random': dvr,
                        'z_score': z,
                        'n_random': len(seeds),
                    }

                    row += f'  {ranked_acc:>7.4f} {dvr:>+7.4f} {z:>+7.2f}'

                    # Track sign for sign test (ranked below random = effect)
                    if seeds:
                        sign_counts[bn]['total'] += 1
                        if ranked_acc < rand_mean:
                            sign_counts[bn]['below'] += 1

                    # Mann-Whitney U (only with >= 30 random trials)
                    if n_random >= 30 and len(seeds) >= 30:
                        from scipy.stats import mannwhitneyu
                        # Compare single ranked value against random distribution
                        # Use [ranked_acc]*1 vs seeds; one-sided (ranked < random)
                        try:
                            _, mw_p = mannwhitneyu(
                                [ranked_acc], seeds, alternative='less')
                        except ValueError:
                            mw_p = 1.0
                        sig = '***' if mw_p < 0.001 else '**' if mw_p < 0.01 else '*' if mw_p < 0.05 else ''
                        bn_stats['mann_whitney_p'] = mw_p
                        row += f' {mw_p:>6.3f}{sig}'

                    frac_stats[bn] = bn_stats

                cat_stats[fk] = frac_stats
                print(row)

            # ── Sign test across fractions ──
            print(f'\n      Sign test (ranked < random at how many fractions):')
            for bn in bench_names:
                sc = sign_counts[bn]
                n_below = sc['below']
                n_total = sc['total']
                if n_total > 0:
                    # Binomial test: H0 = ranked is equally likely above/below random
                    sign_p = _sign_test(n_below, n_total)
                    direction = 'BELOW' if n_below > n_total / 2 else 'above'
                    sig = '***' if sign_p < 0.001 else '**' if sign_p < 0.01 else '*' if sign_p < 0.05 else ''
                    print(f'        {bn}: {n_below}/{n_total} fractions {direction} '
                          f'(sign p={sign_p:.4f}) {sig}')
                    cat_stats[f'sign_test_{bn}'] = {
                        'n_below': n_below,
                        'n_total': n_total,
                        'p_value': sign_p,
                    }

            key_stats[cat] = cat_stats

        all_stats[key] = key_stats

    # ── Save merged comparison ──
    os.makedirs(args.output_dir, exist_ok=True)
    merged = {
        'model_type': args.model_type,
        'model_name': args.model_name,
        'sweeps': {k: v for k, v in all_sweeps.items()},
        'statistics': all_stats,
    }
    out_path = os.path.join(args.output_dir, 'merged_sweep_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'\n  Merged comparison saved to {out_path}')

    # ── Save CSV summary ──
    import pandas as pd
    rows = []
    for key, data in sorted(all_sweeps.items()):
        ranking = data.get('ranking', '')
        hook = data.get('hook_point', '')
        n_random = data.get('n_random_trials', 5)
        for cat in categories:
            curves = data.get('curves', {}).get(cat, {})
            ranked = curves.get('ranked', {})
            stats = all_stats.get(key, {}).get(cat, {})
            for fk in sorted(ranked.keys()):
                row = {'ranking': ranking, 'hook': hook, 'category': cat,
                       'fraction': float(fk), 'K': ranked[fk].get('K', 0),
                       'n_random': n_random}
                frac_stats = stats.get(fk, {})
                for bn in (bench_names or []):
                    r = ranked[fk].get(bn, {})
                    bs = frac_stats.get(bn, {})
                    row[f'{bn}_acc'] = r.get('accuracy')
                    row[f'{bn}_delta_base'] = r.get('delta')
                    row[f'{bn}_delta_rand'] = r.get('delta_vs_random')
                    row[f'{bn}_z_score'] = bs.get('z_score')
                    if 'mann_whitney_p' in bs:
                        row[f'{bn}_mw_p'] = bs.get('mann_whitney_p')
                rows.append(row)

    csv_path = os.path.join(args.output_dir, 'sweep_comparison.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'  CSV saved to {csv_path}')

    print(f'\n  Phase 2 complete.')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Normalize qwen25vl-7b / qwen25vl-3b → qwen2vl: same architecture
    if args.model_type in ('qwen25vl-7b', 'qwen25vl-3b'):
        args.model_type = 'qwen2vl'

    if args.phase == 1:
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)


if __name__ == '__main__':
    main()