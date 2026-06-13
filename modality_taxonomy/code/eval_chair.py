#!/usr/bin/env python3
"""
eval_chair.py -- CHAIR hallucination evaluation for VLMs

Implements the CHAIR (Caption Hallucination Assessment with Image Relevance)
protocol from Rohrbach et al. (EMNLP 2018), matching the SRF paper
(Ali, Zoabi & Wolf, arXiv 2511.12220).

Protocol:
  1. Sample N random COCO val2014 images
  2. Generate free-form captions with the VLM using
     "Please describe this image in detail."
  3. Score against COCO instance annotations:
       CHAIRs = % of sentences containing >= 1 hallucinated object
       CHAIRi = % of mentioned objects that are hallucinated

Usage:
  python eval_chair.py \
      --vlm_path results/24-srf/full/llava-onevision-7b/srf_a0.5_m10 \
      --model_type llava-ov \
      --coco_ann_path data/annotations/instances_val2014.json \
      --coco_img_dir data/val2014 \
      --n_images 500 \
      --output_dir results/chair_eval/llava-ov_srf
"""

import argparse
import json
import os
import random
import re
import sys
import time

import nltk
import torch
from PIL import Image
from tqdm import tqdm


# ================================================================
# Section 1 -- Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description='CHAIR hallucination evaluation')
    p.add_argument('--vlm_path', required=True,
                   help='Path to VLM checkpoint (HF format)')
    p.add_argument('--model_type', required=True,
                   choices=['llava-ov', 'qwen2vl', 'internvl',
                            'llava-hf', 'llava-llama3'],
                   help='VLM architecture type')
    p.add_argument('--coco_ann_path', required=True,
                   help='Path to instances_val2014.json')
    p.add_argument('--coco_img_dir', default='data/val2014',
                   help='Directory with COCO val2014 images')
    p.add_argument('--n_images', type=int, default=500,
                   help='Number of random images to caption')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for image sampling')
    p.add_argument('--prompt', default='Please describe this image in detail.',
                   help='Caption prompt (matches SRF paper)')
    p.add_argument('--max_new_tokens', type=int, default=512,
                   help='Max tokens for caption generation')
    p.add_argument('--output_dir', required=True,
                   help='Directory to save captions and scores')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


# ================================================================
# Section 2 -- COCO object synonyms (from Rohrbach et al.)
# ================================================================

# Maps COCO category name -> list of common synonyms.
# Taken from the original CHAIR implementation.
SYNONYMS = {
    'person': ['person', 'man', 'woman', 'child', 'boy', 'girl',
               'people', 'men', 'women', 'children', 'kid', 'kids',
               'lady', 'guy', 'player', 'rider', 'pedestrian',
               'baby', 'toddler', 'adult', 'teenager', 'teen'],
    'bicycle': ['bicycle', 'bike', 'cycling'],
    'car': ['car', 'automobile', 'vehicle', 'sedan', 'suv', 'van', 'taxi'],
    'motorcycle': ['motorcycle', 'motorbike', 'scooter'],
    'airplane': ['airplane', 'plane', 'aircraft', 'jet', 'aeroplane'],
    'bus': ['bus'],
    'train': ['train', 'locomotive', 'railway'],
    'truck': ['truck', 'lorry', 'pickup'],
    'boat': ['boat', 'ship', 'vessel', 'sailboat', 'canoe', 'kayak', 'yacht'],
    'traffic light': ['traffic light', 'stoplight', 'traffic signal'],
    'fire hydrant': ['fire hydrant', 'hydrant'],
    'stop sign': ['stop sign'],
    'parking meter': ['parking meter'],
    'bench': ['bench', 'park bench'],
    'bird': ['bird', 'parrot', 'pigeon', 'sparrow', 'seagull', 'goose',
             'duck', 'crow', 'robin', 'owl', 'eagle', 'hawk', 'pelican'],
    'cat': ['cat', 'kitten', 'kitty', 'feline'],
    'dog': ['dog', 'puppy', 'pup', 'canine', 'hound'],
    'horse': ['horse', 'pony', 'stallion', 'mare', 'foal'],
    'sheep': ['sheep', 'lamb', 'ewe', 'ram'],
    'cow': ['cow', 'cattle', 'bull', 'calf', 'ox', 'steer', 'bovine'],
    'elephant': ['elephant'],
    'bear': ['bear', 'polar bear', 'grizzly'],
    'zebra': ['zebra'],
    'giraffe': ['giraffe'],
    'backpack': ['backpack', 'rucksack', 'knapsack', 'bag'],
    'umbrella': ['umbrella', 'parasol'],
    'handbag': ['handbag', 'purse', 'clutch'],
    'tie': ['tie', 'necktie', 'bowtie'],
    'suitcase': ['suitcase', 'luggage', 'briefcase'],
    'frisbee': ['frisbee', 'disc'],
    'skis': ['skis', 'ski'],
    'snowboard': ['snowboard'],
    'sports ball': ['ball', 'sports ball', 'baseball', 'basketball',
                    'soccer ball', 'football', 'tennis ball'],
    'kite': ['kite'],
    'baseball bat': ['baseball bat', 'bat'],
    'baseball glove': ['baseball glove', 'glove', 'mitt'],
    'skateboard': ['skateboard'],
    'surfboard': ['surfboard'],
    'tennis racket': ['tennis racket', 'racket', 'racquet'],
    'bottle': ['bottle'],
    'wine glass': ['wine glass', 'glass', 'goblet'],
    'cup': ['cup', 'mug'],
    'fork': ['fork'],
    'knife': ['knife'],
    'spoon': ['spoon'],
    'bowl': ['bowl'],
    'banana': ['banana'],
    'apple': ['apple'],
    'sandwich': ['sandwich', 'sub', 'burger', 'hamburger'],
    'orange': ['orange'],
    'broccoli': ['broccoli'],
    'carrot': ['carrot'],
    'hot dog': ['hot dog', 'hotdog'],
    'pizza': ['pizza'],
    'donut': ['donut', 'doughnut'],
    'cake': ['cake'],
    'chair': ['chair', 'seat'],
    'couch': ['couch', 'sofa', 'loveseat'],
    'potted plant': ['potted plant', 'plant', 'flower', 'houseplant'],
    'bed': ['bed'],
    'dining table': ['dining table', 'table', 'desk'],
    'toilet': ['toilet', 'restroom'],
    'tv': ['tv', 'television', 'monitor', 'screen'],
    'laptop': ['laptop', 'notebook', 'computer'],
    'mouse': ['mouse'],
    'remote': ['remote', 'remote control'],
    'keyboard': ['keyboard'],
    'cell phone': ['cell phone', 'phone', 'cellphone', 'smartphone',
                   'mobile phone', 'mobile'],
    'microwave': ['microwave', 'oven'],
    'oven': ['oven', 'stove'],
    'toaster': ['toaster'],
    'sink': ['sink', 'basin'],
    'refrigerator': ['refrigerator', 'fridge'],
    'book': ['book'],
    'clock': ['clock'],
    'vase': ['vase'],
    'scissors': ['scissors', 'shears'],
    'teddy bear': ['teddy bear', 'stuffed animal', 'stuffed bear',
                   'teddy', 'plush'],
    'hair drier': ['hair drier', 'hair dryer', 'dryer', 'blow dryer'],
    'toothbrush': ['toothbrush'],
}


def build_synonym_lookup():
    """Build reverse lookup: synonym word -> set of COCO category names."""
    reverse = {}
    for coco_cat, syns in SYNONYMS.items():
        for syn in syns:
            for word in syn.split():
                word_lower = word.lower().strip()
                if word_lower:
                    if word_lower not in reverse:
                        reverse[word_lower] = set()
                    reverse[word_lower].add(coco_cat)
    return reverse


# ================================================================
# Section 3 -- CHAIR metric computation
# ================================================================

def load_coco_annotations(ann_path):
    """Load COCO instance annotations.

    Returns:
        image_to_objects: dict {image_id: set of COCO category names}
        id_to_name: dict {category_id: category name}
    """
    print(f'  Loading COCO annotations from {ann_path}...')
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

    image_to_objects = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cat_name = id_to_name[ann['category_id']]
        if img_id not in image_to_objects:
            image_to_objects[img_id] = set()
        image_to_objects[img_id].add(cat_name)

    print(f'  {len(image_to_objects)} images with annotations, '
          f'{len(id_to_name)} categories')
    return image_to_objects, id_to_name


def extract_objects_from_caption(caption, synonym_lookup):
    """Extract COCO objects mentioned in a caption using NLTK lemmatization."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = nltk.word_tokenize(caption.lower())
    lemmas = [lemmatizer.lemmatize(w) for w in words]

    mentioned = set()
    for lemma in lemmas:
        if lemma in synonym_lookup:
            mentioned.update(synonym_lookup[lemma])
    return mentioned


def compute_chair(captions, image_to_objects, synonym_lookup):
    """Compute CHAIR metrics.

    CHAIRs = sentences with >= 1 hallucinated object / total sentences
    CHAIRi = hallucinated objects / total mentioned objects
    """
    n_sentences_with_halluc = 0
    n_total_sentences = 0
    n_hallucinated_objects = 0
    n_total_mentioned_objects = 0
    per_caption = []

    for item in captions:
        img_id = item['image_id']
        caption = item['caption']
        gt_objects = image_to_objects.get(img_id, set())
        mentioned = extract_objects_from_caption(caption, synonym_lookup)
        hallucinated = mentioned - gt_objects

        n_total_sentences += 1
        if len(hallucinated) > 0:
            n_sentences_with_halluc += 1
        n_total_mentioned_objects += len(mentioned)
        n_hallucinated_objects += len(hallucinated)

        per_caption.append({
            'image_id': img_id,
            'caption': caption,
            'gt_objects': sorted(gt_objects),
            'mentioned_objects': sorted(mentioned),
            'hallucinated_objects': sorted(hallucinated),
        })

    chairs = n_sentences_with_halluc / max(n_total_sentences, 1)
    chairi = n_hallucinated_objects / max(n_total_mentioned_objects, 1)

    return {
        'CHAIRs': round(chairs * 100, 2),
        'CHAIRi': round(chairi * 100, 2),
        'n_images': len(captions),
        'n_sentences_with_halluc': n_sentences_with_halluc,
        'n_total_mentioned_objects': n_total_mentioned_objects,
        'n_hallucinated_objects': n_hallucinated_objects,
        'per_caption': per_caption,
    }


# ================================================================
# Section 4 -- VLM caption generation
# ================================================================

def load_model(model_type, vlm_path, device):
    """Load VLM for caption generation."""
    print(f'  Loading {model_type} from {vlm_path}...')

    if model_type == 'llava-ov':
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                   AutoProcessor)
        processor = AutoProcessor.from_pretrained(vlm_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).eval().to(device)

    elif model_type == 'qwen2vl':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(vlm_path)
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).eval().to(device)

    elif model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            vlm_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True).eval().to(device)
        processor = AutoTokenizer.from_pretrained(
            vlm_path, trust_remote_code=True)

    elif model_type == 'llava-hf':
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(vlm_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval().to(device)

    elif model_type == 'llava-llama3':
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        processor = AutoProcessor.from_pretrained(vlm_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval().to(device)

    else:
        raise ValueError(f'Unsupported model_type: {model_type}')

    return model, processor


def generate_caption(model, processor, image, prompt, model_type,
                     device, max_new_tokens=512):
    """Generate a free-form caption for one image."""
    with torch.no_grad():
        if model_type == 'llava-ov':
            msgs = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=image, text=text,
                               return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            plen = inputs['input_ids'].shape[1]
            return processor.decode(out[0][plen:],
                                     skip_special_tokens=True).strip()

        elif model_type == 'qwen2vl':
            msgs = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=image, text=text,
                               return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            plen = inputs['input_ids'].shape[1]
            return processor.decode(out[0][plen:],
                                     skip_special_tokens=True).strip()

        elif model_type == 'internvl':
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            tf = T.Compose([
                T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            pv = tf(image).unsqueeze(0).to(torch.bfloat16).to(device)
            return model.chat(
                processor, pv, f'<image>\n{prompt}',
                dict(max_new_tokens=max_new_tokens, do_sample=False))

        elif model_type == 'llava-hf':
            text = f'USER: <image>\n{prompt}\nASSISTANT:'
            inputs = processor(text=text, images=image,
                               return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            generated = processor.decode(out[0], skip_special_tokens=True)
            if 'ASSISTANT:' in generated:
                return generated.split('ASSISTANT:')[-1].strip()
            return generated.strip()

        elif model_type == 'llava-llama3':
            msgs = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=image, text=text,
                               return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            plen = inputs['input_ids'].shape[1]
            return processor.decode(out[0][plen:],
                                     skip_special_tokens=True).strip()


# ================================================================
# Section 5 -- Main
# ================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  CHAIR Hallucination Evaluation')
    print(f'  Model: {args.vlm_path}')
    print(f'  Type:  {args.model_type}')
    print(f'  N:     {args.n_images} images')
    print(f'{"="*60}\n')

    # -- Load COCO annotations --
    image_to_objects, id_to_name = load_coco_annotations(args.coco_ann_path)
    synonym_lookup = build_synonym_lookup()

    # -- Sample random images --
    all_img_ids = sorted(image_to_objects.keys())
    random.seed(args.seed)
    sampled_ids = random.sample(all_img_ids,
                                min(args.n_images, len(all_img_ids)))
    print(f'  Sampled {len(sampled_ids)} images')

    # -- Check for existing captions (resume support) --
    captions_path = os.path.join(args.output_dir, 'captions.jsonl')
    existing_captions = {}
    if os.path.isfile(captions_path):
        with open(captions_path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                existing_captions[item['image_id']] = item
        print(f'  Found {len(existing_captions)} existing captions (resuming)')

    # -- Generate captions --
    remaining = [i for i in sampled_ids if i not in existing_captions]
    if remaining:
        print(f'\n  Generating captions for {len(remaining)} images...')
        model, processor = load_model(
            args.model_type, args.vlm_path, args.device)

        with open(captions_path, 'a') as f:
            for img_id in tqdm(remaining, desc='  Captioning'):
                img_file = f'COCO_val2014_{img_id:012d}.jpg'
                img_path = os.path.join(args.coco_img_dir, img_file)

                if not os.path.isfile(img_path):
                    continue

                try:
                    img = Image.open(img_path).convert('RGB')
                    caption = generate_caption(
                        model, processor, img, args.prompt,
                        args.model_type, args.device, args.max_new_tokens)
                    item = {'image_id': img_id, 'caption': caption}
                    f.write(json.dumps(item) + '\n')
                    f.flush()
                    existing_captions[img_id] = item
                except Exception as e:
                    print(f'    Warning: failed on {img_file}: {e}')
                    continue

        del model
        torch.cuda.empty_cache()

    # -- Compute CHAIR --
    print(f'\n  Computing CHAIR metrics...')
    all_captions = [existing_captions[i] for i in sampled_ids
                    if i in existing_captions]

    results = compute_chair(all_captions, image_to_objects, synonym_lookup)

    # -- Save results --
    full_path = os.path.join(args.output_dir, 'chair_results_full.json')
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2)

    summary = {k: v for k, v in results.items() if k != 'per_caption'}
    summary['model_path'] = args.vlm_path
    summary['model_type'] = args.model_type
    summary['prompt'] = args.prompt
    summary_path = os.path.join(args.output_dir, 'chair_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*60}')
    print(f'  CHAIR Results')
    print(f'  CHAIRs: {results["CHAIRs"]:.1f}%')
    print(f'  CHAIRi: {results["CHAIRi"]:.1f}%')
    print(f'  ({results["n_images"]} images, '
          f'{results["n_total_mentioned_objects"]} objects mentioned, '
          f'{results["n_hallucinated_objects"]} hallucinated)')
    print(f'{"="*60}')
    print(f'  Results saved to {summary_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
