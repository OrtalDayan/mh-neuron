#!/usr/bin/env python3
"""Prepare MathVerse subsets for ablation experiments."""
import argparse, json, os
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/mathverse')
    parser.add_argument('--all_versions', action='store_true')
    args = parser.parse_args()

    from datasets import load_dataset

    print('Loading MathVerse testmini...')
    ds = load_dataset('AI4Math/MathVerse', 'testmini', split='testmini')
    print(f'  Total: {len(ds)} questions')

    versions = Counter(ds['problem_version'])
    print(f'  Versions:')
    for v, c in sorted(versions.items()):
        print(f'    {v}: {c}')

    if args.all_versions:
        target_versions = list(versions.keys())
    else:
        target_versions = ['Text Dominant']

    filtered = [(i, item) for i, item in enumerate(ds)
                if item['problem_version'] in target_versions
                and item['question_type'] == 'multi-choice']

    print(f'  Filtered: {len(filtered)} multi-choice questions')

    img_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    questions = []
    for idx, (ds_idx, item) in enumerate(filtered):
        img_filename = f'mathverse_{ds_idx:05d}.png'
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            if item.get('image') is not None:
                item['image'].save(img_path)
            else:
                from PIL import Image
                Image.new('RGB', (224, 224), (255, 255, 255)).save(img_path)

        questions.append({
            'index': idx, 'ds_index': ds_idx,
            'problem_version': item['problem_version'],
            'question_type': item['question_type'],
            'question': item.get('query_wo', item.get('question', '')),
            'answer': str(item['answer']).strip(),
            'image': img_filename,
        })

    with open(os.path.join(args.output_dir, 'questions.json'), 'w') as f:
        json.dump(questions, f, indent=2)

    ver_counts = Counter(q['problem_version'] for q in questions)
    print(f'  Saved {len(questions)} questions to {args.output_dir}/questions.json')
    for v, c in sorted(ver_counts.items()):
        print(f'    {v}: {c}')

if __name__ == '__main__':
    main()
