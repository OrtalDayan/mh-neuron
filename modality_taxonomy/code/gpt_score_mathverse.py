#!/usr/bin/env python3
"""Score MathVerse predictions using GPT (matching VLMEvalKit's evaluation).

Reads prediction JSON files saved by equal_fraction_ablation.py and runs
VLMEvalKit's two-step GPT evaluation:
  1. GPT extracts the answer from the model's response
  2. GPT judges if the extracted answer matches the ground truth

Usage:
    # Score a single prediction file
    python gpt_score_mathverse.py --pred_file predictions/pred_gate_up_visual_MV_TD_f0.10_ranked.json

    # Score all prediction files in a directory
    python gpt_score_mathverse.py --pred_dir results/24-ranked-ablation/full/llava-next-llama3-8b/predictions/

    # Score with a specific model
    python gpt_score_mathverse.py --pred_dir predictions/ --judge gpt-4o-mini
"""
import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


FAIL_MSG = 'Failed to obtain answer via API.'


def get_extract_prompt(prediction):
    """Build GPT prompt to extract answer from model response (VLMEvalKit-style)."""
    examples = [
        "1.\nModel response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\\n\\n(-2, 1)'\nExtracted Answer: (-2, 1)\n",
        "2.\nModel response: 'at those points.\\n\\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\\n\\nD. They give the solutions to the equation $f(t)=g(t)$.\"'\nExtracted Answer: D\n",
        "3.\nModel response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\\\((-4, 1]\\\\).\\n\\nFinal values:\\nDomain: \\\\((-3, 3]\\\\)\\nRange: \\\\((-4, 1]\\\\)'\nExtracted Answer: Domain: \\\\((-3, 3]\\\\)\\nRange: \\\\((-4, 1]\\\\)\n",
        "4.\nModel response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'\nExtracted Answer: null\n",
        "5.\nModel response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\\n\\nd = 17.6 / cos(38°)\\n\\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'\nExtracted answer: 22.3\n",
        "6.\nModel response:  have all the coefficients for the quadratic function:\\\\( f(x) = ax^2 + bx + c \\\\)\\n\\\\( f(x) = -1x^2 - 2x + 1 \\\\)\\n\\nTherefore, the equation for the graphed function \\\\( f \\\\) is:\\n\\\\( f(x) = -x^2 - 2x + 1 \\\\)\"'\nExtracted answer: f(x) = -x^2 - 2x + 1\n",
    ]
    task = "I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n"
    demo = task + '\n\n'.join(examples) + '\n\n'
    test = f"7.\nModel response: '{prediction}'\nExtracted Answer: "
    return demo + test


def get_score_prompt(question, answer, extract):
    """Build GPT prompt to judge if extracted answer matches ground truth."""
    examples = [
        "[Question]: Write the set of numbers represented on the number line in interval notation.\n[Standard Answer]: (-2,1]\n[Model_answer] : Extracted Answer: \\\\((-2, 1)\\\\)\nJudgement: 0\n",
        "[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2√{3}\nC:√{3}\nD:2√{2}\n[Standard Answer]: C\n[Model_answer] : B:2√{3}\nJudgement: 0\n",
        "[Question]: Find the domain and range of the function f using interval notation.\n[Standard Answer]: domain: [-4, 0) and range: (-3, 1]\n[Model_answer] : Range: \\\\((-4, 1]\\\\)\nJudgement: 0\n",
        "[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2√{3}\nC:√{3}\nD:2√{2}\n[Standard Answer]: C\n[Model_answer] : null\nJudgement: 0\n",
    ]
    task = "Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.\nPlease note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.\nIf they are consistent, Judement is 1; if they are different, Judement is 0.\n\n"
    demo = task + '\n\n'.join(examples) + '\n\n'
    test = f"[Question]: {question}\n[Standard Answer]: {answer}\n[Model_answer] : {extract}\nJudgement:"
    return demo + test


def call_gpt(client, model, prompt, retries=5):
    """Call GPT with retries."""
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=i * 0.5,
                max_tokens=256,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
            else:
                return FAIL_MSG


def score_predictions(pred_file, client, judge_model='gpt-4o-mini'):
    """Score predictions in a single file using GPT extraction + scoring."""
    with open(pred_file) as f:
        preds = json.load(f)

    correct = 0
    total = len(preds)
    results = []

    for item in tqdm(preds, desc=f'Scoring {os.path.basename(pred_file)}'):
        prediction = item['prediction']
        answer = item['answer']
        question = item['question']

        # Step 1: GPT extract answer
        extract_prompt = get_extract_prompt(prediction)
        extract = call_gpt(client, judge_model, extract_prompt)

        if extract == FAIL_MSG:
            results.append({**item, 'extract': '', 'score': False})
            continue

        # Shortcut: exact match
        if extract.strip() == answer.strip():
            results.append({**item, 'extract': extract, 'score': True})
            correct += 1
            continue

        # Step 2: GPT judge
        score_prompt = get_score_prompt(question, answer, extract)
        score_str = call_gpt(client, judge_model, score_prompt)

        if score_str.strip() == '1':
            score = True
            correct += 1
        else:
            score = False

        results.append({**item, 'extract': extract, 'score': score})

    accuracy = correct / max(total, 1)
    return {
        'accuracy': round(accuracy, 4),
        'n_questions': total,
        'n_correct': correct,
        'results': results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_file', help='Single prediction JSON file')
    p.add_argument('--pred_dir', help='Directory of prediction JSON files')
    p.add_argument('--output_dir', default=None, help='Output directory (default: same as pred)')
    p.add_argument('--judge', default='gpt-4o-mini')
    p.add_argument('--api_key', default=None, help='OpenAI API key (or set OPENAI_API_KEY)')
    args = p.parse_args()

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        # Try VLMEvalKit .env
        env_path = os.path.join(os.path.dirname(__file__), '..', 'modern_vlms', 'VLMEvalKit', '.env')
        if os.path.isfile(env_path):
            for line in open(env_path):
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
    assert api_key, 'No API key found. Set OPENAI_API_KEY or pass --api_key'

    client = OpenAI(api_key=api_key)

    # Collect prediction files
    pred_files = []
    if args.pred_file:
        pred_files.append(args.pred_file)
    elif args.pred_dir:
        pred_files = sorted(Path(args.pred_dir).glob('pred_*.json'))
    else:
        p.error('Provide --pred_file or --pred_dir')

    print(f'Scoring {len(pred_files)} prediction files with {args.judge}')

    for pf in pred_files:
        pf = str(pf)
        out_dir = args.output_dir or os.path.dirname(pf)
        out_name = os.path.basename(pf).replace('pred_', 'gpt_score_').replace('.json', f'_{args.judge}.json')
        out_path = os.path.join(out_dir, out_name)

        if os.path.isfile(out_path):
            print(f'  [skip] {out_name} — already scored')
            continue

        print(f'\n  Scoring: {os.path.basename(pf)}')
        result = score_predictions(pf, client, args.judge)
        print(f'  Accuracy: {result["accuracy"]:.4f} ({result["n_correct"]}/{result["n_questions"]})')

        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved: {out_path}')


if __name__ == '__main__':
    main()
