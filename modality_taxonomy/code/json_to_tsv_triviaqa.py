"""
Convert TriviaQA verified-web-dev.json -> ~/LMUData/TriviaQA.tsv

VLMEvalKit expects: index, question, answer, aliases (pipe-delimited), split, category

Usage
-----
    python code/json_to_tsv_triviaqa.py \
        --input  data/triviaqa/qa/verified-web-dev.json \
        --output ~/LMUData/TriviaQA.tsv \
        --split  verified-web-dev \
        [--num 100]   # optional: cap to first N questions (for debugging)

The JSON format we handle is the standard TriviaQA release:
    {"Data": [
        {"QuestionId": "...", "Question": "...",
         "Answer": {"Value": "...", "Aliases": [...], "NormalizedAliases": [...]}
        },
        ...
    ]}
"""
import argparse
import json
import os
import os.path as osp
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to verified-web-dev.json')
    ap.add_argument('--output', required=True, help='output TSV (e.g. ~/LMUData/TriviaQA.tsv)')
    ap.add_argument('--split', default='verified-web-dev',
                    help='split tag to store in the split column')
    ap.add_argument('--category', default='',
                    help='optional category tag')
    ap.add_argument('--num', type=int, default=None,
                    help='optional cap: keep only first N questions')
    args = ap.parse_args()

    input_path = osp.expanduser(args.input)
    output_path = osp.expanduser(args.output)
    assert osp.exists(input_path), f'Input JSON not found: {input_path}'

    with open(input_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    items = payload.get('Data', payload)  # handle both wrapped and raw list
    if args.num is not None:
        items = items[:args.num]

    rows = []
    for i, item in enumerate(items):
        question = item.get('Question', '').strip()
        ans_field = item.get('Answer', {})
        # Canonical answer
        answer = ans_field.get('Value', '') if isinstance(ans_field, dict) else str(ans_field)
        # Aliases — prefer NormalizedAliases when present (more forgiving),
        # fall back to Aliases, then to just [answer]
        aliases = []
        if isinstance(ans_field, dict):
            aliases = ans_field.get('NormalizedAliases') or ans_field.get('Aliases') or []
        if answer and answer not in aliases:
            aliases = [answer] + list(aliases)
        # Pipe-delimited (safe for TSV — no pipes occur in TriviaQA answers)
        aliases_str = '|'.join(str(a) for a in aliases)

        rows.append({
            'index': i,
            'question': question,
            'answer': answer,
            'aliases': aliases_str,
            'split': args.split,
            'category': args.category,
        })

    df = pd.DataFrame(rows, columns=['index', 'question', 'answer', 'aliases', 'split', 'category'])
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print(f'Wrote {len(df)} rows -> {output_path}')
    print(f'Columns: {list(df.columns)}')
    print(df.head(2).to_string())


if __name__ == '__main__':
    main()
