#!/usr/bin/env python3
"""
One-time cleanup: fix unit mismatch in existing TriviaQA result JSONs.

Background: VLMEvalKit's TriviaQA evaluator writes accuracy as a 0-1 fraction
(e.g., 0.6388 = 63.88% accuracy), while parse_vlmeval_score in run1_ablation.py
treated all CSV values as percent-scale and divided by 100. As a result:
    - accuracy_pct stored = raw CSV value (e.g., 0.6388) ← should be 63.88
    - accuracy stored     = raw CSV value / 100 (e.g., 0.006388) ← should be 0.6388

This script walks every results/24-ranked-ablation/full/<model>/run1/ directory,
finds run1_*TriviaQA*.json files that have been saved alongside their CSV copies
(_score.csv), and re-reads the CSV to write back correctly normalized values.

POPE and MathVerse JSONs are left untouched because their CSVs already use
percent-scale, so the original parse was correct for those.

Idempotent: detects already-fixed records (accuracy_pct > 1.0) and skips them.
"""

import csv
import glob
import json
import os
import sys


def fix_one_json(json_path):
    """
    Re-read accuracy from sibling _score.csv and patch the JSON.
    Returns:
        'fixed'    — JSON updated in place
        'already'  — already in percent scale (e.g., re-run after deploy)
        'no_csv'   — sibling _score.csv missing, can't fix
        'error'    — exception raised, see stderr
    """
    csv_path = json_path.replace('.json', '_score.csv')
    if not os.path.exists(csv_path):
        return 'no_csv'
    try:
        # Read the raw CSV value
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            for key in ('Overall', 'accuracy'):
                if key in row and row[key] not in ('nan', '', None):
                    raw = float(row[key])
                    break
            else:
                return 'no_csv'

        # Read the JSON
        with open(json_path) as f:
            data = json.load(f)

        # Skip if already-fixed (accuracy_pct already in percent scale)
        if data.get('accuracy_pct', 0) > 1.0:
            return 'already'

        # Apply the same normalization as the patched parse_vlmeval_score
        if raw <= 1.0:
            pct = raw * 100.0
        else:
            pct = raw
        # Patch the JSON in place
        data['accuracy_pct'] = pct
        data['accuracy'] = pct / 100.0  # 0-1 normalized
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return 'fixed'

    except Exception as e:
        print(f'  ERROR fixing {json_path}: {e}', file=sys.stderr)
        return 'error'


def main():
    """
    Walk every model's run1 dir under step 24 results, fix all
    run1_*TriviaQA*.json files in place. Print a per-model summary.
    """
    base = 'results/24-ranked-ablation'
    if not os.path.isdir(base):
        print(f'Not found: {base} — run from project root.', file=sys.stderr)
        sys.exit(1)

    pattern = os.path.join(base, '*', '*', 'run1', 'run1_*TriviaQA*.json')
    json_files = sorted(glob.glob(pattern))
    print(f'Found {len(json_files)} TriviaQA JSON files across all models.\n')

    counters = {'fixed': 0, 'already': 0, 'no_csv': 0, 'error': 0}
    per_model = {}
    for jp in json_files:
        # Extract model name from path: results/24-...full/<model>/run1/...
        parts = jp.split(os.sep)
        model = parts[-3] if len(parts) >= 3 else 'unknown'
        status = fix_one_json(jp)
        counters[status] += 1
        per_model.setdefault(model, {'fixed': 0, 'already': 0, 'no_csv': 0, 'error': 0})
        per_model[model][status] += 1

    print('Per-model summary:')
    for model in sorted(per_model):
        c = per_model[model]
        print(f'  {model:50s}  fixed={c["fixed"]}  already={c["already"]}  '
              f'no_csv={c["no_csv"]}  error={c["error"]}')
    print()
    print('Total:')
    print(f'  Fixed:    {counters["fixed"]} JSONs updated')
    print(f'  Already:  {counters["already"]} JSONs already in percent scale (skipped)')
    print(f'  No CSV:   {counters["no_csv"]} JSONs without sibling _score.csv (cannot fix)')
    print(f'  Errors:   {counters["error"]} JSONs failed to update')


if __name__ == '__main__':
    main()
