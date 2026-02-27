#!/usr/bin/env python3
"""Summarize all dose-response ablation results into one table.

Usage:
    # Summarize all three ranking methods (default dirs):
    python summarize_dose_results.py

    # Explicit dirs:
    python summarize_dose_results.py \
        --results_dir results/taxonomy/full/llava-1.5-7b/ablation_label \
                      results/taxonomy/full/llava-1.5-7b/ablation_cett \
                      results/taxonomy/full/llava-1.5-7b/ablation_combined

    # Save CSV:
    python summarize_dose_results.py --csv results/dose_summary.csv

    # Only show label and cett:
    python summarize_dose_results.py --ranking label cett
"""

import argparse                                                             # Line 1: CLI argument parsing
import json                                                                 # Line 2: read JSON result files
import os                                                                   # Line 3: directory traversal
import glob                                                                 # Line 4: file pattern matching


def parse_condition_file(path):
    """Parse a single ablation_condition_*.json file.

    Args:
        path: full path to the JSON file

    Returns:
        dict with ranking, taxonomy, type, dose_pct, and all metrics,
        or None if file is invalid.
    """
    with open(path) as f:                                                   # Line 5: open JSON
        data = json.load(f)                                                 # Line 6: parse JSON

    condition = data.get('condition', '')                                    # Line 7: e.g. "ablate_vis_dose1.0pct_pmbt_cett"
    n_ablated = data.get('n_neurons_ablated', 0)                            # Line 8: total neurons removed
    pct_total = data.get('pct_total_neurons', None)                         # Line 9: % of total (if stored)

    # ── Determine ranking method ──
    # Line 10: First try the explicit JSON field (present in newer runs)
    ranking_method = data.get('ranking_method', None)

    # Line 11: Fallback: infer from the parent directory name
    #          ablation_label/ → label, ablation_cett/ → cett, etc.
    if ranking_method is None:
        for part in path.replace('\\', '/').split('/'):                      # Line 12: check each path component
            if part.startswith('ablation_'):
                suffix = part.replace('ablation_', '').replace('_test', '')
                if suffix in ('label', 'cett', 'combined'):
                    ranking_method = suffix
                    break
        if ranking_method is None:
            ranking_method = 'label'                                        # Line 13: default

    # ── Determine taxonomy from condition suffix ──
    if '_pmbt' in condition:                                                # Line 14: check for pmbt in name
        taxonomy = 'pmbt'
    elif '_ft' in condition:                                                # Line 15: check for ft in name
        taxonomy = 'ft'
    else:
        # Line 16: Infer from directory path (pmbt/ vs ft/ subdirectory)
        if '/pmbt/' in path:
            taxonomy = 'pmbt'
        elif '/ft/' in path:
            taxonomy = 'ft'
        else:
            taxonomy = 'ft'                                                 # Line 17: default to ft (legacy)

    # ── Determine neuron type and dose ──
    dose_pct = data.get('dose_pct', None)                                   # Line 18: dose percentage
    dose_type = data.get('dose_type', None)                                 # Line 19: neuron type

    if dose_type is None:                                                   # Line 20: fallback — parse name
        if 'baseline' in condition:
            dose_type = 'baseline'
        elif 'ablate_vis' in condition:
            dose_type = 'visual'
        elif 'ablate_text' in condition:
            dose_type = 'text'
        elif 'ablate_multi' in condition:
            dose_type = 'multimodal'
        elif 'ablate_unknown' in condition:
            dose_type = 'unknown'
        elif 'random' in condition:
            dose_type = 'random'
        else:
            dose_type = condition

    # ── Extract metrics ──
    pope = data.get('pope', {})                                             # Line 21: POPE metrics
    chair = data.get('chair', {})                                           # Line 22: CHAIR metrics
    vqa = data.get('vqav2', {})                                             # Line 23: VQAv2 metrics
    textqa = data.get('textqa', {})                                         # Line 24: TextQA metrics
    vqa_perc = data.get('vqa_perception', {})                               # Line 25: VQA perception split
    vqa_know = data.get('vqa_knowledge', {})                                # Line 26: VQA knowledge split

    return {                                                                # Line 27: unified record
        'ranking': ranking_method,
        'taxonomy': taxonomy,
        'type': dose_type if dose_type else 'baseline',
        'dose_pct': dose_pct if dose_pct else 0.0,
        'n_ablated': n_ablated,
        'pct_total': pct_total if pct_total else 0.0,
        'pope_acc': pope.get('accuracy', None),
        'pope_f1': pope.get('f1', None),
        'pope_yes': pope.get('yes_ratio', None),
        'chair_i': chair.get('chair_i', None),
        'chair_s': chair.get('chair_s', None),
        'vqav2_acc': vqa.get('accuracy', None),
        'vqa_perc_acc': vqa_perc.get('accuracy', None),
        'vqa_know_acc': vqa_know.get('accuracy', None),
        'textqa_acc': textqa.get('accuracy', None),
        'time_sec': data.get('time_sec', None),
    }


def find_result_files(results_dirs):
    """Find all ablation_condition_*.json files recursively."""
    all_files = set()                                                       # Line 28: deduplicate
    for d in results_dirs:
        pattern = os.path.join(d, '**', 'ablation_condition_*.json')        # Line 29: recursive glob
        all_files.update(glob.glob(pattern, recursive=True))
    return sorted(all_files)                                                # Line 30: sorted list


def main():
    parser = argparse.ArgumentParser(
        description='Summarize dose-response ablation results across '
                    'ranking methods (label, cett, combined)')
    parser.add_argument('--results_dir', nargs='+',                         # Line 31: multiple dirs
                        default=[
                            'results/taxonomy/full/llava-1.5-7b/ablation_label',
                            'results/taxonomy/full/llava-1.5-7b/ablation_cett',
                            'results/taxonomy/full/llava-1.5-7b/ablation_combined',
                        ],
                        help='Root directories to search (accepts multiple)')
    parser.add_argument('--csv', default=None,                              # Line 32: CSV output
                        help='Save results to CSV file')
    parser.add_argument('--ranking', nargs='*', default=None,               # Line 33: filter
                        help='Only show specific rankings (e.g. --ranking label cett)')
    args = parser.parse_args()

    files = find_result_files(args.results_dir)                             # Line 34: find all JSONs
    if not files:
        print('No ablation_condition_*.json files found in:')
        for d in args.results_dir:
            tag = '\u2713' if os.path.isdir(d) else '\u2717 NOT FOUND'
            print(f'  {d}  {tag}')
        return

    # Parse all files
    rows = []                                                               # Line 35: accumulate
    for f in files:
        try:
            row = parse_condition_file(f)
            if row:
                rows.append(row)
        except Exception as e:
            print(f'  [warn] Failed to parse {f}: {e}')

    # Filter by ranking method
    if args.ranking:                                                        # Line 36: optional filter
        rows = [r for r in rows if r['ranking'] in args.ranking]

    if not rows:
        print('No results found after filtering.')
        return

    # Sort: ranking → taxonomy → type → dose_pct
    type_order = {'baseline': 0, 'visual': 1, 'text': 2,
                  'multimodal': 3, 'unknown': 4, 'random': 5}
    ranking_order = {'label': 0, 'cett': 1, 'combined': 2}
    rows.sort(key=lambda r: (                                               # Line 37: sort
        ranking_order.get(r['ranking'], 99),
        r['taxonomy'],
        type_order.get(r['type'], 99),
        r['dose_pct'] or 0.0
    ))

    # Find baselines — keyed by (ranking, taxonomy)
    baselines = {}                                                          # Line 38: baseline lookup
    for r in rows:
        if r['type'] == 'baseline':
            baselines[(r['ranking'], r['taxonomy'])] = r

    # Check if VQA split data exists
    has_vqa_split = any(r['vqa_perc_acc'] is not None for r in rows)

    # Build header
    _D = '\u0394'  # Δ delta symbol
    _M = '\u2014'  # — em dash
    hdr = [f'{"Rank":>8s}', f'{"Tax":>4s}', f'{"Type":>12s}',
           f'{"Dose%":>5s}', f'{"#Abl":>6s}',
           f'{"POPE":>6s}', f'{_D + "POPE":>6s}',
           f'{"CHR_i":>6s}', f'{_D + "CHR":>6s}',
           f'{"VQA2":>6s}', f'{_D + "VQA2":>6s}']
    if has_vqa_split:
        hdr += [f'{"VPerc":>6s}', f'{_D + "Perc":>6s}',
                f'{"VKnow":>6s}', f'{_D + "Know":>6s}']
    hdr += [f'{"TxtQA":>6s}', f'{_D + "Txt":>6s}']
    header = '  '.join(hdr)

    print(f'\n{"="*len(header)}')
    print('DOSE-RESPONSE ABLATION SUMMARY')
    print(f'{"="*len(header)}')
    print(f'  Found {len(rows)} conditions from {len(files)} files')
    print(f'  Rankings:   {sorted(set(r["ranking"] for r in rows))}')
    print(f'  Taxonomies: {sorted(set(r["taxonomy"] for r in rows))}')
    print(f'  Baselines:  {list(baselines.keys())}')
    if has_vqa_split:
        print(f'  VQA split:  YES (perception + knowledge columns)')
    print(f'{"="*len(header)}\n')
    print(header)
    print('-' * len(header))

    prev_ranking = prev_tax = prev_type = None
    for r in rows:
        if prev_ranking and r['ranking'] != prev_ranking:
            print(f'{"="*len(header)}')
        elif prev_tax and r['taxonomy'] != prev_tax:
            print(f'{"-"*len(header)}')
        elif prev_type and r['type'] != prev_type:
            print()
        prev_ranking, prev_tax, prev_type = r['ranking'], r['taxonomy'], r['type']

        bl = baselines.get((r['ranking'], r['taxonomy']), {})

        def delta(val, key):
            bv = bl.get(key) if isinstance(bl, dict) else None
            return val - bv if val is not None and bv is not None else None

        d_pope  = delta(r['pope_acc'],     'pope_acc')
        d_chair = delta(r['chair_i'],      'chair_i')
        d_vqa   = delta(r['vqav2_acc'],    'vqav2_acc')
        d_perc  = delta(r['vqa_perc_acc'], 'vqa_perc_acc')
        d_know  = delta(r['vqa_know_acc'], 'vqa_know_acc')
        d_txt   = delta(r['textqa_acc'],   'textqa_acc')

        def fmt(v, w=6):
            return f'{v:>{w}.4f}' if v is not None else _M.rjust(w)
        def fmtd(v, w=6):
            if v is None: return _M.rjust(w)
            s = '+' if v >= 0 else ''
            return f'{s}{v:>.4f}'[:w].rjust(w)

        dose_s = f'{r["dose_pct"]:.1f}' if r['dose_pct'] else '  ' + _M
        n_s = f'{r["n_ablated"]:>6d}' if r['n_ablated'] else '     0'

        parts = [f'{r["ranking"]:>8s}', f'{r["taxonomy"]:>4s}',
                 f'{r["type"]:>12s}', f'{dose_s:>5s}', f'{n_s}',
                 fmt(r['pope_acc']), fmtd(d_pope),
                 fmt(r['chair_i']), fmtd(d_chair),
                 fmt(r['vqav2_acc']), fmtd(d_vqa)]
        if has_vqa_split:
            parts += [fmt(r['vqa_perc_acc']), fmtd(d_perc),
                      fmt(r['vqa_know_acc']), fmtd(d_know)]
        parts += [fmt(r['textqa_acc']), fmtd(d_txt)]
        print('  '.join(parts))

    print(f'\n{"="*len(header)}')

    # Save CSV
    if args.csv:
        import csv
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved CSV \u2192 {args.csv}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# """Summarize all dose-response ablation results into one table.

# Usage:
#     python summarize_dose_results.py --results_dir results/taxonomy/full/llava-1.5-7b/ablation

# Reads all ablation_condition_*.json files from pmbt/dose_response/ and
# ft/dose_response/ subdirectories, extracts key metrics, and prints a
# unified table sorted by taxonomy → type → dose.
# """

# import argparse                                                             # Line 1: CLI argument parsing
# import json                                                                 # Line 2: read JSON result files
# import os                                                                   # Line 3: directory traversal
# import glob                                                                 # Line 4: file pattern matching


# def parse_condition_file(path):
#     """Parse a single ablation_condition_*.json file.

#     Args:
#         path: full path to the JSON file

#     Returns:
#         dict with taxonomy, type, dose_pct, and all metrics,
#         or None if file is invalid.
#     """
#     with open(path) as f:                                                   # Line 5: open JSON
#         data = json.load(f)                                                 # Line 6: parse JSON

#     condition = data.get('condition', '')                                    # Line 7: e.g. "ablate_vis_dose1.0pct_pmbt"
#     n_ablated = data.get('n_neurons_ablated', 0)                            # Line 8: total neurons removed
#     pct_total = data.get('pct_total_neurons', None)                         # Line 9: % of total (if stored)

#     # Determine taxonomy from suffix
#     if condition.endswith('_pmbt'):                                          # Line 10: check suffix
#         taxonomy = 'pmbt'
#     elif '_pmbt' not in condition and '_ft' not in condition:
#         # FT results have no suffix (label_suffix = "")
#         taxonomy = 'ft'
#     else:
#         taxonomy = 'unknown'

#     # Determine type and dose from condition name
#     dose_pct = data.get('dose_pct', None)                                   # Line 11: dose percentage
#     dose_type = data.get('dose_type', None)                                 # Line 12: neuron type

#     # Fallback: parse from condition string
#     if dose_type is None:                                                   # Line 13: extract type from name
#         if 'baseline' in condition:
#             dose_type = 'baseline'
#         elif 'ablate_vis' in condition:
#             dose_type = 'visual'
#         elif 'ablate_text' in condition:
#             dose_type = 'text'
#         elif 'ablate_multi' in condition:
#             dose_type = 'multimodal'
#         elif 'ablate_unknown' in condition:
#             dose_type = 'unknown'
#         elif 'random' in condition:
#             dose_type = 'random'
#         else:
#             dose_type = condition

#     # Extract metrics
#     pope = data.get('pope', {})                                             # Line 14: POPE metrics dict
#     chair = data.get('chair', {})                                           # Line 15: CHAIR metrics dict
#     vqa = data.get('vqav2', {})                                             # Line 16: VQAv2 metrics dict
#     textqa = data.get('textqa', {})                                         # Line 17: TextQA metrics dict

#     return {                                                                # Line 18: return unified record
#         'taxonomy': taxonomy,
#         'type': dose_type if dose_type else 'baseline',
#         'dose_pct': dose_pct if dose_pct else 0.0,
#         'n_ablated': n_ablated,
#         'pct_total': pct_total,
#         'pope_acc': pope.get('accuracy', None),
#         'pope_f1': pope.get('f1', None),
#         'pope_yes': pope.get('yes_ratio', None),
#         'chair_i': chair.get('chair_i', None),
#         'chair_s': chair.get('chair_s', None),
#         'vqav2_acc': vqa.get('accuracy', None),
#         'textqa_acc': textqa.get('accuracy', None),
#         'time_sec': data.get('time_sec', None),
#     }


# def find_result_files(results_dir):
#     """Find all ablation_condition_*.json files recursively.

#     Args:
#         results_dir: root directory to search

#     Returns:
#         list of file paths
#     """
#     pattern = os.path.join(results_dir, '**', 'ablation_condition_*.json')  # Line 19: recursive glob
#     return sorted(glob.glob(pattern, recursive=True))                       # Line 20: return sorted list


# def main():
#     parser = argparse.ArgumentParser(description='Summarize dose-response results')
#     parser.add_argument('--results_dir',                                    # Line 21: root results directory
#                         default='results/taxonomy/full/llava-1.5-7b/ablation',
#                         help='Root directory containing pmbt/ and ft/ subdirs')
#     parser.add_argument('--csv', default=None,                              # Line 22: optional CSV output
#                         help='Save results to CSV file')
#     args = parser.parse_args()

#     files = find_result_files(args.results_dir)                             # Line 23: find all JSON files
#     if not files:
#         print(f'No ablation_condition_*.json files found in {args.results_dir}')
#         return

#     # Parse all files
#     rows = []                                                               # Line 24: accumulate parsed results
#     for f in files:
#         try:
#             row = parse_condition_file(f)                                   # Line 25: parse each file
#             if row:
#                 rows.append(row)
#         except Exception as e:
#             print(f'  [warn] Failed to parse {f}: {e}')

#     # Sort: taxonomy → type → dose_pct
#     type_order = {'baseline': 0, 'visual': 1, 'text': 2,                   # Line 26: define sort order
#                   'multimodal': 3, 'unknown': 4, 'random': 5}
#     rows.sort(key=lambda r: (                                               # Line 27: sort rows
#         r['taxonomy'],
#         type_order.get(r['type'], 99),
#         r['dose_pct'] or 0.0
#     ))

#     # Find baselines for delta computation
#     baselines = {}                                                          # Line 28: store baselines per taxonomy
#     for r in rows:
#         if r['type'] == 'baseline':
#             baselines[r['taxonomy']] = r

#     # Print header
#     header = (f'{"Tax":>4s}  {"Type":>12s}  {"Dose%":>5s}  {"#Abl":>6s}  '  # Line 29: format header
#               f'{"POPE":>6s}  {"ΔPOPE":>6s}  '
#               f'{"CHR_i":>6s}  {"ΔCHR":>6s}  '
#               f'{"VQA2":>6s}  {"ΔVQA2":>6s}  '
#               f'{"TxtQA":>6s}  {"ΔTxt":>6s}')
#     print(f'\n{"="*len(header)}')
#     print('DOSE-RESPONSE ABLATION SUMMARY')
#     print(f'{"="*len(header)}')
#     print(f'  Found {len(rows)} conditions from {len(files)} files')
#     print(f'  Baselines: {list(baselines.keys())}')
#     print(f'{"="*len(header)}\n')
#     print(header)
#     print('-' * len(header))

#     prev_tax = None                                                         # Line 30: track taxonomy for separator
#     prev_type = None                                                        # Line 31: track type for separator
#     for r in rows:
#         # Print separator between taxonomies
#         if prev_tax and r['taxonomy'] != prev_tax:                          # Line 32: taxonomy separator
#             print(f'{"="*len(header)}')
#         elif prev_type and r['type'] != prev_type:                          # Line 33: type separator
#             print()
#         prev_tax = r['taxonomy']
#         prev_type = r['type']

#         # Compute deltas vs baseline
#         bl = baselines.get(r['taxonomy'], {})                               # Line 34: get baseline for this tax
#         d_pope = (r['pope_acc'] - bl.get('pope_acc', 0)                     # Line 35: POPE delta
#                   if r['pope_acc'] is not None and bl.get('pope_acc') else None)
#         d_chair = (r['chair_i'] - bl.get('chair_i', 0)                     # Line 36: CHAIR delta
#                    if r['chair_i'] is not None and bl.get('chair_i') else None)
#         d_vqa = (r['vqav2_acc'] - bl.get('vqav2_acc', 0)                   # Line 37: VQAv2 delta
#                  if r['vqav2_acc'] is not None and bl.get('vqav2_acc') else None)
#         d_txt = (r['textqa_acc'] - bl.get('textqa_acc', 0)                  # Line 38: TextQA delta
#                  if r['textqa_acc'] is not None and bl.get('textqa_acc') else None)

#         # Format values
#         def fmt(v, w=6):                                                    # Line 39: format helper
#             return f'{v:>{w}.4f}' if v is not None else f'{"—":>{w}s}'
#         def fmtd(v, w=6):                                                   # Line 40: format delta with sign
#             if v is None: return f'{"—":>{w}s}'
#             sign = '+' if v >= 0 else ''
#             return f'{sign}{v:>.4f}'[:w].rjust(w)

#         dose_str = f'{r["dose_pct"]:.1f}' if r['dose_pct'] else '  —'     # Line 41: dose string
#         n_str = f'{r["n_ablated"]:>6d}' if r['n_ablated'] else '     0'    # Line 42: neuron count

#         print(f'{r["taxonomy"]:>4s}  {r["type"]:>12s}  {dose_str:>5s}  {n_str}  '
#               f'{fmt(r["pope_acc"])}  {fmtd(d_pope)}  '
#               f'{fmt(r["chair_i"])}  {fmtd(d_chair)}  '
#               f'{fmt(r["vqav2_acc"])}  {fmtd(d_vqa)}  '
#               f'{fmt(r["textqa_acc"])}  {fmtd(d_txt)}')

#     print(f'\n{"="*len(header)}')

#     # Save CSV if requested
#     if args.csv:                                                            # Line 43: optional CSV export
#         import csv
#         with open(args.csv, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=rows[0].keys())           # Line 44: write CSV header
#             writer.writeheader()
#             writer.writerows(rows)                                          # Line 45: write all rows
#         print(f'\nSaved CSV → {args.csv}')


# if __name__ == '__main__':
#     main()
