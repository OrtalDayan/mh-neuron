"""
merge_halluc_scores.py — Merge per-layer ablation scores from halluc_score_neurons.py
======================================================================================

When halluc_score_neurons.py is run with --layer_start / --layer_end (per-layer
LSF mode), each job writes its own files:
    ablation_scores_layers{S}-{E}.json
    ablation_scores_tqa_layers{S}-{E}.json    (if --halluc_triviaqa was set)
    baseline_results_layers{S}-{E}.json

This script:
  1. Discovers all per-layer files in <output_dir>
  2. Merges them into the standard names:
        ablation_scores.json       (POPE ΔH per neuron)
        ablation_scores_tqa.json   (TriviaQA ΔH_TQA per neuron, if present)
        baseline_results.json      (one representative baseline)
  3. Optionally invokes halluc_score_neurons.py --skip_ablation to run enrichment
     on the merged scores.

Usage:
    python merge_halluc_scores.py \\
        --output_dir results/10-halluc_scores/full/idefics2-8b \\
        [--run_enrichment]                       # run halluc_score_neurons.py
                                                 #   --skip_ablation after merge

When --run_enrichment is set, the user must also pass --halluc_score_script
and the same --model_type, --model_path, --model_name, --n_layers,
--n_neurons, --label_dir, --pope_path, --pope_img_dir args that the
per-layer jobs were submitted with (so enrichment uses the same labels).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys


def find_per_layer_files(output_dir, pattern_prefix):
    """Find all per-layer JSON files matching a given prefix.

    Args:
        output_dir: directory to search
        pattern_prefix: e.g. 'ablation_scores' or 'ablation_scores_tqa'

    Returns:
        List of (filepath, layer_start, layer_end) tuples, sorted by start.
    """
    pattern = os.path.join(output_dir, f'{pattern_prefix}_layers*.json')
    files = glob.glob(pattern)
    out = []
    for fp in files:
        m = re.search(r'_layers(\d+)-(\d+)\.json$', os.path.basename(fp))
        if m:
            out.append((fp, int(m.group(1)), int(m.group(2))))
    out.sort(key=lambda x: x[1])
    return out


def merge_score_files(file_list, output_path, label):
    """Merge per-layer score JSONs (dicts of 'layer_neuron' -> float) into one.

    Args:
        file_list: list of (filepath, start, end) tuples
        output_path: path to write merged JSON
        label: e.g. 'POPE' or 'TriviaQA' for log messages

    Returns:
        Total number of merged neuron entries.
    """
    if not file_list:
        return 0
    merged = {}
    seen_keys = set()
    layer_coverage = []
    for fp, s, e in file_list:
        with open(fp) as f:
            d = json.load(f)
        layer_coverage.append((s, e))
        n_added = 0
        for k, v in d.items():
            if k in seen_keys:
                # Should not happen if layer ranges don't overlap
                print(f'  WARN: duplicate key {k} in {os.path.basename(fp)}; '
                      f'overwriting')
            merged[k] = v
            seen_keys.add(k)
            n_added += 1
        print(f'  [{label}] {os.path.basename(fp)}: {n_added:,} neurons '
              f'(layers {s}-{e-1})')
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'  [{label}] Merged {len(merged):,} neurons → {output_path}')
    return len(merged), layer_coverage


def merge_baselines(file_list, output_path):
    """Merge baseline_results_layers*.json files.

    Each file contains the same baseline (computed from the same contrastive
    set on each per-layer worker), so we just take the first one as canonical.
    Sanity-check that they agree.
    """
    if not file_list:
        return None
    first_fp = file_list[0][0]
    with open(first_fp) as f:
        canonical = json.load(f)

    # Sanity check: hallucination_rate should match across all per-layer baselines
    base_hr = canonical.get('hallucination_rate')
    mismatches = 0
    for fp, s, e in file_list[1:]:
        with open(fp) as f:
            d = json.load(f)
        if d.get('hallucination_rate') != base_hr:
            mismatches += 1
    if mismatches > 0:
        print(f'  WARN: {mismatches} per-layer baselines disagree on '
              f'hallucination_rate. Using first ({base_hr}).')
    with open(output_path, 'w') as f:
        json.dump(canonical, f, indent=2)
    print(f'  Baseline merged → {output_path} (hr={base_hr})')
    return canonical


def check_layer_coverage(coverage, expected_n_layers):
    """Verify that merged per-layer files cover [0, expected_n_layers) without gaps."""
    if not coverage:
        return False, 'No per-layer files found'
    coverage = sorted(coverage)
    covered = set()
    for s, e in coverage:
        for layer in range(s, e):
            if layer in covered:
                return False, f'Layer {layer} covered by multiple files'
            covered.add(layer)
    expected = set(range(expected_n_layers))
    missing = expected - covered
    extra = covered - expected
    if missing:
        return False, f'Missing layers: {sorted(missing)}'
    if extra:
        return False, f'Unexpected layers beyond [0,{expected_n_layers}): {sorted(extra)}'
    return True, f'All {expected_n_layers} layers covered'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output_dir', required=True,
                    help='Directory containing per-layer ablation_scores_layers*.json '
                         'files. Merged outputs are written to the same directory.')
    ap.add_argument('--n_layers', type=int, default=None,
                    help='Expected total number of layers for coverage check '
                         '(e.g. 32 for 8B LLaMA-class, 28 for Qwen2). '
                         'If not set, coverage check is skipped.')
    ap.add_argument('--run_enrichment', action='store_true',
                    help='After merging, invoke halluc_score_neurons.py --skip_ablation '
                         'to run Phase 2 enrichment analysis on the merged scores.')
    ap.add_argument('--halluc_score_script',
                    default='code/halluc_score_neurons.py',
                    help='Path to halluc_score_neurons.py (used when --run_enrichment).')
    ap.add_argument('--python', default=sys.executable,
                    help='Python interpreter to use for the enrichment subprocess.')
    # Pass-through args for the enrichment subprocess (only used when --run_enrichment)
    ap.add_argument('--enrichment_args', nargs=argparse.REMAINDER,
                    help='All remaining args are passed verbatim to halluc_score_neurons.py '
                         '(e.g. --model_type idefics2 --model_path ... --label_dir ... '
                         '--pope_path ... etc.).')
    args = ap.parse_args()

    print(f'\n{"="*60}')
    print(f'MERGE PER-LAYER HALLUCINATION SCORES')
    print(f'  Directory: {args.output_dir}')
    print(f'{"="*60}\n')

    # ── Discover per-layer files ──
    pope_files = find_per_layer_files(args.output_dir, 'ablation_scores')
    # Filter out the TQA files (which also match 'ablation_scores' prefix)
    pope_files = [(fp, s, e) for (fp, s, e) in pope_files
                  if '_tqa_' not in os.path.basename(fp)]
    tqa_files = find_per_layer_files(args.output_dir, 'ablation_scores_tqa')
    bl_files = find_per_layer_files(args.output_dir, 'baseline_results')

    print(f'Found {len(pope_files)} POPE shards, {len(tqa_files)} TQA shards, '
          f'{len(bl_files)} baseline shards.\n')

    if not pope_files:
        print(f'ERROR: No ablation_scores_layers*.json files found in {args.output_dir}')
        sys.exit(1)

    # ── Merge POPE ablation scores ──
    pope_out = os.path.join(args.output_dir, 'ablation_scores.json')
    n_pope, pope_coverage = merge_score_files(pope_files, pope_out, 'POPE')

    # ── Merge TQA ablation scores ──
    if tqa_files:
        tqa_out = os.path.join(args.output_dir, 'ablation_scores_tqa.json')
        n_tqa, _ = merge_score_files(tqa_files, tqa_out, 'TriviaQA')
    else:
        print('No TQA per-layer files — skipping TQA merge.')

    # ── Merge baselines ──
    if bl_files:
        bl_out = os.path.join(args.output_dir, 'baseline_results.json')
        merge_baselines(bl_files, bl_out)
    else:
        print('No baseline per-layer files — skipping baseline merge.')

    # ── Coverage sanity check ──
    if args.n_layers is not None:
        ok, msg = check_layer_coverage(pope_coverage, args.n_layers)
        symbol = '✓' if ok else '✗'
        print(f'\n{symbol} Coverage check: {msg}')
        if not ok:
            print(f'WARNING: enrichment results will be incomplete '
                  f'until missing per-layer jobs are re-submitted.')

    # ── Optionally run enrichment ──
    if args.run_enrichment:
        if not args.enrichment_args:
            print(f'ERROR: --run_enrichment requires --enrichment_args followed by '
                  f'all the same --model_type/--model_path/--label_dir/etc. flags '
                  f'used in per-layer jobs.')
            sys.exit(1)
        cmd = [args.python, args.halluc_score_script,
               '--skip_ablation',
               '--ablation_scores', pope_out,
               '--output_dir', args.output_dir]
        # Forward the merged TriviaQA ΔH_TQA so the enrichment subprocess also
        # emits enrichment_results_tqa.json (the text-side double-dissociation).
        _tqa_out = os.path.join(args.output_dir, 'ablation_scores_tqa.json')
        if os.path.isfile(_tqa_out):
            cmd.extend(['--ablation_scores_tqa', _tqa_out])
        cmd.extend(args.enrichment_args)
        print(f'\n{"="*60}')
        print(f'INVOKING ENRICHMENT')
        print(f'  Command: {" ".join(cmd)}')
        print(f'{"="*60}\n')
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f'ERROR: enrichment subprocess exited with code {rc}')
            sys.exit(rc)

    print(f'\nDone. Merged outputs in {args.output_dir}')


if __name__ == '__main__':
    main()
