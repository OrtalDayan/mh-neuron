#!/usr/bin/env python3
"""
eval_single_model.py — Evaluate a single merged model on all benchmarks.

Called by run_pipeline.sh Phase B to enable parallel evaluation
(1 GPU per model variant).

Usage:
    python eval_single_model.py \
        --state_path results/13-weight-merge/.../model/merged_state_dict.pt \
        --model_type llava-llama3 \
        --vlm_path llava-hf/llama3-llava-next-8b-hf \
        --output_dir results/13-weight-merge/.../text_inject_lambda0.9/ \
        --eval_pope --pope_path data/POPE/... --pope_img_dir data/val2014 \
        --eval_chair --coco_ann_dir data/annotations \
        --eval_vlmevalkit --vlmevalkit_benchmarks MathVista_MINI MMStar ...
"""

import argparse
import json
import os
import sys
import torch


def _resolve_hf_local(model_id):
    """Resolve a HuggingFace model ID to its local cache snapshot path.

    On compute nodes without internet, from_pretrained('org/model')
    fails because it tries to download. This finds the cached copy.
    Returns the original model_id if it's already a local path.
    """
    # Already a local path
    if os.path.isdir(model_id):
        return model_id

    # Try huggingface_hub first
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        pass

    # Manual HF cache search
    for cache_root in [
        os.environ.get('HF_HOME', ''),
        os.environ.get('TRANSFORMERS_CACHE', ''),
        os.path.expanduser('~/.cache/huggingface/hub'),
        os.path.join(os.environ.get('WORK_DIR', '.'),
                     '.cache/huggingface/hub'),
    ]:
        if not cache_root or not os.path.isdir(cache_root):
            continue
        model_dir = os.path.join(cache_root,
                                 'models--' + model_id.replace('/', '--'),
                                 'snapshots')
        if os.path.isdir(model_dir):
            snapshots = sorted(os.listdir(model_dir))
            if snapshots:
                resolved = os.path.join(model_dir, snapshots[-1])
                print(f'  Resolved {model_id} → {resolved}')
                return resolved

    # Fallback: return original (will fail on compute node)
    print(f'  WARNING: Could not resolve {model_id} to local cache')
    return model_id


def main():
    p = argparse.ArgumentParser(description='Evaluate a single merged model')

    # Model
    p.add_argument('--state_path', default=None,
                   help='Path to merged_state_dict.pt (omit for baseline eval)')
    p.add_argument('--model_type', required=True)
    p.add_argument('--vlm_path', required=True,
                   help='HF path to base VLM (for architecture + tokenizer)')
    p.add_argument('--output_dir', required=True,
                   help='Directory to save eval results + HF checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--baseline', action='store_true', default=False,
                   help='Baseline mode: evaluate original VLM without any merge')

    # POPE
    p.add_argument('--eval_pope', action='store_true', default=False)
    p.add_argument('--pope_dir', default='data/POPE/output/coco',
                   help='Directory containing coco_pope_{random,popular,adversarial}.json')
    p.add_argument('--pope_img_dir', default='data/val2014')
    p.add_argument('--n_pope_questions', type=int, default=0,
                   help='Max questions per strategy (0 = all, default for publication)')

    # CHAIR
    p.add_argument('--eval_chair', action='store_true', default=False)
    p.add_argument('--coco_ann_dir', default='data/annotations')
    p.add_argument('--chair_n_images', type=int, default=500)
    p.add_argument('--chair_max_tokens', type=int, default=64)
    p.add_argument('--chair_seeds', default='42,123,7',
                   help='Comma-separated seeds for CHAIR (3 seeds for publication)')

    # VLMEvalKit
    p.add_argument('--eval_vlmevalkit', action='store_true', default=False)
    p.add_argument('--vlmevalkit_benchmarks', nargs='+',
                   default=['MathVista_MINI', 'MathVerse_MINI',
                            'MathVision', 'DynaMath', 'MMStar'],
                   help='VLMEvalKit benchmark names (5 BRV paper benchmarks)')
    p.add_argument('--vlmevalkit_dir', default=None)
    p.add_argument('--vlmevalkit_n_gpus', type=int, default=1)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = os.path.basename(args.output_dir.rstrip('/'))

    # Done flags are per-eval-type so parallel jobs don't conflict:
    #   pope_{strategy}_done.flag — for each POPE strategy (random, popular, adversarial)
    #   chair_seed{N}_done.flag  — for each CHAIR seed
    #   vlmeval_<bench>_done.flag — for each VLMEvalKit benchmark

    print(f'\n{"="*60}')
    print(f'  Evaluating: {run_name}')
    if args.baseline:
        print(f'  Mode: BASELINE (original VLM, no merge)')
    else:
        print(f'  State: {args.state_path}')
    print(f'  Device: {args.device}')
    print(f'{"="*60}\n')

    # Import from neuron_weight_merge (must be on sys.path)
    from neuron_weight_merge import (load_model_for_eval, eval_pope,
                                     eval_chair, save_as_hf_checkpoint,
                                     eval_vlmevalkit)

    # Load state dict (None for baseline)
    merged_state = None
    if not args.baseline and args.state_path:
        print(f'  Loading merged state dict...')
        merged_state = torch.load(args.state_path, map_location='cpu')

    results = {'run_name': run_name}

    # ── POPE (all 3 strategies) ───────────────────────────────────
    POPE_STRATEGIES = ['random', 'popular', 'adversarial']

    if args.eval_pope and os.path.isdir(args.pope_dir):
        pope_all_results = {}
        model_for_pope = None

        for strategy in POPE_STRATEGIES:
            pope_file = os.path.join(args.pope_dir,
                                     f'coco_pope_{strategy}.json')
            strategy_flag = os.path.join(args.output_dir,
                                         f'pope_{strategy}_done.flag')

            if os.path.isfile(strategy_flag):
                print(f'  [skip] POPE {strategy} already done')
                # Load existing results
                strat_result_file = os.path.join(args.output_dir,
                                                  f'pope_{strategy}_results.json')
                if os.path.isfile(strat_result_file):
                    import json as _json
                    pope_all_results[strategy] = _json.load(
                        open(strat_result_file))
                continue

            if not os.path.isfile(pope_file):
                print(f'  [skip] POPE {strategy} file not found: {pope_file}')
                continue

            # Load model once, reuse for all strategies
            if model_for_pope is None:
                print(f'  Loading model for POPE eval...')
                model_for_pope, processor = load_model_for_eval(
                    args.model_type, args.vlm_path,
                    merged_state=merged_state, device=args.device)

            n_q = args.n_pope_questions if args.n_pope_questions > 0 else 0
            n_label = f'{n_q} questions' if n_q > 0 else 'all questions'
            print(f'  POPE {strategy} ({n_label})...')

            pope_res = eval_pope(model_for_pope, processor, pope_file,
                                 args.pope_img_dir,
                                 n_q if n_q > 0 else 99999,
                                 args.model_type, args.device)
            pope_all_results[strategy] = pope_res
            print(f'  POPE {strategy}: acc={pope_res["accuracy"]:.4f} '
                  f'prec={pope_res["precision"]:.4f} '
                  f'F1={pope_res["f1"]:.4f} '
                  f'halluc={pope_res["hallucination_rate"]:.4f}')

            # Save per-strategy results + flag
            with open(os.path.join(args.output_dir,
                      f'pope_{strategy}_results.json'), 'w') as f:
                json.dump(pope_res, f, indent=2, default=str)
            with open(strategy_flag, 'w') as f:
                f.write('done\n')

        if model_for_pope is not None:
            del model_for_pope
            torch.cuda.empty_cache()

        if pope_all_results:
            results['pope'] = pope_all_results
            # Save combined results
            with open(os.path.join(args.output_dir,
                      'pope_results.json'), 'w') as f:
                json.dump(pope_all_results, f, indent=2, default=str)
            print(f'  ✓ POPE done ({len(pope_all_results)} strategies)')

    elif args.eval_pope:
        print(f'  [skip] POPE dir not found: {args.pope_dir}')

    # ── CHAIR (multiple seeds for publication) ─────────────────────
    chair_seeds = [int(s) for s in args.chair_seeds.split(',')]

    if args.eval_chair:
        coco_img_dir = args.pope_img_dir  # val2014
        chair_all_seeds = {}
        model_for_chair = None

        for seed in chair_seeds:
            seed_flag = os.path.join(args.output_dir,
                                     f'chair_seed{seed}_done.flag')
            seed_result_file = os.path.join(args.output_dir,
                                            f'chair_seed{seed}_results.json')

            if os.path.isfile(seed_flag):
                print(f'  [skip] CHAIR seed={seed} already done')
                if os.path.isfile(seed_result_file):
                    chair_all_seeds[seed] = json.load(open(seed_result_file))
                continue

            # Load model once, reuse across seeds
            if model_for_chair is None:
                print(f'  Loading model for CHAIR eval...')
                model_for_chair, processor = load_model_for_eval(
                    args.model_type, args.vlm_path,
                    merged_state=merged_state, device=args.device)

            print(f'  CHAIR eval (seed={seed}, {args.chair_n_images} images)...')
            chair_res = eval_chair(model_for_chair, processor,
                                   args.model_type, args.device,
                                   coco_img_dir, args.coco_ann_dir,
                                   args.chair_n_images,
                                   args.chair_max_tokens, seed=seed)

            if chair_res:
                chair_all_seeds[seed] = chair_res
                print(f'  CHAIR seed={seed}: CHAIRi={chair_res["CHAIRi"]:.4f} '
                      f'CHAIRs={chair_res["CHAIRs"]:.4f}')
                with open(seed_result_file, 'w') as f:
                    json.dump(chair_res, f, indent=2, default=str)

            with open(seed_flag, 'w') as f:
                f.write('done\n')

        if model_for_chair is not None:
            del model_for_chair
            torch.cuda.empty_cache()

        # Aggregate mean ± std across seeds
        if chair_all_seeds:
            import numpy as np
            chairi_vals = [r['CHAIRi'] for r in chair_all_seeds.values()]
            chairs_vals = [r['CHAIRs'] for r in chair_all_seeds.values()]
            chair_agg = {
                'CHAIRi_mean': float(np.mean(chairi_vals)),
                'CHAIRi_std': float(np.std(chairi_vals)),
                'CHAIRs_mean': float(np.mean(chairs_vals)),
                'CHAIRs_std': float(np.std(chairs_vals)),
                'n_seeds': len(chair_all_seeds),
                'per_seed': {str(k): v for k, v in chair_all_seeds.items()},
            }
            results['chair'] = chair_agg
            print(f'  CHAIR ({len(chair_all_seeds)} seeds): '
                  f'CHAIRi={chair_agg["CHAIRi_mean"]:.4f}±{chair_agg["CHAIRi_std"]:.4f} '
                  f'CHAIRs={chair_agg["CHAIRs_mean"]:.4f}±{chair_agg["CHAIRs_std"]:.4f}')
            with open(os.path.join(args.output_dir,
                      'chair_results.json'), 'w') as f:
                json.dump(chair_agg, f, indent=2, default=str)
            print(f'  ✓ CHAIR done')

    # ── VLMEvalKit (one or more benchmarks) ──────────────────────
    if args.eval_vlmevalkit:
        # Check which benchmarks are already done
        benchmarks_to_run = []
        for bench in args.vlmevalkit_benchmarks:
            bench_flag = os.path.join(args.output_dir,
                                      f'vlmeval_{bench}_done.flag')
            if os.path.isfile(bench_flag):
                print(f'  [skip] {bench} already done')
            else:
                benchmarks_to_run.append(bench)

        if benchmarks_to_run:
            vlmeval_work = os.path.join(args.output_dir,
                                        'vlmevalkit_results')

            if args.baseline:
                # Baseline: symlink original VLM to hf_checkpoint so the
                # model_name matches VLMEvalKit's registered 'hf_checkpoint'
                hf_dir = os.path.join(args.output_dir, 'hf_checkpoint')
                if not os.path.lexists(hf_dir):
                    resolved = _resolve_hf_local(args.vlm_path)
                    try:
                        os.symlink(os.path.abspath(resolved), hf_dir)
                        print(f'  Baseline: {args.vlm_path} → {hf_dir}')
                    except FileExistsError:
                        print(f'  Baseline symlink already exists: {hf_dir}')
                else:
                    print(f'  Baseline checkpoint exists: {hf_dir}')
            else:
                hf_dir = os.path.join(args.output_dir, 'hf_checkpoint')
                print(f'  Saving HF checkpoint for VLMEvalKit...')
                save_as_hf_checkpoint(merged_state, args.model_type,
                                      args.vlm_path, hf_dir)

            print(f'  VLMEvalKit ({len(benchmarks_to_run)} benchmarks): '
                  f'{", ".join(benchmarks_to_run)}')
            vlmeval_res = eval_vlmevalkit(
                hf_dir, benchmarks_to_run, args.model_type,
                args.vlmevalkit_n_gpus, args.vlmevalkit_dir,
                vlmeval_work, model_name=run_name)

            if vlmeval_res:
                results['vlmevalkit'] = vlmeval_res
                for bench, metrics in vlmeval_res.items():
                    overall = metrics.get('Overall',
                              metrics.get('Accuracy', ''))
                    if overall:
                        print(f'    {bench}: {overall}')

            # Write per-benchmark done flags + results (only for successful ones)
            for bench in benchmarks_to_run:
                bench_data = vlmeval_res.get(bench, {}) if vlmeval_res else {}

                bench_result = os.path.join(args.output_dir,
                                            f'vlmeval_{bench}_results.json')
                with open(bench_result, 'w') as f:
                    json.dump(bench_data, f, indent=2, default=str)

                # Only write done flag if we got actual results
                if bench_data:
                    bench_flag = os.path.join(args.output_dir,
                                              f'vlmeval_{bench}_done.flag')
                    with open(bench_flag, 'w') as f:
                        f.write('done\n')
                    print(f'    ✓ {bench} done')
                else:
                    print(f'    ✗ {bench} failed (no results, will retry next run)')

            print(f'  ✓ VLMEvalKit done')

    del merged_state

    print(f'\n  ✓ {run_name} complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())