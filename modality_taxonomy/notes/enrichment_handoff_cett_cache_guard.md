# Handoff — add "skip if cached" guard to `build_contrastive_pope_set`

**Audience:** a Claude Code session running in the enrichment-analysis worktree (Session A — `enrichment-checkpoint` branch). That session has no context from the step-weight-merging session where this handoff was prepared, so this document is self-contained.

## Task

Add an early-return guard at the top of `build_contrastive_pope_set(args)` in `code/halluc_score_neurons.py` so that step-10 re-launches skip the contrastive POPE preprocessing pass when its outputs (`cett_diff_scores.json`, `contrastive_pope.jsonl`, `contrastive_stats.json`) already exist in `<args.output_dir>/`.

## Context

The step-weight-merging session just killed 560 step-10 (halluc_score) jobs on `waic-risk` that had broken `-w done(L0)` dependencies (LSF rotated the L0 parent jobs out of history after May 29). The L0 outputs for all 27 (9 models × 3 variants: `gate`, `gate_up`, `ft`) are already on disk under `results/10-halluc_scores/full/{model}/{variant}/`. Step-10 re-launches per (model × variant) are imminent.

Every step-10 re-launch currently re-runs `build_contrastive_pope_set(args)`. That function does a full POPE-scoring forward pass to compute CETT-diff and writes three artefacts:
- `<output_dir>/cett_diff_scores.json`
- `<output_dir>/contrastive_pope.jsonl`
- `<output_dir>/contrastive_stats.json`

All 27 of those triples already exist from the prior L0 run, so the preprocessing pass is wasted GPU work — about 1 GPU-hour total across the 27 re-runs, plus a model load per re-run that holds a GPU slot for ~30 s.

## File to edit

`code/halluc_score_neurons.py` — function `build_contrastive_pope_set(args)`.

## Function signature and current behaviour (line numbers approximate — verify against current head)

- Called from `main()` at line **2493**: `contrastive_path, n_clean, cett_diff_scores = build_contrastive_pope_set(args)`.
- Returns: `(contrastive_path: str, n_clean: int, cett_diff_scores: dict[tuple[int,int], float])`.
- Inside the function:
  - Runs POPE inference, splits results into `n_halluc` / `n_correct` groups, accumulates per-layer × per-neuron CETT sums.
  - Lines **862–877**: builds the `cett_diff` dict and writes it as `cett_diff_scores.json` at `<args.output_dir>/cett_diff_scores.json` with keys serialised as `f'{layer}_{neuron}'`.
  - Earlier in the same function it also writes `contrastive_pope.jsonl` (the filtered POPE question set) and `contrastive_stats.json` to the same directory.

## Required guard

At the **top** of `build_contrastive_pope_set`, before any model loading or forward passes, check whether the three artefacts already exist. If they do (and the user hasn't explicitly requested a restart), load them and return early.

Suggested implementation — adapt to match the script's local style:

```python
def build_contrastive_pope_set(args):
    cett_diff_path     = os.path.join(args.output_dir, 'cett_diff_scores.json')
    contrastive_path   = os.path.join(args.output_dir, 'contrastive_pope.jsonl')
    contrastive_stats  = os.path.join(args.output_dir, 'contrastive_stats.json')

    if (getattr(args, 'resume', True)
            and os.path.exists(cett_diff_path)
            and os.path.exists(contrastive_path)
            and os.path.exists(contrastive_stats)):
        with open(cett_diff_path) as f:
            cett_serializable = json.load(f)
        cett_diff_scores = {
            tuple(int(x) for x in k.split('_')): v
            for k, v in cett_serializable.items()
        }
        with open(contrastive_stats) as f:
            n_clean = json.load(f).get('n_clean')
        if n_clean is None:
            n_clean = sum(1 for _ in open(contrastive_path))
        print(f'[resume] Reusing cached contrastive POPE preprocessing '
              f'from {args.output_dir} — skipping CETT-diff GPU pass '
              f'({len(cett_diff_scores)} neurons, n_clean={n_clean})')
        return contrastive_path, n_clean, cett_diff_scores

    # ── original body of build_contrastive_pope_set continues here ──
    ...
```

## Things to verify / preserve

1. **`--resume` is already the default** (`add_argument('--resume', default=True)`, line ~216). The corresponding opt-out flag is `--restart` (line ~220) which deletes checkpoints up-front. The guard above predicates on `getattr(args, 'resume', True)` — confirm the exact flag name (`resume` vs `restart` vs both) matches what the script uses elsewhere.
2. **`n_clean` provenance.** Check whether `contrastive_stats.json` already contains a field named `n_clean` (or equivalent). If not, count lines in `contrastive_pope.jsonl` — both are valid recoveries.
3. **Key deserialisation must be lossless.** The original write at lines 874–875 uses `f'{k[0]}_{k[1]}': v`. The reload code must split on `_` and `int()` both halves. Verify the keys really look like `"5_1234"` only — if any layer-name scheme contains `_`, the split is brittle.
4. **No partial state.** Require *all three* files to exist before triggering the skip. If any is missing, re-run the full preprocessing (don't try to patch in a partial cache).
5. **Guard placement is critical.** Top of the function, before any model load or POPE inference. Not inside the per-layer ablation loop.

## Verification plan after the edit

1. Pick one (model, variant) where L0 finished — e.g. `qwen2-vl-7b/pmbt_gate_up/` (all three artefacts exist).
2. Re-launch step 10 for that combination via `run_pipeline.sh`. Confirm the script prints `[resume] Reusing cached contrastive POPE preprocessing …` and does **not** load the VLM during preprocessing.
3. Confirm L1+ shard jobs are still submitted normally (no regression in the per-layer scheduling path).
4. As a negative test, pick a fresh `(model, variant)` with no `cett_diff_scores.json` on disk and confirm the original preprocessing still runs end-to-end.

## Scope / ownership

This edit is in `code/halluc_score_neurons.py`, which belongs to the enrichment-analysis worktree (Session A — `enrichment-checkpoint` branch per the MH-Neuron project memory). The step-weight-merging worktree where the dependency-deadlock was diagnosed does **not** own this file and should not edit it directly.

## Background that may be useful

- Project memory `project_enrichment_blockers` notes "0/9 complete; 2 unfixed `halluc_score_neurons.py` bugs (model-type gap, InternVL image_flags)" — those are separate from the caching guard and not in scope here.
- The bigger picture: 27 re-launches × ~1 GPU-min preprocessing each ≈ 1 GPU-hour saved; more importantly, the model load itself stalls a GPU slot for ~30 s × 27 ≈ 15 min of slot occupancy that's pure overhead.
- `--resume` is already the script's default behaviour for per-batch L0 checkpoints (lines 216–218 of `halluc_score_neurons.py`). This guard extends the same "reuse what's on disk" philosophy to the preprocessing layer.

---

*Generated 2026-06-03 from the step-weight-merging session diagnosing the May 29 dependency-deadlock on `waic-risk`. Paste verbatim into the enrichment-analysis session.*
