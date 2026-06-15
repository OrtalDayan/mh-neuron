# Handoff — fix MMStar eval failing with `AssertionError` on the judge allow-list

**Audience:** the worktree that owns the VLMEvalKit_brv fork. That session has no context from the step-weight-merging session where this handoff was prepared, so this document is self-contained.

## Task

Make step-25 MMStar eval jobs actually produce score CSVs. Currently every MMStar job — across **every** model+task-vector pair, going back to April — fails silently with:

```
ERROR: Model <model> x Dataset MMStar combination failed: , skipping this combination.
Traceback (most recent call last):
  File "VLMEvalKit_brv/run.py", line 413, in main
    eval_results = dataset.evaluate(result_file, **judge_kwargs)
  File "VLMEvalKit_brv/vlmeval/dataset/image_mcq.py", line 212, in evaluate
    assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
AssertionError
```

The LSF job marks DONE successfully — VLMEvalKit's run.py wraps the per-(model, dataset) loop in a try/except — so the failure is invisible without reading the `.err` files.

## Why it's the universal MMStar blocker

The step-weight-merging session today verified: **zero** MMStar `_score.csv` files exist anywhere under `mh-neuron/modality_taxonomy/results/25-merge/<model>/<tv>/` for any of the 5 model+TV pairs that have data. Old April logs and today's freshly-submitted MMStar jobs both fail the same way. This isn't an "MMStar hasn't been re-run yet" gap — MMStar has *never* succeeded on this branch.

## Root cause

`vlmeval/dataset/image_mcq.py:212` hard-asserts the judge model is one of three exact strings:

```python
assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
```

`run_pipeline.sh` (line ~7224) forces MMStar to use `exact_matching`:

```bash
[[ "$_BN" == "MMStar" || "$_BN" == "MME" || "$_BN" == "POPE" || "$_BN" == "HallusionBench" ]] \
    && _BN_JUDGE="exact_matching"
```

So on paper the `--judge exact_matching` flag should keep MMStar in-bounds of the assertion. But the assertion is firing in practice, meaning either:

1. `--judge exact_matching` isn't reaching `image_mcq.py:212` — some intermediate code is overwriting `model` to a default (most likely `gpt-4o-mini`, which run_pipeline.sh's `--judge` flag in the eval bsub command also passes as the GPT judge for benchmarks that DO need GPT scoring).
2. The string is reaching but being modified — capitalization, trailing whitespace, alias rewriting.
3. The `model` variable at line 212 refers to something different than the `--judge` arg (maybe the MMStar dataset class is hardcoded to look at a different config field).

## Required diagnosis (suggested first step)

Read `vlmeval/dataset/image_mcq.py` around the MMStar evaluate path (the class is probably `MMStar` or `ImageMCQDataset`):

```bash
grep -n "MMStar\|class.*MCQ\|def evaluate" vlmeval/dataset/image_mcq.py | head -30
```

Then trace what `model` is at line 212:
- Is it the `--judge` arg passed via `judge_kwargs`?
- Or is it pulled from a class attribute / env var / dataset config?

A 5-second probe is to add `print(f'DEBUG: model={model!r}')` immediately before line 212 and re-run one MMStar job to see what the actual value is.

## Fix paths (rank-order by reversibility)

### Fix A — Expand the allow-list (least invasive)

If `model` at line 212 is being set to a value like `gpt-4o-mini` or some other modern judge despite the `--judge exact_matching` flag, just expand the allow-list to include the actual judges in use:

```python
assert model in [
    'chatgpt-0125', 'exact_matching', 'gpt-4-0125',
    'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo',
]
```

The list as written is plainly stale — `gpt-4o-mini` has been the standard judge since at least 2024 and is the default in `run_pipeline.sh`.

### Fix B — Fix the `--judge` plumbing

If the deeper bug is that `--judge exact_matching` isn't actually being propagated to MMStar's `evaluate(...)`, find where `model` is read and ensure it falls back to the `judge_kwargs['model']` value:

```python
model = judge_kwargs.get('model', 'exact_matching')
```

This is the cleaner long-term fix but requires understanding the MMStar dataset's class hierarchy.

### Fix C — Bypass the assertion

Remove the assertion entirely and let MMStar try whatever judge it's given. May break other things downstream — not recommended unless A/B are blocked.

## Verification plan after the edit

1. Apply chosen fix.
2. From `mh-neuron/modality_taxonomy/`:
   ```bash
   bash code/run_pipeline.sh --step 25 --model-type idefics2 \
       --p5-mode uniform --p5-baseline --p5-benchmarks "MMStar"
   ```
3. Wait for both bsub jobs (baseline + uniform_a0.9 eval) to complete (~30–60 min each).
4. Check `results/25-merge/idefics2-8b/mammoth1/baseline/eval/` and `.../uniform_a0.9/eval/` for `*MMStar*_score.csv` files. They should now exist and contain real numbers.
5. Smoke test on the other 4 models (qwen2-vl, llava-llama3, llava-mistral, llava-1.5). If they all produce CSVs, the fix is complete.

## Scope / ownership

`vlmeval/dataset/image_mcq.py` belongs to the VLMEvalKit_brv fork. The step-weight-merging worktree where this bug was diagnosed does **not** own this file and should not patch it directly.

## Background that may be useful

- The assertion list `['chatgpt-0125', 'exact_matching', 'gpt-4-0125']` looks like a snapshot from early 2024, when those were the relevant OpenAI judge models.
- Step 25 always wants `exact_matching` for MMStar (it's a multi-choice benchmark — no GPT-judge scoring needed), so Fix A is the lowest-risk path: just acknowledge that some other judge name is sneaking in.
- This affects all 5 model+TV pairs that have on-disk data; once fixed, the `extract_merge_coverage.py` script will surface MMStar values for all of them immediately on next-run.

---

*Generated 2026-06-04 from the step-weight-merging session, diagnosing why step-25 MMStar jobs mark DONE in LSF without producing score CSVs across all 5 model+TV pairs. Paste verbatim into the VLMEvalKit-fork session.*
