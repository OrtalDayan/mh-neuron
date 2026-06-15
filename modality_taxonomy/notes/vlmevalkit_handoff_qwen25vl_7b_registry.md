# Handoff — register `Qwen2.5-VL-7B-Instruct` in VLMEvalKit's `supported_VLM`

**Audience:** a Claude Code session running in the worktree that owns `modern_vlms/VLMEvalKit_brv/` (the VLMEvalKit fork). That session has no context from the step-weight-merging session where this handoff was prepared, so this document is self-contained.

## Task

Add a single entry to `modern_vlms/VLMEvalKit_brv/vlmeval/config.py` registering `'Qwen2.5-VL-7B-Instruct'` as a key in the `supported_VLM` dict, so that `run.py --model Qwen2.5-VL-7B-Instruct` actually resolves to a model class instead of raising `KeyError`.

## Context

The step-weight-merging session attempted step-25 baseline + uniform_a0.9 evaluation for `qwen2.5-vl-7b` (+Qwen2.5-Math-7B). `run_pipeline.sh` line 7175 sends:

```
--model Qwen2.5-VL-7B-Instruct
```

VLMEvalKit's run.py wraps the lookup in a try/except (run.py line 442) that catches *any* exception and "skips this combination" without raising. Result: each LSF job exits cleanly with no scores, no error in the .log, the cause hidden in the .err file:

```
KeyError: 'Qwen2.5-VL-7B-Instruct'
```

The registry currently has:
- `Qwen2.5-VL-3B-Instruct` (line 303) — present
- `Qwen2-VL-2B-Instruct`, `-7B-Instruct`, `-72B-Instruct` (lines 294–302) — present
- `Qwen2.5-VL-7B-Instruct` — **missing**

This affects all 10 step-25 eval jobs the script tried to submit for qwen2.5-vl-7b (both baseline and uniform_a0.9). All would have failed even if the merge had succeeded.

## File to edit

`modern_vlms/VLMEvalKit_brv/vlmeval/config.py` — the `supported_VLM` dict around line 303.

## Required addition

Insert immediately after line 303 (the existing 3B entry), keeping the same `Qwen2VLChat` class and arg structure:

```python
'Qwen2.5-VL-7B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-VL-7B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28, use_custom_prompt=False),
```

Reference — the existing 3B entry that defines the pattern:

```python
'Qwen2.5-VL-3B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-VL-3B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28, use_custom_prompt=False),
```

Only `3B` → `7B` changes in two places. `Qwen2VLChat` is the right class (Qwen2.5-VL shares the Qwen2-VL inference interface; the 3B entry uses the same class).

## Things to verify / preserve

1. **`Qwen2VLChat` import.** Already imported in config.py (used by both Qwen2-VL and Qwen2.5-VL-3B entries above). No new import needed.
2. **Model weights availability.** The 7B base model is locally cached at `modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct`. HF download would be a fallback if the local path isn't resolved. If you want to point the registry at the local cache, replace `'Qwen/Qwen2.5-VL-7B-Instruct'` with the absolute path — but **don't do this**: the registry is shared, and an absolute path in one user's tree breaks others. The HF identifier is correct; the `HF_HOME` env var (set in `run_pipeline.sh`) routes downloads to the shared cache.
3. **`use_custom_prompt=False`** matches the 3B entry. Don't drop it. Without it, Qwen2VLChat applies a chat-template wrapper that's not what BRV / our step-25 expects.
4. **`min_pixels` / `max_pixels`** values match the 3B entry. Keep them.
5. **Don't reformat the surrounding entries**. Single-line addition only.

## Verification plan after the edit

1. From `modern_vlms/VLMEvalKit_brv/`:
   ```bash
   .venv/bin/python -c "from vlmeval.config import supported_VLM; print('Qwen2.5-VL-7B-Instruct' in supported_VLM)"
   ```
   Should print `True`.

2. Dry-run a single benchmark (any) for the model on a small dataset:
   ```bash
   .venv/bin/python run.py --data MathVista_MINI --model Qwen2.5-VL-7B-Instruct --reuse --work-dir /tmp/qwen25vl7b_smoke
   ```
   Should load the model (HF download or local cache hit), run inference, and produce a score CSV — no `KeyError`.

3. Then re-run step 25 from the main worktree:
   ```bash
   bash code/run_pipeline.sh --step 25 --model-type qwen25vl-7b --p5-mode uniform --p5-baseline
   ```
   The 10 baseline + 10 uniform eval jobs should now actually produce scores under `results/25-merge/qwen2.5-vl-7b/qwen25-math/{baseline,uniform_a0.9}/eval/`.

## Scope / ownership

This edit is in `modern_vlms/VLMEvalKit_brv/vlmeval/config.py`, which belongs to the worktree owning the VLMEvalKit fork. The step-weight-merging worktree where the bug was diagnosed does **not** own this file and should not edit it directly.

## Background that may be useful

- A related bug for `llava-ov` was already fixed in the same diagnostic session: `run_pipeline.sh` line 7174 sent the wrong key (`llava_onevision_qwen2_7b`) instead of `llava_onevision_qwen2_7b_ov`. That fix lives in `mh-neuron/modality_taxonomy/code/run_pipeline.sh` (uncommitted on branch `pmbt-alpha-0001`).
- The InternVL eval jobs fail differently — `'InternLM2ForCausalLM' object has no attribute 'generate'`, a transformers/HF-cache compatibility issue with the model class definition in `~/.cache/huggingface/.../modeling_internvl_chat.py`. Out of scope here; tracked as a separate item.
- The merge jobs for qwen25vl-7b and internvl were also killed by `TERM_THREADLIMIT` (exit 143). That was patched in the same session by prepending `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to the merge bsub command in run_pipeline.sh.

---

*Generated 2026-06-04 from the step-weight-merging session, diagnosing why 42 step-25 eval jobs marked "DONE successfully" produced zero score CSVs. Paste verbatim into the VLMEvalKit-fork session.*
