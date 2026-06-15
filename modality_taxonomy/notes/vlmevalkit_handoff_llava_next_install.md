# Handoff — install LLaVA-NeXT into VLMEvalKit_brv's venv

**Audience:** the worktree that owns `modern_vlms/VLMEvalKit_brv/.venv`. That session has no context from the step-weight-merging session where this handoff was prepared, so this document is self-contained.

## Task

Install the `llava` Python package (from the LLaVA-NeXT repo) into the eval venv so that step-25 evaluation jobs for **both `llava-1.5-7b` and `llava-onevision-7b`** stop silently failing with:

```
ModuleNotFoundError: No module named 'llava'
File "VLMEvalKit_brv/vlmeval/vlm/llava/llava.py", line 21
    from llava.model.builder import load_pretrained_model
```

This affects 2 of the 9 enrichment-target models.

## Why

`vlmeval/vlm/llava/llava.py` is the model class for both LLaVA-1.5 and LLaVA-OneVision (via `LLaVA_Next` / `LLaVA_OneVision` subclasses). It imports `from llava.model.builder import load_pretrained_model` at class init time — if the `llava` package isn't in `sys.path`, every eval job dies inside VLMEvalKit's swallow-and-skip handler. From LSF's perspective the job marks DONE successfully; the failure is only visible in the `.err` log.

The package previously existed in the venv (older score CSVs at `25-merge/llava-1.5-7b/MetaMath-7B-V1.0/baseline/` prove it worked at some point). At some point the venv was rebuilt or repaired and the package was dropped. Today's fresh step-25 launches for both models all fail with the same error.

## Install command

The CRITICAL hint is already in the error message itself:

```
Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`
```

Adapted to this project's uv-managed venv:

```bash
uv pip install --python /home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/modern_vlms/VLMEvalKit_brv/.venv/bin/python \
    'git+https://github.com/LLaVA-VL/LLaVA-NeXT.git'
```

The version installed previously was sufficient to score both LLaVA-1.5 and LLaVA-OneVision baselines; the upstream `main` branch should be at least as capable. If pinning is desired, use `@<commit-sha>` in the URL.

## Things to verify after install

1. **Import check** from inside the venv:
   ```bash
   /home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/modern_vlms/VLMEvalKit_brv/.venv/bin/python -c \
       "from llava.model.builder import load_pretrained_model; print('OK')"
   ```
2. **VLMEvalKit smoke test**:
   ```bash
   cd /home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
   modern_vlms/VLMEvalKit_brv/.venv/bin/python modern_vlms/VLMEvalKit_brv/run.py \
       --data MathVista_MINI --model llava_v1.5_7b --judge gpt-4o-mini --reuse \
       --work-dir /tmp/llava_smoke
   ```
   Should load the model, run inference, and produce a score CSV.

3. **Step-25 re-launch from this worktree** (both models in parallel):
   ```bash
   for mt in llava-hf llava-ov; do
     bash code/run_pipeline.sh --step 25 --model-type "$mt" \
       --p5-mode uniform --p5-baseline
   done
   ```
   Idempotence will fill the missing benchmarks; ETA ~1–2 hr per model.

## Side effects to be aware of

- Installing `llava` (and its dependencies) may upgrade `transformers`, `accelerate`, or `torch` in the venv. Watch for version conflicts with other model classes (Qwen2-VL, Qwen2.5-VL, InternVL) that may have implicit pins via `vlmeval/vlm/*`. If post-install some other model regresses, the canonical fix is to keep `llava` at the older version it was compatible with — see the previous git log for that venv's `pip freeze` output if available.
- Disk usage: the package itself is small, but loading LLaVA-1.5 / LLaVA-OneVision weights for the first time may trigger HF cache downloads (already present for the established models — should be a no-op).

## Scope / ownership

The `modern_vlms/VLMEvalKit_brv/.venv` belongs to the VLMEvalKit-fork worktree. The step-weight-merging worktree where the missing-package bug was diagnosed does **not** own the venv and should not modify it directly.

## Background that may be useful

- llava-1.5 specifically also has a separate constraint: it **cannot run TriviaQA** because the model requires an image input and TriviaQA is text-only. This is documented in the step-weight-merging session's memory (`feedback_llava15_no_triviaqa.md`) and is orthogonal to this handoff — installing `llava` does not enable llava-1.5 TriviaQA, but does enable all other benchmarks for llava-1.5 plus the full benchmark suite for llava-onevision-7b.
- The other 4 models in the 9-target set have separate blockers tracked in `notes/vlmevalkit_handoff_qwen25vl_7b_registry.md`, `notes/internvl_handoff_generation_mixin.md`, `notes/vlmevalkit_handoff_mmstar_judge_assertion.md`, plus a hard script-side block for qwen2.5-vl-3b.

---

*Generated 2026-06-04 from the step-weight-merging session, after observing that both llava-1.5 and llava-ov eval jobs share the identical missing-`llava`-package failure. Paste verbatim into the VLMEvalKit-fork session.*
