# Handoff — fix InternVL2.5 inference: `'InternLM2ForCausalLM' object has no attribute 'generate'`

**Audience:** the worktree that owns the InternVL evaluation toolchain (the modern_vlms VLMEvalKit_brv fork and/or its HF-cache). That session has no context from the step-weight-merging session where this handoff was prepared, so this document is self-contained.

## Task

Make step-25 baseline + uniform_a0.9 eval jobs for `internvl2.5-8b` actually produce score CSVs instead of failing silently. Currently every per-benchmark eval job marks DONE in LSF without writing a CSV, with the .err log containing:

```
ERROR: Model InternVL2_5-8B x Dataset DynaMath combination failed:
  'InternLM2ForCausalLM' object has no attribute 'generate',
  skipping this combination.
```

The traceback originates in `vlmeval/vlm/internvl_chat.py:471 generate_inner → 442 generate_v2 → response = self.model.chat(...)`, which internally calls `.generate()` on `InternLM2ForCausalLM`.

## Root cause

The InternVL2.5-8B HF cache ships its own custom `modeling_internvl_chat.py` and `modeling_internlm2.py`. The `InternLM2ForCausalLM` class in those files does *not* inherit from `transformers.GenerationMixin`. Newer transformers versions (post-4.50) stopped having `PreTrainedModel` itself inherit from `GenerationMixin`, so model classes that don't explicitly subclass `GenerationMixin` lose the `.generate()` method.

There's an HF-version warning visible in the same .err log that spells this out:

> InternLM2ForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly defined. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.

## Three possible fixes (pick whichever fits your branch's invariants)

### Fix A — Patch the HF-cache `modeling_internlm2.py` to inherit `GenerationMixin`

The custom model code is at:

```
~/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2_5_hyphen_8B/<rev>/modeling_internlm2.py
```

Find the class declaration:

```python
class InternLM2ForCausalLM(InternLM2PreTrainedModel):
```

and change it to:

```python
from transformers.generation import GenerationMixin

class InternLM2ForCausalLM(InternLM2PreTrainedModel, GenerationMixin):
```

This restores `.generate()` directly. The HF-cache is shared on disk so the patch survives across re-launches once applied, but it's lost if the cache is wiped or the rev is updated.

### Fix B — Pin transformers to ≤ 4.49 in the eval venv

The `modern_vlms/VLMEvalKit_brv/.venv` is uv-managed. Pin transformers:

```bash
uv pip install --python <path-to-VLMEvalKit_brv/.venv/bin/python> 'transformers<4.50'
```

This restores the old behaviour where `PreTrainedModel` inherits from `GenerationMixin`, so all models including InternLM2 get `.generate()` for free. Caveat: other models (especially Qwen2-VL, Qwen2.5-VL) may regress on the older transformers.

### Fix C — Wrap `self.model` in a `GenerationMixin`-bearing subclass at eval time

Inside `VLMEvalKit_brv/vlmeval/vlm/internvl_chat.py:__init__`, after the model is loaded, dynamically rebind:

```python
from transformers.generation import GenerationMixin
if not isinstance(self.model.language_model, GenerationMixin):
    self.model.language_model.__class__ = type(
        self.model.language_model.__class__.__name__,
        (self.model.language_model.__class__, GenerationMixin),
        {},
    )
```

Localized to InternVL; no transformers downgrade; no HF-cache patch needed. But fragile if HF-cache rev changes upstream.

## Recommended

**Fix A** if the HF-cache is stable in your environment (it is for at least the InternVL2_5-8B rev currently pinned). One file, two lines, easy to review.

## Verification plan

1. Apply chosen fix.
2. From `mh-neuron/modality_taxonomy/`:
   ```bash
   bash code/run_pipeline.sh --step 25 --model-type internvl \
       --p5-mode uniform --p5-baseline \
       --p5-benchmarks "MathVista_MINI,DynaMath"
   ```
3. Wait for the bsub jobs to finish (~30–60 min). Check `results/25-merge/internvl2.5-8b/internlm2-math-plus-7b/baseline/eval/` for `*_score.csv` files.
4. If the .err log no longer contains `has no attribute 'generate'`, the fix works.

## Related work already done

- **Merge `dtype` bug fixed** in `code/merge_pmbt.py` (commits TBA on `pmbt-alpha-0001` and `session/step_weight_merging`). The InternVL-specific code path now passes `torch_dtype=` instead of `dtype=` to `AutoModel.from_pretrained`, so the merge `.pth` produces correctly. Eval is the only remaining blocker.
- **Merge bsub thread cap fix** in `run_pipeline.sh` (committed in `1e09898` on `pmbt-alpha-0001`). Prevents the merge job from being killed by `TERM_THREADLIMIT`.

## Scope / ownership

`modeling_internlm2.py` (and its hosting HF cache rev), `internvl_chat.py`, and the VLMEvalKit_brv venv all belong to the worktree that owns VLMEvalKit_brv. The step-weight-merging worktree where the bug was diagnosed does **not** own these files and should not patch them directly.

---

*Generated 2026-06-04 from the step-weight-merging session, diagnosing why step-25 eval jobs for internvl2.5-8b mark DONE in LSF without producing score CSVs. Paste verbatim into the VLMEvalKit-fork session.*
