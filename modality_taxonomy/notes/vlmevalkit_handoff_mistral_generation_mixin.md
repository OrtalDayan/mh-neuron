# Handoff — fix LLaVA-Next-Mistral TriviaQA: `'MistralModel' object has no attribute 'generate'`

**Audience:** the worktree that owns `modern_vlms/VLMEvalKit_brv/.venv` and the HF cache. Same fork-side ownership as the InternVL `.generate()` handoff. This document is self-contained.

## Task

Make step-25 TriviaQA eval jobs for `llava-next-mistral-7b` actually produce score CSVs. Currently the jobs mark DONE in LSF but fail silently inside VLMEvalKit's swallow-and-skip handler with:

```
ERROR: Model llava_next_mistral_7b x Dataset TriviaQA combination failed:
  'MistralModel' object has no attribute 'generate', skipping this combination.
```

The traceback points to `vlmeval/vlm/llava/llava.py:428`:

```
output = self.model.language_model.generate(**text_only, **gen_kwargs)
```

where `self.model.language_model` is a `MistralModel` instance that does not expose a `.generate()` method.

## Root cause

Same family of issue as the InternVL `.generate()` blocker
(`notes/internvl_handoff_generation_mixin.md`). With newer transformers
(post-4.50), `PreTrainedModel` no longer inherits from
`GenerationMixin` — a model class that doesn't explicitly subclass
`GenerationMixin` loses the `.generate()` method.

The specific path that fires here is the **text-only generation
fall-back** inside `llava/llava.py:generate_inner` for image-less
benchmarks (TriviaQA). The image-bearing benchmarks (MathVista,
MathVerse, MathVision, DynaMath, POPE, MME, MMStar, HallusionBench)
take a different code path that uses the multi-modal LLaVA wrapper and
*does* have `.generate()`, so those land fine for mistral. **TriviaQA
is the only affected benchmark for mistral.**

## Scope of impact

| Benchmark | Affected for mistral? |
|---|---|
| MathVista, 5 MathVerse splits, MathVision, DynaMath, POPE, MME, HallusionBench, MMStar* | No (uses multi-modal `model.generate`) |
| **TriviaQA** | **Yes** (uses `model.language_model.generate`) |

\* MMStar fails for a different (universal) reason — see `notes/vlmevalkit_handoff_mmstar_judge_assertion.md`.

## Three possible fixes (mirror of the InternVL handoff)

### Fix A — Patch the HF-cache `modeling_mistral.py`

If the LLaVA-Next-Mistral model loads its LM via HF auto-class for
`mistralai/Mistral-7B-v0.1` (or whatever the local checkpoint pins to),
the underlying `MistralForCausalLM` should inherit `GenerationMixin`
in modern transformers. If it doesn't (older HF cache), add
`GenerationMixin` as a second parent.

### Fix B — Pin transformers to ≤ 4.49 in the eval venv

```bash
uv pip install --python <path-to-VLMEvalKit_brv/.venv/bin/python> 'transformers<4.50'
```

This is the same workaround as InternVL — if one is applied, both
unblock. Caveat: other model classes may regress.

### Fix C — Wrap `self.model.language_model` at eval time

Inside `vlmeval/vlm/llava/llava.py`, after the model is loaded, dynamically
rebind:

```python
from transformers.generation import GenerationMixin
lm = self.model.language_model
if not isinstance(lm, GenerationMixin):
    lm.__class__ = type(
        lm.__class__.__name__,
        (lm.__class__, GenerationMixin),
        {},
    )
```

Localised to LLaVA-Next-Mistral, no transformers downgrade, no HF-cache
patch. Same fragility caveat as the InternVL Fix C.

## Recommended

If you're already considering applying **Fix B** (transformers pin) for the
InternVL handoff, it covers Mistral TriviaQA for free.

If you'd rather keep transformers current, **Fix C** localised to
`vlmeval/vlm/llava/llava.py` is the smallest reversible patch.

## Verification plan

1. Apply chosen fix.
2. From `mh-neuron/modality_taxonomy/`:
   ```bash
   bash code/run_pipeline.sh --step 25 --model-type llava-mistral \
       --p5-mode uniform --p5-baseline --p5-benchmarks TriviaQA
   ```
3. Wait for both bsub jobs (~20–35 min each — the verified-web-dev split is small).
4. Check `results/25-merge/llava-next-mistral-7b/mammoth1/{baseline,uniform_a0.9}/eval/` for `*TriviaQA*_score.csv`.

## Scope / ownership

`modern_vlms/VLMEvalKit_brv/vlmeval/vlm/llava/llava.py`, the HF cache, and
the eval venv all belong to the VLMEvalKit-fork worktree. The
step-weight-merging worktree should not patch these directly.

## Related work already filed

- `notes/internvl_handoff_generation_mixin.md` — same root cause, different
  model class (`InternLM2ForCausalLM` instead of `MistralModel`).
- `notes/vlmevalkit_handoff_llava_next_install.md` — covers the
  *unrelated* `No module named 'llava'` blocker that affects llava-1.5
  and llava-onevision-7b (not mistral).
- `notes/vlmevalkit_handoff_qwen25vl_7b_registry.md`
- `notes/vlmevalkit_handoff_mmstar_judge_assertion.md`

---

*Generated 2026-06-04 from the step-weight-merging session, after observing the silent TriviaQA failure for llava-next-mistral-7b in the same submission batch that succeeded for MME and HallusionBench.*
