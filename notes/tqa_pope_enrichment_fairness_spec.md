# Spec: Put POPE and TriviaQA enrichment on equal footing

**For:** Claude Code, working in the MH-Neurons repo.
**File of interest:** `code/halluc_score_neurons.py` (defines `compute_enrichment`,
`compute_hallucination_rate`, `compute_triviaqa_error_rate`, both contrastive builders),
and `code/merge_halluc_scores.py` (the Phase-2 `--run_enrichment` entry point).

## Goal (read this first — it is the whole point)

Make the **TriviaQA enrichment leg** and the **POPE enrichment leg** comparable, so
that an odds ratio > 1 on one task and < 1 on the other reflects genuine modality
specialization and not a scale / metric / sampling artifact. Then **report whatever the
odds ratios are**, including if they do NOT show a visual↑POPE/↓TQA, text↓POPE/↑TQA
dissociation.

This is a neutral fairness fix, NOT an attempt to produce a dissociation. Do not tune
thresholds, top-k, filtering, or metric definitions toward any target pattern. If the
legs come out fair and the dissociation is absent or partial, that is the correct
result and must be reported as-is.

## Background: what is already fair (do NOT change)

- `compute_enrichment` (≈ line 2248) is task-agnostic: it takes a `neuron_scores` dict
  and a `top_k_pct`, selects the top-k by `np.argsort`, and runs Fisher's exact test
  against the same `flat_labels` universe with the same random baseline. Fairness does
  not live here.
- Ablation granularity: both legs run through the same `ablation_worker` in the same
  batches (POPE via `compute_hallucination_rate`, TQA via `compute_triviaqa_error_rate`),
  so the intervention unit is already matched. Keep it that way.

## The three things that are NOT fair, and must be fixed

### 1. Metric shape (primary defect)

- POPE: `compute_hallucination_rate(..., halluc_measure=...)` supports a **dense**
  `'logit_margin'` mode (continuous P(yes) from the Yes/No logit margin) in addition to
  `'binary'`. The code comment at ~line 1505 explains binary single-neuron ΔH is
  near-all-zero and degenerate.
- TQA: `compute_triviaqa_error_rate` is **binary only** (generate 32 tokens, alias-match,
  error = n_incorrect / n_total). There is no dense analogue, and open-ended correctness
  flips even more rarely under single-neuron ablation than POPE's yes/no — so TQA ΔH is
  the *more* degenerate of the two.

**Required:** both legs must use the **same metric family** with comparable density.
Two acceptable ways to achieve this — pick ONE and apply it to BOTH legs identically:

  - **(Preferred) Dense answer-confidence delta on both.**
    - POPE: keep / use `halluc_measure='logit_margin'` → ΔH = change in mean P(yes) on
      gt=no questions.
    - TQA: build the missing dense analogue — ΔH_TQA = change in the model's confidence
      in the gold answer at the answer position (e.g. Δ of the negative log-prob, or
      Δ P(gold-alias first token), under ablation vs baseline). This is a continuous
      per-neuron signal that mirrors the POPE logit-margin signal. It does NOT exist yet
      and must be implemented inside `compute_triviaqa_error_rate` (or a sibling
      function) using the same `_AnswerLogitCapture` / lm_head hook machinery already
      used for POPE's logit_margin path.
  - **(Acceptable fallback) Binary on both, but only if neither leg is degenerate.**
    If you keep binary ΔH for both, you MUST confirm that
    `compute_enrichment`'s degeneracy guard (≈ lines 2301–2321) does NOT fire for
    *either* leg. If it fires for either, the binary route is invalid and you must use
    the dense route above. Do not silently report ORs from a flagged-degenerate leg.

Note: there is already a dense, matched pair available if you prefer not to build a new
TQA signal — `cett_diff_scores` (POPE) and `cett_diff_scores_tqa` (TQA). Using CETT-diff
for BOTH legs is a legitimate matched comparison. State clearly in the output which score
family was used; do not mix (e.g. logit-margin ΔH for POPE vs CETT-diff for TQA is NOT
matched).

### 2. Contrastive filtering parity

Both legs have contrastive builders (`build_contrastive_pope_set`,
`build_contrastive_triviaqa_set`). Verify and, if needed, align:

- Same clean-example **cap** (POPE ~1000; TQA `--triviaqa_cap` default 1000 — confirm
  they match).
- Same **resampling protocol** for the consistency filter: number of samples K and
  temperature used to decide "consistently correct/incorrect". POPE uses K=10 @ T=0.7
  per the paper; confirm TQA uses the same K and temperature, or document the difference.
- Same **consistency criterion** (retain only questions the baseline model answers
  stably, so ΔH is measured against a stable baseline rather than noise).

Report the final retained-set sizes for both legs side by side.

### 3. Identical enrichment call path

- The TQA leg must be scored through the same Phase-2 `--run_enrichment` path as POPE,
  with the **same** `top_k_pct`, the **same** `flat_labels` neuron universe, and the
  **same** `n_random_trials` and `seed`. Do not run TQA enrichment as an offline one-off
  with different parameters.
- Emit both `enrichment_results.json` (POPE) and `enrichment_results_tqa.json` (TQA) from
  the same routine in the same run, so they are guaranteed parameter-matched.

## Pre-flight checks (run BEFORE trusting any number)

These guard against the unresolved issues from prior analysis. Print results and stop if
they fail.

1. **Labels match the paper.** Load the PMBT labels each leg uses and print the
   visual/text/multimodal/unknown distribution per model. Diff against paper Table 1:
   - LLaVA-OV-7B: 45.2 / 34.1 / 5.8 / 14.9
   - InternVL2.5-8B: 36.3 / 52.6 / 11.1 / 0.0
   - Qwen2.5-VL-7B: 41.8 / 38.1 / 8.0 / 1.7
   If a model's `unknown` is not ~0 where Table 1 says 0.0 (InternVL especially), the
   labels are stale/wrong and enrichment on them is meaningless — flag and stop.
2. **Generation cap of the labels.** Report which `_min{min}_max{max}` directory suffix
   the label dir carries (bare = 300/100; suffixed = a non-default cap). Both legs must
   use the SAME label set. Print it; do not assume.
3. **Which `halluc_measure` produced the on-disk POPE scores** (binary vs logit_margin).
   If unknown, recompute rather than guess. The TQA leg's metric must be matched to
   whatever family POPE ends up using (see defect #1).
4. **Degeneracy guard status per leg.** After building each score dict, report whether
   `compute_enrichment`'s degenerate-driving-set condition is true for POPE and for TQA.
   If true for either, the binary route is void.

## Sign convention

Confirm both ΔH signs are oriented the same way: "larger score = removing this neuron
hurts the task more" (POPE: ablation raises hallucination; TQA: ablation raises error /
lowers gold-answer confidence). `compute_enrichment` takes the top-k *largest* scores, so
both legs must put "most task-damaging neurons" at the top. Document the convention used.

## Required outputs

1. `enrichment_results.json` and `enrichment_results_tqa.json`, parameter-matched, per
   model/hook combo on disk.
2. A combined 2×2 dissociation table per model/hook:

   | category | POPE OR (p) | TQA OR (p) |
   |---|---|---|
   | visual | … | … |
   | text | … | … |
   | multimodal | … | … |
   | unknown | … | … |

3. A one-line provenance header per combo: score family used (logit-margin ΔH /
   binary ΔH / CETT-diff / combined), top_k_pct, retained POPE n, retained TQA n,
   label-dir suffix, and degeneracy-guard status for each leg.
4. A coverage summary across all model×hook combos on disk: which cells show the flip,
   which don't, which are flagged degenerate or label-mismatched.

## Hard constraints

- Do not change `compute_enrichment`'s statistics, top-k logic, or random baseline.
- Do not pick the score family, top-k, or filter that "looks best." Pick the matched
  pair, run it, report it.
- Do not relabel a binary correctness metric and a logit-margin metric both as "ΔH" in
  the output — name each leg's actual metric.
- If a leg is degenerate or label-mismatched, the correct action is to flag and stop for
  that combo, not to report its odds ratios.
- Read-only on existing result/label files except where you are explicitly writing the
  new `enrichment_results*.json`. Do not regenerate labels or descriptions as part of
  this task; if labels fail the Table-1 check, report it and stop.
