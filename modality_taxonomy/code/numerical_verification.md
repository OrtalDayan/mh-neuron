# 1.1 Numerical Verification Checklist — ECCV #9948

For each row, run the lookup command on the cluster and write the result in the
"Cluster value" column. Mismatches go into a fix list for Day 3 trim.

Total expected time: 15–20 min. Lookups are independent — do them in any order.

---

## A. Claims verifiable from per-model classification JSONs

These read from:
  results/3-classify/full/llava-onevision-7b/llm_fixed_threshold/classification_stats_all.json
  results/3-classify/full/llava-onevision-7b/llm_permutation/permutation_stats_all.json

| # | Rebuttal claim | Cluster lookup |
|---|----------------|----------------|
| A1 | "POPE 90.2" (LLaVA-OV baseline POPE accuracy) | See section C below — POPE is from VLMEvalKit, not classification JSON |
| A2 | "12–20% → 0–1.7%" (FT-multimodal "unknown" collapse under PMBT, W1) | Run on each of 3 primary models the cluster has (llava-1.5-7b, llava-onevision-7b, qwen2.5-vl-7b OR internvl2.5-8b). For each: `jq '.stats' classification_stats_all.json` (FT) and `jq '.stats' permutation_stats_all.json` (PMBT). Compute unknown_count / total_neurons × 100 in each. Verify FT range is 12–20% and PMBT range is 0–1.7% across the 3 models. |
| A3 | "15.8–31.7%" (Tab. 1: FT-multimodal share for 3 primary models) | `jq '.stats.multimodal / (.stats.visual + .stats.text + .stats.multimodal + .stats.unknown) * 100' classification_stats_all.json` — run on each of 3 primary models, expect range 15.8–31.7%. |
| A4 | "15–45%" (Intro: FT-multimodal share across all four models) | Same lookup as A3 but include the fourth model. Range should widen from 15.8–31.7 to 15–45. |
| A5 | "15–32%" (Abstract: FT-multimodal "reclassified by PMBT") | This is a different stat — fraction of FT-multimodal neurons that PMBT relabels as something else. Compute: count FT-multimodal neurons whose PMBT label ≠ multimodal, divide by total FT-multimodal count. Needs `neuron_labels_all.json` + `neuron_labels_permutation_all.json`. Run a quick Python: load both, inner-join by (layer, neuron_idx), filter `ft.label == "multimodal"`, count `pmbt.label != "multimodal"`. Range across 3 primary models should be 15–32%. |

---

## B. Claims verifiable from per-neuron hallucination scores

These read from `results/10-halluc_scores/full/{model}/` for each of 3 primary
models. The exact file name depends on what step 10 wrote — likely a JSON or NPY
with per-(layer, neuron) hallucination scores plus category labels merged in.

| # | Rebuttal claim | Cluster lookup |
|---|----------------|----------------|
| B1 | "fold 1.10–1.35×" (visual neurons enriched among hallucination drivers) | For each of 3 primary models, identify the top-5% hallucination-driver set (by combined score). Compute: P(visual neuron is in top-5%) / P(visual neuron in random 5%) — this is the fold enrichment. Verify range 1.10–1.35× across 3 models. The exact aggregation formula should match what's in §4 of the submitted paper — read it and replicate. |
| B2 | "fold 0.58–0.83×" (text neurons depleted among hallucination drivers) | Same as B1 but for text neurons; expect fold < 1 in all 3 models. |
| B3 | "depletion 0.83×" (in §1 R3 "Empirical evidence" paragraph; should be the value for one specific model — likely LLaVA-OV) | Look up the LLaVA-OV-specific text-fold value. Should be ~0.83×. The current rebuttal cites this as the model-specific value, distinct from the cross-model range B2. |
| B4 | "CETT-diff 1.09×" (text neurons in §2 W3 causal-vs-coactive paragraph) | This is the CETT-difference metric (paper notation). Compute or look up in the per-model halluc_scores file. Verify ≈1.09× for text neurons. |
| B5 | "ΔH 0.83×" (text neurons in same W3 paragraph) | Same source, different aggregate. Verify ≈0.83×. |

---

## C. Claims verifiable from VLMEvalKit baselines (POPE, CHAIR_i)

These read from:
  results/7-equal-fraction-ablation/full/llava-onevision-7b/random_trials/vlmeval_baseline/

| # | Rebuttal claim | Cluster lookup |
|---|----------------|----------------|
| C1 | "POPE 90.2" (LLaVA-OV baseline POPE accuracy) | `find . -name "*POPE*_score*" -path "*vlmeval_baseline*" -exec cat {} \;` — read the score CSV. POPE accuracy is averaged across 3 strategies; expect overall ≈90.2%. |
| C2 | "POPE 92.0" (LLaVA-OV with steering at α=0.5, +1.8) | Find the corresponding steering result. Path likely under `results/11-steering/full/llava-onevision-7b/dh/visual/alpha_0.5/`. Look for POPE eval JSON. Expect 92.0% (i.e., +1.8 over 90.2). |
| C3 | "CHAIR_i 26.1" (LLaVA-OV baseline) | `find . -name "*CHAIR*_score*" -path "*vlmeval_baseline*"`. CHAIR_i is per-instance hallucination rate, expect 26.1. NOTE: CHAIR_i is an error rate (lower = better). |
| C4 | "CHAIR_i 24.3" (steering -1.8) | Same path as C2 steering result, look for CHAIR. Expect 24.3. |

---

## D. Claims that need cross-checking against the submitted paper

These are not on the cluster as JSONs — they're claims about what the paper says.
You verify by reading the paper PDF or its `.tex`.

| # | Rebuttal claim | Source to check |
|---|----------------|-----------------|
| D1 | "Fig. 2(e,f) errors hidden behind p_m=0.98" | Look at submitted paper Fig. 2(e) and Fig. 2(f) and their captions. Confirm p_m=0.98 is the relevant value. |
| D2 | "Supp F dose 0.1–5%" | Read Supplementary §F. What dose values does it actually report? |
| D3 | "Supp J cross-architecture steering" | Confirm Supp §J exists and shows cross-arch steering. |

---

## E. Claims that need external paper lookup (NOT on cluster)

| # | Rebuttal claim | Source to check |
|---|----------------|-----------------|
| E1 | "VCD reports POPE gains of ≈+1.5 to +2.3" on LLaVA-1.5-7B | VCD paper: Leng et al., "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding," CVPR 2024. Look at their POPE table. Find the LLaVA-1.5-7B row. Verify the +1.5 to +2.3 range matches their reported gains across POPE strategies. |
| E2 | "VCD baseline ~82" (LLaVA-1.5-7B POPE baseline) | Same paper, same table — the baseline column. Verify ≈82%. |
| E3 | "≈18% relative error reduction" (1.8 / (100-90.2) ≈ 18.4%) | Pure arithmetic check: (92.0 − 90.2) / (100 − 90.2) = 1.8/9.8 = 18.37%. ✓ This one is fine if C1 and C2 verify. |

---

## What to do with the results

After running the lookups, classify each row into:

- **MATCH** — number is correct, no action.
- **MINOR DRIFT** — number is close but not exact (e.g., rebuttal says 90.2 but JSON has 90.18). Round-off; usually fine. Still record the exact value for your records.
- **MISMATCH** — number is wrong (e.g., rebuttal says fold 1.35× but JSON has 1.42×). FIX in `.tex` before Day 3 trim.
- **SOURCE NOT FOUND** — file doesn't exist where expected. Either the experiment hasn't been run or the path is different. Flag for Shai meeting tomorrow — may indicate something needs rerunning.
- **EXTERNAL CHECK PENDING** — for E1/E2, you may not have the VCD paper in front of you right now. Defer until you can access it (preferably before Shai meeting).

If anything comes back MISMATCH, tell me the corrected number and I'll patch the rebuttal `.tex` and rebuild the PDF. Do this BEFORE the Shai meeting — walking in with corrected numbers in hand looks much better than discovering them mid-meeting.

---

## Priority within the 15–20 min budget

If you run out of time and can only do a subset, prioritize in this order:

1. **C1, C2, C3, C4** (POPE + CHAIR_i baselines and steering results) — these are the most-cited numbers in the rebuttal and the easiest for a reviewer to check against the literature.
2. **B1, B2** (fold ranges) — these support the W3 effect-size argument.
3. **A2, A5** (the abstract and W1 reclassification numbers) — these are flagged in 2P7W's review as inconsistencies.
4. **D2** (Supp F dose) — quick read of the paper, can do anytime.
5. **B3, B4, B5** — load-bearing for the W3 causal-vs-coactive paragraph but more granular.
6. **E1, E2** — external, may need to defer until you have the VCD paper open.

Items A1, A3, A4 are lower priority — they're consistent with each other and with the paper, so they tend to drift together and are less reviewer-visible.
