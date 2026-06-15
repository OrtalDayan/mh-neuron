# Local Results — Step-25 Baseline vs. Uniform 0.9

Derived from `code/extract_merge_coverage.py`, which walks every `*_score.csv` under `results/25-merge/{model}/{tv}/`, picks the latest by mtime per (mode, benchmark) tuple, and surfaces the headline score regardless of which of the historical output-path layouts wrote it. Supersedes the earlier hand-curated extractions.

**Regenerate**: `python3 code/extract_merge_coverage.py --alpha 0.9 --csv /tmp/coverage.csv` from `modality_taxonomy/`.

---

## Coverage at a glance — 5 model × task-vector pairs

| # | Model | Task vector | Baseline | +Uniform 0.9 |
|---|---|---|---|---|
| 1 | idefics2-8b | MAmmoTH-7B-Mistral | 9 / 14 | 9 / 14 |
| 2 | llava-1.5-7b | MetaMath-7B-V1.0 | 9 / 14 | 9 / 14 |
| 3 | llava-next-llama3-8b | dart-math-prop2diff | 9 / 14 | 9 / 14 |
| 4 | llava-next-mistral-7b | MAmmoTH-7B-Mistral | 8 / 14 | 8 / 14 |
| 5 | qwen2-vl-7b | Qwen2-Math-7B | 9 / 14 | **2 / 14** |

Denominator 14 = the canonical Step-25 benchmark set (MathVista, 5 MathVerse splits, MathVision, MMStar, DynaMath, MM-Math, POPE, MME, TriviaQA, HallusionBench).

**Universal holes (zero rows across all 5 pairs):** MMStar · MM-Math · MME · TriviaQA · HallusionBench. The mistral MM-Math jobs running right now (`25_lm_m1_{bl,uni9}_mmm`) will fill one of those cells for one pair.

**Pair-specific gap:** qwen2-vl-7b's +Uniform 0.9 column only has T-D + POPE — the *latest-by-mtime* CSVs for uniform are from a Jun-1 partial re-score; the older April CSVs (which the Jun-1 PDF used) still exist on disk but lose the tie-break.

---

## 1. idefics2-8b (+MAmmoTH-7B-Mistral)

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 52.00 | 53.10 | **+1.10** |
| MathVerse-Overall (mean of 5) | 18.17 | 18.78 | +0.61 |
| MathVerse-T-D | 21.83 | 22.84 | **+1.01** |
| MathVerse-T-L | 20.18 | 20.69 | +0.51 |
| MathVerse-V-I | 19.54 | 19.67 | +0.13 |
| MathVerse-V-D | 19.04 | 18.78 | −0.26 |
| MathVerse-V-O | 10.28 | 11.93 | **+1.65** |
| MathVision | 16.78 | 17.76 | +0.98 |
| DynaMath-Avg | 21.80 | 23.01 | **+1.21** |
| POPE-Overall | 86.36 | 86.41 | +0.05 |

## 2. llava-1.5-7b (+MetaMath-7B-V1.0)

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 24.10 | 23.70 | −0.40 |
| MathVerse-Overall (mean of 5) | 15.03 | 14.90 | −0.13 |
| MathVerse-T-D | 15.61 | 14.85 | −0.76 |
| MathVerse-T-L | 14.59 | 14.59 | +0.00 |
| MathVerse-V-I | 14.85 | 14.72 | −0.13 |
| MathVerse-V-D | 15.36 | 15.10 | −0.26 |
| MathVerse-V-O | 14.72 | 15.23 | +0.51 |
| MathVision | 15.79 | 17.11 | **+1.32** |
| DynaMath-Avg | 12.73 | 11.86 | −0.87 |
| POPE-Overall | 84.80 | 84.10 | −0.70 |

## 3. llava-next-llama3-8b (+dart-math-prop2diff)

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 37.80 | 38.00 | +0.20 |
| MathVerse-Overall (mean of 5) | 18.45 | 19.03 | +0.58 |
| MathVerse-T-D | 22.34 | 23.60 | **+1.26** |
| MathVerse-T-L | 19.67 | 20.05 | +0.38 |
| MathVerse-V-I | 19.80 | 20.05 | +0.25 |
| MathVerse-V-D | 15.48 | 15.86 | +0.38 |
| MathVerse-V-O | 14.97 | 15.61 | +0.64 |
| MathVision | 14.14 | 14.80 | +0.66 |
| DynaMath-Avg | 22.61 | 24.23 | **+1.62** |
| POPE-Overall | 87.14 | 87.44 | +0.30 |

## 4. llava-next-mistral-7b (+MAmmoTH-7B-Mistral)

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 35.40 | 36.00 | +0.60 |
| MathVerse-Overall (mean of 5) | 18.27 | 17.92 | −0.35 |
| MathVerse-T-D | 21.19 | 21.07 | −0.12 |
| MathVerse-T-L | 17.89 | 18.27 | +0.38 |
| MathVerse-V-I | 19.16 | 19.16 | +0.00 |
| MathVerse-V-D | 17.01 | 16.75 | −0.26 |
| MathVerse-V-O | 16.12 | 14.34 | **−1.78** |
| MathVision | 11.18 | 13.82 | **+2.64** |
| DynaMath-Avg | 17.78 | 19.26 | **+1.48** |
| POPE-Overall | — | — | — |

**MM-Math in flight**: jobs `521563` (baseline) and `521565` (uniform) at ~39%/44% as of 2026-06-04 morning; ETA ~20–27 hours more. They will add a single MM-Math row to this table when they complete.

## 5. qwen2-vl-7b (+Qwen2-Math-7B)

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 61.10 | — | — |
| MathVerse-Overall (mean of 5) | 30.68 | — (only T-D) | — |
| MathVerse-T-D | 34.77 | 35.03 | +0.26 |
| MathVerse-T-L | 30.58 | — | — |
| MathVerse-V-I | 30.71 | — | — |
| MathVerse-V-D | 32.23 | — | — |
| MathVerse-V-O | 25.13 | — | — |
| MathVision | 20.72 | — | — |
| DynaMath-Avg | 33.55 | — | — |
| POPE-Overall | 86.48 | 86.12 | −0.36 |

**Uniform-side gap explanation:** The Jun-1 PDF showed a complete +Uniform 0.9 row for qwen2-vl-7b. Those numbers came from older `eval_uniform_0.9/Qwen2-VL-7B-Instruct0.9/` CSVs (April). A partial Jun-1 re-score landed newer CSVs for T-D and POPE only, and the script's "latest by mtime" policy picks those up. The April CSVs still exist; to surface them in this table the script would need an extra knob (e.g. `--prefer-complete` or path-priority overrides) or the qwen2-vl uniform run should be re-launched from a clean state to produce a fresh complete set.

---

## Cross-cutting observations

- **+Uniform 0.9 is broadly mildly positive across the 4 complete pairs** — best on MathVision (+2.64 mistral, +1.32 llava-1.5, +0.98 idefics2) and DynaMath (+1.62 llava-llama3, +1.48 mistral, +1.21 idefics2). The only pair where uniform clearly hurts is llava-1.5 + MetaMath-7B-V1.0, where 6 of 9 benchmarks regress.
- **MathVerse-V-O is the most volatile column** — −1.78 for mistral, +1.65 for idefics2, +0.51 for llava-1.5. Idiosyncratic per-pair behaviour rather than a uniform tendency.
- **The number-quality gap between local and the BRV paper** (documented in the original Jun-1 PDF as a 1–3 pp MathVerse shortfall for LLaVA-Next-LLaMA3) is unchanged — those numbers come from the same `dart-prop/eval_baseline/` and `eval_uniform_0.9/` CSVs.

---

## What changed from the previous version of this supplement

- Mistral row was already complete after the 2026-06-04 update; that's preserved.
- llava-1.5-7b + MetaMath-7B-V1.0 was the only "extra" row in the previous version; it's still present, with one minor difference: this version drops the DynaMath-Worst column (no recovery script for it in `extract_merge_coverage.py` yet — would need a special case in `extract_score`).
- idefics2-8b, llava-next-llama3-8b, qwen2-vl-7b were *not* in the previous supplement (they were in the Jun-1 PDF only). They're now folded in.
- The "Model × task-vector pairs found on disk but with no usable data" section is dropped because the script's `discover_model_tv_pairs` already filters those out.

## Provenance

- Script: `code/extract_merge_coverage.py` — single-file utility, runs from `modality_taxonomy/` against the `mh-neuron/modality_taxonomy/results/25-merge/` tree by default.
- Per-benchmark extraction conventions documented inline in the script's `extract_score()` function.
- Numbers above are the script's stdout output, with the per-pair Δ column computed inline in this doc (`+Uniform − Baseline`).
- MathVerse-Overall = simple mean of the 5 split rows, matching BRV's methodology.
- Generated 2026-06-04 from the step-weight-merging session.
