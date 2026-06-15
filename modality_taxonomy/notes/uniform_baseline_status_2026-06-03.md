# Baseline + Uniform 0.9 — cross-model status (2026-06-03)

Snapshot of which of the 9 enrichment-target models currently have **Baseline** and **+Uniform 0.9** scores on disk in `mh-neuron/modality_taxonomy/results/25-merge/`. All values in percent; Δ = +Uniform − Baseline.

---

## Coverage summary

| # | Model | Math LLM (task vector) | Baseline | +Uniform 0.9 | Source |
|---|---|---|---|---|---|
| 1 | llava-1.5-7b | MetaMath-7B-V1.0 | ✓ | ✓ | Jun-3 supplement |
| 2 | llava-next-llama3-8b | dart-math (prop2diff) | ✓ | ✓ | Jun-1 PDF |
| 3 | idefics2-8b | MAmmoTH-7B-Mistral | ✓ | ✓ | Jun-1 PDF |
| 4 | qwen2-vl-7b | Qwen2-Math-7B | ✓ | ✓ | Jun-1 PDF |
| 5 | llava-next-mistral-7b | MAmmoTH-7B-Mistral | ✓ | ✓ | Jun-3 supplement |
| 6 | llava-onevision-7b | Qwen2-Math-7B | ✗ (empty) | ✗ (empty) | — |
| 7 | qwen2.5-vl-7b | Qwen2.5-Math | ✗ (empty) | ✗ (empty) | — |
| 8 | internvl2.5-8b | InternLM2-Math-Plus-7B | ✗ (empty) | ✗ (empty) | — |
| 9 | qwen2.5-vl-3b | — | — | — | no 25-merge dir at all |

**Totals:** 5 complete · 0 partial · 3 empty result dirs · 1 with no merge attempt.

---

## Tables for the 5 complete models

### 1. llava-1.5-7b (+MetaMath-7B-V1.0)

11 benchmark rows. No MMStar or MathVista General/Math splits in the CSVs. Includes POPE.

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 24.10 | 23.70 | −0.40 |
| MathVerse-Overall (mean) | 15.03 | 14.90 | −0.13 |
| MathVerse-T-D | 15.61 | 14.85 | −0.76 |
| MathVerse-T-L | 14.59 | 14.59 | +0.00 |
| MathVerse-V-I | 14.85 | 14.72 | −0.13 |
| MathVerse-V-D | 15.36 | 15.10 | −0.26 |
| MathVerse-V-O | 14.72 | 15.23 | +0.51 |
| MathVision | 15.79 | 17.11 | **+1.32** |
| DynaMath-Avg | 12.73 | 11.86 | −0.87 |
| DynaMath-Worst | 1.20 | 1.80 | +0.60 |
| POPE-Overall | 84.80 | 84.10 | −0.70 |

### 2. llava-next-llama3-8b (+dart-math-prop2diff)

| Benchmark | Type | Baseline | +Uniform 0.9 | Δ |
|---|---|---:|---:|---:|
| MathVista-Overall | perception | 37.80 | 37.90 | +0.10 |
| MathVista-General | perception | 51.96 | 49.13 | −2.83 |
| MathVista-Math | reasoning | 25.74 | 28.33 | +2.59 |
| MathVerse-Overall (mean) | reasoning | 18.45 | 19.03 | +0.58 |
| MathVerse-T-D | reasoning | 22.34 | 23.98 | +1.64 |
| MathVerse-T-L | reasoning | 19.67 | 19.92 | +0.25 |
| MathVerse-V-I | reasoning | 19.80 | 20.18 | +0.38 |
| MathVerse-V-D | reasoning | 15.48 | 15.48 | +0.00 |
| MathVerse-V-O | reasoning | 14.97 | 15.61 | +0.64 |
| MathVision | reasoning | 14.14 | 14.47 | +0.33 |
| MMStar-Overall | perception | 43.33 | 43.60 | +0.27 |
| MMStar-Math | reasoning | 29.60 | 33.60 | **+4.00** |
| DynaMath-Avg | reasoning | 22.75 | 24.35 | +1.60 |
| DynaMath-Worst | reasoning | 3.19 | 4.39 | +1.20 |

### 3. idefics2-8b (+MAmmoTH-7B-Mistral)

| Benchmark | Type | Baseline | +Uniform 0.9 | Δ |
|---|---|---:|---:|---:|
| MathVista-Overall | perception | 52.00 | 52.50 | +0.50 |
| MathVista-General | perception | 56.30 | 59.13 | **+2.83** |
| MathVista-Math | reasoning | 48.33 | 46.85 | −1.48 |
| MathVerse-Overall (mean) | reasoning | 18.38 | 18.96 | +0.58 |
| MathVerse-T-D | reasoning | 21.70 | 24.62 | **+2.92** |
| MathVerse-T-L | reasoning | 20.81 | 21.57 | +0.76 |
| MathVerse-V-I | reasoning | 20.05 | 19.67 | −0.38 |
| MathVerse-V-D | reasoning | 19.04 | 17.89 | −1.15 |
| MathVerse-V-O | reasoning | 10.28 | 11.04 | +0.76 |
| MathVision | reasoning | 16.45 | 14.80 | −1.65 |
| MMStar-Overall | perception | 49.33 | 48.40 | −0.93 |
| MMStar-Math | reasoning | 39.60 | 40.00 | +0.40 |
| DynaMath-Avg | reasoning | 21.80 | 23.01 | +1.21 |
| DynaMath-Worst | reasoning | 3.19 | 4.79 | +1.60 |

### 4. qwen2-vl-7b (+Qwen2-Math-7B)

| Benchmark | Type | Baseline | +Uniform 0.9 | Δ |
|---|---|---:|---:|---:|
| MathVista-Overall | perception | 61.10 | 58.30 | −2.80 |
| MathVista-General | perception | 69.78 | 66.09 | **−3.69** |
| MathVista-Math | reasoning | 53.70 | 51.67 | −2.03 |
| MathVerse-Overall (mean) | reasoning | 30.68 | 30.41 | −0.27 |
| MathVerse-T-D | reasoning | 34.77 | 33.88 | −0.89 |
| MathVerse-T-L | reasoning | 30.58 | 30.08 | −0.50 |
| MathVerse-V-I | reasoning | 30.71 | 30.08 | −0.63 |
| MathVerse-V-D | reasoning | 32.23 | 31.09 | −1.14 |
| MathVerse-V-O | reasoning | 25.13 | 26.90 | +1.77 |
| MathVision | reasoning | 20.72 | 20.07 | −0.65 |
| MMStar-Overall | perception | 59.60 | 56.47 | **−3.13** |
| MMStar-Math | reasoning | 58.40 | 55.20 | **−3.20** |
| DynaMath-Avg | reasoning | 33.55 | 31.40 | −2.15 |
| DynaMath-Worst | reasoning | 10.38 | 10.78 | +0.40 |

### 5. llava-next-mistral-7b (+MAmmoTH-7B-Mistral)

10 benchmark rows. No MMStar or MathVista General/Math splits in the CSVs.

| Benchmark | Baseline | +Uniform 0.9 | Δ |
|---|---:|---:|---:|
| MathVista-Overall | 35.40 | 36.00 | +0.60 |
| MathVerse-Overall (mean) | 18.27 | 17.92 | −0.35 |
| MathVerse-T-D | 21.19 | 21.07 | −0.12 |
| MathVerse-T-L | 17.89 | 18.27 | +0.38 |
| MathVerse-V-I | 19.16 | 19.16 | +0.00 |
| MathVerse-V-D | 17.01 | 16.75 | −0.26 |
| MathVerse-V-O | 16.12 | 14.34 | **−1.78** |
| MathVision | 11.18 | 13.82 | **+2.64** |
| DynaMath-Avg | 17.78 | 19.26 | +1.48 |
| DynaMath-Worst | 2.20 | 2.40 | +0.20 |

---

## To get to "all 9 covered"

| Model | Missing | Action needed |
|---|---|---|
| llava-onevision-7b | Baseline + Uniform eval | Full step-25 baseline + uniform run. Check whether `…/uniform_a0.9/merged_model_0.9.pth` exists; if so, only the evals need launching. |
| qwen2.5-vl-7b | Baseline + Uniform eval | Same as llava-OV. |
| internvl2.5-8b | Baseline + Uniform eval | Same as llava-OV. |
| qwen2.5-vl-3b | Everything | No `25-merge/qwen2.5-vl-3b/` dir exists. `run_pipeline.sh` doesn't have a math-LLM auto-route for `qwen25vl-3b` (line 7029–7044 only routes `llava-llama3`, `idefics2`, `qwen2vl`). Need to add the route, pick a math LLM, then run the whole step-25 pipeline. |

---

## Provenance

- Numbers for models #2–#4 come from the table in `notes/local_results_uniform_lambda_0.9.pdf` (Jun 1, 16:58).
- Numbers for #1 and #5 come from re-extracting score CSVs under `mh-neuron/modality_taxonomy/results/25-merge/{model}/{tv}/{baseline,uniform_a0.9}/eval/...` on 2026-06-03 (saved in `notes/local_results_supplement_2026-06-03.md`).
- MathVista-Overall: column `acc` row "Overall" (already percent).
- MathVerse splits: column `accuracy` row 1 (already percent). Overall = simple mean of the five, matching BRV's methodology.
- MathVision: column `acc` row "Overall" (already percent).
- DynaMath-Average/Worst: column `Overall`, rows "Average"/"Worst Case", multiplied by 100.
- MMStar: from PDF (Jun 1) only — not re-extracted today.
- POPE-Overall: column `Overall` row "Overall" (already percent).

---

*Generated 2026-06-03 from the step-weight-merging session for at-your-pace reading. Open in any editor / `less` / `bat` / Markdown viewer.*
