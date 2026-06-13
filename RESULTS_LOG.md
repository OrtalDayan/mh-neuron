# Results Log

## 2026-05-30 — α=0.001 PMBT reclassification across 9 VLMs + α=0.05↔α=0.001 comparison

### Question
What does the PMBT modality-category distribution look like for each of the 9 classify-supported VLM model-types × {gate_up, gate} hooks at α=0.001, and how does it compare to α=0.05?

### Method
- Ran step 3 (classify) for all 9 model-types × {gate_up, gate}: `bash modality_taxonomy/code/run_pipeline.sh --step 3 --model-type <t> --hook-point {gate_up,gate} --long-desc`. `llava-mistral` needed classify-script support added first (commit `be114e2`: aliases to the `llava-llama3` LLaVA-Next path).
- α=0.001 is the script default in `neuron_modality_statistical.py` (`--alpha` arg). Per-layer `permutation_stats_layersN-N+1.json` merged via `--step 4` into `permutation_stats_all.json`. 18 merged files in total (9 models × 2 hooks).
- α=0.05 baseline: previously-preserved `..._alpha_0.05` sibling dirs at the same `_min100_max2048` desc variant. Available for 8 gate_up runs + llama3 gate = **9 comparable pairs**. Aggregated by summing each layer's `stats` dict.
- Description input: full 23K-image `detail_23k`; `n_permutations=1000`; min-tokens=100, max-tokens=2048 (via `--long-desc`).

### Key findings

**α=0.001 distributions, full reclassification (18 runs):**

| model | hook | visual | text | multi | unknown | n |
|---|---|---|---|---|---|---|
| llava-hf | gate_up | 31.2% | 57.0% | 11.7% | 0.1% | 352,256 |
| llava-hf | gate | 31.8% | 58.4% | 9.8% | 0.0% | 352,256 |
| llava-ov | gate_up | 55.8% | 31.0% | 12.5% | 0.8% | 530,432 |
| llava-ov | gate | 47.1% | 34.0% | 14.2% | 4.7% | 530,432 |
| internvl | gate_up | 30.6% | 42.2% | **27.0%** | 0.2% | 458,752 |
| internvl | gate | 34.4% | 46.8% | 18.8% | 0.0% | 458,752 |
| qwen2vl | gate_up | 51.2% | 33.3% | 14.6% | 0.8% | 530,432 |
| qwen2vl | gate | 45.6% | 33.7% | 14.6% | 6.2% | 530,432 |
| qwen25vl-7b | gate_up | 43.7% | 36.6% | 18.5% | 1.1% | 530,432 |
| qwen25vl-7b | gate | 39.6% | 37.1% | 17.6% | 5.7% | 530,432 |
| qwen25vl-3b | gate_up | 49.5% | 33.1% | 12.5% | 5.0% | 396,288 |
| qwen25vl-3b | gate | 43.7% | 32.6% | 10.4% | **13.2%** | 396,288 |
| llava-llama3 | gate_up | 34.5% | 51.8% | 13.7% | 0.1% | 458,752 |
| llava-llama3 | gate | 30.2% | 57.7% | 12.0% | 0.0% | 458,752 |
| idefics2 | gate_up | 55.6% | 35.9% | 8.3% | 0.1% | 458,752 |
| idefics2 | gate | 57.8% | 35.8% | 6.3% | 0.0% | 458,752 |
| llava-mistral | gate_up | 24.8% | 64.2% | 10.8% | 0.2% | 458,752 |
| llava-mistral | gate | 21.1% | 69.2% | 9.7% | 0.0% | 458,752 |

Source: `modality_taxonomy/results/3-classify/full/<MODEL_NAME>/llm_permutation{_gate_up,}_min100_max2048/permutation_stats_all.json` (read `stats` field).

**α=0.05 → α=0.001 effect (9 comparable pairs, percentage-point Δ):**

| model × hook | Δ visual | Δ text | Δ multimodal | Δ unknown |
|---|---|---|---|---|
| internvl gate_up | −7.9 | −4.1 | **+12.0** | −0.0 |
| qwen25vl-7b gate_up | −4.6 | −3.2 | +7.8 | +0.0 |
| qwen2vl gate_up | −4.6 | −2.3 | +7.1 | −0.2 |
| llava-ov gate_up | −4.2 | −1.8 | +6.0 | +0.0 |
| llava-llama3 gate_up | −3.4 | −2.6 | +6.0 | +0.0 |
| qwen25vl-3b gate_up | −3.4 | −2.3 | +5.5 | +0.2 |
| llava-1.5 gate_up | −2.8 | −2.3 | +5.1 | +0.0 |
| llava-llama3 gate | −2.8 | −2.3 | +5.1 | +0.0 |
| idefics2 gate_up | −2.4 | −0.9 | **+3.4** | −0.0 |

Source: per-layer JSONs aggregated from `modality_taxonomy/results/3-classify/full/<m>/llm_permutation{_gate_up,}_min100_max2048_alpha_0.05/permutation_stats_layers*.json`, compared to the merged α=0.001 file at the sibling path.

**Three robust findings:**

1. **Stricter α moves neurons from visual+text → multimodal, NOT into unknown.** Holds across every one of the 9 architectures examined. `Δ unknown` ≤ 0.2 pp anywhere; multimodal Δ between +3.4 and +12.0 pp.

2. **`unknown` is an *activity gate*, not a significance bucket** (`neuron_modality_statistical.py:654-664`): `total_high < min_high_tokens` → label "unknown", **independent of `alpha`**. The brief's "more unknown = more honesty about what we don't know" rationale doesn't hold in this codebase. Replace with: "α=0.001 → fewer over-confident single-modality assignments, more multimodal." Same scientific argument (lower FDR / higher precision), but the mechanism in this codebase routes the borderline neurons into *multimodal*, not *unknown*. The FDR-control rationale from Nanda 2024 / Gordon et al. 2021 still holds.

3. **InternVL is the most α-sensitive** (multimodal +12 pp); **idefics2 the least** (+3.4 pp). The shift size doesn't track parameter count or backbone family cleanly — appears to track the proportion of borderline-significant neurons at α=0.05.

### Caveats
- The α=0.05↔α=0.001 comparison covers **9 of 18** model×hook combinations. The other 9 (7 models' gate runs + mistral both hooks) had no preserved α=0.05 baseline at the `_min100_max2048` desc variant: mistral was never classified before; for the rest, the old gate-hook α=0.05 labels live in the non-suffixed `llm_permutation/` dirs under an older `max=300` description variant — not strictly apples-to-apples and not aggregated here.
- "Multimodal grows" was characterized via aggregated per-category counts in the merged `permutation_stats_all.json`. Per-layer behaviour was earlier spot-checked on 16 LLaVA-1.5 gate_up layers and showed the same pattern; not exhaustively re-verified across all 18 runs in this writeup.
- α=0.001 reclassification is the script default (`--alpha 0.001`). Reproducibility depends on the merge step finding labels under the `_min100_max2048` suffix path, which requires `--long-desc` on every `run_pipeline.sh` invocation.
- `llava-mistral` classify support was added by aliasing to `llava-llama3` (commit `be114e2`). The `get_layer_names` else fallback returns `model.layers.{i}.mlp.act_fn` for both llama3 and mistral *gate*-hook resolution; the gate runs completed and produced sensible-looking distributions (visual+text+multi ≈ 100%, unknown ~0), but the underlying module path may be incorrect for HF LLaVA-Next (expected `model.language_model.layers.{i}.mlp.act_fn`). Treat the llama3 and mistral *gate* numbers as provisional until a `model.get_submodule(...)` check is done.
- α=0.05 baselines preserved at `..._alpha_0.05` siblings for 9 pairs only; mistral has no prior labels, and most models' gate-hook baselines weren't moved aside (no need — the new gate-hook target dirs were already empty).

---

## 2026-05-30 — Weight-merging (step 25): baseline/uniform/PMBT inventory + uniform λ tuned vs BRV

### Question
For the cross-modal weight-merging study (step 25), what results exist (baseline / uniform / PMBT p<0.05), which uniform λ best matches the BRV paper (*Bring Reason to Vision*, ICML 2025), and is the data ready for the planned p<0.001 PMBT runs?

### Method
- Inventoried `results/25-merge` for the 4 merge pairs: `llava-next-llama3-8b/dart-prop`, `idefics2-8b/mammoth1`, `qwen2-vl-7b/qwen2-math`, `llava-1.5-7b/MetaMath-7B-V1.0`.
- Re-extracted headline metrics in BRV Table-2/3/7 layout. **MathVista All/General/Math uses the native `category` field** (`math-targeted-vqa`=Math / `general-vqa`=Gen) via VLMEvalKit `post_check` (`vlmeval/dataset/utils/mathvista.py`), NOT skill-grouping — verified Overall reproduces the `_score.csv` (e.g. qwen 61.1). MathVerse = 5 split `_score_score.csv` files (Overall = mean of splits); MMStar All/Math from `_acc.csv`; DM = DynaMath Average; MV = MathVision Overall.
- Swept uniform λ∈{0.8, 0.85, 0.9}; compared each to BRV's reported row for the **same VLM+task-vector** by mean absolute error. Pulled BRV repo (`shiqichen17/VLM_Merging`) merge configs.

### Key findings

**1. Uniform λ0.9 is the paper-matching baseline for every model.** MAE vs BRV's λ0.9 row (excl. MathVerse), monotonic 0.9 < 0.85 < 0.8:

| model (+task vector) | MAE λ0.8 | λ0.85 | λ0.9 |
|---|---|---|---|
| LLaVA-Next +Dart-Prop | 1.17 | 0.40 | **0.16** |
| Idefics2 +MAmmoTH-1 | 1.14 | 0.71 | **0.34** |
| Qwen2-VL +Qwen2-Math | 11.79 | 7.01 | **2.43** |

LLaVA-Next / Idefics2 near-exact reproductions of BRV (several cells identical). Qwen2-VL is uniformly ~2–3 pt below the paper across all benchmarks — likely a checkpoint/eval-version diff, not a λ effect (λ trend still points cleanly to 0.9).

**2. BRV does not report a per-model best λ.** Paper states λ=0.9 only for LLaVA-Next; describes a {0.8,0.85,0.9} search for Idefics2/InternVL without printing the selection; gives no λ for Qwen2-VL (Appendix E). BRV repo `scripts/merge/example_detailed.sh` **hardcodes `--alpha 0.9` for every model/task-vector merge**; `merge.py` mode `base` = `alpha*VLM + (1-alpha)*math`, so alpha = VLM weight = λ. → **uniform baseline fixed at λ0.9** (memory: `project-uniform-lambda-09`).

**3. BRV-format results (native MathVista split), baseline vs uniform λ0.9 vs best PMBT p<0.05:**

| model | method | MVista A·Gen·Math | MVerse-O | MMStar A·Math | DM | MV |
|---|---|---|---|---|---|---|
| LLaVA-Next | Baseline | 37.8·52.0·25.7 | 18.5 | 43.3·29.6 | 22.6 | 14.1 |
| | Uniform λ0.9 | 38.0·49.1·28.5 | 19.1 | 43.6·33.6 | 24.2 | 14.8 |
| | PMBT t0.7·v1.0·m1.0 | 39.2·49.6·30.4 | 20.2 | 43.7·32.4 | 23.9 | 14.1 |
| Idefics2 | Baseline | 52.1·56.1·48.7 | 18.2 | 49.3·39.6 | 21.8 | 16.8 |
| | Uniform λ0.9 | 53.1·58.5·48.5 | 18.7 | 48.4·40.0 | 23.0 | 17.8 |
| | PMBT t0.8·v1.0·m0.9 | 53.6·59.6·48.5 | 18.6 | 48.3·42.4 | 20.8 | 17.1 |
| Qwen2-VL | Baseline | 61.1·69.8·53.7 | 30.7 | 59.6·58.4 | 33.6 | 20.7 |
| | Uniform λ0.9 | 58.3·66.1·51.7 | 30.4 | 56.5·55.2 | 31.4 | 20.1 |
| | PMBT t0.85·v1.0·m1.0 | 58.5·67.4·50.9 | 29.3 | 57.9·58.0 | 33.1 | 19.1 |
| LLaVA-1.5 | Baseline | 24.1·32.4·17.0 | 15.0 | 31.1·22.4 | 12.7 | 15.8 |
| | Uniform λ0.9 | 23.7·30.0·18.3 | 14.9 | 32.7·27.6 | 11.9 | 17.1 |

Trend matches BRV: Idefics2 & LLaVA-Next gain on math (MathVista-Math, MMStar-Math, DM, MV up); **Qwen2-VL is hurt by any merge** (reproduces BRV Table 7 / App E — already math-pretrained). PMBT damages Qwen less than uniform (MMStar-Math −0.4 vs −3.2; DM −0.5 vs −2.2).

**4. Benchmark coverage vs BRV:** we cover all BRV benchmarks (MathVista, MathVerse, MathVision, DynaMath, MMStar, MM-Math) plus extras (POPE, MME, TriviaQA). Nothing missing.

**5. p<0.001 readiness:** base classify ran at α=0.001 (script default) and completed May 30 for all 9 model-types × {gate, gate_up} — the un-suffixed `llm_permutation*_min100_max2048/neuron_labels_permutation_all.json` ARE the p<0.001 labels. The `25-merge/.../*_p0.001/` merge dirs are **stale/mislabeled** (merged model named without p-suffix; numbers = p<0.05) — NOT genuine p<0.001 merges. **p<0.001 PMBT merges not yet run.** Baseline + uniform λ0.9 are p-threshold-independent anchors, reusable as-is.

### Caveats
- Qwen2-VL ~2–3 pt below BRV across all benchmarks (uniform offset; check Qwen2-Math / Qwen2-VL checkpoint revision vs paper).
- Our MathVerse runs sit systematically below the paper (eval/gpt-extraction diff) — excluded from the λ MAE.
- llava-1.5-7b has uniform λ0.9 only (no 0.8/0.85) and no PMBT selective; not a BRV model.
- PMBT p<0.05 "best" config picked by MathVista-Math among non-degenerate runs; many low-`t` Qwen configs collapsed (MathVerse→0) and were excluded.
- Extraction artifacts currently in `/tmp` (ephemeral): `merge_numbers.csv`, `merge_brv_native.md`, `merge_baseline_uniform_colored.html`, `compare_paper.py`, `brv_native.py`.
