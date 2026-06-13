#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A1 2×2 FACTORIAL DISPATCH — layer-scheduling ablation
#
#  Motivation: SRF (Ali, Zoabi, Wolf 2025, arXiv:2511.12220) show that
#  VLM hallucinations can be mitigated by editing the UPPER HALF of
#  transformer layers (L = 16..32 for 32-layer models). Their Table 4
#  ablation shows layers [16-32] win (+2.97 on A-OKVQA) while [24-32]
#  alone has zero effect. The winning scope is the full upper half.
#
#  We test whether the PMBT modality taxonomy is redundant or complementary
#  with SRF-style late-layer scheduling. The 2×2 factorial:
#
#                   | Full scope (have ✓)   | Layers 16+ only (this dispatch)
#    ---------------|----------------------|--------------------------
#    Uniform α      | Tables 1-3 Uniform    | Cell A
#    PMBT per-cat   | Tables 1-3 Full PMBT  | Cell D
#
#  Main effect of layer-restrict  = (A - C)  and  (D - B), averaged
#  Main effect of PMBT taxonomy   = (B - C)  and  (D - A), averaged
#  Interaction (PMBT × SRF-scope) = (D - B) - (A - C)
#    → If small: PMBT is redundant once we restrict to SRF's upper half
#    → If large: PMBT adds value even when scoped to SRF's range
#
#  Per model: 2 dispatches × (1 merge + 2 eval jobs) = 6 bsubs
#  Total: 3 models × 6 = 18 bsubs on waic-risk, ~1.5h wall clock
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

LOG=/tmp/a1_2x2_dispatch.log
> "$LOG"
echo "===== A1 2×2 dispatch — $(date) =====" | tee -a "$LOG"

# Benchmarks for Round 1 — minimal decision set.
# Add MME, TQA, MMStar, MathVista in Round 2 if A1 shows promise.
BENCHES="POPE,MathVerse_MINI_Text_Dominant"

# Layer threshold: merge from layer 16 onward (out of 32 total layers).
# Matching SRF (Ali et al. 2025), who empirically show that corrections to
# L = {16, 17, ..., 32} are the winning range for their spectral filter.
# Their Table 4 ablation shows that narrower late-only ranges ([24-32]) have
# essentially no effect (72.49 vs greedy 72.49), while [16-32] yields +2.97.
# Matching their threshold gives our experiment a citable precedent and
# makes the PMBT-vs-late-scheduling comparison directly interpretable.
LAYER_START=16

# ─────────────────────────────────────────────────────────────────
# Model 1: LLaVA-Next-LLaMA3-8B + Dart-Math
# ─────────────────────────────────────────────────────────────────

echo ""                                                               | tee -a "$LOG"
echo "=== [1/3] LLaVA Cell A: Uniform α=0.9 + layers 16+ ==="         | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

echo ""                                                               | tee -a "$LOG"
echo "=== [1/3] LLaVA Cell D: PMBT (0.7/1.0/1.0) + layers 16+ ==="   | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Model 2: Qwen2-VL-7B + Qwen2-Math-7B
# ─────────────────────────────────────────────────────────────────

echo ""                                                               | tee -a "$LOG"
echo "=== [2/3] Qwen Cell A: Uniform α=0.9 + layers 16+ ==="          | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

echo ""                                                               | tee -a "$LOG"
echo "=== [2/3] Qwen Cell D: PMBT (0.9/1.0/1.0) + layers 16+ ==="    | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.9 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Model 3: Idefics2-8B + MAmmoTH-7B-Mistral
# ─────────────────────────────────────────────────────────────────

echo ""                                                               | tee -a "$LOG"
echo "=== [3/3] Idefics2 Cell A: Uniform α=0.9 + layers 16+ ==="      | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

echo ""                                                               | tee -a "$LOG"
echo "=== [3/3] Idefics2 Cell D: PMBT (0.8/1.0/0.9) + layers 16+ ==="| tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.8 --p5-alpha-visual 1.0 --p5-alpha-multi 0.9 \
    --p5-alpha-other 1.0 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

echo ""                                                               | tee -a "$LOG"
echo "===== Dispatch complete — $(date) =====" | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep L16                 # jobs should contain _L16 suffix"
echo "  bjobs -w | grep -c L16              # should be 18 (6 merges + 12 evals)"
echo ""
echo "Output locations (each combines cells A + D per model):"
echo "  results/25-merge/llava-next-llama3-8b/dart-prop/{uniform_a0.9_L16-end,pmbt_t0.7_v1.0_m1.0_*_L16-end}/"
echo "  results/25-merge/qwen2-vl-7b/qwen2-math/{uniform_a0.9_L16-end,pmbt_t0.9_v1.0_m1.0_*_L16-end}/"
echo "  results/25-merge/idefics2-8b/mammoth1/{uniform_a0.9_L16-end,pmbt_t0.8_v1.0_m0.9_*_L16-end}/"
echo ""
echo "Expected wall-clock: ~1-2 hours (merge ~3 min + POPE+MV-TD ~30 min)"
echo ""
echo "When done, extract + 2×2 interaction:"
echo "  python3 code/extract_merge_results.py 2>/dev/null | grep -B 1 -A 3 'L16-end'"
