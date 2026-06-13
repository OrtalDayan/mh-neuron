#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A2 3-MODEL DISPATCH
#
#  Dispatches confidence-weighted PMBT (A2 feature) on all 3 models,
#  with each benchmark getting its own bsub job for maximum parallelism.
#
#  Structure per model:
#    - 1 merge job (uses --p5-confidence-weighted → soft α)
#    - N eval jobs (one per benchmark, each waiting for the merge)
#
#  For 3 models × 6 benchmarks = 21 total bsubs on waic-risk.
#  Expect ~2-4 hours wall clock if queue is flowing.
#
#  Each model uses its table-1-best Full PMBT triplet; A2 adds soft α
#  weighting on top. Reference metrics in comments below.
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

LOG=/tmp/a2_3model_dispatch.log
> "$LOG"
echo "===== A2 3-model dispatch — $(date) =====" | tee -a "$LOG"

# Benchmarks — shared across all 3 models. Adjust if you want fewer/more.
# Each benchmark becomes a separate bsub job (line 7560 of run_pipeline.sh).
BENCHES="POPE,MME,TriviaQA,MathVerse_MINI_Text_Dominant,MMStar,MathVista_MINI"

# ─────────────────────────────────────────────────────────────────
# Model 1: LLaVA-Next-LLaMA3-8B + Dart-Math
#   Ref (hard labels, Full PMBT t=0.7):
#     MathMn=20.3, POPE-F1=87.7, MME-P=1513, MME-R=281.4, TQA=79.1
#   A2 goal: POPE-F1 >= 87.7 AND math subset >= hard-label values
# ─────────────────────────────────────────────────────────────────
echo ""                                                          | tee -a "$LOG"
echo "=== Model 1/3: LLaVA + Dart-Math (α_text=0.7) ==="        | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-confidence-weighted \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Model 2: Qwen2-VL-7B + Qwen2-Math-7B
#   Ref (hard labels, Full PMBT t=0.9):
#     MathMn=31.1, perception unscored (that's why we need this run)
#   A2 goal: Math >= 31.1 AND perception matches or exceeds baseline
# ─────────────────────────────────────────────────────────────────
echo ""                                                          | tee -a "$LOG"
echo "=== Model 2/3: Qwen2-VL + Qwen2-Math (α_text=0.9) ==="    | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.9 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-confidence-weighted \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Model 3: Idefics2-8B + MAmmoTH-7B-Mistral
#   Ref (hard labels, Full PMBT t=0.8/1.0/0.9):
#     MathMn=22.1, POPE-F1=86.7, MME-P=1524, MME-R=337.1, TQA=77.1
#   A2 goal: POPE-F1 >= 86.7 AND MME stays at or above 337
# ─────────────────────────────────────────────────────────────────
echo ""                                                          | tee -a "$LOG"
echo "=== Model 3/3: Idefics2 + MAmmoTH (α_text=0.8, α_multi=0.9) ==="  | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.8 --p5-alpha-visual 1.0 --p5-alpha-multi 0.9 \
    --p5-alpha-other 1.0 \
    --p5-confidence-weighted \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

echo ""                                                          | tee -a "$LOG"
echo "===== Dispatch complete — $(date) =====" | tee -a "$LOG"
echo ""
echo "Monitor:   bjobs -w | grep confw"
echo "Count:     bjobs -w | grep -c confw    # should be 21 (3 merges + 18 evals)"
echo "Log:       tail -f $LOG"
echo ""
echo "Expected wall clock: 2-4 hours if waic-risk has headroom."
echo ""
echo "When done, extract results with:"
echo "  python3 code/extract_merge_results.py 2>/dev/null | grep -B 1 -A 3 '_confw'"
