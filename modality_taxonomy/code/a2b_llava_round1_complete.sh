#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a2b_llava_round1_complete.sh — Fill all missing benchmark cells for Round 1
#
#  Current status of Round 1 LLaVA:
#    LABEL (smoke): has TriviaQA only (no POPE, no MVerseTD)
#    A_rd C=0.10:   has POPE + MVerseTD (no TriviaQA)
#    B_pure C=0.02: has POPE + MVerseTD (no TriviaQA)
#    B_pure C=0.10: NOTHING (merge was skipped in the original dispatch)
#    B_rdxnorm:     has POPE + MVerseTD (no TriviaQA)
#
#  This script completes all 5×3 = 15 cells (3 benchmarks × 5 variants).
#  Re-dispatches each variant with ALL 3 benchmarks — pipeline skips
#  already-done work (idempotent), so we only dispatch what's missing.
#
#  Total new bsubs: 1 merge (Bpure C=0.10) + 8 evals = 9 LSF jobs
#  Wall clock: ~30-60 min
# ═══════════════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

BENCHES="POPE,MathVerse_MINI_Text_Dominant,TriviaQA"
AT=0.7; AV=1.0; AM=1.0; AO=1.0
C_small=0.02; C_med=0.10; N_scale=2.0

# ─── Variant 3 FIRST (needs merge + all 3 evals) ───
echo ""
echo "=== [1/5] B_pure C=$C_med — submit merge + 3 evals (was previously skipped) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_pure \
    --p5-signal-scale-c $C_med \
    --p5-benchmarks "$BENCHES"

# ─── LABEL: need POPE + MVerseTD (TriviaQA already done) ───
echo ""
echo "=== [2/5] LABEL — submit POPE + MVerseTD (TriviaQA already done) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode label \
    --p5-benchmarks "$BENCHES"

# ─── A_rd: TriviaQA only (POPE + MVerseTD done) ───
echo ""
echo "=== [3/5] A_rd — submit TriviaQA (POPE + MVerseTD already done) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode A_rd \
    --p5-signal-scale-c $C_med \
    --p5-benchmarks "$BENCHES"

# ─── B_pure C=0.02: TriviaQA only ───
echo ""
echo "=== [4/5] B_pure C=$C_small — submit TriviaQA (POPE + MVerseTD already done) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_pure \
    --p5-signal-scale-c $C_small \
    --p5-benchmarks "$BENCHES"

# ─── B_rdxnorm: TriviaQA only ───
echo ""
echo "=== [5/5] B_rdxnorm — submit TriviaQA (POPE + MVerseTD already done) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_rdxnorm \
    --p5-signal-scale-c $C_med \
    --p5-signal-norm-scale $N_scale \
    --p5-benchmarks "$BENCHES"

echo ""
echo "===== Round 1 completion dispatch done ====="
echo ""
echo "Monitor:"
echo "  bjobs -w | grep ll3"
echo ""
echo "When bjobs clears, run:"
echo "  bash a2b_llava_extract.sh"
