#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a2b_llava_cleanup.sh — Recovers from the job-name collision in Round 1
#
#  Problem:  Round 1 dispatch had a bug where B_pure C=0.02 and B_pure C=0.10
#            share the same job name "_Bp", causing variant 3 (C=0.10) to be
#            skipped as "already running" while variant 2 was still merging.
#
#  What this script does:
#  - Re-dispatches each of the 4 variants in sequence
#  - For variants 1, 2, 4: merges already exist → pipeline dispatches evals
#  - For variant 3: merge slot is now free → submits both merge + eval
#  - The TriviaQA smoke test is skipped (already done separately)
# ═══════════════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

BENCHES="POPE,MathVerse_MINI_Text_Dominant"
AT=0.7; AV=1.0; AM=1.0; AO=1.0
C_small=0.02; C_med=0.10; N_scale=2.0

# ─── Variant 3 FIRST (to dispatch the missing merge before job name collides again) ───
echo ""
echo "=== [3/4] RECOVER: Path B_pure C=$C_med (previously skipped) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_pure \
    --p5-signal-scale-c $C_med \
    --p5-benchmarks "$BENCHES"

# ─── Variant 1: dispatch eval (merge already done) ───
echo ""
echo "=== [1/4] Dispatch eval: Path A (A_rd) C=$C_med ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode A_rd \
    --p5-signal-scale-c $C_med \
    --p5-benchmarks "$BENCHES"

# ─── Variant 2: dispatch eval (merge already done) ───
echo ""
echo "=== [2/4] Dispatch eval: Path B_pure C=$C_small ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_pure \
    --p5-signal-scale-c $C_small \
    --p5-benchmarks "$BENCHES"

# ─── Variant 4: dispatch eval (merge already done) ───
echo ""
echo "=== [4/4] Dispatch eval: Path B_rdxnorm C=$C_med N=$N_scale ==="
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
echo "===== Cleanup dispatch complete ====="
echo ""
echo "Monitor:"
echo "  bjobs -w | grep ll3"
echo ""
echo "Check merge files exist for all 4 variants:"
echo "  for d in Ard_C0.10 Bpure_C0.02 Bpure_C0.10 Brn_C0.10_N2.0; do"
echo "    ls -la results/25-merge/llava-next-llama3-8b/dart-prop/pmbt_t0.7_v1.0_m1.0_a1.0_o1.0_\${d}/*.pth 2>/dev/null"
echo "  done"
