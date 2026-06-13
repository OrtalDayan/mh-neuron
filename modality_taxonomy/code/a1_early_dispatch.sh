#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a1_early_dispatch.sh — Early-layer-only merging experiment (L0-15)
#
#  Motivation: A1 2×2 showed PMBT+full beats PMBT+L16 by -3.31 interaction.
#  This implies layers 0-15 are where PMBT's advantage comes from.
#  This script tests: merge only L0-15 and compare.
#
#  Cells dispatched:
#    Cell E (Uniform+L0-15): --p5-mode base --p5-layer-end 15
#    Cell F (PMBT+L0-15):    --p5-mode pmbt --p5-layer-end 15
#
#  Compare to existing cells (all already measured):
#    Cell C (Uniform+full):  23.60 | 33.88 | 22.59  (LLaVA | Qwen | Idef)
#    Cell B (PMBT+full):     26.02 | 34.14 | 21.70
#    Cell A (Uniform+L16+): 23.35 | 35.03 | 22.84
#    Cell D (PMBT+L16+):    22.46 | 33.50 | 22.46
#
#  LLaVA only. Replicate on Qwen/Idef if LLaVA shows interesting result.
#  6 bsubs (2 merges + 4 evals). ~45 min.
# ═══════════════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

BENCHES="POPE,MathVerse_MINI_Text_Dominant"
LAYER_END=15    # merge layers 0..15 inclusive; 16..31 kept at VLM baseline

echo "===== A1 early-only dispatch — $(date) ====="
echo ""

# ─── LLaVA Cell E: Uniform + L0-15 ───
echo "=== [1/2] LLaVA Cell E: Uniform α=0.9 + layers 0-15 ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-end $LAYER_END \
    --p5-benchmarks "$BENCHES"

# ─── LLaVA Cell F: PMBT + L0-15 ───
echo "=== [2/2] LLaVA Cell F: PMBT (0.7/1.0/1.0) + layers 0-15 ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-layer-end $LAYER_END \
    --p5-benchmarks "$BENCHES"

echo ""
echo "===== Early-only dispatch done — $(date) ====="
echo ""
echo "Expected output dirs:"
echo "  results/25-merge/llava-next-llama3-8b/dart-prop/uniform_a0.9_L0-15/"
echo "  results/25-merge/llava-next-llama3-8b/dart-prop/pmbt_t0.7_v1.0_m1.0_a1.0_o1.0_L0-15/"
echo ""
echo "NOTE: merges first, then re-run this script to trigger evals (they defer on"
echo "      first run because .pth doesn't exist yet)."
echo ""
echo "Monitor: bjobs -w | grep L0-15"
