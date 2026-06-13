#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a1_complete.sh — Dispatch evals for A1 2×2 cells + fill Cell B gaps
#
#  After a1_2x2_dispatch.sh runs and the 6 merge jobs finish, run this to:
#    - Dispatch POPE + MVerseTD for all 6 new cells (A and D per model)
#    - Fill any missing Cell B (PMBT+full) benchmarks that Tables 1-3 didn't run
#
#  Idempotent: the pipeline skips already-completed evals.
#  Total bsubs: ~12-15 (varies with which Cell B cells already have data).
#  Wall clock: ~30-60 min.
# ═══════════════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

BENCHES="POPE,MathVerse_MINI_Text_Dominant"
LAYER_START=16

echo "===== A1 completion dispatch — $(date) ====="
echo ""

# ═══════════════════════════════════════════════════════════════════
#  Cell D dispatches (PMBT + L16+) — trigger evals now that merges done
# ═══════════════════════════════════════════════════════════════════

echo "=== [1/9] LLaVA Cell D: PMBT+L16 evals ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES"

echo "=== [2/9] LLaVA Cell A: Uniform+L16 evals ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES"

echo "=== [3/9] Qwen Cell D: PMBT+L16 evals ==="
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.9 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES"

echo "=== [4/9] Qwen Cell A: Uniform+L16 evals ==="
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES"

echo "=== [5/9] Idefics2 Cell D: PMBT+L16 evals ==="
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.8 --p5-alpha-visual 1.0 --p5-alpha-multi 0.9 \
    --p5-alpha-other 1.0 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES"

echo "=== [6/9] Idefics2 Cell A: Uniform+L16 evals ==="
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode base --p5-alpha 0.9 \
    --p5-layer-start $LAYER_START \
    --p5-benchmarks "$BENCHES"

# ═══════════════════════════════════════════════════════════════════
#  Cell B gap fills (PMBT + full scope — reference from Tables 1-3)
#  The pipeline is idempotent; benchmarks that are already done will skip.
# ═══════════════════════════════════════════════════════════════════

echo "=== [7/9] LLaVA Cell B: PMBT+full evals (idempotent) ==="
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "$BENCHES"

echo "=== [8/9] Qwen Cell B: PMBT+full evals (idempotent) ==="
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.9 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "$BENCHES"

echo "=== [9/9] Idefics2 Cell B: PMBT+full evals (idempotent) ==="
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.8 --p5-alpha-visual 1.0 --p5-alpha-multi 0.9 \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "$BENCHES"

echo ""
echo "===== A1 completion dispatch done — $(date) ====="
echo ""
echo "Monitor with:"
echo "  bjobs -w | grep -E 'll3|qw|idef'"
echo ""
echo "When bjobs clears, run:"
echo "  bash code/a1_extract.sh"
