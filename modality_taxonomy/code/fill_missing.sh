#!/bin/bash
# =================================================================
#  Dispatch all missing evals to fill the ~10% remaining gap
#  in the three per-model comparison tables.
#
#  Safe to re-run — run_pipeline.sh will [skip] anything that
#  already has a score CSV or is currently running.
#
#  Bad-node handling: run_pipeline.sh defaults to excluding lgn20
#  and lgn25 (via BAD_NODES="lgn20,lgn25"). To override per-block,
#  append e.g. --exclude-nodes "lgn20,lgn25,lgn17" after --long-desc.
# =================================================================

set -u
cd ~/mh-neuron/modality_taxonomy

LOG=/tmp/fill_missing_dispatch.log
> "$LOG"

echo "===== Fill Missing Results — $(date) =====" | tee -a "$LOG"
echo ""                                            | tee -a "$LOG"

# -----------------------------------------------------------------
# (1) Qwen2-VL Full PMBT perception  (9 cells: POPE, TriviaQA, MME)
# -----------------------------------------------------------------
echo "=== (1) Qwen2-VL Full PMBT t0.9/1.0/1.0_o1.0 perception ===" | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.9 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "POPE,TriviaQA,MME" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------
# (2) Qwen2-VL DP PMBT math tail  (7 cells: MV, MVerse×5, MVision)
# -----------------------------------------------------------------
echo ""                                                               | tee -a "$LOG"
echo "=== (2) Qwen2-VL DP PMBT t0.9/1.0/1.0 math tail ==="             | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.9 --p5-alpha-visual 1.0 --p5-alpha-multi 1.0 \
    --p5-pmbt-scope mlp --p5-mlp-projs down \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "MathVista,MathVerse,MathVision" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------
# (3) LLaVA Uniform a0.8 MME  (2 cells)
# -----------------------------------------------------------------
echo ""                                                    | tee -a "$LOG"
echo "=== (3) LLaVA Uniform a0.8 MME ==="                  | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode uniform --p5-alpha 0.8 \
    --p5-benchmarks "MME" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------
# (4) MM-Math + DynaMath tail for selected DP configs  (use waic-long
#     because MM-Math takes 16-40 hours)
# -----------------------------------------------------------------
echo ""                                                               | tee -a "$LOG"
echo "=== (4) LLaVA DP t0.6/1.0/0.9 MM-Math (waic-long) ==="           | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-long --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.6 --p5-alpha-visual 1.0 --p5-alpha-multi 0.9 \
    --p5-pmbt-scope mlp --p5-mlp-projs down \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "MM-Math" 2>&1 | tee -a "$LOG"

echo ""                                                               | tee -a "$LOG"
echo "=== (5) Idefics2 DP t0.9/0.9/0.9 MM-Math (waic-long) ==="       | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type idefics2 \
    --gmem 80 --queue waic-long --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 0.9 \
    --p5-alpha-text 0.9 --p5-alpha-visual 0.9 --p5-alpha-multi 0.9 \
    --p5-pmbt-scope mlp --p5-mlp-projs down \
    --p5-alpha-other 0.9 \
    --p5-benchmarks "MM-Math" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------
# (6) Equal-α DynaMath + MM-Math for all 9 configs (3 models × 3 t)
#     DynaMath on waic-risk (~30 min), MM-Math on waic-long
# -----------------------------------------------------------------
echo ""                                                                    | tee -a "$LOG"
echo "=== (6) Equal-α DynaMath + MM-Math tail (9 configs × 2 bench) ==="   | tee -a "$LOG"
for model in llava-llama3 idefics2 qwen2vl; do
    for t in 0.7 0.8 0.9; do
        # DynaMath — fast, normal queue
        bash code/run_pipeline.sh --step 25 --model-type "$model" \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$t" --p5-alpha-visual "$t" --p5-alpha-multi "$t" \
            --p5-alpha-other 1.0 \
            --p5-benchmarks "DynaMath" 2>&1 | tee -a "$LOG"

        # MM-Math — long queue
        bash code/run_pipeline.sh --step 25 --model-type "$model" \
            --gmem 80 --queue waic-long --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$t" --p5-alpha-visual "$t" --p5-alpha-multi "$t" \
            --p5-alpha-other 1.0 \
            --p5-benchmarks "MM-Math" 2>&1 | tee -a "$LOG"
    done
done

# -----------------------------------------------------------------
# (7) Equal-α t=0.7 specific subset completions
#     Qwen2-VL: MVe-VO + MMStar (t=0.7 only; t=0.8/0.9 are fully scored)
#     LLaVA:    MVe-TD + MVe-VI (t=0.7 only)
# -----------------------------------------------------------------
echo ""                                                                | tee -a "$LOG"
echo "=== (7) Equal-α t=0.7 subset fills ==="                          | tee -a "$LOG"

# Qwen t=0.7: MMStar + MathVerse (the VO subset is part of MathVerse full dispatch)
bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 0.7 --p5-alpha-multi 0.7 \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "MMStar,MathVerse" 2>&1 | tee -a "$LOG"

# LLaVA t=0.7: MathVerse (includes TD + VI subsets)
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text 0.7 --p5-alpha-visual 0.7 --p5-alpha-multi 0.7 \
    --p5-alpha-other 1.0 \
    --p5-benchmarks "MathVerse" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------
# Dispatch summary
# -----------------------------------------------------------------
echo ""                                                                    | tee -a "$LOG"
echo "================================================================="   | tee -a "$LOG"
echo "DISPATCH SUMMARY"                                                    | tee -a "$LOG"
echo "================================================================="   | tee -a "$LOG"
echo "[submit]            count: $(grep -c '^  \[submit\]' $LOG)"          | tee -a "$LOG"
echo "[skip] score exists:       $(grep -c '\[skip\].*score exists' $LOG)" | tee -a "$LOG"
echo "[skip] Merge exists:       $(grep -c '\[skip\] Merge exists' $LOG)"  | tee -a "$LOG"
echo "[skip] already running:    $(grep -c 'already running' $LOG)"        | tee -a "$LOG"
echo "[wait]              count: $(grep -c '\[wait\]' $LOG)"               | tee -a "$LOG"
echo ""                                                                    | tee -a "$LOG"
echo "Queue state:"                                                        | tee -a "$LOG"
echo "  Running: $(bjobs 2>/dev/null | grep -c RUN)"                      | tee -a "$LOG"
echo "  Pending: $(bjobs 2>/dev/null | grep -c PEND)"                     | tee -a "$LOG"
echo ""
echo "Dispatch complete. Log saved to $LOG"