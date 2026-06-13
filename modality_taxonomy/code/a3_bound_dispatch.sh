#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A3 BOUND DISPATCH — bound the α=0.5 peak (LLaVA only, 4 cells)
#
#  The initial 4×3 grid found peak at (α=0.5, p<0.01) with MV-TD = 27.03.
#  To confirm α=0.5 is the genuine optimum, we bound it from both sides:
#    α_text=0.4 (aggressive side)
#    α_text=0.6 (conservative side)
#  at the two strict thresholds where the peak appeared:
#    p<0.01, p<0.001
#
#  Expected outcomes:
#    - If both < 27.03: α=0.5 confirmed as optimum
#    - If α=0.6 > 27.03: peak is actually at α=0.6, broader "safe" zone
#    - If α=0.4 > 27.03: peak shifts further toward aggressive (surprising)
#
#  4 cells × (1 merge + 3 evals) = 16 bsub jobs, ~45 min wall-clock
#
#  PREREQUISITES (should already be done):
#   - run_pipeline.sh patched with --p5-pval-tag
#   - Re-thresholded label dirs exist for p<0.01 and p<0.001
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

# Sanity checks
if ! grep -q "P5_PVAL_TAG" code/run_pipeline.sh; then
    echo "ERROR: run_pipeline.sh not patched. Run: bash code/patch_pvaltag.sh"
    exit 1
fi
for tag in p0.01 p0.001; do
    f="results/3-classify/full/llava-next-llama3-8b/llm_permutation_gate_up_min100_max2048_${tag}/neuron_labels_permutation_all.json"
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing re-thresholded labels: $f"
        exit 1
    fi
done

LOG=/tmp/a3_bound_dispatch.log
> "$LOG"
echo "===== A3 bound dispatch (LLaVA, α=0.4 and α=0.6) — $(date) =====" | tee -a "$LOG"

# Same benchmarks as the main grid for consistency
BENCHES="MathVerse_MINI_Text_Dominant,POPE,MathVista_MINI"

# Cells to run
P_TAGS=("p0.01" "p0.001")
ALPHAS=("0.4" "0.6")

N_TOTAL=0
for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${ALPHAS[@]}"; do
        N_TOTAL=$((N_TOTAL + 1))
        _label="${P_TAG}_a${ALPHA_TEXT}"

        echo ""                                                              | tee -a "$LOG"
        echo "=== [$N_TOTAL/4] LLaVA bound α=${ALPHA_TEXT} at ${P_TAG} ==="  | tee -a "$LOG"

        bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            --p5-pval-tag "$P_TAG" \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

echo ""                                                               | tee -a "$LOG"
echo "===== Bound dispatch complete — $(date) ====="                 | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep -E 'pt[46]v10m10.*p0\\.00?1'"
echo ""
echo "Output locations:"
echo "  results/25-merge/llava-next-llama3-8b/dart-prop/pmbt_t{0.4,0.6}_v1.0_m1.0_a1.0_o1.0_{p0.01,p0.001}/"
echo ""
echo "When done (~45 min):"
echo "  python3 code/a3_grid_extract_v2.py"
echo ""
echo "Note: a3_grid_extract_v2.py will show pending for α=0.4 and α=0.6 rows"
echo "since it only loops over α ∈ {0.7, 0.5, 0.3}. After this completes,"
echo "extend the extractor to include 0.4 and 0.6."
