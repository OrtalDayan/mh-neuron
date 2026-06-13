#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A3 GRID DISPATCH — Qwen + Idefics (generalization of LLaVA finding)
#
#  Tests whether the (p<0.01, α_text=0.5) optimum found on LLaVA
#  generalizes to Qwen2-VL + Qwen2-Math and Idefics2 + MAmmoTH.
#
#  Grids:
#   QWEN: 4 p × 4 α_text = 16 cells (single convention α_v=α_m=α_o=1.0)
#     α_text ∈ {0.9, 0.7, 0.5, 0.3}   (0.9 = BRV baseline)
#     p ∈ {0.1, 0.05, 0.01, 0.001}
#
#   IDEFICS Conv A: 16 cells (LLaVA-parallel convention, α_m=1.0)
#     α_text ∈ {0.8, 0.7, 0.5, 0.3}   (0.8 = BRV baseline)
#     p ∈ {0.1, 0.05, 0.01, 0.001}
#
#   IDEFICS Conv B: 16 cells (BRV-original convention, α_m=0.9)
#     α_text ∈ {0.8, 0.7, 0.5, 0.3}
#     p ∈ {0.1, 0.05, 0.01, 0.001}
#
#  Total: 48 cells. Some baselines exist from A1 (auto-skipped by pipeline).
#  Per cell: 1 merge + 3 evals = 4 bsubs
#  Estimated: ~184 bsubs, 4-8h wall-clock
#
#  PREREQUISITES:
#   - run_pipeline.sh patched with --p5-pval-tag
#   - rethreshold_labels_v2.py run for qwen2-vl-7b and idefics2-8b
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

# Sanity checks
if ! grep -q "P5_PVAL_TAG" code/run_pipeline.sh; then
    echo "ERROR: run_pipeline.sh not patched. Run: bash code/patch_pvaltag.sh"
    exit 1
fi

for model in qwen2-vl-7b idefics2-8b; do
    for tag in p0.1 p0.01 p0.001; do
        f="results/3-classify/full/${model}/llm_permutation_gate_up_min100_max2048_${tag}/neuron_labels_permutation_all.json"
        if [[ ! -f "$f" ]]; then
            echo "ERROR: Missing re-thresholded labels: $f"
            echo "Run: python3 code/rethreshold_labels_v2.py"
            exit 1
        fi
    done
done

LOG=/tmp/a3_grid_qwen_idefics.log
> "$LOG"
echo "===== A3 Qwen+Idefics grid — $(date) =====" | tee -a "$LOG"

BENCHES="MathVerse_MINI_Text_Dominant,POPE,MathVista_MINI"
P_TAGS=("p0.1" "" "p0.01" "p0.001")   # '' = default p<0.05

# ═════════════════════════════════════════════════════════════════
# QWEN: single convention (α_v=α_m=α_o=1.0), α_text ∈ {0.9, 0.7, 0.5, 0.3}
# ═════════════════════════════════════════════════════════════════

QWEN_ALPHAS=("0.9" "0.7" "0.5" "0.3")
N=0

for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${QWEN_ALPHAS[@]}"; do
        N=$((N + 1))
        _label="${P_TAG:-p0.05}_a${ALPHA_TEXT}"

        echo ""                                                          | tee -a "$LOG"
        echo "=== [QWEN $N/16] ${_label} ==="                            | tee -a "$LOG"

        PVAL_FLAG=""
        [[ -n "$P_TAG" ]] && PVAL_FLAG="--p5-pval-tag $P_TAG"

        bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            $PVAL_FLAG \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

# ═════════════════════════════════════════════════════════════════
# IDEFICS Conv A: α_v=1.0, α_m=1.0 (LLaVA-parallel)
# ═════════════════════════════════════════════════════════════════

IDEFICS_ALPHAS=("0.8" "0.7" "0.5" "0.3")
N=0

for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${IDEFICS_ALPHAS[@]}"; do
        N=$((N + 1))
        _label="Conv-A_${P_TAG:-p0.05}_a${ALPHA_TEXT}"

        echo ""                                                          | tee -a "$LOG"
        echo "=== [IDEFICS-A $N/16] ${_label} ==="                      | tee -a "$LOG"

        PVAL_FLAG=""
        [[ -n "$P_TAG" ]] && PVAL_FLAG="--p5-pval-tag $P_TAG"

        bash code/run_pipeline.sh --step 25 --model-type idefics2 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            $PVAL_FLAG \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

# ═════════════════════════════════════════════════════════════════
# IDEFICS Conv B: α_v=1.0, α_m=0.9 (BRV original)
# ═════════════════════════════════════════════════════════════════

N=0

for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${IDEFICS_ALPHAS[@]}"; do
        N=$((N + 1))
        _label="Conv-B_${P_TAG:-p0.05}_a${ALPHA_TEXT}"

        echo ""                                                          | tee -a "$LOG"
        echo "=== [IDEFICS-B $N/16] ${_label} ==="                      | tee -a "$LOG"

        PVAL_FLAG=""
        [[ -n "$P_TAG" ]] && PVAL_FLAG="--p5-pval-tag $P_TAG"

        bash code/run_pipeline.sh --step 25 --model-type idefics2 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 0.9 --p5-alpha-other 1.0 \
            $PVAL_FLAG \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

echo ""                                                               | tee -a "$LOG"
echo "===== Qwen+Idefics dispatch complete — $(date) ====="          | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep -E '25_(qvl|idf)' | wc -l"
echo ""
echo "When done (4-8h):"
echo "  python3 code/a3_grid_extract_all.py"
