#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A3 SCOPE-VARIANTS DISPATCH — Qwen + Idefics (both conventions)
#
#  Mirrors the LLaVA scope-variants experiment onto other models.
#  Tests whether the downproj / KV-merge / MLP-only hypotheses generalize.
#
#  VARIANTS:
#    V1 downproj  --mlp_projs down --merge_scope mlp
#    V2 kvmerge   --kv_merge
#    V3 mlponly   --merge_scope mlp (all MLP, no attention)
#
#  GRID: α_text ∈ {0.4, 0.5, 0.6, 0.7}, p ∈ {p0.01, p0.001}
#
#  CELLS:
#    Qwen (α_v=α_m=α_o=1.0):           3 var × 4 α × 2 p = 24 cells
#    Idefics Conv A (α_m=1.0):         3 var × 4 α × 2 p = 24 cells
#    Idefics Conv B (α_m=0.9, BRV):    3 var × 4 α × 2 p = 24 cells
#    TOTAL: 72 cells × 4 bsubs = 288 jobs, 6-10h wall clock
#
#  PREREQUISITES (already satisfied):
#   - run_pipeline.sh patched with --p5-pval-tag
#   - Re-thresholded label dirs exist for qwen2-vl-7b and idefics2-8b
#     (created by rethreshold_labels_v2.py)
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

# Sanity checks
if ! grep -q "P5_PVAL_TAG" code/run_pipeline.sh; then
    echo "ERROR: run_pipeline.sh not patched. Run: bash code/patch_pvaltag.sh"
    exit 1
fi
for model in qwen2-vl-7b idefics2-8b; do
    for tag in p0.01 p0.001; do
        f="results/3-classify/full/${model}/llm_permutation_gate_up_min100_max2048_${tag}/neuron_labels_permutation_all.json"
        if [[ ! -f "$f" ]]; then
            echo "ERROR: Missing re-thresholded labels: $f"
            echo "Run: python3 code/rethreshold_labels_v2.py"
            exit 1
        fi
    done
done

LOG=/tmp/a3_scope_variants_qw_idf.log
> "$LOG"
echo "===== A3 Qwen+Idefics scope-variants — $(date) =====" | tee -a "$LOG"

BENCHES="MathVerse_MINI_Text_Dominant,POPE,MathVista_MINI"
P_TAGS=("p0.01" "p0.001")
ALPHAS=("0.7" "0.6" "0.5" "0.4")

# ═════════════════════════════════════════════════════════════════
# QWEN: α_v=α_m=α_o=1.0, 3 variants × 4 α × 2 p = 24 cells
# ═════════════════════════════════════════════════════════════════

N=0
for VARIANT_NAME in "V1_downproj" "V2_kvmerge" "V3_mlponly"; do
    case "$VARIANT_NAME" in
        V1_downproj)
            SCOPE_FLAGS="--p5-pmbt-scope mlp --p5-mlp-projs down"
            ;;
        V2_kvmerge)
            SCOPE_FLAGS="--p5-kv-merge"
            ;;
        V3_mlponly)
            SCOPE_FLAGS="--p5-pmbt-scope mlp"
            ;;
    esac
    for P_TAG in "${P_TAGS[@]}"; do
        for ALPHA_TEXT in "${ALPHAS[@]}"; do
            N=$((N + 1))
            echo ""                                                          | tee -a "$LOG"
            echo "=== [QWEN $VARIANT_NAME $N/24] α=$ALPHA_TEXT $P_TAG ==="  | tee -a "$LOG"

            bash code/run_pipeline.sh --step 25 --model-type qwen2vl \
                --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
                --p5-mode pmbt --p5-alpha 1.0 \
                --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
                --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
                --p5-pval-tag "$P_TAG" \
                $SCOPE_FLAGS \
                --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
        done
    done
done

# ═════════════════════════════════════════════════════════════════
# IDEFICS Conv A: α_v=1.0, α_m=1.0, α_o=1.0 (LLaVA-parallel)
# ═════════════════════════════════════════════════════════════════

N=0
for VARIANT_NAME in "V1_downproj" "V2_kvmerge" "V3_mlponly"; do
    case "$VARIANT_NAME" in
        V1_downproj)
            SCOPE_FLAGS="--p5-pmbt-scope mlp --p5-mlp-projs down"
            ;;
        V2_kvmerge)
            SCOPE_FLAGS="--p5-kv-merge"
            ;;
        V3_mlponly)
            SCOPE_FLAGS="--p5-pmbt-scope mlp"
            ;;
    esac
    for P_TAG in "${P_TAGS[@]}"; do
        for ALPHA_TEXT in "${ALPHAS[@]}"; do
            N=$((N + 1))
            echo ""                                                               | tee -a "$LOG"
            echo "=== [IDEFICS-A $VARIANT_NAME $N/24] α=$ALPHA_TEXT $P_TAG ===" | tee -a "$LOG"

            bash code/run_pipeline.sh --step 25 --model-type idefics2 \
                --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
                --p5-mode pmbt --p5-alpha 1.0 \
                --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
                --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
                --p5-pval-tag "$P_TAG" \
                $SCOPE_FLAGS \
                --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
        done
    done
done

# ═════════════════════════════════════════════════════════════════
# IDEFICS Conv B: α_v=1.0, α_m=0.9, α_o=1.0 (BRV-original)
# ═════════════════════════════════════════════════════════════════

N=0
for VARIANT_NAME in "V1_downproj" "V2_kvmerge" "V3_mlponly"; do
    case "$VARIANT_NAME" in
        V1_downproj)
            SCOPE_FLAGS="--p5-pmbt-scope mlp --p5-mlp-projs down"
            ;;
        V2_kvmerge)
            SCOPE_FLAGS="--p5-kv-merge"
            ;;
        V3_mlponly)
            SCOPE_FLAGS="--p5-pmbt-scope mlp"
            ;;
    esac
    for P_TAG in "${P_TAGS[@]}"; do
        for ALPHA_TEXT in "${ALPHAS[@]}"; do
            N=$((N + 1))
            echo ""                                                               | tee -a "$LOG"
            echo "=== [IDEFICS-B $VARIANT_NAME $N/24] α=$ALPHA_TEXT $P_TAG ===" | tee -a "$LOG"

            bash code/run_pipeline.sh --step 25 --model-type idefics2 \
                --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
                --p5-mode pmbt --p5-alpha 1.0 \
                --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
                --p5-alpha-multi 0.9 --p5-alpha-other 1.0 \
                --p5-pval-tag "$P_TAG" \
                $SCOPE_FLAGS \
                --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
        done
    done
done

echo ""                                                               | tee -a "$LOG"
echo "===== Scope-variants Qwen+Idefics dispatch complete — $(date) =====" | tee -a "$LOG"
echo ""
echo "Monitor (wait ~10 min for merges, then re-run to trigger evals):"
echo "  bjobs -w | grep -cE '25_(qw_qm|idef_m1).*_(dp|kv|mlpo)'"
echo ""
echo "Two-phase pattern:"
echo "  Round 1 (this dispatch): fires 72 merges. Evals print [wait]."
echo "  Round 2 (re-run after ~20 min): fires 216 evals."
echo ""
echo "bash code/a3_scope_variants_dispatch_qw_idf.sh    # round 2 after merges"
