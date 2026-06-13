#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A3 QWEN α=0.9 SCOPE-VARIANTS — stricter p thresholds
#
#  Fills the gap: Qwen's main-table-optimal α=0.9 was never tested with
#  scope variants at stricter p thresholds. This mirrors the successful
#  Idefics finding (V1 downproj + α=0.7 + p<0.001) but at Qwen's sweet spot.
#
#  VARIANTS:
#    V1 downproj  --p5-pmbt-scope mlp --p5-mlp-projs down
#    V2 kvmerge   --p5-kv-merge
#    V3 mlponly   --p5-pmbt-scope mlp
#
#  GRID: Qwen, α_text=0.9, p ∈ {p0.01, p0.001}
#  CELLS: 3 variants × 1 α × 2 p = 6 cells × 4 bsubs = 24 jobs
#  Wall clock: ~1-2h
#
#  PREREQUISITES (already satisfied):
#   - run_pipeline.sh patched with --p5-pval-tag
#   - Re-thresholded Qwen label dirs exist for p<0.01 and p<0.001
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

# Sanity checks
if ! grep -q "P5_PVAL_TAG" code/run_pipeline.sh; then
    echo "ERROR: run_pipeline.sh not patched. Run: bash code/patch_pvaltag.sh"
    exit 1
fi
for tag in p0.01 p0.001; do
    f="results/3-classify/full/qwen2-vl-7b/llm_permutation_gate_up_min100_max2048_${tag}/neuron_labels_permutation_all.json"
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing re-thresholded labels: $f"
        echo "Run: python3 code/rethreshold_labels_v2.py"
        exit 1
    fi
done

LOG=/tmp/a3_qwen_a09_scope_variants.log
> "$LOG"
echo "===== A3 Qwen α=0.9 scope-variants — $(date) =====" | tee -a "$LOG"

BENCHES="MathVerse_MINI_Text_Dominant,POPE,MathVista_MINI"
P_TAGS=("p0.01" "p0.001")
ALPHA_TEXT="0.9"

N=0
TOTAL=6

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
        N=$((N + 1))
        echo ""                                                               | tee -a "$LOG"
        echo "=== [QWEN-A0.9 $VARIANT_NAME $N/$TOTAL] α=$ALPHA_TEXT $P_TAG ==="| tee -a "$LOG"

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

echo ""                                                               | tee -a "$LOG"
echo "===== Qwen α=0.9 scope-variants dispatch complete — $(date) =====" | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep -cE '25_qw_qm_pt9v10m10.*_(dp|kv|mlpo)'"
echo ""
echo "Two-phase:"
echo "  Round 1 (this): fires 6 merges"
echo "  Round 2 (re-run after ~15-20 min): fires 18 evals"
echo ""
echo "When done, extract:"
echo "  python3 code/extract_merge_results.py 2>/dev/null | grep '^pmbt_t0.9_v1.0_m1.0_.*_p0\\.0'"
