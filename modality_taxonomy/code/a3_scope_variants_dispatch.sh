#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A3 SCOPE-VARIANTS DISPATCH — test merge-scope ablations at best thresholds
#
#  Motivation:
#   Previous experiments tested mlponly, downproj, KV merge as SEPARATE
#   dispatches at α=0.7/0.8/0.9 p=0.05 default. We never tested them at
#   the new optimal (p<0.01, α=0.5) or nearby corners.
#
#   This dispatch asks: do scope restrictions / KV inclusion help or hurt
#   at the newly-found best threshold?
#
#  VARIANTS:
#    V1 downproj  --mlp_projs down --merge_scope mlp  (MLP down_proj only)
#    V2 kvmerge   --kv_merge                          (all standard + K/V)
#    V3 mlponly   --merge_scope mlp                   (all MLP, no attention)
#
#  GRID: LLaVA, α_text ∈ {0.4, 0.5, 0.6, 0.7}, p ∈ {p0.01, p0.001}
#  TOTAL: 3 variants × 4 α × 2 p = 24 cells, 96 bsubs
#  Wall clock: ~2-4h (shares queue with running A3 grid)
#
#  PREREQUISITES:
#   - run_pipeline.sh patched with --p5-pval-tag
#   - Re-thresholded label dirs exist for p<0.01 and p<0.001 (LLaVA)
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

LOG=/tmp/a3_scope_variants.log
> "$LOG"
echo "===== A3 scope-variants dispatch — $(date) =====" | tee -a "$LOG"

BENCHES="MathVerse_MINI_Text_Dominant,POPE,MathVista_MINI"

P_TAGS=("p0.01" "p0.001")
ALPHAS=("0.7" "0.6" "0.5" "0.4")

N=0
TOTAL=$((3 * ${#P_TAGS[@]} * ${#ALPHAS[@]}))

# ─────────────────────────────────────────────────────────────────
# V1: downproj-only (MLP scope, only the down_proj matrix)
# ─────────────────────────────────────────────────────────────────
for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${ALPHAS[@]}"; do
        N=$((N + 1))
        echo ""                                                          | tee -a "$LOG"
        echo "=== [V1 downproj $N/$TOTAL] α=$ALPHA_TEXT $P_TAG ==="      | tee -a "$LOG"

        bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            --p5-pval-tag "$P_TAG" \
            --p5-pmbt-scope mlp \
            --p5-mlp-projs down \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

# ─────────────────────────────────────────────────────────────────
# V2: KV merge (standard both-scope + attention K/V included)
# ─────────────────────────────────────────────────────────────────
for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${ALPHAS[@]}"; do
        N=$((N + 1))
        echo ""                                                          | tee -a "$LOG"
        echo "=== [V2 kvmerge $N/$TOTAL] α=$ALPHA_TEXT $P_TAG ==="       | tee -a "$LOG"

        bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            --p5-pval-tag "$P_TAG" \
            --p5-kv-merge \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

# ─────────────────────────────────────────────────────────────────
# V3: MLP-only (no attention at all, all three MLP projections)
# ─────────────────────────────────────────────────────────────────
for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${ALPHAS[@]}"; do
        N=$((N + 1))
        echo ""                                                          | tee -a "$LOG"
        echo "=== [V3 mlponly $N/$TOTAL] α=$ALPHA_TEXT $P_TAG ==="       | tee -a "$LOG"

        bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            --p5-pval-tag "$P_TAG" \
            --p5-pmbt-scope mlp \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

echo ""                                                               | tee -a "$LOG"
echo "===== Scope-variants dispatch complete — $(date) ====="        | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep -cE '25_ll3_dp.*_downproj|_kv|_mlponly'"
echo ""
echo "Output dirs will have suffixes encoding the scope choice:"
echo "  pmbt_t0.X_v1.0_m1.0_a1.0_o1.0_downproj_p0.01/"
echo "  pmbt_t0.X_v1.0_m1.0_a1.0_o1.0_kv_p0.01/         (or similar)"
echo "  pmbt_t0.X_v1.0_m1.0_a1.0_o1.0_mlponly_p0.01/   (or similar)"
echo ""
echo "When done, extract by searching the suffixes:"
echo "  python3 code/extract_merge_results.py | grep -E 'pmbt_t0\\.[4567].*_(downproj|kv|mlponly).*_p0\\.00?1'"
