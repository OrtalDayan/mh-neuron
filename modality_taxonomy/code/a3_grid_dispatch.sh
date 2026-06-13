#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  A3 GRID DISPATCH — p-threshold × α_text sensitivity sweep (LLaVA only)
#
#  Motivation: We address two concerns in one sweep:
#   (1) Seed robustness of permutation-based classifier — addressed
#       indirectly by showing the effect is stable across p-thresholds.
#       Stricter thresholds (p<0.01) exclude borderline neurons whose
#       labels would flip across seeds.
#   (2) Hyperparameter optimality — α_text=0.7 was inherited from BRV.
#       We've never tested whether more aggressive blending helps.
#
#  The 2D grid (LLaVA only):
#
#              | p<0.1 | p<0.05 (main) | p<0.01 | p<0.001
#    ---------------------------------------------
#    α_text=0.7 | cell | EXISTS: 26.02 | cell   | cell
#    α_text=0.5 | cell | cell          | cell   | cell
#    α_text=0.3 | cell | cell          | cell   | cell
#
#  12 cells total, 11 new.
#  Each cell: 1 merge + 3 evals (MathVerse-TD + POPE + MathVista_MINI)
#  Total bsubs: 11 merges + 33 evals = 44 jobs
#
#  PREREQUISITES:
#   - patch_pvaltag.sh applied to run_pipeline.sh
#   - rethreshold_labels.py run (generates _p0.1, _p0.01, _p0.001 dirs)
#
#  Prior: α_text values fixed (α_visual=1.0, α_multi=1.0, α_other=1.0)
#  match the existing PMBT config for LLaVA+Dart-Prop.
# ═══════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

# Sanity checks
if ! grep -q "P5_PVAL_TAG" code/run_pipeline.sh; then
    echo "ERROR: run_pipeline.sh not patched. Run: bash code/patch_pvaltag.sh"
    exit 1
fi
for tag in p0.1 p0.01 p0.001; do
    f="results/3-classify/full/llava-next-llama3-8b/llm_permutation_gate_up_min100_max2048_${tag}/neuron_labels_permutation_all.json"
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing re-thresholded labels: $f"
        echo "Run: python3 code/rethreshold_labels.py"
        exit 1
    fi
done

LOG=/tmp/a3_grid_dispatch.log
> "$LOG"
echo "===== A3 grid dispatch — $(date) =====" | tee -a "$LOG"

# Benchmarks: MV-TD + POPE + MathVista (per user request for full suite)
BENCHES="MathVerse_MINI_Text_Dominant,POPE,MathVista_MINI"

# Grid definition
# p=0.05 with α=0.7 already exists in results/ — auto-skipped by pipeline idempotency
P_TAGS=("p0.1" "" "p0.01" "p0.001")   # empty string = default p<0.05 labels
ALPHAS=("0.7" "0.5" "0.3")

N_TOTAL=0
N_SKIP=0

for P_TAG in "${P_TAGS[@]}"; do
    for ALPHA_TEXT in "${ALPHAS[@]}"; do
        N_TOTAL=$((N_TOTAL + 1))
        # Human-readable label for logs
        _label="${P_TAG:-p0.05}_a${ALPHA_TEXT}"

        echo ""                                                          | tee -a "$LOG"
        echo "=== [$N_TOTAL/12] LLaVA $_label ==="                      | tee -a "$LOG"

        # Build pval-tag argument (empty = omit flag = use default p<0.05)
        PVAL_FLAG=""
        [[ -n "$P_TAG" ]] && PVAL_FLAG="--p5-pval-tag $P_TAG"

        bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
            --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
            --p5-mode pmbt --p5-alpha 1.0 \
            --p5-alpha-text "$ALPHA_TEXT" --p5-alpha-visual 1.0 \
            --p5-alpha-multi 1.0 --p5-alpha-other 1.0 \
            $PVAL_FLAG \
            --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"
    done
done

echo ""                                                               | tee -a "$LOG"
echo "===== Dispatch complete — $(date) =====" | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep '25_'                   # all step-25 jobs"
echo "  bjobs -w | grep -cE 'pt[57]v10m10.*p00?1?[0-9]?_?'  # should be up to 44"
echo ""
echo "Output locations:"
echo "  results/25-merge/llava-next-llama3-8b/dart-prop/pmbt_t{0.7,0.5,0.3}_v1.0_m1.0_a1.0_o1.0{,_p0.1,_p0.01,_p0.001}/"
echo ""
echo "Expected wall-clock: ~2-4 hours (11 merges ~3min each + 33 evals varying)"
echo ""
echo "When done, extract the 4×3 grid:"
echo "  python3 code/a3_grid_extract.py"
