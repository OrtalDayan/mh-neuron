#!/bin/bash
# launch_step24_D_all_models.sh
#
# Queues the missing step-24 D-ranking cells across all 7 paper models.
# Pipeline's built-in skip-existing-cells logic (run_pipeline.sh:6785) will
# only submit cells that don't already exist as non-empty files. Verified
# 1945 cells missing across the 7 models per the inventory script.
#
# CRITICAL: We override --p4-sweep-fracs to use double-decimal format
# (0.10 not 0.1) because the existing on-disk cells use that format. Without
# this override, the skip logic would fail to recognize existing cells and
# re-queue all 7 × 432 = 3024 jobs instead of the missing ~1945.
#
# Usage:
#   bash code/launch_step24_D_all_models.sh                # submit all 7
#   bash code/launch_step24_D_all_models.sh llava-llama3   # submit one model
#   bash code/launch_step24_D_all_models.sh dryrun         # preview, don't submit

set -u                                                # error on unset vars; no -e because we want
                                                      # the loop to continue if one model fails

# ── Model list: (display_name, model-type flag value) ──
# model-type → MODEL_NAME mapping is set inside run_pipeline.sh
MODELS=(
    "llava-next-llama3-8b   llava-llama3"
    "idefics2-8b            idefics2"
    "llava-onevision-7b     llava-ov"
    "qwen2-vl-7b            qwen2vl"
    "llava-1.5-7b           llava-liuhaotian"
    "internvl2.5-8b         internvl"
    "qwen2.5-vl-3b          qwen25vl-3b"
)

# ── Step-24 flags shared across all models ──
# - hook-point gate_up: matches the gate_up cells we've been analyzing
# - p4-ranking D: only ranking we now care about (norm/D_x_norm dropped)
# - p4-sweep-fracs: double-decimal format to match existing filenames
# - p4-benchmarks: all 4 paper benches (defaults to only 2)
# - p4-categories: visual,text,multimodal (default — explicit for clarity)
# - long-desc: matches existing label-dir naming (llm_permutation_gate_up_min100_max2048)
COMMON_FLAGS=(
    --step 24
    --hook-point gate_up
    --p4-ranking D
    --p4-sweep-fracs "0.01,0.05,0.10,0.25,0.50,1.00"
    --p4-benchmarks "POPE,MV_Text_Dominant,MV_Vision_Only,TriviaQA"
    --p4-categories "visual,text,multimodal"
    --long-desc
)

# ── Argument handling: optional model filter or dryrun ──
FILTER="${1:-}"
DRYRUN=false
[[ "$FILTER" == "dryrun" ]] && { DRYRUN=true; FILTER=""; }    # treat 'dryrun' as a no-filter flag

# ── Header ──
echo "═══════════════════════════════════════════════════════════"
echo "  Step-24 D-ranking sweep launcher"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════"
$DRYRUN && echo "  (DRYRUN — no jobs submitted)"
[[ -n "$FILTER" ]] && echo "  Filter: only model-type matching '$FILTER'"
echo ""

# ── Per-model launch loop ──
total_subbed=0
total_skipped=0
for entry in "${MODELS[@]}"; do
    # Split "name model_type" into two variables. read -r prevents backslash escapes.
    read -r MODEL_NAME MODEL_TYPE <<< "$entry"

    # Skip if user passed a model-type filter that doesn't match
    [[ -n "$FILTER" && "$MODEL_TYPE" != "$FILTER" ]] && continue

    echo "── $MODEL_NAME ($MODEL_TYPE) ──"

    if $DRYRUN; then
        # Print the command we WOULD run, without executing
        echo "  bash code/run_pipeline.sh --model-type $MODEL_TYPE ${COMMON_FLAGS[*]}"
        echo ""
        continue
    fi

    # Capture pipeline output so we can grep for SUBMITTED/SKIPPED counts at the end
    LOG_TMP=$(mktemp /tmp/step24_launch_${MODEL_TYPE}_XXXX.log)

    # Run the pipeline. The pipeline itself calls bsub for each missing cell.
    bash code/run_pipeline.sh \
        --model-type "$MODEL_TYPE" \
        "${COMMON_FLAGS[@]}" \
        2>&1 | tee "$LOG_TMP"

    # Parse the pipeline submit/skip lines. grep -c can occasionally emit
    # multi-line output that breaks $((...)) arithmetic; strip non-digits
    # and default empties to 0 to make this robust.
    n_sub=$(grep -cE '^\s*\[submit\]' "$LOG_TMP" 2>/dev/null || true)
    n_skip=$(grep -cE '^\s*\[skip\]'   "$LOG_TMP" 2>/dev/null || true)
    n_sub=${n_sub//[^0-9]/}; n_sub=${n_sub:-0}              # strip non-digits, default 0
    n_skip=${n_skip//[^0-9]/}; n_skip=${n_skip:-0}
    total_subbed=$((total_subbed + n_sub))
    total_skipped=$((total_skipped + n_skip))
    echo "  → submitted=$n_sub  skipped=$n_skip"
    echo ""
done

# ── Footer ──
echo "═══════════════════════════════════════════════════════════"
if $DRYRUN; then
    echo "  DRYRUN complete — no jobs submitted"
else
    echo "  TOTAL submitted: $total_subbed"
    echo "  TOTAL skipped (already done): $total_skipped"
    echo ""
    echo "  Monitor with:"
    echo "    bjobs -w | grep '24_'"
    echo "    bjobs -w | awk '\$3==\"PEND\" && \$7~/^24_/' | wc -l   # pending count"
fi
echo "═══════════════════════════════════════════════════════════"