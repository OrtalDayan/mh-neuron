#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a1_extract.sh — Collect A1 2×2 factorial results
#
#  Usage: bash a1_extract.sh
#
#  Shows the 2×2 per model:
#                   | Full scope   | L16+ (SRF-scope)
#    Uniform α      | Cell C (Tab) | Cell A (NEW)
#    PMBT per-cat   | Cell B (Tab) | Cell D (NEW)
#
#  Reference cells (B and C) come from existing Tables 1-3 runs.
#  New cells (A and D) come from the a1_2x2_dispatch.sh jobs.
# ═══════════════════════════════════════════════════════════════════════════

cd ~/mh-neuron/modality_taxonomy

extract_score() {
    # $1 = search dir, $2 = benchmark pattern
    local f=$(find "$1" -maxdepth 5 -name "*${2}*score*.csv" ! -path "*T2026*" 2>/dev/null | head -1)
    [[ -f "$f" ]] || { echo "PENDING|-"; return; }
    if [[ "$2" == "POPE" ]]; then
        local f1=$(awk -F',' '$1 == "\"Overall\"" { gsub(/"/, "", $2); print $2 }' "$f")
        local acc=$(awk -F',' '$1 == "\"Overall\"" { gsub(/"/, "", $3); print $3 }' "$f")
        echo "${f1:-?}|${acc:-?}"
    elif [[ "$2" == *"MathVerse"* ]]; then
        local acc=$(awk -F',' 'NR==2 { gsub(/"/, "", $1); print $1 }' "$f")
        echo "${acc:-?}|-"
    else
        local overall=$(awk -F',' 'NR==2 { gsub(/"/, "", $1); printf "%.4f", $1 }' "$f")
        echo "${overall:-?}|-"
    fi
}

# Model, VLM dir, PMBT α triplet suffix
# For each model, list the directories to check
print_model_2x2() {
    local model="$1"
    local base="$2"
    local pmbt_dir="$3"
    local uniform_dir="$4"
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  $model"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "  2×2 CELLS:"
    echo "    Cell A (Uniform+L16): $base/${uniform_dir}_L16-end/"
    echo "    Cell B (PMBT+full):   $base/${pmbt_dir}/"
    echo "    Cell C (Uniform+full): $base/${uniform_dir}/"
    echo "    Cell D (PMBT+L16):    $base/${pmbt_dir}_L16-end/"
    echo ""
    
    # Extract POPE F1 for all 4 cells
    local a_pope=$(extract_score "$base/${uniform_dir}_L16-end" "POPE")
    local b_pope=$(extract_score "$base/${pmbt_dir}" "POPE")
    local c_pope=$(extract_score "$base/${uniform_dir}" "POPE")
    local d_pope=$(extract_score "$base/${pmbt_dir}_L16-end" "POPE")
    
    local a_pope_f1="${a_pope%%|*}"
    local b_pope_f1="${b_pope%%|*}"
    local c_pope_f1="${c_pope%%|*}"
    local d_pope_f1="${d_pope%%|*}"
    
    # Extract MVerse-TD
    local a_mv=$(extract_score "$base/${uniform_dir}_L16-end" "MathVerse*Text_Dominant")
    local b_mv=$(extract_score "$base/${pmbt_dir}" "MathVerse*Text_Dominant")
    local c_mv=$(extract_score "$base/${uniform_dir}" "MathVerse*Text_Dominant")
    local d_mv=$(extract_score "$base/${pmbt_dir}_L16-end" "MathVerse*Text_Dominant")
    
    local a_mv_acc="${a_mv%%|*}"
    local b_mv_acc="${b_mv%%|*}"
    local c_mv_acc="${c_mv%%|*}"
    local d_mv_acc="${d_mv%%|*}"
    
    echo "  POPE-F1:"
    printf "  %-20s  %-12s  %-12s\n" "" "Full scope" "L16+ (SRF)"
    printf "  %-20s  %-12s  %-12s\n" "Uniform α" "$c_pope_f1" "$a_pope_f1"
    printf "  %-20s  %-12s  %-12s\n" "PMBT per-cat" "$b_pope_f1" "$d_pope_f1"
    
    echo ""
    echo "  MathVerse-TD acc:"
    printf "  %-20s  %-12s  %-12s\n" "" "Full scope" "L16+ (SRF)"
    printf "  %-20s  %-12s  %-12s\n" "Uniform α" "$c_mv_acc" "$a_mv_acc"
    printf "  %-20s  %-12s  %-12s\n" "PMBT per-cat" "$b_mv_acc" "$d_mv_acc"
    
    # 2×2 interaction analysis (only if all 4 cells have MV-TD numbers)
    echo ""
    if [[ "$a_mv_acc" =~ ^[0-9.]+$ ]] && [[ "$b_mv_acc" =~ ^[0-9.]+$ ]] && \
       [[ "$c_mv_acc" =~ ^[0-9.]+$ ]] && [[ "$d_mv_acc" =~ ^[0-9.]+$ ]]; then
        python3 -c "
a=$a_mv_acc; b=$b_mv_acc; c=$c_mv_acc; d=$d_mv_acc
print(f'  2×2 Interaction (MV-TD):')
print(f'    Layer-restriction effect on Uniform:   A - C = {a-c:+.2f}')
print(f'    Layer-restriction effect on PMBT:      D - B = {d-b:+.2f}')
print(f'    PMBT effect at full scope:             B - C = {b-c:+.2f}')
print(f'    PMBT effect at L16+ scope:             D - A = {d-a:+.2f}')
print(f'    Interaction: (D-B) - (A-C) =           {(d-b)-(a-c):+.2f}')
print(f'      (Positive: PMBT benefits MORE from L16+ scope; Negative: less)')
"
    else
        echo "  (2×2 interaction pending — need all 4 cells)"
    fi
}

echo "══════════════════════════════════════════════════════════════════════════"
echo "  A1 2×2 FACTORIAL RESULTS — Layer Scheduling × PMBT"
echo "══════════════════════════════════════════════════════════════════════════"

print_model_2x2 "LLaVA-Next-LLaMA3 + Dart-Math" \
    "results/25-merge/llava-next-llama3-8b/dart-prop" \
    "pmbt_t0.7_v1.0_m1.0_a1.0_o1.0" \
    "uniform_a0.9"

print_model_2x2 "Qwen2-VL + Qwen2-Math" \
    "results/25-merge/qwen2-vl-7b/qwen2-math" \
    "pmbt_t0.9_v1.0_m1.0_a1.0_o1.0" \
    "uniform_a0.9"

print_model_2x2 "Idefics2 + MAmmoTH" \
    "results/25-merge/idefics2-8b/mammoth1" \
    "pmbt_t0.8_v1.0_m0.9_a1.0_o1.0" \
    "uniform_a0.9"

echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job status: $(bjobs -w 2>/dev/null | grep -c -E 'L16|_L1')"
echo "══════════════════════════════════════════════════════════════════════════"
