#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a2b_llava_extract.sh — Collect Round 1 Path A/B results into one table
#
#  Runs after all bjobs clear. Safe to run early (shows PENDING for missing).
#  Usage: bash a2b_llava_extract.sh
# ═══════════════════════════════════════════════════════════════════════════

cd ~/mh-neuron/modality_taxonomy

BASE="results/25-merge/llava-next-llama3-8b/dart-prop"
PMBT_DIR="pmbt_t0.7_v1.0_m1.0_a1.0_o1.0"

echo ""
echo "==========================================================================="
echo "  ROUND 1 — LLaVA Path A/B Results"
echo "==========================================================================="

# (label, dir_suffix) pairs
VARIANTS=(
    "LABEL|"
    "A_rd|_Ard_C0.10"
    "B_pure_002|_Bpure_C0.02"
    "B_pure_010|_Bpure_C0.10"
    "B_rdxnorm|_Brn_C0.10_N2.0"
)

get_score_file() {
    # $1 = variant dir suffix
    # $2 = benchmark pattern (TriviaQA, POPE, MathVerse*Text_Dominant)
    find "$BASE/${PMBT_DIR}${1}" -maxdepth 4 -name "*${2}*score*.csv" ! -path "*T2026*" 2>/dev/null | head -1
}

extract_triviaQA() {
    local f=$(get_score_file "$1" "TriviaQA")
    [[ -f "$f" ]] || { echo "PENDING"; return; }
    awk -F',' 'NR==2 { gsub(/"/, "", $1); printf "%.4f\n", $1 }' "$f"
}

extract_POPE() {
    # Returns F1 | Acc | Precision
    local f=$(get_score_file "$1" "POPE")
    [[ -f "$f" ]] || { echo "PENDING|-|-"; return; }
    # POPE CSV columns: "split","Overall","acc","precision","recall"
    #   where "Overall" column is actually the F1 score.
    # We want the row labeled "Overall" (averaged across splits).
    local f1=$(awk -F',' '$1 == "\"Overall\"" { gsub(/"/, "", $2); print $2 }' "$f")
    local acc=$(awk -F',' '$1 == "\"Overall\"" { gsub(/"/, "", $3); print $3 }' "$f")
    local prec=$(awk -F',' '$1 == "\"Overall\"" { gsub(/"/, "", $4); print $4 }' "$f")
    echo "${f1:-?}|${acc:-?}|${prec:-?}"
}

extract_MVerseTD() {
    local f=$(get_score_file "$1" "MathVerse*Text_Dominant")
    [[ -f "$f" ]] || { echo "PENDING"; return; }
    awk -F',' 'NR==2 { gsub(/"/, "", $1); print $1 }' "$f"
}

echo ""
echo "── TriviaQA (Overall) ──"
printf "  %-12s %-20s %-10s\n" "Variant" "Dir suffix" "Overall"
echo "  -----------------------------------------------"
for v in "${VARIANTS[@]}"; do
    label="${v%%|*}"
    suffix="${v##*|}"
    score=$(extract_triviaQA "$suffix")
    printf "  %-12s %-20s %-10s\n" "$label" "${suffix:-(none)}" "$score"
done

echo ""
echo "── POPE (Overall F1 / Acc / Precision) ──"
printf "  %-12s %-20s %-8s %-8s %-8s\n" "Variant" "Dir suffix" "F1" "Acc" "Prec"
echo "  --------------------------------------------------------------"
for v in "${VARIANTS[@]}"; do
    label="${v%%|*}"
    suffix="${v##*|}"
    scores=$(extract_POPE "$suffix")
    f1="${scores%%|*}"
    rest="${scores#*|}"
    acc="${rest%%|*}"
    prec="${rest#*|}"
    printf "  %-12s %-20s %-8s %-8s %-8s\n" "$label" "${suffix:-(none)}" "$f1" "$acc" "$prec"
done

echo ""
echo "── MathVerse_MINI_Text_Dominant (overall accuracy) ──"
printf "  %-12s %-20s %-8s\n" "Variant" "Dir suffix" "Acc"
echo "  ---------------------------------------------"
for v in "${VARIANTS[@]}"; do
    label="${v%%|*}"
    suffix="${v##*|}"
    score=$(extract_MVerseTD "$suffix")
    printf "  %-12s %-20s %-8s\n" "$label" "${suffix:-(none)}" "$score"
done

echo ""
echo "── Raw score files (for copy-paste verification) ──"
for v in "${VARIANTS[@]}"; do
    label="${v%%|*}"
    suffix="${v##*|}"
    dir="$BASE/${PMBT_DIR}${suffix}"
    echo ""
    echo "  [$label]"
    find "$dir" -maxdepth 4 -name "*score*.csv" ! -path "*T2026*" 2>/dev/null | while read f; do
        echo "    $f"
    done
done

echo ""
echo "── Job status ──"
echo "  Pending/running jobs:  $(bjobs -w 2>/dev/null | grep -c ll3)"
echo "  (If > 0, some evals are still running)"
echo ""
echo "Reference baselines (Tables 1-3 LLaVA Full PMBT):"
echo "  TriviaQA:    77.89 (we just measured this as LABEL smoke test)"
echo "  POPE-F1:     ~87.7 (approximate)"
echo "  MVerse-TD:   need to verify from existing data"
echo ""
echo "== DONE =="