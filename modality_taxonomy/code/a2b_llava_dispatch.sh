#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  a2b_llava_dispatch.sh — 4-variant signal-mode experiment on LLaVA
#
#  Tests whether rate_diff-based or continuous-α signal modes outperform the
#  categorical PMBT baseline (hard labels from classifier).
#
#  Variants dispatched:
#    1. Path A (A_rd)     : rate_diff magnitude as confidence, A2-style formula
#                           α = 1 − (|D|/C)(1−α_label)
#                           Still uses categorical α_label; structurally a no-op
#                           for visual/multi when α_label=1.0 (control variant).
#
#    2. Path B C=0.02     : near-categorical continuous α
#                           Neurons saturate to full α_text/α_visual quickly;
#                           only boundary neurons get smooth interpolation.
#
#    3. Path B C=0.10     : moderate continuous α (main hypothesis)
#                           Median text neuron (rd=-0.012) gets α≈0.876;
#                           only p95 text tail (rd≤-0.12) reaches full α_text=0.7.
#                           Most text neurons treated as "mild text."
#
#    4. Path B × norm     : continuous α weighted by down_proj norm
#                           Combines modality preference with functional impact
#                           (Step 24 showed D×norm identifies high-impact neurons).
#
#  Baseline for comparison: existing hard-label PMBT (Tables 1-3 row).
#
#  Benchmarks (Round 1): POPE, MathVerse_MINI_Text_Dominant
#    → 4 variants × 1 merge + 2 evals = 12 bsubs total on waic-risk.
#  Wall clock: ~1.5-2h (LLaVA merge ~3 min + evals ~30-60 min each).
# ═══════════════════════════════════════════════════════════════════════════

set -u
cd ~/mh-neuron/modality_taxonomy

LOG=/tmp/a2b_llava_dispatch.log
> "$LOG"
echo "===== A2→B LLaVA dispatch — $(date) =====" | tee -a "$LOG"

# Benchmarks for Round 1 (minimal decision set)
BENCHES="POPE,MathVerse_MINI_Text_Dominant"

# LLaVA α triplet from existing Full PMBT (Tables 1-3)
AT=0.7       # α_text
AV=1.0       # α_visual
AM=1.0       # α_multi
AO=1.0       # α_other (layernorms, embeddings)

# Signal scale params — chosen from rate_diff probe
C_small=0.02   # near-categorical; saturates >p75 of text neurons
C_med=0.10     # moderate; saturates ~p95 of text neurons (main variant)
N_scale=2.0    # down_proj norm scale; matches LLaVA's ~max (2.125)
               # Chosen so that median neuron (norm=0.68) gets w_j = 0.34
               # (strong gating toward α_multi), while outlier neurons (norm≈2)
               # saturate to w_j = 1.0 (full continuous effect).
               # Qwen max ≈ 1.82, Idefics2 max ≈ 0.37 — use different N_scale
               # per model in Round 2 (or renormalize formula).

# ─────────────────────────────────────────────────────────────────
# Variant 1: Path A (A_rd) at C=0.10
# ─────────────────────────────────────────────────────────────────
echo ""                                                                | tee -a "$LOG"
echo "=== [1/4] Path A (A_rd) — rate_diff as confidence, C=$C_med ===" | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode A_rd \
    --p5-signal-scale-c $C_med \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Variant 2: Path B_pure at C=0.02 (near-categorical)
# ─────────────────────────────────────────────────────────────────
echo ""                                                                       | tee -a "$LOG"
echo "=== [2/4] Path B_pure — continuous α, C=$C_small (near-categorical) ===" | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_pure \
    --p5-signal-scale-c $C_small \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Variant 3: Path B_pure at C=0.10 (main hypothesis)
# ─────────────────────────────────────────────────────────────────
echo ""                                                                     | tee -a "$LOG"
echo "=== [3/4] Path B_pure — continuous α, C=$C_med (main variant) ==="   | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_pure \
    --p5-signal-scale-c $C_med \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Variant 4: Path B × norm at C=0.10
# ─────────────────────────────────────────────────────────────────
echo ""                                                                            | tee -a "$LOG"
echo "=== [4/4] Path B_rdxnorm — rate_diff × norm, C=$C_med N=$N_scale ==="       | tee -a "$LOG"
bash code/run_pipeline.sh --step 25 --model-type llava-llama3 \
    --gmem 80 --queue waic-risk --hook-point gate_up --long-desc \
    --p5-mode pmbt --p5-alpha 1.0 \
    --p5-alpha-text $AT --p5-alpha-visual $AV --p5-alpha-multi $AM \
    --p5-alpha-other $AO \
    --p5-signal-mode B_rdxnorm \
    --p5-signal-scale-c $C_med \
    --p5-signal-norm-scale $N_scale \
    --p5-benchmarks "$BENCHES" 2>&1 | tee -a "$LOG"

echo ""                                                                | tee -a "$LOG"
echo "===== Dispatch complete — $(date) =====" | tee -a "$LOG"
echo ""
echo "Monitor:"
echo "  bjobs -w | grep -E 'Ard|Bp|Brn'       # job names contain variant tag"
echo "  bjobs -w | grep -c merge              # expect 4 merge jobs"
echo ""
echo "Output directories (one per variant):"
echo "  results/25-merge/llava-next-llama3-8b/dart-prop/"
echo "    pmbt_t${AT}_v${AV}_m${AM}_a1.0_o${AO}_Ard_C${C_med}/"
echo "    pmbt_t${AT}_v${AV}_m${AM}_a1.0_o${AO}_Bpure_C${C_small}/"
echo "    pmbt_t${AT}_v${AV}_m${AM}_a1.0_o${AO}_Bpure_C${C_med}/"
echo "    pmbt_t${AT}_v${AV}_m${AM}_a1.0_o${AO}_Brn_C${C_med}_N${N_scale}/"
echo ""
echo "After jobs finish (~1.5-2h), extract results:"
echo "  find results/25-merge/llava-next-llama3-8b -path '*_Ard_*' -o -path '*_Bpure_*' -o -path '*_Brn_*' | \\"
echo "    xargs -I{} find {} -name '*_score.csv' 2>/dev/null"
