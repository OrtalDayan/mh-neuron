#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Step-10 hallucination scoring fan-out for 4 BRV models.
#
# - 3 models have Phase 0 done (idefics2, llava-llama3, qwen25vl-3b)
#   → submit Phase 1 per-layer jobs with --skip_contrastive_prep
# - 1 model has no Phase 0 (qwen2.5-vl-7b)
#   → submit Phase 0 first (no --skip_contrastive_prep, layer 0 only)
#     then we'll submit Phase 1 layers 1..N once Phase 0 finishes
#
# - After all Phase 1 layers per model complete, Phase 2 aggregation runs
#   with a `done()` dependency on every Phase 1 job for that model.
#
# 10B_*_L<N>  = Phase 1 per-layer ablation (~30-60 min × top_k_pct/100 × n_neurons / batch)
# 10C_*       = Phase 2 aggregation (CPU only, 10 min)
# ─────────────────────────────────────────────────────────────────────────

# set -euo pipefail (removed)
WORK_DIR=/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
cd "$WORK_DIR"

VENV=modern_vlms/.venv
QUEUE=waic-risk
WALLTIME="23:59"      # explicit 24h cap so single layer can't run forever

# ── Per-model config: tag | model_type | model_path | n_layers | n_neurons | model_dir ──
declare -A CFG=(
    [idef]="idefics2|modern_vlms/pretrained/idefics2-8b|32|14336|idefics2-8b"
    [ll3]="llava-llama3|llava-hf/llama3-llava-next-8b-hf|32|14336|llava-next-llama3-8b"
    [q3]="qwen25vl-3b|modern_vlms/pretrained/Qwen2.5-VL-3B-Instruct|36|11008|qwen2.5-vl-3b"
    [q27]="qwen2vl|modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct|28|18944|qwen2.5-vl-7b"
)

# Models with Phase 0 already done
P0_DONE=(idef ll3 q3)
# Models needing fresh Phase 0
P0_NEEDED=(q27)

submit_layer_job() {
    local TAG=$1
    local LAYER=$2
    local SKIP_PHASE0=$3    # "yes" → add --skip_contrastive_prep
    IFS='|' read -r MTYPE MPATH NLAYERS NNEURONS MDIR <<< "${CFG[$TAG]}"
    
    local JOBNAME="10B_${TAG}_L${LAYER}_b100"
    local LOG_DIR="logs/full/$MDIR/10-halluc_score"
    local OUT_DIR="results/10-halluc_scores/full/$MDIR"
    mkdir -p "$LOG_DIR" "$OUT_DIR"
    
    local PHASE0_FLAG=""
    [[ "$SKIP_PHASE0" == "yes" ]] && PHASE0_FLAG="--skip_contrastive_prep"
    
    local LABEL_DIR="results/3-classify/full/$MDIR/llm_permutation"
    
    bsub -q $QUEUE -J "$JOBNAME" \
         -gpu "num=1:gmem=40G" \
         -R "rusage[mem=65536]" \
         -oo "$LOG_DIR/${JOBNAME}.log" \
         -eo "$LOG_DIR/${JOBNAME}.err" \
         "cd $WORK_DIR && $VENV/bin/python code/halluc_score_neurons_batch100.py \
            --model_type $MTYPE \
            --model_path $MPATH \
            --model_name $MDIR \
            --n_layers $NLAYERS \
            --n_neurons $NNEURONS \
            --label_dir $LABEL_DIR \
            --pope_path data/POPE/output/coco/coco_pope_random.json \
            --pope_img_dir data/val2014 \
            --triviaqa_path data/triviaqa/qa/verified-web-dev.json \
            --triviaqa_num 2000 \
            --triviaqa_cap 1000 \
            --output_dir $OUT_DIR \
            --contrastive \
            --halluc_triviaqa \
            --top_k_pct 5.0 --batch_neurons 100 \
            $PHASE0_FLAG \
            --layer_start $LAYER \
            --layer_end $((LAYER + 1))"
}

# === Submit Phase 1 fan-out for the 3 models with Phase 0 done ===
echo "=========================================================="
echo "  Step-10 Phase 1 fan-out for 3 models (Phase 0 done)"
echo "=========================================================="
for TAG in "${P0_DONE[@]}"; do
    IFS='|' read -r MTYPE MPATH NLAYERS NNEURONS MDIR <<< "${CFG[$TAG]}"
    echo ""
    echo "→ $MDIR (model_type=$MTYPE, n_layers=$NLAYERS, n_neurons=$NNEURONS)"
    for L in $(seq 0 $((NLAYERS - 1))); do
        submit_layer_job "$TAG" "$L" "yes" 2>&1 | grep -E "Job <" | head -1
    done
done

# === For qwen2.5-vl-7b: submit Phase 0 (layer 0 + prep) ===
# Phase 1 fanout for layers 1..27 will need to wait for Phase 0 to finish
echo ""
echo "=========================================================="
echo "  qwen2.5-vl-7b: Phase 0 (layer 0 + contrastive prep)"
echo "=========================================================="
for TAG in "${P0_NEEDED[@]}"; do
    IFS='|' read -r MTYPE MPATH NLAYERS NNEURONS MDIR <<< "${CFG[$TAG]}"
    echo ""
    echo "→ $MDIR (Phase 0)"
    submit_layer_job "$TAG" "0" "no" 2>&1 | grep -E "Job <" | head -1
    echo "  [NOTE] After this completes, manually submit Phase 1 layers 1..27 with --skip_contrastive_prep"
done

echo ""
echo "=========================================================="
echo "  Summary"
echo "=========================================================="
sleep 5
bjobs -w | grep "10B_" | head -10
echo ""
echo "  Total 10B jobs submitted: $(bjobs -w | grep -c '10B_')"
echo "  Track with:  bjobs -w | grep 10B_"
echo ""
echo "  PHASE 2 NOTE: After all Phase 1 layers complete for a model,"
echo "  re-run halluc_score_neurons_batch100.py with --skip_ablation and full layer range"
echo "  to aggregate into combined_halluc_scores.json and ablation_scores_delta_h*.json"
