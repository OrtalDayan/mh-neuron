#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Step-10 hallucination scoring fan-out for 4 BRV models.
# (Fixed version: no -W flag, no strict mode, safer error handling)
#
# - 3 models have Phase 0 done (idefics2, llava-llama3, qwen25vl-3b)
#   → submit Phase 1 per-layer jobs with --skip_contrastive_prep
# - 1 model has no Phase 0 (qwen2.5-vl-7b)
#   → submit Phase 0 first (layer 0 only, no skip flag)
#
# 10B_*_L<N>  = Phase 1 per-layer ablation
# ─────────────────────────────────────────────────────────────────────────

WORK_DIR=/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
cd "$WORK_DIR" || { echo "FATAL: cannot cd to $WORK_DIR"; exit 1; }

VENV=modern_vlms/.venv
QUEUE=waic-long

# Skip submission if a job with this name is already queued/running
already_submitted() {
    local jobname=$1
    bjobs -w 2>/dev/null | awk '{print $7}' | grep -qx "$jobname"
}

submit_layer_job() {
    local TAG=$1
    local LAYER=$2
    local SKIP_PHASE0=$3
    local MTYPE=$4
    local MPATH=$5
    local NLAYERS=$6
    local NNEURONS=$7
    local MDIR=$8
    
    local JOBNAME="10B_${TAG}_L${LAYER}"
    
    if already_submitted "$JOBNAME"; then
        echo "  [SKIP] $JOBNAME already submitted"
        return 0
    fi
    
    local LOG_DIR="logs/full/$MDIR/10-halluc_score"
    local OUT_DIR="results/10-halluc_scores/full/$MDIR"
    mkdir -p "$LOG_DIR" "$OUT_DIR"
    
    local PHASE0_FLAG=""
    [[ "$SKIP_PHASE0" == "yes" ]] && PHASE0_FLAG="--skip_contrastive_prep"
    
    local LABEL_DIR="results/3-classify/full/$MDIR/llm_permutation"
    
    # Submit; capture stdout to detect "Job <NNN>" line
    local OUT
    OUT=$(bsub -q $QUEUE -J "$JOBNAME" \
         -gpu "num=1:gmem=40G" \
         -R "rusage[mem=65536]" \
         -oo "$LOG_DIR/${JOBNAME}.log" \
         -eo "$LOG_DIR/${JOBNAME}.err" \
         "cd $WORK_DIR && $VENV/bin/python code/halluc_score_neurons.py \
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
            --top_k_pct 5.0 \
            $PHASE0_FLAG \
            --layer_start $LAYER \
            --layer_end $((LAYER + 1))" 2>&1)
    
    local JID=$(echo "$OUT" | grep -oE "Job <[0-9]+>" | head -1)
    if [[ -n "$JID" ]]; then
        echo "  ✓ $JOBNAME → $JID"
    else
        echo "  ✗ $JOBNAME FAILED: $(echo "$OUT" | tail -3 | head -c 200)"
    fi
}

# ── Phase 0 done: idef, ll3, q3 ──
echo "=========================================================="
echo "  Phase 1 fan-out for 3 models with Phase 0 done"
echo "=========================================================="

echo ""
echo "→ idefics2-8b (32 layers, 14336 neurons)"
for L in $(seq 0 31); do
    submit_layer_job idef "$L" yes idefics2 modern_vlms/pretrained/idefics2-8b 32 14336 idefics2-8b
done

echo ""
echo "→ llava-next-llama3-8b (32 layers, 14336 neurons)"
for L in $(seq 0 31); do
    submit_layer_job ll3 "$L" yes llava-llama3 llava-hf/llama3-llava-next-8b-hf 32 14336 llava-next-llama3-8b
done

echo ""
echo "→ qwen2.5-vl-3b (36 layers, 11008 neurons)"
for L in $(seq 0 35); do
    submit_layer_job q3 "$L" yes qwen25vl-3b modern_vlms/pretrained/Qwen2.5-VL-3B-Instruct 36 11008 qwen2.5-vl-3b
done

# ── Phase 0 needed: q27 (just layer 0 + prep for now) ──
echo ""
echo "=========================================================="
echo "  qwen2.5-vl-7b: Phase 0 (layer 0 + contrastive prep)"
echo "=========================================================="
submit_layer_job q27 0 no qwen2vl modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct 28 18944 qwen2.5-vl-7b

# ── Summary ──
echo ""
echo "=========================================================="
echo "  Summary"
echo "=========================================================="
sleep 5
TOTAL=$(bjobs -w 2>/dev/null | grep -c "10B_")
PEND=$(bjobs -w 2>/dev/null | grep "10B_" | grep -c "PEND")
RUN=$(bjobs -w 2>/dev/null | grep "10B_" | grep -c "RUN")
echo "  Total 10B jobs: $TOTAL (PEND=$PEND, RUN=$RUN)"
echo ""
echo "  Track:  bjobs -w | grep 10B_"
