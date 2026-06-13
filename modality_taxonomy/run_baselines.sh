#!/bin/bash
# run_baselines.sh — evaluate VCD/ICD/SID on POPE+CHAIR via the pipeline's conventions
set -euo pipefail
# Pull the same defaults run_pipeline.sh uses
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
MODEL_TYPE="${MODEL_TYPE:-llava-ov}"
MODEL_NAME="${MODEL_NAME:-llava-onevision-7b}"
MODE_DIR="${MODE_DIR:-full}"
QUEUE="${QUEUE:-waic-risk}"
POPE_PATH="${POPE_PATH:-data/POPE/output/coco/coco_pope_random.json}"
POPE_IMG_DIR="${POPE_IMG_DIR:-data/val2014}"
CHAIR_ANN_PATH="${CHAIR_ANN_PATH:-data/annotations/instances_val2014.json}"
CHAIR_NUM_IMAGES="${CHAIR_NUM_IMAGES:-500}"
VLMEVAL_PY="$(pwd)/modern_vlms/.venv/bin/python"
LOCAL="${LOCAL:-false}"
mkdir -p logs/baselines
for METHOD in vcd icd sid; do
    OUT_DIR="results/19-evaluate/${MODE_DIR}/${MODEL_NAME}/baseline_${METHOD}"
    JOB="bl_${METHOD}_${MODEL_NAME}"
    if [[ -f "$OUT_DIR/summary.json" ]]; then
        echo "[skip] $JOB — already done ($OUT_DIR/summary.json)"
        continue
    fi
    CMD="$VLMEVAL_PY code/eval_decoding_baselines.py \
        --vlm_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --method $METHOD \
        --pope_path $POPE_PATH \
        --pope_img_dir $POPE_IMG_DIR \
        --coco_ann_path $CHAIR_ANN_PATH \
        --coco_img_dir $POPE_IMG_DIR \
        --n_images $CHAIR_NUM_IMAGES \
        --output_dir $OUT_DIR"
    if [[ "$LOCAL" == "true" ]]; then
        echo "→ running $JOB locally"
        $CMD 2>&1 | tee logs/baselines/${METHOD}.log
    else
        echo "→ submitting $JOB"
        bsub -q "$QUEUE" -J "$JOB" \
            -gpu "num=1:gmem=80G" \
            -R "rusage[mem=65536] order[-gpu_maxfactor]" \
            -oo logs/baselines/${METHOD}.log \
            -eo logs/baselines/${METHOD}.err \
            "cd $(pwd) && $CMD"
    fi
done
