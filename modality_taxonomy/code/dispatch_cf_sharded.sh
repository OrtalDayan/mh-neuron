#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Sharded CF Pilot Dispatch — Multi-GPU
# ═══════════════════════════════════════════════════════════════════
# Submits N parallel shard jobs, then a merge job that waits for all
# shards to finish via bsub's job-dependency mechanism.
#
# Wall time estimates (LLaVA-Next-LLaMA3, 500 samples, 3 layers):
#   N=4:  ~22 minutes
#   N=8:  ~11 minutes
#
# Usage:
#   bash dispatch_cf_sharded.sh --smoke               # 2 shards, 10 samples
#   bash dispatch_cf_sharded.sh --full                # 4 shards (default)
#   bash dispatch_cf_sharded.sh --shards 8 --full     # 8 shards
#   bash dispatch_cf_sharded.sh --queue waic-long --full     # different queue
#   bash dispatch_cf_sharded.sh --help                # full options list
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────
WORK_DIR="${HOME}/mh-neuron/modality_taxonomy"
PYTHON="python3"
SCRIPT="code/pilot_cf_sharded.py"

MODEL_NAME="llava-next-llama3-8b"
MODEL_PATH="llava-hf/llama3-llava-next-8b-hf"
QUEUE="waic-short"
GPU_GMEM="80G"

COCO_CAPTIONS="data/coco_captions.json"
COCO_IMG_DIR="data/coco/train2017"
PMBT_LABEL_DIR="results/3-classify/full/${MODEL_NAME}/llm_permutation_gate_up_min100_max2048"

# ─── Parse args ────────────────────────────────────────────────────
N_SHARDS=4
MODE="full"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards) N_SHARDS="$2"; shift 2 ;;
        --queue)  QUEUE="$2";    shift 2 ;;
        --gmem)   GPU_GMEM="$2"; shift 2 ;;
        --smoke)  MODE="smoke"; shift ;;
        --full)   MODE="full";  shift ;;
        --dry)    MODE="dry";   shift ;;
        -h|--help)
            echo "Usage: $0 [--smoke|--full|--dry] [options]"
            echo ""
            echo "  --smoke           Smoke test (2 shards, 10 samples, ~10 min)"
            echo "  --full            Full pilot (default 4 shards, ~1 hr)"
            echo "  --dry             Print plan without submitting"
            echo "  --shards N        Number of shards (default: 4)"
            echo "  --queue NAME      bsub queue (default: waic-short)"
            echo "                    Common: waic-short, waic-long, waic-medium"
            echo "  --gmem SIZE       GPU memory tier (default: 80G)"
            exit 0 ;;
        *) echo "Unknown arg: $1"; echo "Try $0 --help"; exit 1 ;;
    esac
done

# ─── Mode-specific config ──────────────────────────────────────────
case "$MODE" in
    smoke)
        N_SAMPLES=10
        LAYERS="14"
        K_IMAGE=3
        K_TEXT=3
        N_NOISE=10
        NOISE_K=3
        N_SHARDS=2
        PAIRED_PATH="data/cf_paired_smoketest.json"
        SHARD_DIR="results/cf_pilot_smoketest/shards"
        OUTPUT_DIR="results/cf_pilot_smoketest"
        TAG="smoke"
        ;;
    full)
        N_SAMPLES=500
        LAYERS="14,18,22"
        K_IMAGE=5
        K_TEXT=5
        N_NOISE=500
        NOISE_K=5
        PAIRED_PATH="data/cf_paired_500.json"
        SHARD_DIR="results/cf_pilot_v1/shards"
        OUTPUT_DIR="results/cf_pilot_v1"
        TAG="full"
        ;;
    dry)
        echo "Dry mode: would submit ${N_SHARDS} shards + 1 merge job"
        exit 0
        ;;
esac

LOG_DIR="${WORK_DIR}/logs/cf_pilot"
mkdir -p "$LOG_DIR"

DATESTAMP=$(date +%Y%m%d_%H%M)

echo "═══════════════════════════════════════════════════════════════"
echo "  CF Pilot Dispatch — ${MODE} (${N_SHARDS} shards)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Samples:    ${N_SAMPLES}  Layers: ${LAYERS}"
echo "  K_image:    ${K_IMAGE}    K_text: ${K_TEXT}"
echo "  Noise:      ${N_NOISE} groups × ${NOISE_K}"
echo "  Shard dir:  ${SHARD_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo ""

# ─── Build paired data (one-time, on login node) ───────────────────
if [[ ! -f "${WORK_DIR}/${PAIRED_PATH}" ]]; then
    echo "Building paired data..."
    cd "$WORK_DIR"
    $PYTHON $SCRIPT shard \
        --shard_idx 0 --n_shards 1 \
        --paired_data_path "$PAIRED_PATH" \
        --coco_captions_path "$COCO_CAPTIONS" \
        --n_pilot_samples "$N_SAMPLES" \
        --K_image "$K_IMAGE" --K_text "$K_TEXT" \
        --build_paired_only 2>&1 | head -20 || true
    echo ""
fi

# ─── Common arg string ─────────────────────────────────────────────
COMMON_ARGS="--paired_data_path ${PAIRED_PATH} \
    --coco_captions_path ${COCO_CAPTIONS} \
    --coco_img_dir ${COCO_IMG_DIR} \
    --n_pilot_samples ${N_SAMPLES} \
    --K_image ${K_IMAGE} --K_text ${K_TEXT} \
    --n_noise_pairs ${N_NOISE} --noise_K ${NOISE_K} \
    --layers ${LAYERS} --hook_point gate_up \
    --pmbt_label_dir ${PMBT_LABEL_DIR} \
    --shard_dir ${SHARD_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_PATH} --model_type llava-llama3 \
    --device cuda:0 --seed 42 \
    --skip_paired_build"

# ─── Submit N shard jobs ───────────────────────────────────────────
echo "Submitting ${N_SHARDS} shard jobs..."
SHARD_JOBS=()
for ((s=0; s<N_SHARDS; s++)); do
    JN="cf_${TAG}_shard_${s}_${DATESTAMP}"
    SHARD_JOBS+=("$JN")
    CMD="cd ${WORK_DIR} && ${PYTHON} ${SCRIPT} shard \
        --shard_idx ${s} --n_shards ${N_SHARDS} \
        ${COMMON_ARGS}"

    bsub -q "$QUEUE" \
         -J "$JN" \
         -gpu "num=1:gmem=${GPU_GMEM}" \
         -oo "${LOG_DIR}/${JN}.log" \
         -eo "${LOG_DIR}/${JN}.err" \
         -- "$CMD"
    echo "  Submitted shard $s: $JN"
done

# ─── Submit merge job (depends on all shards completing) ───────────
# Build the dependency string: "done(JOB1) && done(JOB2) && ..."
DEPS=""
for jn in "${SHARD_JOBS[@]}"; do
    if [[ -z "$DEPS" ]]; then
        DEPS="done(${jn})"
    else
        DEPS="${DEPS} && done(${jn})"
    fi
done
echo "  Dependency: $DEPS"
JN_MERGE="cf_${TAG}_merge_${DATESTAMP}"
CMD_MERGE="cd ${WORK_DIR} && ${PYTHON} ${SCRIPT} merge \
    --n_shards ${N_SHARDS} ${COMMON_ARGS}"

bsub -q "$QUEUE" \
     -J "$JN_MERGE" \
     -gpu "num=1:gmem=${GPU_GMEM}" \
     -w "$DEPS" \
     -oo "${LOG_DIR}/${JN_MERGE}.log" \
     -eo "${LOG_DIR}/${JN_MERGE}.err" \
     -- "$CMD_MERGE"
echo ""
echo "Submitted merge: $JN_MERGE"
echo "  Depends on: ${SHARD_JOBS[*]}"
echo ""
echo "Monitor: bjobs"
echo "Logs:    ${LOG_DIR}/cf_${TAG}_*_${DATESTAMP}.{log,err}"
echo ""
echo "When done, results at: ${OUTPUT_DIR}/pilot_summary.json"