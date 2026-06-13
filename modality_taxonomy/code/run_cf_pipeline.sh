#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_cf_pipeline.sh — Counterfactual Perturbation Classifier pipeline
# ═══════════════════════════════════════════════════════════════════
# Companion to run_pipeline.sh. Adds two new steps:
#   - build_cf_paired: build COCO paired-data file (login node, no GPU)
#   - cf_classify:     run CF classifier (sharded across GPUs by layer range)
#
# Reuses run_pipeline.sh's conventions:
#   - Python venv selection by model type (modern_vlms/.venv for non-InternVL
#     modern VLMs)
#   - Queue and GPU memory tier escalation (waic-short → waic-long)
#   - Output directory naming (results/3-classify/<MODE>/<MODEL_NAME>/...)
#   - Skip detection (won't redo already-completed work)
#   - Log directory structure (logs/<MODE>/<MODEL_NAME>/3-classify/)
#
# Usage:
#   # Step 1: build paired data (one-time, on login node)
#   bash code/run_cf_pipeline.sh --step build_cf_paired --model-type llava-llama3
#   bash code/run_cf_pipeline.sh --step build_cf_paired --model-type llava-llama3 --smoke
#
#   # Step 2: run CF classifier (sharded across GPUs)
#   bash code/run_cf_pipeline.sh --step cf_classify --model-type llava-llama3
#   bash code/run_cf_pipeline.sh --step cf_classify --model-type llava-llama3 --smoke
#   bash code/run_cf_pipeline.sh --step cf_classify --model-type llava-llama3 --layers 14,18,22
#   bash code/run_cf_pipeline.sh --step cf_classify --model-type llava-llama3 --shards 8
#
# Smoke test (--smoke): 10 samples × 1 layer; takes ~5 minutes total.
# Full run:             500 samples × all layers; takes 30-60 minutes on
#                       4 A100s with sharding.
#
# Options:
#   --step <name>        build_cf_paired | cf_classify | all
#   --model-type <name>  llava-llama3 | qwen2vl | idefics2 | internvl | llava-ov
#   --model-path <path>  override default model path
#   --smoke              quick smoke test (10 samples, 1 layer)
#   --layers <list>      comma-separated layer indices (default: all)
#   --shards <N>         number of GPU shards for cf_classify (default: 4)
#   --queue <q>          bsub queue (default: waic-short)
#   --gmem <list>        GPU memory tiers (default: 80G,40G,10G)
#   --gmem-wait <sec>    seconds before tier escalation (default: 120)
#   --n-samples <N>      override default number of samples (full=500, smoke=10)
#   --K-image <K>        image variants per sample (default: 5)
#   --K-text <K>         text variants per sample (default: 5)
#   --noise-K <K>        random pairings per noise group (default: 5)
#   --n-noise-pairs <N>  number of noise groups (default: 500)
#   --hook-point <name>  gate | gate_up | attn (default: gate_up)
#   --skip-noise-pass    skip noise pass; 3-way classification
#   --local              run locally without bsub
#   --output-suffix <s>  appended to llm_cf_permutation_*<suffix>
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────
WORK_DIR="${HOME}/mh-neuron/modality_taxonomy"
STEP="all"
MODEL_TYPE="llava-llama3"
MODEL_PATH=""              # filled in by per-backend defaults
MODEL_NAME=""              # filled in by per-backend defaults
SMOKE=false
LAYERS=""                  # empty = all layers
SHARDS=""                  # empty = auto-default to one shard per layer
QUEUE="waic-short"
GPU_GMEM_TIERS=("80G")     # default to 80G only; pass --gmem 80G,40G,10G for fallback
GMEM_WAIT=120
N_SAMPLES=""               # filled in by smoke vs full
K_IMAGE=5
K_TEXT=5
NOISE_K=5
N_NOISE_PAIRS=500
HOOK_POINT="gate_up"
SKIP_NOISE_PASS=false
LOCAL=false
OUTPUT_SUFFIX=""
N_PERMUTATIONS=1000
ALPHA=0.05
NOISE_PERCENTILE=95.0
SEED=42

# Data paths (match run_pipeline.sh conventions)
COCO_ANN_PATH="/home/projects/bagon/shared/coco2017/annotations/captions_train2017.json"
COCO_IMG_DIR="/home/projects/bagon/shared/coco2017/images/train2017/"
DETAIL_23K_PATH="${WORK_DIR}/data/detail_23k.json"

# ─── Parse args ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)            STEP="$2"; shift 2 ;;
        --model-type)      MODEL_TYPE="$2"; shift 2 ;;
        --model-path)      MODEL_PATH="$2"; shift 2 ;;
        --model-name)      MODEL_NAME="$2"; shift 2 ;;
        --smoke)           SMOKE=true; shift ;;
        --layers)          LAYERS="$2"; shift 2 ;;
        --shards)          SHARDS="$2"; shift 2 ;;
        --queue)           QUEUE="$2"; shift 2 ;;
        --gmem)            IFS=',' read -ra GPU_GMEM_TIERS <<< "$2"; shift 2 ;;
        --gmem-wait)       GMEM_WAIT="$2"; shift 2 ;;
        --n-samples)       N_SAMPLES="$2"; shift 2 ;;
        --K-image)         K_IMAGE="$2"; shift 2 ;;
        --K-text)          K_TEXT="$2"; shift 2 ;;
        --noise-K)         NOISE_K="$2"; shift 2 ;;
        --n-noise-pairs)   N_NOISE_PAIRS="$2"; shift 2 ;;
        --hook-point)      HOOK_POINT="$2"; shift 2 ;;
        --skip-noise-pass) SKIP_NOISE_PASS=true; shift ;;
        --local)           LOCAL=true; shift ;;
        --output-suffix)   OUTPUT_SUFFIX="$2"; shift 2 ;;
        --coco-ann-path)   COCO_ANN_PATH="$2"; shift 2 ;;
        --coco-img-dir)    COCO_IMG_DIR="$2"; shift 2 ;;
        --detail-23k-path) DETAIL_23K_PATH="$2"; shift 2 ;;
        --n-permutations)  N_PERMUTATIONS="$2"; shift 2 ;;
        --alpha)           ALPHA="$2"; shift 2 ;;
        --noise-percentile) NOISE_PERCENTILE="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        -h|--help)
            sed -n '4,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$WORK_DIR"

# ─── Per-backend MODEL_PATH defaults (matches run_pipeline.sh) ─────
if [[ -z "$MODEL_PATH" ]]; then
    case "$MODEL_TYPE" in
        llava-llama3)  MODEL_PATH="llava-hf/llama3-llava-next-8b-hf" ;;
        llava-mistral) MODEL_PATH="llava-hf/llava-v1.6-mistral-7b-hf" ;;
        qwen2vl)       MODEL_PATH="modern_vlms/pretrained/Qwen2-VL-7B-Instruct" ;;
        qwen25vl-7b)   MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct" ;;
        qwen25vl-3b)   MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-3B-Instruct" ;;
        idefics2)      MODEL_PATH="modern_vlms/pretrained/idefics2-8b" ;;
        internvl)      MODEL_PATH="modern_vlms/pretrained/InternVL2_5-8B" ;;
        llava-ov)
            _OV_SNAP="$WORK_DIR/.cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots"
            if [[ -d "$_OV_SNAP" ]]; then
                MODEL_PATH="$_OV_SNAP/$(ls "$_OV_SNAP" | head -1)"
            else
                MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"
            fi
            ;;
        *) echo "ERROR: unknown model_type $MODEL_TYPE"; exit 1 ;;
    esac
fi

# ─── Per-backend MODEL_NAME defaults ───────────────────────────────
if [[ -z "$MODEL_NAME" ]]; then
    case "$MODEL_TYPE" in
        llava-llama3)  MODEL_NAME="llava-next-llama3-8b" ;;
        llava-mistral) MODEL_NAME="llava-next-mistral-7b" ;;
        qwen2vl)       MODEL_NAME="qwen2-vl-7b" ;;
        qwen25vl-7b)   MODEL_NAME="qwen2.5-vl-7b" ;;
        qwen25vl-3b)   MODEL_NAME="qwen2.5-vl-3b" ;;
        idefics2)      MODEL_NAME="idefics2-8b" ;;
        internvl)      MODEL_NAME="internvl2.5-8b" ;;
        llava-ov)      MODEL_NAME="llava-ov-7b" ;;
    esac
fi

# ─── Python interpreter selection (matches run_pipeline.sh) ─────────
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: $PYTHON not found"; exit 1
    fi
elif [[ "$MODEL_TYPE" =~ ^(qwen2vl|qwen25vl-7b|qwen25vl-3b|idefics2|llava-ov|llava-mistral|llava-llama3)$ ]]; then
    PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: $PYTHON not found"; exit 1
    fi
    unset VIRTUAL_ENV 2>/dev/null || true
    if [[ -n "${PYTHONPATH:-}" ]]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd ':' -)
        export PYTHONPATH
    fi
else
    PYTHON="$WORK_DIR/.venv/bin/python"
fi

# Shared HF cache
export HF_HOME="${HF_HOME:-$WORK_DIR/.cache/huggingface}"

# ─── Smoke vs full defaults ────────────────────────────────────────
if $SMOKE; then
    [[ -z "$N_SAMPLES" ]] && N_SAMPLES=10
    [[ -z "$LAYERS" ]] && LAYERS="14"
    K_IMAGE=3
    K_TEXT=3
    NOISE_K=3
    N_NOISE_PAIRS=10
    PAIRED_DATA_PATH="data/cf_paired_smoketest.json"
    LOG_TAG="smoke"
else
    [[ -z "$N_SAMPLES" ]] && N_SAMPLES=500
    PAIRED_DATA_PATH="data/cf_paired_${N_SAMPLES}.json"
    LOG_TAG="full"
fi

# ─── Output / log directories (match run_pipeline.sh structure) ────
MODE_DIR="full"
OUTPUT_DIR="results/3-classify/$MODE_DIR"
LOG_DIR="logs/$MODE_DIR/$MODEL_NAME/3-classify"
mkdir -p "$LOG_DIR"

DATESTAMP=$(date +%Y%m%d_%H%M)
HOOK_SUFFIX="_${HOOK_POINT}"
CF_OUT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_cf_permutation${HOOK_SUFFIX}${OUTPUT_SUFFIX}"

# ─── Helpers ───────────────────────────────────────────────────────
is_job_pending() {
    local jn="$1"
    bjobs -J "$jn" 2>/dev/null | grep -q "PEND" && return 0 || return 1
}

is_job_active() {
    local jn="$1"
    bjobs -J "$jn" 2>/dev/null | grep -qE "PEND|RUN" && return 0 || return 1
}

bsub_tiered() {
    # Submit a job with GPU memory tier escalation.
    # Args before --: bsub args. Args after --: command.
    local bsub_args=()
    local cmd=""
    local job_name="" log_file="" err_file=""
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            shift; cmd="$*"; break
        fi
        if [[ "$1" == "-J" ]]; then
            bsub_args+=("$1"); shift
            job_name="$1"
            bsub_args+=("$1"); shift; continue
        fi
        if [[ "$1" == "-oo" ]]; then
            bsub_args+=("$1"); shift
            log_file="$1"
            bsub_args+=("$1"); shift; continue
        fi
        if [[ "$1" == "-eo" ]]; then
            bsub_args+=("$1"); shift
            err_file="$1"
            bsub_args+=("$1"); shift; continue
        fi
        bsub_args+=("$1"); shift
    done
    [[ -n "$log_file" ]] && rm -f "$log_file"
    [[ -n "$err_file" ]] && rm -f "$err_file"

    local n_tiers=${#GPU_GMEM_TIERS[@]}
    local first_gmem="${GPU_GMEM_TIERS[0]}"

    bsub "${bsub_args[@]}" -gpu "num=1:gmem=${first_gmem}" "$cmd"

    if (( n_tiers <= 1 )); then
        return
    fi

    # Background escalation monitor
    (
        set +e
        for ((t=1; t<n_tiers; t++)); do
            sleep "$GMEM_WAIT"
            if is_job_pending "$job_name"; then
                next_gmem="${GPU_GMEM_TIERS[$t]}"
                echo "  [escalate] $job_name → gmem=$next_gmem"
                bkill -J "$job_name" 2>/dev/null || true
                sleep 2
                bsub "${bsub_args[@]}" -gpu "num=1:gmem=${next_gmem}" "$cmd"
            else
                break
            fi
        done
    ) &
}

# ═══════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
echo "  CF Pipeline — step: $STEP  |  model: $MODEL_TYPE ($MODEL_NAME)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Python:        $PYTHON"
echo "  Model path:    $MODEL_PATH"
echo "  Layers:        ${LAYERS:-all}"
echo "  Hook point:    $HOOK_POINT"
echo "  Samples:       $N_SAMPLES (K_image=$K_IMAGE, K_text=$K_TEXT)"
echo "  Paired data:   $PAIRED_DATA_PATH"
echo "  Output:        $CF_OUT_DIR"
echo "  Queue:         $QUEUE  |  GPU tiers: ${GPU_GMEM_TIERS[*]}"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP: build_cf_paired
# ═══════════════════════════════════════════════════════════════════
if [[ "$STEP" == "build_cf_paired" || "$STEP" == "all" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP: build_cf_paired"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -s "$PAIRED_DATA_PATH" ]]; then
        echo "  [skip] $PAIRED_DATA_PATH already exists"
    else
        BUILD_CMD="cd $WORK_DIR && $PYTHON code/build_cf_paired_data.py \
            --coco_ann_path $COCO_ANN_PATH \
            --detail_23k_path $DETAIL_23K_PATH \
            --output $PAIRED_DATA_PATH \
            --n_samples $N_SAMPLES \
            --K_image $K_IMAGE \
            --K_text $K_TEXT \
            --device auto \
            --seed $SEED"

        if $LOCAL; then
            # Run locally — uses GPU if login node has one, else CPU (slow on full)
            echo "  Building paired data locally…"
            eval "$BUILD_CMD"
        else
            # Submit as a small bsub job to get a GPU.
            # Embedding 23K captions is ~1-3 min on GPU vs many minutes on CPU.
            BUILD_JOB_NAME="cf_${LOG_TAG}_build_paired_${MODEL_NAME}_${DATESTAMP}"
            BUILD_LOG="$LOG_DIR/${BUILD_JOB_NAME}.log"
            BUILD_ERR="$LOG_DIR/${BUILD_JOB_NAME}.err"

            echo "  Submitting paired-data build to $QUEUE (GPU embedding)…"
            BSUB_ARGS=(-q "$QUEUE" \
                -J "$BUILD_JOB_NAME" \
                -oo "$BUILD_LOG" \
                -eo "$BUILD_ERR")
            bsub_tiered "${BSUB_ARGS[@]}" -- "$BUILD_CMD"
            echo "  → Build job: $BUILD_JOB_NAME"
            echo "  Logs: $BUILD_LOG"

            # If we're going to run cf_classify too, cf_classify shards must
            # wait for build to complete. Track the build job for that.
            CF_BUILD_JOB="$BUILD_JOB_NAME"
        fi
    fi

    if [[ "$STEP" == "build_cf_paired" ]]; then
        echo ""
        if $LOCAL; then
            echo "Paired data ready at $PAIRED_DATA_PATH"
        else
            echo "Paired-data build submitted. Once $BUILD_JOB_NAME completes,"
            echo "paired data will be at $PAIRED_DATA_PATH"
        fi
        exit 0
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# STEP: cf_classify
# ═══════════════════════════════════════════════════════════════════
if [[ "$STEP" == "cf_classify" || "$STEP" == "all" ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP: cf_classify"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check that paired data exists OR that we just submitted a build job
    # for it. If a build job was submitted via --step all, classify shards
    # use bsub -w "done(BUILD_JOB)" to wait for the file to be created.
    if [[ ! -s "$PAIRED_DATA_PATH" ]] && [[ -z "${CF_BUILD_JOB:-}" ]]; then
        echo "ERROR: paired data not found at $PAIRED_DATA_PATH"
        echo "Build it first: bash code/run_cf_pipeline.sh --step build_cf_paired ..."
        echo "Or run all steps together: bash code/run_cf_pipeline.sh --step all ..."
        exit 1
    fi
    if [[ ! -s "$PAIRED_DATA_PATH" ]]; then
        echo "  Note: paired data will be built by job $CF_BUILD_JOB."
        echo "  Classify shards will wait via bsub -w dependency."
    fi

    mkdir -p "$CF_OUT_DIR"

    # Determine number of layers in the model
    case "$MODEL_TYPE" in
        llava-llama3|llava-mistral|qwen2vl|qwen25vl-7b|llava-ov|internvl|idefics2)
            N_LAYERS=32 ;;
        qwen25vl-3b)  N_LAYERS=36 ;;
        *) N_LAYERS=32 ;;
    esac

    # Determine which layers to process
    if [[ -n "$LAYERS" ]]; then
        IFS=',' read -ra LAYER_LIST <<< "$LAYERS"
    else
        LAYER_LIST=()
        for ((l=0; l<N_LAYERS; l++)); do
            LAYER_LIST+=("$l")
        done
    fi

    # Default sharding: one GPU per layer being processed.
    # User-supplied --shards overrides this (SHARDS already non-empty).
    if [[ -z "$SHARDS" ]]; then
        SHARDS=${#LAYER_LIST[@]}
    fi

    echo "  Layers to process: ${LAYER_LIST[*]}"
    echo "  Sharding: $SHARDS shards"

    # Group layers into shards
    LAYERS_PER_SHARD=$(( (${#LAYER_LIST[@]} + SHARDS - 1) / SHARDS ))
    SUBMITTED_JOBS=()

    for ((s=0; s<SHARDS; s++)); do
        START_IDX=$((s * LAYERS_PER_SHARD))
        END_IDX=$((START_IDX + LAYERS_PER_SHARD))
        if (( END_IDX > ${#LAYER_LIST[@]} )); then
            END_IDX=${#LAYER_LIST[@]}
        fi
        if (( START_IDX >= ${#LAYER_LIST[@]} )); then
            break
        fi

        # This shard's layers (subset of LAYER_LIST)
        SHARD_LAYERS=()
        for ((i=START_IDX; i<END_IDX; i++)); do
            SHARD_LAYERS+=("${LAYER_LIST[$i]}")
        done
        SHARD_LAYER_START="${SHARD_LAYERS[0]}"
        SHARD_LAYER_END=$((${SHARD_LAYERS[-1]} + 1))

        JOB_NAME="cf_${LOG_TAG}_${MODEL_NAME}_s${s}_${DATESTAMP}"
        LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
        ERR_FILE="$LOG_DIR/${JOB_NAME}.err"

        # Skip if all layers in this shard already have outputs
        ALL_DONE=true
        for l in "${SHARD_LAYERS[@]}"; do
            # Layer name varies by hook_point; check the stats file as a proxy
            STATS_FILE="$CF_OUT_DIR/cf_permutation_stats_layers${SHARD_LAYER_START}-${SHARD_LAYER_END}.json"
            if [[ ! -s "$STATS_FILE" ]]; then
                ALL_DONE=false
                break
            fi
        done

        if $ALL_DONE; then
            echo "  [skip] Shard $s — already done"
            continue
        fi

        SKIP_NOISE_FLAG=""
        $SKIP_NOISE_PASS && SKIP_NOISE_FLAG="--skip_noise_pass"

        # Only pass --output_suffix when it's non-empty; otherwise argparse
        # consumes the next flag as its value and aborts with exit code 2.
        SUFFIX_FLAG=""
        [[ -n "$OUTPUT_SUFFIX" ]] && SUFFIX_FLAG="--output_suffix $OUTPUT_SUFFIX"

        CMD="cd $WORK_DIR && $PYTHON code/cf_classify.py \
            --model_type $MODEL_TYPE \
            --original_model_path $MODEL_PATH \
            --model $MODEL_NAME \
            --output_dir $OUTPUT_DIR \
            --paired_data_path $PAIRED_DATA_PATH \
            --coco_img_dir $COCO_IMG_DIR \
            --layer_start $SHARD_LAYER_START \
            --layer_end $SHARD_LAYER_END \
            --hook_point $HOOK_POINT \
            $SUFFIX_FLAG \
            --K_image $K_IMAGE \
            --K_text $K_TEXT \
            --n_noise_pairs $N_NOISE_PAIRS \
            --noise_K $NOISE_K \
            --noise_percentile $NOISE_PERCENTILE \
            --n_permutations $N_PERMUTATIONS \
            --alpha $ALPHA \
            --num_samples $N_SAMPLES \
            --seed $SEED \
            --device 0 \
            $SKIP_NOISE_FLAG"

        if $LOCAL; then
            echo "  Running shard $s locally (layers $SHARD_LAYER_START-$SHARD_LAYER_END)…"
            eval "$CMD" 2>&1 | tee "$LOG_FILE"
        else
            BSUB_ARGS=(-q "$QUEUE" \
                -J "$JOB_NAME" \
                -oo "$LOG_FILE" \
                -eo "$ERR_FILE")
            # If build_paired was submitted as a bsub job earlier, wait for it
            if [[ -n "${CF_BUILD_JOB:-}" ]]; then
                BSUB_ARGS+=(-w "done($CF_BUILD_JOB)")
            fi
            bsub_tiered "${BSUB_ARGS[@]}" -- "$CMD"
            echo "  → Shard $s: $JOB_NAME (layers $SHARD_LAYER_START-$SHARD_LAYER_END)"
            SUBMITTED_JOBS+=("$JOB_NAME")
        fi
    done

    if ! $LOCAL && (( ${#SUBMITTED_JOBS[@]} > 0 )); then
        echo ""
        echo "Submitted ${#SUBMITTED_JOBS[@]} shard jobs"
        echo "Monitor: bjobs"
        echo "Logs:    $LOG_DIR/cf_${LOG_TAG}_*_${DATESTAMP}.{log,err}"
        echo "Results: $CF_OUT_DIR/"
    fi
fi

echo ""
echo "Done."
