#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_pipeline.sh — Full Xu et al. neuron classification pipeline
#
# Backend: liuhaotian/llava-v1.5-7b (original LLaVA model)
#
# Chains:  1 describe → 2 merge_desc → 3 classify → 4 merge_class → 5 ablation → merge_ablation → 6 visualize → 7 attention → 8 statistics
# Keeps Python scripts separate; this script handles orchestration.
#
# Usage:
#   bash run_pipeline.sh                              # full pipeline, 32 GPUs
#   bash run_pipeline.sh --step 1                     # (describe) generate VLM descriptions
#   bash run_pipeline.sh --step 2                     # (merge_desc) merge description shards
#   bash run_pipeline.sh --step 3                     # (classify) FT + permutation classification
#   bash run_pipeline.sh --step 4                     # (merge_class) merge classification shards
#   bash run_pipeline.sh --step 5                     # (ablation) ablation validation
#   bash run_pipeline.sh --step merge_ablation         # merge existing ablation shard results
#   bash run_pipeline.sh --step 6                     # (visualize) Figure 3 activation maps
#   bash run_pipeline.sh --step 6 --viz-fig3          # reproduce exact Xu Figure 3 panels (a)-(f)
#   bash run_pipeline.sh --step 6 --viz-fig3 --viz-taxonomy both  # FT + PMBT labels
#   bash run_pipeline.sh --step 6 --viz-fig89         # reproduce Xu Figures 8 & 9
#   bash run_pipeline.sh --step 6 --viz-supplementary # reproduce Figures 15-17
#   bash run_pipeline.sh --step 7 --attn-image-id 000000189475  # attention analysis (single image)
#   bash run_pipeline.sh --step 7 --attn-image-path /path/to/image.jpg --attn-words "dough nut pink"
#   bash run_pipeline.sh --step all_att --mode test            # uses 6 default fig3 images for attn
#   bash run_pipeline.sh --step 8                     # (statistics) all charts (FT + PMBT)
#   bash run_pipeline.sh --step all                   # steps 1-6, 8 (standard chain)
#   bash run_pipeline.sh --step all_att              # steps 1-8 (includes attention)
#   bash run_pipeline.sh --mode test                  # quick test (6 fig3 images, 2 layers, 1 GPU)
#   bash run_pipeline.sh --mode test --clean           # remove markers → force re-run (keeps results)
#   bash run_pipeline.sh --mode test --clean 3         # remove markers from step 3 onwards
#   bash run_pipeline.sh --mode test --clean --wipe    # delete all results + logs
#   bash run_pipeline.sh --mode test --clean 3 --wipe  # delete results + logs from step 3 onwards
#   bash run_pipeline.sh --shards 16                  # use 16 GPUs
#   bash run_pipeline.sh --local                      # run locally (no bsub, no sharding)
#   bash run_pipeline.sh --queue waic-risk            # use waic-risk queue
#   bash run_pipeline.sh --suffix _v2                 # output to results/classification_v2
#   bash run_pipeline.sh --prune-images 200           # use 200 images for ablation
#   bash run_pipeline.sh --ablation-shards 16         # use 16 GPUs per ablation config
#   bash run_pipeline.sh --step 5 --pope-path pope/coco_pope_random.json
#   bash run_pipeline.sh --step 5 --pope-path pope/coco_pope_random.json \
#       --chair-ann-path /path/to/instances_val2014.json \
#       --pope-img-dir /path/to/val2014/
#
# GPU tiered escalation:
#   All jobs start at gmem=80G. If still PEND after 2 min, killed and
#   resubmitted at gmem=40G. After another 2 min → gmem=10G.
#   Override tiers: --gmem 80G,40G,10G   Override wait: --gmem-wait 120
#
# Idempotent: re-running skips jobs whose output already exists or
#             that are currently PEND/RUN in LSF.
#
# Examples:
#   bash run_pipeline.sh --mode test --local
#   bash run_pipeline.sh --shards 32
#   bash run_pipeline.sh --step 3 --shards 32
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────
QUEUE="waic-risk"
QUEUE_SET=false
LOCAL=false
STEP="all"             # 1|2|3|4|5|6|7|8|all|all_att  (or: describe|merge_desc|classify|merge_class|ablation|visualize|attention|statistics)
MODE="full"            # test | full
SHARDS=32              # GPUs
MODEL_TYPE="llava-ov"    # model backend (llava-liuhaotian / llava-hf / llava-ov / internvl / qwen2vl)
MODEL_PATH="liuhaotian/llava-v1.5-7b"  # HF Hub ID or local path to model weights
MODEL_NAME="llava-1.5-7b"  # must match --model default in $CLASSIFY_SCRIPT
OUT_SUFFIX_USER=""        # user-provided suffix for output dirs
CLASSIFY_SCRIPT="code/neuron_modality_statistical.py"  # classification script for step 3 (classify)
ABLATION_SCRIPT="code/neuron_ablation_validate.py"      # ablation validation script for step 5 (ablation)
VIZ_SCRIPT="code/visualize_neuron_activations.py"        # Figure 3 visualization script for step visualize
PLOT_SCRIPT="code/plot_neuron_statistics.py"              # Figures 5/6/7 statistics charts for step 8 (statistics)
ATTN_SCRIPT="code/attention_analysis.py"                  # attention analysis script for step 7 (attention)
VTP_SCRIPT="code/test_visual_token_pressure.py"              # VTP hypothesis analysis for step 9 (vtp)
ATTN_IMAGE_ID=""                                          # COCO image ID for attention analysis
ATTN_IMAGE_PATH=""                                        # direct image path (overrides ATTN_IMAGE_ID)
ATTN_HIGHLIGHTED_WORDS="hot white blue tie"               # words Xu highlighted (space-separated)
ATTN_HEATMAP_LAYERS="0 7 15 23 28 31"                    # layers to show in heatmap
ATTN_LAYER=""                                             # layer index for auto-highlight mode
ATTN_NEURON_IDX=""                                        # neuron index for auto-highlight mode
ATTN_TOP_K="5"                                            # top-K tokens for auto-highlight mode
ATTN_N_SAMPLES="1"                                           # number of top-ranked images to analyze
ATTN_GPUS="1"                                                # number of GPUs for parallel sharding
DESC_SUFFIX_USER=""        # suffix for description files (defaults to OUT_SUFFIX_USER)
OUTPUT_DIR_USER=""         # override classification output dir
PRUNE_IMAGES=100           # number of images for ablation validation
POPE_PATH="data/POPE/output/coco/coco_pope_random.json"   # path to POPE jsonl
CHAIR_ANN_PATH="data/annotations/instances_val2014.json"  # path to instances_val2014.json
POPE_IMG_DIR="data/val2014"                                 # path to COCO val images for POPE/CHAIR
CHAIR_NUM_IMAGES=500       # number of images for CHAIR evaluation
TRIVIAQA_PATH="data/triviaqa/qa/verified-web-dev.json"       # path to TriviaQA verified-web-dev.json
TRIVIAQA_NUM=2000                                            # number of TriviaQA questions
MMLU_DIR="data/mmlu/"                                       # path to MMLU data/ directory
MMLU_NUM=2000                                                # number of MMLU questions
VSR_PATH="data/vsr/"                                         # path to VSR (Visual Spatial Reasoning) data directory
SCIENCEQA_PATH="data/scienceqa/"                             # path to ScienceQA data directory
VIZ_FIG3=false             # if true, visualize step reproduces Xu Figure 3 panels
VIZ_FIG89=false            # if true, visualize step reproduces Xu Figures 8 & 9
VIZ_SUPPLEMENTARY=false    # if true, visualize step reproduces supplementary Figures 15-17
VIZ_TAXONOMY="both"        # ft | pmbt | both — which taxonomy labels to show in Figure 3 headers

# ── Advanced ablation settings (8 approaches) ──
ABLATION_METHOD="zero"         # zero | mean | noise | clamp_high | clamp_low
ABLATION_CURVE=true            # if true, sweep top_n values (ablation curve)
ABLATION_TOP_N=""              # specific top_n value (empty = ablate all)
ABLATION_LAYER_RANGE=""        # e.g. "0-10" or "22-31" (empty = all layers)
ABLATION_CURVE_STEPS=""                           # auto-computed from fracs × N_LLM_NEURONS (override with --ablation-curve-steps)
ABLATION_CURVE_STEPS_SET=false                    # tracks whether user explicitly set --ablation-curve-steps
ABLATION_CURVE_FRACS="0.001,0.005,0.01,0.05"     # 0.1%, 0.5%, 1%, 5% of total LLM neurons
ABLATION_N_STATS=50            # reference images for mean/noise/clamp stats
ABLATION_RANKING="label"       # label | cett
ABLATION_N_CETT=30             # reference images for CETT computation
ABLATION_ALL=false             # if true, submit all approaches as parallel GPU jobs
ABLATION_TAXONOMY="pmbt"       # ft | pmbt | both — which taxonomy to ablate (used when --ablation-all is NOT set) (default changed to pmbt only)
ABLATION_SHARDS=15              # GPUs for ablation (per taxonomy config)
CLEAN=false                    # if true, delete markers to force re-run (test mode only)
WIPE=false                     # if true, delete entire result directories (used with --clean --wipe)
CLEAN_FROM="auto"                  # clean from this step onwards (auto=match --step, or explicit 1-8)
CLEAN_TO="auto"                    # clean up to this step (auto=same as CLEAN_FROM, i.e. single step)

# ── Step 10 (halluc_score) settings ──
HALLUC_SCORE_SCRIPT="code/halluc_score_neurons.py"
MERGE_STEERING_SCRIPT="code/merge_steering_results.py"    # step 12: merge steering results across alphas
PLOT_STEERING_SCRIPT="code/plot_steering_results.py"      # step 13: ECCV publication figures
HALLUC_SCORE_SHARDS=8             # GPUs for halluc scoring (max useful: 32, one per layer)
HALLUC_POPE_SAMPLES=""            # number of POPE samples (empty = all ~3000)
HALLUC_CONTRASTIVE=""             # set to "1" to enable contrastive POPE filtering
HALLUC_SKIP_ABLATION=""           # set to "1" to skip ablation loop, reuse existing scores
HALLUC_TRIVIAQA=""                # set to "1" to enable TriviaQA dual-source scoring (Section 3.5)

# ── Step 11 (steering) settings ──
STEERING_ALPHAS="0,0.25,0.5,0.75,1.5,2.0,2.5,3.0"   # comma-separated alpha values
STEERING_SHARDS=15                          # GPUs per alpha value (same sharding as step 5)
STEERING_TAXONOMY="pmbt"                    # ft | pmbt | both
HALLUC_SCORE_METHOD="combined"              # combined (default) | dh | cett — which step 10 ranking to use

# GPU memory tiers — escalate through tiers when jobs stay PEND
GPU_GMEM_TIERS=("20G")                # override with --gmem 40G,20G
RUN_SUFFIX=""                          # append to log names + output dirs (e.g. _gmem_80)
LAYER_LIST=""                          # comma-separated layers for step 3 (e.g. 8,9,13,14,31)
GMEM_WAIT=120                          # seconds to wait before escalating (override with --gmem-wait)
GPU_RES_BASE="rusage[mem=24576] order[-gpu_maxfactor]"

# Dataset constants
N_TOTAL_IMAGES=23000
N_LAYERS=32
GEN_LIMIT=""                       # --limit N: override N_TOTAL_IMAGES for quick testing

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)    LOCAL=true; shift ;;
        --step)     STEP="$2"; shift 2 ;;
        --mode)     MODE="$2"; shift 2 ;;
        --shards)   SHARDS="$2"; shift 2 ;;
        --limit)    GEN_LIMIT="$2"; shift 2 ;;
        --queue)    QUEUE="$2"; QUEUE_SET=true; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --suffix)   OUT_SUFFIX_USER="$2"; shift 2 ;;
        --classify-script) CLASSIFY_SCRIPT="$2"; shift 2 ;;
        --ablation-script) ABLATION_SCRIPT="$2"; shift 2 ;;
        --desc-suffix) DESC_SUFFIX_USER="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR_USER="$2"; shift 2 ;;
        --prune-images) PRUNE_IMAGES="$2"; shift 2 ;;
        --pope-path) POPE_PATH="$2"; shift 2 ;;
        --chair-ann-path) CHAIR_ANN_PATH="$2"; shift 2 ;;
        --pope-img-dir) POPE_IMG_DIR="$2"; shift 2 ;;
        --chair-num-images) CHAIR_NUM_IMAGES="$2"; shift 2 ;;
        --triviaqa-path) TRIVIAQA_PATH="$2"; shift 2 ;;
        --triviaqa-num) TRIVIAQA_NUM="$2"; shift 2 ;;
        --mmlu-dir) MMLU_DIR="$2"; shift 2 ;;
        --mmlu-num) MMLU_NUM="$2"; shift 2 ;;
        --vsr-path) VSR_PATH="$2"; shift 2 ;;
        --scienceqa-path) SCIENCEQA_PATH="$2"; shift 2 ;;
        --viz-fig3) VIZ_FIG3=true; shift ;;
        --viz-fig89) VIZ_FIG89=true; shift ;;
        --viz-supplementary) VIZ_SUPPLEMENTARY=true; shift ;;
        --viz-taxonomy) VIZ_TAXONOMY="$2"; shift 2 ;;
        --gmem)     IFS=',' read -ra GPU_GMEM_TIERS <<< "$2"; shift 2 ;;
        --gmem-wait) GMEM_WAIT="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --layer-list) LAYER_LIST="$2"; shift 2 ;;
        --ablation-method) ABLATION_METHOD="$2"; shift 2 ;;
        --ablation-curve) ABLATION_CURVE=true; shift ;;
        --ablation-top-n) ABLATION_TOP_N="$2"; shift 2 ;;
        --ablation-layer-range) ABLATION_LAYER_RANGE="$2"; shift 2 ;;
        --ablation-curve-steps) ABLATION_CURVE_STEPS="$2"; ABLATION_CURVE_STEPS_SET=true; shift 2 ;;
        --ablation-curve-fracs) ABLATION_CURVE_FRACS="$2"; shift 2 ;;
        --ablation-n-stats) ABLATION_N_STATS="$2"; shift 2 ;;
        --ablation-ranking) ABLATION_RANKING="$2"; shift 2 ;;
        --ablation-n-cett) ABLATION_N_CETT="$2"; shift 2 ;;
        --ablation-all) ABLATION_ALL=true; shift ;;
        --ablation-taxonomy) ABLATION_TAXONOMY="$2"; shift 2 ;;
        --ablation-shards) ABLATION_SHARDS="$2"; shift 2 ;;
        --attn-image-id) ATTN_IMAGE_ID="$2"; shift 2 ;;
        --attn-image-path) ATTN_IMAGE_PATH="$2"; shift 2 ;;
        --attn-words) ATTN_HIGHLIGHTED_WORDS="$2"; shift 2 ;;
        --attn-layers) ATTN_HEATMAP_LAYERS="$2"; shift 2 ;;
        --attn-layer) ATTN_LAYER="$2"; shift 2 ;;
        --attn-neuron) ATTN_NEURON_IDX="$2"; shift 2 ;;
        --attn-topk) ATTN_TOP_K="$2"; shift 2 ;;
        --attn-nsamples) ATTN_N_SAMPLES="$2"; shift 2 ;;
        --attn-gpus) ATTN_GPUS="$2"; shift 2 ;;
        --halluc-score-shards) HALLUC_SCORE_SHARDS="$2"; shift 2 ;;
        --halluc-pope-samples) HALLUC_POPE_SAMPLES="$2"; shift 2 ;;
        --halluc-contrastive) HALLUC_CONTRASTIVE="1"; shift 1 ;;
        --halluc-skip-ablation) HALLUC_SKIP_ABLATION="1"; shift 1 ;;
        --halluc-triviaqa) HALLUC_TRIVIAQA="1"; shift 1 ;;
        --halluc-score-method) HALLUC_SCORE_METHOD="$2"; shift 2 ;;
        --steering-alphas) STEERING_ALPHAS="$2"; shift 2 ;;
        --steering-shards) STEERING_SHARDS="$2"; shift 2 ;;
        --steering-taxonomy) STEERING_TAXONOMY="$2"; shift 2 ;;
        --clean)        CLEAN=true
                        # Optional step number: --clean 3 cleans from step 3 onwards
                        # If omitted, defaults to the --step value (set after parsing)
                        if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                            CLEAN_FROM="$2"; shift 2
                        else
                            CLEAN_FROM="auto"; shift
                        fi
                        ;;
        --wipe)         WIPE=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Normalize model-type aliases ────────────────────────────────────────────
# Accept short names at the CLI so users can type e.g. --model-type llava
# instead of --model-type llava-liuhaotian.  Full names still work.
_normalize_model_type() {
    case "$1" in
        llava)  echo "llava-liuhaotian" ;;
        intern) echo "internvl" ;;
        qwen)   echo "qwen2vl" ;;
        *)      echo "$1" ;;
    esac
}
if [[ "$MODEL_TYPE" == *","* ]]; then
    _normalized=""
    IFS=',' read -ra _parts <<< "$MODEL_TYPE"
    for _p in "${_parts[@]}"; do
        _n=$(_normalize_model_type "$_p")
        _normalized="${_normalized:+${_normalized},}${_n}"
    done
    MODEL_TYPE="$_normalized"
elif [[ "$MODEL_TYPE" != "all" ]]; then
    MODEL_TYPE=$(_normalize_model_type "$MODEL_TYPE")
fi

# ── Validate model-type ─────────────────────────────────────────────────────
_VALID_MODELS="llava-liuhaotian llava-hf llava-ov llava-ov-si internvl qwen2vl all"
_VALID_ALIASES="llava intern qwen"
if [[ "$MODEL_TYPE" != *","* ]]; then
    _found=false
    for _vm in $_VALID_MODELS; do
        [[ "$MODEL_TYPE" == "$_vm" ]] && _found=true && break
    done
    if ! $_found; then
        echo "ERROR: unknown --model-type '$MODEL_TYPE'"
        echo ""
        echo "  Valid names:    llava-liuhaotian  llava-hf  llava-ov  internvl  qwen2vl  all"
        echo "  Short aliases:  llava             —         —         intern    qwen"
        echo ""
        echo "  Examples:"
        echo "    bash code/run_pipeline.sh --model-type llava ..."
        echo "    bash code/run_pipeline.sh --model-type intern ..."
        echo "    bash code/run_pipeline.sh --model-type qwen,intern ..."
        exit 1
    fi
else
    IFS=',' read -ra _check_parts <<< "$MODEL_TYPE"
    for _cp in "${_check_parts[@]}"; do
        _found=false
        for _vm in $_VALID_MODELS; do
            [[ "$_cp" == "$_vm" ]] && _found=true && break
        done
        if ! $_found; then
            echo "ERROR: unknown model '$_cp' in --model-type '$MODEL_TYPE'"
            echo ""
            echo "  Valid names:    llava-liuhaotian  llava-hf  llava-ov  internvl  qwen2vl"
            echo "  Short aliases:  llava             —         —         intern    qwen"
            exit 1
        fi
    done
fi

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$_SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"
mkdir -p logs

# ── Multi-model dispatch ──────────────────────────────────────────────────
# --model-type all              → run for all 4 backends sequentially
# --model-type internvl,qwen2vl → run for selected backends
ALL_MODELS=(llava-liuhaotian llava-hf llava-ov llava-ov-si internvl qwen2vl)
if [[ "$MODEL_TYPE" == "all" ]] || [[ "$MODEL_TYPE" == *","* ]]; then
    if [[ "$MODEL_TYPE" == "all" ]]; then
        RUN_MODELS=("${ALL_MODELS[@]}")
    else
        IFS=',' read -ra RUN_MODELS <<< "$MODEL_TYPE"
    fi
    echo "Multi-model dispatch: ${RUN_MODELS[*]}"
    ANY_FAILED=0
    for m in "${RUN_MODELS[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Backend: $m"
        echo "════════════════════════════════════════════════════"
        # Rebuild args: strip --model-type and its value, inject --model-type $m
        REBUILT_ARGS=()
        SKIP_NEXT=false
        for arg in "$@"; do
            if $SKIP_NEXT; then SKIP_NEXT=false; continue; fi
            if [[ "$arg" == "--model-type" ]]; then SKIP_NEXT=true; continue; fi
            REBUILT_ARGS+=("$arg")
        done
        bash "$0" "${REBUILT_ARGS[@]}" --model-type "$m" || ANY_FAILED=1
    done
    if (( ANY_FAILED )); then
        echo "WARNING: one or more backends failed"
        exit 1
    fi
    exit 0
fi

# ── Per-backend MODEL_PATH defaults ─────────────────────────────────────────
# Applied only when the user has not passed --model-path (i.e. MODEL_PATH is
# still the liuhaotian default).  This lets you run:
#   bash run_pipeline.sh --model-type internvl ...
# without also having to specify --model-path every time.
if [[ "$MODEL_PATH" == "liuhaotian/llava-v1.5-7b" ]]; then          # not overridden by user
    if [[ "$MODEL_TYPE" == "internvl" ]]; then
        MODEL_PATH="modern_vlms/pretrained/InternVL2_5-8B"           # default InternVL2.5-8B weights
    elif [[ "$MODEL_TYPE" == "qwen2vl" ]]; then
        MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct"  # default Qwen2.5-VL-7B weights
    elif [[ "$MODEL_TYPE" == "llava-hf" ]]; then
        MODEL_PATH="llava-hf/llava-1.5-7b-hf"                       # HF-format LLaVA 1.5 (has preprocessor_config.json)
    elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
        # Resolve local HF cache snapshot — avoids Hub download on cluster nodes
        _OV_SNAP="$WORK_DIR/.cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots"
        if [[ -d "$_OV_SNAP" ]]; then
            _OV_HASH=$(ls "$_OV_SNAP" | head -1)                     # pick first (only) snapshot hash
            MODEL_PATH="$_OV_SNAP/$_OV_HASH"                         # full local path — no network needed
        else
            MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"    # fallback to Hub if cache missing
        fi
    elif [[ "$MODEL_TYPE" == "llava-ov-si" ]]; then
        # Same pattern for the single-image stage-2 variant
        _OVSI_SNAP="$WORK_DIR/.cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-si-hf/snapshots"
        if [[ -d "$_OVSI_SNAP" ]]; then
            _OVSI_HASH=$(ls "$_OVSI_SNAP" | head -1)
            MODEL_PATH="$_OVSI_SNAP/$_OVSI_HASH"
        else
            MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-si-hf"    # fallback to Hub if cache missing
        fi
    fi
fi

# ── Per-backend MODEL_NAME defaults ─────────────────────────────────────────
# Applied only when the user has not overridden MODEL_NAME (i.e. it is still
# the liuhaotian default).  Ensures output directories, --model args, and
# marker file paths all reflect the actual model being run.
if [[ "$MODEL_NAME" == "llava-1.5-7b" ]]; then                           # not overridden by user
    if [[ "$MODEL_TYPE" == "internvl" ]]; then
        MODEL_NAME="internvl2.5-8b"                                       # InternVL2.5-8B output dir name
    elif [[ "$MODEL_TYPE" == "qwen2vl" ]]; then
        MODEL_NAME="qwen2.5-vl-7b"                                        # Qwen2.5-VL-7B output dir name
    elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
        MODEL_NAME="llava-onevision-7b"                                    # LLaVA-OneVision-7B output dir name
    fi
    # llava-hf keeps MODEL_NAME="llava-1.5-7b" (same model as llava-liuhaotian)
fi


# ── Per-backend N_LAYERS defaults ───────────────────────────────────────────
# Qwen2-7B backbone (used by llava-ov and qwen2vl) has 28 transformer layers,
# not 32.  Override the default so sharding in step 3 does not produce
# out-of-range layer indices.  InternVL2.5-8B uses InternLM2 (32 layers).
if [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then
    N_LAYERS=28
fi
# ── Per-backend N_LLM_NEURONS (n_layers × intermediate_size) ──────────────
# Total MLP neuron count across all transformer layers.  Used to compute
# percentage-based ablation curve steps (0.1%, 0.5%, 1%, 5% of total).
#   llava-liuhaotian / llava-hf:  32 × 11008 = 352256  (LLaMA-2-7B)
#   llava-ov / llava-ov-si / qwen2vl:  28 × 18944 = 530432  (Qwen2-7B)
#   internvl:  32 × 14336 = 458752  (InternLM2-8B)
case "$MODEL_TYPE" in
    llava-liuhaotian|llava-hf) N_LLM_NEURONS=352256 ;;
    llava-ov|llava-ov-si|qwen2vl) N_LLM_NEURONS=530432 ;;
    internvl) N_LLM_NEURONS=458752 ;;
    *) N_LLM_NEURONS=530432 ;;  # safe default (Qwen2-7B)
esac

# ── Auto-compute ABLATION_CURVE_STEPS from fractions ──────────────────────
# If the user did not explicitly pass --ablation-curve-steps, compute the
# integer top_n values from ABLATION_CURVE_FRACS × N_LLM_NEURONS.
# Example for llava-ov (530432 total):
#   0.001 → 530,  0.005 → 2652,  0.01 → 5304,  0.05 → 26522
if ! $ABLATION_CURVE_STEPS_SET; then
    _computed_steps=""
    IFS=',' read -ra _fracs <<< "$ABLATION_CURVE_FRACS"
    for _f in "${_fracs[@]}"; do
        _n=$(awk "BEGIN { printf \"%d\", ${_f} * ${N_LLM_NEURONS} + 0.5 }")
        _computed_steps="${_computed_steps:+${_computed_steps},}${_n}"
    done
    ABLATION_CURVE_STEPS="$_computed_steps"
fi
# ── Short model alias for job names ─────────────────────────────────────────
case "$MODEL_TYPE" in
    llava-liuhaotian) SHORT_MODEL="l" ;;
    llava-hf)         SHORT_MODEL="lhf" ;;
    llava-ov)         SHORT_MODEL="lo" ;;
    llava-ov-si)      SHORT_MODEL="losi" ;;
    internvl)         SHORT_MODEL="int" ;;
    qwen2vl)          SHORT_MODEL="qw" ;;
    *)                SHORT_MODEL="${MODEL_TYPE:0:4}" ;;
esac

# ── Log directory and suffix — separate logs by backend ─────────────────────
# Stores logs under logs/<mode>/<MODEL_TYPE>/ with _<MODEL_TYPE> suffix on each file,
# so different backends never overwrite each other's logs.
MODE_DIR=$( [[ "$MODE" == "test" ]] && echo "test" || echo "full" )
LOG_DIR="logs/${MODE_DIR}/${MODEL_TYPE}"
LOG_SUFFIX="_${MODEL_TYPE}${RUN_SUFFIX}"
mkdir -p "$LOG_DIR"

# ── Python interpreter — modern VLMs use their own venv ──────────────────
# The old .venv (managed by uv) has transformers==4.37.2 + an older torch
# which crashes with: ImportError: cannot import name '_get_cpp_backtrace'.
# InternVL / Qwen2VL / LLaVA-HF / LLaVA-OV need modern_vlms/.venv which has compatible versions.
#
# CRITICAL: We must also unset VIRTUAL_ENV and strip PYTHONPATH so the old
# .venv site-packages don't leak into the new interpreter via env inheritance.
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: $PYTHON not found — run:  cd modern_vlms/intervl_env && uv venv .venv_internvl --python 3.10 && uv pip install -r pyproject.toml"
        exit 1
    fi
elif [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then
    PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: $PYTHON not found — run:  cd modern_vlms && python -m venv .venv && pip install -r requirements.txt"
        exit 1
    fi
    # Purge old venv from environment so its packages don't shadow modern_vlms
    unset VIRTUAL_ENV 2>/dev/null || true
    # Remove any .venv site-packages entries from PYTHONPATH
    if [[ -n "${PYTHONPATH:-}" ]]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd ':' -)
        export PYTHONPATH
    fi
    # Verify the interpreter loads the correct torch
    echo "  Verifying modern venv torch..."
    if ! "$PYTHON" -c "from torch._C import _get_cpp_backtrace" 2>/dev/null; then
        echo "ERROR: modern_vlms/.venv torch is also broken. Reinstall:"
        echo "  cd modern_vlms && .venv/bin/pip install --force-reinstall torch"
        exit 1
    fi
    echo "  ✓ torch._C OK"
else
    PYTHON="$WORK_DIR/.venv/bin/python"
fi

# ── Shared HuggingFace cache (so all cluster nodes use the same download) ──
export HF_HOME="${HF_HOME:-$WORK_DIR/.cache/huggingface}"

# ── Mode-specific settings ────────────────────────────────────
if [[ "$MODE" == "test" ]]; then
    # Quick test: 6 fig3 images, 2 layers, top_n=5, no sharding
    GEN_ARGS="--test_fig3"
    DESC_FILE="results/1-describe/test/generated_descriptions_fig3_${MODEL_TYPE}.json"
    CLASSIFY_ARGS="--num_images 6 --top_n 5 --layer_start 0 --layer_end 2 --n_permutations 100"
    OUT_SUFFIX="_test_${MODEL_TYPE}"
    SHARDS_EFFECTIVE=1   # no sharding in test mode
    $QUEUE_SET || QUEUE="waic-risk"  # default queue for all modes
else
    # Full run: sharded across GPUs
    GEN_ARGS=""
    _DS="_${MODEL_NAME}${DESC_SUFFIX_USER:+_${DESC_SUFFIX_USER}}"
    DESC_FILE="results/1-describe/full/generated_descriptions${_DS}.json"
    CLASSIFY_ARGS=""
    OUT_SUFFIX="${OUT_SUFFIX_USER}"
    SHARDS_EFFECTIVE=$SHARDS
fi

# ── --limit N: quick-test override for step 1 ───────────────────────────────
# Caps N_TOTAL_IMAGES and forces a single shard so you can verify the pipeline
# end-to-end with just a few images before committing to a full run.
if [[ -n "$GEN_LIMIT" ]]; then
    N_TOTAL_IMAGES=$GEN_LIMIT
    SHARDS_EFFECTIVE=1
    echo "  ⚠ --limit $GEN_LIMIT: overriding N_TOTAL_IMAGES=$GEN_LIMIT, SHARDS_EFFECTIVE=1"
fi

# ── Normalise --step value ──────────────────────────────────────────────
# Accept numbers (1-8), short names, and descriptive names.
# Everything maps to the internal gate names used by step sections below.
case "$STEP" in
    1|describe|gd)            STEP="gd" ;;
    2|merge_desc|merge_gd)    STEP="merge_gd" ;;
    3|classify|cn)            STEP="cn" ;;
    4|merge_class|merge_nc)   STEP="merge_nc" ;;
    5|ablation|prune)         STEP="prune" ;;
    merge_ablation)           STEP="merge_ablation" ;;
    6|visualize|viz)          STEP="visualize" ;;
    7|attention|attn)         STEP="attn" ;;
    8|statistics|plot|stats)  STEP="plot" ;;
    find_fig3|fig3_neurons)    STEP="find_fig3" ;;
    check_collisions|collisions)  STEP="check_collisions" ;;
    9|vtp)                    STEP="vtp" ;;
    10|halluc_score)           STEP="halluc_score" ;;
    11|steering|steer)         STEP="steering" ;;
    12|merge_steering)         STEP="merge_steering" ;;
    13|plot_steering|eccv)     STEP="plot_steering" ;;
    all|all_att)             ;;  # keep as-is
    *) echo "ERROR: unknown step '$STEP'"; echo "  Valid: 1-11, merge_ablation, find_fig3, check_collisions, all, all_att"; exit 1 ;;
esac

# ── Resolve --clean default: if "auto", infer from --step ───────────────
if [[ "$CLEAN_FROM" == "auto" ]]; then
    case "$STEP" in
        gd)        CLEAN_FROM=1 ;;
        merge_gd)  CLEAN_FROM=2 ;;
        cn)        CLEAN_FROM=3 ;;
        merge_nc)  CLEAN_FROM=4 ;;
        prune)     CLEAN_FROM=5 ;;
        merge_ablation) CLEAN_FROM=5 ;;
        visualize) CLEAN_FROM=6 ;;
        attn)      CLEAN_FROM=7 ;;
        plot)      CLEAN_FROM=8 ;;
        vtp)       CLEAN_FROM=9 ;;
        halluc_score) CLEAN_FROM=10 ;;
        steering)  CLEAN_FROM=11 ;;
        merge_steering) CLEAN_FROM=12 ;;
        plot_steering) CLEAN_FROM=13 ;;
        *)         CLEAN_FROM=1 ;;  # all / all_att → clean everything
    esac
fi
# Default CLEAN_TO: same as CLEAN_FROM (single-step clean).
# For 'all' / 'all_att', clean all steps (1–13).
if [[ "$CLEAN_TO" == "auto" ]]; then
    case "$STEP" in
        all|all_att) CLEAN_TO=13 ;;
        *)           CLEAN_TO="$CLEAN_FROM" ;;
    esac
fi

# ── Step-all flags ──────────────────────────────────────────────────────
# "all"      = steps 1-4, 5-6, 8    (standard chain, skips attn)
# "all_att" = steps 1-8             (includes attention maps)
STEP_ALL=false
STEP_ALL_FULL=false
if [[ "$STEP" == "all" || "$STEP" == "all_att" ]]; then
    STEP_ALL=true
    [[ "$STEP" == "all_att" ]] && STEP_ALL_FULL=true
fi

# ── Job name bases — step number + short model alias ────────────────────────
# Produces concise LSF job names like 1_lav, 3_int_2, 5_q, 8_lav1
JN1="1_${SHORT_MODEL}"     # describe
JN2="2_${SHORT_MODEL}"     # merge_descriptions
JN3="3_${SHORT_MODEL}"     # classify (FT + permutation)
JN4="4_${SHORT_MODEL}"     # merge_classifications
JN5="5_${SHORT_MODEL}"     # ablation_validate
JN6="6_${SHORT_MODEL}"     # activation_maps
JN6p="6p_${SHORT_MODEL}"   # patch_fig3
JN7="7_${SHORT_MODEL}"     # attention_maps
JN8="8_${SHORT_MODEL}"     # statistics
JN9="9_${SHORT_MODEL}"     # vtp_analysis
JN10="10_${SHORT_MODEL}"   # halluc_score
JN11="11_${SHORT_MODEL}"   # steering

# For step 3 (classify): can't exceed 32 layers
CLASSIFY_SHARDS=$SHARDS_EFFECTIVE
if (( CLASSIFY_SHARDS > N_LAYERS )); then
    CLASSIFY_SHARDS=$N_LAYERS
fi

# ── LSF helpers ──────────────────────────────────────────────
# Check if a job is currently PEND or RUN in LSF
is_job_active() {
    local name=$1
    bjobs -J "$name" -noheader 2>/dev/null | grep -qE "PEND|RUN"
}

# Check if a job is specifically PEND (not yet running)
is_job_pending() {
    local name=$1
    bjobs -J "$name" -noheader 2>/dev/null | grep -q "PEND"
}

# Submit a GPU job with tiered gmem escalation.
# Usage: bsub_tiered <bsub_args...> -- <command>
# Submits at GPU_GMEM_TIERS[0], then spawns a background monitor that
# re-submits at the next tier if the job is still PEND after GMEM_WAIT seconds.
bsub_tiered() {
    # Split args at '--' into bsub_args and cmd
    local bsub_args=()
    local cmd=""
    local job_name=""
    local log_file="" err_file=""
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            shift; cmd="$*"; break
        fi
        # capture -J value for job name
        if [[ "$1" == "-J" ]]; then
            bsub_args+=("$1"); shift
            job_name="$1"
            bsub_args+=("$1"); shift
            continue
        fi
        # capture -oo/-eo paths for log clearing
        if [[ "$1" == "-oo" ]]; then
            bsub_args+=("$1"); shift
            log_file="$1"
            bsub_args+=("$1"); shift
            continue
        fi
        if [[ "$1" == "-eo" ]]; then
            bsub_args+=("$1"); shift
            err_file="$1"
            bsub_args+=("$1"); shift
            continue
        fi
        bsub_args+=("$1"); shift
    done

    # Clear previous logs so they don't accumulate across runs
    [[ -n "$log_file" ]] && rm -f "$log_file"
    [[ -n "$err_file" ]] && rm -f "$err_file"

    local n_tiers=${#GPU_GMEM_TIERS[@]}
    local first_gmem="${GPU_GMEM_TIERS[0]}"

    # Submit at first tier
    bsub "${bsub_args[@]}" -gpu "num=1:gmem=$first_gmem" -R "$GPU_RES_BASE" "$cmd"

    # If only one tier, no escalation needed
    if (( n_tiers <= 1 )); then
        return
    fi

    # Spawn background escalation monitor
    (
        set +e  # don't exit on bkill/bjobs failures
        for ((t=1; t<n_tiers; t++)); do
            sleep "$GMEM_WAIT"
            # Check if still pending
            if is_job_pending "$job_name"; then
                next_gmem="${GPU_GMEM_TIERS[$t]}"
                echo "  [escalate] $job_name still PEND after ${GMEM_WAIT}s → resubmit gmem=$next_gmem"
                bkill -J "$job_name" 2>/dev/null || true
                sleep 2  # let LSF process the kill
                bsub "${bsub_args[@]}" -gpu "num=1:gmem=$next_gmem" -R "$GPU_RES_BASE" "$cmd"
            else
                # Job is running or done, stop escalating
                break
            fi
        done
    ) &
}

# Counters
SUBMITTED=0
SKIPPED=0
MERGE_SUBMITTED=0             # 1 if merge job was submitted this run
CLS_SUBMITTED_JOBS_ALL=""      # space-separated list of cn job names for merge_nc deps
MERGE_NC_SUBMITTED=""          # merge_nc job name if submitted this run (for plot deps)

OUTPUT_DIR="${OUTPUT_DIR_USER:-results/3-classify/${MODE_DIR}}${RUN_SUFFIX}"

# ── --clean in full mode: require confirmation to prevent accidents ────
if $CLEAN && [[ "$MODE" == "full" ]]; then
    if $WIPE; then
        echo "WARNING: --clean --wipe in full mode will DELETE results + logs for $MODEL_NAME (steps $CLEAN_FROM–$CLEAN_TO)."
    else
        echo "WARNING: --clean in full mode will remove markers for $MODEL_NAME (steps $CLEAN_FROM–$CLEAN_TO)."
    fi
    read -p "  Are you sure? [y/N] " _confirm
    if [[ "$_confirm" != "y" && "$_confirm" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ── Model-specific GPU memory override ────────────────────────
# InternVL2.5-8B has InternViT-6B (~12GB) + InternLM2-7B (~14GB) = ~26GB
# in bfloat16, which OOMs on 20G GPUs.  Auto-escalate to 40G unless the
# user explicitly set --gmem on the command line.
if [[ "$MODEL_TYPE" == "internvl" && "${GPU_GMEM_TIERS[0]}" == "20G" ]]; then
    GPU_GMEM_TIERS=("40G")
    GPU_RES_BASE="rusage[mem=49152] order[-gpu_maxfactor]"
    echo "  [auto] InternVL detected → GPU gmem=40G, CPU mem=48GB"
fi

# ── Banner ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Xu et al. Neuron Classification Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Mode:               $MODE"
echo "  Backend:            $MODEL_TYPE"
echo "  Model name:         $MODEL_NAME"
echo "  Job alias:          $SHORT_MODEL (e.g. 1_${SHORT_MODEL}, 3_${SHORT_MODEL})"
echo "  Python:             $PYTHON"
echo "  Model path:         $MODEL_PATH"
echo "  Log dir:            $LOG_DIR"
echo "  Step:               $STEP"
echo "  GPUs:               $SHARDS_EFFECTIVE"
[[ -n "$GEN_LIMIT" ]] && echo "  Image limit:        $GEN_LIMIT (--limit)"
echo "  Classify shards:    $CLASSIFY_SHARDS (max $N_LAYERS layers)"
echo "  Local:              $LOCAL"
echo "  Queue:              $QUEUE"
echo "  Work dir:           $WORK_DIR"
echo "  GPU gmem tiers:     ${GPU_GMEM_TIERS[*]} (wait ${GMEM_WAIT}s between)"
echo "  Desc file:          $DESC_FILE"
echo "  Output dir:         $OUTPUT_DIR"
echo "  Ablation script:    $ABLATION_SCRIPT"
echo "  Attention script:   $ATTN_SCRIPT"
echo "  Prune images:       $PRUNE_IMAGES"
echo "  Ablation parallel:  $ABLATION_ALL (--ablation-all for 18 parallel GPU jobs)"
echo "  Ablation taxonomy:  $ABLATION_TAXONOMY (ft | pmbt | both)"  # default is now pmbt
echo "  Ablation shards:    $ABLATION_SHARDS GPUs per config"
echo "  LLM neurons:        $N_LLM_NEURONS (${N_LAYERS} layers × intermediate_size)"
echo "  Ablation curve:     $ABLATION_CURVE (fracs: $ABLATION_CURVE_FRACS → steps: $ABLATION_CURVE_STEPS)"
echo "  Clean before run:   $CLEAN (step $CLEAN_FROM–$CLEAN_TO, wipe=$WIPE)  --clean [N] markers only, add --wipe for full delete"
echo "  Idempotent:         yes (skips completed/active jobs)"
echo "═══════════════════════════════════════════════════════════"

# Create directories (no log cleanup — needed for completion detection)
mkdir -p "$WORK_DIR/results/1-describe/${MODE_DIR}/shards"
mkdir -p "$WORK_DIR/$LOG_DIR"

# ── CLEAN: remove markers (or full dirs with --wipe) to force re-run ───
if $CLEAN; then
    echo ""
    if $WIPE; then
        echo "  ── WIPE MODE ($MODE): deleting results + logs step $CLEAN_FROM–$CLEAN_TO for $MODEL_NAME ──"
    else
        echo "  ── CLEAN MODE ($MODE): removing markers step $CLEAN_FROM–$CLEAN_TO for $MODEL_NAME ──"
    fi

    # Kill active LSF jobs for steps CLEAN_FROM–CLEAN_TO
    declare -A STEP_JOBS=(
        [1]="$JN1"  [2]="$JN2"  [3]="$JN3"  [4]="$JN4"
        [5]="$JN5"  [6]="$JN6 $JN6p"  [7]="$JN7"  [8]="$JN8"
    )
    for s in $(seq "$CLEAN_FROM" "$CLEAN_TO"); do
        for jname in ${STEP_JOBS[$s]:-}; do
            if is_job_active "$jname" 2>/dev/null; then
                bkill -J "$jname" 2>/dev/null && echo "    bkill $jname"
            fi
        done
    done
    sleep 2  # let LSF register job kills before skip-checks

    # Step 1: descriptions
    if (( CLEAN_FROM <= 1 && 1 <= CLEAN_TO )); then
        if $WIPE; then
            [[ -f "$DESC_FILE" ]] && rm -f "$DESC_FILE" && echo "    rm $DESC_FILE"
            # Also remove shards
            for sf in results/1-describe/${MODE_DIR}/shards/gen_desc*"${MODEL_NAME}"*.json; do
                [[ -f "$sf" ]] && rm -f "$sf" && echo "    rm $sf"
            done
        else
            [[ -f "$DESC_FILE" ]] && rm -f "$DESC_FILE" && echo "    rm $DESC_FILE (marker)"
        fi
    fi

    # Step 3: classification stats (markers that trigger skip)
    if (( CLEAN_FROM <= 3 && 3 <= CLEAN_TO )); then
        _FT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"
        _PM_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
        if $WIPE; then
            [[ -d "$OUTPUT_DIR/$MODEL_NAME" ]] && rm -rf "$OUTPUT_DIR/$MODEL_NAME" \
                && echo "    rm -rf $OUTPUT_DIR/$MODEL_NAME"
        else
            # Remove classification_stats JSON files (skip markers)
            for f in "$_FT_DIR"/classification_stats_*.json "$_PM_DIR"/permutation_stats_*.json; do
                [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
            done
        fi
    fi

    # Step 4: merged classification stats
    if (( CLEAN_FROM <= 4 && 4 <= CLEAN_TO )); then
        _MERGE_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_all.json"
        [[ -f "$_MERGE_FILE" ]] && rm -f "$_MERGE_FILE" && echo "    rm $_MERGE_FILE (marker)"
    fi

    # Step 5: ablation summaries
    if (( CLEAN_FROM <= 5 && 5 <= CLEAN_TO )); then
        _ABL_DIR="$OUTPUT_DIR/$MODEL_NAME/ablation"
        if $WIPE; then
            [[ -d "$_ABL_DIR" ]] && rm -rf "$_ABL_DIR" && echo "    rm -rf $_ABL_DIR"
        else
            # Remove ablation_summary.json from each sub-config
            for f in "$_ABL_DIR"/*/ablation_summary.json; do
                [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
            done
        fi
    fi

    # Step 6: visualization marker
    if (( CLEAN_FROM <= 6 && 6 <= CLEAN_TO )); then
        _PATCH="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/fig3_patched.marker"
        if $WIPE; then
            _VIZ="$OUTPUT_DIR/$MODEL_NAME/fig3"
            [[ -d "$_VIZ" ]] && rm -rf "$_VIZ" && echo "    rm -rf $_VIZ"
        fi
        [[ -f "$_PATCH" ]] && rm -f "$_PATCH" && echo "    rm $_PATCH (marker)"
    fi

    # Step 7: attention marker
    if (( CLEAN_FROM <= 7 && 7 <= CLEAN_TO )); then
        _ATTN_DIR="results/7-attention_maps"
        if $WIPE; then
            [[ -d "$_ATTN_DIR" ]] && rm -rf "$_ATTN_DIR" && echo "    rm -rf $_ATTN_DIR"
        else
            for f in "$_ATTN_DIR"/done*.marker; do
                [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
            done
        fi
    fi

    # Step 8: plot marker
    if (( CLEAN_FROM <= 8 && 8 <= CLEAN_TO )); then
        _PLOT_DIR="results/8-statistics/cross-model/${MODE_DIR}"
        _PLOT_MARKER="$_PLOT_DIR/done.marker"
        if $WIPE; then
            [[ -d "$_PLOT_DIR" ]] && rm -rf "$_PLOT_DIR" && echo "    rm -rf $_PLOT_DIR"
        else
            [[ -f "$_PLOT_MARKER" ]] && rm -f "$_PLOT_MARKER" && echo "    rm $_PLOT_MARKER (marker)"
        fi
    fi

    # Step 10: hallucination scores
    if (( CLEAN_FROM <= 10 && 10 <= CLEAN_TO )); then
        _HS_DIR="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}"
        if $WIPE; then
            [[ -d "$_HS_DIR" ]] && rm -rf "$_HS_DIR" && echo "    rm -rf $_HS_DIR"
        else
            [[ -f "$_HS_DIR/done.marker" ]] && rm -f "$_HS_DIR/done.marker" && echo "    rm $_HS_DIR/done.marker (marker)"
        fi
    fi

    # Step 11: steering results
    if (( CLEAN_FROM <= 11 && 11 <= CLEAN_TO )); then
        _ST_DIR="${OUTPUT_DIR_USER:-results/3-classify/${MODE_DIR}}/$MODEL_NAME/ablation/steering"
        if $WIPE; then
            [[ -d "$_ST_DIR" ]] && rm -rf "$_ST_DIR" && echo "    rm -rf $_ST_DIR"
        else
            # Remove per-condition marker files
            for f in "$_ST_DIR"/*/alpha_*/ablation_summary.json; do
                [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
            done
        fi
    fi

    # Steps 12-13: merged steering + ECCV plots
    if (( CLEAN_FROM <= 13 && 12 <= CLEAN_TO )); then
        _PLOT_ST_DIR="results/13-plots/${MODE_DIR}/${MODEL_NAME}"
        if $WIPE; then
            [[ -d "$_PLOT_ST_DIR" ]] && rm -rf "$_PLOT_ST_DIR" && echo "    rm -rf $_PLOT_ST_DIR"
        else
            [[ -f "$_PLOT_ST_DIR/steering_merged.json" ]] && rm -f "$_PLOT_ST_DIR/steering_merged.json" && echo "    rm $_PLOT_ST_DIR/steering_merged.json (marker)"
        fi
    fi

    # Logs: only delete with --wipe
    if $WIPE; then
        for s in $(seq "$CLEAN_FROM" "$CLEAN_TO"); do
            STEP_LOG_NAMES=("1-describe" "2-merge_descriptions" "3-classify" "4-merge_classifications" "5-ablation_validate" "6-activation_maps" "7-attention_maps" "8-statistics" "9-vtp" "10-halluc_score" "11-steering" "12-merge_steering" "13-plot_steering")
            SDIR="$LOG_DIR/${STEP_LOG_NAMES[$((s-1))]}"
            if [[ -d "$SDIR" ]]; then
                rm -rf "$SDIR" && echo "    rm -rf $SDIR"
            fi
        done
    fi

    if $WIPE; then
        echo "  ── WIPE complete (steps $CLEAN_FROM–$CLEAN_TO: results + logs deleted) ──"
    else
        echo "  ── CLEAN complete (steps $CLEAN_FROM–$CLEAN_TO: markers removed, results preserved) ──"
    fi
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# STEP 1 (describe): Generate descriptions (sharded by image range)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "gd" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: Generate descriptions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/1-describe"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

if [[ "$MODE" == "test" ]]; then
    JOB_NAME="${JN1}"
    echo ""
    echo "  ── generate_descriptions --test_fig3 ──"
    # Skip if test description file exists or job is active
    if [[ -s "$DESC_FILE" ]] || is_job_active "$JOB_NAME"; then
        echo "  [skip] $JOB_NAME — already done or active"
        SKIPPED=$((SKIPPED + 1))
    elif $LOCAL; then
        (cd "$WORK_DIR" && $PYTHON code/generate_descriptions.py \
            --model_type "$MODEL_TYPE" \
            --original_model_path "$MODEL_PATH" \
            --model_path "$MODEL_PATH" \
            --output_path "results/1-describe/test/generated_descriptions.json" \
            $GEN_ARGS) \
            2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        bsub_tiered -q $QUEUE \
             -J "$JOB_NAME" \
             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
             -- "cd $WORK_DIR && $PYTHON code/generate_descriptions.py \
                     --model_type $MODEL_TYPE \
                     --original_model_path $MODEL_PATH \
                     --model_path $MODEL_PATH \
                     --output_path results/1-describe/test/generated_descriptions.json \
                     $GEN_ARGS"
        echo "  → Job: $JOB_NAME (1 GPU, tiers: ${GPU_GMEM_TIERS[*]})"
        SUBMITTED=$((SUBMITTED + 1))
    fi
else
    # Full mode: shard across GPUs
    echo ""
    echo "  ── generate_descriptions — $SHARDS_EFFECTIVE shards ──"
    echo "  Images per shard: ~$((N_TOTAL_IMAGES / SHARDS_EFFECTIVE))"

    GEN_SUBMITTED_JOBS=()  # track which shard jobs were submitted this run
    for ((s=0; s<SHARDS_EFFECTIVE; s++)); do
        START_IDX=$((s * N_TOTAL_IMAGES / SHARDS_EFFECTIVE))
        END_IDX=$(((s + 1) * N_TOTAL_IMAGES / SHARDS_EFFECTIVE))
        JOB_NAME="gen_${s}_${SHORT_MODEL}"
        SHARD_FILE="results/1-describe/full/shards/gen_desc${_DS}_shard${s}.json"

        # Skip if shard output exists or job is active
        if [[ -s "$SHARD_FILE" ]] || is_job_active "$JOB_NAME"; then
            echo "  [skip] Shard $s ($JOB_NAME) — already done or active"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if $LOCAL; then
            echo "  Shard $s: images [$START_IDX, $END_IDX) → $SHARD_FILE"
            (cd "$WORK_DIR" && $PYTHON code/generate_descriptions.py \
                --model_type "$MODEL_TYPE" \
                --original_model_path "$MODEL_PATH" \
                --model_path "$MODEL_PATH" \
                --output_path "$SHARD_FILE" \
                --start_idx $START_IDX \
                --end_idx $END_IDX) \
                2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            bsub_tiered -q $QUEUE \
                 -J "$JOB_NAME" \
                 -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
                 -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
                 -- "cd $WORK_DIR && $PYTHON code/generate_descriptions.py \
                     --model_type $MODEL_TYPE \
                     --original_model_path $MODEL_PATH \
                     --model_path $MODEL_PATH \
                     --output_path $SHARD_FILE \
                     --start_idx $START_IDX \
                     --end_idx $END_IDX"
            echo "  → Shard $s: [$START_IDX, $END_IDX) tiers: ${GPU_GMEM_TIERS[*]} → $JOB_NAME"
            GEN_SUBMITTED_JOBS+=("$JOB_NAME")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done

    # ── Merge shards into single file ─────────────────────────
    MERGE_JOB="${JN2}"
    MERGE_CMD="$PYTHON -c \"
import json, glob, os
merged = {}
for f in sorted(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json')):
    with open(f) as fp:
        merged.update(json.load(fp))
os.makedirs(os.path.dirname('${DESC_FILE}'), exist_ok=True)
with open('${DESC_FILE}', 'w') as fp:
    json.dump(merged, fp, indent=2)
n_shards = len(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json'))
print(f'Merged {len(merged)} descriptions from {n_shards} shards → ${DESC_FILE}')
\""

    echo ""
    echo "  ── Merge → $DESC_FILE ──"

    # Skip if merged file exists or job is active
    if [[ -s "$DESC_FILE" ]] || is_job_active "$MERGE_JOB"; then
        echo "  [skip] $MERGE_JOB — already done or active"
        SKIPPED=$((SKIPPED + 1))
    elif $LOCAL; then
        (cd "$WORK_DIR" && eval "$MERGE_CMD") 2>&1 | tee "${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log"
    else
        # Build dependency on ALL active gen jobs (submitted this run OR still running)
        BSUB_MERGE_ARGS=(-q "$QUEUE" \
             -J "$MERGE_JOB" \
             -oo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log" \
             -eo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.err")
        DEP_PARTS=""
        for ((gs=0; gs<SHARDS_EFFECTIVE; gs++)); do
            gn="gen_${gs}_${SHORT_MODEL}"
            if is_job_active "$gn"; then
                if [[ -n "$DEP_PARTS" ]]; then
                    DEP_PARTS="$DEP_PARTS && done($gn)"
                else
                    DEP_PARTS="done($gn)"
                fi
            fi
        done
        if [[ -n "$DEP_PARTS" ]]; then
            BSUB_MERGE_ARGS+=(-w "$DEP_PARTS")
            echo "  → Job: $MERGE_JOB (depends on ${#GEN_SUBMITTED_JOBS[@]} shard jobs)"
        else
            echo "  → Job: $MERGE_JOB (no deps — all shards already done)"
        fi
        rm -f "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.err"
        bsub "${BSUB_MERGE_ARGS[@]}" "cd $WORK_DIR && $MERGE_CMD"
        MERGE_SUBMITTED=1
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 1 (describe)

# ═══════════════════════════════════════════════════════════════
# STEP 2 (merge_desc): Merge generated description shards
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "merge_gd" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Merge generated description shards"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/2-merge_descriptions"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

MERGE_CMD="$PYTHON -c \"
import json, glob, os
merged = {}
for f in sorted(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json')):
    with open(f) as fp:
        merged.update(json.load(fp))
os.makedirs(os.path.dirname('${DESC_FILE}'), exist_ok=True)
with open('${DESC_FILE}', 'w') as fp:
    json.dump(merged, fp, indent=2)
n_shards = len(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json'))
print(f'Merged {len(merged)} descriptions from {n_shards} shards → ${DESC_FILE}')
\""

echo ""
echo "  ── Merging → $DESC_FILE ──"
(cd "$WORK_DIR" && eval "$MERGE_CMD") 2>&1 | tee "${STEP_LOG_DIR}/merge_gd${LOG_SUFFIX}.log"

fi  # end step 2 (merge_desc)

# ═══════════════════════════════════════════════════════════════
# STEP 3 (classify): Classify neurons (sharded by layer range)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "cn" || $STEP_ALL == true ]]; then

# In test + viz-fig3 mode, classify is unnecessary — patch_fig3 handles
# the 6 specific neurons directly, so skip the entire cn step.
STEP_LOG_DIR="${LOG_DIR}/3-classify"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"
if [[ "$MODE" == "test" ]] && $VIZ_FIG3 && [[ $STEP_ALL == true ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 3: SKIPPED (test + --viz-fig3 → patch_fig3 handles neurons directly)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: Classify neurons"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "test" ]]; then
  if [[ -n "$LAYER_LIST" ]]; then
    # Test mode with --layer-list: submit one job per requested layer
    CLS_OUTPUT_DIR="${OUTPUT_DIR}"
    echo ""
    echo "  ── $CLASSIFY_SCRIPT — test mode, layer filter: $LAYER_LIST ──"
    [[ -n "$RUN_SUFFIX" ]] && echo "  Output dir:  $CLS_OUTPUT_DIR (suffixed)"
    GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
    # Strip --layer_start/--layer_end from CLASSIFY_ARGS so we can override per layer
    CLS_ARGS_NO_LAYER=$(echo "$CLASSIFY_ARGS" | sed 's/--layer_start [0-9]*//;s/--layer_end [0-9]*//')
    CLS_SUBMITTED_JOBS=()
    IFS=',' read -ra _LAYERS <<< "$LAYER_LIST"
    for _L in "${_LAYERS[@]}"; do
        JOB_NAME="${JN3}_${_L}_g${GMEM_TAG}"
        STATS_FILE="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_layers${_L}-$((_L+1)).json"
        if [[ -s "$STATS_FILE" ]] || is_job_active "$JOB_NAME"; then
            echo "  [skip] $JOB_NAME layer $_L — already done or active"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if $LOCAL; then
            (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
                --model_type "$MODEL_TYPE" \
                --original_model_path "$MODEL_PATH" \
                --text_source generated \
                --generated_desc_path "$DESC_FILE" \
                --output_dir "$CLS_OUTPUT_DIR" \
                --model "$MODEL_NAME" \
                --layer_start $_L --layer_end $((_L+1)) \
                $CLS_ARGS_NO_LAYER) \
                2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            BSUB_ARGS=(-q "$QUEUE" \
                -J "$JOB_NAME" \
                -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
            if [[ $STEP_ALL == true ]] && is_job_active "${JN1}"; then
                BSUB_ARGS+=(-w "done(${JN1})")
            fi
            bsub_tiered "${BSUB_ARGS[@]}" \
                -- "cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
                    --model_type $MODEL_TYPE \
                    --original_model_path $MODEL_PATH \
                    --text_source generated \
                    --generated_desc_path $DESC_FILE \
                    --output_dir $CLS_OUTPUT_DIR \
                    --model $MODEL_NAME \
                    --layer_start $_L --layer_end $((_L+1)) \
                    $CLS_ARGS_NO_LAYER"
            echo "  → Layer $_L: $JOB_NAME (tiers: ${GPU_GMEM_TIERS[*]})"
            CLS_SUBMITTED_JOBS+=("$JOB_NAME")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    CLS_SUBMITTED_JOBS_ALL="${CLS_SUBMITTED_JOBS[*]}"
  else
    # Test mode: single job (original behavior)
    JOB_NAME="${JN3}"
    STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_layers0-2.json"
    echo ""
    echo "  ── $CLASSIFY_SCRIPT — test mode ──"

    # Skip if stats file exists or job is active
    if [[ -s "$STATS_FILE" ]] || is_job_active "$JOB_NAME"; then
        echo "  [skip] $JOB_NAME — already done or active"
        SKIPPED=$((SKIPPED + 1))
        # Track active job so merge_nc waits for it
        if is_job_active "$JOB_NAME"; then
            CLS_SUBMITTED_JOBS_ALL="$JOB_NAME"
        fi
    elif $LOCAL; then
        (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
            --model_type "$MODEL_TYPE" \
            --original_model_path "$MODEL_PATH" \
            --text_source generated \
            --generated_desc_path "$DESC_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --model "$MODEL_NAME" \
            $CLASSIFY_ARGS) \
            2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        BSUB_ARGS=(-q "$QUEUE" \
            -J "$JOB_NAME" \
            -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
        if [[ $STEP_ALL == true ]] && is_job_active "${JN1}"; then
            BSUB_ARGS+=(-w "done(${JN1})")
        fi
        bsub_tiered "${BSUB_ARGS[@]}" \
            -- "cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
                --model_type $MODEL_TYPE \
                --original_model_path $MODEL_PATH \
                --text_source generated \
                --generated_desc_path $DESC_FILE \
                --output_dir $OUTPUT_DIR \
                --model $MODEL_NAME \
                $CLASSIFY_ARGS"
        echo "  → Job: $JOB_NAME (tiers: ${GPU_GMEM_TIERS[*]})"
        SUBMITTED=$((SUBMITTED + 1))
        CLS_SUBMITTED_JOBS_ALL="$JOB_NAME"
    fi
  fi  # end --layer-list check
else
    # Full mode: shard by layer range
    CLS_OUTPUT_DIR="${OUTPUT_DIR}"
    echo ""
    echo "  ── $CLASSIFY_SCRIPT — $CLASSIFY_SHARDS shards ──"
    echo "  Layers per shard: ~$((N_LAYERS / CLASSIFY_SHARDS))"
    [[ -n "$RUN_SUFFIX" ]] && echo "  Output dir:  $CLS_OUTPUT_DIR (suffixed)"
    [[ -n "$LAYER_LIST" ]] && echo "  Layer filter: $LAYER_LIST (skipping all others)"

    CLS_SUBMITTED_JOBS=()
    GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
    for ((s=0; s<CLASSIFY_SHARDS; s++)); do
        LAYER_START=$((s * N_LAYERS / CLASSIFY_SHARDS))
        LAYER_END=$(((s + 1) * N_LAYERS / CLASSIFY_SHARDS))

        # Skip shard if --layer-list is set and none of its layers match
        if [[ -n "$LAYER_LIST" ]]; then
            _MATCH=false
            for ((_l=LAYER_START; _l<LAYER_END; _l++)); do
                if [[ ",$LAYER_LIST," == *",$_l,"* ]]; then _MATCH=true; break; fi
            done
            if ! $_MATCH; then continue; fi
        fi

        JOB_NAME="${JN3}_${s}_g${GMEM_TAG}"
        STATS_FILE="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_layers${LAYER_START}-${LAYER_END}.json"

        # Skip if stats file exists or job is active
        if [[ -s "$STATS_FILE" ]] || is_job_active "$JOB_NAME"; then
            echo "  [skip] Shard $s ($JOB_NAME) layers [$LAYER_START, $LAYER_END) — already done or active"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if $LOCAL; then
            echo "  Shard $s: layers [$LAYER_START, $LAYER_END) → $CLS_OUTPUT_DIR"
            (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
                --model_type "$MODEL_TYPE" \
                --original_model_path "$MODEL_PATH" \
                --text_source generated \
                --generated_desc_path "$DESC_FILE" \
                --output_dir "$CLS_OUTPUT_DIR" \
                --model "$MODEL_NAME" \
                --layer_start $LAYER_START \
                --layer_end $LAYER_END) \
                2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            # Depend on merge job only if it was submitted this run
            BSUB_ARGS=(-q "$QUEUE" \
                -J "$JOB_NAME" \
                -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
            if [[ $STEP_ALL == true ]] && { [[ "$MERGE_SUBMITTED" == "1" ]] || is_job_active "$JN2"; }; then
                BSUB_ARGS+=(-w "done($JN2)")
            fi

            bsub_tiered "${BSUB_ARGS[@]}" \
                -- "cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
                    --model_type $MODEL_TYPE \
                    --original_model_path $MODEL_PATH \
                    --text_source generated \
                    --generated_desc_path $DESC_FILE \
                    --output_dir $CLS_OUTPUT_DIR \
                    --model $MODEL_NAME \
                    --layer_start $LAYER_START \
                    --layer_end $LAYER_END"
            echo "  → Shard $s: layers [$LAYER_START, $LAYER_END) tiers: ${GPU_GMEM_TIERS[*]} → $JOB_NAME"
            CLS_SUBMITTED_JOBS+=("$JOB_NAME")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done

    CLS_SUBMITTED_JOBS_ALL="${CLS_SUBMITTED_JOBS[*]}"
fi

fi  # end test+viz-fig3 skip check

fi  # end step 3 (classify)

# ═══════════════════════════════════════════════════════════════
# STEP 4 (merge_class): Merge per-shard classification results
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "merge_nc" || $STEP_ALL == true ]]; then

STEP_LOG_DIR="${LOG_DIR}/4-merge_classifications"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"
if [[ "$MODE" == "test" ]] && $VIZ_FIG3 && [[ $STEP_ALL == true ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 4: SKIPPED (test + --viz-fig3)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4: Merge classification results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MERGE_NC_CMD="cd $WORK_DIR && python3 code/merge_classification.py \
    --model_type $MODEL_TYPE \
    --model $MODEL_NAME \
    --output_dir $OUTPUT_DIR --plot"

# Skip if merged output already exists and no classify jobs are pending
MERGE_STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_all.json"
MERGE_NC_JOB_CHK="${JN4}"
if [[ -s "$MERGE_STATS_FILE" ]] && [[ -z "${CLS_SUBMITTED_JOBS_ALL:-}" ]] && ! is_job_active "$MERGE_NC_JOB_CHK"; then
    echo "  [skip] $MERGE_NC_JOB_CHK — already done ($(basename "$MERGE_STATS_FILE") exists)"
    SKIPPED=$((SKIPPED + 1))
elif is_job_active "$MERGE_NC_JOB_CHK"; then
    echo "  [skip] $MERGE_NC_JOB_CHK — already active"
    SKIPPED=$((SKIPPED + 1))
else

echo ""
echo "  ── Merging → $OUTPUT_DIR ──"

if $LOCAL || [[ "$STEP" == "merge_nc" ]]; then
    # Run inline (local mode or standalone merge_nc)
    (cd "$WORK_DIR" && python3 code/merge_classification.py \
        --model_type "$MODEL_TYPE" \
        --model "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" --plot)
else
    # Submit as LSF job dependent on all cn jobs
    MERGE_NC_JOB="${JN4}"
    BSUB_MERGE_NC_ARGS=(-q "$QUEUE" \
        -J "$MERGE_NC_JOB" \
        -oo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.err")
    # Build dependency on ALL active cn jobs (submitted this run OR still running)
    if [[ -n "${CLS_SUBMITTED_JOBS_ALL:-}" ]]; then
        DEP_PARTS=""
        for jn in $CLS_SUBMITTED_JOBS_ALL; do
            if [[ -n "$DEP_PARTS" ]]; then
                DEP_PARTS="$DEP_PARTS && done($jn)"
            else
                DEP_PARTS="done($jn)"
            fi
        done
        BSUB_MERGE_NC_ARGS+=(-w "$DEP_PARTS")
        echo "  → Job: $MERGE_NC_JOB (depends on cn jobs)"
    else
        # Check for any active cls jobs from previous runs
        DEP_PARTS=""
        for ((cs=0; cs<CLASSIFY_SHARDS; cs++)); do
            cn="${JN3}_${cs}"
            if is_job_active "$cn"; then
                if [[ -n "$DEP_PARTS" ]]; then
                    DEP_PARTS="$DEP_PARTS && done($cn)"
                else
                    DEP_PARTS="done($cn)"
                fi
            fi
        done
        if [[ -n "$DEP_PARTS" ]]; then
            BSUB_MERGE_NC_ARGS+=(-w "$DEP_PARTS")
            echo "  → Job: $MERGE_NC_JOB (depends on active cls jobs from previous run)"
        else
            echo "  → Job: $MERGE_NC_JOB (no deps — all cn jobs already done)"
        fi
    fi
    rm -f "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.err"
    bsub "${BSUB_MERGE_NC_ARGS[@]}" "$MERGE_NC_CMD"
    SUBMITTED=$((SUBMITTED + 1))
    MERGE_NC_SUBMITTED="$MERGE_NC_JOB"
fi

fi  # end merge_nc skip check

fi  # end test+viz-fig3 skip check

fi  # end step 4 (merge_class)

# ═══════════════════════════════════════════════════════════════
# STEP find_fig3: Find candidate neurons for Figure 3
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "find_fig3" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Find Figure 3 neurons (from classification results)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FIG3_DATA_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"

if [[ ! -d "$FIG3_DATA_DIR/topn_heap" ]]; then
    echo "  ERROR: Top-N Heap not found at $FIG3_DATA_DIR/topn_heap"
    echo "  Run steps 1-4 first:  bash code/run_pipeline.sh --model-type $MODEL_TYPE --mode $MODE --step all"
    exit 1
fi

echo "  Data dir:   $FIG3_DATA_DIR"
echo "  Model type: $MODEL_TYPE"
echo ""

$PYTHON code/find_fig3_neurons.py \
    --data_dir "$FIG3_DATA_DIR" \
    --model_type "$MODEL_TYPE"

fi  # end find_fig3

# ═══════════════════════════════════════════════════════════════
# STEP check_collisions: Check for image token collisions in descriptions
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "check_collisions" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Check image-token collisions in generated descriptions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ! -f "$DESC_FILE" ]]; then
    echo "  ERROR: Description file not found: $DESC_FILE"
    echo "  Run step 1 first:  bash code/run_pipeline.sh --model-type $MODEL_TYPE --mode $MODE --step 1"
    exit 1
fi

echo "  Model type: $MODEL_TYPE"
echo "  Model path: $MODEL_PATH"
echo "  Desc file:  $DESC_FILE"
echo ""

$PYTHON code/check_token_collisions.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --desc_path "$DESC_FILE"

fi  # end check_collisions



# ═══════════════════════════════════════════════════════════════
# STEP 5 (ablation): Ablation validation of neuron taxonomy
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "prune" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 5: Ablation validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/5-ablation_validate"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"
ABLATION_OUT="${OUTPUT_DIR}/$MODEL_NAME/ablation"
GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))

if [[ "$MODE" == "test" ]]; then
    # CHANGED: Added ablate_multimodal and ablate_unknown to conditions
    BASE_PRUNE_ARGS="--num_images 5 --conditions baseline ablate_vis ablate_text ablate_multimodal ablate_unknown random --pope_num_questions 3"
    PRUNE_IMAGES_EFF=5
    CHAIR_NUM_IMAGES=2
    [[ "$ABLATION_SHARDS" == "15" ]] && ABLATION_SHARDS=12  # test mode: default to 12 GPUs (unless user overrode)
    # Reduce benchmark questions in test mode (override only if user didn't set explicitly)
    [[ "$TRIVIAQA_NUM" == "2000" ]] && TRIVIAQA_NUM=10
    [[ "$MMLU_NUM" == "2000" ]]     && MMLU_NUM=10
else
    # CHANGED: Added ablate_multimodal and ablate_unknown to conditions
    BASE_PRUNE_ARGS="--num_images $PRUNE_IMAGES --conditions baseline ablate_vis ablate_text ablate_multimodal ablate_unknown random"
    PRUNE_IMAGES_EFF=$PRUNE_IMAGES
fi

# CHANGED: Add new relational benchmarks to the arguments if the folders exist
if [[ -d "$VSR_PATH" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --vsr_path $VSR_PATH"
fi

if [[ -d "$SCIENCEQA_PATH" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --scienceqa_path $SCIENCEQA_PATH"
fi

if [[ -n "$POPE_PATH" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --pope_path $POPE_PATH"
    if [[ -n "$POPE_IMG_DIR" ]]; then
        BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --pope_img_dir $POPE_IMG_DIR"
    fi
fi

if [[ -n "$CHAIR_ANN_PATH" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --chair_ann_path $CHAIR_ANN_PATH --chair_num_images $CHAIR_NUM_IMAGES"
    if [[ -n "$POPE_IMG_DIR" ]]; then
        BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --chair_img_dir $POPE_IMG_DIR"
    fi
fi

if [[ -n "$TRIVIAQA_PATH" ]] && [[ "$MODEL_TYPE" != "llava-hf" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --triviaqa_path $TRIVIAQA_PATH --triviaqa_num_questions $TRIVIAQA_NUM"
fi

if [[ -n "$MMLU_DIR" ]] && [[ "$MODEL_TYPE" != "llava-hf" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --mmlu_dir $MMLU_DIR --mmlu_num_questions $MMLU_NUM"
fi

submit_ablation_job() {
    local JOB_SUFFIX="$1"
    local OUT_SUBDIR="$2"
    local EXTRA_ARGS="$3"
    local _LABELS_DIR="$4"
    local _LABEL_SOURCE="$5"
    local JOB_BASE="${JN5}_${JOB_SUFFIX}_g${GMEM_TAG}"
    local JOB_OUT="${ABLATION_OUT}/${OUT_SUBDIR}"
    local JOB_SUMMARY="${JOB_OUT}/ablation_summary.json"
    local FULL_ARGS="$BASE_PRUNE_ARGS $EXTRA_ARGS"

    if [[ -s "$JOB_SUMMARY" ]] || is_job_active "$JOB_BASE"; then
        echo "  [skip] $JOB_BASE — already done or active"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local COMMON_CMD="$PYTHON $ABLATION_SCRIPT \
        --model_type $MODEL_TYPE \
        --original_model_path $MODEL_PATH \
        --labels_dir $_LABELS_DIR \
        --label_source $_LABEL_SOURCE \
        --output_dir $JOB_OUT \
        $FULL_ARGS"

    # ── Single shard (test mode or ABLATION_SHARDS=1) ──────────────
    if (( ABLATION_SHARDS <= 1 )); then
        local JOB_NAME="$JOB_BASE"
        echo "  → $JOB_NAME → $JOB_OUT  [${_LABEL_SOURCE}]"
        rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

        if $LOCAL; then
            (cd "$WORK_DIR" && $COMMON_CMD) \
                2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            local BSUB_ARGS=(-q "$QUEUE" \
                -J "$JOB_NAME" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err")
            if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
                BSUB_ARGS+=(-w "done(${JN4})")
            fi
            bsub_tiered "${BSUB_ARGS[@]}" -- "cd $WORK_DIR && $COMMON_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
        echo "  → Job: $JOB_NAME (1 GPU)"
        return
    fi

    # ── Multi-shard (ABLATION_SHARDS > 1) ──────────────────────────
    # Compute total (condition × top_n) runs — this is what the Python
    # script slices with --start_idx / --end_idx, NOT image count.
    local N_CONDITIONS=0
    local _in_conds=false
    for _w in $BASE_PRUNE_ARGS; do
        if [[ "$_w" == "--conditions" ]]; then _in_conds=true; continue; fi
        if $_in_conds; then
            [[ "$_w" == --* ]] && break
            N_CONDITIONS=$((N_CONDITIONS + 1))
        fi
    done
    local N_TOP_N=1
    if [[ "$FULL_ARGS" == *"--curve"* ]]; then
        IFS=',' read -ra _cs <<< "$ABLATION_CURVE_STEPS"
        N_TOP_N=$(( ${#_cs[@]} + 1 ))   # +1 for the "all" step
    fi
    local N_TOTAL_RUNS=$(( N_CONDITIONS * N_TOP_N ))

    echo "  → $JOB_BASE → $JOB_OUT  [${_LABEL_SOURCE}] (${ABLATION_SHARDS} shards, ${N_TOTAL_RUNS} runs = ${N_CONDITIONS} conds × ${N_TOP_N} top_n)"
    local SHARD_JOB_NAMES=""
    local ALL_SHARDS_DONE=true
    local SHARDS_SKIPPED=0
    for (( s=0; s<ABLATION_SHARDS; s++ )); do
        local SHARD_START=$(( s * N_TOTAL_RUNS / ABLATION_SHARDS ))
        local SHARD_END=$(( (s + 1) * N_TOTAL_RUNS / ABLATION_SHARDS ))
        local SHARD_NAME="${JOB_BASE}_${s}"
        local SHARD_OUT="${JOB_OUT}/shards/shard_${s}"
        local SHARD_SUMMARY="${SHARD_OUT}/ablation_summary.json"

        # Skip shard if its summary exists (no dep needed)
        if [[ -s "$WORK_DIR/$SHARD_SUMMARY" ]]; then
            SHARDS_SKIPPED=$((SHARDS_SKIPPED + 1))
            continue
        fi
        # Skip if job is active (add dep for merge to wait)
        if is_job_active "$SHARD_NAME"; then
            SHARDS_SKIPPED=$((SHARDS_SKIPPED + 1))
            ALL_SHARDS_DONE=false
            [[ -n "$SHARD_JOB_NAMES" ]] && SHARD_JOB_NAMES+=" && "
            SHARD_JOB_NAMES+="done($SHARD_NAME)"
            continue
        fi
        ALL_SHARDS_DONE=false

        rm -f "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.err"

        local BSUB_ARGS=(-q "$QUEUE" \
            -J "$SHARD_NAME" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.err")
        if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
            BSUB_ARGS+=(-w "done(${JN4})")
        fi
        bsub_tiered "${BSUB_ARGS[@]}" \
            -- "cd $WORK_DIR && $COMMON_CMD \
                --start_idx $SHARD_START --end_idx $SHARD_END \
                --output_dir $SHARD_OUT"

        [[ -n "$SHARD_JOB_NAMES" ]] && SHARD_JOB_NAMES+=" && "
        SHARD_JOB_NAMES+="done($SHARD_NAME)"
        SUBMITTED=$((SUBMITTED + 1))
    done
    (( SHARDS_SKIPPED > 0 )) && echo "    [skip] $SHARDS_SKIPPED/$ABLATION_SHARDS shards already done or active"

    # Merge shard results into final output
    local MERGE_NAME="${JOB_BASE}_merge"
    if [[ -s "$JOB_SUMMARY" ]]; then
        echo "    [skip] merge — $JOB_SUMMARY already exists"
    elif $ALL_SHARDS_DONE && (( SHARDS_SKIPPED == ABLATION_SHARDS )); then
        echo "    [merge] All shards done — merging inline"
        (cd "$WORK_DIR" && $PYTHON $ABLATION_SCRIPT \
            --merge_shards $JOB_OUT/shards \
            --output_dir $JOB_OUT)
    else
        rm -f "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.err"
        bsub -q "$QUEUE" -J "$MERGE_NAME" -w "$SHARD_JOB_NAMES" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $PYTHON $ABLATION_SCRIPT \
                --merge_shards $JOB_OUT/shards \
                --output_dir $JOB_OUT"
        SUBMITTED=$((SUBMITTED + 1))
    fi
    echo "  → ${ABLATION_SHARDS} shard jobs + 1 merge (tiers: ${GPU_GMEM_TIERS[*]})"
}

if $ABLATION_ALL; then

TAXONOMY_CONFIGS=(
    "xu|llm_fixed_threshold|ft"
    "llm_permutation|llm_permutation|perm"
)

echo ""
echo "  ── PARALLEL MODE: 9 approaches × 2 taxonomies × ${ABLATION_SHARDS} GPUs ──"

for TAXCFG in "${TAXONOMY_CONFIGS[@]}"; do
    IFS='|' read -r TAX_SOURCE TAX_SUBDIR TAX_PREFIX <<< "$TAXCFG"
    TAX_LABELS_DIR="$LABELS_BASE/$MODEL_NAME/$TAX_SUBDIR"

    echo ""
    echo "  ── Taxonomy: ${TAX_SOURCE} (${TAX_LABELS_DIR}) ──"

    submit_ablation_job "${TAX_PREFIX}_standard"      "${TAX_PREFIX}/1_standard"      "--ablation_method zero"                                                                                           "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_curve_label"   "${TAX_PREFIX}/2_curve_label"   "--ablation_method zero --curve --curve_steps $ABLATION_CURVE_STEPS --ranking_method label"                       "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_curve_cett"    "${TAX_PREFIX}/3_curve_cett"    "--ablation_method zero --curve --curve_steps $ABLATION_CURVE_STEPS --ranking_method cett --n_cett_images $ABLATION_N_CETT" "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_layer_0_10"    "${TAX_PREFIX}/4_layer_0_10"    "--ablation_method zero --layer_range 0-10"                                                                        "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_layer_11_21"   "${TAX_PREFIX}/5_layer_11_21"   "--ablation_method zero --layer_range 11-21"                                                                       "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_layer_22_31"   "${TAX_PREFIX}/6_layer_22_31"   "--ablation_method zero --layer_range 22-31"                                                                       "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_mean"          "${TAX_PREFIX}/7_mean"          "--ablation_method mean --n_stats_images $ABLATION_N_STATS"                                                        "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_clamp"         "${TAX_PREFIX}/9_clamp_high"    "--ablation_method clamp_high --n_stats_images $ABLATION_N_STATS"                                                  "$TAX_LABELS_DIR" "$TAX_SOURCE"
    submit_ablation_job "${TAX_PREFIX}_top500"        "${TAX_PREFIX}/10_top_n_500"    "--ablation_method zero --top_n 500"                                                                               "$TAX_LABELS_DIR" "$TAX_SOURCE"
done

else

EXTRA_PRUNE="--ablation_method $ABLATION_METHOD"
EXTRA_PRUNE="$EXTRA_PRUNE --n_stats_images $ABLATION_N_STATS"
EXTRA_PRUNE="$EXTRA_PRUNE --ranking_method $ABLATION_RANKING"
EXTRA_PRUNE="$EXTRA_PRUNE --n_cett_images $ABLATION_N_CETT"

if $ABLATION_CURVE; then EXTRA_PRUNE="$EXTRA_PRUNE --curve --curve_steps $ABLATION_CURVE_STEPS"; fi
if [[ -n "$ABLATION_TOP_N" ]]; then EXTRA_PRUNE="$EXTRA_PRUNE --top_n $ABLATION_TOP_N"; fi
if [[ -n "$ABLATION_LAYER_RANGE" ]]; then EXTRA_PRUNE="$EXTRA_PRUNE --layer_range $ABLATION_LAYER_RANGE"; fi

# ── Run for each requested taxonomy ──────────────────────────
if [[ "$ABLATION_TAXONOMY" == "ft" || "$ABLATION_TAXONOMY" == "both" ]]; then
    LABELS_DIR="$LABELS_BASE/$MODEL_NAME/llm_fixed_threshold"
    submit_ablation_job "ft" "ft" "$EXTRA_PRUNE" "$LABELS_DIR" "xu"
fi

if [[ "$ABLATION_TAXONOMY" == "pmbt" || "$ABLATION_TAXONOMY" == "both" ]]; then
    LABELS_DIR="$LABELS_BASE/$MODEL_NAME/llm_permutation"
    submit_ablation_job "perm" "perm" "$EXTRA_PRUNE" "$LABELS_DIR" "llm_permutation"
fi
fi  # end ABLATION_ALL check

fi  # end step 5 (ablation)

# ═══════════════════════════════════════════════════════════════
# STEP merge_ablation: merge shards produced by the ablation step
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "merge_ablation" ]]; then

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP merge_ablation: merging ablation shard results"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ABLATION_OUT="${OUTPUT_DIR}/$MODEL_NAME/ablation"
    if [[ ! -d "$ABLATION_OUT" ]]; then
        echo "  ERROR: ablation output not found: $ABLATION_OUT"
        exit 1
    fi

    find "$ABLATION_OUT" -type d -name shards | while read -r sharddir; do
        parentdir="$(dirname "$sharddir")"
        echo "  merging shards in $parentdir"
        $PYTHON $ABLATION_SCRIPT \
            --merge_shards "$sharddir" \
            --output_dir "$parentdir"
    done

    exit 0
fi

# ═══════════════════════════════════════════════════════════════
# STEP 6 (visualize)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "visualize" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 6: Figure 3 activation visualizations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/6-activation_maps"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

VIZ_DATA_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"
VIZ_OUT_DIR="$OUTPUT_DIR/$MODEL_NAME/fig3"
VIZ_FT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"
VIZ_PMBT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
COCO_IMG_DIR="${COCO_IMG_DIR:-/home/projects/bagon/shared/coco2017/images/train2017/}"

VIZ_ARGS="--types visual text multimodal"
if $VIZ_FIG3; then
    VIZ_ARGS="--fig3"
elif $VIZ_FIG89; then
    VIZ_ARGS="--fig89"
elif $VIZ_SUPPLEMENTARY; then
    VIZ_ARGS="--supplementary"
fi

PATCH_MARKER="$VIZ_DATA_DIR/fig3_patched.marker"
if $VIZ_FIG3 && [[ ! -f "$PATCH_MARKER" ]]; then
    PATCH_SCRIPT="code/patch_fig3_activations.py"
    if $LOCAL; then
        (cd "$WORK_DIR" && $PYTHON $PATCH_SCRIPT \
            --data_dir "$VIZ_DATA_DIR" \
            --coco_img_dir "$COCO_IMG_DIR" \
            --generated_desc_path "$DESC_FILE" \
            --model_type "$MODEL_TYPE" \
            --original_model_path "$MODEL_PATH" \
            --model_name "$MODEL_NAME" \
            --device 0) \
            2>&1 | tee "${STEP_LOG_DIR}/patch_fig3${LOG_SUFFIX}.log"
        touch "$PATCH_MARKER"
    else
        GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
        PATCH_JOB="${JN6p}_g${GMEM_TAG}"
        if ! is_job_active "$PATCH_JOB"; then
            PATCH_BSUB_ARGS=(-q "$QUEUE" \
                -J "$PATCH_JOB" \
                -oo "$WORK_DIR/${STEP_LOG_DIR}/${PATCH_JOB}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/${STEP_LOG_DIR}/${PATCH_JOB}${LOG_SUFFIX}.err")
            # Wait for gen_desc in test mode so the description file exists
            if [[ "$MODE" == "test" && "$STEP" == "all" ]] && is_job_active "${JN1}"; then
                PATCH_BSUB_ARGS+=(-w "done(${JN1})")
            fi
            bsub_tiered "${PATCH_BSUB_ARGS[@]}" \
                -- "cd $WORK_DIR && $PYTHON $PATCH_SCRIPT \
                    --data_dir $VIZ_DATA_DIR \
                    --coco_img_dir $COCO_IMG_DIR \
                    --generated_desc_path $DESC_FILE \
                    --model_type $MODEL_TYPE \
                    --original_model_path $MODEL_PATH \
                    --model_name $MODEL_NAME \
                    --device 0 \
                    && touch $PATCH_MARKER"
        fi
    fi
fi

JOB_NAME="${JN6}"
if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
elif $LOCAL; then
    (cd "$WORK_DIR" && $PYTHON $VIZ_SCRIPT \
        --data_dir "$VIZ_DATA_DIR" \
        --coco_img_dir "$COCO_IMG_DIR" \
        --generated_desc_path "$DESC_FILE" \
        --model_type "$MODEL_TYPE" \
        --model_name "$MODEL_NAME" \
        --pmbt_data_dir "$VIZ_PMBT_DIR" \
        --output_dir "$VIZ_OUT_DIR" \
        $VIZ_ARGS) \
        2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
else
    BSUB_ARGS=(-q "$QUEUE" \
        -J "$JOB_NAME" \
        -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
        -R "rusage[mem=98304]" -M 98304)
    if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
        BSUB_ARGS+=(-w "done(${JN4})")
    fi
    if is_job_active "${JN6p}"; then
        BSUB_ARGS+=(-w "done(${JN6p})")
    fi
    rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
    bsub "${BSUB_ARGS[@]}" \
        "cd $WORK_DIR && $PYTHON $VIZ_SCRIPT \
            --data_dir $VIZ_DATA_DIR \
            --coco_img_dir $COCO_IMG_DIR \
            --generated_desc_path $DESC_FILE \
            --model_type $MODEL_TYPE \
            --model_name $MODEL_NAME \
            --pmbt_data_dir $VIZ_PMBT_DIR \
            --output_dir $VIZ_OUT_DIR \
            $VIZ_ARGS"
    echo "  → Job: $JOB_NAME (CPU only)"
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end step 6 (visualize)

# ═══════════════════════════════════════════════════════════════
# STEP 7 (attention)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "attn" ]] || $STEP_ALL_FULL; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 7: Attention analysis for reclassified neurons"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/7-attention_maps"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

ATTN_OUT_DIR="results/7-attention_maps"

# Default test images (the 6 fig3 images from Xu et al.)
ATTN_DEFAULT_IMAGES=(000000403170 000000065793 000000156852 000000323964 000000276332 000000060034)

# Build list of image IDs to process
ATTN_IMAGE_IDS=()
if [[ -n "$ATTN_IMAGE_ID" ]]; then
    ATTN_IMAGE_IDS=("$ATTN_IMAGE_ID")
elif [[ -z "$ATTN_IMAGE_PATH" && -z "$ATTN_LAYER" && "$ATTN_N_SAMPLES" -le 1 ]]; then
    # No explicit target — use defaults in test mode, skip otherwise
    if [[ "$MODE" == "test" ]]; then
        ATTN_IMAGE_IDS=("${ATTN_DEFAULT_IMAGES[@]}")
        echo "  Using default test images: ${ATTN_IMAGE_IDS[*]}"
    elif $STEP_ALL_FULL; then
        echo "  [skip] step 7 (attn) — no --attn-image-id / --attn-image-path / --attn-nsamples given"
    else
        echo "  ERROR: --step attn requires --attn-image-id, --attn-image-path, or --attn-nsamples > 1"
        exit 1
    fi
fi

COCO_IMG_DIR="${COCO_IMG_DIR:-/home/projects/bagon/shared/coco2017/images/train2017/}"


# Override heatmap layers for 28-layer models (Qwen2-7B backbone) if user
# hasn't explicitly set --attn-layers.  Default "0 7 15 23 28 31" has indices
# 28 and 31 which are out of range for models with 28 layers (indices 0-27).
if [[ "$ATTN_HEATMAP_LAYERS" == "0 7 15 23 28 31" ]]; then
    if [[ "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" || "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "internvl" ]]; then
        ATTN_HEATMAP_LAYERS="0 5 10 15 20 27"
    fi
fi
# Build common args (model, paths, heatmap layers)
ATTN_COMMON_ARGS=""
ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --heatmap_layers $ATTN_HEATMAP_LAYERS"
ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --model_type $MODEL_TYPE"
if [[ "$MODEL_TYPE" == "llava-hf" ]]; then
    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llava-1.5-7b-hf"
elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llava-onevision-qwen2-7b-ov-hf"
else
    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --original_model_path $MODEL_PATH"
fi
ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --coco_img_dir $COCO_IMG_DIR"
ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --generated_desc_path $DESC_FILE"
ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --output_dir $ATTN_OUT_DIR"
if [[ -n "$ATTN_HIGHLIGHTED_WORDS" ]]; then
    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --highlighted_words $ATTN_HIGHLIGHTED_WORDS"
fi

# ── Image ID loop (default test images or --attn-image-id) ──────────
for _IMG_ID in "${ATTN_IMAGE_IDS[@]}"; do
    ATTN_TAG="_${_IMG_ID}"
    ATTN_MARKER="$ATTN_OUT_DIR/done${ATTN_TAG}.marker"
    JOB_NAME="${JN7}${ATTN_TAG}"
    ATTN_ARGS="--image_id $_IMG_ID $ATTN_COMMON_ARGS"

    if [[ -f "$ATTN_MARKER" ]] || is_job_active "$JOB_NAME"; then
        echo "  [skip] $JOB_NAME — already done or active"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $LOCAL; then
        (cd "$WORK_DIR" && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0) \
            2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        touch "$ATTN_MARKER"
    else
        rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
        BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
            -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
        if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
            BSUB_ARGS+=(-w "done(${JN4})")
        fi
        bsub_tiered "${BSUB_ARGS[@]}" \
            -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0 \
                && touch $ATTN_MARKER"
        echo "  → Job: $JOB_NAME (image $_IMG_ID)"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

# ── Custom image path (--attn-image-path) ───────────────────────────
if [[ -n "$ATTN_IMAGE_PATH" ]]; then
    ATTN_TAG="_custom"
    ATTN_MARKER="$ATTN_OUT_DIR/done${ATTN_TAG}.marker"
    JOB_NAME="${JN7}${ATTN_TAG}"
    ATTN_ARGS="--image_path $ATTN_IMAGE_PATH $ATTN_COMMON_ARGS"

    if [[ -f "$ATTN_MARKER" ]] || is_job_active "$JOB_NAME"; then
        echo "  [skip] $JOB_NAME — already done or active"
        SKIPPED=$((SKIPPED + 1))
    elif $LOCAL; then
        (cd "$WORK_DIR" && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0) \
            2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        touch "$ATTN_MARKER"
    else
        rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
        bsub_tiered -q "$QUEUE" -J "$JOB_NAME" \
            -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
            -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0 \
                && touch $ATTN_MARKER"
        echo "  → Job: $JOB_NAME (custom path)"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

# ── Neuron auto-highlight mode (--attn-layer + --attn-neuron-idx + --attn-nsamples) ──
if [[ -n "$ATTN_LAYER" && -n "$ATTN_NEURON_IDX" && "$ATTN_N_SAMPLES" -gt 1 ]]; then
    ATTN_TAG="_layer${ATTN_LAYER}_neuron${ATTN_NEURON_IDX}"
    ATTN_MARKER="$ATTN_OUT_DIR/done${ATTN_TAG}.marker"
    JOB_NAME="${JN7}${ATTN_TAG}"
    ATTN_ARGS="$ATTN_COMMON_ARGS --layer $ATTN_LAYER --neuron_idx $ATTN_NEURON_IDX --top_k $ATTN_TOP_K --n_samples $ATTN_N_SAMPLES"
    ATTN_ARGS="$ATTN_ARGS --data_dir $OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"

    if [[ -f "$ATTN_MARKER" ]] || is_job_active "$JOB_NAME"; then
        echo "  [skip] $JOB_NAME — already done or active"
        SKIPPED=$((SKIPPED + 1))
    elif [[ "$ATTN_GPUS" -gt 1 ]]; then
        N_SHARDS="$ATTN_GPUS"
        SAMPLES_PER_SHARD=$(( (ATTN_N_SAMPLES + N_SHARDS - 1) / N_SHARDS ))
        SHARD_JOB_NAMES=""
        for (( s=0; s<N_SHARDS; s++ )); do
            S_START=$(( s * SAMPLES_PER_SHARD ))
            S_END=$(( S_START + SAMPLES_PER_SHARD ))
            if [[ "$S_END" -gt "$ATTN_N_SAMPLES" ]]; then S_END="$ATTN_N_SAMPLES"; fi
            if [[ "$S_START" -ge "$ATTN_N_SAMPLES" ]]; then break; fi
            SHARD_NAME="${JOB_NAME}_shard${s}"
            SHARD_ARGS="$ATTN_ARGS --sample_start $S_START --sample_end $S_END"
            bsub_tiered -q "$QUEUE" -J "$SHARD_NAME" \
                -oo "$WORK_DIR/${STEP_LOG_DIR}/${SHARD_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/${STEP_LOG_DIR}/${SHARD_NAME}${LOG_SUFFIX}.err" \
                -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $SHARD_ARGS --device 0"
            SUBMITTED=$((SUBMITTED + 1))
            if [[ -z "$SHARD_JOB_NAMES" ]]; then SHARD_JOB_NAMES="done($SHARD_NAME)"
            else SHARD_JOB_NAMES="$SHARD_JOB_NAMES && done($SHARD_NAME)"; fi
        done
        MERGE_NAME="${JOB_NAME}_merge"
        bsub -q "$QUEUE" -J "$MERGE_NAME" -w "$SHARD_JOB_NAMES" \
            -R "rusage[mem=4096]" \
            -oo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT \
                --merge --layer $ATTN_LAYER --neuron_idx $ATTN_NEURON_IDX \
                --n_samples $ATTN_N_SAMPLES --output_dir $ATTN_OUT_DIR \
                && touch $ATTN_MARKER"
        SUBMITTED=$((SUBMITTED + 1))
    elif $LOCAL; then
        (cd "$WORK_DIR" && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0) \
            2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        touch "$ATTN_MARKER"
    else
        rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
        bsub_tiered -q "$QUEUE" -J "$JOB_NAME" \
            -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
            -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0 \
                && touch $ATTN_MARKER"
        echo "  → Job: $JOB_NAME (neuron auto-highlight, ${ATTN_N_SAMPLES} samples)"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi
fi  # end step 7 (attention)

# ═══════════════════════════════════════════════════════════════
# STEP 8 (statistics)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "plot" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 8: Fig7 cross-model comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/8-statistics"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

PLOT_OUT_BASE="results/8-statistics/cross-model/${MODE_DIR}"
PLOT_DPI=200
PLOT_MARKER="$PLOT_OUT_BASE/done.marker"

JOB_NAME="${JN8}"
if [[ -f "$PLOT_MARKER" ]] || is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already done or active"
    SKIPPED=$((SKIPPED + 1))
else

PLOT_WRAPPER="$WORK_DIR/${STEP_LOG_DIR}/plot_wrapper${OUT_SUFFIX}.sh"
cat > "$PLOT_WRAPPER" << 'PLOTEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$WORK_DIR"

# ── Auto-discover all models with classification results ──
FT_DIRS=()
PMBT_DIRS=()
NAMES=()
TYPES=()

# Known model → model_type mapping
declare -A NAME_TO_TYPE=(
    ["llava-1.5-7b"]="llava-liuhaotian"
    ["internvl2.5-8b"]="internvl"
    ["qwen2.5-vl-7b"]="qwen2vl"
    ["llava-onevision-7b"]="llava-ov"
)

echo "  Scanning for available models in $OUTPUT_DIR ..."
# Fixed left-to-right column order for the cross-model figure:
#   LLaVA-1.5  →  LLaVA-OV  →  Qwen  →  InternVL
ORDERED_MODELS=("llava-1.5-7b" "llava-onevision-7b" "qwen2.5-vl-7b" "internvl2.5-8b")
for mname in "${ORDERED_MODELS[@]}"; do
    model_dir="$OUTPUT_DIR/$mname/"
    [[ ! -d "$model_dir" ]] && continue   # skip if this model hasn't been run yet

    ft_dir="$model_dir/llm_fixed_threshold"
    pmbt_dir="$model_dir/llm_permutation"
    mtype="${NAME_TO_TYPE[$mname]:-llava-hf}"

    # Check FT has merged labels (at least one layer dir with neuron_labels.json)
    ft_has_labels=false
    if [[ -d "$ft_dir" ]]; then
        for lbl in "$ft_dir"/*/neuron_labels.json; do
            [[ -f "$lbl" ]] && ft_has_labels=true && break
        done
    fi

    # Check PMBT has merged labels
    pmbt_has_labels=false
    if [[ -d "$pmbt_dir" ]]; then
        for lbl in "$pmbt_dir"/*/neuron_labels_permutation.json; do
            [[ -f "$lbl" ]] && pmbt_has_labels=true && break
        done
        # Also check neuron_labels.json fallback
        if ! $pmbt_has_labels; then
            for lbl in "$pmbt_dir"/*/neuron_labels.json; do
                [[ -f "$lbl" ]] && pmbt_has_labels=true && break
            done
        fi
    fi

    if $ft_has_labels; then
        FT_DIRS+=("$ft_dir")
        NAMES+=("$mname")
        TYPES+=("$mtype")
        echo "    ✓ $mname (FT: $ft_dir)"
    fi

    if $pmbt_has_labels; then
        PMBT_DIRS+=("$pmbt_dir")
        echo "    ✓ $mname (PMBT: $pmbt_dir)"
    fi
done

PLOT_OK=0

# ── Combined FT + PMBT cross-model figure (PNG) ──────────
if [[ ${#FT_DIRS[@]} -ge 1 && ${#PMBT_DIRS[@]} -ge 1 ]]; then
    echo ""; echo "  [1] Fig8 combined FT+PMBT PNG (${#FT_DIRS[@]} models)"

    # Build per-model label-file list for PMBT row
    PMBT_LFILES=()
    for pd in "${PMBT_DIRS[@]}"; do
        found_perm=false
        for lbl in "$pd"/*/neuron_labels_permutation.json; do
            [[ -f "$lbl" ]] && found_perm=true && break
        done
        if $found_perm; then
            PMBT_LFILES+=("neuron_labels_permutation.json")
        else
            PMBT_LFILES+=("neuron_labels.json")
        fi
    done

    if $PYTHON $PLOT_SCRIPT \
        --ft_dirs   "${FT_DIRS[@]}" \
        --pmbt_dirs "${PMBT_DIRS[@]}" \
        --model_names "${NAMES[@]}" \
        --ft_model_types   "${TYPES[@]}" \
        --pmbt_model_types "${TYPES[@]}" \
        --pmbt_label_files "${PMBT_LFILES[@]}" \
        --output_dir "$PLOT_OUT_BASE/cross-model" \
        --dpi "$PLOT_DPI" --format png --fig8; then
        PLOT_OK=$((PLOT_OK + 1))
    else
        echo "  FAILED: Fig8 combined PNG"
    fi
else
    echo "  SKIP Fig8 PNG: need both FT and PMBT models"
fi

# ── Combined FT + PMBT cross-model figure (PDF) ──────────
if [[ ${#FT_DIRS[@]} -ge 1 && ${#PMBT_DIRS[@]} -ge 1 ]]; then
    echo ""; echo "  [2] Fig8 combined FT+PMBT PDF"

    if $PYTHON $PLOT_SCRIPT \
        --ft_dirs   "${FT_DIRS[@]}" \
        --pmbt_dirs "${PMBT_DIRS[@]}" \
        --model_names "${NAMES[@]}" \
        --ft_model_types   "${TYPES[@]}" \
        --pmbt_model_types "${TYPES[@]}" \
        --pmbt_label_files "${PMBT_LFILES[@]}" \
        --output_dir "$PLOT_OUT_BASE/cross-model" \
        --dpi 300 --format pdf --fig8; then
        PLOT_OK=$((PLOT_OK + 1))
    else
        echo "  FAILED: Fig8 combined PDF"
    fi
fi

echo ""; echo "  ALL PLOTS COMPLETE → $PLOT_OUT_BASE ($PLOT_OK succeeded)"
if [[ "$PLOT_OK" -eq 0 ]]; then
    echo "  ERROR: No plots succeeded — not marking as done"
    exit 1
fi
PLOTEOF

chmod +x "$PLOT_WRAPPER"

if $LOCAL; then
    PYTHON="$PYTHON" PLOT_SCRIPT="$PLOT_SCRIPT" OUTPUT_DIR="$OUTPUT_DIR" \
    PLOT_OUT_BASE="$PLOT_OUT_BASE" PLOT_DPI="$PLOT_DPI" WORK_DIR="$WORK_DIR" \
        bash "$PLOT_WRAPPER" 2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        touch "$PLOT_MARKER"
    else
        echo "  WARNING: plot wrapper failed — not marking as done"
    fi
else
    BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
        -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
    if [[ $STEP_ALL == true ]]; then
        # Prefer waiting on merge_nc (submitted this run OR still active)
        if [[ -n "${MERGE_NC_SUBMITTED:-}" ]]; then
            BSUB_ARGS+=(-w "done($MERGE_NC_SUBMITTED)")
        elif is_job_active "${JN4}"; then
            BSUB_ARGS+=(-w "done(${JN4})")
        # Fallback: wait for classify if merge_nc was skipped
        elif [[ -n "${CLS_SUBMITTED_JOBS_ALL:-}" ]]; then
            BSUB_ARGS+=(-w "done(${CLS_SUBMITTED_JOBS_ALL%% *})")
        elif [[ "$MODE" == "test" ]] && is_job_active "${JN3}"; then
            BSUB_ARGS+=(-w "done(${JN3})")
        fi
    fi
    rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
    bsub "${BSUB_ARGS[@]}" \
        "cd $WORK_DIR && \
         PYTHON='$PYTHON' PLOT_SCRIPT='$PLOT_SCRIPT' OUTPUT_DIR='$OUTPUT_DIR' \
         PLOT_OUT_BASE='$PLOT_OUT_BASE' PLOT_DPI='$PLOT_DPI' WORK_DIR='$WORK_DIR' \
         bash $PLOT_WRAPPER && touch $PLOT_MARKER"
    echo "  → Job: $JOB_NAME (CPU only)"
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end skip check

fi  # end step 8 (statistics)

# ═══════════════════════════════════════════════════════════════
# STEP 10 (halluc_score): Hallucination taxonomy — per-neuron ablation on POPE + enrichment
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "halluc_score" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 10: Hallucination taxonomy (per-neuron POPE ablation → enrichment)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/10-halluc_score"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

HALLUC_SCORES_DIR="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}${RUN_SUFFIX}"
LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"

# ── Resolve taxonomy label dir ────────────────────────────────
# Prefer full-mode labels (all layers classified) even when running step 10
# in test mode. Test mode here means "fewer POPE questions" not "use
# incomplete labels".  Fall back to current MODE_DIR if full doesn't exist.
if [[ "$STEERING_TAXONOMY" == "ft" ]]; then
    _tax_subdir="llm_fixed_threshold"
    HS_TAXONOMY="ft"
elif [[ "$STEERING_TAXONOMY" == "pmbt" ]]; then
    _tax_subdir="llm_permutation"
    HS_TAXONOMY="pmbt"
else
    _tax_subdir="llm_permutation"
    HS_TAXONOMY="pmbt"
fi

_full_labels="results/3-classify/full/$MODEL_NAME/$_tax_subdir"
_mode_labels="$LABELS_BASE/$MODEL_NAME/$_tax_subdir"
if [[ -d "$_full_labels" ]]; then
    HS_LABEL_DIR="$_full_labels"
    echo "  Using full-mode labels: $HS_LABEL_DIR"
elif [[ -d "$_mode_labels" ]]; then
    HS_LABEL_DIR="$_mode_labels"
    echo "  Using ${MODE}-mode labels: $HS_LABEL_DIR"
else
    echo "  ERROR: No labels found at $_full_labels or $_mode_labels"
    echo "         Run steps 1-4 first to classify neurons."
    exit 1
fi

# -- Test vs Full mode --
if [[ "$MODE" == "test" ]]; then
    HS_N_POPE=5
    HS_N_LAYERS=1
    HS_N_GPUS=1
    HS_BATCH_NEURONS=50
    echo "  [test] $HS_N_POPE POPE, $HS_N_LAYERS layer, $HS_N_GPUS GPU, batch=$HS_BATCH_NEURONS"
else
    HS_N_POPE="${HALLUC_POPE_SAMPLES:-500}"
    HS_N_LAYERS=$N_LAYERS
    HS_N_GPUS=$HALLUC_SCORE_SHARDS
    HS_BATCH_NEURONS=500
    echo "  [full] $HS_N_POPE POPE, $HS_N_LAYERS layers, $HS_N_GPUS GPUs, batch=$HS_BATCH_NEURONS"
fi
HS_COMMON_ARGS="--model_type $MODEL_TYPE --model_path $MODEL_PATH"
HS_COMMON_ARGS="$HS_COMMON_ARGS --model_name $MODEL_NAME"
HS_COMMON_ARGS="$HS_COMMON_ARGS --label_dir $HS_LABEL_DIR --taxonomy $HS_TAXONOMY"
HS_COMMON_ARGS="$HS_COMMON_ARGS --pope_path $POPE_PATH --pope_img_dir $POPE_IMG_DIR"
HS_COMMON_ARGS="$HS_COMMON_ARGS --n_pope_questions $HS_N_POPE"
HS_COMMON_ARGS="$HS_COMMON_ARGS --n_gpus $HS_N_GPUS"
HS_COMMON_ARGS="$HS_COMMON_ARGS --n_layers $HS_N_LAYERS"
HS_COMMON_ARGS="$HS_COMMON_ARGS --batch_neurons $HS_BATCH_NEURONS"
HS_COMMON_ARGS="$HS_COMMON_ARGS --output_dir $HALLUC_SCORES_DIR"

# Contrastive filtering
if [[ "$HALLUC_CONTRASTIVE" == "1" ]]; then
    HS_COMMON_ARGS="$HS_COMMON_ARGS --contrastive"
    if [[ "$MODE" == "test" ]]; then
        HS_COMMON_ARGS="$HS_COMMON_ARGS --contrastive_start_per_split 5 --contrastive_samples 2 --contrastive_cap_per_split 3"
        echo "  [contrastive-test] 5/split × 2 samples → cap 3/split (~30 forward passes)"
    else
        echo "  [contrastive] Enabled: 1250/split × 10 samples → cap 333/split ≈ 1000 clean"
    fi
fi

# TriviaQA dual-source scoring (Section 3.5)
if [[ "$HALLUC_TRIVIAQA" == "1" ]]; then
    HS_COMMON_ARGS="$HS_COMMON_ARGS --halluc_triviaqa"
    HS_COMMON_ARGS="$HS_COMMON_ARGS --triviaqa_path $TRIVIAQA_PATH"
    HS_COMMON_ARGS="$HS_COMMON_ARGS --triviaqa_num $TRIVIAQA_NUM"
    if [[ "$MODE" == "test" ]]; then
        HS_COMMON_ARGS="$HS_COMMON_ARGS --triviaqa_cap 10"
        echo "  [triviaqa-test] cap=10 questions for test mode"
    else
        echo "  [triviaqa] Enabled: pool=$TRIVIAQA_NUM → cap=1000 clean questions"
    fi
fi

# Skip ablation loop — reuse existing ΔH scores from previous run
if [[ "$HALLUC_SKIP_ABLATION" == "1" ]]; then
    EXISTING_SCORES="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}/ablation_scores.json"
    if [[ -f "$WORK_DIR/$EXISTING_SCORES" ]]; then
        HS_COMMON_ARGS="$HS_COMMON_ARGS --skip_ablation --ablation_scores $EXISTING_SCORES"
        echo "  [skip-ablation] Reusing existing ΔH scores: $EXISTING_SCORES"
        echo "  → Only running: contrastive preprocessing + CETT-diff + enrichment (~70 min)"
    else
        echo "  ERROR: --halluc-skip-ablation set but no existing scores at $EXISTING_SCORES"
        echo "         Run step 10 without --halluc-skip-ablation first."
        exit 1
    fi
fi

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JOB_NAME="${JN10}_g${GMEM_TAG}"
echo "  → $JOB_NAME → $HALLUC_SCORES_DIR"
rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
      "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

FULL_CMD="$PYTHON $HALLUC_SCORE_SCRIPT $HS_COMMON_ARGS"

if $LOCAL; then
    (cd "$WORK_DIR" && $FULL_CMD) \
        2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
else
    # Request N GPUs on a single node — script handles internal sharding
    BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
        -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err")
    if (( HS_N_GPUS > 1 )); then
        # Multi-GPU: use bsub directly (bsub_tiered hardcodes num=1)
        BSUB_ARGS+=(-gpu "num=$HS_N_GPUS:gmem=${GPU_GMEM_TIERS[0]}")
        BSUB_ARGS+=(-R "rusage[mem=153600] order[-gpu_maxfactor]")
        bsub "${BSUB_ARGS[@]}" "cd $WORK_DIR && $FULL_CMD"
    else
        # Single-GPU: use tiered submission
        bsub_tiered "${BSUB_ARGS[@]}" -- "cd $WORK_DIR && $FULL_CMD"
    fi
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end step 10 (halluc_score)

# ═══════════════════════════════════════════════════════════════
# STEP 11 (steering): Activation steering with neuron_ablation_validate.py --alpha
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "steering" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 11: Activation steering (7 benchmarks × alpha sweep)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/11-steering"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"
STEERING_OUT="${OUTPUT_DIR}/$MODEL_NAME/ablation/steering"
GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))

# ── Test mode: 1 alpha, all conditions, no curve, minimal benchmarks ──
if [[ "$MODE" == "test" ]]; then
    ST_ALPHAS="0.5"
    ST_BASE_ARGS="--num_images 0 --conditions baseline ablate_vis ablate_text ablate_multimodal ablate_unknown random ablate_encoder ablate_projector --pope_num_questions 5"
    ST_SHARDS=1
    ST_CHAIR_NUM=5
    [[ "$TRIVIAQA_NUM" == "2000" ]] && TRIVIAQA_NUM=5
    [[ "$MMLU_NUM" == "2000" ]]     && MMLU_NUM=5
    echo "  [test] alpha=0.5, 8 conditions (all + encoder/projector), CHAIR=5, 1 GPU"
else
    ST_ALPHAS="$STEERING_ALPHAS"
    ST_BASE_ARGS="--conditions baseline ablate_vis ablate_text ablate_multimodal ablate_unknown random ablate_encoder ablate_projector"
    ST_SHARDS=$STEERING_SHARDS
    ST_CHAIR_NUM=$CHAIR_NUM_IMAGES

    if $ABLATION_CURVE; then
        ST_BASE_ARGS="$ST_BASE_ARGS --curve --curve_steps $ABLATION_CURVE_STEPS"
    fi

    IFS=',' read -ra _alphas <<< "$ST_ALPHAS"
    echo "  [full] ${#_alphas[@]} alpha values × ${ST_SHARDS} GPUs each"
fi

# ── Append benchmark paths ──
if [[ -n "$POPE_PATH" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --pope_path $POPE_PATH"
    [[ -n "$POPE_IMG_DIR" ]] && ST_BASE_ARGS="$ST_BASE_ARGS --pope_img_dir $POPE_IMG_DIR"
fi
if [[ -n "$CHAIR_ANN_PATH" ]] && (( ST_CHAIR_NUM > 0 )); then
    ST_BASE_ARGS="$ST_BASE_ARGS --chair_ann_path $CHAIR_ANN_PATH --chair_num_images $ST_CHAIR_NUM"
    [[ -n "$POPE_IMG_DIR" ]] && ST_BASE_ARGS="$ST_BASE_ARGS --chair_img_dir $POPE_IMG_DIR"
fi
if [[ -n "$TRIVIAQA_PATH" ]] && [[ "$MODEL_TYPE" != "llava-hf" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --triviaqa_path $TRIVIAQA_PATH --triviaqa_num_questions $TRIVIAQA_NUM"
fi
if [[ -n "$MMLU_DIR" ]] && [[ "$MODEL_TYPE" != "llava-hf" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --mmlu_dir $MMLU_DIR --mmlu_num_questions $MMLU_NUM"
fi
if [[ -d "$VSR_PATH" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --vsr_path $VSR_PATH"
fi
if [[ -d "$SCIENCEQA_PATH" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --scienceqa_path $SCIENCEQA_PATH"
fi

# ── Hallucination score ranking (from step 10) ──
# Resolve the correct JSON file based on HALLUC_SCORE_METHOD
HALLUC_SCORES_DIR_ST="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}"
case "$HALLUC_SCORE_METHOD" in
    combined) HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/combined_halluc_scores.json" ;;
    dh)       HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/ablation_scores.json" ;;
    cett)     HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/cett_diff_scores.json" ;;
    *)        echo "  WARNING: Unknown halluc score method '$HALLUC_SCORE_METHOD', using combined"
              HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/combined_halluc_scores.json" ;;
esac

if [[ -f "$WORK_DIR/$HALLUC_SCORES_FILE" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --halluc_scores_path $HALLUC_SCORES_FILE"
    echo "  [halluc ranking] $HALLUC_SCORE_METHOD → $HALLUC_SCORES_FILE"
else
    echo "  [halluc ranking] $HALLUC_SCORES_FILE not found — using classification confidence"
    echo "         (run step 10 with --halluc-contrastive to generate halluc scores)"
fi

# ── Function to submit one alpha value ──
submit_steering_alpha() {
    local ALPHA="$1"
    local TAX_SOURCE="$2"
    local TAX_LABELS_DIR="$3"
    local TAX_PREFIX="$4"
    local JOB_SUFFIX="${TAX_PREFIX}_a${ALPHA}"
    local JOB_BASE="${JN11}_${JOB_SUFFIX}_g${GMEM_TAG}"
    local JOB_OUT="${STEERING_OUT}/${TAX_PREFIX}/alpha_${ALPHA}"
    local JOB_SUMMARY="${JOB_OUT}/ablation_summary.json"

    if [[ -s "$JOB_SUMMARY" ]] || is_job_active "$JOB_BASE"; then
        echo "  [skip] $JOB_BASE — already done or active"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local COMMON_CMD="$PYTHON $ABLATION_SCRIPT \
        --model_type $MODEL_TYPE \
        --original_model_path $MODEL_PATH \
        --labels_dir $TAX_LABELS_DIR \
        --label_source $TAX_SOURCE \
        --output_dir $JOB_OUT \
        --alpha $ALPHA \
        $ST_BASE_ARGS"

    # ── Single shard ──
    if (( ST_SHARDS <= 1 )); then
        local JOB_NAME="$JOB_BASE"
        echo "  → $JOB_NAME (α=$ALPHA, ${TAX_SOURCE}) → $JOB_OUT"
        rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

        if $LOCAL; then
            (cd "$WORK_DIR" && $COMMON_CMD) \
                2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            bsub_tiered -q "$QUEUE" -J "$JOB_NAME" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
                -- "cd $WORK_DIR && $COMMON_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
        return
    fi

    # ── Multi-shard: split (condition × top_n) runs across GPUs ──
    # Count conditions
    local N_CONDITIONS=0
    local _in_conds=false
    for _w in $ST_BASE_ARGS; do
        if [[ "$_w" == "--conditions" ]]; then _in_conds=true; continue; fi
        if $_in_conds; then
            [[ "$_w" == --* ]] && break
            N_CONDITIONS=$((N_CONDITIONS + 1))
        fi
    done
    local N_TOP_N=1
    if [[ "$ST_BASE_ARGS" == *"--curve"* ]]; then
        IFS=',' read -ra _cs <<< "$ABLATION_CURVE_STEPS"
        N_TOP_N=$(( ${#_cs[@]} + 1 ))
    fi
    local N_TOTAL_RUNS=$(( N_CONDITIONS * N_TOP_N ))

    echo "  → $JOB_BASE (α=$ALPHA, ${TAX_SOURCE}, ${ST_SHARDS} shards, ${N_TOTAL_RUNS} runs)"
    local SHARD_JOB_NAMES=""
    local ALL_SHARDS_DONE=true
    local SHARDS_SKIPPED=0
    for (( s=0; s<ST_SHARDS; s++ )); do
        local SHARD_START=$(( s * N_TOTAL_RUNS / ST_SHARDS ))
        local SHARD_END=$(( (s + 1) * N_TOTAL_RUNS / ST_SHARDS ))
        local SHARD_NAME="${JOB_BASE}_${s}"
        local SHARD_OUT="${JOB_OUT}/shards/shard_${s}"
        local SHARD_SUMMARY="${SHARD_OUT}/ablation_summary.json"

        # Skip shard if its summary exists (no dep needed)
        if [[ -s "$WORK_DIR/$SHARD_SUMMARY" ]]; then
            SHARDS_SKIPPED=$((SHARDS_SKIPPED + 1))
            continue
        fi
        # Skip if job is active (add dep for merge to wait)
        if is_job_active "$SHARD_NAME"; then
            SHARDS_SKIPPED=$((SHARDS_SKIPPED + 1))
            ALL_SHARDS_DONE=false
            [[ -n "$SHARD_JOB_NAMES" ]] && SHARD_JOB_NAMES+=" && "
            SHARD_JOB_NAMES+="done($SHARD_NAME)"
            continue
        fi
        ALL_SHARDS_DONE=false

        rm -f "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.err"

        bsub_tiered -q "$QUEUE" -J "$SHARD_NAME" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.err" \
            -- "cd $WORK_DIR && $COMMON_CMD \
                --start_idx $SHARD_START --end_idx $SHARD_END \
                --output_dir $SHARD_OUT"

        [[ -n "$SHARD_JOB_NAMES" ]] && SHARD_JOB_NAMES+=" && "
        SHARD_JOB_NAMES+="done($SHARD_NAME)"
        SUBMITTED=$((SUBMITTED + 1))
    done
    (( SHARDS_SKIPPED > 0 )) && echo "    [skip] $SHARDS_SKIPPED/$ST_SHARDS shards already done or active"

    # Merge shard results
    local MERGE_NAME="${JOB_BASE}_merge"
    if [[ -s "$JOB_SUMMARY" ]]; then
        echo "    [skip] merge — $JOB_SUMMARY already exists"
    elif $ALL_SHARDS_DONE && (( SHARDS_SKIPPED == ST_SHARDS )); then
        echo "    [merge] All shards done — merging inline"
        (cd "$WORK_DIR" && $PYTHON $ABLATION_SCRIPT \
            --merge_shards $JOB_OUT/shards \
            --output_dir $JOB_OUT)
    else
        rm -f "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.err"
        bsub -q "$QUEUE" -J "$MERGE_NAME" -w "$SHARD_JOB_NAMES" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $PYTHON $ABLATION_SCRIPT \
                --merge_shards $JOB_OUT/shards \
                --output_dir $JOB_OUT"
        SUBMITTED=$((SUBMITTED + 1))
    fi
    echo "  → ${ST_SHARDS} shard jobs + 1 merge (tiers: ${GPU_GMEM_TIERS[*]})"
}

# ── Loop over alpha values × taxonomies ──
# Prefer full-mode labels (same logic as step 10)
_ft_full="results/3-classify/full/$MODEL_NAME/llm_fixed_threshold"
_ft_mode="$LABELS_BASE/$MODEL_NAME/llm_fixed_threshold"
if [[ -d "$_ft_full" ]]; then _ST_FT_DIR="$_ft_full"
elif [[ -d "$_ft_mode" ]]; then _ST_FT_DIR="$_ft_mode"
else _ST_FT_DIR="$_ft_mode"; fi

_pm_full="results/3-classify/full/$MODEL_NAME/llm_permutation"
_pm_mode="$LABELS_BASE/$MODEL_NAME/llm_permutation"
if [[ -d "$_pm_full" ]]; then _ST_PM_DIR="$_pm_full"
elif [[ -d "$_pm_mode" ]]; then _ST_PM_DIR="$_pm_mode"
else _ST_PM_DIR="$_pm_mode"; fi

IFS=',' read -ra ALPHA_LIST <<< "$ST_ALPHAS"
for ALPHA in "${ALPHA_LIST[@]}"; do
    echo ""
    echo "  ── Alpha = $ALPHA ──"

    if [[ "$STEERING_TAXONOMY" == "ft" || "$STEERING_TAXONOMY" == "both" ]]; then
        submit_steering_alpha "$ALPHA" "xu" \
            "$_ST_FT_DIR" "ft"
    fi

    if [[ "$STEERING_TAXONOMY" == "pmbt" || "$STEERING_TAXONOMY" == "both" ]]; then
        submit_steering_alpha "$ALPHA" "llm_permutation" \
            "$_ST_PM_DIR" "perm"
    fi
done

fi  # end step 11 (steering)


# ═══════════════════════════════════════════════════════════════
# STEP 12 (merge_steering): Merge all alpha results into one JSON
# ═══════════════════════════════════════════════════════════════

if [[ "$STEP" == "merge_steering" ]]; then

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 12: Merge steering results"
echo "═══════════════════════════════════════════════════════════"

STEERING_OUT="${OUTPUT_DIR}/$MODEL_NAME/ablation/steering"
PLOT_OUT_DIR="results/13-plots/${MODE_DIR}/${MODEL_NAME}${RUN_SUFFIX}"

# Determine taxonomy prefix
case "$STEERING_TAXONOMY" in
    ft)   MERGE_TAX="ft" ;;
    pmbt) MERGE_TAX="perm" ;;
    *)    MERGE_TAX="perm" ;;
esac

MERGE_CMD="$PYTHON $MERGE_STEERING_SCRIPT \
    --steering_dir $STEERING_OUT \
    --taxonomy $MERGE_TAX \
    --model_name $MODEL_NAME \
    --output_dir $PLOT_OUT_DIR"

MERGED_JSON="$PLOT_OUT_DIR/steering_merged.json"

if [[ -f "$WORK_DIR/$MERGED_JSON" ]]; then
    echo "  [skip] $MERGED_JSON already exists"
    SKIPPED=$((SKIPPED + 1))
else
    echo "  → Merging $STEERING_OUT/$MERGE_TAX/alpha_*/"
    if $LOCAL; then
        (cd "$WORK_DIR" && $MERGE_CMD)
    else
        (cd "$WORK_DIR" && $MERGE_CMD)
    fi
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end step 12 (merge_steering)


# ═══════════════════════════════════════════════════════════════
# STEP 13 (plot_steering): Generate ECCV publication figures
# ═══════════════════════════════════════════════════════════════

if [[ "$STEP" == "plot_steering" ]]; then

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 13: Generate ECCV figures"
echo "═══════════════════════════════════════════════════════════"

PLOT_OUT_DIR="results/13-plots/${MODE_DIR}/${MODEL_NAME}${RUN_SUFFIX}"
MERGED_JSON="$PLOT_OUT_DIR/steering_merged.json"
HALLUC_SCORES_DIR_PLOT="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}${RUN_SUFFIX}"

if [[ ! -f "$WORK_DIR/$MERGED_JSON" ]]; then
    echo "  ERROR: $MERGED_JSON not found."
    echo "         Run step 12 first: bash run_pipeline.sh --step 12"
    exit 1
fi

PLOT_CMD="$PYTHON $PLOT_STEERING_SCRIPT \
    --merged_json $MERGED_JSON \
    --model_name $MODEL_NAME \
    --output_dir $PLOT_OUT_DIR \
    --format pdf"

# Add enrichment dir if available
if [[ -d "$WORK_DIR/$HALLUC_SCORES_DIR_PLOT" ]]; then
    PLOT_CMD="$PLOT_CMD --enrichment_dir $HALLUC_SCORES_DIR_PLOT"
fi

echo "  → Generating figures to $PLOT_OUT_DIR/"
if $LOCAL; then
    (cd "$WORK_DIR" && $PLOT_CMD)
else
    (cd "$WORK_DIR" && $PLOT_CMD)
fi
SUBMITTED=$((SUBMITTED + 1))

fi  # end step 13 (plot_steering)

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
if $LOCAL; then
    echo "  PIPELINE COMPLETE (local mode)"
else
    echo "  SUBMITTED: $SUBMITTED   SKIPPED: $SKIPPED (already done/active)"
    echo ""
    echo "  Monitor:  bjobs -q $QUEUE       (jobs: *_${SHORT_MODEL}*)"
    echo "  Logs:     ls ${LOG_DIR}/*/*.log"
fi
echo ""
echo "  Description file:   $DESC_FILE"
echo "  Classification dir: $OUTPUT_DIR/"
echo "  Ablation dir:       results/3-classify/${MODE_DIR}/$MODEL_NAME/ablation/"
echo "  Steering dir:       results/3-classify/${MODE_DIR}/$MODEL_NAME/ablation/steering/"
echo "  ECCV plots:         results/13-plots/${MODE_DIR}/$MODEL_NAME/"
echo "═══════════════════════════════════════════════════════════"
exit 0