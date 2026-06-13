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
#   bash run_pipeline.sh --step 14                    # (layer_plots) per-layer trend figure + LaTeX tables
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
MODEL_TYPE="llava-ov"    # model backend (llava-liuhaotian / llava-mistral / llava-llama3 / llava-ov / internvl / qwen2vl)
MODEL_PATH="liuhaotian/llava-v1.5-7b"  # HF Hub ID or local path to model weights
MODEL_NAME="llava-1.5-7b"  # must match --model default in $CLASSIFY_SCRIPT
MODEL_NAME_SET=false       # true if user passed --model-name explicitly
OUT_SUFFIX_USER=""        # user-provided suffix for output dirs
CLASSIFY_SCRIPT="code/neuron_modality_statistical.py"  # classification script for step 3 (classify)
ABLATION_SCRIPT="code/neuron_ablation_validate.py"      # ablation validation script for step 5 (ablation)
VIZ_SCRIPT="code/visualize_neuron_activations.py"        # Figure 3 visualization script for step visualize
PLOT_SCRIPT="code/plot_neuron_statistics.py"              # Figures 5/6/7 statistics charts for step 8 (statistics)
PLOT_DPI=200                                              # output DPI for all plot scripts (steps 8 and 14)
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

# ── Step 16/17 (weight merging) settings ──
MERGE_BASE_LLM_PATH=""           # base LLM path (auto-set per model)
MERGE_MATH_LLM_PATH=""           # math LLM path for text_inject (Method 1)
MERGE_DONOR_VLM_PATH=""          # donor VLM path for visual_transplant (Method 5)
MERGE_DONOR_LABEL_DIR=""         # PMBT label dir for donor VLM
MERGE_LAMBDA_SWEEP="0.05 0.1 0.15 0.2 0.25 0.3 0.5 0.7"  # lambda sweep: BRV range (0.05-0.3) + extreme (0.5-0.7)
MERGE_INCLUDE_UNIFORM=true       # also run uniform (no-mask) BRV baseline — on by default
MERGE_INCLUDE_MULTIMODAL=true    # also run text+multimodal mask variant — on by default
MERGE_INCLUDE_VISUAL_ONLY=true  # also run visual-only mask variant (negative control) — on by default
MERGE_INCLUDE_VISUAL_MULTI=true # also run visual+multimodal mask variant — on by default
MERGE_INCLUDE_RANDOM=true      # also run random mask (same count as text, sparsity control) — on by default
MERGE_INCLUDE_MULTIMODAL_ONLY=true # also run multimodal-only mask variant — on by default

# Step 18 (compose_merge): compose step-16 text injection + step-24 SRF, and step-23 SNRF + step-24 SRF
COMBINE_LAMBDA_16="0.1"          # which step-16 lambda to use for the combined model
COMBINE_LAMBDA_17="0.1"          # which step-17 lambda to use for the combined model

# Step 19 (evaluate): VLMEvalKit evaluation of merged models (steps 16, 17, 18)
EVAL_BENCHMARKS="MathVista_MINI MathVerse_MINI_Vision_Only MMStar POPE"  # benchmarks to run
EVAL_WHICH="baseline 16 text_multi visual_only visual_multi uniform multimodal_only random snrf snrf_random srf srf_random"   # which models to evaluate
MERGE_SAVE_MODEL=true            # save merged model weights to disk (always on by default)

# Step 21 (tune_lambda): BRV-style lambda tuning on MathVista
TUNE_LAMBDAS="0.05 0.1 0.15 0.2 0.25 0.3"    # lambda values to search over (must exist in step 16 sweep)
TUNE_MASKS="16 text_multi uniform visual_only visual_multi multimodal_only random"          # which masks to tune (16=text_inject, text_multi, visual_only, visual_multi, uniform, multimodal_only, random)

# ── Step 10 (halluc_score) settings ──
HALLUC_SCORE_SCRIPT="code/halluc_score_neurons.py"
MERGE_STEERING_SCRIPT="code/merge_steering_results.py"    # step 12: merge steering results across alphas
PLOT_STEERING_SCRIPT="code/plot_steering_results.py"      # step 13: ECCV publication figures
MERGE_SCRIPT="code/neuron_weight_merge.py"          # step 16/17: PMBT-guided weight merging
SNRF_SCRIPT="code/neuron_snrf_merge.py"             # step 23: SNRF + PMBT (Layer 1b)
SRF_SCRIPT="code/neuron_srf_merge.py"               # step 24: SRF + PMBT (Layer 1c)
SUMMARIZE_SCRIPT="code/summarize_eval_results.py"   # step 20: summarize benchmark results
HALLUC_SCORE_SHARDS=8             # GPUs for halluc scoring (max useful: 32, one per layer)
HALLUC_POPE_SAMPLES=""            # number of POPE samples (empty = all ~3000)
HALLUC_CONTRASTIVE="1"             # contrastive POPE filtering — on by default (needed for CETT-diff)
HALLUC_SKIP_ABLATION=""           # set to "1" to skip ablation loop, reuse existing scores
HALLUC_TRIVIAQA="1"                # TriviaQA dual-source scoring (Section 3.5) — on by default
HALLUC_ENCODER_PROJECTOR="1"       # encoder/projector neuron scoring — on by default
HALLUC_PLOT_ONLY=""               # set to "1" to only regenerate plots from existing results (no GPU needed)
HALLUC_MERGE_ONLY=""              # set to "1" to skip splits/preprocess/layers, only run merge (data must exist)

# ── Step 11 (steering) settings ──
STEERING_ALPHAS="0,0.25,0.5,0.75,1.5,2.0,2.5,3.0"   # comma-separated alpha values
STEERING_SHARDS=15                          # GPUs per alpha value (same sharding as step 5)
STEERING_TAXONOMY="pmbt"                    # ft | pmbt | both
HALLUC_SCORE_METHOD="dh"                    # dh (default) | combined | cett — which step 10 ranking to use
STEP5B_RANKING="all"                       # all | cett | selectivity | combined — neuron ranking for step 5b

# GPU memory tiers — escalate through tiers when jobs stay PEND
GPU_GMEM_TIERS=("20G")                # override with --gmem 40G,20G
RUN_SUFFIX=""                          # append to log names + output dirs (e.g. _gmem_80)
LAYER_LIST=""                          # comma-separated layers for step 3 (e.g. 8,9,13,14,31)
GMEM_WAIT=120                          # seconds to wait before escalating (override with --gmem-wait)
GPU_RES_BASE="rusage[mem=24576] order[-gpu_maxfactor]"
GPU_EXCLUSIVE=false                    # --exclusive: request exclusive GPU (mode=exclusive_process)

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
        --model-name) MODEL_NAME="$2"; MODEL_NAME_SET=true; shift 2 ;;
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
        --exclusive) GPU_EXCLUSIVE=true; shift ;;
        --gmem-wait) GMEM_WAIT="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --layer-list) LAYER_LIST="$2"; shift 2 ;;
        --ablation-method) ABLATION_METHOD="$2"; shift 2 ;;
        --ablation-curve) ABLATION_CURVE=true; shift ;;
        --no-ablation-curve) ABLATION_CURVE=false; shift ;;
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
        --merge-base-llm) MERGE_BASE_LLM_PATH="$2"; shift 2 ;;
        --merge-math-llm) MERGE_MATH_LLM_PATH="$2"; shift 2 ;;
        --merge-donor-vlm) MERGE_DONOR_VLM_PATH="$2"; shift 2 ;;
        --merge-donor-labels) MERGE_DONOR_LABEL_DIR="$2"; shift 2 ;;
        --merge-lambda) MERGE_LAMBDA_SWEEP="$2"; shift 2 ;;
        --merge-lambda16) MERGE_LAMBDA16="$2"; shift 2 ;;
        --merge-lambda17) MERGE_LAMBDA17="$2"; shift 2 ;;
        --merge-save-model) MERGE_SAVE_MODEL=true; shift ;;
        --no-save-model) MERGE_SAVE_MODEL=false; shift ;;
        --merge-uniform) MERGE_INCLUDE_UNIFORM=true; shift ;;
        --combine-lambda-16) COMBINE_LAMBDA_16="$2"; shift 2 ;;
        --combine-lambda-17) COMBINE_LAMBDA_17="$2"; shift 2 ;;
        --eval-benchmarks) EVAL_BENCHMARKS="$2"; shift 2 ;;
        --eval-which) EVAL_WHICH="$2"; shift 2 ;;
        --tune-lambdas) TUNE_LAMBDAS="$2"; shift 2 ;;
        --tune-masks) TUNE_MASKS="$2"; shift 2 ;;
        --halluc-pope-samples) HALLUC_POPE_SAMPLES="$2"; shift 2 ;;
        --halluc-contrastive) HALLUC_CONTRASTIVE="1"; shift 1 ;;
        --no-halluc-contrastive) HALLUC_CONTRASTIVE=""; shift 1 ;;
        --halluc-skip-ablation) HALLUC_SKIP_ABLATION="1"; shift 1 ;;
        --halluc-triviaqa) HALLUC_TRIVIAQA="1"; shift 1 ;;
        --no-halluc-triviaqa) HALLUC_TRIVIAQA=""; shift 1 ;;
        --no-encoder-projector) HALLUC_ENCODER_PROJECTOR=""; shift 1 ;;
        --halluc-plot-only) HALLUC_PLOT_ONLY="1"; shift 1 ;;
        --halluc-merge-only) HALLUC_MERGE_ONLY="1"; shift 1 ;;
        --halluc-score-method) HALLUC_SCORE_METHOD="$2"; shift 2 ;;
        --step5b-ranking) STEP5B_RANKING="$2"; shift 2 ;;
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
_VALID_MODELS="llava-liuhaotian llava-mistral llava-llama3 llava-ov llava-ov-si internvl qwen2vl idefics2 all"
_VALID_ALIASES="llava intern qwen"
if [[ "$MODEL_TYPE" != *","* ]]; then
    _found=false
    for _vm in $_VALID_MODELS; do
        [[ "$MODEL_TYPE" == "$_vm" ]] && _found=true && break
    done
    if ! $_found; then
        echo "ERROR: unknown --model-type '$MODEL_TYPE'"
        echo ""
        echo "  Valid names:    llava-liuhaotian  llava-mistral  llava-llama3  llava-ov  internvl  qwen2vl  all"
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
            echo "  Valid names:    llava-liuhaotian  llava-mistral  llava-llama3  llava-ov  internvl  qwen2vl"
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
ALL_MODELS=(llava-liuhaotian llava-mistral llava-llama3 llava-ov llava-ov-si internvl qwen2vl)
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
    elif [[ "$MODEL_TYPE" == "llava-mistral" ]]; then
        MODEL_PATH="llava-hf/llava-v1.6-mistral-7b-hf"               # LLaVA-Next-Mistral-7B (BRV reproduction)
    elif [[ "$MODEL_TYPE" == "llava-llama3" ]]; then
        MODEL_PATH="llava-hf/llama3-llava-next-8b-hf"                # LLaVA-Next-LLaMA3-8B (BRV main model)
    elif [[ "$MODEL_TYPE" == "idefics2" ]]; then
        MODEL_PATH="modern_vlms/pretrained/idefics2-8b"              # Idefics2-8B local snapshot
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
if ! $MODEL_NAME_SET && [[ "$MODEL_NAME" == "llava-1.5-7b" ]]; then       # not overridden by user
    if [[ "$MODEL_TYPE" == "internvl" ]]; then
        MODEL_NAME="internvl2.5-8b"                                       # InternVL2.5-8B output dir name
    elif [[ "$MODEL_TYPE" == "qwen2vl" ]]; then
        MODEL_NAME="qwen2.5-vl-7b"                                        # Qwen2.5-VL-7B output dir name
    elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
        MODEL_NAME="llava-onevision-7b"                                    # LLaVA-OneVision-7B output dir name
    elif [[ "$MODEL_TYPE" == "llava-mistral" ]]; then
        MODEL_NAME="llava-next-mistral-7b"                                 # LLaVA-Next-Mistral-7B output dir name
    elif [[ "$MODEL_TYPE" == "llava-llama3" ]]; then
        MODEL_NAME="llava-next-llama3-8b"                                  # LLaVA-Next-LLaMA3-8B output dir name
    elif [[ "$MODEL_TYPE" == "idefics2" ]]; then
        MODEL_NAME="idefics2-8b"                                           # Idefics2-8B output dir name
    fi
    # llava-liuhaotian keeps MODEL_NAME="llava-1.5-7b"
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
#   llava-liuhaotian:  32 × 11008 = 352256  (LLaMA-2-7B)
#   llava-mistral:  32 × 14336 = 458752  (Mistral-7B)
#   llava-llama3:  32 × 14336 = 458752  (LLaMA-3-8B)
#   llava-ov / llava-ov-si / qwen2vl:  28 × 18944 = 530432  (Qwen2-7B)
#   internvl:  32 × 14336 = 458752  (InternLM2-8B)
case "$MODEL_TYPE" in
    llava-liuhaotian) N_LLM_NEURONS=352256 ;;
    llava-mistral|llava-llama3) N_LLM_NEURONS=458752 ;;
    idefics2)         N_LLM_NEURONS=458752 ;;  # Mistral-7B backbone (32 × 14336)
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
    llava-mistral)     SHORT_MODEL="lm" ;;
    llava-llama3)      SHORT_MODEL="ll3" ;;
    llava-ov)         SHORT_MODEL="lo" ;;
    llava-ov-si)      SHORT_MODEL="losi" ;;
    internvl)         SHORT_MODEL="int" ;;
    qwen2vl)          SHORT_MODEL="qw" ;;
    idefics2)         SHORT_MODEL="idef" ;;
    *)                SHORT_MODEL="${MODEL_TYPE:0:4}" ;;
esac

# Differentiate same-MODEL_TYPE size variants by appending size to SHORT_MODEL.    # Line SZ1: comment
# Without this, e.g. Qwen2-VL-7B and Qwen2.5-VL-3B both map to SHORT_MODEL=qw      # Line SZ2: explanation
# and their LSF job names collide.                                                  # Line SZ3
if [[ "$MODEL_NAME" == *"3b"* || "$MODEL_NAME" == *"3B"* ]]; then                  # Line SZ4: detect 3B
    SHORT_MODEL="${SHORT_MODEL}3"                                                   # Line SZ5: append "3"
fi

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
elif [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" || "$MODEL_TYPE" == "llava-mistral" || "$MODEL_TYPE" == "llava-llama3" || "$MODEL_TYPE" == "idefics2" ]]; then
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
    5|ranked_ablation|5b)     STEP="ranked_ablation" ;;
    27|ablation_old|prune)    STEP="prune" ;;
    merge_ablation)           STEP="merge_ablation" ;;
    6|visualize|viz)          STEP="visualize" ;;
    7|attention|attn)         STEP="attn" ;;
    8|statistics|plot|stats)  STEP="plot" ;;
    14|layer_plots|layer_tables) STEP="layer_plots" ;;
    15|enrichment_plots)          STEP="enrichment_plots" ;;
    16|text_inject)              STEP="text_inject" ;;
    17|srf|srf_edit)                     STEP="srf" ;;
    18|compose_merge|combine_merges) STEP="compose_merge" ;;
    19|evaluate|benchmark_eval)      STEP="evaluate" ;;
    20|summarize|summary)             STEP="summarize" ;;
    21|tune_lambda)                     STEP="tune_lambda" ;;
    22|select_lambda)                    STEP="select_lambda" ;;
    23|snrf|snrf_merge)                  STEP="snrf" ;;
    24|visual_transplant)        STEP="visual_transplant" ;;
    29|weight_diff_rank|rank)    STEP="weight_diff_rank" ;;
    26|vit_analysis|vit)         STEP="vit_analysis" ;;
    25|compose_layer1)                   STEP="compose_layer1" ;;
    find_fig3|fig3_neurons)    STEP="find_fig3" ;;
    check_collisions|collisions)  STEP="check_collisions" ;;
    9|vtp)                    STEP="vtp" ;;
    10|halluc_score)           STEP="halluc_score" ;;
    11|steering|steer)         STEP="steering" ;;
    12|merge_steering)         STEP="merge_steering" ;;
    13|plot_steering|eccv)     STEP="plot_steering" ;;
    all|all_att)             ;;  # keep as-is
    *) echo "ERROR: unknown step '$STEP'"; echo "  Valid: 1-26, 27 (old ablation), 29, merge_ablation, find_fig3, check_collisions, all, all_att"; exit 1 ;;
esac

# ── Resolve --clean default: if "auto", infer from --step ───────────────
if [[ "$CLEAN_FROM" == "auto" ]]; then
    case "$STEP" in
        gd)        CLEAN_FROM=1 ;;
        merge_gd)  CLEAN_FROM=2 ;;
        cn)        CLEAN_FROM=3 ;;
        merge_nc)  CLEAN_FROM=4 ;;
        prune)     CLEAN_FROM=27 ;;
        ranked_ablation) CLEAN_FROM=5 ;;
        merge_ablation) CLEAN_FROM=5 ;;
        visualize) CLEAN_FROM=6 ;;
        attn)      CLEAN_FROM=7 ;;
        plot)      CLEAN_FROM=8 ;;
        layer_plots) CLEAN_FROM=14 ;;
        enrichment_plots) CLEAN_FROM=15 ;;
        text_inject)      CLEAN_FROM=16 ;;
        visual_transplant) CLEAN_FROM=24 ;;
        srf) CLEAN_FROM=17 ;;
        compose_merge)    CLEAN_FROM=18 ;;
        evaluate)         CLEAN_FROM=19 ;;
        vtp)       CLEAN_FROM=9 ;;
        halluc_score) CLEAN_FROM=10 ;;
        steering)  CLEAN_FROM=11 ;;
        merge_steering) CLEAN_FROM=12 ;;
        plot_steering) CLEAN_FROM=13 ;;
        vit_analysis) CLEAN_FROM=26 ;;
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
JN11="11_${SHORT_MODEL}${RUN_SUFFIX}"   # steering — RUN_SUFFIX prevents job-name collisions across runs
JN14="14_${SHORT_MODEL}"   # layer_plots (per-layer trend figure + LaTeX tables)

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
    local gpu_mode=""
    $GPU_EXCLUSIVE && gpu_mode=":mode=exclusive_process"

    # Submit at first tier
    bsub "${bsub_args[@]}" -gpu "num=1:gmem=${first_gmem}${gpu_mode}" -R "$GPU_RES_BASE" "$cmd"

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
                bsub "${bsub_args[@]}" -gpu "num=1:gmem=${next_gmem}${gpu_mode}" -R "$GPU_RES_BASE" "$cmd"
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

    # Step 14: layer_plots marker
    if (( CLEAN_FROM <= 14 && 14 <= CLEAN_TO )); then
        _LP_DIR="results/14-layer-plots/${MODE_DIR}"
        _LP_MARKER="$_LP_DIR/done.marker"
        if $WIPE; then
            [[ -d "$_LP_DIR" ]] && rm -rf "$_LP_DIR" && echo "    rm -rf $_LP_DIR"
        else
            [[ -f "$_LP_MARKER" ]] && rm -f "$_LP_MARKER" && echo "    rm $_LP_MARKER (marker)"
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
        _ST_DIR="results/11-steering/${MODE_DIR}/${MODEL_NAME}/${HALLUC_SCORE_METHOD}"
        if $WIPE; then
            [[ -d "$_ST_DIR" ]] && rm -rf "$_ST_DIR" && echo "    rm -rf $_ST_DIR"
        else
            for f in "$_ST_DIR"/*/alpha_*/ablation_summary.json; do
                [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
            done
        fi
    fi

    # Steps 12-13: merged steering + ECCV plots
    if (( CLEAN_FROM <= 13 && 12 <= CLEAN_TO )); then
        _PLOT_ST_DIR="results/13-plots/${MODE_DIR}/${MODEL_NAME}/${HALLUC_SCORE_METHOD}"
        if $WIPE; then
            [[ -d "$_PLOT_ST_DIR" ]] && rm -rf "$_PLOT_ST_DIR" && echo "    rm -rf $_PLOT_ST_DIR"
        else
            for f in "$_PLOT_ST_DIR"/steering_merged_*.json; do
                [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
            done
        fi
    fi

    # Step 14: layer plots
    if (( CLEAN_FROM <= 14 && 14 <= CLEAN_TO )); then
        _LP_MARKER="results/14-layer-plots/${MODE_DIR}/done.marker"
        rm -f "$WORK_DIR/$_LP_MARKER" && echo "    rm $_LP_MARKER"
        if $WIPE; then
            _LP_DIR="results/14-layer-plots/${MODE_DIR}"
            [[ -d "$WORK_DIR/$_LP_DIR" ]] && rm -rf "$WORK_DIR/$_LP_DIR" && echo "    rm -rf $_LP_DIR"
        fi
    fi

    # Step 15: enrichment plots
    if (( CLEAN_FROM <= 15 && 15 <= CLEAN_TO )); then
        _EP_MARKER="results/15-enrichment-plots/full/done.marker"
        rm -f "$WORK_DIR/$_EP_MARKER" && echo "    rm $_EP_MARKER"
        if $WIPE; then
            _EP_DIR="results/15-enrichment-plots/full"
            [[ -d "$WORK_DIR/$_EP_DIR" ]] && rm -rf "$WORK_DIR/$_EP_DIR" && echo "    rm -rf $_EP_DIR"
        fi
    fi

    # Logs: only delete with --wipe
    if $WIPE; then
        for s in $(seq "$CLEAN_FROM" "$CLEAN_TO"); do
            STEP_LOG_NAMES=("1-describe" "2-merge_descriptions" "3-classify" "4-merge_classifications" "5-ablation_validate" "6-activation_maps" "7-attention_maps" "8-statistics" "9-vtp" "10-halluc_score" "11-steering" "12-merge_steering" "13-plot_steering" "14-layer-plots" "15-enrichment-plots")
            if (( s >= 1 && s <= ${#STEP_LOG_NAMES[@]} )); then
                SDIR="$LOG_DIR/${STEP_LOG_NAMES[$((s-1))]}"
                if [[ -d "$SDIR" ]]; then
                    rm -rf "$SDIR" && echo "    rm -rf $SDIR"
                fi
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
# STEP 5 (ranked_ablation): Dose-response taxonomy validation
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "ranked_ablation" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 5: Ranked Fraction Ablation (dose-response validation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STEP5B_SCRIPT="code/ranked_fraction_ablation.py"
STEP5B_LOG="${LOG_DIR}/5b-ranked-ablation"
mkdir -p "$WORK_DIR/$STEP5B_LOG"

LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"
LABEL_DIR="${LABELS_BASE}/${MODEL_NAME}/llm_permutation"

# Fractions and conditions
STEP5B_FRACTIONS="0.01,0.05,0.10,0.20,0.50,1.00"
STEP5B_CONDITIONS="visual,text,multimodal,unknown,random"

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))

# Expand "all" to the three ranking types
if [[ "$STEP5B_RANKING" == "all" ]]; then
    _RANKING_LIST="selectivity cett combined"
else
    _RANKING_LIST="$STEP5B_RANKING"
fi

echo "  Model:      $MODEL_NAME ($MODEL_TYPE)"
echo "  Rankings:   $_RANKING_LIST"
echo "  Fractions:  $STEP5B_FRACTIONS"
echo "  Conditions: $STEP5B_CONDITIONS"
echo "  GPU gmem:   ${GPU_GMEM_TIERS[0]}"

for _RANK_TYPE in $_RANKING_LIST; do

echo ""
echo "  ════════════════════════════════════════════════════════"
echo "  Ranking: $_RANK_TYPE"
echo "  ════════════════════════════════════════════════════════"

STEP5B_OUT="results/5b-ranked-ablation/${MODE_DIR}/${MODEL_NAME}/${_RANK_TYPE}"
mkdir -p "$WORK_DIR/$STEP5B_OUT"

# ── Symlink cached data from step 10 if available ──
STEP10_DIR="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}"
if [[ -d "$WORK_DIR/$STEP10_DIR" ]]; then
    for _f in contrastive_pope.jsonl contrastive_stats.json \
              cett_diff_scores.json cett_mean_scores.json \
              contrastive_triviaqa.jsonl cett_diff_scores_tqa.json \
              cett_mean_scores_tqa.json; do
        if [[ -f "$WORK_DIR/$STEP10_DIR/$_f" && ! -f "$WORK_DIR/$STEP5B_OUT/$_f" ]]; then
            ln -sf "$(realpath "$WORK_DIR/$STEP10_DIR/$_f")" "$WORK_DIR/$STEP5B_OUT/$_f"
        fi
    done
fi

# Common args for this ranking type
STEP5B_COMMON="--model_type $MODEL_TYPE --model_path $MODEL_PATH \
    --model_name $MODEL_NAME --n_layers $N_LAYERS \
    --label_dir $LABEL_DIR --taxonomy pmbt \
    --pope_path $POPE_PATH --pope_img_dir $POPE_IMG_DIR \
    --ranking $_RANK_TYPE \
    --output_dir $STEP5B_OUT"

# Add TriviaQA if supported by model type
if [[ "$MODEL_TYPE" != "llava-mistral" && "$MODEL_TYPE" != "llava-llama3" \
      && "$MODEL_TYPE" != "llava-liuhaotian" && "$MODEL_TYPE" != "llava" ]]; then
    STEP5B_COMMON="$STEP5B_COMMON --triviaqa_path $TRIVIAQA_PATH --triviaqa_num $TRIVIAQA_NUM"
fi

# ── Special case: random_trials (phases 3+4 instead of 0+1+2) ──
if [[ "$_RANK_TYPE" == "random_trials" ]]; then
    _RT_N_TRIALS=30
    _RT_FRACTIONS="0.10,0.20,0.30"
    echo "  Random-sample taxonomy validation"
    echo "  Trials:     $_RT_N_TRIALS per condition"
    echo "  Fractions:  $_RT_FRACTIONS"

    # InternVL needs 80G
    if [[ "$MODEL_TYPE" == "internvl" ]]; then
        _RT_GMEM="80G"
    else
        _RT_GMEM="${GPU_GMEM_TIERS[0]}"
    fi

    # ── Phase 0: Generate contrastive data if missing ──
    _RT_P0_DEP=""
    _CONTRASTIVE="$WORK_DIR/$STEP5B_OUT/contrastive_pope.jsonl"
    if [[ -f "$_CONTRASTIVE" ]]; then
        echo "  [skip] Contrastive POPE already exists"
    else
        JOB_RT_P0="5rt_${SHORT_MODEL}_p0"
        echo "  [P0] $JOB_RT_P0 → generate contrastive data"
        if $LOCAL; then
            (cd "$WORK_DIR" && $PYTHON $STEP5B_SCRIPT --phase 0 $STEP5B_COMMON) \
                2>&1 | tee "$WORK_DIR/$STEP5B_LOG/${JOB_RT_P0}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_RT_P0" \
                -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP5B_LOG/${JOB_RT_P0}.log" \
                -eo "$WORK_DIR/$STEP5B_LOG/${JOB_RT_P0}.err" \
                "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 0 $STEP5B_COMMON"
            SUBMITTED=$((SUBMITTED + 1))
            _RT_P0_DEP="done($JOB_RT_P0)"
        fi
    fi

    IFS=',' read -ra _RT_FRACS <<< "$_RT_FRACTIONS"
    IFS=',' read -ra _RT_CONDS <<< "$STEP5B_CONDITIONS"

    RT_JOBS=()
    for _COND in "${_RT_CONDS[@]}"; do
        for _FRAC in "${_RT_FRACS[@]}"; do
            _FRAC_FNAME=$(echo "$_FRAC" | sed 's/\./p/')
            for (( _T=0; _T<$_RT_N_TRIALS; _T++ )); do
                _T_FMT=$(printf "%03d" "$_T")
                _RESULT_FILE="$WORK_DIR/$STEP5B_OUT/random_trial_${_COND}_${_FRAC_FNAME}_t${_T_FMT}.json"
                JOB_NAME="5rt_${SHORT_MODEL}_${_COND:0:3}_${_FRAC_FNAME}_t${_T_FMT}"

                if [[ -f "$_RESULT_FILE" ]]; then
                    continue
                fi

                echo "  → $JOB_NAME"
                if $LOCAL; then
                    (cd "$WORK_DIR" && $PYTHON $STEP5B_SCRIPT --phase 3 \
                        --condition "$_COND" --fraction "$_FRAC" --trial "$_T" \
                        $STEP5B_COMMON) \
                        2>&1 | tee "$WORK_DIR/$STEP5B_LOG/${JOB_NAME}.log"
                else
                    _RT_DEP_FLAG=""
                    [[ -n "$_RT_P0_DEP" ]] && _RT_DEP_FLAG="-w $_RT_P0_DEP"
                    bsub -q "$QUEUE" -J "$JOB_NAME" \
                        $_RT_DEP_FLAG \
                        -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                        -R "$GPU_RES_BASE" \
                        -oo "$WORK_DIR/$STEP5B_LOG/${JOB_NAME}.log" \
                        -eo "$WORK_DIR/$STEP5B_LOG/${JOB_NAME}.err" \
                        "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 3 \
                            --condition $_COND --fraction $_FRAC --trial $_T \
                            $STEP5B_COMMON"
                    RT_JOBS+=("$JOB_NAME")
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
        done
    done

    # ── Phase 4: Merge trials + statistics (depends on all phase 3 jobs) ──
    if [[ ${#RT_JOBS[@]} -gt 0 ]]; then
        JOB_MERGE="5rt_${SHORT_MODEL}_merge"
        DEP_STR=""
        for _jn in "${RT_JOBS[@]}"; do
            [[ -n "$DEP_STR" ]] && DEP_STR="$DEP_STR && "
            DEP_STR="${DEP_STR}done($_jn)"
        done

        echo ""
        echo "  [M] $JOB_MERGE → merge ${#RT_JOBS[@]} trials + statistics"
        bsub -q "$QUEUE" -J "$JOB_MERGE" \
            -w "$DEP_STR" \
            -R "rusage[mem=4096]" \
            -oo "$WORK_DIR/$STEP5B_LOG/${JOB_MERGE}.log" \
            -eo "$WORK_DIR/$STEP5B_LOG/${JOB_MERGE}.err" \
            "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 4 $STEP5B_COMMON"
        SUBMITTED=$((SUBMITTED + 1))
    fi

    echo "  Random trials: ${#RT_JOBS[@]} jobs submitted"
    continue  # skip phases 0-1-2 for this ranking type
fi

# ── Phase 0: Compute ranking (if not already done) ──
P0_DEP=""  # set to "done(job_name)" if phase 0 submitted
if [[ -f "$WORK_DIR/$STEP5B_OUT/neuron_ranking.json" ]]; then
    echo "  [skip] Phase 0: neuron_ranking.json already exists for $_RANK_TYPE"
else
    JOB_P0="5b_${SHORT_MODEL}_p0_${_RANK_TYPE}"
    if [[ "$_RANK_TYPE" == "selectivity" ]]; then
        echo "  [P0] $JOB_P0 → compute $_RANK_TYPE ranking (CPU only)"
        if $LOCAL; then
            (cd "$WORK_DIR" && $PYTHON $STEP5B_SCRIPT --phase 0 $STEP5B_COMMON) \
                2>&1 | tee "$WORK_DIR/$STEP5B_LOG/${JOB_P0}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_P0" \
                -R "rusage[mem=8192]" \
                -oo "$WORK_DIR/$STEP5B_LOG/${JOB_P0}.log" \
                -eo "$WORK_DIR/$STEP5B_LOG/${JOB_P0}.err" \
                "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 0 $STEP5B_COMMON"
            SUBMITTED=$((SUBMITTED + 1))
            P0_DEP="done($JOB_P0)"
        fi
    else
        echo "  [P0] $JOB_P0 → compute $_RANK_TYPE ranking (GPU for CETT)"
        if $LOCAL; then
            (cd "$WORK_DIR" && $PYTHON $STEP5B_SCRIPT --phase 0 $STEP5B_COMMON) \
                2>&1 | tee "$WORK_DIR/$STEP5B_LOG/${JOB_P0}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_P0" \
                -gpu "num=1:gmem=80G" \
                -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP5B_LOG/${JOB_P0}.log" \
                -eo "$WORK_DIR/$STEP5B_LOG/${JOB_P0}.err" \
                "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 0 $STEP5B_COMMON"
            SUBMITTED=$((SUBMITTED + 1))
            P0_DEP="done($JOB_P0)"
        fi
    fi
fi

# ── Phase 1: Submit evaluation jobs (depend on phase 0 if submitted) ──
echo ""
echo "  ── Phase 1 [$_RANK_TYPE]: Submitting evaluation jobs ──"
[[ -n "$P0_DEP" ]] && echo "  (pending on $P0_DEP)"

# InternVL needs 80G for all GPU jobs
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    _P1_GMEM="80G"
else
    _P1_GMEM="${GPU_GMEM_TIERS[0]}"
fi
echo "  Phase 1 GPU gmem: $_P1_GMEM"

IFS=',' read -ra _FRACS <<< "$STEP5B_FRACTIONS"
IFS=',' read -ra _CONDS <<< "$STEP5B_CONDITIONS"

PHASE1_JOBS=()

for _COND in "${_CONDS[@]}"; do
    for _FRAC in "${_FRACS[@]}"; do
        _FRAC_FNAME=$(echo "$_FRAC" | sed 's/\./p/')
        _RESULT_FILE="$WORK_DIR/$STEP5B_OUT/ablation_result_${_COND}_${_FRAC_FNAME}.json"
        JOB_NAME="5b_${SHORT_MODEL}_${_RANK_TYPE:0:3}_${_COND}_${_FRAC_FNAME}"

        if [[ -f "$_RESULT_FILE" ]]; then
            echo "  [skip] $_COND @ $_FRAC — result exists"
            continue
        fi

        echo "  → $JOB_NAME ($_COND top $_FRAC)"
        if $LOCAL; then
            (cd "$WORK_DIR" && $PYTHON $STEP5B_SCRIPT --phase 1 \
                --condition "$_COND" --fraction "$_FRAC" $STEP5B_COMMON) \
                2>&1 | tee "$WORK_DIR/$STEP5B_LOG/${JOB_NAME}.log"
        else
            _P1_DEP_FLAG=""
            [[ -n "$P0_DEP" ]] && _P1_DEP_FLAG="-w $P0_DEP"
            bsub -q "$QUEUE" -J "$JOB_NAME" \
                $_P1_DEP_FLAG \
                -gpu "num=1:gmem=$_P1_GMEM" \
                -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP5B_LOG/${JOB_NAME}.log" \
                -eo "$WORK_DIR/$STEP5B_LOG/${JOB_NAME}.err" \
                "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 1 \
                    --condition $_COND --fraction $_FRAC $STEP5B_COMMON"
            PHASE1_JOBS+=("$JOB_NAME")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done


# ── Phase 2: Merge (with dependency on Phase 1 jobs) ──
if [[ ${#PHASE1_JOBS[@]} -gt 0 ]]; then
    JOB_MERGE="5b_${SHORT_MODEL}_${_RANK_TYPE:0:3}_merge"
    DEP_STR=""
    for _jn in "${PHASE1_JOBS[@]}"; do
        [[ -n "$DEP_STR" ]] && DEP_STR="$DEP_STR && "
        DEP_STR="${DEP_STR}done($_jn)"
    done

    echo ""
    echo "  [M] $JOB_MERGE → merge results (depends on ${#PHASE1_JOBS[@]} Phase 1 jobs)"
    bsub -q "$QUEUE" -J "$JOB_MERGE" \
        -w "$DEP_STR" \
        -R "rusage[mem=4096]" \
        -oo "$WORK_DIR/$STEP5B_LOG/${JOB_MERGE}.log" \
        -eo "$WORK_DIR/$STEP5B_LOG/${JOB_MERGE}.err" \
        "cd $WORK_DIR && $PYTHON $STEP5B_SCRIPT --phase 2 $STEP5B_COMMON"
    SUBMITTED=$((SUBMITTED + 1))
fi

echo "  Phase 1 [$_RANK_TYPE]: ${#PHASE1_JOBS[@]} jobs submitted"

done  # end ranking type loop

fi  # end step 5 (ranked_ablation)



# ═══════════════════════════════════════════════════════════════
# STEP 27 (old ablation): Full-category ablation validation of neuron taxonomy
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "prune" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 27: Full-category ablation validation (legacy step 5)"
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

if [[ -n "$TRIVIAQA_PATH" ]] && [[ "$MODEL_TYPE" != "llava-mistral" && "$MODEL_TYPE" != "llava-llama3" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
    BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --triviaqa_path $TRIVIAQA_PATH --triviaqa_num_questions $TRIVIAQA_NUM"
fi

if [[ -n "$MMLU_DIR" ]] && [[ "$MODEL_TYPE" != "llava-mistral" && "$MODEL_TYPE" != "llava-llama3" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
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
if [[ "$MODEL_TYPE" == "llava-mistral" ]]; then
    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llava-v1.6-mistral-7b-hf"
elif [[ "$MODEL_TYPE" == "llava-llama3" ]]; then
    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llama3-llava-next-8b-hf"
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
    mtype="${NAME_TO_TYPE[$mname]:-llava-mistral}"

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
# STEP 14 (layer_plots): per-layer trend figure + LaTeX tables
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "layer_plots" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 14: Per-layer trend figure + LaTeX tables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/14-layer-plots"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

LP_OUT_DIR="results/14-layer-plots/${MODE_DIR}"
LP_MARKER="$LP_OUT_DIR/done.marker"

JOB_NAME="${JN14}"
if [[ -f "$LP_MARKER" ]] || is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already done or active"
    SKIPPED=$((SKIPPED + 1))
else

mkdir -p "$LP_OUT_DIR"

# ── Write generate_layer_tables_plots.py inline ───────────────────────────
LP_SCRIPT="$WORK_DIR/${STEP_LOG_DIR}/generate_layer_tables_plots.py"
cat > "$LP_SCRIPT" << 'LPEOF'
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()
parser.add_argument("--classify-dir", default="results/3-classify/full")
parser.add_argument("--output-dir",   default="results/14-layer-plots/full")
parser.add_argument("--dpi",          type=int, default=200)
args = parser.parse_args()

OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_REGISTRY = [
    ("llava-1.5-7b",       "LLaVA-1.5-7b"),
    ("llava-onevision-7b", "LLaVA-OV-7B"),
    ("qwen2.5-vl-7b",      "Qwen2.5-VL-7B"),
    ("internvl2.5-8b",     "InternVL2.5-8B"),
]

def find_models(classify_dir):
    # Hardcoded exact paths based on actual cluster directory structure.
    B = classify_dir  # results/3-classify
    EXACT = [
        ("llava-1.5-7b",       "LLaVA-1.5-7b",
         f"{B}/full_empty/llava-onevision-7b/llava-1.5-7b/llm_fixed_threshold/classification_stats_all.json",
         f"{B}/full_empty/llava-onevision-7b/llava-1.5-7b/llm_permutation/permutation_stats_all.json"),
        ("llava-onevision-7b", "LLaVA-OV-7B",
         f"{B}/full_empty/llava-onevision-7b/llava-onevision-7b/llm_fixed_threshold/classification_stats_all.json",
         f"{B}/full_empty/llava-onevision-7b/llava-onevision-7b/llm_permutation/permutation_stats_all.json"),
        ("qwen2.5-vl-7b",      "Qwen2.5-VL-7B",
         f"{B}/full_empty/qwen2.5-vl-7b/llm_fixed_threshold/classification_stats_all.json",
         f"{B}/full_empty/qwen2.5-vl-7b/llm_permutation/permutation_stats_all.json"),
        ("internvl2.5-8b",     "InternVL2.5-8B",
         f"{B}/full_empty/internvl2.5-8b/llm_fixed_threshold/classification_stats_all.json",
         f"{B}/full_empty/internvl2.5-8b/llm_permutation/permutation_stats_all.json"),
    ]
    found = []
    for model_id, label, ft, pmbt in EXACT:
        if os.path.isfile(ft) and os.path.isfile(pmbt):
            found.append((model_id, label, ft, pmbt))
            print(f"  ✓ {model_id}  FT={ft}")
        else:
            print(f"  ✗ {model_id} — missing FT={ft}", file=sys.stderr)
    if not found:
        print("ERROR: no models found", file=sys.stderr); sys.exit(1)
    return found

MODELS = find_models(args.classify_dir)

TYPES  = ["visual","text","multimodal","unknown"]
LABELS = ["Visual","Text","Multimodal","Unknown"]
COLORS = ["#d62728","#1f77b4","#2ca02c","#7f7f7f"]
LSTYLES = ["-","-","-","--"]

def load_per_layer(filepath):
    with open(filepath) as f: d = json.load(f)
    return {int(k): v for k, v in d["per_layer_stats"].items()}

def to_pct(per_layer):
    out = {}
    for layer, counts in sorted(per_layer.items()):
        total = sum(counts[t] for t in TYPES)
        out[layer] = {t: 100.0*counts[t]/total if total>0 else 0.0 for t in TYPES}
    return out

# ── Trend plots ─────────────────────────────────────────────────────────────
n = len(MODELS)
fig, axes = plt.subplots(n, 2, figsize=(12, 3.5*n), sharex=False, sharey=True,
                         constrained_layout=True)
if n == 1: axes = axes[np.newaxis, :]

for row, (mid, mlabel, ft_path, pmbt_path) in enumerate(MODELS):
    for col, (fpath, mtitle) in enumerate([(ft_path,"FT"),(pmbt_path,"PMBT")]):
        ax = axes[row, col]
        pct = to_pct(load_per_layer(fpath))
        layers = sorted(pct.keys())
        x = np.array(layers)
        for t, lbl, color, ls in zip(TYPES, LABELS, COLORS, LSTYLES):
            ax.plot(x, [pct[l][t] for l in layers], color=color, linestyle=ls,
                    linewidth=1.6, label=lbl, marker="o", markersize=2.5, markeredgewidth=0)
        ax.set_xlim(layers[0]-.5, layers[-1]+.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(layers[::4])
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3)
        if row == 0: ax.set_title(mtitle, fontsize=13, fontweight="bold", pad=6)
        if col == 0: ax.set_ylabel(mlabel, fontsize=10, fontweight="bold")
        if row == n-1: ax.set_xlabel("Layer", fontsize=9)

handles, lbls = axes[0,0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5,-0.02),
           ncol=4, fontsize=10, frameon=True, edgecolor="grey")
fig.suptitle("Per-Layer Neuron Type Proportions — FT vs PMBT",
             fontsize=13, fontweight="bold", y=1.01)
for ext in ["pdf","png"]:
    out = os.path.join(OUT_DIR, f"fig_layer_trends.{ext}")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ── LaTeX tables ─────────────────────────────────────────────────────────────
def fmt(v): return f"{v:.1f}"

def make_longtable(model_label, pl_ft, pl_pmbt, n_neurons, n_layers):
    pct_ft   = to_pct(pl_ft)
    pct_pmbt = to_pct(pl_pmbt)
    all_layers = sorted(set(pct_ft)|set(pct_pmbt))
    slug = model_label.lower().replace(" ","-").replace(".","")
    hdr = (r"  \toprule" + "\n"
           r"  & \multicolumn{4}{c}{\textbf{FT}} && \multicolumn{4}{c}{\textbf{PMBT}} \\" + "\n"
           r"  \cmidrule(lr){2-5}\cmidrule(lr){7-10}" + "\n"
           r"  \textbf{Layer} & Vis. & Text & Multi. & Unk. && Vis. & Text & Multi. & Unk. \\" + "\n"
           r"  \midrule")
    lines = [r"\begin{longtable}{@{}r rrrr c rrrr@{}}",
             r"  \caption{Per-layer neuron type proportions (\%) for " + model_label +
             f" ({n_neurons:,} neurons, {n_layers} layers)." + r"}",
             r"  \label{tab:layer-" + slug + r"} \\",
             hdr, r"  \endfirsthead", hdr, r"  \endhead",
             r"  \midrule \multicolumn{10}{r}{\textit{continued on next page}} \\",
             r"  \endfoot", r"  \bottomrule", r"  \endlastfoot"]
    for l in all_layers:
        ft   = pct_ft.get(l,   {t:0.0 for t in TYPES})
        pmbt = pct_pmbt.get(l, {t:0.0 for t in TYPES})
        lines.append(f"  {l} & {fmt(ft['visual'])} & {fmt(ft['text'])} & "
                     f"{fmt(ft['multimodal'])} & {fmt(ft['unknown'])} && "
                     f"{fmt(pmbt['visual'])} & {fmt(pmbt['text'])} & "
                     f"{fmt(pmbt['multimodal'])} & {fmt(pmbt['unknown'])} \\\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)

tex_out = os.path.join(OUT_DIR, "supp_layer_tables.tex")
with open(tex_out, "w") as f:
    f.write("% Auto-generated by generate_layer_tables_plots.py\n")
    f.write("% Requires: \\usepackage{longtable, booktabs}\n\n")
    for mid, mlabel, ft_path, pmbt_path in MODELS:
        pl_ft   = load_per_layer(ft_path)
        pl_pmbt = load_per_layer(pmbt_path)
        with open(ft_path) as fj: meta = json.load(fj)
        stats = meta.get("stats", {})
        n_neurons = sum(stats.values()) if isinstance(stats, dict) else 0
        f.write(f"% ── {mlabel} ──\n{make_longtable(mlabel, pl_ft, pl_pmbt, n_neurons, len(pl_ft))}\n\n")
print(f"Saved: {tex_out}")
print("Done.")
LPEOF

# ── Write generate_specialist_figure.py inline ────────────────────────────
SP_SCRIPT="$WORK_DIR/${STEP_LOG_DIR}/generate_specialist_figure.py"
cat > "$SP_SCRIPT" << 'SPEOF'
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()
parser.add_argument("--classify-dir", default="results/3-classify/full")
parser.add_argument("--output-dir",   default="results/14-layer-plots/full")
parser.add_argument("--dpi",          type=int, default=200)
parser.add_argument("--pct",          type=float, default=3.0)
args = parser.parse_args()

OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_REGISTRY = [
    ("llava-1.5-7b",       "LLaVA-1.5-7b"),
    ("llava-onevision-7b", "LLaVA-OV-7B"),
    ("qwen2.5-vl-7b",      "Qwen2.5-VL-7B"),
    ("internvl2.5-8b",     "InternVL2.5-8B"),
]
TYPES   = ["visual","text","multimodal"]
LABELS  = ["Visual","Text","Multimodal"]
COLORS  = ["#d62728","#1f77b4","#2ca02c"]
MARKERS = ["o","s","^"]

def find_models(classify_dir):
    """
    Hardcoded exact paths based on actual cluster directory structure.
    classify_dir is used as the root (results/3-classify).
    """
    BASE = classify_dir  # results/3-classify
    EXACT = [
        ("llava-1.5-7b",       "LLaVA-1.5-7b",
         f"{BASE}/full_empty/llava-onevision-7b/llava-1.5-7b/llm_fixed_threshold/classification_stats_all.json",
         f"{BASE}/full_empty/llava-onevision-7b/llava-1.5-7b/llm_permutation/permutation_stats_all.json"),
        ("llava-onevision-7b", "LLaVA-OV-7B",
         f"{BASE}/full_empty/llava-onevision-7b/llava-onevision-7b/llm_fixed_threshold/classification_stats_all.json",
         f"{BASE}/full_empty/llava-onevision-7b/llava-onevision-7b/llm_permutation/permutation_stats_all.json"),
        ("qwen2.5-vl-7b",      "Qwen2.5-VL-7B",
         f"{BASE}/full_empty/qwen2.5-vl-7b/llm_fixed_threshold/classification_stats_all.json",
         f"{BASE}/full_empty/qwen2.5-vl-7b/llm_permutation/permutation_stats_all.json"),
        ("internvl2.5-8b",     "InternVL2.5-8B",
         f"{BASE}/full_empty/internvl2.5-8b/llm_fixed_threshold/classification_stats_all.json",
         f"{BASE}/full_empty/internvl2.5-8b/llm_permutation/permutation_stats_all.json"),
    ]
    found = []
    for model_id, label, ft, pmbt in EXACT:
        if os.path.isfile(ft) and os.path.isfile(pmbt):
            found.append((model_id, label, ft, pmbt))
            print(f"  ✓ {model_id}")
        else:
            print(f"  ✗ {model_id} — missing:\n    FT:   {ft}\n    PMBT: {pmbt}", file=sys.stderr)
    if not found:
        print("ERROR: no models found", file=sys.stderr); sys.exit(1)
    return found

def load_stats(filepath):
    with open(filepath) as f: d = json.load(f)
    per_layer = {int(k): v for k, v in d["per_layer_stats"].items()}
    total = sum(d["stats"].values())
    return per_layer, total

MODELS = find_models(args.classify_dir)
n = len(MODELS)
fig, axes = plt.subplots(n, 2, figsize=(12, 3.5*n), sharex=False, sharey=False,
                         constrained_layout=True)
if n == 1: axes = axes[np.newaxis, :]

for row, (mid, mlabel, ft_path, pmbt_path) in enumerate(MODELS):
    pl_ft,   total_ft   = load_stats(ft_path)
    pl_pmbt, total_pmbt = load_stats(pmbt_path)
    n_layers = len(pl_ft)
    baseline = (args.pct / 100.0 * total_ft) / n_layers

    for col, (pl, mtitle) in enumerate([(pl_ft,"FT"),(pl_pmbt,"PMBT")]):
        ax = axes[row, col]
        layers = sorted(pl.keys())
        for t, lbl, color, mrk in zip(TYPES, LABELS, COLORS, MARKERS):
            counts = np.array([pl[l][t] for l in layers])
            ax.plot(layers, counts, color=color, marker=mrk,
                    markersize=3.5, linewidth=1.5, label=lbl)
        ax.axhline(baseline, color="black", linestyle=":", linewidth=0.9,
                   label=f"{args.pct:.0f}%/layer ≈ {int(baseline):,}")
        ax.set_xlim(layers[0]-.5, layers[-1]+.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(layers[::4])
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3)
        if row == 0: ax.set_title(mtitle, fontsize=13, fontweight="bold", pad=6)
        if col == 0: ax.set_ylabel(mlabel, fontsize=10, fontweight="bold")
        if row == n-1: ax.set_xlabel("Layer", fontsize=9)

handles, lbls = axes[0,0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5,-0.02),
           ncol=4, fontsize=10, frameon=True, edgecolor="grey")
fig.suptitle(f"Per-Layer Specialist Neuron Counts — FT vs PMBT  "
             f"(dotted = {args.pct:.0f}%/layer baseline)",
             fontsize=13, fontweight="bold", y=1.01)
for ext in ["pdf","png"]:
    out = os.path.join(OUT_DIR, f"fig_specialist_layers.{ext}")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
print("Done.")
SPEOF

# ── Write generate_confidence_figures.py inline ───────────────────────────
CF_SCRIPT="$WORK_DIR/${STEP_LOG_DIR}/generate_confidence_figures.py"
cat > "$CF_SCRIPT" << 'CFEOF'
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--classify-dir", required=True)
parser.add_argument("--output-dir",   default="results/14-layer-plots/full")
parser.add_argument("--dpi",          type=int, default=200)
parser.add_argument("--max-neurons",  type=int, default=None)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

MODEL_REGISTRY = [
    ("llava-1.5-7b",       "LLaVA-1.5-7b"),
    ("llava-onevision-7b", "LLaVA-OV-7B"),
    ("qwen2.5-vl-7b",      "Qwen2.5-VL-7B"),
    ("internvl2.5-8b",     "InternVL2.5-8B"),
]
TYPE_COLORS = {"visual":"#d62728","text":"#1f77b4","multimodal":"#2ca02c","unknown":"#7f7f7f"}
BINS = 50

def iter_layer_dirs(base_dir):
    """Yield (layer_int, full_path) for each per-layer subdir."""
    import re
    if not os.path.isdir(base_dir): return
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full): continue
        # Match patterns like:
        #   model.language_model.layers.11.mlp.act_fn
        #   layer_11  /  11
        m = re.search(r'layers?[._](\d+)', name)
        if m:
            yield int(m.group(1)), full
            continue
        # Fallback: any trailing integer in the name
        m2 = re.search(r'(\d+)', name)
        if m2:
            yield int(m2.group(1)), full

def load_ft_neurons(ft_base, max_n=None):
    neurons = []
    for _, ld in iter_layer_dirs(ft_base):
        fpath = os.path.join(ld, "neuron_labels.json")
        if not os.path.isfile(fpath): continue
        with open(fpath) as f: data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        for n in items:
            neurons.append({"label":n.get("label","unknown"),
                            "pv":float(n.get("pv",0)),"pt":float(n.get("pt",0)),
                            "pm":float(n.get("pm",0)),"pu":float(n.get("pu",0))})
            if max_n and len(neurons) >= max_n: return neurons
    return neurons

def load_pmbt_neurons(pmbt_base, max_n=None):
    neurons = []
    for _, ld in iter_layer_dirs(pmbt_base):
        fpath = os.path.join(ld, "neuron_labels_permutation.json")
        if not os.path.isfile(fpath): continue
        with open(fpath) as f: data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        for n in items:
            neurons.append({"label":n.get("label","unknown"),
                            "p_value":float(n.get("p_value",1)),
                            "observed_rate_diff":float(n.get("observed_rate_diff",0))})
            if max_n and len(neurons) >= max_n: return neurons
    return neurons

print("Loading neuron label files...")
model_data = []
BASE = args.classify_dir  # results/3-classify
EXACT_DIRS = [
    ("llava-1.5-7b",       "LLaVA-1.5-7b",
     f"{BASE}/full_empty/llava-onevision-7b/llava-1.5-7b/llm_fixed_threshold",
     f"{BASE}/full_empty/llava-onevision-7b/llava-1.5-7b/llm_permutation"),
    ("llava-onevision-7b", "LLaVA-OV-7B",
     f"{BASE}/full_empty/llava-onevision-7b/llava-onevision-7b/llm_fixed_threshold",
     f"{BASE}/full_empty/llava-onevision-7b/llava-onevision-7b/llm_permutation"),
    ("qwen2.5-vl-7b",      "Qwen2.5-VL-7B",
     f"{BASE}/full_empty/qwen2.5-vl-7b/llm_fixed_threshold",
     f"{BASE}/full_empty/qwen2.5-vl-7b/llm_permutation"),
    ("internvl2.5-8b",     "InternVL2.5-8B",
     f"{BASE}/full_empty/internvl2.5-8b/llm_fixed_threshold",
     f"{BASE}/full_empty/internvl2.5-8b/llm_permutation"),
]
for model_id, model_label, ft_base, pmbt_base in EXACT_DIRS:
    if not os.path.isdir(ft_base) or not os.path.isdir(pmbt_base):
        print(f"  ✗ {model_id} — missing dirs:\n    FT:   {ft_base}\n    PMBT: {pmbt_base}", file=sys.stderr); continue
    ft_n   = load_ft_neurons(ft_base,   args.max_neurons)
    pmbt_n = load_pmbt_neurons(pmbt_base, args.max_neurons)
    if not ft_n or not pmbt_n:
        print(f"  ✗ {model_id} — no neuron files", file=sys.stderr); continue
    print(f"  ✓ {model_id}: FT={len(ft_n):,}  PMBT={len(pmbt_n):,}")
    model_data.append((model_label, ft_n, pmbt_n))

if not model_data:
    print("  [warn] No neuron_labels.json files found — skipping Sec D figures.", file=sys.stderr)
    print("  Run step 4 (merge) first to generate per-neuron label files.")
    sys.exit(0)

n_models = len(model_data)
FT_PROB_TYPES = [("pv","Visual ($p_v$)","#d62728"),("pt","Text ($p_t$)","#1f77b4"),
                 ("pm","Multimodal ($p_m$)","#2ca02c"),("pu","Unknown ($p_u$)","#7f7f7f")]

# Fig D1: FT probability distributions
fig1, axes1 = plt.subplots(n_models, 4, figsize=(14, 3.2*n_models), constrained_layout=True)
if n_models == 1: axes1 = axes1[np.newaxis,:]
for row,(mlabel,ft_n,_) in enumerate(model_data):
    for col,(field,ctitle,color) in enumerate(FT_PROB_TYPES):
        ax = axes1[row,col]
        vals = np.array([n[field] for n in ft_n])
        ax.hist(vals, bins=BINS, range=(0,1), color=color, alpha=0.75,
                edgecolor="white", linewidth=0.3)
        ax.set_xlim(0,1); ax.set_xlabel("Probability", fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x,_: f"{int(x/1000)}k" if x>=1000 else str(int(x))))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        if row==0: ax.set_title(ctitle, fontsize=11, fontweight="bold", pad=5)
        if col==0: ax.set_ylabel(mlabel, fontsize=9, fontweight="bold")
fig1.suptitle("FT Probability Distributions per Neuron Type", fontsize=12, fontweight="bold", y=1.01)
for ext in ["pdf","png"]:
    p = os.path.join(args.output_dir, f"fig_D1_ft_probs.{ext}")
    fig1.savefig(p, dpi=args.dpi, bbox_inches="tight"); print(f"Saved: {p}")
plt.close(fig1)

# Fig D2: PMBT p-value + D distributions
fig2, axes2 = plt.subplots(n_models, 2, figsize=(10, 3.2*n_models), constrained_layout=True)
if n_models == 1: axes2 = axes2[np.newaxis,:]
for row,(mlabel,_,pmbt_n) in enumerate(model_data):
    ax = axes2[row,0]
    pvals = np.array([n["p_value"] for n in pmbt_n])
    ax.hist(pvals, bins=BINS, range=(0,1), color="#5c4e8a", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1.2, label=r"$\alpha=0.05$")
    ax.set_xlim(0,1); ax.set_xlabel("$p$-value", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x,_: f"{int(x/1000)}k" if x>=1000 else str(int(x))))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_ylabel(mlabel, fontsize=9, fontweight="bold")
    if row==0: ax.set_title("PMBT $p$-value Distribution", fontsize=11, fontweight="bold", pad=5)
    ax = axes2[row,1]
    dvals = np.clip([n["observed_rate_diff"] for n in pmbt_n], -1, 1)
    bins_d = np.linspace(-1,1,BINS+1)
    dvals = np.array(dvals)
    ax.hist(dvals, bins=bins_d, color="#888888", alpha=0.5, edgecolor="white", linewidth=0.3)
    ax.hist(dvals[dvals>=0], bins=bins_d[bins_d>=0], color="#d62728", alpha=0.7,
            edgecolor="white", linewidth=0.3, label="Visual ($D>0$)")
    ax.hist(dvals[dvals<0],  bins=bins_d[bins_d<=0], color="#1f77b4", alpha=0.7,
            edgecolor="white", linewidth=0.3, label="Text ($D<0$)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1,1); ax.set_xlabel("Rate difference $D$", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x,_: f"{int(x/1000)}k" if x>=1000 else str(int(x))))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    if row==0: ax.set_title("PMBT Rate Difference $D$", fontsize=11, fontweight="bold", pad=5)
fig2.suptitle("PMBT Statistical Distributions", fontsize=12, fontweight="bold", y=1.01)
for ext in ["pdf","png"]:
    p = os.path.join(args.output_dir, f"fig_D2_pmbt_stats.{ext}")
    fig2.savefig(p, dpi=args.dpi, bbox_inches="tight"); print(f"Saved: {p}")
plt.close(fig2)

# Fig D3: Confidence by label
try:
    from scipy.stats import gaussian_kde; HAS_SCIPY=True
except ImportError:
    HAS_SCIPY=False

CONF_LABELS = ["visual","text","multimodal"]
fig3, axes3 = plt.subplots(n_models, 2, figsize=(10, 3.2*n_models), constrained_layout=True)
if n_models == 1: axes3 = axes3[np.newaxis,:]
for row,(mlabel,ft_n,pmbt_n) in enumerate(model_data):
    for col,(neurons,get_conf,ctitle,xlbl) in enumerate([
        (ft_n,   lambda n: max(n["pv"],n["pt"],n["pm"]),
         "FT Confidence $\\max(p_v,p_t,p_m)$", "Confidence"),
        (pmbt_n, lambda n: -np.log10(max(n["p_value"], 1e-300)),
         "PMBT Significance $-\\log_{10}(p)$", "$-\\log_{10}(p)$"),
    ]):
        ax = axes3[row,col]
        by_label = {lbl:[] for lbl in CONF_LABELS}
        for n in neurons:
            lbl = n["label"]
            if lbl in by_label: by_label[lbl].append(get_conf(n))
        # For PMBT column: cap at -log10(0.05)~1.3 ... but p=0 neurons
        # dominate so cap at 5 to show separation between significant (spike at right)
        # and non-significant multimodal neurons (spread near 0)
        if col == 1:
            x_max = 5.0  # shows p=0.05 at ~1.3, p=1e-5 at 5
        else:
            x_max = 1.0
        for lbl in CONF_LABELS:
            vals = np.array(by_label[lbl])
            if len(vals)<10: continue
            color = TYPE_COLORS[lbl]
            if HAS_SCIPY and len(vals)>20:
                kde = gaussian_kde(vals, bw_method=0.15)
                xs = np.linspace(0, x_max, 400)
                ax.plot(xs, kde(xs), color=color, linewidth=1.8, label=lbl.capitalize())
                ax.fill_between(xs, kde(xs), alpha=0.12, color=color)
            else:
                ax.hist(vals, bins=40, range=(0, x_max), color=color, alpha=0.4,
                        density=True, label=lbl.capitalize())
        ax.set_xlim(0, x_max)
        ax.set_xlabel(xlbl, fontsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        if col==0: ax.set_ylabel(mlabel, fontsize=9, fontweight="bold")
        if row==0:
            ax.set_title(ctitle, fontsize=11, fontweight="bold", pad=5)
            ax.legend(fontsize=8, frameon=False)
fig3.suptitle("Classification Confidence by Label — FT vs PMBT",
              fontsize=12, fontweight="bold", y=1.01)
for ext in ["pdf","png"]:
    p = os.path.join(args.output_dir, f"fig_D3_confidence_by_label.{ext}")
    fig3.savefig(p, dpi=args.dpi, bbox_inches="tight"); print(f"Saved: {p}")
plt.close(fig3)
print("Done.")
CFEOF

# ── Auto-detect classify dir (handle full vs full_empty layout) ───────────
# Pass the parent results/3-classify root — the Python scripts use recursive
# glob so they will find models regardless of intermediate directory name.
CLASSIFY_DIR="results/3-classify"
echo "  Using classify dir: $CLASSIFY_DIR"

# ── Build commands ────────────────────────────────────────────────────────
LP_CMD="$PYTHON $LP_SCRIPT --classify-dir $CLASSIFY_DIR --output-dir $LP_OUT_DIR --dpi $PLOT_DPI"
SP_CMD="$PYTHON $SP_SCRIPT --classify-dir $CLASSIFY_DIR --output-dir $LP_OUT_DIR --dpi $PLOT_DPI"
CF_CMD="$PYTHON $CF_SCRIPT --classify-dir $CLASSIFY_DIR --output-dir $LP_OUT_DIR --dpi $PLOT_DPI"

if $LOCAL; then
    echo "  Running locally..."
    { eval "$LP_CMD" && eval "$SP_CMD"; } \
        2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        touch "$LP_MARKER"
        echo "  ✓ LP+SP done. Outputs: $LP_OUT_DIR"
        echo "  Running confidence figures (optional)..."
        eval "$CF_CMD" 2>&1 | tee -a "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" || \
            echo "  [warn] CF skipped — neuron_labels.json not found (run step 4 first)"
    else
        echo "  ERROR: layer_plots failed"
        exit 1
    fi
else
    BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
        -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
    rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
    bsub "${BSUB_ARGS[@]}" \
        "cd $WORK_DIR && $LP_CMD && $SP_CMD && touch $LP_MARKER && ($CF_CMD || echo '[warn] CF skipped — neuron_labels.json not found')"
    echo "  → Job: $JOB_NAME (CPU only)"
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end skip check

fi  # end step 14 (layer_plots)

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

# ── Plot-only mode: regenerate charts from existing results (no GPU) ──
if [[ "$HALLUC_PLOT_ONLY" == "1" ]]; then
    echo "  [plot-only] Regenerating plots from existing results in $HALLUC_SCORES_DIR"
    _HS_DIR="$WORK_DIR/$HALLUC_SCORES_DIR"
    if [[ ! -f "$_HS_DIR/enrichment_results_delta_h.json" ]]; then
        echo "  ERROR: No enrichment_results_delta_h.json found in $_HS_DIR"
        echo "         Run step 10 fully first to generate results."
        exit 1
    fi

    $PYTHON -c "
import json, numpy as np, sys, os
sys.path.insert(0, os.path.join('$WORK_DIR', 'code'))
from halluc_score_neurons import plot_enrichment_results

out_dir = '$_HS_DIR'
with open(os.path.join(out_dir, 'enrichment_results_delta_h.json')) as f:
    results = json.load(f)

tqa_path = os.path.join(out_dir, 'enrichment_results_delta_h_tqa.json')
tqa_results = None
if os.path.exists(tqa_path):
    with open(tqa_path) as f:
        tqa_results = json.load(f)
    print(f'  Loaded TQA enrichment → side-by-side charts')

heatmap = np.load(os.path.join(out_dir, 'per_layer_enrichment.npy'))
cats = ['visual', 'text', 'multimodal', 'unknown']

plot_enrichment_results(results, heatmap, cats, out_dir, '$MODEL_NAME', tqa_results=tqa_results)
print(f'  Plots saved to {out_dir}/')
"
    echo "  ✓ Step 10 plot-only complete."
    exit 0
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

# TriviaQA dual-source scoring (Section 3.5) — on by default
if [[ "$HALLUC_TRIVIAQA" == "1" ]]; then
    HS_COMMON_ARGS="$HS_COMMON_ARGS --triviaqa_path $TRIVIAQA_PATH"
    HS_COMMON_ARGS="$HS_COMMON_ARGS --triviaqa_num $TRIVIAQA_NUM"
    if [[ "$MODE" == "test" ]]; then
        HS_COMMON_ARGS="$HS_COMMON_ARGS --triviaqa_cap 10"
        echo "  [triviaqa-test] cap=10 questions for test mode"
    else
        echo "  [triviaqa] Enabled: pool=$TRIVIAQA_NUM → cap=1000 clean questions"
    fi
else
    HS_COMMON_ARGS="$HS_COMMON_ARGS --no_triviaqa"
    echo "  [triviaqa] Disabled (--no-halluc-triviaqa)"
fi

# Encoder/projector neuron scoring — on by default
if [[ "$HALLUC_ENCODER_PROJECTOR" != "1" ]]; then
    HS_COMMON_ARGS="$HS_COMMON_ARGS --no_encoder_projector"
    echo "  [encoder/projector] Disabled (--no-encoder-projector)"
else
    echo "  [encoder/projector] Enabled: scoring encoder + projector neurons (ΔH + CETT + combined)"
fi

# Skip ablation loop — reuse existing ΔH scores from previous run
if [[ "$HALLUC_SKIP_ABLATION" == "1" ]]; then
    EXISTING_SCORES="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}/ablation_scores_delta_h.json"
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
BASE_CMD="$PYTHON $HALLUC_SCORE_SCRIPT $HS_COMMON_ARGS"

# ── GPU memory tiers for step 10 ────────────────────────────────
# Preprocessing loads model + runs 37.5K forward passes → needs more VRAM
# Per-layer ablation loads model + runs ~1K forward passes per batch → less VRAM
# InternVL needs 80G for all jobs (larger model / trust_remote_code overhead)
HS_GMEM_PREPROCESS="80G"
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    HS_GMEM_LAYER="80G"
else
    HS_GMEM_LAYER="40G"
fi

# ── Jobs S0-S2: One per POPE split (3 parallel, ~1.3h each, gmem=80G) ────
if [[ "$HALLUC_MERGE_ONLY" == "1" ]]; then
    echo "  [halluc-merge-only] Skipping splits, preprocess, and layer jobs"
    echo "  [halluc-merge-only] Assumes all data already exists in $HALLUC_SCORES_DIR"
else

# Marker file: written by preprocess on success, checked by layer jobs
PREPROCESS_MARKER="$WORK_DIR/$HALLUC_SCORES_DIR/.preprocess_done"

POPE_SPLITS=("random" "popular" "adversarial")
SPLIT_JOB_NAMES=()

if [[ -f "$PREPROCESS_MARKER" ]]; then
    echo "  [skip] Preprocess marker exists — skipping all split + preprocess jobs"
else

for SPLIT in "${POPE_SPLITS[@]}"; do
    JOB_SPLIT="${JN10}_split_${SPLIT}"
    SPLIT_JOB_NAMES+=("$JOB_SPLIT")
    SPLIT_CMD="$BASE_CMD --preprocess_only --pope_split $SPLIT --n_gpus 1"

    echo "  [S] $JOB_SPLIT → contrastive sampling ($SPLIT) [gmem=$HS_GMEM_PREPROCESS]"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_SPLIT}${LOG_SUFFIX}."{log,err}

    if $LOCAL; then
        (cd "$WORK_DIR" && $SPLIT_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_SPLIT}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$JOB_SPLIT" \
            -gpu "num=1:gmem=$HS_GMEM_PREPROCESS" \
            -R "$GPU_RES_BASE" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_SPLIT}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_SPLIT}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $SPLIT_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

fi  # end if NOT marker exists (split jobs)

# ── Job P: Merge splits + CETT-diff + TriviaQA (depends on all 3 splits) ──
JOB_PREPROCESS="${JN10}_preprocess"
PREPROCESS_CMD="$BASE_CMD --merge_preprocess --n_gpus 1 --pope_splits_dir $(dirname $POPE_PATH)"
# Append marker touch on success
PREPROCESS_CMD_FULL="$PREPROCESS_CMD && touch $PREPROCESS_MARKER"

echo "  [P] $JOB_PREPROCESS → merge splits + CETT-diff + TriviaQA [gmem=$HS_GMEM_PREPROCESS]"
rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_PREPROCESS}${LOG_SUFFIX}."{log,err}

if $LOCAL; then
    (cd "$WORK_DIR" && $PREPROCESS_CMD_FULL) \
        2>&1 | tee "$STEP_LOG_DIR/${JOB_PREPROCESS}${LOG_SUFFIX}.log"
else
    if [[ -f "$PREPROCESS_MARKER" ]]; then
        echo "  [skip] Preprocess marker exists — preprocess already done"
    else
        # Build dependency: done(split_random) && done(split_popular) && done(split_adversarial)
        SPLIT_DEP=""
        for jn in "${SPLIT_JOB_NAMES[@]}"; do
            [[ -n "$SPLIT_DEP" ]] && SPLIT_DEP="$SPLIT_DEP && "
            SPLIT_DEP="${SPLIT_DEP}done($jn)"
        done
        bsub -q "$QUEUE" -J "$JOB_PREPROCESS" \
            -w "$SPLIT_DEP" \
            -gpu "num=1:gmem=$HS_GMEM_PREPROCESS" \
            -R "$GPU_RES_BASE" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_PREPROCESS}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_PREPROCESS}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $PREPROCESS_CMD_FULL"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

# ── Jobs L0..LN: One per layer (1 GPU ≥40G each, ~2-3 hours) ────
LAYER_JOB_NAMES=()
for L in $(seq 0 $((HS_N_LAYERS - 1))); do
    JOB_LAYER="${JN10}_L${L}"
    LAYER_JOB_NAMES+=("$JOB_LAYER")
    LAYER_CMD="$BASE_CMD --layer_start $L --layer_end $((L + 1)) --n_gpus 1"

    echo "  [L] $JOB_LAYER → layer $L ablation [gmem=$HS_GMEM_LAYER]"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_LAYER}${LOG_SUFFIX}."{log,err}

    if $LOCAL; then
        (cd "$WORK_DIR" && $LAYER_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_LAYER}${LOG_SUFFIX}.log"
    else
        # Use marker file instead of LSF done() — immune to job history expiry
        _LAYER_DEP_ARGS=()
        if [[ ! -f "$PREPROCESS_MARKER" ]]; then
            _LAYER_DEP_ARGS+=(-w "done($JOB_PREPROCESS)")
        fi
        bsub -q "$QUEUE" -J "$JOB_LAYER" \
            "${_LAYER_DEP_ARGS[@]}" \
            -gpu "num=1:gmem=$HS_GMEM_LAYER" \
            -R "$GPU_RES_BASE" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_LAYER}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_LAYER}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $LAYER_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

fi  # end if NOT halluc-merge-only

# ── Job M: Merge + Enrichment + Plots (CPU only, depends on all layers) ──
JOB_MERGE="${JN10}_merge"
MERGE_CMD="$BASE_CMD --merge_layer_scores"

echo "  [M] $JOB_MERGE → merge scores + enrichment + plots"
rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_MERGE}${LOG_SUFFIX}."{log,err}

if $LOCAL; then
    (cd "$WORK_DIR" && $MERGE_CMD) \
        2>&1 | tee "$STEP_LOG_DIR/${JOB_MERGE}${LOG_SUFFIX}.log"
else
    if [[ "$HALLUC_MERGE_ONLY" == "1" ]]; then
        # No dependencies — data already exists, but may need GPU for TriviaQA checks
        bsub -q "$QUEUE" -J "$JOB_MERGE" \
            -gpu "num=1:gmem=${HS_GMEM_LAYER}" \
            -R "$GPU_RES_BASE" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_MERGE}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_MERGE}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $MERGE_CMD"
    else
        # Build dependency string: done(L0) && done(L1) && ... && done(LN)
        DEP_STR=""
        for jn in "${LAYER_JOB_NAMES[@]}"; do
            [[ -n "$DEP_STR" ]] && DEP_STR="$DEP_STR && "
            DEP_STR="${DEP_STR}done($jn)"
        done
        bsub -q "$QUEUE" -J "$JOB_MERGE" \
            -w "$DEP_STR" \
            -R "rusage[mem=16384]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_MERGE}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_MERGE}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $MERGE_CMD"
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
STEERING_OUT="results/11-steering/${MODE_DIR}/${MODEL_NAME}/${HALLUC_SCORE_METHOD}"
echo "  [method] $HALLUC_SCORE_METHOD → output: $STEERING_OUT"
GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))

# ── Test mode: 1 alpha, all conditions, no curve, minimal benchmarks ──
if [[ "$MODE" == "test" ]]; then
    ST_ALPHAS="0.5"
    ST_CONDITIONS="baseline ablate_vis ablate_text ablate_multimodal ablate_unknown random ablate_encoder ablate_projector"
    ST_BASE_ARGS="--num_images 0 --pope_num_questions 5"
    ST_SHARDS=1
    ST_CHAIR_NUM=5
    [[ "$TRIVIAQA_NUM" == "2000" ]] && TRIVIAQA_NUM=5
    [[ "$MMLU_NUM" == "2000" ]]     && MMLU_NUM=5
    echo "  [test] alpha=0.5, 8 conditions (all + encoder/projector), CHAIR=5, 1 GPU"
else
    ST_ALPHAS="$STEERING_ALPHAS"
    ST_CONDITIONS="baseline ablate_vis ablate_text ablate_multimodal ablate_unknown random ablate_encoder ablate_projector"
    ST_BASE_ARGS=""
    ST_SHARDS=$STEERING_SHARDS
    ST_CHAIR_NUM=$CHAIR_NUM_IMAGES

    if $ABLATION_CURVE; then
        ST_BASE_ARGS="$ST_BASE_ARGS --curve --curve_steps $ABLATION_CURVE_STEPS"
    elif [[ -n "$ABLATION_TOP_N" ]]; then                                  # Line N1: curve off, top_n set
        ST_BASE_ARGS="$ST_BASE_ARGS --top_n $ABLATION_TOP_N"               # Line N2: pass single-point top_n
    fi

    IFS=',' read -ra _alphas <<< "$ST_ALPHAS"
    echo "  [full] ${#_alphas[@]} alpha values × ${ST_SHARDS} GPUs each"
fi

# Add multimodal sub-conditions when TQA scores are available
_TQA_CHECK="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}/ablation_scores_delta_h_tqa.json"
if [[ -f "$WORK_DIR/$_TQA_CHECK" ]]; then
    ST_CONDITIONS="$ST_CONDITIONS ablate_multi_pope ablate_multi_tqa ablate_multi_mean"
    echo "  [multi sub-conditions] +3 conditions: ablate_multi_pope, ablate_multi_tqa, ablate_multi_mean"
fi

# Prepend --conditions to the args
ST_BASE_ARGS="--conditions $ST_CONDITIONS $ST_BASE_ARGS"

# ── Append benchmark paths ──
if [[ -n "$POPE_PATH" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --pope_path $POPE_PATH"
    [[ -n "$POPE_IMG_DIR" ]] && ST_BASE_ARGS="$ST_BASE_ARGS --pope_img_dir $POPE_IMG_DIR"
fi
if [[ -n "$CHAIR_ANN_PATH" ]] && (( ST_CHAIR_NUM > 0 )); then
    ST_BASE_ARGS="$ST_BASE_ARGS --chair_ann_path $CHAIR_ANN_PATH --chair_num_images $ST_CHAIR_NUM"
    [[ -n "$POPE_IMG_DIR" ]] && ST_BASE_ARGS="$ST_BASE_ARGS --chair_img_dir $POPE_IMG_DIR"
fi
if [[ -n "$TRIVIAQA_PATH" ]] && [[ "$MODEL_TYPE" != "llava-mistral" && "$MODEL_TYPE" != "llava-llama3" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --triviaqa_path $TRIVIAQA_PATH --triviaqa_num_questions $TRIVIAQA_NUM"
fi
if [[ -n "$MMLU_DIR" ]] && [[ "$MODEL_TYPE" != "llava-mistral" && "$MODEL_TYPE" != "llava-llama3" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
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
if [[ "$HALLUC_SCORE_METHOD" == "d" ]]; then                                # Line D-shell-1: D-ranking branch
    ST_BASE_ARGS="$ST_BASE_ARGS --ranking_method d"                         # Line D-shell-2: pass --ranking_method d to python
    HALLUC_SCORES_FILE=""                                                   # Line D-shell-3: clear so file-load below is skipped
    echo "  [d-ranking] using |observed_rate_diff| from labels (no halluc-scores file)"  # Line D-shell-4: log
else
    case "$HALLUC_SCORE_METHOD" in
        combined) HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/combined_halluc_scores.json" ;;
        dh)       HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/ablation_scores_delta_h.json" ;;
        cett)     HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/cett_diff_scores.json" ;;
        *)        echo "  WARNING: Unknown halluc score method '$HALLUC_SCORE_METHOD', using combined"
                  HALLUC_SCORES_FILE="$HALLUC_SCORES_DIR_ST/combined_halluc_scores.json" ;;
    esac
fi

if [[ -n "$HALLUC_SCORES_FILE" && -f "$WORK_DIR/$HALLUC_SCORES_FILE" ]]; then  # Line D-shell-5: also gate on non-empty
    ST_BASE_ARGS="$ST_BASE_ARGS --halluc_scores_path $HALLUC_SCORES_FILE"
    echo "  [halluc ranking] $HALLUC_SCORE_METHOD → $HALLUC_SCORES_FILE"
elif [[ "$HALLUC_SCORE_METHOD" != "d" ]]; then                              # Line D-shell-6: skip warning for d
    echo "  [halluc ranking] $HALLUC_SCORES_FILE not found — using classification confidence"
    echo "         (run step 10 with --halluc-contrastive to generate halluc scores)"
fi

# Pass TriviaQA scores for text neuron ranking (if step 10 produced them)
HALLUC_SCORES_TQA_FILE="$HALLUC_SCORES_DIR_ST/ablation_scores_delta_h_tqa.json"
if [[ -f "$WORK_DIR/$HALLUC_SCORES_TQA_FILE" ]]; then
    ST_BASE_ARGS="$ST_BASE_ARGS --halluc_scores_tqa_path $HALLUC_SCORES_TQA_FILE"
    echo "  [halluc ranking TQA] text neurons ranked by TriviaQA ΔH_TQA"
else
    echo "  [halluc ranking TQA] ablation_scores_delta_h_tqa.json not found — text neurons use POPE ranking"
    echo "         (run step 10 with TriviaQA enabled to generate TQA scores)"
fi

# Pass encoder neuron scores for neuron-level steering (if step 10 produced them)
# Prefer combined > dh > cett for encoder/projector rankings
for _comp in encoder projector; do
    _COMP_UPPER=$(echo "$_comp" | tr '[:lower:]' '[:upper:]')
    _found=""
    for _method in combined_halluc_scores ablation_scores_delta_h cett_diff_scores; do
        _COMP_FILE="$HALLUC_SCORES_DIR_ST/${_method}_${_comp}.json"
        if [[ -f "$WORK_DIR/$_COMP_FILE" ]]; then
            ST_BASE_ARGS="$ST_BASE_ARGS --halluc_scores_${_comp}_path $_COMP_FILE"
            echo "  [halluc ranking $_COMP_UPPER] neuron-level steering via ${_method}_${_comp}.json"
            _found="1"
            break
        fi
    done
    if [[ -z "$_found" ]]; then
        echo "  [halluc ranking $_COMP_UPPER] no neuron scores found — using full-component scaling"
        echo "         (run step 10 with encoder/projector scoring to enable neuron-level steering)"
    fi
done

# ── Function to submit one alpha value ──
submit_steering_alpha() {
    local ALPHA="$1"
    local TAX_SOURCE="$2"
    local TAX_LABELS_DIR="$3"
    local TAX_PREFIX="$4"
    local JOB_SUFFIX="${TAX_PREFIX}_a${ALPHA}_${HALLUC_SCORE_METHOD}"
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

STEERING_OUT="results/11-steering/${MODE_DIR}/${MODEL_NAME}/${HALLUC_SCORE_METHOD}"
PLOT_OUT_DIR="results/13-plots/${MODE_DIR}/${MODEL_NAME}${RUN_SUFFIX}/${HALLUC_SCORE_METHOD}"
echo "  [method] $HALLUC_SCORE_METHOD → steering: $STEERING_OUT"

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

MERGED_JSON="$PLOT_OUT_DIR/steering_merged_${HALLUC_SCORE_METHOD}.json"
_RAW_MERGED="$PLOT_OUT_DIR/steering_merged.json"

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
    # Rename with method suffix for easy identification
    if [[ -f "$WORK_DIR/$_RAW_MERGED" ]]; then
        mv "$WORK_DIR/$_RAW_MERGED" "$WORK_DIR/$MERGED_JSON"
        echo "  → Renamed: steering_merged.json → steering_merged_${HALLUC_SCORE_METHOD}.json"
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

PLOT_OUT_DIR="results/13-plots/${MODE_DIR}/${MODEL_NAME}${RUN_SUFFIX}/${HALLUC_SCORE_METHOD}"
MERGED_JSON="$PLOT_OUT_DIR/steering_merged_${HALLUC_SCORE_METHOD}.json"
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

# ── Derive a short, clean filename suffix from MODEL_NAME ──
# Maps: llava-onevision-7b → llava_ov, internvl2.5-8b → internvl, qwen2.5-vl-7b → qwen
case "$MODEL_NAME" in
    llava-onevision*)  FIG_SUFFIX="llava_ov" ;;
    llava-1.5*)        FIG_SUFFIX="llava_15" ;;
    internvl*)         FIG_SUFFIX="internvl" ;;
    qwen*)             FIG_SUFFIX="qwen" ;;
    *)                 FIG_SUFFIX=$(echo "$MODEL_NAME" | tr '.-' '_') ;;
esac

echo "  → Generating figures to $PLOT_OUT_DIR/  (suffix: _${FIG_SUFFIX})"
if $LOCAL; then
    (cd "$WORK_DIR" && $PLOT_CMD)
else
    (cd "$WORK_DIR" && $PLOT_CMD)
fi

# ── Rename fig_*.pdf → fig_*_<model>_<method>.pdf so filenames are unique per model+method ──
echo "  → Renaming figures with model suffix '_${FIG_SUFFIX}' and method '${HALLUC_SCORE_METHOD}'..."
for f in "$WORK_DIR/$PLOT_OUT_DIR"/fig_*.pdf; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f" .pdf)"         # e.g. fig_A_dose_response
    newname="${base}_${FIG_SUFFIX}_${HALLUC_SCORE_METHOD}.pdf"   # e.g. fig_A_dose_response_llava_ov_dh.pdf
    mv "$f" "$(dirname "$f")/$newname"
    echo "    $(basename "$f") → $newname"
done

SUBMITTED=$((SUBMITTED + 1))

fi  # end step 13 (plot_steering)

# ═══════════════════════════════════════════════════════════════
# STEP 15 (enrichment_plots): Section H figures — OR forest plot + per-layer heatmaps
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "enrichment_plots" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 15: Enrichment plots (OR forest + per-layer heatmaps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EP_OUT="results/15-enrichment-plots/full"
EP_LOG_DIR="logs/full/15-enrichment-plots"
mkdir -p "$WORK_DIR/$EP_OUT" "$WORK_DIR/$EP_LOG_DIR"

EP_MARKER="$WORK_DIR/$EP_OUT/done.marker"
if [[ -f "$EP_MARKER" && "$CLEAN" != "1" ]]; then
    echo "  Already done (marker exists). Use --clean to rerun."
    SKIPPED=$((SKIPPED + 1))
else
    rm -f "$EP_MARKER"

    EP_SCRIPT="$WORK_DIR/$EP_LOG_DIR/generate_enrichment_plots.py"

    # ── Write plotting script ──────────────────────────────────
    cat > "$EP_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
Step 15: Generate Section H supplementary figures from step-10 enrichment outputs.

Produces:
  fig_G1_or_forest.pdf              — OR forest plot, 3 models × 4 categories
  per_layer_enrichment_heatmap-llava-ov.pdf
  per_layer_enrichment_heatmap_internvl.pdf
  per_layer_enrichment_heatmap_qwen.pdf
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────
WORK_DIR  = sys.argv[1]   # repo root
OUT_DIR   = sys.argv[2]   # results/15-enrichment-plots/full

# Hardcoded paths to step-10 enrichment outputs
MODELS = [
    ("LLaVA-OV-7B",    "llava-onevision-7b"),
    ("Qwen2.5-VL-7B",  "qwen2.5-vl-7b"),
    ("InternVL2.5-8B", "internvl2.5-8b"),
]

ENRICH_ROOT = os.path.join(WORK_DIR, "results/10-halluc_scores/full")

CATS      = ["visual", "text", "multimodal", "unknown"]  # column order in .npy
CAT_LABELS = ["Visual", "Text", "Multimodal", "Unknown"]
# Colours consistent with rest of paper
CAT_COLORS = {"visual": "#d62728", "text": "#1f77b4",
               "multimodal": "#2ca02c", "unknown": "#7f7f7f"}

# Minimum number of neurons in driving set to trust enrichment statistics.
# Categories below this threshold are shown as greyed-out / excluded.
MIN_DRIVING_NEURONS = 50

DPI = 200

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
def load_model(model_dir_name):
    """Load enrichment_results_delta_h.json and per_layer_enrichment.npy for one model."""
    base = os.path.join(ENRICH_ROOT, model_dir_name)
    with open(os.path.join(base, "enrichment_results_delta_h.json")) as f:
        er = json.load(f)
    heatmap = np.load(os.path.join(base, "per_layer_enrichment.npy"))
    return er, heatmap

# Filter to models that have completed step 10
_MODELS_ALL = MODELS[:]
MODELS = []
for _label, _dirname in _MODELS_ALL:
    _base = os.path.join(ENRICH_ROOT, _dirname)
    if os.path.isfile(os.path.join(_base, "enrichment_results_delta_h.json")) \
       and os.path.isfile(os.path.join(_base, "per_layer_enrichment.npy")):
        MODELS.append((_label, _dirname))
    else:
        print(f"  {_label}: SKIPPED (step 10 not complete)")

if not MODELS:
    print("ERROR: No models have completed step 10. Nothing to plot.")
    sys.exit(1)
print(f"  Plotting {len(MODELS)}/{len(_MODELS_ALL)} models: {[m[0] for m in MODELS]}")

# ── Figure 1: OR forest plot ──────────────────────────────────
def plot_forest(models_data, out_path):
    """
    OR forest plot — 3 models, 4 categories each = 12 rows.
    X axis: odds ratio (log scale).
    Error bars: 95% CI from random baseline fold enrichment, converted to OR scale.
    Vertical dashed line at OR = 1.0.
    """
    fig, ax = plt.subplots(figsize=(7, 6))  # width, height in inches

    n_models = len(models_data)
    n_cats   = len(CATS)

    # Build row positions — group by model, gap between models
    row_labels = []  # (y_pos, label, color, or_val, ci_low, ci_high, sig)
    y = 0
    group_centers = []  # for model name annotation

    for model_label, er in models_data:
        group_y_start = y
        for cat in CATS:
            cv  = er["categories"][cat]
            bv  = er["random_baseline"][cat]

            n_driving = cv.get("n_in_driving", 999)
            if n_driving < MIN_DRIVING_NEURONS:
                # Insufficient neurons — plot greyed-out placeholder
                row_labels.append((y, CAT_LABELS[CATS.index(cat)],
                                    "#cccccc", None, None, None, False, 1.0))
                y += 1
                continue

            or_val  = cv["odds_ratio"]
            fold    = cv["fold_enrichment"]
            p       = cv["p_value"]
            sig     = cv["significant"] == "True" or cv["significant"] is True

            # CI: baseline gives fold CI → scale to OR space
            # OR ≈ fold when base rate is small, so use fold CI scaled by OR/fold
            scale = or_val / fold if fold > 0 else 1.0
            ci_low  = bv["ci_95_low"]  * scale
            ci_high = bv["ci_95_high"] * scale

            # Clip extreme CI for unknown (baseline CI can be 0–2.353)
            ci_low  = max(ci_low,  0.01)
            ci_high = min(ci_high, 10.0)

            row_labels.append((y, CAT_LABELS[CATS.index(cat)],
                                CAT_COLORS[cat], or_val, ci_low, ci_high, sig, p))
            y += 1
        group_centers.append((group_y_start + y - 1) / 2.0)
        y += 0.8  # gap between model groups

    # Plot rows
    for (yp, lbl, col, orv, cil, cih, sig, pval) in row_labels:
        if orv is None:
            # Insufficient data — show label only, greyed out
            ax.text(1.0, yp, f"  {lbl} (n<{MIN_DRIVING_NEURONS})",
                    va='center', fontsize=7, color='#aaaaaa')
            continue
        # CI bar
        ax.plot([cil, cih], [yp, yp], color="grey", linewidth=1.2, zorder=1)
        # OR diamond
        ax.scatter([orv], [yp], color=col, s=60, zorder=3,
                   marker="D" if sig else "o")
        # p-value significance stars
        if pval == 0.0 or pval < 1e-200:
            stars = "***"
        elif pval < 1e-10:
            stars = "***"
        elif pval < 0.001:
            stars = "**"
        elif pval < 0.05:
            stars = "*"
        else:
            stars = "ns"
        ax.text(cih + 0.03, yp, stars, va='center', fontsize=7, color=col)

    # Model name annotations on right y-axis
    y_positions = [r[0] for r in row_labels]
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[1] for r in row_labels], fontsize=8)

    # Model group labels on left
    for i, (model_label, _) in enumerate(models_data):
        ax.text(-0.28, group_centers[i], model_label,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=8, fontweight='bold')

    # Reference line at OR=1
    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8, zorder=0)

    ax.set_xlabel("Odds Ratio (OR)", fontsize=10)
    ax.set_title("Odds Ratio Forest Plot — Hallucination-Driving Neuron Enrichment",
                 fontsize=10, pad=8)
    ax.set_xscale('log')
    ax.set_xlim(0.03, 12)
    ax.invert_yaxis()  # top model first

    # Legend
    handles = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[CATS.index(c)])
               for c in CATS]
    handles += [Line2D([0],[0], marker='D', color='grey', label='Significant (p<0.05)',
                        markerfacecolor='grey', markersize=6, linestyle='None'),
                Line2D([0],[0], marker='o', color='grey', label='Non-significant',
                        markerfacecolor='grey', markersize=6, linestyle='None')]
    ax.legend(handles=handles, fontsize=7, loc='lower right')

    ax.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# ── Figure 2-4: Per-layer enrichment heatmaps ─────────────────
def plot_heatmap(heatmap, model_label, out_path):
    """
    Per-layer enrichment heatmap.
    Rows = layers, Cols = [Visual, Text, Multimodal] (Unknown excluded — near-zero everywhere).
    Colour = fold enrichment, diverging colormap properly centred at 1.0.
    """
    from matplotlib.colors import TwoSlopeNorm

    # Drop Unknown column (index 3) — always near zero, adds no information
    heatmap_plot = heatmap[:, :3]          # shape (n_layers, 3)
    col_labels   = ["Visual", "Text", "Multimodal"]

    n_layers = heatmap_plot.shape[0]

    # Compute symmetric bounds around 1.0
    vmin = max(0.0, float(np.nanmin(heatmap_plot)))
    vmax = float(np.nanpercentile(heatmap_plot, 99))
    vmax = max(vmax, 1.5)   # always show at least up to 1.5
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(3.5, max(4, n_layers * 0.22)))

    im = ax.imshow(heatmap_plot, aspect='auto', cmap='RdBu_r',
                   norm=norm, interpolation='nearest')

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=6)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_title(f"Per-Layer Fold Enrichment — {model_label}", fontsize=10, pad=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Fold Enrichment", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Mark null line at fold=1.0 on colorbar
    cbar.ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# ── Main ──────────────────────────────────────────────────────
print("Loading enrichment data...")
all_data = []
heatmaps = {}
for label, dirname in MODELS:
    er, hm = load_model(dirname)
    all_data.append((label, er))
    heatmaps[label] = hm
    print(f"  {label}: {hm.shape[0]} layers loaded")

print("\nGenerating OR forest plot...")
plot_forest(
    all_data,
    os.path.join(OUT_DIR, "fig_G1_or_forest.pdf")
)

print("\nGenerating per-layer heatmaps...")
heatmap_names = {
    "LLaVA-OV-7B":    "per_layer_enrichment_heatmap-llava-ov.pdf",
    "InternVL2.5-8B": "per_layer_enrichment_heatmap_internvl.pdf",
    "Qwen2.5-VL-7B":  "per_layer_enrichment_heatmap_qwen.pdf",
}
for label, _ in MODELS:
    plot_heatmap(
        heatmaps[label],
        label,
        os.path.join(OUT_DIR, heatmap_names[label])
    )

print("\nGenerating three-method comparison figure...")

def plot_method_comparison(pope_data, tqa_data, out_path):
    """
    Grouped bar chart: 3 models × 3 methods, showing fold enrichment
    for all 4 categories side by side.
    One subplot per model, bars grouped by category, coloured by method.
    Row 1: POPE-based enrichment (visual hallucination)
    Row 2: TriviaQA-based enrichment (text hallucination)
    """
    methods      = ["dh", "cett_diff", "combined"]
    method_labels = [r"$\Delta H$ only", "CETT-diff only", "Combined"]
    method_colors = ["#4878cf", "#6acc65", "#d65f5f"]

    n_models = len(MODELS)
    n_rows = 2 if tqa_data else 1
    fig, axes = plt.subplots(n_rows, n_models,
                              figsize=(4.5 * n_models, 4.5 * n_rows),
                              sharey=False, squeeze=False)

    bar_width = 0.22
    x = np.arange(len(CATS))  # 4 category positions

    row_labels = ["POPE (Visual Hallucination)"]
    row_data = [pope_data]
    if tqa_data:
        row_labels.append("TriviaQA (Text Hallucination)")
        row_data.append(tqa_data)

    for row_idx, (row_label, data_dict) in enumerate(zip(row_labels, row_data)):
        for col_idx, (model_label, _) in enumerate(MODELS):
            ax = axes[row_idx][col_idx]
            has_data = False
            for mi, (method, mlabel, mcol) in enumerate(
                    zip(methods, method_labels, method_colors)):
                er = data_dict.get(model_label, {}).get(method)
                if er is None:
                    continue
                has_data = True
                folds = []
                for cat in CATS:
                    n_drv = er["categories"][cat].get("n_in_driving", 999)
                    folds.append(er["categories"][cat]["fold_enrichment"]
                                 if n_drv >= MIN_DRIVING_NEURONS else 0.0)
                offset = (mi - 1) * bar_width  # centre the 3 bars
                bars = ax.bar(x + offset, folds, bar_width,
                              label=mlabel, color=mcol, alpha=0.85,
                              edgecolor='white', linewidth=0.5)

                # Add significance stars above bars
                for xi, cat in enumerate(CATS):
                    n_drv = er["categories"][cat].get("n_in_driving", 999)
                    if n_drv < MIN_DRIVING_NEURONS:
                        continue
                    pval = er["categories"][cat]["p_value"]
                    if pval == 0.0 or pval < 1e-200:
                        stars = "***"
                    elif pval < 0.001:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                    if stars:
                        bar_top = folds[CATS.index(cat)]
                        ax.text(xi + offset, bar_top + 0.02, stars,
                                ha='center', va='bottom', fontsize=6, color=mcol)

            ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(CAT_LABELS, fontsize=9)
            if row_idx == 0:
                ax.set_title(model_label, fontsize=10, fontweight='bold')
            ax.set_ylabel(f"Fold Enrichment\n({row_label})", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', linestyle=':', alpha=0.3)
            if not has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='grey')

    # Shared legend on last axis of first row
    axes[0][-1].legend(fontsize=8, loc='upper right')

    fig.suptitle("Three-Method Enrichment Comparison\n"
                 r"($\Delta H$, CETT-diff, Combined)"
                 " — POPE (top) vs TriviaQA (bottom)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# Load all 3 method results for each model — POPE-based
method_files_pope = {
    "dh":        "enrichment_results_delta_h.json",
    "cett_diff": "enrichment_results_cett_diff.json",
    "combined":  "enrichment_results_combined.json",
}

# Load TriviaQA-based enrichment (may not exist for all models)
method_files_tqa = {
    "dh":        "enrichment_results_delta_h_tqa.json",
    "cett_diff": "enrichment_results_cett_diff_tqa.json",
    "combined":  "enrichment_results_combined.json",  # fallback to POPE combined if TQA combined not available
}

models_data_pope = {}
models_data_tqa = {}
for label, dirname in MODELS:
    models_data_pope[label] = {}
    models_data_tqa[label] = {}
    base = os.path.join(ENRICH_ROOT, dirname)

    # POPE enrichment (required)
    for method, fname in method_files_pope.items():
        fpath = os.path.join(base, fname)
        with open(fpath) as f:
            models_data_pope[label][method] = json.load(f)

    # TriviaQA enrichment (optional)
    tqa_loaded = 0
    for method, fname in method_files_tqa.items():
        fpath = os.path.join(base, fname)
        # For combined, try TQA-specific file first
        if method == "combined":
            tqa_combined_path = os.path.join(base, "enrichment_results_combined_tqa.json")
            if os.path.isfile(tqa_combined_path):
                fpath = tqa_combined_path
        if os.path.isfile(fpath):
            with open(fpath) as f:
                models_data_tqa[label][method] = json.load(f)
            tqa_loaded += 1

    print(f"  {label}: POPE 3/3, TriviaQA {tqa_loaded}/3 method files loaded")

has_tqa = any(len(v) > 0 for v in models_data_tqa.values())

plot_method_comparison(
    models_data_pope,
    models_data_tqa if has_tqa else None,
    os.path.join(OUT_DIR, "fig_D_enrichment_comparison.pdf")
)

print("\nAll enrichment plots done.")
PYEOF

    # ── Run the script ─────────────────────────────────────────
    echo "  Running enrichment plotting script..."
    (cd "$WORK_DIR" && python3 "$EP_SCRIPT" "$WORK_DIR" "$WORK_DIR/$EP_OUT") \
        2>&1 | tee "$WORK_DIR/$EP_LOG_DIR/enrichment_plots.log"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        touch "$EP_MARKER"
        echo "  ✓ Step 15 complete. Marker written."
        echo "  Outputs in: $EP_OUT/"
        ls "$WORK_DIR/$EP_OUT/"*.pdf 2>/dev/null | sed 's/^/    /'
    else
        echo "  ERROR: enrichment_plots.py failed — check $EP_LOG_DIR/enrichment_plots.log"
        exit 1
    fi
fi

fi  # end step 15 (enrichment_plots)


# ═══════════════════════════════════════════════════════════════
# STEP 16 (text_inject): Task Arithmetic + PMBT text mask (Method 1)
#   Injects math task vector ONLY into text-classified neurons.
#   Reference: "Bring Reason to Vision" (Chen et al., ICML 2025)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "text_inject" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 16: Task Arithmetic + PMBT text mask (Method 1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/16-text-inject"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

MERGE_OUT_DIR="results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}"

# ── Resolve base LLM and math LLM paths per backbone ──────────
# These must share the same backbone as the VLM.
# Override with --merge-base-llm and --merge-math-llm if needed.
if [[ -z "$MERGE_BASE_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)
            MERGE_BASE_LLM_PATH="Qwen/Qwen2.5-7B"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Qwen2.5 backbone, verified diff=81.8)"
            ;;
        llava-ov)
            MERGE_BASE_LLM_PATH="Qwen/Qwen2-7B"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Qwen2 backbone, verified diff=12.7)"
            ;;
        internvl)
            MERGE_BASE_LLM_PATH="internlm/internlm2_5-7b-chat"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (InternLM2.5 backbone, verified diff=11.8)"
            ;;
        llava-mistral)
            MERGE_BASE_LLM_PATH="mistralai/Mistral-7B-v0.1"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Mistral backbone)"
            ;;
        llava-llama3)
            MERGE_BASE_LLM_PATH="modern_vlms/pretrained/llama3-8b-from-llava"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (LLaMA-3-8B backbone, BRV main model)"
            ;;
        llava-liuhaotian)
            MERGE_BASE_LLM_PATH="NousResearch/Llama-2-7b-hf"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (LLaMA-2-7B backbone)"
            ;;
    esac
fi
if [[ -z "$MERGE_MATH_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)
            MERGE_MATH_LLM_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (R1 reasoning distilled into Qwen2.5)"
            ;;
        llava-ov)
            MERGE_MATH_LLM_PATH="Qwen/Qwen2-Math-7B"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH"
            ;;
        internvl)
            MERGE_MATH_LLM_PATH="internlm/internlm2-math-plus-7b"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH"
            ;;
        llava-mistral)
            MERGE_MATH_LLM_PATH="hkust-nlp/dart-math-mistral-7b-uniform"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (Dart-Math, BRV best)"
            ;;
        llava-llama3)
            MERGE_MATH_LLM_PATH="modern_vlms/pretrained/dart-math-llama3-8b-prop2diff"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (Dart-Math-LLaMA3, BRV main donor)"
            ;;
        llava-liuhaotian)
            MERGE_MATH_LLM_PATH="meta-math/MetaMath-7B-V1.0"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (MetaMath, strongest LLaMA-2-7B math SFT)"
            ;;
    esac
fi

# Resolve PMBT label dir (prefer full-mode)
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    MERGE_LABEL_DIR="$_PM_MODE"
else
    echo "  ERROR: No PMBT labels found. Run steps 1-4 first."
    exit 1
fi
echo "  PMBT labels: $MERGE_LABEL_DIR"

# Build optional flags
_MERGE_EXTRA="--save_model"
$MERGE_SAVE_MODEL || _MERGE_EXTRA="$_MERGE_EXTRA --no_save_model"
$MERGE_INCLUDE_UNIFORM && _MERGE_EXTRA="$_MERGE_EXTRA --include_uniform_baseline"
$MERGE_INCLUDE_MULTIMODAL && _MERGE_EXTRA="$_MERGE_EXTRA --include_multimodal"
$MERGE_INCLUDE_VISUAL_ONLY && _MERGE_EXTRA="$_MERGE_EXTRA --include_visual_only"
$MERGE_INCLUDE_VISUAL_MULTI && _MERGE_EXTRA="$_MERGE_EXTRA --include_visual_multi"
$MERGE_INCLUDE_RANDOM && _MERGE_EXTRA="$_MERGE_EXTRA --include_random"
$MERGE_INCLUDE_MULTIMODAL_ONLY && _MERGE_EXTRA="$_MERGE_EXTRA --include_multimodal_only"
[[ "$MODE" == "test" ]] && _MERGE_EXTRA="$_MERGE_EXTRA --n_pope_questions 50"

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JN16="${JN10/10/16}"
JOB_NAME="${JN16}_g${GMEM_TAG}"
RESULT_FILE="$MERGE_OUT_DIR/text_inject/merge_results.json"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    MERGE_CMD="$PYTHON $MERGE_SCRIPT \
        --method text_inject \
        --vlm_path $MODEL_PATH \
        --base_llm_path $MERGE_BASE_LLM_PATH \
        --math_llm_path $MERGE_MATH_LLM_PATH \
        --label_dir $MERGE_LABEL_DIR \
        --model_type $MODEL_TYPE \
        --n_layers $N_LAYERS \
        --output_dir $MERGE_OUT_DIR/text_inject \
        --lambda_sweep $MERGE_LAMBDA_SWEEP \
        --pope_path $POPE_PATH \
        --pope_img_dir $POPE_IMG_DIR \
        --eval_pope \
        $_MERGE_EXTRA"

    echo "  → $JOB_NAME → $MERGE_OUT_DIR/text_inject"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

    if $LOCAL; then
        (cd "$WORK_DIR" && $MERGE_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        # Request 96GB CPU RAM — loading 3 × 7B models simultaneously peaks ~84GB
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=98304] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $MERGE_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 16 (text_inject)

# ═══════════════════════════════════════════════════════════════
# STEP 24 (visual_transplant): Cross-VLM Visual Neuron Transplant (Method 5)
#   Injects visual task vector from a better VLM ONLY into visual neurons.
#   Symmetric complement to Method 1.
#
#   Verified donor direction (step 10 contrastive POPE, same protocol):
#     DONOR  → LLaVA-OV    (76  hallucinated / 999 questions = 7.6%)
#     TARGET → Qwen2.5-VL  (411 hallucinated / 999 questions = 41.1%)
#
#   Run as:
#     bash run_pipeline.sh --step 24 --model-type qwen2vl --gmem 80
#
#   Donor path is auto-resolved from the local HF cache (LLaVA-OV).
#   Override with --merge-donor-vlm <path> if needed.
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "visual_transplant" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 24: Cross-VLM Visual Neuron Transplant (Method 5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/17-visual-transplant"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

MERGE_OUT_DIR="results/17-weight-merge/${MODE_DIR}/${MODEL_NAME}"

if [[ -z "$MERGE_DONOR_VLM_PATH" ]]; then
    # Auto-resolve LLaVA-OV as the verified donor (7.6% halluc vs Qwen 41.1%)
    _DONOR_SNAP="$WORK_DIR/.cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots"
    if [[ -d "$_DONOR_SNAP" ]]; then
        _DONOR_HASH=$(ls "$_DONOR_SNAP" | head -1)
        MERGE_DONOR_VLM_PATH="$_DONOR_SNAP/$_DONOR_HASH"
        echo "  [auto] Donor VLM (LLaVA-OV): $MERGE_DONOR_VLM_PATH"
    else
        MERGE_DONOR_VLM_PATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"
        echo "  [auto] Donor VLM (LLaVA-OV): $MERGE_DONOR_VLM_PATH (Hub fallback)"
    fi
fi

# Resolve base LLM (same backbone as both VLMs)
if [[ -z "$MERGE_BASE_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)
            MERGE_BASE_LLM_PATH="Qwen/Qwen2.5-7B"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (verified diff=81.8)"
            ;;
        llava-ov)
            MERGE_BASE_LLM_PATH="Qwen/Qwen2-7B"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (verified diff=12.7)"
            ;;
        internvl)
            MERGE_BASE_LLM_PATH="internlm/internlm2_5-7b-chat"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (verified diff=11.8)"
            ;;
        llava-mistral)
            MERGE_BASE_LLM_PATH="mistralai/Mistral-7B-v0.1"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH"
            ;;
        llava-llama3)
            MERGE_BASE_LLM_PATH="modern_vlms/pretrained/llama3-8b-from-llava"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH"
            ;;
        *)
            echo "  ERROR: Set --merge-base-llm explicitly for this model type."
            exit 1
            ;;
    esac
fi

# Resolve PMBT label dirs
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then MERGE_LABEL_DIR="$_PM_MODE"
else echo "  ERROR: No PMBT labels found. Run steps 1-4 first."; exit 1; fi
echo "  Target PMBT labels: $MERGE_LABEL_DIR"

# ── Auto-resolve donor PMBT label dir ──────────────────────────
# If --merge-donor-labels was not passed explicitly, infer the donor model
# name from the donor VLM path and look for its step-3/4 classification output.
# Matching is substring-based so both HF Hub IDs and local paths work.
if [[ -z "$MERGE_DONOR_LABEL_DIR" ]]; then
    _donor_path_lower="${MERGE_DONOR_VLM_PATH,,}"  # lowercase for matching
    if [[ "$_donor_path_lower" == *"qwen2.5"* && "$_donor_path_lower" == *"vl"* ]]; then
        _donor_model_name="qwen2.5-vl-7b"
    elif [[ "$_donor_path_lower" == *"llava"* && "$_donor_path_lower" == *"onevision"* ]]; then
        _donor_model_name="llava-onevision-7b"
    elif [[ "$_donor_path_lower" == *"internvl"* ]]; then
        _donor_model_name="internvl2.5-8b"
    elif [[ "$_donor_path_lower" == *"qwen2"* && "$_donor_path_lower" == *"vl"* ]]; then
        _donor_model_name="qwen2.5-vl-7b"  # fallback for Qwen2-VL variants
    else
        _donor_model_name=""
    fi

    if [[ -n "$_donor_model_name" ]]; then
        _donor_full="results/3-classify/full/$_donor_model_name/llm_permutation"
        _donor_mode="$OUTPUT_DIR/$_donor_model_name/llm_permutation"
        if [[ -d "$_donor_full" ]]; then
            MERGE_DONOR_LABEL_DIR="$_donor_full"
            echo "  [auto] Donor PMBT labels: $MERGE_DONOR_LABEL_DIR"
        elif [[ -d "$_donor_mode" ]]; then
            MERGE_DONOR_LABEL_DIR="$_donor_mode"
            echo "  [auto] Donor PMBT labels: $MERGE_DONOR_LABEL_DIR"
        else
            echo "  ERROR: Donor PMBT labels not found for '$_donor_model_name'."
            echo "         Looked in: $_donor_full"
            echo "         Run steps 1-4 for the donor model first:"
            echo "           bash run_pipeline.sh --step all --model-type qwen2vl"
            echo "         Or pass --merge-donor-labels /path/to/labels explicitly."
            exit 1
        fi
    else
        echo "  ERROR: Could not infer donor model name from path: $MERGE_DONOR_VLM_PATH"
        echo "         Pass --merge-donor-labels explicitly."
        exit 1
    fi
fi

# Build optional flags
_MERGE_EXTRA="--save_model"
$MERGE_SAVE_MODEL || _MERGE_EXTRA="$_MERGE_EXTRA --no_save_model"
$MERGE_INCLUDE_UNIFORM && _MERGE_EXTRA="$_MERGE_EXTRA --include_uniform_baseline"
$MERGE_INCLUDE_MULTIMODAL && _MERGE_EXTRA="$_MERGE_EXTRA --include_multimodal"
$MERGE_INCLUDE_VISUAL_ONLY && _MERGE_EXTRA="$_MERGE_EXTRA --include_visual_only"
$MERGE_INCLUDE_VISUAL_MULTI && _MERGE_EXTRA="$_MERGE_EXTRA --include_visual_multi"
$MERGE_INCLUDE_RANDOM && _MERGE_EXTRA="$_MERGE_EXTRA --include_random"
$MERGE_INCLUDE_MULTIMODAL_ONLY && _MERGE_EXTRA="$_MERGE_EXTRA --include_multimodal_only"
[[ -n "$MERGE_DONOR_LABEL_DIR" ]] && _MERGE_EXTRA="$_MERGE_EXTRA --donor_label_dir $MERGE_DONOR_LABEL_DIR"
[[ "$MODE" == "test" ]] && _MERGE_EXTRA="$_MERGE_EXTRA --n_pope_questions 50"

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JN17="${JN10/10/17}"
JOB_NAME="${JN17}_g${GMEM_TAG}"
RESULT_FILE="$MERGE_OUT_DIR/visual_transplant/merge_results.json"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    MERGE_CMD="$PYTHON $MERGE_SCRIPT \
        --method visual_transplant \
        --vlm_path $MODEL_PATH \
        --base_llm_path $MERGE_BASE_LLM_PATH \
        --donor_vlm_path $MERGE_DONOR_VLM_PATH \
        --label_dir $MERGE_LABEL_DIR \
        --model_type $MODEL_TYPE \
        --n_layers $N_LAYERS \
        --output_dir $MERGE_OUT_DIR/visual_transplant \
        --lambda_sweep $MERGE_LAMBDA_SWEEP \
        --pope_path $POPE_PATH \
        --pope_img_dir $POPE_IMG_DIR \
        --eval_pope \
        $_MERGE_EXTRA"

    echo "  → $JOB_NAME → $MERGE_OUT_DIR/visual_transplant"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

    if $LOCAL; then
        (cd "$WORK_DIR" && $MERGE_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        # Request 96GB CPU RAM — loading 3 × 7B models simultaneously peaks ~84GB
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=98304] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $MERGE_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 24 (visual_transplant)

# ═══════════════════════════════════════════════════════════════
# STEP 18 (compose_merge): Compose text injection + SRF into single models
#
#   Produces two composed models:
#     A) Step 16 (text inject) + Step 24 (SRF)  = Layer 1a + 1c
#     B) Step 23 (SNRF)       + Step 24 (SRF)  = Layer 1b + 1c
#
#   Because step 16/23 touch ONLY text neurons and step 24 touches ONLY
#   visual neurons, the two edits are disjoint and can be composed without
#   conflict. The composed model should be Pareto-better than either
#   individual merge on both reasoning (MathVista) and hallucination (POPE).
#
#   Required: step 24 (SRF) must have been run first to produce hallucination_modes.pt
#             step 16 and/or step 23 must have been run first
#
#   Run as:
#     bash run_pipeline.sh --step 18 --model-type qwen2vl --gmem 80
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "compose_merge" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 18: Compose text edit + visual SRF"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/18-compose-merge"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# ── Check SRF outputs exist ──
SRF_MODES_PT="results/24-srf/${MODE_DIR}/${MODEL_NAME}/hallucination_modes.pt"
if [[ ! -f "$WORK_DIR/$SRF_MODES_PT" ]]; then
    echo "  ERROR: No SRF hallucination modes found at $SRF_MODES_PT"
    echo "         Run step 24 (SRF) first."
    exit 1
fi

# ── Resolve PMBT label dir ──
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    MERGE_LABEL_DIR="$_PM_MODE"
else
    echo "  ERROR: No PMBT labels found."
    exit 1
fi

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
COMPOSE_BASE_OUT="results/18-compose-merge/${MODE_DIR}/${MODEL_NAME}"
mkdir -p "$WORK_DIR/$COMPOSE_BASE_OUT"

# ── Composition A: Step 16 (Layer 1a: text inject) + Step 24 (Layer 1c: SRF) ──
BEST_LAMBDA_JSON="results/22-select-lambda/${MODE_DIR}/${MODEL_NAME}/lambda_summary.json"
if [[ -n "${MERGE_LAMBDA16:-}" ]]; then
    # User explicitly passed --merge-lambda16
    COMPOSE_LAMBDA16="$MERGE_LAMBDA16"
    echo "  [manual] Lambda16=$COMPOSE_LAMBDA16 (from --merge-lambda16)"
    L1A_MODEL="results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${COMPOSE_LAMBDA16}/model"
elif [[ -f "$WORK_DIR/$BEST_LAMBDA_JSON" ]]; then
    # Read best tag (e.g. "text_l0.15") and extract lambda → build directory name
    BEST_TAG=$(python3 -c "import json; d=json.load(open('$WORK_DIR/$BEST_LAMBDA_JSON')); print(d.get('best_brv', d.get('best', '')))" 2>/dev/null)
    COMPOSE_LAMBDA16=$(echo "$BEST_TAG" | grep -oP '(?<=_l)\d+\.\d+' || true)
    if [[ -z "$COMPOSE_LAMBDA16" ]]; then
        echo "  [warn] Could not parse lambda from step 22 tag '$BEST_TAG', using default 0.1"
        COMPOSE_LAMBDA16="0.1"
    fi
    L1A_MODEL="results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${COMPOSE_LAMBDA16}/model"
else
    # Fallback: use default lambda
    COMPOSE_LAMBDA16="0.1"
    L1A_MODEL="results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${COMPOSE_LAMBDA16}/model"
fi

if [[ -d "$WORK_DIR/$L1A_MODEL" ]]; then
    L1A_1C_OUT="$COMPOSE_BASE_OUT/composed_1a_1c"
    JOB_A="18_${SHORT_MODEL}_1a1c"

    if [[ -d "$WORK_DIR/$L1A_1C_OUT/model" ]]; then
        echo "  [skip] Composition A (1a+1c) — already exists at $L1A_1C_OUT"
    elif is_job_active "$JOB_A" 2>/dev/null; then
        echo "  [skip] $JOB_A — already active"
    else
        echo "  [A] Layer 1a + 1c: $L1A_MODEL + SRF"
        COMPOSE_A_CMD="$PYTHON $SRF_SCRIPT \
            --stage edit \
            --vlm_path $WORK_DIR/$L1A_MODEL \
            --model_type $MODEL_TYPE \
            --label_dir $MERGE_LABEL_DIR \
            --eigenvecs_dir results/24-srf/${MODE_DIR}/${MODEL_NAME} \
            --output_dir $L1A_1C_OUT \
            --alpha 0.5 \
            --n_modes 10 \
            --min_layer_pct 0.5 \
            --save_model"

        if $LOCAL; then
            (cd "$WORK_DIR" && $COMPOSE_A_CMD) \
                2>&1 | tee "$STEP_LOG_DIR/${JOB_A}${LOG_SUFFIX}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_A" \
                -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}" \
                -R "rusage[mem=98304] order[-gpu_maxfactor]" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_A}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_A}${LOG_SUFFIX}.err" \
                "cd $WORK_DIR && $COMPOSE_A_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    fi
else
    echo "  [skip] Composition A (1a+1c) — step 16 model not found at $L1A_MODEL"
    echo "         Run step 16 first."
fi

# ── Composition B: Step 23 (Layer 1b: SNRF) + Step 24 (Layer 1c: SRF) ──
L1B_MODEL=$(find results/23-snrf/${MODE_DIR}/${MODEL_NAME}/ -name "pytorch_model.bin" -path "*/snrf_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
if [[ -z "$L1B_MODEL" ]]; then
    # Try safetensors
    L1B_MODEL=$(find results/23-snrf/${MODE_DIR}/${MODEL_NAME}/ -name "model.safetensors" -path "*/snrf_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
fi

if [[ -n "$L1B_MODEL" ]]; then
    L1B_1C_OUT="$COMPOSE_BASE_OUT/composed_1b_1c"
    JOB_B="18_${SHORT_MODEL}_1b1c"

    if [[ -d "$WORK_DIR/$L1B_1C_OUT/model" ]]; then
        echo "  [skip] Composition B (1b+1c) — already exists at $L1B_1C_OUT"
    elif is_job_active "$JOB_B" 2>/dev/null; then
        echo "  [skip] $JOB_B — already active"
    else
        echo "  [B] Layer 1b + 1c: $L1B_MODEL + SRF"
        COMPOSE_B_CMD="$PYTHON $SRF_SCRIPT \
            --stage edit \
            --vlm_path $L1B_MODEL \
            --model_type $MODEL_TYPE \
            --label_dir $MERGE_LABEL_DIR \
            --eigenvecs_dir results/24-srf/${MODE_DIR}/${MODEL_NAME} \
            --output_dir $L1B_1C_OUT \
            --alpha 0.5 \
            --n_modes 10 \
            --min_layer_pct 0.5 \
            --save_model"

        if $LOCAL; then
            (cd "$WORK_DIR" && $COMPOSE_B_CMD) \
                2>&1 | tee "$STEP_LOG_DIR/${JOB_B}${LOG_SUFFIX}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_B" \
                -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}" \
                -R "rusage[mem=98304] order[-gpu_maxfactor]" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_B}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_B}${LOG_SUFFIX}.err" \
                "cd $WORK_DIR && $COMPOSE_B_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    fi
else
    echo "  [skip] Composition B (1b+1c) — step 23 SNRF model not found"
    echo "         Run step 23 first."
fi

fi  # end step 18 (compose_merge)


# ═══════════════════════════════════════════════════════════════
# STEP 19 (evaluate): VLMEvalKit benchmark evaluation
#
#   Evaluates model variants using VLMEvalKit:
#     A) step 16 output  → reasoning benchmarks (MathVista, MathVerse) PRIMARY
#                          + POPE as guard rail
#     B) step 24 output  → hallucination benchmarks (POPE) PRIMARY
#                          + MathVista as guard rail
#     C) step 18 output  → BOTH: expects improvement on both axes simultaneously
#        - composed_1a_1c (step 16 + step 24)
#        - composed_1b_1c (step 23 + step 24)
#        This is the Pareto dominance result.
#
#   All are compared against the unmerged baseline model.
#
#   Benchmarks:
#     MathVista_MINI        — visual math reasoning (primary for A, C)
#     MathVerse_MINI_Vision_Only — diagram math, vision-only mode (primary for A, C)
#     MathVision            — competition-level math reasoning (primary for A, C)
#     DynaMath              — dynamic math reasoning (primary for A, C)
#     MMStar                — general multimodal reasoning (guard rail; matches BRV eval suite)
#     POPE                  — object existence hallucination (primary for B, C; guard rail for A)
#     HallusionBench        — visual illusion + language hallucination (primary for B, C; guard rail for A)
#
# ═══════════════════════════════════════════════════════════════
# STEP 19 (evaluate): VLMEvalKit benchmark evaluation
#
#   Evaluates selected models against 7 benchmarks.
#   Which models to evaluate is controlled by --eval-which:
#     baseline    — original VLM (always included)
#     16          — step 16 text_inject only (PMBT text neurons)
#     text_multi  — text + multimodal neurons (broader PMBT mask)
#     24          — step 24 visual_transplant only
#     18          — step 18 composed (text_inject + visual_transplant)
#     uniform     — BRV-style uniform merge (no PMBT mask, all neurons)
#
#   Default: --eval-which "baseline 16 text_multi uniform"
#   Example: --eval-which "baseline 16 text_multi 18 uniform"  (full suite)
#
#   Auto-converts merged_state_dict.pt → full HF checkpoint if needed.
#   Auto-registers models in VLMEvalKit config.py.
#
#   Run as:
#     bash run_pipeline.sh --step 19 --model-type qwen2vl --gmem 40
#     bash run_pipeline.sh --step 19 --model-type qwen2vl --eval-which "baseline 16"
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "evaluate" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 19: VLMEvalKit benchmark evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/19-evaluate"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

EVAL_OUT_DIR="results/19-evaluate/${MODE_DIR}/${MODEL_NAME}"
VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
fi
VLMEVAL_BENCHMARKS="MathVista_MINI MathVerse_MINI_Vision_Only MathVision DynaMath MMStar POPE HallusionBench MMMU_DEV_VAL MMMU_Pro MME ScienceQA_IMG LLaVABench A-OKVQA CHAIR"
# Per-method benchmarks matching original papers:
#   SNRF (Cui et al. 2602.19058): MathVista, MMMU_DEV_VAL, MMMU_PRO, MME, POPE, ScienceQA_IMG
#   SRF  (Ali et al. 2511.12220): POPE, LLaVABench, A-OKVQA, CHAIR
#   All methods now evaluated on unified superset above.
VLMEVAL_CONFIG="$WORK_DIR/modern_vlms/VLMEvalKit/vlmeval/config.py"

# ── Auto-detect lambda from step 22 if not manually set ─────────────────────
LAMBDA_JSON="$WORK_DIR/results/22-select-lambda/${MODE_DIR}/${MODEL_NAME}/lambda_summary.json"
if [[ -z "${MERGE_LAMBDA16:-}" ]] && [[ -f "$LAMBDA_JSON" ]]; then
    # Extract best_pmbt tag (e.g. "text_l0.15" or "tmulti_l0.1")
    BEST_TAG=$(python3 -c "import json; d=json.load(open('$LAMBDA_JSON')); print(d.get('best_pmbt',''))" 2>/dev/null)
    if [[ -n "$BEST_TAG" ]]; then
        # Extract lambda value from tag: "text_l0.15" → "0.15", "tmulti_l0.1" → "0.1"
        AUTO_LAMBDA=$(echo "$BEST_TAG" | grep -oP '(?<=_l)\d+\.\d+' || true)
        if [[ -n "$AUTO_LAMBDA" ]]; then
            COMPOSE_LAMBDA16="$AUTO_LAMBDA"
            echo "  [auto] Lambda16=$COMPOSE_LAMBDA16 (from step 22: $BEST_TAG)"
        else
            COMPOSE_LAMBDA16="0.1"
            echo "  [warn] Could not parse lambda from step 22 tag '$BEST_TAG', using default 0.1"
        fi
    else
        COMPOSE_LAMBDA16="0.1"
        echo "  [warn] No best_pmbt in $LAMBDA_JSON, using default lambda 0.1"
    fi
else
    COMPOSE_LAMBDA16="${MERGE_LAMBDA16:-0.1}"
    if [[ -n "${MERGE_LAMBDA16:-}" ]]; then
        echo "  [manual] Lambda16=$COMPOSE_LAMBDA16 (from --merge-lambda16)"
    elif [[ ! -f "$LAMBDA_JSON" ]]; then
        echo "  [warn] No step 22 results found at $LAMBDA_JSON, using default lambda 0.1"
    fi
fi
COMPOSE_LAMBDA17="${MERGE_LAMBDA17:-0.1}"

# Which models to evaluate (override with --eval-which "baseline 16 17 18")
echo "  Evaluating: $EVAL_WHICH"
echo "  Benchmarks: $VLMEVAL_BENCHMARKS"
echo "  Lambda16=$COMPOSE_LAMBDA16  Lambda17=$COMPOSE_LAMBDA17"

# ── Helper: convert merged_state_dict.pt → HF checkpoint ─────────────────────
convert_if_needed() {
    local model_dir="$1"
    local model_label="$2"

    # Already a full HF checkpoint?
    if [[ -f "$model_dir/config.json" ]]; then
        echo "  ✓ $model_label: HF checkpoint exists"
        return 0
    fi

    # Has raw state dict?
    if [[ ! -f "$model_dir/merged_state_dict.pt" ]]; then
        echo "  ERROR: $model_label: no config.json or merged_state_dict.pt in $model_dir"
        return 1
    fi

    echo "  ⏳ $model_label: converting merged_state_dict.pt → HF checkpoint..."
    CONVERT_JOB="19_convert_${SHORT_MODEL}_${model_label}"

    # Build model-type-specific conversion command
    # InternVL uses its own venv for model loading
    local CONVERT_PYTHON="$VLMEVAL_PYTHON"
    if [[ "$MODEL_TYPE" == "internvl" ]]; then
        CONVERT_PYTHON="$PYTHON"  # uses the InternVL venv set earlier
    fi

    case "$MODEL_TYPE" in
        internvl)
            CONVERT_CMD="$CONVERT_PYTHON -c \"
import torch, shutil, glob, os
from transformers import AutoModel, AutoTokenizer
base = '$MODEL_PATH'
merged = '$model_dir'
print('Loading base model...')
model = AutoModel.from_pretrained(base, torch_dtype='auto', trust_remote_code=True, low_cpu_mem_usage=True)
print('Loading merged weights...')
state = torch.load(f'{merged}/merged_state_dict.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
print('Saving HF checkpoint...')
model.save_pretrained(merged)
tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
tokenizer.save_pretrained(merged)
# Copy custom modeling files (modeling_internvl_chat.py etc.) needed for trust_remote_code
for f in glob.glob(os.path.join(base, '*.py')):
    dst = os.path.join(merged, os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy2(f, dst)
        print(f'  Copied {os.path.basename(f)}')
print('Done')
\""
            ;;
        *)
            CONVERT_CMD="$VLMEVAL_PYTHON -c \"
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
base = '$MODEL_PATH'
merged = '$model_dir'
print('Loading base model...')
model = AutoModelForVision2Seq.from_pretrained(base, torch_dtype='auto', attn_implementation='sdpa')
print('Loading merged weights...')
state = torch.load(f'{merged}/merged_state_dict.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
print('Saving HF checkpoint...')
model.save_pretrained(merged)
processor = AutoProcessor.from_pretrained(base)
processor.save_pretrained(merged)
print('Done')
\""
            ;;
    esac

    rm -f "$WORK_DIR/$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}."{log,err}

    if $LOCAL; then
        (cd "$WORK_DIR" && eval $CONVERT_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$CONVERT_JOB" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=65536] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $CONVERT_CMD"
        SUBMITTED=$((SUBMITTED + 1))
        echo "  → submitted $CONVERT_JOB (eval jobs will depend on it)"
    fi
    # Return the job name so eval jobs can depend on it
    LAST_CONVERT_JOB="$CONVERT_JOB"
    return 0
}

# ── Helper: register a model in VLMEvalKit config.py ─────────────────────────
register_model() {
    local vlm_name="$1"
    local vlm_path="$2"
    if ! grep -q "\"$vlm_name\"" "$VLMEVAL_CONFIG" 2>/dev/null; then
        # Build the correct partial() call based on model type
        case "$MODEL_TYPE" in
            qwen2vl)
                ENTRY_CLASS="Qwen2VLChat"
                ENTRY_EXTRA=", min_pixels=1280*28*28, max_pixels=16384*28*28"
                ANCHOR='"Qwen2-VL-7B-Instruct": partial('
                ;;
            llava-ov|llava_ov)
                ENTRY_CLASS="LLaVA_OneVision_HF"
                ENTRY_EXTRA=""
                ANCHOR='"llava-onevision-qwen2-7b-ov-hf": partial('
                ;;
            llava-mistral)
                ENTRY_CLASS="LLaVA_Next"
                ENTRY_EXTRA=""
                ANCHOR='"llava_next_mistral_7b": partial('
                ;;
            llava-llama3)
                ENTRY_CLASS="LLaVA_Next"
                ENTRY_EXTRA=""
                ANCHOR='"llava_next_mistral_7b": partial('
                ;;
            internvl)
                ENTRY_CLASS="InternVLChat"
                ENTRY_EXTRA=', version="V2.0", max_new_tokens=512'
                ANCHOR='"InternVL2_5-8B": partial('
                ;;
            *)
                echo "  WARNING: Unknown model type $MODEL_TYPE for VLMEvalKit registration"
                return 1
                ;;
        esac

        $VLMEVAL_PYTHON -c "
cfg_path = '$VLMEVAL_CONFIG'
with open(cfg_path) as f:
    cfg = f.read()
entry = '    \"$vlm_name\": partial($ENTRY_CLASS, model_path=\"$vlm_path\"$ENTRY_EXTRA),\n'
anchor = '$ANCHOR'
if anchor in cfg:
    cfg = cfg.replace(anchor, entry + '    ' + anchor)
    with open(cfg_path, 'w') as f:
        f.write(cfg)
    print('  Registered: $vlm_name')
else:
    print('  WARNING: anchor not found in config.py: ' + anchor)
"
    else
        echo "  Already registered: $vlm_name"
    fi
}

# ── Build list of models to evaluate ─────────────────────────────────────────
declare -A EVAL_MAP          # tag → VLMEvalKit model name
declare -A EVAL_PATHS        # tag → model path
declare -A EVAL_CONVERT_DEP  # tag → conversion job name (empty if no conversion needed)
EVAL_TAGS=()

for WHICH in $EVAL_WHICH; do
    LAST_CONVERT_JOB=""

    case "$WHICH" in
        baseline)
            TAG="baseline"
            VLM_NAME="${MODEL_NAME}_baseline"
            VLM_PATH="$MODEL_PATH"
            ;;
        16|step16|text_inject)
            TAG="step16_text_inject_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: step 16 model not found at $VLM_PATH"
                echo "         Run step 16 first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        text_multi)
            TAG="text_multi_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_text_multi_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: text_multi model not found at $VLM_PATH"
                echo "         Run step 16 with --include_multimodal first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        visual_only)
            TAG="visual_only_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_visual_only_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: visual_only model not found at $VLM_PATH"
                echo "         Run step 16 with --include_visual_only first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        visual_multi)
            TAG="visual_multi_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_visual_multi_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: visual_multi model not found at $VLM_PATH"
                echo "         Run step 16 with --include_visual_multi first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        random)
            TAG="random_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_random_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: random model not found at $VLM_PATH"
                echo "         Run step 16 with --include_random first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        multimodal_only)
            TAG="multi_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_multimodal_only_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: multimodal_only model not found at $VLM_PATH"
                echo "         Run step 16 with --include_multimodal_only first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        snrf)
            TAG="snrf_r16_b0p5"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/23-snrf/${MODE_DIR}/${MODEL_NAME}/snrf_r16_b0p5"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: SNRF model not found at $VLM_PATH"
                echo "         Run step 23 first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        snrf_random)
            TAG="snrf_random_r16_b0p5"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/23-snrf/${MODE_DIR}/${MODEL_NAME}/snrf_random_r16_b0p5"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: SNRF random model not found at $VLM_PATH"
                echo "         Run step 23 with --include_random first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        srf)
            TAG="srf_a0.5_m10"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/24-srf/${MODE_DIR}/${MODEL_NAME}/srf_a0.5_m10"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: SRF model not found at $VLM_PATH"
                echo "         Run step 24 first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        srf_random)
            TAG="srf_random_a0.5_m10"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/24-srf/${MODE_DIR}/${MODEL_NAME}/srf_random_a0.5_m10"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: SRF random model not found at $VLM_PATH"
                echo "         Run step 24 with --include_random first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        24|step24|visual_transplant)
            TAG="step17_visual_transplant_l${COMPOSE_LAMBDA17}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/17-weight-merge/${MODE_DIR}/${MODEL_NAME}/visual_transplant/visual_transplant_lambda${COMPOSE_LAMBDA17}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: step 24 visual_transplant model not found at $VLM_PATH"
                echo "         Run step 24 first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        vis_multi)
            TAG="vis_multi_l${COMPOSE_LAMBDA17}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/17-weight-merge/${MODE_DIR}/${MODEL_NAME}/visual_transplant/visual_transplant_vis_multi_lambda${COMPOSE_LAMBDA17}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: vis_multi model not found at $VLM_PATH"
                echo "         Run step 24 with --include_multimodal first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        17_text_only)
            TAG="17_text_only_l${COMPOSE_LAMBDA17}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/17-weight-merge/${MODE_DIR}/${MODEL_NAME}/visual_transplant/visual_transplant_text_only_lambda${COMPOSE_LAMBDA17}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: 17_text_only model not found at $VLM_PATH"
                echo "         Run step 24 with --include_visual_only first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        17_uniform)
            TAG="17_uniform_l${COMPOSE_LAMBDA17}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/17-weight-merge/${MODE_DIR}/${MODEL_NAME}/visual_transplant/visual_transplant_uniform_lambda${COMPOSE_LAMBDA17}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: 17_uniform model not found at $VLM_PATH"
                echo "         Run step 24 with --include_uniform_baseline first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        18|step18|composed|1a1c|composed_1a_1c)
            TAG="step18_composed_1a_1c"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/18-compose-merge/${MODE_DIR}/${MODEL_NAME}/composed_1a_1c/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: step 18 composed_1a_1c model not found at $VLM_PATH"
                echo "         Run step 18 first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        1b1c|composed_1b_1c)
            TAG="step18_composed_1b_1c"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/18-compose-merge/${MODE_DIR}/${MODEL_NAME}/composed_1b_1c/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: step 18 composed_1b_1c model not found at $VLM_PATH"
                echo "         Run step 18 first (requires step 23 + step 24)."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        uniform|brv)
            TAG="uniform_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_uniform_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: uniform model not found at $VLM_PATH"
                echo "         Run step 16 with --merge-uniform first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        25|compose_layer1|composed_layer1)
            # Find the composed Layer 1a+1c model
            COMP_MODEL=$(find results/25-compose-layer1/${MODE_DIR}/${MODEL_NAME}/ -name "pytorch_model.bin" -path "*/srf_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
            if [[ -z "$COMP_MODEL" || ! -d "$COMP_MODEL" ]]; then
                echo "  ERROR: Composed Layer 1 model not found. Run step 25 first."
                continue
            fi
            TAG="composed_layer1"
            VLM_NAME="${MODEL_NAME}_composed_layer1"
            VLM_PATH="$WORK_DIR/$COMP_MODEL"
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        *)
            echo "  WARNING: Unknown eval target '$WHICH' — skipping"
            continue
            ;;
    esac

    EVAL_TAGS+=("$TAG")
    EVAL_MAP["$TAG"]="$VLM_NAME"
    EVAL_PATHS["$TAG"]="$VLM_PATH"
    EVAL_CONVERT_DEP["$TAG"]="$LAST_CONVERT_JOB"

    register_model "$VLM_NAME" "$VLM_PATH"
    echo "    $TAG → $VLM_NAME ($VLM_PATH)"
done

# ── Submit one job per model × benchmark ─────────────────────────────────────
for TAG in "${EVAL_TAGS[@]}"; do
    MODEL_EVAL_NAME="${EVAL_MAP[$TAG]}"
    CONVERT_DEP="${EVAL_CONVERT_DEP[$TAG]}"

    for BENCH in $VLMEVAL_BENCHMARKS; do
        JOB_NAME="19_eval_${SHORT_MODEL}_${TAG}_${BENCH}"

        # CHAIR uses eval_chair.py instead of VLMEvalKit
        if [[ "$BENCH" == "CHAIR" ]]; then
            CHAIR_COCO_ANN="${CHAIR_ANN_PATH:-$WORK_DIR/data/annotations/instances_val2014.json}"
            CHAIR_COCO_IMG="${POPE_IMG_DIR:-$WORK_DIR/data/val2014}"
            CHAIR_N="${CHAIR_NUM_IMAGES:-500}"
            if [[ ! -f "$CHAIR_COCO_ANN" ]]; then
                echo "  [skip] $JOB_NAME — COCO annotations not found at $CHAIR_COCO_ANN"
                continue
            fi
            CHAIR_VLM_PATH="${EVAL_PATHS[$TAG]}"
            CHAIR_OUT="$EVAL_OUT_DIR/$TAG/CHAIR"
            if [[ -f "$CHAIR_OUT/chair_summary.json" ]]; then
                echo "  [skip] $JOB_NAME — already done"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            EVAL_CMD="$VLMEVAL_PYTHON $WORK_DIR/code/eval_chair.py \
                --vlm_path $CHAIR_VLM_PATH \
                --model_type $MODEL_TYPE \
                --coco_ann_path $CHAIR_COCO_ANN \
                --coco_img_dir $CHAIR_COCO_IMG \
                --n_images $CHAIR_N \
                --output_dir $CHAIR_OUT"
        else
            EVAL_CMD="$VLMEVAL_PYTHON $WORK_DIR/modern_vlms/VLMEvalKit/run.py \
                --data $BENCH \
                --model $MODEL_EVAL_NAME \
                --work-dir $EVAL_OUT_DIR/$TAG \
                --verbose"
        fi

        echo "  → $JOB_NAME"
        rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
              "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

        if $LOCAL; then
            (cd "$WORK_DIR" && $EVAL_CMD) \
                2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            _GPU_MODE="num=1:gmem=${GPU_GMEM_TIERS[0]}"
            $GPU_EXCLUSIVE && _GPU_MODE="${_GPU_MODE}:mode=exclusive_process" || true
            BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
                -gpu "$_GPU_MODE" \
                -R "rusage[mem=65536] order[-gpu_maxfactor]" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err")
            # Add dependency on conversion job if needed
            if [[ -n "$CONVERT_DEP" ]]; then
                BSUB_ARGS+=(-w "done($CONVERT_DEP)")
            fi
            bsub "${BSUB_ARGS[@]}" "cd $WORK_DIR && $EVAL_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

fi  # end step 19 (evaluate)


# ═══════════════════════════════════════════════════════════════
# STEP 20 (summarize): Collect all step 19 results into one CSV
#
#   Scans results/19-evaluate/<mode>/<model>/ for score CSVs,
#   extracts the primary metric per benchmark, and produces:
#     results/20-summary/<mode>/<model>/eval_summary.csv
#
#   No GPU needed — CPU only.
#
#   Run as:
#     bash run_pipeline.sh --step 20 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "summarize" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 20: Summarize evaluation results (baseline vs composed)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EVAL_DIR="results/19-evaluate/${MODE_DIR}/${MODEL_NAME}"
SUMMARY_DIR="results/20-summary/${MODE_DIR}/${MODEL_NAME}"

$PYTHON $SUMMARIZE_SCRIPT \
    --eval_dir "$EVAL_DIR" \
    --output_dir "$SUMMARY_DIR" \
    --model_name "$MODEL_NAME"

echo "  ✓ Step 20 complete."

fi  # end step 20 (summarize)


# ═══════════════════════════════════════════════════════════════
# STEP 21 (tune_lambda): BRV-style lambda tuning on MathVista
#
#   Evaluates text_inject (and optionally other masks) at multiple
#   lambda values on MathVista_MINI only, then reports the best.
#   Follows BRV methodology: tune on MathVista, apply everywhere.
#
#   Models at each lambda already exist from step 16's lambda sweep.
#   This step converts + evaluates + summarizes.
#
#   Lambdas: --tune-lambdas "0.1 0.15 0.2 0.3"
#   Masks:   --tune-masks "16 text_multi"   (16=text_inject)
#
#   Run as:
#     bash run_pipeline.sh --step 21 --model-type llava-ov --gmem 40
#     bash run_pipeline.sh --step 21 --model-type qwen2vl --gmem 40 --tune-lambdas "0.1 0.15 0.2 0.3"
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "tune_lambda" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 21: BRV-style lambda tuning on MathVista"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/21-tune-lambda"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

TUNE_OUT_DIR="results/21-tune-lambda/${MODE_DIR}/${MODEL_NAME}"
VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
fi
VLMEVAL_CONFIG="$WORK_DIR/modern_vlms/VLMEvalKit/vlmeval/config.py"
TUNE_BENCHMARK="MathVista_MINI"

echo "  Lambdas: $TUNE_LAMBDAS"
echo "  Masks:   $TUNE_MASKS"
echo "  Benchmark: $TUNE_BENCHMARK (BRV methodology)"
echo "  Models from: results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/"
echo ""

# ── Helper: convert merged_state_dict.pt → HF checkpoint ─────────────────────
convert_if_needed() {
    local model_dir="$1"
    local model_label="$2"

    if [[ -f "$model_dir/config.json" ]]; then
        echo "  ✓ $model_label: HF checkpoint exists"
        return 0
    fi

    if [[ ! -f "$model_dir/merged_state_dict.pt" ]]; then
        echo "  ERROR: $model_label: no config.json or merged_state_dict.pt in $model_dir"
        return 1
    fi

    echo "  ⏳ $model_label: converting merged_state_dict.pt → HF checkpoint..."
    CONVERT_JOB="21_convert_${SHORT_MODEL}_${model_label}"

    local CONVERT_PYTHON="$VLMEVAL_PYTHON"
    if [[ "$MODEL_TYPE" == "internvl" ]]; then
        CONVERT_PYTHON="$PYTHON"
    fi

    case "$MODEL_TYPE" in
        internvl)
            CONVERT_CMD="$CONVERT_PYTHON -c \"
import torch, shutil, glob, os
from transformers import AutoModel, AutoTokenizer
base = '$MODEL_PATH'
merged = '$model_dir'
print('Loading base model...')
model = AutoModel.from_pretrained(base, torch_dtype='auto', trust_remote_code=True, low_cpu_mem_usage=True)
print('Loading merged weights...')
state = torch.load(f'{merged}/merged_state_dict.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
print('Saving HF checkpoint...')
model.save_pretrained(merged)
tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
tokenizer.save_pretrained(merged)
# Copy custom modeling files (modeling_internvl_chat.py etc.) needed for trust_remote_code
for f in glob.glob(os.path.join(base, '*.py')):
    dst = os.path.join(merged, os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy2(f, dst)
        print(f'  Copied {os.path.basename(f)}')
print('Done')
\""
            ;;
        *)
            CONVERT_CMD="$VLMEVAL_PYTHON -c \"
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
base = '$MODEL_PATH'
merged = '$model_dir'
print('Loading base model...')
model = AutoModelForVision2Seq.from_pretrained(base, torch_dtype='auto', attn_implementation='sdpa')
print('Loading merged weights...')
state = torch.load(f'{merged}/merged_state_dict.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
print('Saving HF checkpoint...')
model.save_pretrained(merged)
processor = AutoProcessor.from_pretrained(base)
processor.save_pretrained(merged)
print('Done')
\""
            ;;
    esac

    rm -f "$WORK_DIR/$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}."{log,err}

    if $LOCAL; then
        (cd "$WORK_DIR" && eval $CONVERT_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$CONVERT_JOB" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=65536] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${CONVERT_JOB}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $CONVERT_CMD"
        SUBMITTED=$((SUBMITTED + 1))
        echo "  → submitted $CONVERT_JOB"
    fi
    LAST_CONVERT_JOB="$CONVERT_JOB"
    return 0
}

# ── Helper: register a model in VLMEvalKit config.py ─────────────────────────
register_model() {
    local vlm_name="$1"
    local vlm_path="$2"
    if ! grep -q "\"$vlm_name\"" "$VLMEVAL_CONFIG" 2>/dev/null; then
        case "$MODEL_TYPE" in
            qwen2vl)
                ENTRY_CLASS="Qwen2VLChat"
                ENTRY_EXTRA=", min_pixels=1280*28*28, max_pixels=16384*28*28"
                ANCHOR='"Qwen2-VL-7B-Instruct": partial('
                ;;
            llava-ov|llava_ov)
                ENTRY_CLASS="LLaVA_OneVision_HF"
                ENTRY_EXTRA=""
                ANCHOR='"llava-onevision-qwen2-7b-ov-hf": partial('
                ;;
            llava-mistral)
                ENTRY_CLASS="LLaVA_Next"
                ENTRY_EXTRA=""
                ANCHOR='"llava_next_mistral_7b": partial('
                ;;
            llava-llama3)
                ENTRY_CLASS="LLaVA_Next"
                ENTRY_EXTRA=""
                ANCHOR='"llava_next_mistral_7b": partial('
                ;;
            internvl)
                ENTRY_CLASS="InternVLChat"
                ENTRY_EXTRA=', version="V2.0", max_new_tokens=512'
                ANCHOR='"InternVL2_5-8B": partial('
                ;;
            *)
                echo "  WARNING: Unknown model type $MODEL_TYPE for VLMEvalKit registration"
                return 1
                ;;
        esac

        $VLMEVAL_PYTHON -c "
cfg_path = '$VLMEVAL_CONFIG'
with open(cfg_path) as f:
    cfg = f.read()
entry = '    \"$vlm_name\": partial($ENTRY_CLASS, model_path=\"$vlm_path\"$ENTRY_EXTRA),\n'
anchor = '$ANCHOR'
if anchor in cfg:
    cfg = cfg.replace(anchor, entry + '    ' + anchor)
    with open(cfg_path, 'w') as f:
        f.write(cfg)
    print('  Registered: $vlm_name')
else:
    print('  WARNING: anchor not found in config.py: ' + anchor)
"
    else
        echo "  Already registered: $vlm_name"
    fi
}

# ── Map mask tags to directory names ──────────────────────────────────────────
mask_to_dir() {
    local mask="$1"
    local lam="$2"
    case "$mask" in
        16|text_inject)   echo "text_inject_lambda${lam}/model" ;;
        text_multi)       echo "text_inject_text_multi_lambda${lam}/model" ;;
        visual_only)      echo "text_inject_visual_only_lambda${lam}/model" ;;
        visual_multi)     echo "text_inject_visual_multi_lambda${lam}/model" ;;
        uniform)          echo "text_inject_uniform_lambda${lam}/model" ;;
        random)           echo "text_inject_random_lambda${lam}/model" ;;
        multimodal_only)  echo "text_inject_multimodal_only_lambda${lam}/model" ;;
        *)                echo "text_inject_lambda${lam}/model" ;;
    esac
}

mask_to_tag() {
    local mask="$1"
    local lam="$2"
    case "$mask" in
        16|text_inject)   echo "text_l${lam}" ;;
        text_multi)       echo "tmulti_l${lam}" ;;
        visual_only)      echo "visonly_l${lam}" ;;
        visual_multi)     echo "vismulti_l${lam}" ;;
        uniform)          echo "uniform_l${lam}" ;;
        random)           echo "random_l${lam}" ;;
        multimodal_only)  echo "multi_l${lam}" ;;
        *)                echo "text_l${lam}" ;;
    esac
}

# ── Baseline ─────────────────────────────────────────────────────────────────
# Always include baseline for comparison
BASELINE_NAME="${MODEL_NAME}_baseline"
register_model "$BASELINE_NAME" "$MODEL_PATH"

BASELINE_JOB="21_tune_${SHORT_MODEL}_baseline_${TUNE_BENCHMARK}"
EVAL_CMD="$VLMEVAL_PYTHON $WORK_DIR/modern_vlms/VLMEvalKit/run.py \
    --data $TUNE_BENCHMARK \
    --model $BASELINE_NAME \
    --work-dir $TUNE_OUT_DIR/baseline \
    --verbose"

echo "  → $BASELINE_JOB (baseline)"
rm -f "$WORK_DIR/$STEP_LOG_DIR/${BASELINE_JOB}${LOG_SUFFIX}."{log,err}

if $LOCAL; then
    (cd "$WORK_DIR" && $EVAL_CMD) \
        2>&1 | tee "$STEP_LOG_DIR/${BASELINE_JOB}${LOG_SUFFIX}.log"
else
    bsub -q "$QUEUE" -J "$BASELINE_JOB" \
        -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
        -R "rusage[mem=65536] order[-gpu_maxfactor]" \
        -oo "$WORK_DIR/$STEP_LOG_DIR/${BASELINE_JOB}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/$STEP_LOG_DIR/${BASELINE_JOB}${LOG_SUFFIX}.err" \
        "cd $WORK_DIR && $EVAL_CMD"
    SUBMITTED=$((SUBMITTED + 1))
fi

# ── Sweep lambdas × masks ────────────────────────────────────────────────────
MERGE_BASE_DIR="$WORK_DIR/results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject"

for LAM in $TUNE_LAMBDAS; do
    for MASK in $TUNE_MASKS; do
        LAST_CONVERT_JOB=""
        DIR_SUFFIX=$(mask_to_dir "$MASK" "$LAM")
        TAG=$(mask_to_tag "$MASK" "$LAM")
        MODEL_DIR="${MERGE_BASE_DIR}/${DIR_SUFFIX}"

        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "  SKIP: $TAG — model dir not found: $MODEL_DIR"
            continue
        fi

        VLM_NAME="${MODEL_NAME}_tune_${TAG}"
        convert_if_needed "$MODEL_DIR" "$TAG"
        register_model "$VLM_NAME" "$MODEL_DIR"

        # Submit MathVista eval
        JOB_NAME="21_tune_${SHORT_MODEL}_${TAG}_${TUNE_BENCHMARK}"
        EVAL_CMD="$VLMEVAL_PYTHON $WORK_DIR/modern_vlms/VLMEvalKit/run.py \
            --data $TUNE_BENCHMARK \
            --model $VLM_NAME \
            --work-dir $TUNE_OUT_DIR/$TAG \
            --verbose"

        echo "  → $JOB_NAME"
        rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}."{log,err}

        if $LOCAL; then
            (cd "$WORK_DIR" && $EVAL_CMD) \
                2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            _GPU_MODE="num=1:gmem=${GPU_GMEM_TIERS[0]}"
            $GPU_EXCLUSIVE && _GPU_MODE="${_GPU_MODE}:mode=exclusive_process" || true
            BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
                -gpu "$_GPU_MODE" \
                -R "rusage[mem=65536] order[-gpu_maxfactor]" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err")
            if [[ -n "$LAST_CONVERT_JOB" ]]; then
                BSUB_ARGS+=(-w "done($LAST_CONVERT_JOB)")
            fi
            bsub "${BSUB_ARGS[@]}" "cd $WORK_DIR && $EVAL_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

# ── Summary job: collect results and pick best lambda ─────────────────────────
SUMMARY_JOB="21_tune_${SHORT_MODEL}_summary"
SUMMARY_CMD="$VLMEVAL_PYTHON -c \"
import os, json, glob

tune_dir = '$TUNE_OUT_DIR'
benchmark = '$TUNE_BENCHMARK'
model_name = '$MODEL_NAME'

print()
print('=' * 72)
print(f'  LAMBDA TUNING RESULTS: {model_name} on {benchmark}')
print('=' * 72)

results = {}
for d in sorted(os.listdir(tune_dir)):
    score_files = glob.glob(os.path.join(tune_dir, d, '**', f'*{benchmark}*score*.csv'), recursive=True)
    if not score_files:
        score_files = glob.glob(os.path.join(tune_dir, d, '**', f'*{benchmark}*.csv'), recursive=True)
    if not score_files:
        continue
    # Read the CSV — score is in the last column of the 2nd row
    import csv
    for sf in score_files:
        if not os.path.isfile(sf):
            continue
        try:
            with open(sf) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                row = next(reader, None)
                if row and header:
                    score = None
                    for i, h in enumerate(header):
                        if 'overall' in h.lower() or 'acc' in h.lower():
                            try:
                                score = float(row[i])
                            except (ValueError, IndexError):
                                pass
                    if score is None:
                        try:
                            score = float(row[-1])
                        except (ValueError, IndexError):
                            pass
                    if score is not None:
                        results[d] = score
        except (IOError, OSError) as e:
            print(f'  [warn] Could not read {sf}: {e}')

# Print table
hdr_cfg = 'Config'
hdr_mv = 'MathVista'
print(f'{hdr_cfg:<30} {hdr_mv:>10}')
print('-' * 42)
best_tag = None
best_score = -1
brv_best_tag = None
brv_best_score = -1
pmbt_best_tag = None
pmbt_best_score = -1
for tag in sorted(results.keys()):
    score = results[tag]
    marker = ''
    if score > best_score:
        best_score = score
        best_tag = tag
    print(f'{tag:<30} {score:>10.2f}')

print('-' * 42)
if best_tag:
    print(f'Best: {best_tag} = {best_score:.2f}')

# Save results
os.makedirs(tune_dir, exist_ok=True)
with open(os.path.join(tune_dir, 'tune_results.json'), 'w') as f:
    json.dump({'benchmark': benchmark, 'results': results, 'best': best_tag, 'best_score': best_score}, f, indent=2)
print(f'Saved: {tune_dir}/tune_results.json')
print('=' * 72)
\""

echo ""
echo "  [S] $SUMMARY_JOB → collect results and pick best lambda"
rm -f "$WORK_DIR/$STEP_LOG_DIR/${SUMMARY_JOB}${LOG_SUFFIX}."{log,err}

if $LOCAL; then
    (cd "$WORK_DIR" && eval $SUMMARY_CMD) \
        2>&1 | tee "$STEP_LOG_DIR/${SUMMARY_JOB}${LOG_SUFFIX}.log"
else
    # Depend on all tune eval jobs
    DEP_STR=""
    for LAM in $TUNE_LAMBDAS; do
        for MASK in $TUNE_MASKS; do
            TAG=$(mask_to_tag "$MASK" "$LAM")
            JN="21_tune_${SHORT_MODEL}_${TAG}_${TUNE_BENCHMARK}"
            [[ -n "$DEP_STR" ]] && DEP_STR="$DEP_STR && "
            DEP_STR="${DEP_STR}done($JN)"
        done
    done
    # Also depend on baseline
    [[ -n "$DEP_STR" ]] && DEP_STR="$DEP_STR && "
    DEP_STR="${DEP_STR}done($BASELINE_JOB)"

    bsub -q "$QUEUE" -J "$SUMMARY_JOB" \
        -w "$DEP_STR" \
        -R "rusage[mem=8192]" \
        -oo "$WORK_DIR/$STEP_LOG_DIR/${SUMMARY_JOB}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/$STEP_LOG_DIR/${SUMMARY_JOB}${LOG_SUFFIX}.err" \
        "cd $WORK_DIR && $SUMMARY_CMD"
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end step 21 (tune_lambda)


# ═══════════════════════════════════════════════════════════════
# STEP 22 (select_lambda): Combine step 16 POPE + step 21 MathVista → pick best λ
#
#   Reads:
#     - results/16-weight-merge/<mode>/<model>/text_inject/merge_results.json (POPE)
#     - results/21-tune-lambda/<mode>/<model>/*/MathVista*.csv (MathVista)
#
#   Produces:
#     - results/22-select-lambda/<mode>/<model>/lambda_summary.json
#     - Console table with all λ × mask × {POPE, MathVista}
#
#   Run as:
#     bash run_pipeline.sh --step 22 --model-type llava-ov
#     bash run_pipeline.sh --step 22 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "select_lambda" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 22: Select best lambda (POPE from step 16 + MathVista from step 21)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STEP_LOG_DIR="${LOG_DIR}/22-select-lambda"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

SELECT_OUT_DIR="results/22-select-lambda/${MODE_DIR}/${MODEL_NAME}"
mkdir -p "$SELECT_OUT_DIR"

POPE_RESULTS="results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/merge_results.json"
TUNE_DIR="results/21-tune-lambda/${MODE_DIR}/${MODEL_NAME}"

echo "  POPE source:     $POPE_RESULTS"
echo "  MathVista source: $TUNE_DIR"
echo ""

$PYTHON -c "
import os, json, glob, csv

pope_path = '$POPE_RESULTS'
tune_dir  = '$TUNE_DIR'
out_dir   = '$SELECT_OUT_DIR'
model_name = '$MODEL_NAME'

# ── 1. Load POPE scores from step 16 ──────────────────────────
pope_scores = {}  # {run_name: accuracy}
if os.path.exists(pope_path):
    with open(pope_path) as f:
        merge_results = json.load(f)
    for run_name, run_data in merge_results.items():
        if 'pope' in run_data:
            pope_scores[run_name] = run_data['pope']['accuracy']
    print(f'  Loaded {len(pope_scores)} POPE scores from step 16')
else:
    print(f'  WARNING: {pope_path} not found — POPE scores unavailable')

# ── 2. Load MathVista scores from step 21 ─────────────────────
mathvista_scores = {}  # {tag: score}
if os.path.exists(tune_dir):
    for d in sorted(os.listdir(tune_dir)):
        dpath = os.path.join(tune_dir, d)
        if not os.path.isdir(dpath):
            continue
        score_files = glob.glob(os.path.join(dpath, '*MathVista*.csv'))
        if not score_files:
            score_files = glob.glob(os.path.join(dpath, '**', '*MathVista*score*.csv'), recursive=True)
        for sf in score_files:
            try:
                with open(sf) as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    row = next(reader, None)
                    if row and header:
                        score = None
                        for i, h in enumerate(header):
                            if 'overall' in h.lower() or 'acc' in h.lower():
                                try:
                                    score = float(row[i])
                                except (ValueError, IndexError):
                                    pass
                        if score is None:
                            try:
                                score = float(row[-1])
                            except (ValueError, IndexError):
                                pass
                        if score is not None:
                            mathvista_scores[d] = score
            except Exception as e:
                pass
    print(f'  Loaded {len(mathvista_scores)} MathVista scores from step 21')
else:
    print(f'  WARNING: {tune_dir} not found — run step 21 first')

# ── 3. Map step 21 tags to step 16 POPE run names ─────────────
# step 21 tags: text_l0.1, tmulti_l0.2, visonly_l0.3, uniform_l0.3
# step 16 names: text_inject_lambda0.1, text_inject_text_multi_lambda0.2, etc.
def tag_to_pope_name(tag):
    parts = tag.rsplit('_l', 1)
    if len(parts) != 2:
        return None
    mask, lam = parts
    if mask == 'text':
        return f'text_inject_lambda{lam}'
    elif mask == 'tmulti':
        return f'text_inject_text_multi_lambda{lam}'
    elif mask == 'visonly':
        return f'text_inject_visual_only_lambda{lam}'
    elif mask == 'uniform':
        return f'text_inject_uniform_lambda{lam}'
    return None

# ── 4. Get baseline scores ─────────────────────────────────────
baseline_mv = mathvista_scores.get('baseline', None)
# Find baseline POPE from step 16 (run name 'baseline' or look for original)
baseline_pope = pope_scores.get('baseline', None)

# ── 5. Build combined table ────────────────────────────────────
print()
print('=' * 82)
print(f'  LAMBDA SELECTION: {model_name}')
print('=' * 82)
_h = ['Config', 'POPE', 'Δ POPE', 'MathVista', 'Δ MathV']
print(f'{_h[0]:<32} {_h[1]:>8} {_h[2]:>8} {_h[3]:>10} {_h[4]:>8}')
print('-' * 82)

# Baseline row
if baseline_mv is not None or baseline_pope is not None:
    pope_str = f'{baseline_pope*100:.2f}' if baseline_pope else '—'
    mv_str   = f'{baseline_mv:.2f}' if baseline_mv else '—'
    print(f'{\"baseline\":<32} {pope_str:>8} {\"\":>8} {mv_str:>10} {\"\":>8}')
    print('-' * 82)

combined = {}
best_tag = None
best_score = -1
brv_best_tag = None
brv_best_score = -1
pmbt_best_tag = None
pmbt_best_score = -1

for tag in sorted(mathvista_scores.keys()):
    if tag == 'baseline':
        continue
    mv = mathvista_scores[tag]
    pope_name = tag_to_pope_name(tag)
    pope = pope_scores.get(pope_name, None) if pope_name else None

    pope_str = f'{pope*100:.2f}' if pope else '—'
    mv_str   = f'{mv:.2f}'

    delta_pope = ''
    if pope is not None and baseline_pope is not None:
        dp = (pope - baseline_pope) * 100
        delta_pope = f'{dp:+.2f}'

    delta_mv = ''
    if baseline_mv is not None:
        dm = mv - baseline_mv
        delta_mv = f'{dm:+.2f}'

    print(f'{tag:<32} {pope_str:>8} {delta_pope:>8} {mv_str:>10} {delta_mv:>8}')

    combined[tag] = {
        'mathvista': mv,
        'pope': pope * 100 if pope else None,
        'delta_mathvista': mv - baseline_mv if baseline_mv else None,
        'delta_pope': (pope - baseline_pope) * 100 if (pope and baseline_pope) else None,
    }

    # ── Mode A: BRV-style — highest MathVista, no POPE constraint ──
    if mv > brv_best_score:
        brv_best_score = mv
        brv_best_tag = tag

    # ── Mode B: PMBT-style — highest MathVista with POPE guard (≤2pt drop) ──
    pope_ok = True
    if pope is not None and baseline_pope is not None:
        pope_ok = (pope * 100) >= (baseline_pope * 100 - 1.0)
    if pope_ok and mv > pmbt_best_score:
        pmbt_best_score = mv
        pmbt_best_tag = tag

print('-' * 82)

def _print_best(label, tag, criterion):
    if tag:
        bp = combined[tag]
        mv_val = bp[\"mathvista\"]
        dmv = bp[\"delta_mathvista\"]
        mv_s = f'{mv_val:.2f} (Δ{dmv:+.2f})' if dmv is not None else f'{mv_val:.2f}'
        dpope = bp[\"delta_pope\"]
        pope_s = ''
        if bp['pope'] is not None and dpope is not None:
            pope_s = f'  POPE: {bp[\"pope\"]:.2f} (Δ{dpope:+.2f})'
        elif bp['pope'] is not None:
            pope_s = f'  POPE: {bp[\"pope\"]:.2f}'
        print(f'  ★ {label}: {tag}')
        print(f'    MathVista: {mv_s}{pope_s}')
        print(f'    Criterion: {criterion}')
    else:
        print(f'  {label}: No valid configuration found.')

_print_best('BRV-style',  brv_best_tag,  'highest MathVista (no POPE constraint)')
_print_best('PMBT-style', pmbt_best_tag, 'highest MathVista with POPE drop ≤ 1.0 pts')

if brv_best_tag != pmbt_best_tag:
    print(f'  ⚠  Selections differ — BRV={brv_best_tag}, PMBT={pmbt_best_tag}')
print('=' * 82)

# ── 6. Save results ───────────────────────────────────────────
summary = {
    'model': model_name,
    'baseline': {
        'mathvista': baseline_mv,
        'pope': baseline_pope * 100 if baseline_pope else None,
    },
    'configs': combined,
    'best_brv': brv_best_tag,
    'best_brv_mathvista': brv_best_score,
    'best_brv_criterion': 'highest MathVista (BRV: no POPE constraint)',
    'best_pmbt': pmbt_best_tag,
    'best_pmbt_mathvista': pmbt_best_score,
    'best_pmbt_criterion': 'highest MathVista with POPE drop <= 1.0 pts',
    'best': brv_best_tag,
    'best_mathvista': brv_best_score,
}
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'lambda_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n  Saved → {out_dir}/lambda_summary.json')
"

echo "  ✓ Step 22 complete."

fi  # end step 22 (select_lambda)


# ═══════════════════════════════════════════════════════════════
# STEP 23 (snrf): SNRF + PMBT — Shared Neuron Low-Rank Fusion (Layer 1b)
#   Stage 1: profile shared neurons between math LLM + VLM backbone
#   Stage 2: SVD rank-16 injection at PMBT text ∩ shared neuron positions
#   Reference: Cui et al., "Do LLMs and VLMs Share Neurons?" (2026)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "snrf" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 23: SNRF + PMBT — Shared Neuron Low-Rank Fusion (Layer 1b)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/23-snrf"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

SNRF_OUT_DIR="results/23-snrf/${MODE_DIR}/${MODEL_NAME}"

# ── Resolve base LLM and math LLM paths (same as step 16) ────
if [[ -z "$MERGE_BASE_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)    MERGE_BASE_LLM_PATH="Qwen/Qwen2.5-7B" ;;
        llava-ov)   MERGE_BASE_LLM_PATH="Qwen/Qwen2-7B" ;;
        internvl)   MERGE_BASE_LLM_PATH="internlm/internlm2_5-7b-chat" ;;
        llava-llama3) MERGE_BASE_LLM_PATH="modern_vlms/pretrained/llama3-8b-from-llava" ;;
        llava-liuhaotian) MERGE_BASE_LLM_PATH="NousResearch/Llama-2-7b-hf" ;;
    esac
    echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH"
fi
if [[ -z "$MERGE_MATH_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)    MERGE_MATH_LLM_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
        llava-ov)   MERGE_MATH_LLM_PATH="Qwen/Qwen2-Math-7B" ;;
        internvl)   MERGE_MATH_LLM_PATH="internlm/internlm2-math-plus-7b" ;;
        llava-llama3) MERGE_MATH_LLM_PATH="modern_vlms/pretrained/dart-math-llama3-8b-prop2diff" ;;
        llava-liuhaotian) MERGE_MATH_LLM_PATH="meta-math/MetaMath-7B-V1.0" ;;
    esac
    echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH"
fi

# Resolve PMBT label dir
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    MERGE_LABEL_DIR="$_PM_MODE"
else
    echo "  ERROR: No PMBT labels found. Run steps 1-4 first."
    exit 1
fi
echo "  PMBT labels: $MERGE_LABEL_DIR"

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JN23="${SHORT_MODEL}_23snrf"
JOB_NAME="${JN23}_g${GMEM_TAG}"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    SNRF_CMD="$PYTHON $SNRF_SCRIPT \
        --stage both \
        --vlm_path $MODEL_PATH \
        --base_llm_path $MERGE_BASE_LLM_PATH \
        --math_llm_path $MERGE_MATH_LLM_PATH \
        --label_dir $MERGE_LABEL_DIR \
        --model_type $MODEL_TYPE \
        --output_dir $SNRF_OUT_DIR \
        --n_prompts 100 \
        --top_k_pct 0.10 \
        --svd_rank 16 \
        --beta 0.5 \
        --save_model"

    $MERGE_INCLUDE_MULTIMODAL && SNRF_CMD="$SNRF_CMD --include_multimodal"
    $MERGE_INCLUDE_RANDOM && SNRF_CMD="$SNRF_CMD --include_random"

    echo "  → $JOB_NAME → $SNRF_OUT_DIR"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

    if $LOCAL; then
        (cd "$WORK_DIR" && $SNRF_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=98304] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $SNRF_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 23 (snrf)


# ═══════════════════════════════════════════════════════════════
# STEP 17 (srf): PMBT-Guided SRF — Spectral Representation Filtering (Layer 1c)
#   Stage 1: profile hallucination modes from contrastive POPE activations
#   Stage 2: apply spectral filter to down_proj at visual neuron positions
#   Reference: Ali, Zoabi & Wolf, "Suppressing VLM Hallucinations with SRF" (2025)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "srf" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 17: PMBT-Guided SRF — Spectral Representation Filtering (Layer 1c)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/24-srf"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

SRF_OUT_DIR="results/24-srf/${MODE_DIR}/${MODEL_NAME}"

# Resolve PMBT label dir
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    MERGE_LABEL_DIR="$_PM_MODE"
else
    echo "  ERROR: No PMBT labels found. Run steps 1-4 first."
    exit 1
fi
echo "  PMBT labels: $MERGE_LABEL_DIR"

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JN24="${SHORT_MODEL}_24srf"
JOB_NAME="${JN24}_g${GMEM_TAG}"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    # Prefer contrastive POPE from step 10 if it exists
    _SRF_POPE="results/10-halluc_scores/${MODE_DIR}/${MODEL_NAME}/contrastive_pope.jsonl"
    if [[ ! -f "$WORK_DIR/$_SRF_POPE" ]]; then
        _SRF_POPE="$POPE_PATH"  # fallback to raw POPE
        echo "  [info] No contrastive POPE found, using raw: $_SRF_POPE"
    else
        echo "  [info] Using contrastive POPE from step 10: $_SRF_POPE"
    fi

    SRF_CMD="$PYTHON $SRF_SCRIPT \
        --stage both \
        --vlm_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --label_dir $MERGE_LABEL_DIR \
        --pope_path $_SRF_POPE \
        --pope_img_dir $POPE_IMG_DIR \
        --output_dir $SRF_OUT_DIR \
        --alpha 0.5 \
        --n_modes 10 \
        --min_layer_pct 0.5 \
        --n_profile_samples 200 \
        --save_model"

    $MERGE_INCLUDE_RANDOM && SRF_CMD="$SRF_CMD --include_random"

    echo "  → $JOB_NAME → $SRF_OUT_DIR"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

    if $LOCAL; then
        (cd "$WORK_DIR" && $SRF_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=65536] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $SRF_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 17 (srf)


# ═══════════════════════════════════════════════════════════════
# STEP 25 (compose_layer1): Compose Layer 1a + 1c (or 1b + 1c)
#   Applies both the text-neuron weight merge (step 16 or 23) and
#   the visual-neuron spectral filter (step 24) to produce a single
#   model with both edits. Safe because masks are disjoint by construction.
#
#   Usage:
#     bash run_pipeline.sh --step 25 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "compose_layer1" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 25: Compose Layer 1a/1b + Layer 1c into single model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/25-compose-layer1"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

COMPOSE_L1_OUT_DIR="results/25-compose-layer1/${MODE_DIR}/${MODEL_NAME}"
mkdir -p "$COMPOSE_L1_OUT_DIR"

# ── Determine which Layer 1a/1b model to use ─────────────────
# Prefer step 22's best lambda selection for step 16 (Layer 1a)
BEST_LAMBDA_JSON="results/22-select-lambda/${MODE_DIR}/${MODEL_NAME}/lambda_summary.json"
if [[ -f "$BEST_LAMBDA_JSON" ]]; then
    BEST_TAG=$(python3 -c "import json; d=json.load(open('$BEST_LAMBDA_JSON')); print(d.get('best_brv', d.get('best', '')))" 2>/dev/null)
    _LAMBDA25=$(echo "$BEST_TAG" | grep -oP '(?<=_l)\d+\.\d+' || true)
    [[ -z "$_LAMBDA25" ]] && _LAMBDA25="0.1"
    L1A_MODEL="results/16-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${_LAMBDA25}/model"
    echo "  Layer 1a model (step 16, λ=${_LAMBDA25} from step 22): $L1A_MODEL"
else
    # Fallback: use step 23 SNRF output
    L1A_MODEL=$(find results/23-snrf/${MODE_DIR}/${MODEL_NAME}/ -name "pytorch_model.bin" -path "*/snrf_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [[ -n "$L1A_MODEL" ]]; then
        echo "  Layer 1b model (step 23 SNRF): $L1A_MODEL"
    else
        echo "  ERROR: No Layer 1a/1b model found."
        echo "  Run step 16 + 22 (Layer 1a) or step 23 (Layer 1b) first."
        exit 1
    fi
fi

# ── Determine which Layer 1c SRF edits to use ────────────────
SRF_MODES_PT="results/24-srf/${MODE_DIR}/${MODEL_NAME}/hallucination_modes.pt"
if [[ ! -f "$SRF_MODES_PT" ]]; then
    echo "  ERROR: No SRF hallucination modes found at $SRF_MODES_PT"
    echo "  Run step 24 (SRF profile) first."
    exit 1
fi

# ── Resolve PMBT label dir ───────────────────────────────────
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    MERGE_LABEL_DIR="$_PM_MODE"
else
    echo "  ERROR: No PMBT labels found."
    exit 1
fi

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JN25="${SHORT_MODEL}_25comp"
JOB_NAME="${JN25}_g${GMEM_TAG}"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    # Compose by applying SRF edit stage on top of the Layer 1a/1b model
    # This works because:
    #   Layer 1a/1b modified text neuron weights (gate/up/down at text positions)
    #   Layer 1c modifies visual neuron weights (down_proj columns at visual positions)
    #   The two edits are on disjoint neuron populations.
    COMPOSE_CMD="$PYTHON $SRF_SCRIPT \
        --stage edit \
        --vlm_path $L1A_MODEL \
        --model_type $MODEL_TYPE \
        --label_dir $MERGE_LABEL_DIR \
        --eigenvecs_dir results/24-srf/${MODE_DIR}/${MODEL_NAME} \
        --output_dir $COMPOSE_L1_OUT_DIR \
        --alpha 0.5 \
        --n_modes 10 \
        --min_layer_pct 0.5 \
        --save_model"

    echo "  → $JOB_NAME → $COMPOSE_L1_OUT_DIR"
    echo "  Layer 1a/1b base: $L1A_MODEL"
    echo "  SRF modes:        $SRF_MODES_PT"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

    if $LOCAL; then
        (cd "$WORK_DIR" && $COMPOSE_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=65536] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $COMPOSE_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 25 (compose_layer1)


# ═══════════════════════════════════════════════════════════════
# STEP 29 (weight_diff_rank): Weight Diff Effective Rank Analysis
#
#   Compares VLM weights to base LLM weights at PMBT visual vs text
#   neuron positions. Measures effective rank (spectral entropy) of
#   the weight diff to test: sequential fine-tuning concentrates
#   late-added modalities into low-rank subspaces.
#
#   CPU-only (loads both models in float32, computes SVD per layer).
#   ~30 min per model.
#
#   Run as:
#     bash run_pipeline.sh --step 29 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "weight_diff_rank" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 29: Weight Diff Effective Rank Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/29-weight-diff-rank"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

RANK_OUT_DIR="results/29-weight-diff-rank/${MODE_DIR}/${MODEL_NAME}"
mkdir -p "$WORK_DIR/$RANK_OUT_DIR"

# ── Resolve base LLM path (local only — no downloads on compute nodes) ──
_BASE_LLM=""
_HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

_resolve_hf_cache() {
    # Given a HF repo ID like "Qwen/Qwen2.5-7B", find the local cached snapshot
    local repo_id="$1"
    local safe_name="models--${repo_id//\//-}"  # Qwen/Qwen2.5-7B → models--Qwen-Qwen2.5-7B
    # but HF uses models--Qwen--Qwen2.5-7B (double dash for /)
    safe_name="models--$(echo "$repo_id" | sed 's/\//-/; s/-/--/')"
    # Actually HF format: replace / with --
    safe_name="models--$(echo "$repo_id" | sed 's/\//--/g')"
    local snap_dir="$_HF_CACHE/$safe_name/snapshots"
    if [[ -d "$snap_dir" ]]; then
        # Return the first (usually only) snapshot
        local snap=$(ls -1 "$snap_dir" 2>/dev/null | head -1)
        if [[ -n "$snap" ]]; then
            echo "$snap_dir/$snap"
            return 0
        fi
    fi
    return 1
}

case "$MODEL_TYPE" in
    qwen2vl)
        _BASE_LLM=$(_resolve_hf_cache "Qwen/Qwen2.5-7B") || _BASE_LLM="Qwen/Qwen2.5-7B"
        ;;
    llava-ov)
        _BASE_LLM=$(_resolve_hf_cache "Qwen/Qwen2-7B") || _BASE_LLM="Qwen/Qwen2-7B"
        ;;
    internvl)
        _BASE_LLM=$(_resolve_hf_cache "internlm/internlm2_5-7b-chat") || _BASE_LLM="internlm/internlm2_5-7b-chat"
        ;;
    llava-llama3)
        _BASE_LLM="modern_vlms/pretrained/llama3-8b-from-llava"
        ;;
    llava-liuhaotian|llava)
        _BASE_LLM=$(_resolve_hf_cache "NousResearch/Llama-2-7b-hf") \
            || _BASE_LLM=$(_resolve_hf_cache "NousResearch/Llama-2-7b-hf") \
            || _BASE_LLM="NousResearch/Llama-2-7b-hf"
        ;;
    *)
        echo "  ERROR: Unknown model_type $MODEL_TYPE for base LLM resolution"
        exit 1
        ;;
esac
# Allow override
[[ -n "$MERGE_BASE_LLM_PATH" ]] && _BASE_LLM="$MERGE_BASE_LLM_PATH"
# ── Override VLM path for models that need HF-converted checkpoints ──
# liuhaotian/llava-v1.5-7b stores LLM weights separately (not in HF state_dict),
# so loading it via AutoModel gives randomly initialized language_model weights.
# Use llava-hf/llava-1.5-7b-hf instead, which bundles everything.
_RANK_VLM_PATH="$MODEL_PATH"
if [[ "$MODEL_TYPE" == "llava-liuhaotian" || "$MODEL_TYPE" == "llava" ]]; then
    _HF_LLAVA15=$(_resolve_hf_cache "llava-hf/llava-1.5-7b-hf") || true
    if [[ -n "$_HF_LLAVA15" && -d "$_HF_LLAVA15" ]]; then
        _RANK_VLM_PATH="$_HF_LLAVA15"
        echo "  [auto] VLM override: using llava-hf/llava-1.5-7b-hf for weight extraction"
    else
        echo "  WARNING: llava-hf/llava-1.5-7b-hf not cached. Download first:"
        echo "    python3 -c \"from transformers import AutoModelForVision2Seq; m=AutoModelForVision2Seq.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype='auto'); del m\""
    fi
fi

echo "  VLM:      $_RANK_VLM_PATH"
echo "  Base LLM: $_BASE_LLM"
# Warn if path is a HF repo ID (not local)
if [[ ! -d "$_BASE_LLM" && ! -d "$WORK_DIR/$_BASE_LLM" ]]; then
    echo "  WARNING: Base LLM path is not local. Download first:"
    echo "    python3 -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$_BASE_LLM', trust_remote_code=True, torch_dtype='auto')\""
fi

# ── Resolve PMBT label dir ──
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    RANK_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    RANK_LABEL_DIR="$_PM_MODE"
else
    echo "  ERROR: No PMBT labels found."
    exit 1
fi

JOB_NAME="29_rank_${SHORT_MODEL}"

RANK_CMD="$PYTHON code/weight_diff_rank.py \
    --model_type $MODEL_TYPE \
    --vlm_path $_RANK_VLM_PATH \
    --llm_path $_BASE_LLM \
    --model_name $MODEL_NAME \
    --n_layers $N_LAYERS \
    --label_dir $RANK_LABEL_DIR \
    --output_dir $RANK_OUT_DIR \
    --weight_types down_proj,gate_proj,up_proj"

if [[ -f "$WORK_DIR/$RANK_OUT_DIR/weight_diff_rank.json" ]]; then
    echo "  [skip] Results already exist at $RANK_OUT_DIR/weight_diff_rank.json"
else
    echo "  → $JOB_NAME → $RANK_OUT_DIR"
    if $LOCAL; then
        (cd "$WORK_DIR" && $RANK_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        # CPU-only job — no GPU needed, just lots of RAM for two 7B models
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -R "rusage[mem=98304]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $RANK_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

# ── Phase 2: Cross-model comparison (depends on all per-model jobs) ──
# Submit once from any model run — checks if all 3 primary models have results
CROSS_MODELS="llava-onevision-7b,qwen2.5-vl-7b,internvl2.5-8b"
CROSS_RESULT="results/29-weight-diff-rank/${MODE_DIR}/cross_model_rank_comparison.json"
CROSS_JOB="29_rank_cross"

if [[ -f "$WORK_DIR/$CROSS_RESULT" ]]; then
    echo "  [skip] Cross-model comparison already exists"
elif is_job_active "$CROSS_JOB" 2>/dev/null; then
    echo "  [skip] $CROSS_JOB — already active"
else
    # Build dependency string from all 3 per-model jobs
    CROSS_DEP="done(29_rank_lo) && done(29_rank_qw) && done(29_rank_int)"

    CROSS_CMD="$PYTHON code/weight_diff_rank.py \
        --cross_model \
        --base_dir results/29-weight-diff-rank/${MODE_DIR} \
        --models $CROSS_MODELS"

    echo "  → $CROSS_JOB → $CROSS_RESULT (pending on all 3 models)"
    if $LOCAL; then
        (cd "$WORK_DIR" && $CROSS_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${CROSS_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$CROSS_JOB" \
            -w "$CROSS_DEP" \
            -R "rusage[mem=4096]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${CROSS_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${CROSS_JOB}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $CROSS_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 29 (weight_diff_rank)



# ═══════════════════════════════════════════════════════════════════
# STEP 26 (vit_analysis): VIT Weight Change × PMBT Label Correlation
# ═══════════════════════════════════════════════════════════════════
#
# Compares per-neuron weight changes between base LLM and VLM,
# then tests whether visually-responsive neurons (PMBT label = visual)
# are the neurons most modified during visual instruction tuning.
#
# No GPU required — loads state dicts on CPU only.
#
# Usage:
#     bash run_pipeline.sh --step 26 --model-type llava-ov
#     bash run_pipeline.sh --step 26 --model-type qwen2vl,internvl,llava-ov
#
if [[ "$STEP" == "vit_analysis" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 26: VIT Weight Change Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VIT_SCRIPT="code/vit_weight_analysis.py"
VIT_OUT_DIR="results/26-vit-analysis/${MODE_DIR}/${MODEL_NAME}"
VIT_LOG_DIR="${LOG_DIR}/26-vit-analysis"
mkdir -p "$WORK_DIR/$VIT_OUT_DIR" "$WORK_DIR/$VIT_LOG_DIR"

# ── Resolve base LLM path ──────────────────────────────────
VIT_BASE_LLM=""
if [[ -n "$MERGE_BASE_LLM_PATH" ]]; then
    VIT_BASE_LLM="$MERGE_BASE_LLM_PATH"
else
    case "$MODEL_TYPE" in
        qwen2vl)
            VIT_BASE_LLM="Qwen/Qwen2.5-7B"
            ;;
        llava-ov)
            VIT_BASE_LLM="Qwen/Qwen2-7B"
            ;;
        internvl)
            VIT_BASE_LLM="internlm/internlm2_5-7b-chat"
            ;;
        llava-liuhaotian)
            VIT_BASE_LLM="lmsys/vicuna-7b-v1.5"
            ;;
        llava-llama3)
            VIT_BASE_LLM="modern_vlms/pretrained/llama3-8b-from-llava"
            ;;
        llava-mistral)
            VIT_BASE_LLM="mistralai/Mistral-7B-v0.1"
            ;;
    esac
fi

if [[ -z "$VIT_BASE_LLM" ]]; then
    echo "  ERROR: Could not determine base LLM for model type '$MODEL_TYPE'"
    echo "  Use --merge-base-llm to specify manually"
    exit 1
fi

# ── PMBT label directory ──────────────────────────────────
LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"
VIT_LABEL_DIR="${LABELS_BASE}/${MODEL_NAME}/llm_permutation"

echo "  Model:      $MODEL_NAME ($MODEL_TYPE)"
echo "  VLM:        $MODEL_PATH"
echo "  Base LLM:   $VIT_BASE_LLM"
echo "  Labels:     $VIT_LABEL_DIR"
echo "  Output:     $VIT_OUT_DIR"

# ── Check prerequisites ──────────────────────────────────
if [[ ! -d "$WORK_DIR/$VIT_LABEL_DIR" ]]; then
    echo "  ERROR: PMBT label directory not found: $VIT_LABEL_DIR"
    echo "  Run step 3 (classify) first."
    exit 1
fi

# ── Submit job (CPU only, needs ~64GB RAM for two 7B models) ──
JOB_NAME="26_vit_${SHORT_MODEL}"

if [[ -f "$WORK_DIR/$VIT_OUT_DIR/vit_weight_analysis.json" ]]; then
    echo "  [skip] Results already exist at $VIT_OUT_DIR/vit_weight_analysis.json"
    SKIPPED=$((SKIPPED + 1))
else
    VIT_CMD="$PYTHON $VIT_SCRIPT \
        --vlm_path $MODEL_PATH \
        --base_path $VIT_BASE_LLM \
        --model_type $MODEL_TYPE \
        --label_dir $VIT_LABEL_DIR \
        --n_layers $N_LAYERS \
        --output_dir $VIT_OUT_DIR"

    echo "  [$JOB_NAME] Submitting VIT weight analysis..."
    if $LOCAL; then
        (cd "$WORK_DIR" && $VIT_CMD) \
            2>&1 | tee "$WORK_DIR/$VIT_LOG_DIR/${JOB_NAME}.log"
    else
        bsub -q "$QUEUE" -J "$JOB_NAME" \
            -R "rusage[mem=65536]" \
            -oo "$WORK_DIR/$VIT_LOG_DIR/${JOB_NAME}.log" \
            -eo "$WORK_DIR/$VIT_LOG_DIR/${JOB_NAME}.err" \
            "cd $WORK_DIR && $VIT_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

fi  # end step 26 (vit_analysis)



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
echo "  Steering dir:       results/11-steering/${MODE_DIR}/$MODEL_NAME/${HALLUC_SCORE_METHOD}/"
echo "  ECCV plots:         results/13-plots/${MODE_DIR}/$MODEL_NAME/${HALLUC_SCORE_METHOD}/"
echo "  Text-inject dir:    results/16-weight-merge/${MODE_DIR}/$MODEL_NAME/text_inject/"
echo "  Visual-transplant:  results/17-weight-merge/${MODE_DIR}/$MODEL_NAME/visual_transplant/"
echo "  Combined model:     results/18-compose-merge/${MODE_DIR}/$MODEL_NAME/"
echo "  Benchmark results:  results/19-evaluate/${MODE_DIR}/$MODEL_NAME/"
echo "  Eval summary:       results/20-summary/${MODE_DIR}/$MODEL_NAME/"
echo "  SNRF (Layer 1b):    results/23-snrf/${MODE_DIR}/$MODEL_NAME/"
echo "  SRF  (Layer 1c):    results/24-srf/${MODE_DIR}/$MODEL_NAME/"
echo "  VIT analysis:       results/26-vit-analysis/${MODE_DIR}/$MODEL_NAME/"
echo "  Composed (1a+1c):   results/25-compose-layer1/${MODE_DIR}/$MODEL_NAME/"
echo "═══════════════════════════════════════════════════════════"
exit 0