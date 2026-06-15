# #!/bin/bash
# # ═══════════════════════════════════════════════════════════════════
# # run_pipeline.sh — Full Xu et al. neuron classification pipeline
# #
# # Backend: liuhaotian/llava-v1.5-7b (original LLaVA model)
# #
# # Chains:  1 describe → 2 merge_desc → 3 classify → 4 merge_class → 5 ablation → 6 visualize → 7 attention → 8 statistics → 10 halluc_taxonomy
# # Keeps Python scripts separate; this script handles orchestration.
# #
# # Usage:
# #   bash run_pipeline.sh                              # full pipeline, 32 GPUs
# #   ...
# #   bash run_pipeline.sh --step 10                    # (halluc_taxonomy) enrichment across ALL neuron categories
# #   bash run_pipeline.sh --step 10 --halluc-top-k-pct 3       # top 3% as hallucination-driving
# #   ...

# set -euo pipefail

# # ── Defaults ──────────────────────────────────────────────────
# # ... (existing defaults)
# HALLUC_SCRIPT="code/hallucination_taxonomy.py"                               # hallucination taxonomy enrichment for step 10
# HALLUC_TOP_K_PCT="5.0"                                                       # top K% neurons classified as hallucination-driving
# HALLUC_N_POPE="500"                                                          # POPE questions per ablation evaluation
# HALLUC_BATCH_NEURONS="50"                                                    # neurons ablated simultaneously per batch
# HALLUC_N_RANDOM="1000"                                                       # random baseline trials for calibration
# HALLUC_SKIP_ABLATION=false                                                   # if true, load pre-computed ablation scores
# HALLUC_SCORES=""                                                             # path to pre-computed ablation scores JSON

# # ── Parse args ────────────────────────────────────────────────
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         # ... (existing cases)
#         --halluc-top-k-pct) HALLUC_TOP_K_PCT="$2"; shift 2 ;;
#         --halluc-n-pope) HALLUC_N_POPE="$2"; shift 2 ;;
#         --halluc-batch-neurons) HALLUC_BATCH_NEURONS="$2"; shift 2 ;;
#         --halluc-n-random) HALLUC_N_RANDOM="$2"; shift 2 ;;
#         --halluc-skip-ablation) HALLUC_SKIP_ABLATION=true; shift ;;
#         --halluc-scores) HALLUC_SCORES="$2"; shift 2 ;;
#         # ...
#     esac
# done

# # ── Normalise --step value ──────────────────────────────────────────────
# case "$STEP" in
#     # ... (existing steps 1-9)
#     10|halluc_taxonomy|halluc) STEP="halluc" ;;
#     all|all_att)             ;;  
#     *) echo "ERROR: unknown step '$STEP'"; exit 1 ;;
# esac

# # ── Job name bases ──────────────────────────────────────────────────────
# # ...
# JN10="10_${SHORT_MODEL}"   # halluc_taxonomy

# # ... (rest of pipeline steps 1 through 8)

# # ═══════════════════════════════════════════════════════════════
# # STEP 10 (halluc_taxonomy): Hallucination enrichment across ALL neuron categories
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "halluc" || $STEP_ALL == true ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 10: Hallucination Taxonomy (enrichment analysis)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/10-halluc_taxonomy"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# HALLUC_OUT_DIR="results/hallucination_taxonomy/${MODE_DIR}/$MODEL_NAME"
# HALLUC_MARKER="$HALLUC_OUT_DIR/done.marker"
# HALLUC_LABEL_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"

# JOB_NAME="${JN10}"

# if [[ -f "$HALLUC_MARKER" ]] || is_job_active "$JOB_NAME"; then
#     echo "  [skip] $JOB_NAME — already done or active"
#     SKIPPED=$((SKIPPED + 1))
# else
#     HALLUC_CMD="$PYTHON $HALLUC_SCRIPT \
#         --label_dir $HALLUC_LABEL_DIR \
#         --taxonomy pmbt \
#         --model_type $MODEL_TYPE \
#         --model_path $MODEL_PATH \
#         --model_name $MODEL_NAME \
#         --n_layers $N_LAYERS \
#         --pope_path $POPE_PATH \
#         --pope_img_dir $POPE_IMG_DIR \
#         --top_k_pct $HALLUC_TOP_K_PCT \
#         --n_pope_questions $HALLUC_N_POPE \
#         --batch_neurons $HALLUC_BATCH_NEURONS \
#         --n_random_trials $HALLUC_N_RANDOM \
#         --n_gpus $SHARDS_EFFECTIVE \
#         --output_dir $HALLUC_OUT_DIR"

#     if $HALLUC_SKIP_ABLATION; then
#         HALLUC_CMD="$HALLUC_CMD --skip_ablation"
#         [[ -n "$HALLUC_SCORES" ]] && HALLUC_CMD="$HALLUC_CMD --ablation_scores $HALLUC_SCORES"
#     fi

#     if $LOCAL; then
#         (cd "$WORK_DIR" && eval "$HALLUC_CMD") 2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#         [[ ${PIPESTATUS[0]} -eq 0 ]] && touch "$HALLUC_MARKER"
#     else
#         BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
#             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
#         if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
#             BSUB_ARGS+=(-w "done(${JN4})")
#         fi

#         if $HALLUC_SKIP_ABLATION; then
#             bsub "${BSUB_ARGS[@]}" "cd $WORK_DIR && $HALLUC_CMD && touch $HALLUC_MARKER"
#         else
#             bsub_tiered "${BSUB_ARGS[@]}" -- "cd $WORK_DIR && $HALLUC_CMD && touch $HALLUC_MARKER"
#         fi
#         SUBMITTED=$((SUBMITTED + 1))
#     fi
# fi
# fi  # end step 10


# #!/bin/bash
# # ═══════════════════════════════════════════════════════════════════
# # run_pipeline.sh — Full Xu et al. neuron classification pipeline
# #
# # Backend: liuhaotian/llava-v1.5-7b (original LLaVA model)
# #
# # Chains:  1 describe → 2 merge_desc → 3 classify → 4 merge_class → 5 ablation → merge_ablation → 6 visualize → 7 attention → 8 statistics
# # Keeps Python scripts separate; this script handles orchestration.
# #
# # Usage:
# #   bash run_pipeline.sh                              # full pipeline, 32 GPUs
# #   bash run_pipeline.sh --step 1                     # (describe) generate VLM descriptions
# #   bash run_pipeline.sh --step 2                     # (merge_desc) merge description shards
# #   bash run_pipeline.sh --step 3                     # (classify) FT + permutation classification
# #   bash run_pipeline.sh --step 4                     # (merge_class) merge classification shards
# #   bash run_pipeline.sh --step 5                     # (ablation) ablation validation
# #   bash run_pipeline.sh --step merge_ablation         # merge existing ablation shard results
# #   bash run_pipeline.sh --step 6                     # (visualize) Figure 3 activation maps
# #   bash run_pipeline.sh --step 6 --viz-fig3          # reproduce exact Xu Figure 3 panels (a)-(f)
# #   bash run_pipeline.sh --step 6 --viz-fig3 --viz-taxonomy both  # FT + PMBT labels
# #   bash run_pipeline.sh --step 6 --viz-fig89         # reproduce Xu Figures 8 & 9
# #   bash run_pipeline.sh --step 6 --viz-supplementary # reproduce Figures 15-17
# #   bash run_pipeline.sh --step 7 --attn-image-id 000000189475  # attention analysis (single image)
# #   bash run_pipeline.sh --step 7 --attn-image-path /path/to/image.jpg --attn-words "dough nut pink"
# #   bash run_pipeline.sh --step all_att --mode test            # uses 6 default fig3 images for attn
# #   bash run_pipeline.sh --step 8                     # (statistics) all charts (FT + PMBT)
# #   bash run_pipeline.sh --step all                   # steps 1-6, 8 (standard chain)
# #   bash run_pipeline.sh --step all_att              # steps 1-8 (includes attention)
# #   bash run_pipeline.sh --mode test                  # quick test (6 fig3 images, 2 layers, 1 GPU)
# #   bash run_pipeline.sh --mode test --clean           # remove markers → force re-run (keeps results)
# #   bash run_pipeline.sh --mode test --clean 3         # remove markers from step 3 onwards
# #   bash run_pipeline.sh --mode test --clean --wipe    # delete all results + logs
# #   bash run_pipeline.sh --mode test --clean 3 --wipe  # delete results + logs from step 3 onwards
# #   bash run_pipeline.sh --shards 16                  # use 16 GPUs
# #   bash run_pipeline.sh --local                      # run locally (no bsub, no sharding)
# #   bash run_pipeline.sh --queue waic-risk            # use waic-risk queue
# #   bash run_pipeline.sh --suffix _v2                 # output to results/classification_v2
# #   bash run_pipeline.sh --prune-images 200           # use 200 images for ablation
# #   bash run_pipeline.sh --ablation-shards 16         # use 16 GPUs per ablation config
# #   bash run_pipeline.sh --step 5 --pope-path pope/coco_pope_random.json
# #   bash run_pipeline.sh --step 5 --pope-path pope/coco_pope_random.json \
# #       --chair-ann-path /path/to/instances_val2014.json \
# #       --pope-img-dir /path/to/val2014/
# #
# # GPU tiered escalation:
# #   All jobs start at gmem=80G. If still PEND after 2 min, killed and
# #   resubmitted at gmem=40G. After another 2 min → gmem=10G.
# #   Override tiers: --gmem 80G,40G,10G   Override wait: --gmem-wait 120
# #
# # Idempotent: re-running skips jobs whose output already exists or
# #             that are currently PEND/RUN in LSF.
# #
# # Examples:
# #   bash run_pipeline.sh --mode test --local
# #   bash run_pipeline.sh --shards 32
# #   bash run_pipeline.sh --step 3 --shards 32
# # ═══════════════════════════════════════════════════════════════════

# set -euo pipefail

# # ── Defaults ──────────────────────────────────────────────────
# QUEUE="waic-risk"
# QUEUE_SET=false
# LOCAL=false
# STEP="all"             # 1|2|3|4|5|6|7|8|all|all_att  (or: describe|merge_desc|classify|merge_class|ablation|visualize|attention|statistics)
# MODE="full"            # test | full
# SHARDS=32              # GPUs
# MODEL_TYPE="llava-liuhaotian"    # model backend (llava-liuhaotian / llava-hf / llava-ov / internvl / qwen2vl)
# MODEL_PATH="liuhaotian/llava-v1.5-7b"  # HF Hub ID or local path to model weights
# MODEL_NAME="llava-1.5-7b"  # must match --model default in $CLASSIFY_SCRIPT
# OUT_SUFFIX_USER=""        # user-provided suffix for output dirs
# CLASSIFY_SCRIPT="code/neuron_modality_statistical.py"  # classification script for step 3 (classify)
# ABLATION_SCRIPT="code/neuron_ablation_validate.py"      # ablation validation script for step 5 (ablation)
# VIZ_SCRIPT="code/visualize_neuron_activations.py"        # Figure 3 visualization script for step visualize
# PLOT_SCRIPT="code/plot_neuron_statistics.py"              # Figures 5/6/7 statistics charts for step 8 (statistics)
# ATTN_SCRIPT="code/attention_analysis.py"                  # attention analysis script for step 7 (attention)
# VTP_SCRIPT="code/test_visual_token_pressure.py"              # VTP hypothesis analysis for step 9 (vtp)
# ATTN_IMAGE_ID=""                                          # COCO image ID for attention analysis
# ATTN_IMAGE_PATH=""                                        # direct image path (overrides ATTN_IMAGE_ID)
# ATTN_HIGHLIGHTED_WORDS="hot white blue tie"               # words Xu highlighted (space-separated)
# ATTN_HEATMAP_LAYERS="0 7 15 23 28 31"                    # layers to show in heatmap
# ATTN_LAYER=""                                             # layer index for auto-highlight mode
# ATTN_NEURON_IDX=""                                        # neuron index for auto-highlight mode
# ATTN_TOP_K="5"                                            # top-K tokens for auto-highlight mode
# ATTN_N_SAMPLES="1"                                           # number of top-ranked images to analyze
# ATTN_GPUS="1"                                                # number of GPUs for parallel sharding
# DESC_SUFFIX_USER=""        # suffix for description files (defaults to OUT_SUFFIX_USER)
# OUTPUT_DIR_USER=""         # override classification output dir
# PRUNE_IMAGES=100           # number of images for ablation validation
# POPE_PATH="data/POPE/output/coco/coco_pope_random.json"   # path to POPE jsonl
# CHAIR_ANN_PATH="data/annotations/instances_val2014.json"  # path to instances_val2014.json
# POPE_IMG_DIR="data/val2014"                                 # path to COCO val images for POPE/CHAIR
# CHAIR_NUM_IMAGES=500       # number of images for CHAIR evaluation
# TRIVIAQA_PATH="data/triviaqa/qa/verified-web-dev.json"       # path to TriviaQA verified-web-dev.json
# TRIVIAQA_NUM=2000                                            # number of TriviaQA questions
# MMLU_DIR="data/mmlu/"                                       # path to MMLU data/ directory
# MMLU_NUM=2000                                                # number of MMLU questions
# VIZ_FIG3=false             # if true, visualize step reproduces Xu Figure 3 panels
# VIZ_FIG89=false            # if true, visualize step reproduces Xu Figures 8 & 9
# VIZ_SUPPLEMENTARY=false    # if true, visualize step reproduces supplementary Figures 15-17
# VIZ_TAXONOMY="both"        # ft | pmbt | both — which taxonomy labels to show in Figure 3 headers

# # ── Advanced ablation settings (8 approaches) ──
# ABLATION_METHOD="zero"         # zero | mean | noise | clamp_high | clamp_low
# ABLATION_CURVE=false           # if true, sweep top_n values (ablation curve)
# ABLATION_TOP_N=""              # specific top_n value (empty = ablate all)
# ABLATION_LAYER_RANGE=""        # e.g. "0-10" or "22-31" (empty = all layers)
# ABLATION_CURVE_STEPS="100,500,1000,5000"  # comma-separated top_n values for curve
# ABLATION_N_STATS=50            # reference images for mean/noise/clamp stats
# ABLATION_RANKING="label"       # label | cett
# ABLATION_N_CETT=30             # reference images for CETT computation
# ABLATION_ALL=false             # if true, submit all approaches as parallel GPU jobs
# ABLATION_TAXONOMY="pmbt"       # ft | pmbt | both — which taxonomy to ablate (used when --ablation-all is NOT set) (default changed to pmbt only)
# ABLATION_SHARDS=8              # GPUs for ablation (per taxonomy config)
# CLEAN=false                    # if true, delete markers to force re-run (test mode only)
# WIPE=false                     # if true, delete entire result directories (used with --clean --wipe)
# CLEAN_FROM="auto"                  # clean from this step onwards (auto=match --step, or explicit 1-8)

# # GPU memory tiers — escalate through tiers when jobs stay PEND
# GPU_GMEM_TIERS=("20G")                # override with --gmem 40G,20G
# GMEM_WAIT=120                          # seconds to wait before escalating (override with --gmem-wait)
# GPU_RES_BASE="rusage[mem=24576] order[-gpu_maxfactor]"

# # Dataset constants
# N_TOTAL_IMAGES=23000
# N_LAYERS=32
# GEN_LIMIT=""                       # --limit N: override N_TOTAL_IMAGES for quick testing

# # ── Parse args ────────────────────────────────────────────────
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --local)    LOCAL=true; shift ;;
#         --step)     STEP="$2"; shift 2 ;;
#         --mode)     MODE="$2"; shift 2 ;;
#         --shards)   SHARDS="$2"; shift 2 ;;
#         --limit)    GEN_LIMIT="$2"; shift 2 ;;
#         --queue)    QUEUE="$2"; QUEUE_SET=true; shift 2 ;;
#         --model-type) MODEL_TYPE="$2"; shift 2 ;;
#         --model-path) MODEL_PATH="$2"; shift 2 ;;
#         --suffix)   OUT_SUFFIX_USER="$2"; shift 2 ;;
#         --classify-script) CLASSIFY_SCRIPT="$2"; shift 2 ;;
#         --ablation-script) ABLATION_SCRIPT="$2"; shift 2 ;;
#         --desc-suffix) DESC_SUFFIX_USER="$2"; shift 2 ;;
#         --output-dir) OUTPUT_DIR_USER="$2"; shift 2 ;;
#         --prune-images) PRUNE_IMAGES="$2"; shift 2 ;;
#         --pope-path) POPE_PATH="$2"; shift 2 ;;
#         --chair-ann-path) CHAIR_ANN_PATH="$2"; shift 2 ;;
#         --pope-img-dir) POPE_IMG_DIR="$2"; shift 2 ;;
#         --chair-num-images) CHAIR_NUM_IMAGES="$2"; shift 2 ;;
#         --triviaqa-path) TRIVIAQA_PATH="$2"; shift 2 ;;
#         --triviaqa-num) TRIVIAQA_NUM="$2"; shift 2 ;;
#         --mmlu-dir) MMLU_DIR="$2"; shift 2 ;;
#         --mmlu-num) MMLU_NUM="$2"; shift 2 ;;
#         --viz-fig3) VIZ_FIG3=true; shift ;;
#         --viz-fig89) VIZ_FIG89=true; shift ;;
#         --viz-supplementary) VIZ_SUPPLEMENTARY=true; shift ;;
#         --viz-taxonomy) VIZ_TAXONOMY="$2"; shift 2 ;;
#         --gmem)     IFS=',' read -ra GPU_GMEM_TIERS <<< "$2"; shift 2 ;;
#         --gmem-wait) GMEM_WAIT="$2"; shift 2 ;;
#         --ablation-method) ABLATION_METHOD="$2"; shift 2 ;;
#         --ablation-curve) ABLATION_CURVE=true; shift ;;
#         --ablation-top-n) ABLATION_TOP_N="$2"; shift 2 ;;
#         --ablation-layer-range) ABLATION_LAYER_RANGE="$2"; shift 2 ;;
#         --ablation-curve-steps) ABLATION_CURVE_STEPS="$2"; shift 2 ;;
#         --ablation-n-stats) ABLATION_N_STATS="$2"; shift 2 ;;
#         --ablation-ranking) ABLATION_RANKING="$2"; shift 2 ;;
#         --ablation-n-cett) ABLATION_N_CETT="$2"; shift 2 ;;
#         --ablation-all) ABLATION_ALL=true; shift ;;
#         --ablation-taxonomy) ABLATION_TAXONOMY="$2"; shift 2 ;;
#         --ablation-shards) ABLATION_SHARDS="$2"; shift 2 ;;
#         --attn-image-id) ATTN_IMAGE_ID="$2"; shift 2 ;;
#         --attn-image-path) ATTN_IMAGE_PATH="$2"; shift 2 ;;
#         --attn-words) ATTN_HIGHLIGHTED_WORDS="$2"; shift 2 ;;
#         --attn-layers) ATTN_HEATMAP_LAYERS="$2"; shift 2 ;;
#         --attn-layer) ATTN_LAYER="$2"; shift 2 ;;
#         --attn-neuron) ATTN_NEURON_IDX="$2"; shift 2 ;;
#         --attn-topk) ATTN_TOP_K="$2"; shift 2 ;;
#         --attn-nsamples) ATTN_N_SAMPLES="$2"; shift 2 ;;
#         --attn-gpus) ATTN_GPUS="$2"; shift 2 ;;
#         --clean)        CLEAN=true
#                         # Optional step number: --clean 3 cleans from step 3 onwards
#                         # If omitted, defaults to the --step value (set after parsing)
#                         if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
#                             CLEAN_FROM="$2"; shift 2
#                         else
#                             CLEAN_FROM="auto"; shift
#                         fi
#                         ;;
#         --wipe)         WIPE=true; shift ;;
#         *)          echo "Unknown arg: $1"; exit 1 ;;
#     esac
# done

# # ── Normalize model-type aliases ────────────────────────────────────────────
# # Accept short names at the CLI so users can type e.g. --model-type llava
# # instead of --model-type llava-liuhaotian.  Full names still work.
# _normalize_model_type() {
#     case "$1" in
#         llava)  echo "llava-liuhaotian" ;;
#         intern) echo "internvl" ;;
#         qwen)   echo "qwen2vl" ;;
#         *)      echo "$1" ;;
#     esac
# }
# if [[ "$MODEL_TYPE" == *","* ]]; then
#     _normalized=""
#     IFS=',' read -ra _parts <<< "$MODEL_TYPE"
#     for _p in "${_parts[@]}"; do
#         _n=$(_normalize_model_type "$_p")
#         _normalized="${_normalized:+${_normalized},}${_n}"
#     done
#     MODEL_TYPE="$_normalized"
# elif [[ "$MODEL_TYPE" != "all" ]]; then
#     MODEL_TYPE=$(_normalize_model_type "$MODEL_TYPE")
# fi

# # ── Validate model-type ─────────────────────────────────────────────────────
# _VALID_MODELS="llava-liuhaotian llava-hf llava-ov llava-ov-si internvl qwen2vl all"
# _VALID_ALIASES="llava intern qwen"
# if [[ "$MODEL_TYPE" != *","* ]]; then
#     _found=false
#     for _vm in $_VALID_MODELS; do
#         [[ "$MODEL_TYPE" == "$_vm" ]] && _found=true && break
#     done
#     if ! $_found; then
#         echo "ERROR: unknown --model-type '$MODEL_TYPE'"
#         echo ""
#         echo "  Valid names:    llava-liuhaotian  llava-hf  llava-ov  internvl  qwen2vl  all"
#         echo "  Short aliases:  llava             —         —         intern    qwen"
#         echo ""
#         echo "  Examples:"
#         echo "    bash code/run_pipeline.sh --model-type llava ..."
#         echo "    bash code/run_pipeline.sh --model-type intern ..."
#         echo "    bash code/run_pipeline.sh --model-type qwen,intern ..."
#         exit 1
#     fi
# else
#     IFS=',' read -ra _check_parts <<< "$MODEL_TYPE"
#     for _cp in "${_check_parts[@]}"; do
#         _found=false
#         for _vm in $_VALID_MODELS; do
#             [[ "$_cp" == "$_vm" ]] && _found=true && break
#         done
#         if ! $_found; then
#             echo "ERROR: unknown model '$_cp' in --model-type '$MODEL_TYPE'"
#             echo ""
#             echo "  Valid names:    llava-liuhaotian  llava-hf  llava-ov  internvl  qwen2vl"
#             echo "  Short aliases:  llava             —         —         intern    qwen"
#             exit 1
#         fi
#     done
# fi

# _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WORK_DIR="$(cd "$_SCRIPT_DIR/.." && pwd)"
# cd "$WORK_DIR"
# mkdir -p logs

# # ── Multi-model dispatch ──────────────────────────────────────────────────
# # --model-type all              → run for all 4 backends sequentially
# # --model-type internvl,qwen2vl → run for selected backends
# ALL_MODELS=(llava-liuhaotian llava-hf llava-ov llava-ov-si internvl qwen2vl)
# if [[ "$MODEL_TYPE" == "all" ]] || [[ "$MODEL_TYPE" == *","* ]]; then
#     if [[ "$MODEL_TYPE" == "all" ]]; then
#         RUN_MODELS=("${ALL_MODELS[@]}")
#     else
#         IFS=',' read -ra RUN_MODELS <<< "$MODEL_TYPE"
#     fi
#     echo "Multi-model dispatch: ${RUN_MODELS[*]}"
#     ANY_FAILED=0
#     for m in "${RUN_MODELS[@]}"; do
#         echo ""
#         echo "════════════════════════════════════════════════════"
#         echo "  Backend: $m"
#         echo "════════════════════════════════════════════════════"
#         # Rebuild args: strip --model-type and its value, inject --model-type $m
#         REBUILT_ARGS=()
#         SKIP_NEXT=false
#         for arg in "$@"; do
#             if $SKIP_NEXT; then SKIP_NEXT=false; continue; fi
#             if [[ "$arg" == "--model-type" ]]; then SKIP_NEXT=true; continue; fi
#             REBUILT_ARGS+=("$arg")
#         done
#         bash "$0" "${REBUILT_ARGS[@]}" --model-type "$m" || ANY_FAILED=1
#     done
#     if (( ANY_FAILED )); then
#         echo "WARNING: one or more backends failed"
#         exit 1
#     fi
#     exit 0
# fi

# # ── Per-backend MODEL_PATH defaults ─────────────────────────────────────────
# # Applied only when the user has not passed --model-path (i.e. MODEL_PATH is
# # still the liuhaotian default).  This lets you run:
# #   bash run_pipeline.sh --model-type internvl ...
# # without also having to specify --model-path every time.
# if [[ "$MODEL_PATH" == "liuhaotian/llava-v1.5-7b" ]]; then          # not overridden by user
#     if [[ "$MODEL_TYPE" == "internvl" ]]; then
#         MODEL_PATH="modern_vlms/pretrained/InternVL2_5-8B"           # default InternVL2.5-8B weights
#     elif [[ "$MODEL_TYPE" == "qwen2vl" ]]; then
#         MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct"  # default Qwen2.5-VL-7B weights
#     elif [[ "$MODEL_TYPE" == "llava-hf" ]]; then
#         MODEL_PATH="llava-hf/llava-1.5-7b-hf"                       # HF-format LLaVA 1.5 (has preprocessor_config.json)
#     elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
#         MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"        # LLaVA-OneVision-7B (Qwen2 backbone)
#     elif [[ "$MODEL_TYPE" == "llava-ov-si" ]]; then
#         MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-si-hf"        # LLaVA-OV-7B Stage 2 (single-image)
#     fi
# fi

# # ── Per-backend MODEL_NAME defaults ─────────────────────────────────────────
# # Applied only when the user has not overridden MODEL_NAME (i.e. it is still
# # the liuhaotian default).  Ensures output directories, --model args, and
# # marker file paths all reflect the actual model being run.
# if [[ "$MODEL_NAME" == "llava-1.5-7b" ]]; then                           # not overridden by user
#     if [[ "$MODEL_TYPE" == "internvl" ]]; then
#         MODEL_NAME="internvl2.5-8b"                                       # InternVL2.5-8B output dir name
#     elif [[ "$MODEL_TYPE" == "qwen2vl" ]]; then
#         MODEL_NAME="qwen2.5-vl-7b"                                        # Qwen2.5-VL-7B output dir name
#     elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
#         MODEL_NAME="llava-onevision-7b"                                    # LLaVA-OneVision-7B output dir name
#     fi
#     # llava-hf keeps MODEL_NAME="llava-1.5-7b" (same model as llava-liuhaotian)
# fi


# # ── Per-backend N_LAYERS defaults ───────────────────────────────────────────
# # Qwen2-7B backbone (used by llava-ov and qwen2vl) has 28 transformer layers,
# # not 32.  Override the default so sharding in step 3 does not produce
# # out-of-range layer indices.  InternVL2.5-8B uses InternLM2 (32 layers).
# if [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then
#     N_LAYERS=28
# fi
# # ── Short model alias for job names ─────────────────────────────────────────
# case "$MODEL_TYPE" in
#     llava-liuhaotian) SHORT_MODEL="l" ;;
#     llava-hf)         SHORT_MODEL="lhf" ;;
#     llava-ov)         SHORT_MODEL="lo" ;;
#     llava-ov-si)      SHORT_MODEL="losi" ;;
#     internvl)         SHORT_MODEL="int" ;;
#     qwen2vl)          SHORT_MODEL="qw" ;;
#     *)                SHORT_MODEL="${MODEL_TYPE:0:4}" ;;
# esac

# # ── Log directory and suffix — separate logs by backend ─────────────────────
# # Stores logs under logs/<mode>/<MODEL_TYPE>/ with _<MODEL_TYPE> suffix on each file,
# # so different backends never overwrite each other's logs.
# MODE_DIR=$( [[ "$MODE" == "test" ]] && echo "test" || echo "full" )
# LOG_DIR="logs/${MODE_DIR}/${MODEL_TYPE}"
# LOG_SUFFIX="_${MODEL_TYPE}"
# mkdir -p "$LOG_DIR"

# # ── Python interpreter — modern VLMs use their own venv ──────────────────
# # The old .venv (managed by uv) has transformers==4.37.2 + an older torch
# # which crashes with: ImportError: cannot import name '_get_cpp_backtrace'.
# # InternVL / Qwen2VL / LLaVA-HF / LLaVA-OV need modern_vlms/.venv which has compatible versions.
# #
# # CRITICAL: We must also unset VIRTUAL_ENV and strip PYTHONPATH so the old
# # .venv site-packages don't leak into the new interpreter via env inheritance.
# if [[ "$MODEL_TYPE" == "internvl" ]]; then
#     PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
#     if [[ ! -x "$PYTHON" ]]; then
#         echo "ERROR: $PYTHON not found — run:  cd modern_vlms/intervl_env && uv venv .venv_internvl --python 3.10 && uv pip install -r pyproject.toml"
#         exit 1
#     fi
# elif [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then
#     PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
#     if [[ ! -x "$PYTHON" ]]; then
#         echo "ERROR: $PYTHON not found — run:  cd modern_vlms && python -m venv .venv && pip install -r requirements.txt"
#         exit 1
#     fi
#     # Purge old venv from environment so its packages don't shadow modern_vlms
#     unset VIRTUAL_ENV 2>/dev/null || true
#     # Remove any .venv site-packages entries from PYTHONPATH
#     if [[ -n "${PYTHONPATH:-}" ]]; then
#         PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd ':' -)
#         export PYTHONPATH
#     fi
#     # Verify the interpreter loads the correct torch
#     echo "  Verifying modern venv torch..."
#     if ! "$PYTHON" -c "from torch._C import _get_cpp_backtrace" 2>/dev/null; then
#         echo "ERROR: modern_vlms/.venv torch is also broken. Reinstall:"
#         echo "  cd modern_vlms && .venv/bin/pip install --force-reinstall torch"
#         exit 1
#     fi
#     echo "  ✓ torch._C OK"
# else
#     PYTHON="$WORK_DIR/.venv/bin/python"
# fi

# # ── Shared HuggingFace cache (so all cluster nodes use the same download) ──
# export HF_HOME="${HF_HOME:-$WORK_DIR/.cache/huggingface}"

# # ── Mode-specific settings ────────────────────────────────────
# if [[ "$MODE" == "test" ]]; then
#     # Quick test: 6 fig3 images, 2 layers, top_n=5, no sharding
#     GEN_ARGS="--test_fig3"
#     DESC_FILE="results/1-describe/test/generated_descriptions_fig3_${MODEL_TYPE}.json"
#     CLASSIFY_ARGS="--num_images 6 --top_n 5 --layer_start 0 --layer_end 2 --n_permutations 100"
#     OUT_SUFFIX="_test_${MODEL_TYPE}"
#     SHARDS_EFFECTIVE=1   # no sharding in test mode
#     $QUEUE_SET || QUEUE="waic-risk"  # default queue for all modes
# else
#     # Full run: sharded across GPUs
#     GEN_ARGS=""
#     _DS="_${MODEL_NAME}${DESC_SUFFIX_USER:+_${DESC_SUFFIX_USER}}"
#     DESC_FILE="results/1-describe/full/generated_descriptions${_DS}.json"
#     CLASSIFY_ARGS=""
#     OUT_SUFFIX="${OUT_SUFFIX_USER}"
#     SHARDS_EFFECTIVE=$SHARDS
# fi

# # ── --limit N: quick-test override for step 1 ───────────────────────────────
# # Caps N_TOTAL_IMAGES and forces a single shard so you can verify the pipeline
# # end-to-end with just a few images before committing to a full run.
# if [[ -n "$GEN_LIMIT" ]]; then
#     N_TOTAL_IMAGES=$GEN_LIMIT
#     SHARDS_EFFECTIVE=1
#     echo "  ⚠ --limit $GEN_LIMIT: overriding N_TOTAL_IMAGES=$GEN_LIMIT, SHARDS_EFFECTIVE=1"
# fi

# # ── Normalise --step value ──────────────────────────────────────────────
# # Accept numbers (1-8), short names, and descriptive names.
# # Everything maps to the internal gate names used by step sections below.
# case "$STEP" in
#     1|describe|gd)            STEP="gd" ;;
#     2|merge_desc|merge_gd)    STEP="merge_gd" ;;
#     3|classify|cn)            STEP="cn" ;;
#     4|merge_class|merge_nc)   STEP="merge_nc" ;;
#     5|ablation|prune)         STEP="prune" ;;
#     merge_ablation)           STEP="merge_ablation" ;;
#     6|visualize|viz)          STEP="visualize" ;;
#     7|attention|attn)         STEP="attn" ;;
#     8|statistics|plot|stats)  STEP="plot" ;;
#     find_fig3|fig3_neurons)    STEP="find_fig3" ;;
#     check_collisions|collisions)  STEP="check_collisions" ;;
#     9|vtp)                    STEP="vtp" ;;
#     all|all_att)             ;;  # keep as-is
#     *) echo "ERROR: unknown step '$STEP'"; echo "  Valid: 1-8, merge_ablation, find_fig3, check_collisions, all, all_att"; exit 1 ;;
# esac

# # ── Resolve --clean default: if "auto", infer from --step ───────────────
# if [[ "$CLEAN_FROM" == "auto" ]]; then
#     case "$STEP" in
#         gd)        CLEAN_FROM=1 ;;
#         merge_gd)  CLEAN_FROM=2 ;;
#         cn)        CLEAN_FROM=3 ;;
#         merge_nc)  CLEAN_FROM=4 ;;
#         prune)     CLEAN_FROM=5 ;;
#         merge_ablation) CLEAN_FROM=5 ;;
#         visualize) CLEAN_FROM=6 ;;
#         attn)      CLEAN_FROM=7 ;;
#         plot)      CLEAN_FROM=8 ;;
#         vtp)       CLEAN_FROM=9 ;;
#         *)         CLEAN_FROM=1 ;;  # all / all_att → clean everything
#     esac
# fi

# # ── Step-all flags ──────────────────────────────────────────────────────
# # "all"      = steps 1-4, 5-6, 8    (standard chain, skips attn)
# # "all_att" = steps 1-8             (includes attention maps)
# STEP_ALL=false
# STEP_ALL_FULL=false
# if [[ "$STEP" == "all" || "$STEP" == "all_att" ]]; then
#     STEP_ALL=true
#     [[ "$STEP" == "all_att" ]] && STEP_ALL_FULL=true
# fi

# # ── Job name bases — step number + short model alias ────────────────────────
# # Produces concise LSF job names like 1_lav, 3_int_2, 5_q, 8_lav1
# JN1="1_${SHORT_MODEL}"     # describe
# JN2="2_${SHORT_MODEL}"     # merge_descriptions
# JN3="3_${SHORT_MODEL}"     # classify (FT + permutation)
# JN4="4_${SHORT_MODEL}"     # merge_classifications
# JN5="5_${SHORT_MODEL}"     # ablation_validate
# JN6="6_${SHORT_MODEL}"     # activation_maps
# JN6p="6p_${SHORT_MODEL}"   # patch_fig3
# JN7="7_${SHORT_MODEL}"     # attention_maps
# JN8="8_${SHORT_MODEL}"     # statistics
# JN9="9_${SHORT_MODEL}"     # vtp_analysis

# # For step 3 (classify): can't exceed 32 layers
# CLASSIFY_SHARDS=$SHARDS_EFFECTIVE
# if (( CLASSIFY_SHARDS > N_LAYERS )); then
#     CLASSIFY_SHARDS=$N_LAYERS
# fi

# # ── LSF helpers ──────────────────────────────────────────────
# # Check if a job is currently PEND or RUN in LSF
# is_job_active() {
#     local name=$1
#     bjobs -J "$name" -noheader 2>/dev/null | grep -qE "PEND|RUN"
# }

# # Check if a job is specifically PEND (not yet running)
# is_job_pending() {
#     local name=$1
#     bjobs -J "$name" -noheader 2>/dev/null | grep -q "PEND"
# }

# # Submit a GPU job with tiered gmem escalation.
# # Usage: bsub_tiered <bsub_args...> -- <command>
# # Submits at GPU_GMEM_TIERS[0], then spawns a background monitor that
# # re-submits at the next tier if the job is still PEND after GMEM_WAIT seconds.
# bsub_tiered() {
#     # Split args at '--' into bsub_args and cmd
#     local bsub_args=()
#     local cmd=""
#     local job_name=""
#     local log_file="" err_file=""
#     while [[ $# -gt 0 ]]; do
#         if [[ "$1" == "--" ]]; then
#             shift; cmd="$*"; break
#         fi
#         # capture -J value for job name
#         if [[ "$1" == "-J" ]]; then
#             bsub_args+=("$1"); shift
#             job_name="$1"
#             bsub_args+=("$1"); shift
#             continue
#         fi
#         # capture -oo/-eo paths for log clearing
#         if [[ "$1" == "-oo" ]]; then
#             bsub_args+=("$1"); shift
#             log_file="$1"
#             bsub_args+=("$1"); shift
#             continue
#         fi
#         if [[ "$1" == "-eo" ]]; then
#             bsub_args+=("$1"); shift
#             err_file="$1"
#             bsub_args+=("$1"); shift
#             continue
#         fi
#         bsub_args+=("$1"); shift
#     done

#     # Clear previous logs so they don't accumulate across runs
#     [[ -n "$log_file" ]] && rm -f "$log_file"
#     [[ -n "$err_file" ]] && rm -f "$err_file"

#     local n_tiers=${#GPU_GMEM_TIERS[@]}
#     local first_gmem="${GPU_GMEM_TIERS[0]}"

#     # Submit at first tier
#     bsub "${bsub_args[@]}" -gpu "num=1:gmem=$first_gmem" -R "$GPU_RES_BASE" "$cmd"

#     # If only one tier, no escalation needed
#     if (( n_tiers <= 1 )); then
#         return
#     fi

#     # Spawn background escalation monitor
#     (
#         set +e  # don't exit on bkill/bjobs failures
#         for ((t=1; t<n_tiers; t++)); do
#             sleep "$GMEM_WAIT"
#             # Check if still pending
#             if is_job_pending "$job_name"; then
#                 next_gmem="${GPU_GMEM_TIERS[$t]}"
#                 echo "  [escalate] $job_name still PEND after ${GMEM_WAIT}s → resubmit gmem=$next_gmem"
#                 bkill -J "$job_name" 2>/dev/null || true
#                 sleep 2  # let LSF process the kill
#                 bsub "${bsub_args[@]}" -gpu "num=1:gmem=$next_gmem" -R "$GPU_RES_BASE" "$cmd"
#             else
#                 # Job is running or done, stop escalating
#                 break
#             fi
#         done
#     ) &
# }

# # Counters
# SUBMITTED=0
# SKIPPED=0
# MERGE_SUBMITTED=0             # 1 if merge job was submitted this run
# CLS_SUBMITTED_JOBS_ALL=""      # space-separated list of cn job names for merge_nc deps
# MERGE_NC_SUBMITTED=""          # merge_nc job name if submitted this run (for plot deps)

# OUTPUT_DIR="${OUTPUT_DIR_USER:-results/3-classify/${MODE_DIR}}"

# # ── --clean in full mode: require confirmation to prevent accidents ────
# if $CLEAN && [[ "$MODE" == "full" ]]; then
#     if $WIPE; then
#         echo "WARNING: --clean --wipe in full mode will DELETE results + logs for $MODEL_NAME (steps $CLEAN_FROM–8)."
#     else
#         echo "WARNING: --clean in full mode will remove markers for $MODEL_NAME (steps $CLEAN_FROM–8)."
#     fi
#     read -p "  Are you sure? [y/N] " _confirm
#     if [[ "$_confirm" != "y" && "$_confirm" != "Y" ]]; then
#         echo "Aborted."
#         exit 0
#     fi
# fi

# # ── Banner ────────────────────────────────────────────────────
# echo ""
# echo "═══════════════════════════════════════════════════════════"
# echo "  Xu et al. Neuron Classification Pipeline"
# echo "═══════════════════════════════════════════════════════════"
# echo "  Mode:               $MODE"
# echo "  Backend:            $MODEL_TYPE"
# echo "  Model name:         $MODEL_NAME"
# echo "  Job alias:          $SHORT_MODEL (e.g. 1_${SHORT_MODEL}, 3_${SHORT_MODEL})"
# echo "  Python:             $PYTHON"
# echo "  Model path:         $MODEL_PATH"
# echo "  Log dir:            $LOG_DIR"
# echo "  Step:               $STEP"
# echo "  GPUs:               $SHARDS_EFFECTIVE"
# [[ -n "$GEN_LIMIT" ]] && echo "  Image limit:        $GEN_LIMIT (--limit)"
# echo "  Classify shards:    $CLASSIFY_SHARDS (max $N_LAYERS layers)"
# echo "  Local:              $LOCAL"
# echo "  Queue:              $QUEUE"
# echo "  Work dir:           $WORK_DIR"
# echo "  GPU gmem tiers:     ${GPU_GMEM_TIERS[*]} (wait ${GMEM_WAIT}s between)"
# echo "  Desc file:          $DESC_FILE"
# echo "  Output dir:         $OUTPUT_DIR"
# echo "  Ablation script:    $ABLATION_SCRIPT"
# echo "  Attention script:   $ATTN_SCRIPT"
# echo "  Prune images:       $PRUNE_IMAGES"
# echo "  Ablation parallel:  $ABLATION_ALL (--ablation-all for 18 parallel GPU jobs)"
# echo "  Ablation taxonomy:  $ABLATION_TAXONOMY (ft | pmbt | both)"  # default is now pmbt
# echo "  Ablation shards:    $ABLATION_SHARDS GPUs per config"
# echo "  Clean before run:   $CLEAN (from step $CLEAN_FROM, wipe=$WIPE)  --clean [N] markers only, add --wipe for full delete"
# echo "  Idempotent:         yes (skips completed/active jobs)"
# echo "═══════════════════════════════════════════════════════════"

# # Create directories (no log cleanup — needed for completion detection)
# mkdir -p "$WORK_DIR/results/1-describe/${MODE_DIR}/shards"
# mkdir -p "$WORK_DIR/$LOG_DIR"

# # ── CLEAN: remove markers (or full dirs with --wipe) to force re-run ───
# if $CLEAN; then
#     echo ""
#     if $WIPE; then
#         echo "  ── WIPE MODE ($MODE): deleting results + logs from step $CLEAN_FROM onwards for $MODEL_NAME ──"
#     else
#         echo "  ── CLEAN MODE ($MODE): removing markers from step $CLEAN_FROM onwards for $MODEL_NAME ──"
#     fi

#     # Kill active LSF jobs from CLEAN_FROM onwards
#     declare -A STEP_JOBS=(
#         [1]="$JN1"  [2]="$JN2"  [3]="$JN3"  [4]="$JN4"
#         [5]="$JN5"  [6]="$JN6 $JN6p"  [7]="$JN7"  [8]="$JN8"
#     )
#     for s in $(seq "$CLEAN_FROM" 8); do
#         for jname in ${STEP_JOBS[$s]:-}; do
#             if is_job_active "$jname" 2>/dev/null; then
#                 bkill -J "$jname" 2>/dev/null && echo "    bkill $jname"
#             fi
#         done
#     done
#     sleep 2  # let LSF register job kills before skip-checks

#     # Step 1: descriptions
#     if (( CLEAN_FROM <= 1 )); then
#         if $WIPE; then
#             [[ -f "$DESC_FILE" ]] && rm -f "$DESC_FILE" && echo "    rm $DESC_FILE"
#             # Also remove shards
#             for sf in results/1-describe/${MODE_DIR}/shards/gen_desc*"${MODEL_NAME}"*.json; do
#                 [[ -f "$sf" ]] && rm -f "$sf" && echo "    rm $sf"
#             done
#         else
#             [[ -f "$DESC_FILE" ]] && rm -f "$DESC_FILE" && echo "    rm $DESC_FILE (marker)"
#         fi
#     fi

#     # Step 3: classification stats (markers that trigger skip)
#     if (( CLEAN_FROM <= 3 )); then
#         _FT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"
#         _PM_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
#         if $WIPE; then
#             [[ -d "$OUTPUT_DIR/$MODEL_NAME" ]] && rm -rf "$OUTPUT_DIR/$MODEL_NAME" \
#                 && echo "    rm -rf $OUTPUT_DIR/$MODEL_NAME"
#         else
#             # Remove classification_stats JSON files (skip markers)
#             for f in "$_FT_DIR"/classification_stats_*.json "$_PM_DIR"/permutation_stats_*.json; do
#                 [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
#             done
#         fi
#     fi

#     # Step 4: merged classification stats
#     if (( CLEAN_FROM <= 4 )); then
#         _MERGE_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_all.json"
#         [[ -f "$_MERGE_FILE" ]] && rm -f "$_MERGE_FILE" && echo "    rm $_MERGE_FILE (marker)"
#     fi

#     # Step 5: ablation summaries
#     if (( CLEAN_FROM <= 5 )); then
#         _ABL_DIR="$OUTPUT_DIR/$MODEL_NAME/ablation"
#         if $WIPE; then
#             [[ -d "$_ABL_DIR" ]] && rm -rf "$_ABL_DIR" && echo "    rm -rf $_ABL_DIR"
#         else
#             # Remove ablation_summary.json from each sub-config
#             for f in "$_ABL_DIR"/*/ablation_summary.json; do
#                 [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
#             done
#         fi
#     fi

#     # Step 6: visualization marker
#     if (( CLEAN_FROM <= 6 )); then
#         _PATCH="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/fig3_patched.marker"
#         if $WIPE; then
#             _VIZ="$OUTPUT_DIR/$MODEL_NAME/fig3"
#             [[ -d "$_VIZ" ]] && rm -rf "$_VIZ" && echo "    rm -rf $_VIZ"
#         fi
#         [[ -f "$_PATCH" ]] && rm -f "$_PATCH" && echo "    rm $_PATCH (marker)"
#     fi

#     # Step 7: attention marker
#     if (( CLEAN_FROM <= 7 )); then
#         _ATTN_DIR="results/7-attention_maps"
#         if $WIPE; then
#             [[ -d "$_ATTN_DIR" ]] && rm -rf "$_ATTN_DIR" && echo "    rm -rf $_ATTN_DIR"
#         else
#             for f in "$_ATTN_DIR"/done*.marker; do
#                 [[ -f "$f" ]] && rm -f "$f" && echo "    rm $f (marker)"
#             done
#         fi
#     fi

#     # Step 8: plot marker
#     if (( CLEAN_FROM <= 8 )); then
#         _PLOT_DIR="results/8-statistics/$MODEL_NAME/${MODE_DIR}"
#         _PLOT_MARKER="$_PLOT_DIR/done.marker"
#         if $WIPE; then
#             [[ -d "$_PLOT_DIR" ]] && rm -rf "$_PLOT_DIR" && echo "    rm -rf $_PLOT_DIR"
#         else
#             [[ -f "$_PLOT_MARKER" ]] && rm -f "$_PLOT_MARKER" && echo "    rm $_PLOT_MARKER (marker)"
#         fi
#     fi

#     # Logs: only delete with --wipe
#     if $WIPE; then
#         for s in $(seq "$CLEAN_FROM" 8); do
#             STEP_LOG_NAMES=("1-describe" "2-merge_descriptions" "3-classify" "4-merge_classifications" "5-ablation_validate" "6-activation_maps" "7-attention_maps" "8-statistics")
#             SDIR="$LOG_DIR/${STEP_LOG_NAMES[$((s-1))]}"
#             if [[ -d "$SDIR" ]]; then
#                 rm -rf "$SDIR" && echo "    rm -rf $SDIR"
#             fi
#         done
#     fi

#     if $WIPE; then
#         echo "  ── WIPE complete (steps $CLEAN_FROM–8: results + logs deleted) ──"
#     else
#         echo "  ── CLEAN complete (steps $CLEAN_FROM–8: markers removed, results preserved) ──"
#     fi
#     echo ""
# fi

# # ═══════════════════════════════════════════════════════════════
# # STEP 1 (describe): Generate descriptions (sharded by image range)
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "gd" || $STEP_ALL == true ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 1: Generate descriptions"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/1-describe"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# if [[ "$MODE" == "test" ]]; then
#     JOB_NAME="${JN1}"
#     echo ""
#     echo "  ── generate_descriptions --test_fig3 ──"
#     # Skip if test description file exists or job is active
#     if [[ -s "$DESC_FILE" ]] || is_job_active "$JOB_NAME"; then
#         echo "  [skip] $JOB_NAME — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#     elif $LOCAL; then
#         (cd "$WORK_DIR" && $PYTHON code/generate_descriptions.py \
#             --model_type "$MODEL_TYPE" \
#             --original_model_path "$MODEL_PATH" \
#             --model_path "$MODEL_PATH" \
#             --output_path "results/1-describe/test/generated_descriptions.json" \
#             $GEN_ARGS) \
#             2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#     else
#         bsub_tiered -q $QUEUE \
#              -J "$JOB_NAME" \
#              -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#              -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
#              -- "cd $WORK_DIR && $PYTHON code/generate_descriptions.py \
#                      --model_type $MODEL_TYPE \
#                      --original_model_path $MODEL_PATH \
#                      --model_path $MODEL_PATH \
#                      --output_path results/1-describe/test/generated_descriptions.json \
#                      $GEN_ARGS"
#         echo "  → Job: $JOB_NAME (1 GPU, tiers: ${GPU_GMEM_TIERS[*]})"
#         SUBMITTED=$((SUBMITTED + 1))
#     fi
# else
#     # Full mode: shard across GPUs
#     echo ""
#     echo "  ── generate_descriptions — $SHARDS_EFFECTIVE shards ──"
#     echo "  Images per shard: ~$((N_TOTAL_IMAGES / SHARDS_EFFECTIVE))"

#     GEN_SUBMITTED_JOBS=()  # track which shard jobs were submitted this run
#     for ((s=0; s<SHARDS_EFFECTIVE; s++)); do
#         START_IDX=$((s * N_TOTAL_IMAGES / SHARDS_EFFECTIVE))
#         END_IDX=$(((s + 1) * N_TOTAL_IMAGES / SHARDS_EFFECTIVE))
#         JOB_NAME="gen_${s}_${SHORT_MODEL}"
#         SHARD_FILE="results/1-describe/full/shards/gen_desc${_DS}_shard${s}.json"

#         # Skip if shard output exists or job is active
#         if [[ -s "$SHARD_FILE" ]] || is_job_active "$JOB_NAME"; then
#             echo "  [skip] Shard $s ($JOB_NAME) — already done or active"
#             SKIPPED=$((SKIPPED + 1))
#             continue
#         fi
#         if $LOCAL; then
#             echo "  Shard $s: images [$START_IDX, $END_IDX) → $SHARD_FILE"
#             (cd "$WORK_DIR" && $PYTHON code/generate_descriptions.py \
#                 --model_type "$MODEL_TYPE" \
#                 --original_model_path "$MODEL_PATH" \
#                 --model_path "$MODEL_PATH" \
#                 --output_path "$SHARD_FILE" \
#                 --start_idx $START_IDX \
#                 --end_idx $END_IDX) \
#                 2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#         else
#             bsub_tiered -q $QUEUE \
#                  -J "$JOB_NAME" \
#                  -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#                  -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
#                  -- "cd $WORK_DIR && $PYTHON code/generate_descriptions.py \
#                      --model_type $MODEL_TYPE \
#                      --original_model_path $MODEL_PATH \
#                      --model_path $MODEL_PATH \
#                      --output_path $SHARD_FILE \
#                      --start_idx $START_IDX \
#                      --end_idx $END_IDX"
#             echo "  → Shard $s: [$START_IDX, $END_IDX) tiers: ${GPU_GMEM_TIERS[*]} → $JOB_NAME"
#             GEN_SUBMITTED_JOBS+=("$JOB_NAME")
#             SUBMITTED=$((SUBMITTED + 1))
#         fi
#     done

#     # ── Merge shards into single file ─────────────────────────
#     MERGE_JOB="${JN2}"
#     MERGE_CMD="$PYTHON -c \"
# import json, glob, os
# merged = {}
# for f in sorted(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json')):
#     with open(f) as fp:
#         merged.update(json.load(fp))
# os.makedirs(os.path.dirname('${DESC_FILE}'), exist_ok=True)
# with open('${DESC_FILE}', 'w') as fp:
#     json.dump(merged, fp, indent=2)
# n_shards = len(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json'))
# print(f'Merged {len(merged)} descriptions from {n_shards} shards → ${DESC_FILE}')
# \""

#     echo ""
#     echo "  ── Merge → $DESC_FILE ──"

#     # Skip if merged file exists or job is active
#     if [[ -s "$DESC_FILE" ]] || is_job_active "$MERGE_JOB"; then
#         echo "  [skip] $MERGE_JOB — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#     elif $LOCAL; then
#         (cd "$WORK_DIR" && eval "$MERGE_CMD") 2>&1 | tee "${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log"
#     else
#         # Build dependency on ALL active gen jobs (submitted this run OR still running)
#         BSUB_MERGE_ARGS=(-q "$QUEUE" \
#              -J "$MERGE_JOB" \
#              -oo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log" \
#              -eo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.err")
#         DEP_PARTS=""
#         for ((gs=0; gs<SHARDS_EFFECTIVE; gs++)); do
#             gn="gen_${gs}_${SHORT_MODEL}"
#             if is_job_active "$gn"; then
#                 if [[ -n "$DEP_PARTS" ]]; then
#                     DEP_PARTS="$DEP_PARTS && done($gn)"
#                 else
#                     DEP_PARTS="done($gn)"
#                 fi
#             fi
#         done
#         if [[ -n "$DEP_PARTS" ]]; then
#             BSUB_MERGE_ARGS+=(-w "$DEP_PARTS")
#             echo "  → Job: $MERGE_JOB (depends on ${#GEN_SUBMITTED_JOBS[@]} shard jobs)"
#         else
#             echo "  → Job: $MERGE_JOB (no deps — all shards already done)"
#         fi
#         rm -f "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.err"
#         bsub "${BSUB_MERGE_ARGS[@]}" "cd $WORK_DIR && $MERGE_CMD"
#         MERGE_SUBMITTED=1
#         SUBMITTED=$((SUBMITTED + 1))
#     fi
# fi

# fi  # end step 1 (describe)

# # ═══════════════════════════════════════════════════════════════
# # STEP 2 (merge_desc): Merge generated description shards
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "merge_gd" ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 2: Merge generated description shards"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/2-merge_descriptions"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# MERGE_CMD="$PYTHON -c \"
# import json, glob, os
# merged = {}
# for f in sorted(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json')):
#     with open(f) as fp:
#         merged.update(json.load(fp))
# os.makedirs(os.path.dirname('${DESC_FILE}'), exist_ok=True)
# with open('${DESC_FILE}', 'w') as fp:
#     json.dump(merged, fp, indent=2)
# n_shards = len(glob.glob('results/1-describe/full/shards/gen_desc${_DS}_shard*.json'))
# print(f'Merged {len(merged)} descriptions from {n_shards} shards → ${DESC_FILE}')
# \""

# echo ""
# echo "  ── Merging → $DESC_FILE ──"
# (cd "$WORK_DIR" && eval "$MERGE_CMD") 2>&1 | tee "${STEP_LOG_DIR}/merge_gd${LOG_SUFFIX}.log"

# fi  # end step 2 (merge_desc)

# # ═══════════════════════════════════════════════════════════════
# # STEP 3 (classify): Classify neurons (sharded by layer range)
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "cn" || $STEP_ALL == true ]]; then

# # In test + viz-fig3 mode, classify is unnecessary — patch_fig3 handles
# # the 6 specific neurons directly, so skip the entire cn step.
# STEP_LOG_DIR="${LOG_DIR}/3-classify"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"
# if [[ "$MODE" == "test" ]] && $VIZ_FIG3 && [[ $STEP_ALL == true ]]; then
#     echo ""
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     echo "  STEP 3: SKIPPED (test + --viz-fig3 → patch_fig3 handles neurons directly)"
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# else
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 3: Classify neurons"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# if [[ "$MODE" == "test" ]]; then
#     # Test mode: single job
#     JOB_NAME="${JN3}"
#     STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_layers0-2.json"
#     echo ""
#     echo "  ── $CLASSIFY_SCRIPT — test mode ──"

#     # Skip if stats file exists or job is active
#     if [[ -s "$STATS_FILE" ]] || is_job_active "$JOB_NAME"; then
#         echo "  [skip] $JOB_NAME — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#         # Track active job so merge_nc waits for it
#         if is_job_active "$JOB_NAME"; then
#             CLS_SUBMITTED_JOBS_ALL="$JOB_NAME"
#         fi
#     elif $LOCAL; then
#         (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
#             --model_type "$MODEL_TYPE" \
#             --original_model_path "$MODEL_PATH" \
#             --text_source generated \
#             --generated_desc_path "$DESC_FILE" \
#             --output_dir "$OUTPUT_DIR" \
#             --model "$MODEL_NAME" \
#             $CLASSIFY_ARGS) \
#             2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#     else
#         BSUB_ARGS=(-q "$QUEUE" \
#             -J "$JOB_NAME" \
#             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
#         if [[ $STEP_ALL == true ]] && is_job_active "${JN1}"; then
#             BSUB_ARGS+=(-w "done(${JN1})")
#         fi
#         bsub_tiered "${BSUB_ARGS[@]}" \
#             -- "cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
#                 --model_type $MODEL_TYPE \
#                 --original_model_path $MODEL_PATH \
#                 --text_source generated \
#                 --generated_desc_path $DESC_FILE \
#                 --output_dir $OUTPUT_DIR \
#                 --model $MODEL_NAME \
#                 $CLASSIFY_ARGS"
#         echo "  → Job: $JOB_NAME (tiers: ${GPU_GMEM_TIERS[*]})"
#         SUBMITTED=$((SUBMITTED + 1))
#         CLS_SUBMITTED_JOBS_ALL="$JOB_NAME"
#     fi
# else
#     # Full mode: shard by layer range
#     echo ""
#     echo "  ── $CLASSIFY_SCRIPT — $CLASSIFY_SHARDS shards ──"
#     echo "  Layers per shard: ~$((N_LAYERS / CLASSIFY_SHARDS))"

#     CLS_SUBMITTED_JOBS=()
#     for ((s=0; s<CLASSIFY_SHARDS; s++)); do
#         LAYER_START=$((s * N_LAYERS / CLASSIFY_SHARDS))
#         LAYER_END=$(((s + 1) * N_LAYERS / CLASSIFY_SHARDS))
#         JOB_NAME="${JN3}_${s}"
#         STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_layers${LAYER_START}-${LAYER_END}.json"

#         # Skip if stats file exists or job is active
#         if [[ -s "$STATS_FILE" ]] || is_job_active "$JOB_NAME"; then
#             echo "  [skip] Shard $s ($JOB_NAME) layers [$LAYER_START, $LAYER_END) — already done or active"
#             SKIPPED=$((SKIPPED + 1))
#             continue
#         fi
#         if $LOCAL; then
#             echo "  Shard $s: layers [$LAYER_START, $LAYER_END) → $OUTPUT_DIR"
#             (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
#                 --model_type "$MODEL_TYPE" \
#                 --original_model_path "$MODEL_PATH" \
#                 --text_source generated \
#                 --generated_desc_path "$DESC_FILE" \
#                 --output_dir "$OUTPUT_DIR" \
#                 --model "$MODEL_NAME" \
#                 --layer_start $LAYER_START \
#                 --layer_end $LAYER_END) \
#                 2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#         else
#             # Depend on merge job only if it was submitted this run
#             BSUB_ARGS=(-q "$QUEUE" \
#                 -J "$JOB_NAME" \
#                 -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#                 -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
#             if [[ $STEP_ALL == true ]] && { [[ "$MERGE_SUBMITTED" == "1" ]] || is_job_active "$JN2"; }; then
#                 BSUB_ARGS+=(-w "done($JN2)")
#             fi

#             bsub_tiered "${BSUB_ARGS[@]}" \
#                 -- "cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
#                     --model_type $MODEL_TYPE \
#                     --original_model_path $MODEL_PATH \
#                     --text_source generated \
#                     --generated_desc_path $DESC_FILE \
#                     --output_dir $OUTPUT_DIR \
#                     --model $MODEL_NAME \
#                     --layer_start $LAYER_START \
#                     --layer_end $LAYER_END"
#             echo "  → Shard $s: layers [$LAYER_START, $LAYER_END) tiers: ${GPU_GMEM_TIERS[*]} → $JOB_NAME"
#             CLS_SUBMITTED_JOBS+=("$JOB_NAME")
#             SUBMITTED=$((SUBMITTED + 1))
#         fi
#     done

#     CLS_SUBMITTED_JOBS_ALL="${CLS_SUBMITTED_JOBS[*]}"
# fi

# fi  # end test+viz-fig3 skip check

# fi  # end step 3 (classify)

# # ═══════════════════════════════════════════════════════════════
# # STEP 4 (merge_class): Merge per-shard classification results
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "merge_nc" || $STEP_ALL == true ]]; then

# STEP_LOG_DIR="${LOG_DIR}/4-merge_classifications"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"
# if [[ "$MODE" == "test" ]] && $VIZ_FIG3 && [[ $STEP_ALL == true ]]; then
#     echo ""
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     echo "  STEP 4: SKIPPED (test + --viz-fig3)"
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# else

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 4: Merge classification results"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# MERGE_NC_CMD="cd $WORK_DIR && python3 code/merge_classification.py \
#     --model_type $MODEL_TYPE \
#     --model $MODEL_NAME \
#     --output_dir $OUTPUT_DIR --plot"

# # Skip if merged output already exists and no classify jobs are pending
# MERGE_STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold/classification_stats_all.json"
# MERGE_NC_JOB_CHK="${JN4}"
# if [[ -s "$MERGE_STATS_FILE" ]] && [[ -z "${CLS_SUBMITTED_JOBS_ALL:-}" ]] && ! is_job_active "$MERGE_NC_JOB_CHK"; then
#     echo "  [skip] $MERGE_NC_JOB_CHK — already done ($(basename "$MERGE_STATS_FILE") exists)"
#     SKIPPED=$((SKIPPED + 1))
# elif is_job_active "$MERGE_NC_JOB_CHK"; then
#     echo "  [skip] $MERGE_NC_JOB_CHK — already active"
#     SKIPPED=$((SKIPPED + 1))
# else

# echo ""
# echo "  ── Merging → $OUTPUT_DIR ──"

# if $LOCAL || [[ "$STEP" == "merge_nc" ]]; then
#     # Run inline (local mode or standalone merge_nc)
#     (cd "$WORK_DIR" && python3 code/merge_classification.py \
#         --model_type "$MODEL_TYPE" \
#         --model "$MODEL_NAME" \
#         --output_dir "$OUTPUT_DIR" --plot)
# else
#     # Submit as LSF job dependent on all cn jobs
#     MERGE_NC_JOB="${JN4}"
#     BSUB_MERGE_NC_ARGS=(-q "$QUEUE" \
#         -J "$MERGE_NC_JOB" \
#         -oo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.log" \
#         -eo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.err")
#     # Build dependency on ALL active cn jobs (submitted this run OR still running)
#     if [[ -n "${CLS_SUBMITTED_JOBS_ALL:-}" ]]; then
#         DEP_PARTS=""
#         for jn in $CLS_SUBMITTED_JOBS_ALL; do
#             if [[ -n "$DEP_PARTS" ]]; then
#                 DEP_PARTS="$DEP_PARTS && done($jn)"
#             else
#                 DEP_PARTS="done($jn)"
#             fi
#         done
#         BSUB_MERGE_NC_ARGS+=(-w "$DEP_PARTS")
#         echo "  → Job: $MERGE_NC_JOB (depends on cn jobs)"
#     else
#         # Check for any active cls jobs from previous runs
#         DEP_PARTS=""
#         for ((cs=0; cs<CLASSIFY_SHARDS; cs++)); do
#             cn="${JN3}_${cs}"
#             if is_job_active "$cn"; then
#                 if [[ -n "$DEP_PARTS" ]]; then
#                     DEP_PARTS="$DEP_PARTS && done($cn)"
#                 else
#                     DEP_PARTS="done($cn)"
#                 fi
#             fi
#         done
#         if [[ -n "$DEP_PARTS" ]]; then
#             BSUB_MERGE_NC_ARGS+=(-w "$DEP_PARTS")
#             echo "  → Job: $MERGE_NC_JOB (depends on active cls jobs from previous run)"
#         else
#             echo "  → Job: $MERGE_NC_JOB (no deps — all cn jobs already done)"
#         fi
#     fi
#     rm -f "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NC_JOB}${LOG_SUFFIX}.err"
#     bsub "${BSUB_MERGE_NC_ARGS[@]}" "$MERGE_NC_CMD"
#     SUBMITTED=$((SUBMITTED + 1))
#     MERGE_NC_SUBMITTED="$MERGE_NC_JOB"
# fi

# fi  # end merge_nc skip check

# fi  # end test+viz-fig3 skip check

# fi  # end step 4 (merge_class)

# # ═══════════════════════════════════════════════════════════════
# # STEP find_fig3: Find candidate neurons for Figure 3
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "find_fig3" ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  Find Figure 3 neurons (from classification results)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# FIG3_DATA_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"

# if [[ ! -d "$FIG3_DATA_DIR/topn_heap" ]]; then
#     echo "  ERROR: Top-N Heap not found at $FIG3_DATA_DIR/topn_heap"
#     echo "  Run steps 1-4 first:  bash code/run_pipeline.sh --model-type $MODEL_TYPE --mode $MODE --step all"
#     exit 1
# fi

# echo "  Data dir:   $FIG3_DATA_DIR"
# echo "  Model type: $MODEL_TYPE"
# echo ""

# $PYTHON code/find_fig3_neurons.py \
#     --data_dir "$FIG3_DATA_DIR" \
#     --model_type "$MODEL_TYPE"

# fi  # end find_fig3

# # ═══════════════════════════════════════════════════════════════
# # STEP check_collisions: Check for image token collisions in descriptions
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "check_collisions" ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  Check image-token collisions in generated descriptions"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# if [[ ! -f "$DESC_FILE" ]]; then
#     echo "  ERROR: Description file not found: $DESC_FILE"
#     echo "  Run step 1 first:  bash code/run_pipeline.sh --model-type $MODEL_TYPE --mode $MODE --step 1"
#     exit 1
# fi

# echo "  Model type: $MODEL_TYPE"
# echo "  Model path: $MODEL_PATH"
# echo "  Desc file:  $DESC_FILE"
# echo ""

# $PYTHON code/check_token_collisions.py \
#     --model_type "$MODEL_TYPE" \
#     --model_path "$MODEL_PATH" \
#     --desc_path "$DESC_FILE"

# fi  # end check_collisions



# # ═══════════════════════════════════════════════════════════════
# # STEP 5 (ablation): Ablation validation of neuron taxonomy
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "prune" || $STEP_ALL == true ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 5: Ablation validation"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/5-ablation_validate"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"
# ABLATION_OUT="$OUTPUT_DIR/$MODEL_NAME/ablation"

# if [[ "$MODE" == "test" ]]; then
#     BASE_PRUNE_ARGS="--num_images 5 --conditions baseline ablate_vis ablate_text random --pope_num_questions 3"
#     PRUNE_IMAGES_EFF=5
#     CHAIR_NUM_IMAGES=2
#     ABLATION_SHARDS=1  # no sharding in test mode
#     # Reduce benchmark questions in test mode (override only if user didn't set explicitly)
#     [[ "$TRIVIAQA_NUM" == "2000" ]] && TRIVIAQA_NUM=10
#     [[ "$MMLU_NUM" == "2000" ]]     && MMLU_NUM=10
# else
#     BASE_PRUNE_ARGS="--num_images $PRUNE_IMAGES --conditions baseline ablate_vis ablate_text random"
#     PRUNE_IMAGES_EFF=$PRUNE_IMAGES
# fi

# if [[ -n "$POPE_PATH" ]]; then
#     BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --pope_path $POPE_PATH"
#     if [[ -n "$POPE_IMG_DIR" ]]; then
#         BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --pope_img_dir $POPE_IMG_DIR"
#     fi
# fi

# if [[ -n "$CHAIR_ANN_PATH" ]]; then
#     BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --chair_ann_path $CHAIR_ANN_PATH --chair_num_images $CHAIR_NUM_IMAGES"
#     if [[ -n "$POPE_IMG_DIR" ]]; then
#         BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --chair_img_dir $POPE_IMG_DIR"
#     fi
# fi

# if [[ -n "$TRIVIAQA_PATH" ]] && [[ "$MODEL_TYPE" != "llava-hf" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
#     BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --triviaqa_path $TRIVIAQA_PATH --triviaqa_num_questions $TRIVIAQA_NUM"
# fi

# if [[ -n "$MMLU_DIR" ]] && [[ "$MODEL_TYPE" != "llava-hf" && "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
#     BASE_PRUNE_ARGS="$BASE_PRUNE_ARGS --mmlu_dir $MMLU_DIR --mmlu_num_questions $MMLU_NUM"
# fi

# submit_ablation_job() {
#     local JOB_SUFFIX="$1"
#     local OUT_SUBDIR="$2"
#     local EXTRA_ARGS="$3"
#     local _LABELS_DIR="$4"
#     local _LABEL_SOURCE="$5"
#     local JOB_BASE="${JN5}_${JOB_SUFFIX}"
#     local JOB_OUT="${ABLATION_OUT}/${OUT_SUBDIR}"
#     local JOB_SUMMARY="${JOB_OUT}/ablation_summary.json"
#     local FULL_ARGS="$BASE_PRUNE_ARGS $EXTRA_ARGS"

#     if [[ -s "$JOB_SUMMARY" ]] || is_job_active "$JOB_BASE"; then
#         echo "  [skip] $JOB_BASE — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#         return
#     fi

#     local COMMON_CMD="$PYTHON $ABLATION_SCRIPT \
#         --model_type $MODEL_TYPE \
#         --original_model_path $MODEL_PATH \
#         --labels_dir $_LABELS_DIR \
#         --label_source $_LABEL_SOURCE \
#         --output_dir $JOB_OUT \
#         $FULL_ARGS"

#     # ── Single shard (test mode or ABLATION_SHARDS=1) ──────────────
#     if (( ABLATION_SHARDS <= 1 )); then
#         local JOB_NAME="$JOB_BASE"
#         echo "  → $JOB_NAME → $JOB_OUT  [${_LABEL_SOURCE}]"
#         rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
#               "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

#         if $LOCAL; then
#             (cd "$WORK_DIR" && $COMMON_CMD) \
#                 2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
#         else
#             local BSUB_ARGS=(-q "$QUEUE" \
#                 -J "$JOB_NAME" \
#                 -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
#                 -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err")
#             if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
#                 BSUB_ARGS+=(-w "done(${JN4})")
#             fi
#             bsub_tiered "${BSUB_ARGS[@]}" -- "cd $WORK_DIR && $COMMON_CMD"
#             SUBMITTED=$((SUBMITTED + 1))
#         fi
#         echo "  → Job: $JOB_NAME (1 GPU)"
#         return
#     fi

#     # ── Multi-shard (ABLATION_SHARDS > 1) ──────────────────────────
#     echo "  → $JOB_BASE → $JOB_OUT  [${_LABEL_SOURCE}] (${ABLATION_SHARDS} shards)"
#     local SHARD_JOB_NAMES=""
#     for (( s=0; s<ABLATION_SHARDS; s++ )); do
#         local SHARD_START=$(( s * PRUNE_IMAGES_EFF / ABLATION_SHARDS ))
#         local SHARD_END=$(( (s + 1) * PRUNE_IMAGES_EFF / ABLATION_SHARDS ))
#         local SHARD_NAME="${JOB_BASE}_${s}"
#         local SHARD_OUT="${JOB_OUT}/shards/shard_${s}"

#         rm -f "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.log" \
#               "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.err"

#         local BSUB_ARGS=(-q "$QUEUE" \
#             -J "$SHARD_NAME" \
#             -oo "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/$STEP_LOG_DIR/${SHARD_NAME}${LOG_SUFFIX}.err")
#         if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
#             BSUB_ARGS+=(-w "done(${JN4})")
#         fi
#         bsub_tiered "${BSUB_ARGS[@]}" \
#             -- "cd $WORK_DIR && $COMMON_CMD \
#                 --start_idx $SHARD_START --end_idx $SHARD_END \
#                 --output_dir $SHARD_OUT"

#         [[ -n "$SHARD_JOB_NAMES" ]] && SHARD_JOB_NAMES+=" && "
#         SHARD_JOB_NAMES+="done($SHARD_NAME)"
#         SUBMITTED=$((SUBMITTED + 1))
#     done

#     # Merge shard results into final output
#     local MERGE_NAME="${JOB_BASE}_merge"
#     rm -f "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.log" \
#           "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.err"
#     bsub -q "$QUEUE" -J "$MERGE_NAME" -w "$SHARD_JOB_NAMES" \
#         -oo "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.log" \
#         -eo "$WORK_DIR/$STEP_LOG_DIR/${MERGE_NAME}${LOG_SUFFIX}.err" \
#         "cd $WORK_DIR && $PYTHON $ABLATION_SCRIPT \
#             --merge_shards $JOB_OUT/shards \
#             --output_dir $JOB_OUT"
#     SUBMITTED=$((SUBMITTED + 1))
#     echo "  → ${ABLATION_SHARDS} shard jobs + 1 merge (tiers: ${GPU_GMEM_TIERS[*]})"
# }

# if $ABLATION_ALL; then

# TAXONOMY_CONFIGS=(
#     "xu|llm_fixed_threshold|ft"
#     "llm_permutation|llm_permutation|perm"
# )

# echo ""
# echo "  ── PARALLEL MODE: 9 approaches × 2 taxonomies × ${ABLATION_SHARDS} GPUs ──"

# for TAXCFG in "${TAXONOMY_CONFIGS[@]}"; do
#     IFS='|' read -r TAX_SOURCE TAX_SUBDIR TAX_PREFIX <<< "$TAXCFG"
#     TAX_LABELS_DIR="$LABELS_BASE/$MODEL_NAME/$TAX_SUBDIR"

#     echo ""
#     echo "  ── Taxonomy: ${TAX_SOURCE} (${TAX_LABELS_DIR}) ──"

#     submit_ablation_job "${TAX_PREFIX}_standard"      "${TAX_PREFIX}/1_standard"      "--ablation_method zero"                                                                                           "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_curve_label"   "${TAX_PREFIX}/2_curve_label"   "--ablation_method zero --curve --curve_steps $ABLATION_CURVE_STEPS --ranking_method label"                       "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_curve_cett"    "${TAX_PREFIX}/3_curve_cett"    "--ablation_method zero --curve --curve_steps $ABLATION_CURVE_STEPS --ranking_method cett --n_cett_images $ABLATION_N_CETT" "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_layer_0_10"    "${TAX_PREFIX}/4_layer_0_10"    "--ablation_method zero --layer_range 0-10"                                                                        "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_layer_11_21"   "${TAX_PREFIX}/5_layer_11_21"   "--ablation_method zero --layer_range 11-21"                                                                       "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_layer_22_31"   "${TAX_PREFIX}/6_layer_22_31"   "--ablation_method zero --layer_range 22-31"                                                                       "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_mean"          "${TAX_PREFIX}/7_mean"          "--ablation_method mean --n_stats_images $ABLATION_N_STATS"                                                        "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_clamp"         "${TAX_PREFIX}/9_clamp_high"    "--ablation_method clamp_high --n_stats_images $ABLATION_N_STATS"                                                  "$TAX_LABELS_DIR" "$TAX_SOURCE"
#     submit_ablation_job "${TAX_PREFIX}_top500"        "${TAX_PREFIX}/10_top_n_500"    "--ablation_method zero --top_n 500"                                                                               "$TAX_LABELS_DIR" "$TAX_SOURCE"
# done

# else

# EXTRA_PRUNE="--ablation_method $ABLATION_METHOD"
# EXTRA_PRUNE="$EXTRA_PRUNE --n_stats_images $ABLATION_N_STATS"
# EXTRA_PRUNE="$EXTRA_PRUNE --ranking_method $ABLATION_RANKING"
# EXTRA_PRUNE="$EXTRA_PRUNE --n_cett_images $ABLATION_N_CETT"

# if $ABLATION_CURVE; then EXTRA_PRUNE="$EXTRA_PRUNE --curve --curve_steps $ABLATION_CURVE_STEPS"; fi
# if [[ -n "$ABLATION_TOP_N" ]]; then EXTRA_PRUNE="$EXTRA_PRUNE --top_n $ABLATION_TOP_N"; fi
# if [[ -n "$ABLATION_LAYER_RANGE" ]]; then EXTRA_PRUNE="$EXTRA_PRUNE --layer_range $ABLATION_LAYER_RANGE"; fi

# # ── Run for each requested taxonomy ──────────────────────────
# if [[ "$ABLATION_TAXONOMY" == "ft" || "$ABLATION_TAXONOMY" == "both" ]]; then
#     LABELS_DIR="$LABELS_BASE/$MODEL_NAME/llm_fixed_threshold"
#     submit_ablation_job "ft" "ft" "$EXTRA_PRUNE" "$LABELS_DIR" "xu"
# fi

# if [[ "$ABLATION_TAXONOMY" == "pmbt" || "$ABLATION_TAXONOMY" == "both" ]]; then
#     LABELS_DIR="$LABELS_BASE/$MODEL_NAME/llm_permutation"
#     submit_ablation_job "perm" "perm" "$EXTRA_PRUNE" "$LABELS_DIR" "llm_permutation"
# fi
# fi  # end ABLATION_ALL check

# fi  # end step 5 (ablation)

# # ═══════════════════════════════════════════════════════════════
# # STEP merge_ablation: merge shards produced by the ablation step
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "merge_ablation" ]]; then

#     echo ""
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     echo "  STEP merge_ablation: merging ablation shard results"
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

#     ABLATION_OUT="$OUTPUT_DIR/$MODEL_NAME/ablation"
#     if [[ ! -d "$ABLATION_OUT" ]]; then
#         echo "  ERROR: ablation output not found: $ABLATION_OUT"
#         exit 1
#     fi

#     find "$ABLATION_OUT" -type d -name shards | while read -r sharddir; do
#         parentdir="$(dirname "$sharddir")"
#         echo "  merging shards in $parentdir"
#         $PYTHON $ABLATION_SCRIPT \
#             --merge_shards "$sharddir" \
#             --output_dir "$parentdir"
#     done

#     exit 0
# fi

# # ═══════════════════════════════════════════════════════════════
# # STEP 6 (visualize)
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "visualize" || $STEP_ALL == true ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 6: Figure 3 activation visualizations"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/6-activation_maps"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# VIZ_DATA_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"
# VIZ_OUT_DIR="$OUTPUT_DIR/$MODEL_NAME/fig3"
# VIZ_FT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"
# VIZ_PMBT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
# COCO_IMG_DIR="${COCO_IMG_DIR:-/home/projects/bagon/shared/coco2017/images/train2017/}"

# VIZ_ARGS="--types visual text multimodal"
# if $VIZ_FIG3; then
#     VIZ_ARGS="--fig3"
# elif $VIZ_FIG89; then
#     VIZ_ARGS="--fig89"
# elif $VIZ_SUPPLEMENTARY; then
#     VIZ_ARGS="--supplementary"
# fi

# PATCH_MARKER="$VIZ_DATA_DIR/fig3_patched.marker"
# if $VIZ_FIG3 && [[ ! -f "$PATCH_MARKER" ]]; then
#     PATCH_SCRIPT="code/patch_fig3_activations.py"
#     if $LOCAL; then
#         (cd "$WORK_DIR" && $PYTHON $PATCH_SCRIPT \
#             --data_dir "$VIZ_DATA_DIR" \
#             --coco_img_dir "$COCO_IMG_DIR" \
#             --generated_desc_path "$DESC_FILE" \
#             --model_type "$MODEL_TYPE" \
#             --original_model_path "$MODEL_PATH" \
#             --model_name "$MODEL_NAME" \
#             --device 0) \
#             2>&1 | tee "${STEP_LOG_DIR}/patch_fig3${LOG_SUFFIX}.log"
#         touch "$PATCH_MARKER"
#     else
#         PATCH_JOB="${JN6p}"
#         if ! is_job_active "$PATCH_JOB"; then
#             PATCH_BSUB_ARGS=(-q "$QUEUE" \
#                 -J "$PATCH_JOB" \
#                 -oo "$WORK_DIR/${STEP_LOG_DIR}/${PATCH_JOB}${LOG_SUFFIX}.log" \
#                 -eo "$WORK_DIR/${STEP_LOG_DIR}/${PATCH_JOB}${LOG_SUFFIX}.err")
#             # Wait for gen_desc in test mode so the description file exists
#             if [[ "$MODE" == "test" && "$STEP" == "all" ]] && is_job_active "${JN1}"; then
#                 PATCH_BSUB_ARGS+=(-w "done(${JN1})")
#             fi
#             bsub_tiered "${PATCH_BSUB_ARGS[@]}" \
#                 -- "cd $WORK_DIR && $PYTHON $PATCH_SCRIPT \
#                     --data_dir $VIZ_DATA_DIR \
#                     --coco_img_dir $COCO_IMG_DIR \
#                     --generated_desc_path $DESC_FILE \
#                     --model_type $MODEL_TYPE \
#                     --original_model_path $MODEL_PATH \
#                     --model_name $MODEL_NAME \
#                     --device 0 \
#                     && touch $PATCH_MARKER"
#         fi
#     fi
# fi

# JOB_NAME="${JN6}"
# if is_job_active "$JOB_NAME"; then
#     echo "  [skip] $JOB_NAME — already active"
#     SKIPPED=$((SKIPPED + 1))
# elif $LOCAL; then
#     (cd "$WORK_DIR" && $PYTHON $VIZ_SCRIPT \
#         --data_dir "$VIZ_DATA_DIR" \
#         --coco_img_dir "$COCO_IMG_DIR" \
#         --generated_desc_path "$DESC_FILE" \
#         --model_type "$MODEL_TYPE" \
#         --model_name "$MODEL_NAME" \
#         --pmbt_data_dir "$VIZ_PMBT_DIR" \
#         --output_dir "$VIZ_OUT_DIR" \
#         $VIZ_ARGS) \
#         2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
# else
#     BSUB_ARGS=(-q "$QUEUE" \
#         -J "$JOB_NAME" \
#         -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#         -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
#         -R "rusage[mem=98304]" -M 98304)
#     if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
#         BSUB_ARGS+=(-w "done(${JN4})")
#     fi
#     if is_job_active "${JN6p}"; then
#         BSUB_ARGS+=(-w "done(${JN6p})")
#     fi
#     rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
#     bsub "${BSUB_ARGS[@]}" \
#         "cd $WORK_DIR && $PYTHON $VIZ_SCRIPT \
#             --data_dir $VIZ_DATA_DIR \
#             --coco_img_dir $COCO_IMG_DIR \
#             --generated_desc_path $DESC_FILE \
#             --model_type $MODEL_TYPE \
#             --model_name $MODEL_NAME \
#             --pmbt_data_dir $VIZ_PMBT_DIR \
#             --output_dir $VIZ_OUT_DIR \
#             $VIZ_ARGS"
#     echo "  → Job: $JOB_NAME (CPU only)"
#     SUBMITTED=$((SUBMITTED + 1))
# fi

# fi  # end step 6 (visualize)

# # ═══════════════════════════════════════════════════════════════
# # STEP 7 (attention)
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "attn" ]] || $STEP_ALL_FULL; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 7: Attention analysis for reclassified neurons"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/7-attention_maps"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# ATTN_OUT_DIR="results/7-attention_maps"

# # Default test images (the 6 fig3 images from Xu et al.)
# ATTN_DEFAULT_IMAGES=(000000403170 000000065793 000000156852 000000323964 000000276332 000000060034)

# # Build list of image IDs to process
# ATTN_IMAGE_IDS=()
# if [[ -n "$ATTN_IMAGE_ID" ]]; then
#     ATTN_IMAGE_IDS=("$ATTN_IMAGE_ID")
# elif [[ -z "$ATTN_IMAGE_PATH" && -z "$ATTN_LAYER" && "$ATTN_N_SAMPLES" -le 1 ]]; then
#     # No explicit target — use defaults in test mode, skip otherwise
#     if [[ "$MODE" == "test" ]]; then
#         ATTN_IMAGE_IDS=("${ATTN_DEFAULT_IMAGES[@]}")
#         echo "  Using default test images: ${ATTN_IMAGE_IDS[*]}"
#     elif $STEP_ALL_FULL; then
#         echo "  [skip] step 7 (attn) — no --attn-image-id / --attn-image-path / --attn-nsamples given"
#     else
#         echo "  ERROR: --step attn requires --attn-image-id, --attn-image-path, or --attn-nsamples > 1"
#         exit 1
#     fi
# fi

# COCO_IMG_DIR="${COCO_IMG_DIR:-/home/projects/bagon/shared/coco2017/images/train2017/}"


# # Override heatmap layers for 28-layer models (Qwen2-7B backbone) if user
# # hasn't explicitly set --attn-layers.  Default "0 7 15 23 28 31" has indices
# # 28 and 31 which are out of range for models with 28 layers (indices 0-27).
# if [[ "$ATTN_HEATMAP_LAYERS" == "0 7 15 23 28 31" ]]; then
#     if [[ "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" || "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "internvl" ]]; then
#         ATTN_HEATMAP_LAYERS="0 5 10 15 20 27"
#     fi
# fi
# # Build common args (model, paths, heatmap layers)
# ATTN_COMMON_ARGS=""
# ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --heatmap_layers $ATTN_HEATMAP_LAYERS"
# ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --model_type $MODEL_TYPE"
# if [[ "$MODEL_TYPE" == "llava-hf" ]]; then
#     ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llava-1.5-7b-hf"
# elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
#     ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llava-onevision-qwen2-7b-ov-hf"
# else
#     ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --original_model_path $MODEL_PATH"
# fi
# ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --coco_img_dir $COCO_IMG_DIR"
# ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --generated_desc_path $DESC_FILE"
# ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --output_dir $ATTN_OUT_DIR"
# if [[ -n "$ATTN_HIGHLIGHTED_WORDS" ]]; then
#     ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --highlighted_words $ATTN_HIGHLIGHTED_WORDS"
# fi

# # ── Image ID loop (default test images or --attn-image-id) ──────────
# for _IMG_ID in "${ATTN_IMAGE_IDS[@]}"; do
#     ATTN_TAG="_${_IMG_ID}"
#     ATTN_MARKER="$ATTN_OUT_DIR/done${ATTN_TAG}.marker"
#     JOB_NAME="${JN7}${ATTN_TAG}"
#     ATTN_ARGS="--image_id $_IMG_ID $ATTN_COMMON_ARGS"

#     if [[ -f "$ATTN_MARKER" ]] || is_job_active "$JOB_NAME"; then
#         echo "  [skip] $JOB_NAME — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#         continue
#     fi

#     if $LOCAL; then
#         (cd "$WORK_DIR" && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0) \
#             2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#         touch "$ATTN_MARKER"
#     else
#         rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#               "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
#         BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
#             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
#         if [[ $STEP_ALL == true ]] && is_job_active "${JN4}"; then
#             BSUB_ARGS+=(-w "done(${JN4})")
#         fi
#         bsub_tiered "${BSUB_ARGS[@]}" \
#             -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0 \
#                 && touch $ATTN_MARKER"
#         echo "  → Job: $JOB_NAME (image $_IMG_ID)"
#         SUBMITTED=$((SUBMITTED + 1))
#     fi
# done

# # ── Custom image path (--attn-image-path) ───────────────────────────
# if [[ -n "$ATTN_IMAGE_PATH" ]]; then
#     ATTN_TAG="_custom"
#     ATTN_MARKER="$ATTN_OUT_DIR/done${ATTN_TAG}.marker"
#     JOB_NAME="${JN7}${ATTN_TAG}"
#     ATTN_ARGS="--image_path $ATTN_IMAGE_PATH $ATTN_COMMON_ARGS"

#     if [[ -f "$ATTN_MARKER" ]] || is_job_active "$JOB_NAME"; then
#         echo "  [skip] $JOB_NAME — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#     elif $LOCAL; then
#         (cd "$WORK_DIR" && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0) \
#             2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#         touch "$ATTN_MARKER"
#     else
#         rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#               "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
#         bsub_tiered -q "$QUEUE" -J "$JOB_NAME" \
#             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
#             -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0 \
#                 && touch $ATTN_MARKER"
#         echo "  → Job: $JOB_NAME (custom path)"
#         SUBMITTED=$((SUBMITTED + 1))
#     fi
# fi

# # ── Neuron auto-highlight mode (--attn-layer + --attn-neuron-idx + --attn-nsamples) ──
# if [[ -n "$ATTN_LAYER" && -n "$ATTN_NEURON_IDX" && "$ATTN_N_SAMPLES" -gt 1 ]]; then
#     ATTN_TAG="_layer${ATTN_LAYER}_neuron${ATTN_NEURON_IDX}"
#     ATTN_MARKER="$ATTN_OUT_DIR/done${ATTN_TAG}.marker"
#     JOB_NAME="${JN7}${ATTN_TAG}"
#     ATTN_ARGS="$ATTN_COMMON_ARGS --layer $ATTN_LAYER --neuron_idx $ATTN_NEURON_IDX --top_k $ATTN_TOP_K --n_samples $ATTN_N_SAMPLES"
#     ATTN_ARGS="$ATTN_ARGS --data_dir $OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold"

#     if [[ -f "$ATTN_MARKER" ]] || is_job_active "$JOB_NAME"; then
#         echo "  [skip] $JOB_NAME — already done or active"
#         SKIPPED=$((SKIPPED + 1))
#     elif [[ "$ATTN_GPUS" -gt 1 ]]; then
#         N_SHARDS="$ATTN_GPUS"
#         SAMPLES_PER_SHARD=$(( (ATTN_N_SAMPLES + N_SHARDS - 1) / N_SHARDS ))
#         SHARD_JOB_NAMES=""
#         for (( s=0; s<N_SHARDS; s++ )); do
#             S_START=$(( s * SAMPLES_PER_SHARD ))
#             S_END=$(( S_START + SAMPLES_PER_SHARD ))
#             if [[ "$S_END" -gt "$ATTN_N_SAMPLES" ]]; then S_END="$ATTN_N_SAMPLES"; fi
#             if [[ "$S_START" -ge "$ATTN_N_SAMPLES" ]]; then break; fi
#             SHARD_NAME="${JOB_NAME}_shard${s}"
#             SHARD_ARGS="$ATTN_ARGS --sample_start $S_START --sample_end $S_END"
#             bsub_tiered -q "$QUEUE" -J "$SHARD_NAME" \
#                 -oo "$WORK_DIR/${STEP_LOG_DIR}/${SHARD_NAME}${LOG_SUFFIX}.log" \
#                 -eo "$WORK_DIR/${STEP_LOG_DIR}/${SHARD_NAME}${LOG_SUFFIX}.err" \
#                 -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $SHARD_ARGS --device 0"
#             SUBMITTED=$((SUBMITTED + 1))
#             if [[ -z "$SHARD_JOB_NAMES" ]]; then SHARD_JOB_NAMES="done($SHARD_NAME)"
#             else SHARD_JOB_NAMES="$SHARD_JOB_NAMES && done($SHARD_NAME)"; fi
#         done
#         MERGE_NAME="${JOB_NAME}_merge"
#         bsub -q "$QUEUE" -J "$MERGE_NAME" -w "$SHARD_JOB_NAMES" \
#             -R "rusage[mem=4096]" \
#             -oo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/${STEP_LOG_DIR}/${MERGE_NAME}${LOG_SUFFIX}.err" \
#             "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT \
#                 --merge --layer $ATTN_LAYER --neuron_idx $ATTN_NEURON_IDX \
#                 --n_samples $ATTN_N_SAMPLES --output_dir $ATTN_OUT_DIR \
#                 && touch $ATTN_MARKER"
#         SUBMITTED=$((SUBMITTED + 1))
#     elif $LOCAL; then
#         (cd "$WORK_DIR" && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0) \
#             2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#         touch "$ATTN_MARKER"
#     else
#         rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#               "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
#         bsub_tiered -q "$QUEUE" -J "$JOB_NAME" \
#             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
#             -- "cd $WORK_DIR && $PYTHON $ATTN_SCRIPT $ATTN_ARGS --device 0 \
#                 && touch $ATTN_MARKER"
#         echo "  → Job: $JOB_NAME (neuron auto-highlight, ${ATTN_N_SAMPLES} samples)"
#         SUBMITTED=$((SUBMITTED + 1))
#     fi
# fi
# fi  # end step 7 (attention)

# # ═══════════════════════════════════════════════════════════════
# # STEP 8 (statistics)
# # ═══════════════════════════════════════════════════════════════
# if [[ "$STEP" == "plot" || $STEP_ALL == true ]]; then

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  STEP 8: Fig7 cross-model comparison"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# STEP_LOG_DIR="${LOG_DIR}/8-statistics"
# mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

# PLOT_OUT_BASE="results/8-statistics/cross-model/${MODE_DIR}"
# PLOT_DPI=200
# PLOT_MARKER="$PLOT_OUT_BASE/done.marker"

# JOB_NAME="${JN8}"
# if [[ -f "$PLOT_MARKER" ]] || is_job_active "$JOB_NAME"; then
#     echo "  [skip] $JOB_NAME — already done or active"
#     SKIPPED=$((SKIPPED + 1))
# else

# PLOT_WRAPPER="$WORK_DIR/${STEP_LOG_DIR}/plot_wrapper${OUT_SUFFIX}.sh"
# cat > "$PLOT_WRAPPER" << 'PLOTEOF'
# #!/usr/bin/env bash
# set -euo pipefail
# cd "$WORK_DIR"

# # ── Auto-discover all models with classification results ──
# FT_DIRS=()
# PMBT_DIRS=()
# NAMES=()
# TYPES=()

# # Known model → model_type mapping
# declare -A NAME_TO_TYPE=(
#     ["llava-1.5-7b"]="llava-liuhaotian"
#     ["internvl2.5-8b"]="internvl"
#     ["qwen2.5-vl-7b"]="qwen2vl"
#     ["llava-onevision-7b"]="llava-ov"
# )

# echo "  Scanning for available models in $OUTPUT_DIR ..."
# for model_dir in "$OUTPUT_DIR"/*/; do
#     mname=$(basename "$model_dir")
#     ft_dir="$model_dir/llm_fixed_threshold"
#     pmbt_dir="$model_dir/llm_permutation"
#     mtype="${NAME_TO_TYPE[$mname]:-llava-hf}"

#     # Check FT has merged labels (at least one layer dir with neuron_labels.json)
#     ft_has_labels=false
#     if [[ -d "$ft_dir" ]]; then
#         for lbl in "$ft_dir"/*/neuron_labels.json; do
#             [[ -f "$lbl" ]] && ft_has_labels=true && break
#         done
#     fi

#     # Check PMBT has merged labels
#     pmbt_has_labels=false
#     if [[ -d "$pmbt_dir" ]]; then
#         for lbl in "$pmbt_dir"/*/neuron_labels_permutation.json; do
#             [[ -f "$lbl" ]] && pmbt_has_labels=true && break
#         done
#         # Also check neuron_labels.json fallback
#         if ! $pmbt_has_labels; then
#             for lbl in "$pmbt_dir"/*/neuron_labels.json; do
#                 [[ -f "$lbl" ]] && pmbt_has_labels=true && break
#             done
#         fi
#     fi

#     if $ft_has_labels; then
#         FT_DIRS+=("$ft_dir")
#         NAMES+=("$mname")
#         TYPES+=("$mtype")
#         echo "    ✓ $mname (FT: $ft_dir)"
#     fi

#     if $pmbt_has_labels; then
#         PMBT_DIRS+=("$pmbt_dir")
#         echo "    ✓ $mname (PMBT: $pmbt_dir)"
#     fi
# done

# PLOT_OK=0

# # ── FT cross-model comparison (PNG) ──────────────────────
# if [[ ${#FT_DIRS[@]} -ge 1 ]]; then
#     echo ""; echo "  [1] Fig7 FT cross-model PNG (${#FT_DIRS[@]} models)"
#     if $PYTHON $PLOT_SCRIPT \
#         --data_dirs "${FT_DIRS[@]}" \
#         --model_names "${NAMES[@]}" \
#         --model_types "${TYPES[@]}" \
#         --output_dir "$PLOT_OUT_BASE/cross-model" \
#         --title_prefix "Fixed-Threshold " \
#         --dpi "$PLOT_DPI" --format png --fig7; then
#         PLOT_OK=$((PLOT_OK + 1))
#     else
#         echo "  FAILED: FT fig7 PNG"
#     fi
# else
#     echo "  SKIP FT fig7: no models found with neuron labels"
# fi

# # ── FT cross-model comparison (PDF) ──────────────────────
# if [[ ${#FT_DIRS[@]} -ge 1 ]]; then
#     echo ""; echo "  [2] Fig7 FT cross-model PDF"
#     if $PYTHON $PLOT_SCRIPT \
#         --data_dirs "${FT_DIRS[@]}" \
#         --model_names "${NAMES[@]}" \
#         --model_types "${TYPES[@]}" \
#         --output_dir "$PLOT_OUT_BASE/cross-model" \
#         --title_prefix "Fixed-Threshold " \
#         --dpi 300 --format pdf --fig7; then
#         PLOT_OK=$((PLOT_OK + 1))
#     else
#         echo "  FAILED: FT fig7 PDF"
#     fi
# fi

# # ── PMBT cross-model comparison (PNG) ────────────────────
# if [[ ${#PMBT_DIRS[@]} -ge 1 ]]; then
#     # Build label_files list: try permutation first
#     PMBT_LFILES=()
#     for pd in "${PMBT_DIRS[@]}"; do
#         found_perm=false
#         for lbl in "$pd"/*/neuron_labels_permutation.json; do
#             [[ -f "$lbl" ]] && found_perm=true && break
#         done
#         if $found_perm; then
#             PMBT_LFILES+=("neuron_labels_permutation.json")
#         else
#             PMBT_LFILES+=("neuron_labels.json")
#         fi
#     done

#     echo ""; echo "  [3] Fig7 PMBT cross-model PNG (${#PMBT_DIRS[@]} models)"
#     if $PYTHON $PLOT_SCRIPT \
#         --data_dirs "${PMBT_DIRS[@]}" \
#         --model_names "${NAMES[@]}" \
#         --model_types "${TYPES[@]}" \
#         --label_files "${PMBT_LFILES[@]}" \
#         --output_dir "$PLOT_OUT_BASE/cross-model" \
#         --title_prefix "Permutation-Test " \
#         --dpi "$PLOT_DPI" --format png --fig7; then
#         PLOT_OK=$((PLOT_OK + 1))
#     else
#         echo "  FAILED: PMBT fig7 PNG"
#     fi
# fi

# # ── PMBT cross-model comparison (PDF) ────────────────────
# if [[ ${#PMBT_DIRS[@]} -ge 1 ]]; then
#     echo ""; echo "  [4] Fig7 PMBT cross-model PDF"
#     if $PYTHON $PLOT_SCRIPT \
#         --data_dirs "${PMBT_DIRS[@]}" \
#         --model_names "${NAMES[@]}" \
#         --model_types "${TYPES[@]}" \
#         --label_files "${PMBT_LFILES[@]}" \
#         --output_dir "$PLOT_OUT_BASE/cross-model" \
#         --title_prefix "Permutation-Test " \
#         --dpi 300 --format pdf --fig7; then
#         PLOT_OK=$((PLOT_OK + 1))
#     else
#         echo "  FAILED: PMBT fig7 PDF"
#     fi
# fi

# echo ""; echo "  ALL PLOTS COMPLETE → $PLOT_OUT_BASE ($PLOT_OK succeeded)"
# if [[ "$PLOT_OK" -eq 0 ]]; then
#     echo "  ERROR: No plots succeeded — not marking as done"
#     exit 1
# fi
# PLOTEOF

# chmod +x "$PLOT_WRAPPER"

# if $LOCAL; then
#     PYTHON="$PYTHON" PLOT_SCRIPT="$PLOT_SCRIPT" OUTPUT_DIR="$OUTPUT_DIR" \
#     PLOT_OUT_BASE="$PLOT_OUT_BASE" PLOT_DPI="$PLOT_DPI" WORK_DIR="$WORK_DIR" \
#         bash "$PLOT_WRAPPER" 2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
#     if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
#         touch "$PLOT_MARKER"
#     else
#         echo "  WARNING: plot wrapper failed — not marking as done"
#     fi
# else
#     BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
#         -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
#         -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err")
#     if [[ $STEP_ALL == true ]]; then
#         # Prefer waiting on merge_nc (submitted this run OR still active)
#         if [[ -n "${MERGE_NC_SUBMITTED:-}" ]]; then
#             BSUB_ARGS+=(-w "done($MERGE_NC_SUBMITTED)")
#         elif is_job_active "${JN4}"; then
#             BSUB_ARGS+=(-w "done(${JN4})")
#         # Fallback: wait for classify if merge_nc was skipped
#         elif [[ -n "${CLS_SUBMITTED_JOBS_ALL:-}" ]]; then
#             BSUB_ARGS+=(-w "done(${CLS_SUBMITTED_JOBS_ALL%% *})")
#         elif [[ "$MODE" == "test" ]] && is_job_active "${JN3}"; then
#             BSUB_ARGS+=(-w "done(${JN3})")
#         fi
#     fi
#     rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
#     bsub "${BSUB_ARGS[@]}" \
#         "cd $WORK_DIR && \
#          PYTHON='$PYTHON' PLOT_SCRIPT='$PLOT_SCRIPT' OUTPUT_DIR='$OUTPUT_DIR' \
#          PLOT_OUT_BASE='$PLOT_OUT_BASE' PLOT_DPI='$PLOT_DPI' WORK_DIR='$WORK_DIR' \
#          bash $PLOT_WRAPPER && touch $PLOT_MARKER"
#     echo "  → Job: $JOB_NAME (CPU only)"
#     SUBMITTED=$((SUBMITTED + 1))
# fi

# fi  # end skip check

# fi  # end step 8 (statistics)

# # ═══════════════════════════════════════════════════════════════
# # Summary
# # ═══════════════════════════════════════════════════════════════
# echo ""
# echo "═══════════════════════════════════════════════════════════"
# if $LOCAL; then
#     echo "  PIPELINE COMPLETE (local mode)"
# else
#     echo "  SUBMITTED: $SUBMITTED   SKIPPED: $SKIPPED (already done/active)"
#     echo ""
#     echo "  Monitor:  bjobs -q $QUEUE       (jobs: *_${SHORT_MODEL}*)"
#     echo "  Logs:     ls ${LOG_DIR}/*/*.log"
# fi
# echo ""
# echo "  Description file:   $DESC_FILE"
# echo "  Classification dir: $OUTPUT_DIR/"
# echo "  Ablation dir:       results/3-classify/${MODE_DIR}/$MODEL_NAME/ablation/"
# echo "═══════════════════════════════════════════════════════════"
# exit 0

