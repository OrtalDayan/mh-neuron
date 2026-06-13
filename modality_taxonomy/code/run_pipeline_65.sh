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
#   bash run_pipeline.sh --step 9                     # (ablation) ablation validation
#   bash run_pipeline.sh --step merge_ablation         # merge existing ablation shard results
#   bash run_pipeline.sh --step 10                     # (visualize) Figure 3 activation maps
#   bash run_pipeline.sh --step 10 --viz-fig3          # reproduce exact Xu Figure 3 panels (a)-(f)
#   bash run_pipeline.sh --step 10 --viz-fig3 --viz-taxonomy both  # FT + PMBT labels
#   bash run_pipeline.sh --step 10 --viz-fig89         # reproduce Xu Figures 8 & 9
#   bash run_pipeline.sh --step 10 --viz-supplementary # reproduce Figures 15-17
#   bash run_pipeline.sh --step 9 --attn-image-id 000000189475  # attention analysis (single image)
#   bash run_pipeline.sh --step 9 --attn-image-path /path/to/image.jpg --attn-words "dough nut pink"
#   bash run_pipeline.sh --step all_att --mode test            # uses 6 default fig3 images for attn
#   bash run_pipeline.sh --step 10                     # (statistics) all charts (FT + PMBT)
#   bash run_pipeline.sh --step 12                    # (layer_plots) per-layer trend figure + LaTeX tables
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
#   bash run_pipeline.sh --step 9 --pope-path pope/coco_pope_random.json
#   bash run_pipeline.sh --step 9 --pope-path pope/coco_pope_random.json \
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
MODEL_TYPE_SET=false     # tracks if --model-type was explicitly passed
MODEL_PATH="liuhaotian/llava-v1.5-7b"  # HF Hub ID or local path to model weights
MODEL_NAME="llava-1.5-7b"  # must match --model default in $CLASSIFY_SCRIPT
MODEL_NAME_SET=false       # true if user passed --model-name explicitly
OUT_SUFFIX_USER=""        # user-provided suffix for output dirs
CLASSIFY_SCRIPT="code/neuron_modality_statistical.py"  # classification script for step 3 (classify)
CLEAN_DESC_SCRIPT="code/clean_descriptions.py"        # step 1c: remove degenerate tails from descriptions
HOOK_POINT="gate"                                      # gate | gate_up | attn — hook point for classification
IMPORTANCE_WEIGHT=false                                # --importance-weight: weight activations by output-projection norms
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
GEN_MAX_TOKENS=300         # --gen-max-tokens: max_new_tokens for generate_descriptions.py
GEN_MIN_TOKENS=100         # --gen-min-tokens: min_new_tokens for generate_descriptions.py
GEN_LONG_DESC=false        # --long-desc: raise max to 1024 (keeps min=100), save to separate dirs
OUTPUT_DIR_USER=""         # override classification output dir
PRUNE_IMAGES=100           # number of images for ablation validation
POPE_PATH="data/POPE/output/coco/coco_pope_random.json"   # path to POPE jsonl (test mode only)
POPE_DIR="data/POPE/output/coco"                           # directory with all 3 strategies (publication)
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
SUPPRESS_LINE1=false       # if true, omit panel/layer/neuron line from fig3 headers (for combine workflow)

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
RESCORE=false                  # if true, step 22 re-runs GPT scoring only (--reuse, no GPU)
SCORE_LOCAL=false              # if true, step 22 runs local letter-extraction scoring (no API)
INLINE_EVAL=false              # if true, step 22 uses in-memory ablation (phase 5) instead of VLMEvalKit
INLINE_CONDITIONS="all"          # which conditions: "all" or comma-separated (e.g. "baseline")
INLINE_HOOKS="gate,gate_up,attn"  # which hooks: gate (original), gate_up, attn
INLINE_BENCHMARKS="POPE,MV_Text_Dominant,MV_Vision_Only,TriviaQA"  # per benchmark by default
INLINE_LIMIT=0                   # samples per benchmark (0=full, e.g. 20 for smoke test)

# ── Phase 4 (ranked ablation sweep) settings ──
P4_RANKING="D_x_norm"             # D | norm | D_x_norm | D_then_norm
P4_SWEEP_FRACS="0.01,0.05,0.1,0.25,0.5,1.0"  # fractions of each category to ablate
P4_RANDOM_TRIALS=5                 # random baselines per fraction
P4_CATEGORIES="visual,text,multimodal"  # categories to ablate
P4_BENCHMARKS="MV_Text_Dominant,MV_Vision_Only"  # BRV paper extremes (text vs vision)
P4_LIMIT=0                        # samples per benchmark (0=full, e.g. 10 for smoke test)
P4_BASELINE_ONLY=false            # --p4-baseline: run baseline only (no ablation)
P4_COMBINED_ATTN=false            # --p4-combined-attn: ablate BOTH MLP neurons + attention heads at same fraction

CLEAN_FROM="auto"                  # clean from this step onwards (auto=match --step, or explicit 1-8)
CLEAN_TO="auto"                    # clean up to this step (auto=same as CLEAN_FROM, i.e. single step)

# ── Step 16/17 (weight merging) settings ──
MERGE_BASE_LLM_PATH=""           # base LLM path (auto-set per model)
MERGE_MATH_LLM_PATH=""           # math LLM path for text_inject (Method 1)
MERGE_DONOR_VLM_PATH=""          # donor VLM path for visual_transplant (Method 5)
MERGE_DONOR_LABEL_DIR=""         # PMBT label dir for donor VLM
MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8 0.75 0.7 0.5 0.3"  # lambda sweep: BRV range (0.8-0.95) + extreme (0.3-0.7)
                                                             # Auto-narrowed per model type below (step 16)
MERGE_FORMULA="brv"                  # "brv" (θ'=(1-lam)·θ_vlm+lam·θ_reason, trade-off) or "additive" (θ'=θ_vlm+lam·τ, purely additive)
MERGE_INCLUDE_UNIFORM=true       # also run uniform (no-mask) BRV baseline — on by default
MERGE_INCLUDE_MULTIMODAL=true    # also run text+multimodal mask variant — on by default
MERGE_INCLUDE_VISUAL_ONLY=true  # also run visual-only mask variant (negative control) — on by default
MERGE_INCLUDE_VISUAL_MULTI=true # also run visual+multimodal mask variant — on by default
MERGE_INCLUDE_RANDOM=true      # also run random mask (same count as text, sparsity control) — on by default
MERGE_INCLUDE_MULTIMODAL_ONLY=true # also run multimodal-only mask variant — on by default
MERGE_EVAL_POPE=false              # run POPE eval in step 13 (off by default, use --eval-pope to enable)
MERGE_EVAL_CHAIR=false             # run CHAIR eval in step 13 (off by default, use --eval-chair to enable)

# Step 18 (compose_merge): compose step-16 text injection + step-24 SRF, and step-23 SNRF + step-24 SRF
COMBINE_LAMBDA_16="0.1"          # which step-16 lambda to use for the combined model
COMBINE_LAMBDA_17="0.1"          # which step-17 lambda to use for the combined model

# Step 19 (evaluate): VLMEvalKit evaluation of merged models (steps 16, 17, 18)
EVAL_BENCHMARKS="MathVista_MINI MathVerse_MINI_Vision_Only MMStar POPE"  # benchmarks to run
EVAL_WHICH="baseline 16 text_multi visual_only visual_multi uniform multimodal_only random snrf snrf_random srf srf_random"   # which models to evaluate
MERGE_SAVE_MODEL=true            # save merged model weights to disk (always on by default)

# Step 21 (tune_lambda): BRV-style lambda tuning on MathVista
TUNE_LAMBDAS="0.95 0.9 0.85 0.8 0.75 0.7"    # lambda values to search over (must exist in step 16 sweep)
TUNE_MASKS="16 text_multi uniform visual_only visual_multi multimodal_only random"          # which masks to tune (16=text_inject, text_multi, visual_only, visual_multi, uniform, multimodal_only, random)

# Step 25 (weight_merge): BRV-compatible + PMBT-guided weight merging
P5_MERGE_SCRIPT="code/merge_pmbt.py"
P5_MODE="base"                   # base (BRV uniform) | pmbt (selective)
P5_ALPHA="0.9"                   # default alpha (BRV uses 0.9)
P5_ALPHA_TEXT=""                  # alpha for text neurons (pmbt mode)
P5_ALPHA_VISUAL=""               # alpha for visual neurons (pmbt mode)
P5_ALPHA_MULTI=""                # alpha for multimodal neurons (pmbt mode)
P5_PMBT_SCOPE="both"            # mlp | attn | both — which neuron types to merge selectively
P5_KV_MERGE=false               # --p5-kv-merge: apply selective merge to k/v via GQA majority vote
P5_ALPHA_OTHER=""               # --p5-alpha-other: alpha for layernorms/embeddings (empty = use --p5-alpha)
P5_MIXED_HOOK=false             # --p5-mixed-hook: gate_proj uses gate labels, up_proj/down_proj use gate_up labels
P5_MLP_PROJS="gate,up,down"     # --p5-mlp-projs: which MLP projections to merge (comma-separated subset of gate,up,down)
P5_BASELINE=false                # also run baseline eval (no merge)
P5_EVAL=true                     # evaluate merged model after merge
P5_JUDGE="gpt-4o-mini"          # GPT judge model for scoring
P5_BENCHMARKS="MathVista_MINI,MathVerse_MINI_Text_Dominant,MathVerse_MINI_Text_Lite,MathVerse_MINI_Vision_Intensive,MathVerse_MINI_Vision_Dominant,MathVerse_MINI_Vision_Only,MMStar,DynaMath,MathVision_MINI,MM-Math"

# ── Step 10 (halluc_score) settings ──
HALLUC_SCORE_SCRIPT="code/halluc_score_neurons.py"
MERGE_STEERING_SCRIPT="code/merge_steering_results.py"    # step 12: merge steering results across alphas
PLOT_STEERING_SCRIPT="code/plot_steering_results.py"      # step 13: ECCV publication figures
MERGE_SCRIPT="code/neuron_weight_merge.py"          # step 16/17: PMBT-guided weight merging
EVAL_SINGLE_SCRIPT="code/eval_single_model.py"     # step 16 Phase B: per-model parallel eval
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
STEP7_RANKING="random_trials"              # random_trials (default) — equal-fraction taxonomy validation
FULL_ABLATION=false                        # if true, step 7 also runs phase 3 (SNRF-style 100% ablation)
EXTRA_BENCHMARKS=false                     # if true, step 7 also evaluates CHAIR + MMLU (slower)
BASELINE_ONLY=false                        # if true, step 7 runs only baseline (no ablation, no trials)
POPE_ONLY=false                            # if true, step 7 evaluates only POPE (no MathVerse/TriviaQA)
STEP7_BENCHMARKS=""                        # comma-separated benchmarks for trials (empty=all). e.g. POPE,TriviaQA,MV_Text_Dominant,MV_Vision_Only
VLMEVAL=false                              # if true, use VLMEvalKit directly (gold-standard evaluation)
VLMEVAL_JUDGE="gpt-4o-mini"                # GPT model for VLMEvalKit answer extraction
VLMEVAL_DIR="modern_vlms/VLMEvalKit_brv"         # BRV's VLMEvalKit (has --merge_model support)
MATHVERSE_DIR="data/mathverse_vlmeval"       # path to MathVerse data (788 Qs/subtask, matches VLMEvalKit)

# GPU memory tiers — escalate through tiers when jobs stay PEND
GPU_GMEM_TIERS=("20G")                # override with --gmem 40G,20G
RUN_SUFFIX=""                          # append to log names + output dirs (e.g. _gmem_80)
LAYER_LIST=""                          # comma-separated layers for step 3 (e.g. 8,9,13,14,31)
GMEM_WAIT=120                          # seconds to wait before escalating (override with --gmem-wait)
GPU_RES_BASE="rusage[mem=24576] order[-gpu_maxfactor]"
GPU_EXCLUSIVE=true                     # --exclusive: request exclusive GPU (mode=exclusive_process)

# Dataset constants
N_TOTAL_IMAGES=23000
N_LAYERS=32
GEN_LIMIT=""                       # --limit N: override N_TOTAL_IMAGES for quick testing

# ── Parse args ────────────────────────────────────────────────
_USER_SET_LAMBDA=false
_USER_SET_FORMULA=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)    LOCAL=true; shift ;;
        --step)     STEP="$2"; shift 2 ;;
        --mode)     MODE="$2"; shift 2 ;;
        --shards)   SHARDS="$2"; shift 2 ;;
        --limit)    GEN_LIMIT="$2"; shift 2 ;;
        --queue)    QUEUE="$2"; QUEUE_SET=true; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; MODEL_TYPE_SET=true; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; MODEL_NAME_SET=true; shift 2 ;;
        --suffix)   OUT_SUFFIX_USER="$2"; shift 2 ;;
        --classify-script) CLASSIFY_SCRIPT="$2"; shift 2 ;;
        --ablation-script) ABLATION_SCRIPT="$2"; shift 2 ;;
        --desc-suffix) DESC_SUFFIX_USER="$2"; shift 2 ;;
        --gen-max-tokens) GEN_MAX_TOKENS="$2"; shift 2 ;;
        --gen-min-tokens) GEN_MIN_TOKENS="$2"; shift 2 ;;
        --long-desc) GEN_LONG_DESC=true; shift ;;
        --output-dir) OUTPUT_DIR_USER="$2"; shift 2 ;;
        --hook-point) HOOK_POINT="$2"; shift 2 ;;
        --importance-weight) IMPORTANCE_WEIGHT=true; shift ;;
        --p4-ranking) P4_RANKING="$2"; shift 2 ;;
        --p4-sweep-fracs) P4_SWEEP_FRACS="$2"; shift 2 ;;
        --p4-random-trials) P4_RANDOM_TRIALS="$2"; shift 2 ;;
        --p4-categories) P4_CATEGORIES="$2"; shift 2 ;;
        --p4-benchmarks) P4_BENCHMARKS="$2"; shift 2 ;;
        --p4-limit) P4_LIMIT="$2"; shift 2 ;;
        --p4-baseline) P4_BASELINE_ONLY=true; shift ;;
        --p4-combined-attn) P4_COMBINED_ATTN=true; shift ;;
        # Step 25 (weight_merge) args
        --p5-mode) P5_MODE="$2"; [[ "$P5_MODE" == "uniform" ]] && P5_MODE="base"; shift 2 ;;
        --p5-alpha) P5_ALPHA="$2"; shift 2 ;;
        --p5-alpha-text) P5_ALPHA_TEXT="$2"; shift 2 ;;
        --p5-alpha-visual) P5_ALPHA_VISUAL="$2"; shift 2 ;;
        --p5-alpha-multi) P5_ALPHA_MULTI="$2"; shift 2 ;;
        --p5-pmbt-scope) P5_PMBT_SCOPE="$2"; shift 2 ;;
        --p5-kv-merge) P5_KV_MERGE=true; shift ;;
        --p5-alpha-other) P5_ALPHA_OTHER="$2"; shift 2 ;;
        --p5-mixed-hook) P5_MIXED_HOOK=true; shift ;;
        --p5-mlp-projs) P5_MLP_PROJS="$2"; shift 2 ;;
        --p5-baseline) P5_BASELINE=true; shift ;;
        --p5-no-eval) P5_EVAL=false; shift ;;
        --p5-judge) P5_JUDGE="$2"; shift 2 ;;
        --p5-benchmarks) P5_BENCHMARKS="$2"; shift 2 ;;
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
        --suppress-line1) SUPPRESS_LINE1=true; shift ;;
        --gmem)     IFS=',' read -ra GPU_GMEM_TIERS <<< "$2"; shift 2 ;;
        --exclusive) GPU_EXCLUSIVE=true; shift ;;
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
        --merge-base-llm) MERGE_BASE_LLM_PATH="$2"; shift 2 ;;
        --merge-math-llm) MERGE_MATH_LLM_PATH="$2"; shift 2 ;;
        --merge-donor-vlm) MERGE_DONOR_VLM_PATH="$2"; shift 2 ;;
        --merge-donor-labels) MERGE_DONOR_LABEL_DIR="$2"; shift 2 ;;
        --merge-lambda) MERGE_LAMBDA_SWEEP="$2"; _USER_SET_LAMBDA=true; shift 2 ;;
        --merge-formula) MERGE_FORMULA="$2"; _USER_SET_FORMULA=true; shift 2 ;;  # additive (ours) or brv (BRV paper exact)
        --merge-lambda16) MERGE_LAMBDA16="$2"; shift 2 ;;
        --merge-lambda17) MERGE_LAMBDA17="$2"; shift 2 ;;
        --merge-save-model) MERGE_SAVE_MODEL=true; shift ;;
        --no-save-model) MERGE_SAVE_MODEL=false; shift ;;
        --merge-uniform) MERGE_INCLUDE_UNIFORM=true; shift ;;
        --no-uniform) MERGE_INCLUDE_UNIFORM=false; shift ;;
        --no-multimodal) MERGE_INCLUDE_MULTIMODAL=false; shift ;;
        --no-visual-only) MERGE_INCLUDE_VISUAL_ONLY=false; shift ;;
        --no-visual-multi) MERGE_INCLUDE_VISUAL_MULTI=false; shift ;;
        --no-random) MERGE_INCLUDE_RANDOM=false; shift ;;
        --no-multimodal-only) MERGE_INCLUDE_MULTIMODAL_ONLY=false; shift ;;
        --uniform-only) MERGE_INCLUDE_UNIFORM=true; MERGE_INCLUDE_MULTIMODAL=false; MERGE_INCLUDE_VISUAL_ONLY=false; MERGE_INCLUDE_VISUAL_MULTI=false; MERGE_INCLUDE_RANDOM=false; MERGE_INCLUDE_MULTIMODAL_ONLY=false; shift ;;
        --eval-pope) MERGE_EVAL_POPE=true; shift ;;
        --eval-chair) MERGE_EVAL_CHAIR=true; shift ;;
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
        --rescore)      RESCORE=true; shift ;;
        --score-local)  SCORE_LOCAL=true; shift ;;
        --full-ablation) FULL_ABLATION=true; shift ;;
        --extra-benchmarks) EXTRA_BENCHMARKS=true; shift ;;
        --baseline-only) BASELINE_ONLY=true; shift ;;
        --pope-only) POPE_ONLY=true; shift ;;
        --benchmarks) STEP7_BENCHMARKS="$2"; shift 2 ;;
        --vlmeval) VLMEVAL=true; shift ;;
        --judge) VLMEVAL_JUDGE="$2"; shift 2 ;;
        --inline)       INLINE_EVAL=true; shift ;;
        --inline-condition) INLINE_CONDITIONS="$2"; shift 2 ;;
        --inline-hooks) INLINE_HOOKS="$2"; shift 2 ;;
        --inline-benchmarks) INLINE_BENCHMARKS="$2"; shift 2 ;;
        --inline-limit) INLINE_LIMIT="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Normalize model-type aliases ────────────────────────────────────────────
# Accept short names at the CLI so users can type e.g. --model-type llava
# instead of --model-type llava-liuhaotian.  Full names still work.
_normalize_model_type() {
    case "$1" in
        llava|llava-1.5|llava-1.5-7b)  echo "llava-liuhaotian" ;;
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
_VALID_MODELS="llava-liuhaotian llava-mistral llava-llama3 llava-ov llava-ov-si internvl qwen2vl qwen2vl-brv qwen25vl-7b qwen25vl-3b idefics2 all"
_VALID_ALIASES="llava intern qwen"
if [[ "$MODEL_TYPE" != *","* ]]; then
    _found=false
    for _vm in $_VALID_MODELS; do
        [[ "$MODEL_TYPE" == "$_vm" ]] && _found=true && break
    done
    if ! $_found; then
        echo "ERROR: unknown --model-type '$MODEL_TYPE'"
        echo ""
        echo "  Valid names:    llava-liuhaotian  llava-mistral  llava-llama3  llava-ov  internvl  qwen2vl  qwen2vl-brv  idefics2  all"
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
            echo "  Valid names:    llava-liuhaotian  llava-mistral  llava-llama3  llava-ov  internvl  qwen2vl  qwen2vl-brv  idefics2"
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
ALL_MODELS=(llava-liuhaotian llava-mistral llava-llama3 llava-ov llava-ov-si internvl qwen2vl qwen2vl-brv qwen25vl-7b qwen25vl-3b)
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
        MODEL_PATH="modern_vlms/pretrained/Qwen2-VL-7B-Instruct"    # Qwen2-VL-7B weights
    elif [[ "$MODEL_TYPE" == "qwen2vl-brv" ]]; then
        MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct"  # Qwen2.5-VL-7B weights (was default qwen2vl)
    elif [[ "$MODEL_TYPE" == "qwen25vl-7b" ]]; then
        MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct"  # Qwen2.5-VL-7B (same as qwen2vl-brv)
    elif [[ "$MODEL_TYPE" == "qwen25vl-3b" ]]; then
        MODEL_PATH="modern_vlms/pretrained/Qwen2.5-VL-3B-Instruct"  # Qwen2.5-VL-3B
    elif [[ "$MODEL_TYPE" == "idefics2" ]]; then
        MODEL_PATH="modern_vlms/pretrained/idefics2-8b"               # BRV Table 3 exact model (local)
    elif [[ "$MODEL_TYPE" == "llava-mistral" ]]; then
        MODEL_PATH="llava-hf/llava-v1.6-mistral-7b-hf"               # LLaVA-Next-Mistral-7B (BRV reproduction)
    elif [[ "$MODEL_TYPE" == "llava-llama3" ]]; then
        MODEL_PATH="llava-hf/llama3-llava-next-8b-hf"                # LLaVA-Next-LLaMA3-8B (BRV main model)
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
        MODEL_NAME="qwen2-vl-7b"                                          # Qwen2-VL-7B output dir name
    elif [[ "$MODEL_TYPE" == "qwen2vl-brv" ]]; then
        MODEL_NAME="qwen2.5-vl-7b"                                        # Qwen2.5-VL-7B output dir name
    elif [[ "$MODEL_TYPE" == "qwen25vl-7b" ]]; then
        MODEL_NAME="qwen2.5-vl-7b"                                        # Qwen2.5-VL-7B output dir name
    elif [[ "$MODEL_TYPE" == "qwen25vl-3b" ]]; then
        MODEL_NAME="qwen2.5-vl-3b"                                        # Qwen2.5-VL-3B output dir name
    elif [[ "$MODEL_TYPE" == "idefics2" ]]; then
        MODEL_NAME="idefics2-8b"                                           # Idefics2-8B (BRV Table 3)
    elif [[ "$MODEL_TYPE" == "llava-ov" ]]; then
        MODEL_NAME="llava-onevision-7b"                                    # LLaVA-OneVision-7B output dir name
    elif [[ "$MODEL_TYPE" == "llava-mistral" ]]; then
        MODEL_NAME="llava-next-mistral-7b"                                 # LLaVA-Next-Mistral-7B output dir name
    elif [[ "$MODEL_TYPE" == "llava-llama3" ]]; then
        MODEL_NAME="llava-next-llama3-8b"                                  # LLaVA-Next-LLaMA3-8B output dir name
    fi
    # llava-liuhaotian keeps MODEL_NAME="llava-1.5-7b"
fi


# ── Per-backend N_LAYERS defaults ───────────────────────────────────────────
# Qwen2-7B backbone (used by llava-ov and qwen2vl) has 28 transformer layers,
# not 32.  Override the default so sharding in step 3 does not produce
# out-of-range layer indices.  InternVL2.5-8B uses InternLM2 (32 layers).
if [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "qwen2vl-brv" || "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then
    N_LAYERS=28
elif [[ "$MODEL_TYPE" == "qwen25vl-3b" ]]; then
    N_LAYERS=36
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
    llava-mistral|llava-llama3|idefics2) N_LLM_NEURONS=458752 ;;
    llava-ov|llava-ov-si|qwen2vl|qwen2vl-brv|qwen25vl-7b) N_LLM_NEURONS=530432 ;;
    qwen25vl-3b) N_LLM_NEURONS=396288 ;;  # 36 × 11008
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
    qwen2vl-brv)      SHORT_MODEL="qwb" ;;
    qwen25vl-7b)      SHORT_MODEL="q25" ;;
    qwen25vl-3b)      SHORT_MODEL="q253" ;;
    idefics2)         SHORT_MODEL="idef" ;;
    *)                SHORT_MODEL="${MODEL_TYPE:0:4}" ;;
esac

# ── Log directory and suffix — separate logs by backend ─────────────────────
# Stores logs under logs/<mode>/<MODEL_TYPE>/ with _<MODEL_TYPE> suffix on each file,
# so different backends never overwrite each other's logs.
# MODE_DIR:          used for description paths (results/1-describe/{MODE_DIR}/)
# CLASSIFY_MODE_DIR: used for classification output + logs
#   test         → test / test          (6 fig3 images, 2 layers)
#   1-layer-100-imgs-test → full / full  (100 images, 1 layer, saves to full/ to verify paths)
#   1-layer      → full / full          (23K images, 1 layer, saves to full/)
#   full         → full / full          (23K images, all layers)
case "$MODE" in
    test)          MODE_DIR="test";  CLASSIFY_MODE_DIR="test" ;;
    1-layer-100-imgs-test)  MODE_DIR="full";  CLASSIFY_MODE_DIR="full" ;;  # saves to full/ to verify paths
    1-layer)       MODE_DIR="full";  CLASSIFY_MODE_DIR="full" ;;
    *)             MODE_DIR="full";  CLASSIFY_MODE_DIR="full" ;;
esac
LOG_DIR="logs/${CLASSIFY_MODE_DIR}/${MODEL_NAME}"
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
elif [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "qwen2vl-brv" || "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "qwen25vl-3b" || "$MODEL_TYPE" == "idefics2" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" || "$MODEL_TYPE" == "llava-mistral" || "$MODEL_TYPE" == "llava-llama3" ]]; then
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

# ── Token limits for description generation ──────────────────────────────
# --long-desc raises max to 2048 so verbose models can finish naturally.
# Degeneration past natural end is handled by clean_descriptions.py.
if $GEN_LONG_DESC; then
    GEN_MAX_TOKENS=2048
    # Keep GEN_MIN_TOKENS at default (100) — ensures enough text tokens for PMBT
fi
# Build suffix for desc directories when using non-default token limits
_GEN_DIR_SUFFIX=""
if [[ "$GEN_MAX_TOKENS" != "300" || "$GEN_MIN_TOKENS" != "100" ]]; then
    _GEN_DIR_SUFFIX="_min${GEN_MIN_TOKENS}_max${GEN_MAX_TOKENS}"
fi

# ── Mode-specific settings ────────────────────────────────────
if [[ "$MODE" == "test" ]]; then
    # Quick test: 6 fig3 images, 2 layers, top_n=5, no sharding
    GEN_ARGS="--test_fig3"
    DESC_FILE="results/1-describe/test${_GEN_DIR_SUFFIX}/generated_descriptions_fig3_${MODEL_NAME}.json"
    CLASSIFY_ARGS="--num_images 6 --top_n 5 --layer_start 0 --layer_end 2 --n_permutations 100 --hook_point $HOOK_POINT${_GEN_DIR_SUFFIX:+ --output_suffix $_GEN_DIR_SUFFIX}"
    OUT_SUFFIX="_test_${MODEL_TYPE}"
    SHARDS_EFFECTIVE=1   # no sharding in test mode
    $QUEUE_SET || QUEUE="waic-risk"  # default queue for all modes
elif [[ "$MODE" == "1-layer" ]]; then
    # 1-layer smoke test: full 23K images, 1 layer, verifies output dirs (saves to full/)
    GEN_ARGS=""
    _DS="_${MODEL_NAME}${DESC_SUFFIX_USER:+_${DESC_SUFFIX_USER}}"
    DESC_FILE="results/1-describe/full${_GEN_DIR_SUFFIX}/generated_descriptions${_DS}.json"
    CLASSIFY_ARGS="--layer_start 0 --layer_end 1 --hook_point $HOOK_POINT${_GEN_DIR_SUFFIX:+ --output_suffix $_GEN_DIR_SUFFIX}"
    OUT_SUFFIX="${OUT_SUFFIX_USER}"
    SHARDS_EFFECTIVE=1
    $QUEUE_SET || QUEUE="waic-risk"
elif [[ "$MODE" == "1-layer-100-imgs-test" ]]; then
    # 1-layer quick verify: 100 images, 1 layer, own dir (doesn't pollute full results)
    GEN_ARGS=""
    _DS="_${MODEL_NAME}${DESC_SUFFIX_USER:+_${DESC_SUFFIX_USER}}"
    DESC_FILE="results/1-describe/full${_GEN_DIR_SUFFIX}/generated_descriptions${_DS}.json"
    CLASSIFY_ARGS="--num_images 100 --top_n 5 --n_permutations 100 --layer_start 0 --layer_end 1 --hook_point $HOOK_POINT${_GEN_DIR_SUFFIX:+ --output_suffix $_GEN_DIR_SUFFIX}"
    OUT_SUFFIX="${OUT_SUFFIX_USER}"
    SHARDS_EFFECTIVE=1
    $QUEUE_SET || QUEUE="waic-risk"
else
    # Full run: sharded across GPUs
    GEN_ARGS=""
    _DS="_${MODEL_NAME}${DESC_SUFFIX_USER:+_${DESC_SUFFIX_USER}}"
    DESC_FILE="results/1-describe/full${_GEN_DIR_SUFFIX}/generated_descriptions${_DS}.json"
    CLASSIFY_ARGS="--hook_point $HOOK_POINT${_GEN_DIR_SUFFIX:+ --output_suffix $_GEN_DIR_SUFFIX}"
    OUT_SUFFIX="${OUT_SUFFIX_USER}"
    SHARDS_EFFECTIVE=$SHARDS
fi

# ── Importance weighting flag ─────────────────────────────────────────────
# Prepend --importance_weight (before --output_suffix which may have empty value)
$IMPORTANCE_WEIGHT && CLASSIFY_ARGS="--importance_weight $CLASSIFY_ARGS"

# ── Hook point suffix for output directories ──────────────────────────────
# Classification results go to separate directories per hook point
# (e.g., llm_permutation_gate_up/ or llm_permutation_attn/)
# _HOOK_SUFFIX: long form for output dirs (e.g. _gate_up)
# _HOOK_SHORT:  short form for LSF job names and log files (e.g. _gup)
if [[ "$HOOK_POINT" == "gate_up" ]]; then
    _HOOK_SUFFIX="_gate_up"
    _HOOK_SHORT="_gup"
elif [[ "$HOOK_POINT" == "attn" ]]; then
    _HOOK_SUFFIX="_attn"
    _HOOK_SHORT="_att"
else
    _HOOK_SUFFIX=""
    _HOOK_SHORT=""
fi
# Append _iw when importance weighting is enabled
$IMPORTANCE_WEIGHT && _HOOK_SUFFIX="${_HOOK_SUFFIX}_iw" && _HOOK_SHORT="${_HOOK_SHORT}_iw"

# Override LOG_SUFFIX to include hook point — separates log files per hook
LOG_SUFFIX="_${MODEL_TYPE}${_HOOK_SHORT}${RUN_SUFFIX}"

# Add hook + token subdirectory to LOG_DIR (e.g. logs/full/idefics2-8b/gup_min100_max2048/)
_HOOK_NAME="${_HOOK_SHORT#_}"  # strip leading underscore: gup, att, g
_LOG_HOOK_SUFFIX="/${_HOOK_NAME}${_GEN_DIR_SUFFIX}"  # e.g. /gup_min100_max2048

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
    1c|clean_desc)            STEP="clean_desc" ;;
    2|merge_desc|merge_gd)    STEP="merge_gd" ;;
    3|classify|cn)            STEP="cn" ;;
    4|merge_class|merge_nc)   STEP="merge_nc" ;;
    4s|summary)               STEP="summary" ;;
    8c|combine_fig3)          STEP="combine_fig3" ;;
    8a|fig3_all|fig3_hooks)   STEP="fig3_all" ;;
    5|check_collisions|collisions)  STEP="check_collisions" ;;
    6|find_fig3|fig3_neurons)    STEP="find_fig3" ;;
    7|equal_fraction_ablation|5b)     STEP="equal_fraction_ablation" ;;
    8|visualize|viz)          STEP="visualize" ;;
    9|attention|attn)         STEP="attn" ;;
    10|statistics|plot|stats)  STEP="plot" ;;
    11|vit_analysis|vit)         STEP="vit_analysis" ;;
    12|layer_plots|layer_tables) STEP="layer_plots" ;;
    13|text_inject)              STEP="text_inject" ;;
    14|snrf|snrf_merge)                  STEP="snrf" ;;
    15|srf|srf_edit)                     STEP="srf" ;;
    16|tune_lambda)                     STEP="tune_lambda" ;;
    17|select_lambda)                    STEP="select_lambda" ;;
    18|compose_layer1)                   STEP="compose_layer1" ;;
    19|evaluate|benchmark_eval)      STEP="evaluate" ;;
    20|summarize|summary)             STEP="summarize" ;;
    21|weight_diff_rank|rank)    STEP="weight_diff_rank" ;;
    22|mathverse_ablation|mv_ablation) STEP="mathverse_ablation" ;;
    23|compare_hooks|hook_cmp) STEP="compare_hooks" ;;
    24|ranked_ablation|phase4) STEP="ranked_ablation" ;;
    25|weight_merge|merge_pmbt) STEP="weight_merge" ;;
    all|all_att)             ;;  # keep as-is
    *) echo "ERROR: unknown step '$STEP'"; echo "  Valid: 1-23, find_fig3, check_collisions, all, all_att"; exit 1 ;;
esac

# ── Resolve --clean default: if "auto", infer from --step ───────────────
if [[ "$CLEAN_FROM" == "auto" ]]; then
    case "$STEP" in
        gd)        CLEAN_FROM=1 ;;
        merge_gd)  CLEAN_FROM=2 ;;
        cn)        CLEAN_FROM=3 ;;
        merge_nc)  CLEAN_FROM=4 ;;
        check_collisions) CLEAN_FROM=5 ;;
        find_fig3) CLEAN_FROM=6 ;;
        equal_fraction_ablation) CLEAN_FROM=7 ;;
        visualize) CLEAN_FROM=8 ;;
        attn)      CLEAN_FROM=9 ;;
        plot)      CLEAN_FROM=10 ;;
        layer_plots) CLEAN_FROM=12 ;;
        text_inject)      CLEAN_FROM=13 ;;
        srf) CLEAN_FROM=15 ;;
        evaluate)         CLEAN_FROM=19 ;;
        vit_analysis) CLEAN_FROM=11 ;;
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
JN3="3_${SHORT_MODEL}${_HOOK_SHORT}"     # classify (FT + permutation) — suffixed by hook point
JN4="4_${SHORT_MODEL}${_HOOK_SHORT}"     # merge_classifications — suffixed by hook point
JN6p="6p_${SHORT_MODEL}${_HOOK_SHORT}"   # patch_fig3 (hook-specific)
JN8="8_${SHORT_MODEL}${_HOOK_SHORT}"     # visualize Figure 3 panels (hook-specific)
JN12="12_${SHORT_MODEL}"   # layer_plots (per-layer trend figure + LaTeX tables)

# For step 3 (classify): one shard per layer for uniform wall time
CLASSIFY_SHARDS=$N_LAYERS

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

OUTPUT_DIR="${OUTPUT_DIR_USER:-results/3-classify/${CLASSIFY_MODE_DIR}}${RUN_SUFFIX}"

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
echo "  Gen tokens:         max=$GEN_MAX_TOKENS min=$GEN_MIN_TOKENS${_GEN_DIR_SUFFIX:+ (dir suffix: $_GEN_DIR_SUFFIX)}"
echo "  Output dir:         $OUTPUT_DIR"
echo "  Hook point:         $HOOK_POINT (gate=SiLU(gate_proj(x)), gate_up=gate*up, attn=o_proj_input)"
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
mkdir -p "$WORK_DIR/results/1-describe/${MODE_DIR}${_GEN_DIR_SUFFIX}/shards"
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
        [1]="${JN1:-}"  [2]="${JN2:-}"  [3]="${JN3:-}"  [4]="${JN4:-}"
        [5]="${JN7:-}"  [6]="${JN8:-} ${JN8:-}p"  [7]="${JN9:-}"  [8]="${JN10:-}"
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
        _ATTN_DIR="results/9-attention_maps"
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
        _PLOT_DIR="results/10-statistics/cross-model/${CLASSIFY_MODE_DIR}"
        _PLOT_MARKER="$_PLOT_DIR/done.marker"
        if $WIPE; then
            [[ -d "$_PLOT_DIR" ]] && rm -rf "$_PLOT_DIR" && echo "    rm -rf $_PLOT_DIR"
        else
            [[ -f "$_PLOT_MARKER" ]] && rm -f "$_PLOT_MARKER" && echo "    rm $_PLOT_MARKER (marker)"
        fi
    fi

    # Step 14: layer_plots marker
    if (( CLEAN_FROM <= 14 && 14 <= CLEAN_TO )); then
        _LP_DIR="results/12-layer-plots/${CLASSIFY_MODE_DIR}"
        _LP_MARKER="$_LP_DIR/done.marker"
        if $WIPE; then
            [[ -d "$_LP_DIR" ]] && rm -rf "$_LP_DIR" && echo "    rm -rf $_LP_DIR"
        else
            [[ -f "$_LP_MARKER" ]] && rm -f "$_LP_MARKER" && echo "    rm $_LP_MARKER (marker)"
        fi
    fi

    # Step 10: hallucination scores
    if (( CLEAN_FROM <= 10 && 10 <= CLEAN_TO )); then
        _HS_DIR="results/10-halluc_scores/${CLASSIFY_MODE_DIR}/${MODEL_NAME}"
        if $WIPE; then
            [[ -d "$_HS_DIR" ]] && rm -rf "$_HS_DIR" && echo "    rm -rf $_HS_DIR"
        else
            [[ -f "$_HS_DIR/done.marker" ]] && rm -f "$_HS_DIR/done.marker" && echo "    rm $_HS_DIR/done.marker (marker)"
        fi
    fi

    # Step 11: steering results
    if (( CLEAN_FROM <= 11 && 11 <= CLEAN_TO )); then
        _ST_DIR="results/11-steering/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/${HALLUC_SCORE_METHOD}"
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
        _PLOT_ST_DIR="results/13-plots/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/${HALLUC_SCORE_METHOD}"
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
        _LP_MARKER="results/12-layer-plots/${CLASSIFY_MODE_DIR}/done.marker"
        rm -f "$WORK_DIR/$_LP_MARKER" && echo "    rm $_LP_MARKER"
        if $WIPE; then
            _LP_DIR="results/12-layer-plots/${CLASSIFY_MODE_DIR}"
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
            STEP_LOG_NAMES=("1-describe" "2-merge_descriptions" "3-classify" "4-merge_classifications" "5-check_collisions" "6-find_fig3" "7-equal-fraction-ablation" "8-activation_maps" "9-attention_maps" "10-statistics" "9-vit_analysis" "10-statistics-cross" "11-vit_analysis" "12-layer-plots")
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
STEP_LOG_DIR="${LOG_DIR}/1-describe${_LOG_HOOK_SUFFIX}"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

if [[ "$MODE" == "test" ]]; then
    JOB_NAME="${JN1}"
    echo ""
    echo "  ── generate_descriptions --test_fig3 ──"
    [[ -n "$_GEN_DIR_SUFFIX" ]] && echo "  Token limits: max=$GEN_MAX_TOKENS min=$GEN_MIN_TOKENS → dir suffix: ${_GEN_DIR_SUFFIX}"
    # The generate script writes _fig3_${MODEL_TYPE}.json internally.
    # Rename to _fig3_${MODEL_NAME}.json if they differ.
    _GEN_SCRIPT_OUT="results/1-describe/test${_GEN_DIR_SUFFIX}/generated_descriptions_fig3_${MODEL_TYPE}.json"
    _GEN_RENAME_CMD=""
    if [[ "$_GEN_SCRIPT_OUT" != "$DESC_FILE" ]]; then
        _GEN_RENAME_CMD="&& [ -f '$_GEN_SCRIPT_OUT' ] && mv '$_GEN_SCRIPT_OUT' '$DESC_FILE' && echo 'Renamed → $DESC_FILE'"
    fi

    # Skip if test description file exists or job is active
    if [[ -s "$DESC_FILE" ]] || is_job_active "$JOB_NAME"; then
        echo "  [skip] $JOB_NAME — already done or active"
        SKIPPED=$((SKIPPED + 1))
    elif $LOCAL; then
        mkdir -p "results/1-describe/test${_GEN_DIR_SUFFIX}"
        (cd "$WORK_DIR" && $PYTHON code/generate_descriptions.py \
            --model_type "$MODEL_TYPE" \
            --original_model_path "$MODEL_PATH" \
            --model_path "$MODEL_PATH" \
            --output_path "results/1-describe/test${_GEN_DIR_SUFFIX}/generated_descriptions.json" \
            --max_new_tokens $GEN_MAX_TOKENS \
            --min_new_tokens $GEN_MIN_TOKENS \
            $GEN_ARGS \
            $_GEN_RENAME_CMD \
            && $PYTHON code/clean_descriptions.py --input "$DESC_FILE" --backup --verbose) \
            2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        bsub_tiered -q $QUEUE \
             -J "$JOB_NAME" \
             -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
             -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
             -- "mkdir -p results/1-describe/test${_GEN_DIR_SUFFIX} && \
                 cd $WORK_DIR && $PYTHON code/generate_descriptions.py \
                     --model_type $MODEL_TYPE \
                     --original_model_path $MODEL_PATH \
                     --model_path $MODEL_PATH \
                     --output_path results/1-describe/test${_GEN_DIR_SUFFIX}/generated_descriptions.json \
                     --max_new_tokens $GEN_MAX_TOKENS \
                     --min_new_tokens $GEN_MIN_TOKENS \
                     $GEN_ARGS \
                     $_GEN_RENAME_CMD \
                 && $PYTHON code/clean_descriptions.py --input $DESC_FILE --backup --verbose"
        echo "  → Job: $JOB_NAME (1 GPU, tiers: ${GPU_GMEM_TIERS[*]})"
        SUBMITTED=$((SUBMITTED + 1))
    fi
else
    # Full mode: shard across GPUs
    echo ""
    echo "  ── generate_descriptions — $SHARDS_EFFECTIVE shards ──"
    echo "  Images per shard: ~$((N_TOTAL_IMAGES / SHARDS_EFFECTIVE))"
    [[ -n "$_GEN_DIR_SUFFIX" ]] && echo "  Token limits: max=$GEN_MAX_TOKENS min=$GEN_MIN_TOKENS → dir suffix: ${_GEN_DIR_SUFFIX}"

    GEN_SUBMITTED_JOBS=()  # track which shard jobs were submitted this run
    for ((s=0; s<SHARDS_EFFECTIVE; s++)); do
        START_IDX=$((s * N_TOTAL_IMAGES / SHARDS_EFFECTIVE))
        END_IDX=$(((s + 1) * N_TOTAL_IMAGES / SHARDS_EFFECTIVE))
        JOB_NAME="gen_${s}_${SHORT_MODEL}"
        SHARD_FILE="results/1-describe/full${_GEN_DIR_SUFFIX}/shards/gen_desc${_DS}_shard${s}.json"

        # Skip if shard output exists or job is active
        if [[ -s "$SHARD_FILE" ]] || is_job_active "$JOB_NAME"; then
            echo "  [skip] Shard $s ($JOB_NAME) — already done or active"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if $LOCAL; then
            echo "  Shard $s: images [$START_IDX, $END_IDX) → $SHARD_FILE"
            mkdir -p "$(dirname "$SHARD_FILE")"
            (cd "$WORK_DIR" && $PYTHON code/generate_descriptions.py \
                --model_type "$MODEL_TYPE" \
                --original_model_path "$MODEL_PATH" \
                --model_path "$MODEL_PATH" \
                --output_path "$SHARD_FILE" \
                --max_new_tokens $GEN_MAX_TOKENS \
                --min_new_tokens $GEN_MIN_TOKENS \
                --start_idx $START_IDX \
                --end_idx $END_IDX) \
                2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            bsub_tiered -q $QUEUE \
                 -J "$JOB_NAME" \
                 -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
                 -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
                 -- "mkdir -p $(dirname $SHARD_FILE) && \
                     cd $WORK_DIR && $PYTHON code/generate_descriptions.py \
                     --model_type $MODEL_TYPE \
                     --original_model_path $MODEL_PATH \
                     --model_path $MODEL_PATH \
                     --output_path $SHARD_FILE \
                     --max_new_tokens $GEN_MAX_TOKENS \
                     --min_new_tokens $GEN_MIN_TOKENS \
                     --start_idx $START_IDX \
                     --end_idx $END_IDX"
            echo "  → Shard $s: [$START_IDX, $END_IDX) tiers: ${GPU_GMEM_TIERS[*]} → $JOB_NAME"
            GEN_SUBMITTED_JOBS+=("$JOB_NAME")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done

    # ── Merge shards into single file + clean degenerate tails ──
    MERGE_JOB="${JN2}"
    MERGE_CMD="$PYTHON -c \"
import json, glob, os
merged = {}
for f in sorted(glob.glob('results/1-describe/full${_GEN_DIR_SUFFIX}/shards/gen_desc${_DS}_shard*.json')):
    with open(f) as fp:
        merged.update(json.load(fp))
os.makedirs(os.path.dirname('${DESC_FILE}'), exist_ok=True)
with open('${DESC_FILE}', 'w') as fp:
    json.dump(merged, fp, indent=2)
n_shards = len(glob.glob('results/1-describe/full${_GEN_DIR_SUFFIX}/shards/gen_desc${_DS}_shard*.json'))
print(f'Merged {len(merged)} descriptions from {n_shards} shards → ${DESC_FILE}')
\""
    CLEAN_CMD="$PYTHON code/clean_descriptions.py --input ${DESC_FILE} --backup --verbose"

    echo ""
    echo "  ── Merge + Clean → $DESC_FILE ──"

    # Skip if merged file exists or job is active
    if [[ -s "$DESC_FILE" ]] || is_job_active "$MERGE_JOB"; then
        echo "  [skip] $MERGE_JOB — already done or active"
        SKIPPED=$((SKIPPED + 1))
    elif $LOCAL; then
        (cd "$WORK_DIR" && eval "$MERGE_CMD" && eval "$CLEAN_CMD") 2>&1 | tee "${STEP_LOG_DIR}/${MERGE_JOB}${LOG_SUFFIX}.log"
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
        bsub "${BSUB_MERGE_ARGS[@]}" "cd $WORK_DIR && $MERGE_CMD && $CLEAN_CMD"
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
STEP_LOG_DIR="${LOG_DIR}/2-merge_descriptions${_LOG_HOOK_SUFFIX}"
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
# STEP 1c (clean_desc): Remove degenerate tails from descriptions
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "clean_desc" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1c: Clean descriptions (remove degenerate tails)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

_DESC_DIR="results/1-describe/${MODE_DIR}${_GEN_DIR_SUFFIX}"
if [[ -d "$_DESC_DIR" ]]; then
    echo "  Directory: $_DESC_DIR"
    $PYTHON $CLEAN_DESC_SCRIPT --input-dir "$_DESC_DIR" --backup --verbose
else
    echo "  SKIP: directory not found: $_DESC_DIR"
    echo "  Run step 1 first with --long-desc to generate descriptions."
fi

fi  # end step 1c (clean_desc)

# ═══════════════════════════════════════════════════════════════
# STEP 3 (classify): Classify neurons (sharded by layer range)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "cn" || $STEP_ALL == true ]]; then

# In test + viz-fig3 mode, classify is unnecessary — patch_fig3 handles
# the 6 specific neurons directly, so skip the entire cn step.
STEP_LOG_DIR="${LOG_DIR}/3-classify${_LOG_HOOK_SUFFIX}"
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

if [[ "$MODE" == "test" || "$MODE" == "1-layer-100-imgs-test" || "$MODE" == "1-layer" ]]; then
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
        STATS_FILE="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/classification_stats_layers${_L}-$((_L+1)).json"
        PMBT_FILE="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/permutation_stats_layers${_L}-$((_L+1)).json"
        if { [[ -s "$STATS_FILE" ]] && [[ -s "$PMBT_FILE" ]]; } || is_job_active "$JOB_NAME"; then
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
    STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/classification_stats_layers0-2.json"
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

    # Pre-create result directories immediately
    _CLS_FT_DIR="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
    _CLS_PM_DIR="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
    mkdir -p "$_CLS_FT_DIR" "$_CLS_PM_DIR"
    echo "  FT dir:   $_CLS_FT_DIR"
    echo "  PMBT dir: $_CLS_PM_DIR"

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
        STATS_FILE="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/classification_stats_layers${LAYER_START}-${LAYER_END}.json"
        PMBT_FILE="$CLS_OUTPUT_DIR/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/permutation_stats_layers${LAYER_START}-${LAYER_END}.json"

        # Skip only if BOTH FT and PMBT exist, or job is active
        if { [[ -s "$STATS_FILE" ]] && [[ -s "$PMBT_FILE" ]]; } || is_job_active "$JOB_NAME"; then
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
                --layer_end $LAYER_END \
                $CLASSIFY_ARGS) \
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
                    --layer_end $LAYER_END \
                    $CLASSIFY_ARGS"
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

STEP_LOG_DIR="${LOG_DIR}/4-merge_classifications${_LOG_HOOK_SUFFIX}"
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

MERGE_NC_CMD="cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
    --merge \
    --model_type $MODEL_TYPE \
    --model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --hook_point $HOOK_POINT \
    ${_GEN_DIR_SUFFIX:+--output_suffix $_GEN_DIR_SUFFIX} \
    $(if $IMPORTANCE_WEIGHT; then echo '--importance_weight'; fi) \
    && $PYTHON code/print_classification_summary.py --base_dir $OUTPUT_DIR --models $MODEL_NAME"

# Skip if merged output already exists and no classify jobs are pending
MERGE_STATS_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/classification_stats_all.json"
MERGE_PMBT_FILE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}/permutation_stats_all.json"
MERGE_NC_JOB_CHK="${JN4}"
if [[ -s "$MERGE_STATS_FILE" ]] && [[ -s "$MERGE_PMBT_FILE" ]] && [[ -z "${CLS_SUBMITTED_JOBS_ALL:-}" ]] && ! is_job_active "$MERGE_NC_JOB_CHK"; then
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
    (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
        --merge \
        --model_type "$MODEL_TYPE" \
        --model "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --hook_point "$HOOK_POINT" \
        ${_GEN_DIR_SUFFIX:+--output_suffix "$_GEN_DIR_SUFFIX"} \
        $(if $IMPORTANCE_WEIGHT; then echo '--importance_weight'; fi))

    # Print classification distribution summary after merge
    SUMMARY_SCRIPT="code/print_classification_summary.py"
    if [[ -f "$SUMMARY_SCRIPT" ]]; then
        echo ""
        echo "  ── Classification summary (${MODEL_NAME}, ${HOOK_POINT}) ──"
        $PYTHON "$SUMMARY_SCRIPT" --base_dir "$OUTPUT_DIR" --models "$MODEL_NAME"
    fi
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
# STEP 4s (summary): Print classification distribution table
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "summary" ]]; then
    SUMMARY_SCRIPT="code/print_classification_summary.py"
    SUMMARY_OUT="$OUTPUT_DIR/classification_summary.txt"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 4s: Classification summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _SUMMARY_ARGS="--base_dir $OUTPUT_DIR -o $SUMMARY_OUT"
    # If a specific model was requested via --model-type, show only that model
    if $MODEL_TYPE_SET; then
        _SUMMARY_ARGS="$_SUMMARY_ARGS --models $MODEL_NAME"
    fi
    $PYTHON "$SUMMARY_SCRIPT" $_SUMMARY_ARGS
fi

# ═══════════════════════════════════════════════════════════════
# STEP 6 (find_fig3): Find candidate neurons for Figure 3
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

fi  # end step 6 (find_fig3)

# ═══════════════════════════════════════════════════════════════
# STEP 5 (check_collisions): Check for image token collisions in descriptions
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

fi  # end step 5 (check_collisions)



# ═══════════════════════════════════════════════════════════════
# STEP 7 (equal_fraction_ablation): Equal-count taxonomy validation
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "equal_fraction_ablation" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 7: Equal-fraction taxonomy validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STEP7_SCRIPT="code/equal_fraction_ablation.py"

# Add hook suffix to output/log dirs only for non-default hooks (gate is default)
if [[ "$HOOK_POINT" == "gate" ]]; then
    _STEP7_HOOK=""
else
    _STEP7_HOOK="${_HOOK_SUFFIX}"
fi

STEP7_LOG="${LOG_DIR}/7-equal-fraction-ablation${_STEP7_HOOK}"
mkdir -p "$WORK_DIR/$STEP7_LOG"

LABELS_BASE="${OUTPUT_DIR_USER:-$OUTPUT_DIR}"
LABEL_DIR="${LABELS_BASE}/${MODEL_NAME}/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
# Fallback: try without gen suffix if dir doesn't exist (older label runs)
if [[ ! -d "$WORK_DIR/$LABEL_DIR" ]] && [[ -n "$_GEN_DIR_SUFFIX" ]]; then
    _LABEL_DIR_FALLBACK="${LABELS_BASE}/${MODEL_NAME}/llm_permutation${_HOOK_SUFFIX}"
    if [[ -d "$WORK_DIR/$_LABEL_DIR_FALLBACK" ]]; then
        echo "  [label fallback] $LABEL_DIR not found, using $_LABEL_DIR_FALLBACK"
        LABEL_DIR="$_LABEL_DIR_FALLBACK"
    fi
fi

STEP7_CONDITIONS="visual,text,multimodal,random_visual,random_text,random_multimodal"
STEP7_OUT="results/7-equal-fraction-ablation/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/random_trials${_STEP7_HOOK}"
mkdir -p "$WORK_DIR/$STEP7_OUT"

# Short hook suffix for job names (empty for default gate hook)
_JH=""
[[ "$HOOK_POINT" != "gate" ]] && _JH="${_HOOK_SHORT}"

# ── Test mode: minimal run (1 trial, 1 fraction, 2 conditions) ──
if [[ "$MODE" == "test" ]]; then
    _RT_N_TRIALS=1
    _RT_FRACTIONS="0.10"
    STEP7_CONDITIONS="visual,text,random_visual,random_text"
    echo "  [test mode] Reduced: 1 trial, 1 fraction, 4 conditions (2 categories + 2 matched random)"
elif [[ "$HOOK_POINT" == "attn" ]]; then
    # Attention heads: ~1024 total (model-dependent), small multimodal pool
    # Drop 0.05 (too few neurons at small fractions), add 0.70
    _RT_N_TRIALS=30
    _RT_FRACTIONS="0.10,0.20,0.30,0.50,0.70"
    STEP7_CONDITIONS="visual,text,multimodal,random_visual,random_text,random_multimodal"
    echo "  [attn mode] fractions 0.10-0.70 (no 0.05 — too few heads)"
else
    _RT_N_TRIALS=30
    _RT_FRACTIONS="0.05,0.10,0.20,0.30,0.50"
fi

# InternVL needs 80G
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    _RT_GMEM="80G"
else
    _RT_GMEM="${GPU_GMEM_TIERS[0]}"
fi

STEP7_COMMON="--model_type $MODEL_TYPE --model_path $MODEL_PATH \
    --model_name $MODEL_NAME --n_layers $N_LAYERS \
    --label_dir $LABEL_DIR --taxonomy pmbt --hook_point $HOOK_POINT \
    --pope_dir $POPE_DIR --pope_img_dir $POPE_IMG_DIR \
    --ranking random_trials \
    --output_dir $STEP7_OUT"

# Default benchmarks: POPE + MathVerse (TD, VO) + TriviaQA (matches SNRF/step 22)
# --pope-only: skip MathVerse and TriviaQA, evaluate only POPE
if ! $POPE_ONLY; then
    if [[ "$MODE" == "test" ]]; then
        STEP7_COMMON="$STEP7_COMMON --triviaqa_path $TRIVIAQA_PATH --triviaqa_num 50 --text_only_benchmarks"
        STEP7_COMMON="$STEP7_COMMON --mathverse_dir $MATHVERSE_DIR --mathverse_subtasks Text_Dominant,Text_Lite,Vision_Intensive,Vision_Dominant,Vision_Only"
    else
        STEP7_COMMON="$STEP7_COMMON --triviaqa_path $TRIVIAQA_PATH --triviaqa_num $TRIVIAQA_NUM --text_only_benchmarks"
        STEP7_COMMON="$STEP7_COMMON --mathverse_dir $MATHVERSE_DIR --mathverse_subtasks Text_Dominant,Text_Lite,Vision_Intensive,Vision_Dominant,Vision_Only"
    fi
fi
if $EXTRA_BENCHMARKS; then
    if [[ "$MODE" == "test" ]]; then
        STEP7_COMMON="$STEP7_COMMON --chair_ann_path $CHAIR_ANN_PATH --chair_img_dir $POPE_IMG_DIR --chair_num_images 20"
        STEP7_COMMON="$STEP7_COMMON --mmlu_dir $MMLU_DIR --mmlu_num 50"
    else
        STEP7_COMMON="$STEP7_COMMON --chair_ann_path $CHAIR_ANN_PATH --chair_img_dir $POPE_IMG_DIR --chair_num_images 500"
        STEP7_COMMON="$STEP7_COMMON --mmlu_dir $MMLU_DIR --mmlu_num 2000"
    fi
fi

echo "  Model:      $MODEL_NAME ($MODEL_TYPE)"
echo "  Conditions: $STEP7_CONDITIONS"
echo "  Fractions:  $_RT_FRACTIONS"
echo "  Trials:     $_RT_N_TRIALS per condition"
echo "  GPU gmem:   $_RT_GMEM"
echo "  Hook point: $HOOK_POINT"
echo "  Label dir:  $LABEL_DIR"
[[ -n "$STEP7_BENCHMARKS" ]] && echo "  Benchmarks: $STEP7_BENCHMARKS (per-benchmark jobs)" || echo "  Benchmarks: all (single job per trial)"

# ── Baseline-only mode: run phase 3 baseline evaluation, no trials ──
if $BASELINE_ONLY; then
    echo ""
    echo "  ── Baseline-only mode (--baseline-only) ──"

    # ── Custom code: one job per benchmark (exact match, no GPT) ──
    echo ""
    echo "  [custom] Per-benchmark baseline jobs (exact match)"

    # Common args without any benchmark-specific paths
    _BL_BASE="$PYTHON $STEP7_SCRIPT --phase 3 \
        --phase3_condition baseline \
        --phase3_benchmark all \
        --model_type $MODEL_TYPE --model_path $MODEL_PATH \
        --model_name $MODEL_NAME --n_layers $N_LAYERS \
        --ranking random_trials \
        --output_dir $STEP7_OUT"

    # POPE
    JOB_BLP="7bl_${SHORT_MODEL}${_JH}_pope"
    _BL_POPE_RESULT="$WORK_DIR/$STEP7_OUT/phase3_baseline_POPE.json"
    if [[ -f "$_BL_POPE_RESULT" ]]; then
        echo "  [skip] POPE baseline done: $_BL_POPE_RESULT"
    else
        _BL_POPE_CMD="$_BL_BASE --phase3_benchmark POPE --pope_dir $POPE_DIR --pope_img_dir $POPE_IMG_DIR"
        echo "  → $JOB_BLP (POPE)"
        if $LOCAL; then
            (cd "$WORK_DIR" && eval "$_BL_POPE_CMD") 2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_BLP}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_BLP" \
                -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP7_LOG/${JOB_BLP}.log" \
                -eo "$WORK_DIR/$STEP7_LOG/${JOB_BLP}.err" \
                "cd $WORK_DIR && $_BL_POPE_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    fi

    # TriviaQA
    if [[ "$MODE" == "test" ]]; then _TQA_NUM=50; else _TQA_NUM=$TRIVIAQA_NUM; fi
    JOB_BLT="7bl_${SHORT_MODEL}${_JH}_tqa"
    _BL_TQA_RESULT="$WORK_DIR/$STEP7_OUT/phase3_baseline_TriviaQA.json"
    if [[ -f "$_BL_TQA_RESULT" ]]; then
        echo "  [skip] TriviaQA baseline done: $_BL_TQA_RESULT"
    else
        _BL_TQA_CMD="$_BL_BASE --phase3_benchmark TriviaQA --triviaqa_path $TRIVIAQA_PATH --triviaqa_num $_TQA_NUM --text_only_benchmarks"
        echo "  → $JOB_BLT (TriviaQA)"
        if $LOCAL; then
            (cd "$WORK_DIR" && eval "$_BL_TQA_CMD") 2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_BLT}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_BLT" \
                -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP7_LOG/${JOB_BLT}.log" \
                -eo "$WORK_DIR/$STEP7_LOG/${JOB_BLT}.err" \
                "cd $WORK_DIR && $_BL_TQA_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    fi

    # MathVerse subtasks (one job per subtask)
    for _MV_ST in Text_Dominant Text_Lite Vision_Intensive Vision_Dominant Vision_Only; do
        _MV_SHORT=$(echo "$_MV_ST" | sed 's/Vision_Only/vo/;s/Vision_Dominant/vd/;s/Vision_Intensive/vi/;s/Text_Dominant/td/;s/Text_Lite/tl/' | tr '[:upper:]' '[:lower:]')
        JOB_BLMV="7bl_${SHORT_MODEL}${_JH}_mv_${_MV_SHORT}"
        _BL_MV_RESULT="$WORK_DIR/$STEP7_OUT/phase3_baseline_MV_${_MV_ST}.json"
        if [[ -f "$_BL_MV_RESULT" ]]; then
            echo "  [skip] MathVerse $_MV_ST baseline done"
        else
            _BL_MV_CMD="$_BL_BASE --phase3_benchmark MV_${_MV_ST} --mathverse_dir $MATHVERSE_DIR --mathverse_subtasks $_MV_ST"
            echo "  → $JOB_BLMV (MathVerse $_MV_ST)"
            if $LOCAL; then
                (cd "$WORK_DIR" && eval "$_BL_MV_CMD") 2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_BLMV}.log"
            else
                bsub -q "$QUEUE" -J "$JOB_BLMV" \
                    -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                    -R "$GPU_RES_BASE" \
                    -oo "$WORK_DIR/$STEP7_LOG/${JOB_BLMV}.log" \
                    -eo "$WORK_DIR/$STEP7_LOG/${JOB_BLMV}.err" \
                    "cd $WORK_DIR && $_BL_MV_CMD"
                SUBMITTED=$((SUBMITTED + 1))
            fi
        fi
    done

    # ── VLMEvalKit for MathVista + MathVerse (GPT judge, matches BRV exactly) ──
    if $VLMEVAL && ! $POPE_ONLY && [[ "$MODEL_TYPE" != "llava-liuhaotian" ]]; then
        echo ""
        echo "  [vlmeval] MathVista + MathVerse (GPT judge)"
        _VE_WORKDIR="$STEP7_OUT/vlmeval_baseline"
        _VE_MODEL="UNKNOWN"
        [[ "$MODEL_TYPE" == "llava-liuhaotian" ]] && _VE_MODEL="llava-1.5-7b_baseline"
        [[ "$MODEL_TYPE" == "llava-llama3" ]] && _VE_MODEL="llava_next_llama3"
        [[ "$MODEL_TYPE" == "llava-mistral" ]] && _VE_MODEL="llava_next_mistral_7b"
        [[ "$MODEL_TYPE" == "llava-ov" ]] && _VE_MODEL="llava-onevision-7b_baseline"
        [[ "$MODEL_TYPE" == "internvl" ]] && _VE_MODEL="internvl2.5-8b_baseline"
        [[ "$MODEL_TYPE" == "qwen2vl" ]] && _VE_MODEL="Qwen2-VL-7B-Instruct"
        [[ "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "qwen2vl-brv" ]] && _VE_MODEL="qwen2.5-vl-7b_baseline"
        [[ "$MODEL_TYPE" == "qwen25vl-3b" ]] && _VE_MODEL="Qwen2.5-VL-3B-Instruct"
        [[ "$MODEL_TYPE" == "idefics2" ]] && _VE_MODEL="idefics2_8b"

        _VE_DATASETS="MathVista_MINI MathVerse_MINI_Text_Dominant MathVerse_MINI_Text_Lite MathVerse_MINI_Vision_Intensive MathVerse_MINI_Vision_Dominant MathVerse_MINI_Vision_Only"

        # VLMEvalKit always uses the modern_vlms venv (compatible with VLMEvalKit deps)
        _VE_PYTHON="$WORK_DIR/modern_vlms/VLMEvalKit_brv/.venv/bin/python"
        [[ "$MODEL_TYPE" == "internvl" ]] && _VE_PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"

        # Ensure idefics2 uses SDPA (built-in) instead of flash_attention_2 (requires external package)
        sed -i 's/_attn_implementation="flash_attention_2"/_attn_implementation="sdpa"/' "$WORK_DIR/$VLMEVAL_DIR/vlmeval/vlm/idefics"* 2>/dev/null || true

        mkdir -p "$WORK_DIR/$_VE_WORKDIR"
        echo "  Model: $_VE_MODEL"

        # Short judge suffix for job names (gpt-4o-mini → 4om, gpt-4o → 4o)
        _JUDGE_SHORT=$(echo "$VLMEVAL_JUDGE" | sed 's/gpt-//;s/-mini/m/')

        echo "  Datasets: $_VE_DATASETS"
        echo "  Judge: $VLMEVAL_JUDGE (short: $_JUDGE_SHORT)"

        for _VE_D in $_VE_DATASETS; do
            _VE_SHORT=$(echo "$_VE_D" | sed 's/MathVerse_MINI_//;s/MathVista_MINI/mvista/;s/Vision_Only/vo/;s/Vision_Dominant/vd/;s/Vision_Intensive/vi/;s/Text_Dominant/td/;s/Text_Lite/tl/' | tr '[:upper:]' '[:lower:]')
            JOB_VE="7ve_${SHORT_MODEL}_${_VE_SHORT}_${_JUDGE_SHORT}"

            # Skip if result already exists — use _score files with exact judge match
            _VE_EXISTING=""
            if [[ "$VLMEVAL_JUDGE" == "gpt-4o-mini" ]]; then
                _VE_EXISTING=$(find "$WORK_DIR/$_VE_WORKDIR" -name "*${_VE_D}*gpt-4o-mini*_score*" ! -path "*/bak_*" 2>/dev/null | head -1 || true)
            else
                _VE_EXISTING=$(find "$WORK_DIR/$_VE_WORKDIR" -name "*${_VE_D}*${VLMEVAL_JUDGE}*_score*" ! -path "*/bak_*" 2>/dev/null | grep -v "gpt-4o-mini" | head -1 || true)
            fi
            if [[ -n "$_VE_EXISTING" ]]; then
                echo "  [skip] $_VE_D done: $_VE_EXISTING"
                continue
            fi

            _VE_CMD="cd $WORK_DIR && export OPENAI_API_KEY=\$(cat $VLMEVAL_DIR/.env 2>/dev/null | grep OPENAI_API_KEY | cut -d= -f2); $_VE_PYTHON $VLMEVAL_DIR/run.py --data $_VE_D --model $_VE_MODEL --judge $VLMEVAL_JUDGE --work-dir $WORK_DIR/$_VE_WORKDIR"

            echo "  → $JOB_VE ($_VE_D)"
            if $LOCAL; then
                eval "$_VE_CMD" 2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_VE}.log"
            else
                bsub -q "$QUEUE" -J "$JOB_VE" \
                    -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                    -R "$GPU_RES_BASE" \
                    -oo "$WORK_DIR/$STEP7_LOG/${JOB_VE}.log" \
                    -eo "$WORK_DIR/$STEP7_LOG/${JOB_VE}.err" \
                    "$_VE_CMD"
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        echo "  Results: $_VE_WORKDIR/"
    fi

else
# ── Phase 1: One job per (condition, fraction, trial, [benchmark]) ──
IFS=',' read -ra _RT_FRACS <<< "$_RT_FRACTIONS"
IFS=',' read -ra _RT_CONDS <<< "$STEP7_CONDITIONS"

# Benchmark short names for job IDs
declare -A _BENCH_SHORT=(
    [POPE]="pope" [TriviaQA]="tqa"
    [MV_Text_Dominant]="td" [MV_Text_Lite]="tl"
    [MV_Vision_Intensive]="vi" [MV_Vision_Dominant]="vd" [MV_Vision_Only]="vo"
)

# Per-benchmark mode: separate job per benchmark (fast, resumable)
if [[ -n "$STEP7_BENCHMARKS" ]]; then
    IFS=',' read -ra _RT_BENCHMARKS <<< "$STEP7_BENCHMARKS"
    echo "  Per-benchmark mode: ${_RT_BENCHMARKS[*]}"
    echo "  Jobs per model: ${#_RT_CONDS[@]} conditions × ${#_RT_FRACS[@]} fractions × $_RT_N_TRIALS trials × ${#_RT_BENCHMARKS[@]} benchmarks"

    RT_JOBS=()
    for _BENCH in "${_RT_BENCHMARKS[@]}"; do
        _BS="${_BENCH_SHORT[$_BENCH]:-${_BENCH:0:3}}"
        for _COND in "${_RT_CONDS[@]}"; do
            for _FRAC in "${_RT_FRACS[@]}"; do
                _FRAC_FNAME=$(echo "$_FRAC" | sed 's/\./p/')
                for (( _T=0; _T<$_RT_N_TRIALS; _T++ )); do
                    _T_FMT=$(printf "%03d" "$_T")
                    _RESULT_FILE="$WORK_DIR/$STEP7_OUT/random_trial_${_COND}_${_FRAC_FNAME}_t${_T_FMT}_${_BENCH}.json"
                    JOB_NAME="7rt_${SHORT_MODEL}${_JH}_${_BS}_${_COND:0:3}_${_FRAC_FNAME}_t${_T_FMT}"

                    if [[ -f "$_RESULT_FILE" ]]; then
                        continue
                    fi

                    if $LOCAL; then
                        (cd "$WORK_DIR" && $PYTHON $STEP7_SCRIPT --phase 1 \
                            --condition "$_COND" --fraction "$_FRAC" --trial "$_T" \
                            --benchmark "$_BENCH" \
                            $STEP7_COMMON) \
                            2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_NAME}.log"
                    else
                        bsub -q "$QUEUE" -J "$JOB_NAME" \
                            -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                            -R "$GPU_RES_BASE" \
                            -oo "$WORK_DIR/$STEP7_LOG/${JOB_NAME}.log" \
                            -eo "$WORK_DIR/$STEP7_LOG/${JOB_NAME}.err" \
                            "cd $WORK_DIR && $PYTHON $STEP7_SCRIPT --phase 1 \
                                --condition $_COND --fraction $_FRAC --trial $_T \
                                --benchmark $_BENCH \
                                $STEP7_COMMON"
                        RT_JOBS+=("$JOB_NAME")
                        SUBMITTED=$((SUBMITTED + 1))
                    fi
                done
            done
        done
    done

# All-in-one mode: one job evaluates ALL benchmarks (legacy, slower)
else
    echo "  All-in-one mode (use --benchmarks for per-benchmark jobs)"
    RT_JOBS=()
    for _COND in "${_RT_CONDS[@]}"; do
        for _FRAC in "${_RT_FRACS[@]}"; do
            _FRAC_FNAME=$(echo "$_FRAC" | sed 's/\./p/')
            for (( _T=0; _T<$_RT_N_TRIALS; _T++ )); do
                _T_FMT=$(printf "%03d" "$_T")
                _RESULT_FILE="$WORK_DIR/$STEP7_OUT/random_trial_${_COND}_${_FRAC_FNAME}_t${_T_FMT}.json"
                JOB_NAME="7rt_${SHORT_MODEL}${_JH}_${_COND:0:3}_${_FRAC_FNAME}_t${_T_FMT}"

                if [[ -f "$_RESULT_FILE" ]]; then
                    continue
                fi

                if $LOCAL; then
                    (cd "$WORK_DIR" && $PYTHON $STEP7_SCRIPT --phase 1 \
                        --condition "$_COND" --fraction "$_FRAC" --trial "$_T" \
                        $STEP7_COMMON) \
                        2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_NAME}.log"
                else
                    bsub -q "$QUEUE" -J "$JOB_NAME" \
                        -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                        -R "$GPU_RES_BASE" \
                        -oo "$WORK_DIR/$STEP7_LOG/${JOB_NAME}.log" \
                        -eo "$WORK_DIR/$STEP7_LOG/${JOB_NAME}.err" \
                        "cd $WORK_DIR && $PYTHON $STEP7_SCRIPT --phase 1 \
                            --condition $_COND --fraction $_FRAC --trial $_T \
                            $STEP7_COMMON"
                    RT_JOBS+=("$JOB_NAME")
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
        done
    done
fi

# ── Phase 2: Merge trials + statistics (depends on all phase 3 jobs) ──
if [[ ${#RT_JOBS[@]} -gt 0 ]]; then
    JOB_MERGE="7rt_${SHORT_MODEL}_merge"
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
        -oo "$WORK_DIR/$STEP7_LOG/${JOB_MERGE}.log" \
        -eo "$WORK_DIR/$STEP7_LOG/${JOB_MERGE}.err" \
        "cd $WORK_DIR && $PYTHON $STEP7_SCRIPT --phase 2 $STEP7_COMMON"
    SUBMITTED=$((SUBMITTED + 1))
fi

echo "  Random trials: ${#RT_JOBS[@]} jobs submitted"

fi  # end baseline-only / trials branch

# ── Phase 3: SNRF-style 100% ablation (only if --full-ablation) ──
if $FULL_ABLATION; then
    echo ""
    echo "  ── Phase 3: SNRF-style 100% ablation (--full-ablation) ──"
    STEP7_P3_CONDITIONS="baseline visual text multimodal random_visual random_text random_multimodal"
    STEP7_P3_OUT="results/7-equal-fraction-ablation/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/snrf_ablation"
    mkdir -p "$WORK_DIR/$STEP7_P3_OUT"

    P3_JOBS=()
    for _P3_COND in $STEP7_P3_CONDITIONS; do
        _P3_RESULT="$WORK_DIR/$STEP7_P3_OUT/phase3_${_P3_COND}.json"
        JOB_P3="7p3_${SHORT_MODEL}_${_P3_COND}"

        if [[ -f "$_P3_RESULT" ]]; then
            echo "  [skip] $_P3_COND already done"
            continue
        fi

        _P3_CMD="$PYTHON $STEP7_SCRIPT --phase 3 \
            --phase3_condition $_P3_COND \
            --phase3_benchmark all \
            --model_type $MODEL_TYPE --model_path $MODEL_PATH \
            --model_name $MODEL_NAME --n_layers $N_LAYERS \
            --label_dir $LABEL_DIR --taxonomy pmbt \
            --pope_dir $POPE_DIR --pope_img_dir $POPE_IMG_DIR \
            --triviaqa_path $TRIVIAQA_PATH --triviaqa_num $TRIVIAQA_NUM \
            --output_dir $STEP7_P3_OUT"

        echo "  → $JOB_P3"
        if $LOCAL; then
            (cd "$WORK_DIR" && eval "$_P3_CMD") \
                2>&1 | tee "$WORK_DIR/$STEP7_LOG/${JOB_P3}.log"
        else
            bsub -q "$QUEUE" -J "$JOB_P3" \
                -gpu "num=1:gmem=$_RT_GMEM$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP7_LOG/${JOB_P3}.log" \
                -eo "$WORK_DIR/$STEP7_LOG/${JOB_P3}.err" \
                "cd $WORK_DIR && $_P3_CMD"
            P3_JOBS+=("$JOB_P3")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done

    # Merge job after all conditions complete
    if [[ ${#P3_JOBS[@]} -gt 0 ]]; then
        JOB_P3_MERGE="7p3_${SHORT_MODEL}_merge"
        DEP_P3=""
        for _jn in "${P3_JOBS[@]}"; do
            [[ -n "$DEP_P3" ]] && DEP_P3="$DEP_P3 && "
            DEP_P3="${DEP_P3}done($_jn)"
        done
        bsub -q "$QUEUE" -J "$JOB_P3_MERGE" \
            -w "$DEP_P3" \
            -R "rusage[mem=4096]" \
            -oo "$WORK_DIR/$STEP7_LOG/${JOB_P3_MERGE}.log" \
            -eo "$WORK_DIR/$STEP7_LOG/${JOB_P3_MERGE}.err" \
            "cd $WORK_DIR && $PYTHON $STEP7_SCRIPT --phase 3 \
                --phase3_condition merge \
                --model_type $MODEL_TYPE --model_name $MODEL_NAME --n_layers $N_LAYERS \
                --label_dir $LABEL_DIR --taxonomy pmbt \
                --output_dir $STEP7_P3_OUT"
        SUBMITTED=$((SUBMITTED + 1))
    fi

    echo "  Phase 3 (SNRF-style): ${#P3_JOBS[@]} jobs submitted"
    echo "  Results: $STEP7_P3_OUT/"
fi

fi  # end step 7 (equal_fraction_ablation)
# ═══════════════════════════════════════════════════════════════
# STEP 8 (visualize)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "visualize" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 10: Figure 3 activation visualizations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/6-activation_maps${_LOG_HOOK_SUFFIX}"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

VIZ_DATA_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
VIZ_OUT_DIR="$OUTPUT_DIR/$MODEL_NAME/fig3${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
VIZ_FT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_fixed_threshold${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
VIZ_PMBT_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
# Second PMBT dir: gate hook data for multi-hook comparison
# Only passed when current hook is gate (column 1 in combined figure)
# gate_up and attn panels should NOT show PMBT gate labels
VIZ_PMBT_DIR_2=""
VIZ_PMBT_DIR_2_ARGS=""
if [[ "$HOOK_POINT" == "gate" ]]; then
    if [[ -d "$OUTPUT_DIR/$MODEL_NAME/llm_permutation_gate${_GEN_DIR_SUFFIX}" ]]; then
        VIZ_PMBT_DIR_2="$OUTPUT_DIR/$MODEL_NAME/llm_permutation_gate${_GEN_DIR_SUFFIX}"
    else
        VIZ_PMBT_DIR_2="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
    fi
    VIZ_PMBT_DIR_2_ARGS="--pmbt_data_dir_2 $VIZ_PMBT_DIR_2"
fi
COCO_IMG_DIR="${COCO_IMG_DIR:-/home/projects/bagon/shared/coco2017/images/train2017/}"

# Hook display names for figure headers
case "$_HOOK_SUFFIX" in
    _gate_up) _HOOK_DISPLAY_NAME="PMBT" ;;
    _attn)    _HOOK_DISPLAY_NAME="PMBT" ;;
    _gate)    _HOOK_DISPLAY_NAME="PMBT" ;;
    *)        _HOOK_DISPLAY_NAME="PMBT" ;;
esac

VIZ_ARGS="--types visual text multimodal"
if $VIZ_FIG3; then
    VIZ_ARGS="--fig3 --viz-taxonomy $VIZ_TAXONOMY --pmbt_hook_name $_HOOK_DISPLAY_NAME"
    $SUPPRESS_LINE1 && VIZ_ARGS="$VIZ_ARGS --suppress_line1"
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
            --hook_point "$HOOK_POINT" \
            --skip_viz \
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
                    --hook_point $HOOK_POINT \
                    --skip_viz \
                    --device 0 \
                    && touch $PATCH_MARKER"
        fi
    fi
fi

FIG3_CLS_JOBS=()
# ── Mini-classify: run PMBT on just the fig3 layers ──────────
# This gives us neuron_labels.json + neuron_labels_permutation.json
# so the visualize step can show both FT and PMBT statistics.
if $VIZ_FIG3; then
    echo ""
    echo "  ── Mini-classify: PMBT labels for Figure 3 layers ──"

    # Compute fig3 layers (accounts for proportional layer mapping)
    FIG3_LAYERS=$($PYTHON -c "
nl = {'llava-hf':32,'llava-liuhaotian':32,'llava-llama3':32,'internvl':32,'idefics2':32,'qwen25vl-3b':36,'llava-ov':28,'qwen2vl':28,'qwen25vl-7b':28}
n = nl.get('${MODEL_TYPE}', 32)
xu = [2,21,27,29,31]
mapped = sorted(set(l if l < n else min(round(l/31*(n-1)), n-1) for l in xu))
print(','.join(map(str, mapped)))
")
    echo "  Fig3 layers for ${MODEL_TYPE}: $FIG3_LAYERS"

    GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
    FIG3_CLS_JOBS=()
    CLS_FIG3_ARGS="--num_images 6 --top_n 5 --n_permutations 100 --hook_point $HOOK_POINT${_GEN_DIR_SUFFIX:+ --output_suffix $_GEN_DIR_SUFFIX}"

    IFS=',' read -ra _LAYERS <<< "$FIG3_LAYERS"
    for _L in "${_LAYERS[@]}"; do
        FIG3_CLS_JOB="3f_${SHORT_MODEL}_${_HOOK_SHORT}_${_L}"
        # Skip if PMBT labels already exist
        _PMBT_CHECK="$VIZ_PMBT_DIR"
        _HAS_PMBT=$(find "$_PMBT_CHECK" -path "*layers.${_L}.*" -name "neuron_labels_permutation.json" 2>/dev/null | head -1 || true)
        if [[ -n "$_HAS_PMBT" ]]; then
            echo "  [skip] layer $_L — PMBT labels exist"
            continue
        fi
        if is_job_active "$FIG3_CLS_JOB"; then
            echo "  [skip] $FIG3_CLS_JOB — already active"
            FIG3_CLS_JOBS+=("$FIG3_CLS_JOB")
            continue
        fi

        if $LOCAL; then
            echo "  Running classify layer $_L locally..."
            (cd "$WORK_DIR" && $PYTHON $CLASSIFY_SCRIPT \
                --model_type "$MODEL_TYPE" \
                --original_model_path "$MODEL_PATH" \
                --model_path "$MODEL_PATH" \
                --text_source generated \
                --generated_desc_path "$DESC_FILE" \
                --output_dir "$OUTPUT_DIR" \
                --model "$MODEL_NAME" \
                --layer_start "$_L" --layer_end "$((_L + 1))" \
                $CLS_FIG3_ARGS)
        else
            FIG3_CLS_BSUB=(-q "$QUEUE" \
                -J "$FIG3_CLS_JOB" \
                -oo "$WORK_DIR/${STEP_LOG_DIR}/${FIG3_CLS_JOB}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/${STEP_LOG_DIR}/${FIG3_CLS_JOB}${LOG_SUFFIX}.err")
            # Wait for patch job if active
            if is_job_active "${JN6p}"; then
                FIG3_CLS_BSUB+=(-w "done(${JN6p})")
            fi
            bsub_tiered "${FIG3_CLS_BSUB[@]}" \
                -- "cd $WORK_DIR && $PYTHON $CLASSIFY_SCRIPT \
                    --model_type $MODEL_TYPE \
                    --original_model_path $MODEL_PATH \
                    --model_path $MODEL_PATH \
                    --text_source generated \
                    --generated_desc_path $DESC_FILE \
                    --output_dir $OUTPUT_DIR \
                    --model $MODEL_NAME \
                    --layer_start $_L --layer_end $((_L + 1)) \
                    $CLS_FIG3_ARGS"
            FIG3_CLS_JOBS+=("$FIG3_CLS_JOB")
        fi
    done
fi

JOB_NAME="${JN8}"
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
        $VIZ_PMBT_DIR_2_ARGS \
        --output_dir "$VIZ_OUT_DIR" \
        --hook_point "$HOOK_POINT" \
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
    # Wait for fig3 mini-classify jobs
    for _cj in "${FIG3_CLS_JOBS[@]}"; do
        if is_job_active "$_cj"; then
            BSUB_ARGS+=(-w "done($_cj)")
        fi
    done
    rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
    bsub "${BSUB_ARGS[@]}" \
        "cd $WORK_DIR && $PYTHON $VIZ_SCRIPT \
            --data_dir $VIZ_DATA_DIR \
            --coco_img_dir $COCO_IMG_DIR \
            --generated_desc_path $DESC_FILE \
            --model_type $MODEL_TYPE \
            --model_name $MODEL_NAME \
            --pmbt_data_dir $VIZ_PMBT_DIR \
            $VIZ_PMBT_DIR_2_ARGS \
            --output_dir $VIZ_OUT_DIR \
            --hook_point $HOOK_POINT \
            $VIZ_ARGS"
    echo "  → Job: $JOB_NAME (CPU only)"
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end step 8 (visualize)

# ═══════════════════════════════════════════════════════════════
# STEP 8c (combine_fig3): Combine gate + gate_up Figure 3 panels side-by-side
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "combine_fig3" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 8c: Combine Figure 3 panels (gate vs gate_up vs attn)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

_FIG3_GATE_PANELS="$OUTPUT_DIR/$MODEL_NAME/fig3_gate${_GEN_DIR_SUFFIX}/panels"
_FIG3_GUP_PANELS="$OUTPUT_DIR/$MODEL_NAME/fig3_gate_up${_GEN_DIR_SUFFIX}/panels"
_FIG3_ATTN_PANELS="$OUTPUT_DIR/$MODEL_NAME/fig3_attn${_GEN_DIR_SUFFIX}/panels"
_FIG3_COMBINED="$OUTPUT_DIR/$MODEL_NAME/fig3_combined_hooks${_GEN_DIR_SUFFIX}.png"
_COMBINE_SCRIPT="code/combine_fig3_hooks.py"

echo "  Gate panels:    $_FIG3_GATE_PANELS $([ -d \"$_FIG3_GATE_PANELS\" ] && echo ok || echo MISSING)"
echo "  Gate_up panels: $_FIG3_GUP_PANELS $([ -d \"$_FIG3_GUP_PANELS\" ] && echo ok || echo MISSING)"
echo "  Attn panels:    $_FIG3_ATTN_PANELS $([ -d \"$_FIG3_ATTN_PANELS\" ] && echo ok || echo MISSING)"
echo "  Output:         $_FIG3_COMBINED"

_COMBINE_ARGS="--dir1 $_FIG3_GATE_PANELS --dir2 $_FIG3_GUP_PANELS"
[[ -d "$_FIG3_ATTN_PANELS" ]] && _COMBINE_ARGS="$_COMBINE_ARGS --dir3 $_FIG3_ATTN_PANELS"

$PYTHON "$_COMBINE_SCRIPT" \
    $_COMBINE_ARGS \
    --model_name "$MODEL_NAME" \
    --label1 "gate (Xu et al.)" \
    --label2 "gate_up (ours)" \
    --label3 "attn (heads)" \
    --output "$_FIG3_COMBINED"


echo "  Done: $_FIG3_COMBINED"

fi  # end step 8c (combine_fig3)

# ═══════════════════════════════════════════════════════════════
# STEP 8a (fig3_all): Run gate + gate_up + attn panels + combine in one go
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "fig3_all" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 8a: Figure 3 — gate + gate_up + attn + combine"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build pass-through args
_COMMON_ARGS="--model-type $MODEL_TYPE --mode $MODE"
$GEN_LONG_DESC && _COMMON_ARGS="$_COMMON_ARGS --long-desc"
_COMMON_ARGS="$_COMMON_ARGS --viz-fig3 --suppress-line1"
[[ -n "${GPU_GMEM_TIERS[0]:-}" ]] && _COMMON_ARGS="$_COMMON_ARGS --gmem ${GPU_GMEM_TIERS[0]%%G*}"
$QUEUE_SET && _COMMON_ARGS="$_COMMON_ARGS --queue $QUEUE"

# Clean old PNGs
echo "  Cleaning old fig3 panels..."
for _hp in gate gate_up attn; do
    rm -f "$OUTPUT_DIR/$MODEL_NAME/fig3_${_hp}${_GEN_DIR_SUFFIX}/"*.png 2>/dev/null
    rm -rf "$OUTPUT_DIR/$MODEL_NAME/fig3_${_hp}${_GEN_DIR_SUFFIX}/panels" 2>/dev/null
done

# Step 1: gate panels (col 1 — FT + PMBT gate)
echo ""
echo "  ── Step 1/4: gate panels (FT + PMBT gate) ──"
bash "$0" --step 8 $_COMMON_ARGS --hook-point gate --viz-taxonomy both

# Step 2: gate_up panels (col 2 — PMBT gate_up only)
echo ""
echo "  ── Step 2/4: gate_up panels (PMBT gate_up only) ──"
bash "$0" --step 8 $_COMMON_ARGS --hook-point gate_up --viz-taxonomy pmbt

# Step 3: attn panels (col 3 — PMBT attn only)
echo ""
echo "  ── Step 3/4: attn panels (PMBT attn only) ──"
bash "$0" --step 8 $_COMMON_ARGS --hook-point attn --viz-taxonomy pmbt

# Step 4: combine all 3
echo ""
echo "  ── Step 4/4: combining panels (3 columns) ──"

_FIG3_GATE_PANELS="$OUTPUT_DIR/$MODEL_NAME/fig3_gate${_GEN_DIR_SUFFIX}/panels"
_FIG3_GUP_PANELS="$OUTPUT_DIR/$MODEL_NAME/fig3_gate_up${_GEN_DIR_SUFFIX}/panels"
_FIG3_ATT_PANELS="$OUTPUT_DIR/$MODEL_NAME/fig3_attn${_GEN_DIR_SUFFIX}/panels"
_FIG3_COMBINED="$OUTPUT_DIR/$MODEL_NAME/fig3_combined_hooks${_GEN_DIR_SUFFIX}.png"

# Check if visualize jobs are still running
_VIZ_GATE_JOB="8_${SHORT_MODEL}_g"
_VIZ_GUP_JOB="8_${SHORT_MODEL}_gup"
_VIZ_ATT_JOB="8_${SHORT_MODEL}_att"
_ANY_ACTIVE=false
_DEP_EXPR=""
for _jn in "$_VIZ_GATE_JOB" "$_VIZ_GUP_JOB" "$_VIZ_ATT_JOB"; do
    if is_job_active "$_jn"; then
        _ANY_ACTIVE=true
        [[ -n "$_DEP_EXPR" ]] && _DEP_EXPR="$_DEP_EXPR && done($_jn)" || _DEP_EXPR="done($_jn)"
    fi
done

_COMBINE_CMD="$PYTHON code/combine_fig3_hooks.py \
    --dir1 $_FIG3_GATE_PANELS \
    --dir2 $_FIG3_GUP_PANELS \
    --dir3 $_FIG3_ATT_PANELS \
    --model_name $MODEL_NAME \
    --label1 'gate (Xu et al.)' \
    --label2 'gate_up (ours)' \
    --label3 'attn (heads)' \
    --output $_FIG3_COMBINED"

if $_ANY_ACTIVE; then
    echo "  Visualize jobs still running — submitting combine as dependent job"
    bsub -q "$QUEUE" -J "8c_${SHORT_MODEL}" \
        -oo "$WORK_DIR/logs/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/8c_combine${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/logs/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/8c_combine${LOG_SUFFIX}.err" \
        -w "$_DEP_EXPR" \
        "cd $WORK_DIR && $_COMBINE_CMD"
    echo "  → Job: 8c_${SHORT_MODEL} (depends on: $_DEP_EXPR)"
else
    # All done — run combine locally
    _N_READY=0
    [[ -d "$_FIG3_GATE_PANELS" ]] && _N_READY=$((_N_READY + 1))
    [[ -d "$_FIG3_GUP_PANELS" ]] && _N_READY=$((_N_READY + 1))
    [[ -d "$_FIG3_ATT_PANELS" ]] && _N_READY=$((_N_READY + 1))
    if [[ $_N_READY -ge 2 ]]; then
        _COMBINE_ARGS="--dir1 $_FIG3_GATE_PANELS --dir2 $_FIG3_GUP_PANELS"
        [[ -d "$_FIG3_ATT_PANELS" ]] && _COMBINE_ARGS="$_COMBINE_ARGS --dir3 $_FIG3_ATT_PANELS"
        $PYTHON code/combine_fig3_hooks.py \
            $_COMBINE_ARGS \
            --model_name "$MODEL_NAME" \
            --label1 "gate (Xu et al.)" \
            --label2 "gate_up (ours)" \
            --label3 "attn (heads)" \
            --output "$_FIG3_COMBINED"
    else
        echo "  Panels not ready yet ($_N_READY/3 dirs found). Run step 8c after visualize jobs complete."
    fi
fi

fi  # end step 8a (fig3_all)

# ═══════════════════════════════════════════════════════════════
# STEP 9 (attention)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "attn" ]] || $STEP_ALL_FULL; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 11: Attention analysis for reclassified neurons"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/7-attention_maps${_LOG_HOOK_SUFFIX}"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

ATTN_OUT_DIR="results/9-attention_maps"

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
    JOB_NAME="${JN9}${ATTN_TAG}"
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
    JOB_NAME="${JN9}${ATTN_TAG}"
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
    JOB_NAME="${JN9}${ATTN_TAG}"
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
fi  # end step 9 (attention)

# ═══════════════════════════════════════════════════════════════
# STEP 10 (statistics)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "plot" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 10: Fig7 cross-model comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/8-statistics${_LOG_HOOK_SUFFIX}"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

PLOT_OUT_BASE="results/10-statistics/cross-model/${CLASSIFY_MODE_DIR}"
PLOT_MARKER="$PLOT_OUT_BASE/done.marker"

JOB_NAME="${JN10}"
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

fi  # end step 10 (statistics)

# ═══════════════════════════════════════════════════════════════
# STEP 12 (layer_plots): per-layer trend figure + LaTeX tables
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "layer_plots" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 12: Per-layer trend figure + LaTeX tables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/14-layer-plots"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

LP_OUT_DIR="results/12-layer-plots/${CLASSIFY_MODE_DIR}"
LP_MARKER="$LP_OUT_DIR/done.marker"

JOB_NAME="${JN12}"
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
parser.add_argument("--output-dir",   default="results/12-layer-plots/full")
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
parser.add_argument("--output-dir",   default="results/12-layer-plots/full")
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
parser.add_argument("--output-dir",   default="results/12-layer-plots/full")
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

fi  # end step 12 (layer_plots)


# ═══════════════════════════════════════════════════════════════
# STEP 13 (text_inject): Task Arithmetic + PMBT text mask (Method 1)
#   Injects math task vector ONLY into text-classified neurons.
#   Reference: "Bring Reason to Vision" (Chen et al., ICML 2025)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "text_inject" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 13: Task Arithmetic + PMBT text mask (Method 1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/13-text-inject"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

MERGE_OUT_DIR="results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}"

# ── Resolve base LLM and math LLM paths per backbone ──────────
# These must share the same backbone as the VLM.
# Override with --merge-base-llm and --merge-math-llm if needed.
if [[ -z "$MERGE_BASE_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)
            MERGE_BASE_LLM_PATH="Qwen/Qwen2.5-7B"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Qwen2.5 backbone, verified diff=81.8)"
            ;;
        qwen2vl-brv)
            MERGE_BASE_LLM_PATH="Qwen/Qwen2-7B"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Qwen2 backbone, BRV Appendix E exact)"
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
            MERGE_BASE_LLM_PATH="modern_vlms/pretrained/Mistral-7B-v0.1"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Mistral backbone, local)"
            ;;
        llava-llama3)
            MERGE_BASE_LLM_PATH="modern_vlms/pretrained/llama3-8b-from-llava"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (LLaMA-3-8B backbone, BRV main model)"
            ;;
        llava-liuhaotian)
            MERGE_BASE_LLM_PATH="NousResearch/Llama-2-7b-hf"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (LLaMA-2-7B backbone)"
            ;;
        idefics2)
            MERGE_BASE_LLM_PATH="modern_vlms/pretrained/Mistral-7B-v0.1"
            echo "  [auto] base LLM: $MERGE_BASE_LLM_PATH (Mistral backbone, BRV Table 4, local)"
            ;;
    esac
fi
if [[ -z "$MERGE_MATH_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        qwen2vl)
            MERGE_MATH_LLM_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (R1 reasoning distilled into Qwen2.5)"
            ;;
        qwen2vl-brv)
            MERGE_MATH_LLM_PATH="Qwen/Qwen2-Math-7B"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (BRV Appendix E Table 7 exact)"
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
        idefics2)
            MERGE_MATH_LLM_PATH="modern_vlms/pretrained/MAmmoTH-7B-Mistral"
            echo "  [auto] math LLM: $MERGE_MATH_LLM_PATH (MAmmoTH-1, BRV Table 3 best for Idefics2, local)"
            ;;
    esac
fi

# ── Auto-set lambda sweep per model type (BRV paper Table 2/3 best) ──
# If user didn't override with --merge-lambda, use model-specific defaults.
# BRV paper (Chen et al., ICML 2025) reports best λ_BRV values:
#   LLaVA-Next-LLaMA3-8B: λ_BRV=0.9 → task_vector_weight=0.1
#   Idefics2-8B:           λ_BRV=0.85 → task_vector_weight=0.15
#   InternVL2-76B:         λ_BRV=0.9 → task_vector_weight=0.1
# Our lam = task_vector_weight, so lam=0.1 ↔ BRV's λ=0.9.
#
# Default: sweep around the BRV best value for each architecture.
if ! $_USER_SET_LAMBDA; then
    case "$MODEL_TYPE" in
        llava-llama3)
            # BRV best: λ=0.9 → lam=0.1 (Dart-Prop on MathVista)
            MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8 0.7"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (BRV best for LLaVA: λ=0.9)"
            ;;
        llava-hf|llava-liuhaotian)
            # Same backbone family as LLaVA-Next; BRV best: lam=0.1
            MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (BRV best for LLaVA: λ=0.9)"
            ;;
        internvl)
            # BRV best: λ=0.9 → lam=0.1 (Dart-Uniform on InternVL2-76B)
            MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (BRV best for InternVL: λ=0.9)"
            ;;
        qwen2vl)
            # BRV Appendix E (Table 7): Qwen2-VL + Qwen2-Math, moderate gains
            MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8 0.75 0.7"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (Qwen2-VL, broader search)"
            ;;
        qwen2vl-brv)
            # BRV Appendix E: lambda not stated, broader search
            MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8 0.75 0.7"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (BRV Qwen2-VL reproduction, broader search)"
            ;;
        llava-ov)
            # Qwen2 backbone, similar to qwen2vl
            MERGE_LAMBDA_SWEEP="0.95 0.9 0.85 0.8 0.75 0.7"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (LLaVA-OV / Qwen2 backbone)"
            ;;
        idefics2)
            # BRV best: λ=0.85 → lam=0.15 (MAmmoTH-1, Table 3)
            # Idefics2 is already heavily math-finetuned, needs stronger injection
            MERGE_LAMBDA_SWEEP="0.9 0.85 0.8 0.75"
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (BRV best for Idefics2: λ=0.85)"
            ;;
        *)
            echo "  [auto] lambda sweep: $MERGE_LAMBDA_SWEEP (default)"
            ;;
    esac
fi

# ── Auto-set merge formula per model type ────────────────────────
# Default: "brv" for all models (trade-off at target neurons).
# Override with --merge-formula additive if you want purely additive.
if ! $_USER_SET_FORMULA; then
    echo "  [auto] merge formula: $MERGE_FORMULA (trade-off at target neurons)"
fi

# Resolve PMBT label dir (prefer full-mode)
# qwen2vl-brv uses same VLM as qwen2vl, so labels are interchangeable
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
_PM_MODE="$OUTPUT_DIR/$MODEL_NAME/llm_permutation"
if [[ -d "$_PM_FULL" ]]; then
    MERGE_LABEL_DIR="$_PM_FULL"
elif [[ -d "$_PM_MODE" ]]; then
    MERGE_LABEL_DIR="$_PM_MODE"
elif [[ "$MODEL_TYPE" == "qwen2vl-brv" ]]; then
    # Fallback: qwen2vl-brv shares the same VLM arch as qwen2vl
    _PM_QWEN="results/3-classify/full/qwen2-vl-7b/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
    if [[ -d "$_PM_QWEN" ]]; then
        MERGE_LABEL_DIR="$_PM_QWEN"
        echo "  [fallback] Using qwen2-vl-7b labels for qwen2vl-brv"
    else
        echo "  ERROR: No PMBT labels found for qwen2vl-brv. Run steps 1-4 for qwen2vl first."
        exit 1
    fi
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

# ── Test mode: minimal run to verify pipeline works ──────────
# 1 lambda × 2 masks (text_inject + uniform) × POPE 20 questions only
# No CHAIR, no VLMEvalKit, no model saving → finishes in ~5 minutes
if [[ "$MODE" == "test" ]]; then
    MERGE_LAMBDA_SWEEP="0.9"
    _MERGE_EXTRA="--include_uniform_baseline --no_save_model --n_pope_questions 20"
    _TEST_MODE=true
    echo "  [test] lambda=0.9, 2 masks only, POPE 20 questions, no save, no VLMEvalKit"
else
    _TEST_MODE=false
fi

GMEM_NUM=${GPU_GMEM_TIERS[0]%%G*}; GMEM_TAG=$((GMEM_NUM / 10))
JN13="13_${SHORT_MODEL}"
JOB_NAME="${JN13}_g${GMEM_TAG}"
RESULT_FILE="$MERGE_OUT_DIR/text_inject/merge_results.json"

# Map shell model-type to Python model-type (qwen2vl-brv uses same arch as qwen2vl)
MERGE_MODEL_TYPE="$MODEL_TYPE"
[[ "$MODEL_TYPE" == "qwen2vl-brv" ]] && MERGE_MODEL_TYPE="qwen2vl"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    # ── Phase A: Merge (+ inline POPE in test mode) ────────────────
    if $_TEST_MODE; then
        # Test mode: merge + inline POPE eval in one shot
        MERGE_CMD="$PYTHON $MERGE_SCRIPT \
            --method text_inject \
            --vlm_path $MODEL_PATH \
            --base_llm_path $MERGE_BASE_LLM_PATH \
            --math_llm_path $MERGE_MATH_LLM_PATH \
            --label_dir $MERGE_LABEL_DIR \
            --model_type $MERGE_MODEL_TYPE \
            --n_layers $N_LAYERS \
            --output_dir $MERGE_OUT_DIR/text_inject \
            --lambda_sweep $MERGE_LAMBDA_SWEEP \
            --merge_formula $MERGE_FORMULA \
            --pope_path $POPE_PATH \
            --pope_img_dir $POPE_IMG_DIR \
            --eval_pope \
            $_MERGE_EXTRA"
    else
        # Full mode: merge only, no eval (Phase B handles eval)
        MERGE_CMD="$PYTHON $MERGE_SCRIPT \
            --method text_inject \
            --vlm_path $MODEL_PATH \
            --base_llm_path $MERGE_BASE_LLM_PATH \
            --math_llm_path $MERGE_MATH_LLM_PATH \
            --label_dir $MERGE_LABEL_DIR \
            --model_type $MERGE_MODEL_TYPE \
            --n_layers $N_LAYERS \
            --output_dir $MERGE_OUT_DIR/text_inject \
            --lambda_sweep $MERGE_LAMBDA_SWEEP \
            --merge_formula $MERGE_FORMULA \
            --no_eval_pope \
            --device cpu \
            $_MERGE_EXTRA"
    fi

    echo "  ── Phase A: Merge$($_TEST_MODE && echo ' + POPE eval' || echo ' only (CPU)') ──"
    echo "  → $JOB_NAME → $MERGE_OUT_DIR/text_inject"
    rm -f "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err"

    if $LOCAL; then
        (cd "$WORK_DIR" && $MERGE_CMD) \
            2>&1 | tee "$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log"
    else
        if $_TEST_MODE; then
            # Test mode needs 1 GPU for inline POPE eval + 96GB RAM for BRV formula
            bsub -q "$QUEUE" -J "$JOB_NAME" \
                -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                -R "rusage[mem=98304] order[-gpu_maxfactor]" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
                "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 && cd $WORK_DIR && $MERGE_CMD"
        else
            # Full mode: CPU-only merge, no GPU needed
            # Limit threads to avoid TERM_THREADLIMIT on clusters
            # HF_HUB_OFFLINE=1 prevents download threads when loading from cache
            bsub -q "$QUEUE" -J "$JOB_NAME" \
                -R "rusage[mem=98304] order[-gpu_maxfactor]" \
                -oo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP_LOG_DIR/${JOB_NAME}${LOG_SUFFIX}.err" \
                "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 && cd $WORK_DIR && $MERGE_CMD"
        fi
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

# ── Phase B: Parallel evaluation (1 GPU per model × benchmark) ────
# Each (model variant, benchmark) pair gets its own GPU job.
# Skipped in test mode (Phase A already does inline POPE eval).

if $_TEST_MODE; then
    echo ""
    echo "  ── Phase B: SKIPPED (test mode — POPE ran inline in Phase A) ──"
else

echo ""
echo "  ── Phase B: Parallel evaluation (1 GPU per model × benchmark) ──"

SKIP_COMPLETED=true  # skip benchmarks with done flags

# All benchmarks to evaluate via VLMEvalKit (matching BRV evaluate_vlm.sh)
# BRV runs each MathVerse sub-split individually and uses MathVision_MINI.
VLMEVAL_BENCHMARKS=(
    MathVista_MINI
    MathVerse_MINI
    MathVerse_MINI_Vision_Only
    MathVerse_MINI_Vision_Dominant
    MathVerse_MINI_Vision_Intensive
    MathVerse_MINI_Text_Lite
    MathVerse_MINI_Text_Dominant
    MathVision_MINI
    MM-Math
    DynaMath
    MMStar
)
# Total: 11 VLMEvalKit jobs + 1 POPE job + 1 CHAIR job = 13 jobs per model

_EVAL_SUBMITTED=0
_EVAL_SKIPPED=0

# Export OPENAI_API_KEY for VLMEvalKit GPT scoring on compute nodes
_OPENAI_EXPORT=""
if [[ -f "$WORK_DIR/modern_vlms/VLMEvalKit/.env" ]]; then
    _OPENAI_KEY=$(grep OPENAI_API_KEY "$WORK_DIR/modern_vlms/VLMEvalKit/.env" | head -1 | cut -d= -f2-)
    if [[ -n "$_OPENAI_KEY" ]]; then
        _OPENAI_EXPORT="export OPENAI_API_KEY=$_OPENAI_KEY && "
        echo "  [env] OPENAI_API_KEY loaded from .env (${#_OPENAI_KEY} chars)"
    fi
elif [[ -n "$OPENAI_API_KEY" ]]; then
    _OPENAI_EXPORT="export OPENAI_API_KEY=$OPENAI_API_KEY && "
    echo "  [env] OPENAI_API_KEY from environment"
fi

# ── Baseline eval (original VLM, no merge) ────────────────────
_BASELINE_DIR="$MERGE_OUT_DIR/text_inject/baseline/"
mkdir -p "$_BASELINE_DIR"

# POPE baseline (all 3 strategies: random, popular, adversarial)
if $MERGE_EVAL_POPE; then
_BL_POPE_JOB="13e_${SHORT_MODEL}_baseline_pope"
_BL_POPE_FLAG="${_BASELINE_DIR}pope_adversarial_done.flag"  # last strategy = all done
if [[ -f "$_BL_POPE_FLAG" ]] && $SKIP_COMPLETED; then
    _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
elif is_job_active "$_BL_POPE_JOB"; then
    _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
else
    _BL_POPE_CMD="cd $WORK_DIR && $PYTHON $EVAL_SINGLE_SCRIPT \
        --baseline \
        --model_type $MERGE_MODEL_TYPE \
        --vlm_path $MODEL_PATH \
        --output_dir $_BASELINE_DIR \
        --eval_pope \
        --pope_dir $POPE_DIR \
        --pope_img_dir $POPE_IMG_DIR"
    if $LOCAL; then
        eval "$_BL_POPE_CMD" 2>&1 | tee "$STEP_LOG_DIR/${_BL_POPE_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$_BL_POPE_JOB" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=49152] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${_BL_POPE_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${_BL_POPE_JOB}${LOG_SUFFIX}.err" \
            "$_BL_POPE_CMD"
        _EVAL_SUBMITTED=$((_EVAL_SUBMITTED + 1))
    fi
fi
fi  # MERGE_EVAL_POPE

# CHAIR baseline
if $MERGE_EVAL_CHAIR; then
_BL_CHAIR_JOB="13e_${SHORT_MODEL}_baseline_chair"
_BL_CHAIR_FLAG="${_BASELINE_DIR}chair_seed7_done.flag"
if [[ -f "$_BL_CHAIR_FLAG" ]] && $SKIP_COMPLETED; then
    _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
elif is_job_active "$_BL_CHAIR_JOB"; then
    _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
else
    _BL_CHAIR_CMD="cd $WORK_DIR && $PYTHON $EVAL_SINGLE_SCRIPT \
        --baseline \
        --model_type $MERGE_MODEL_TYPE \
        --vlm_path $MODEL_PATH \
        --output_dir $_BASELINE_DIR \
        --eval_chair \
        --pope_img_dir $POPE_IMG_DIR \
        --coco_ann_dir ${COCO_ANN_DIR:-data/annotations} \
        --chair_n_images ${CHAIR_N_IMAGES:-500}"
    if $LOCAL; then
        eval "$_BL_CHAIR_CMD" 2>&1 | tee "$STEP_LOG_DIR/${_BL_CHAIR_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$_BL_CHAIR_JOB" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=49152] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${_BL_CHAIR_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${_BL_CHAIR_JOB}${LOG_SUFFIX}.err" \
            "$_BL_CHAIR_CMD"
        _EVAL_SUBMITTED=$((_EVAL_SUBMITTED + 1))
    fi
fi
fi  # MERGE_EVAL_CHAIR

# VLMEvalKit baseline (1 job per benchmark)
for _bench in "${VLMEVAL_BENCHMARKS[@]}"; do
    _BL_BENCH_JOB="13e_${SHORT_MODEL}_baseline_${_bench}"
    _BL_BENCH_FLAG="${_BASELINE_DIR}vlmeval_${_bench}_done.flag"
    if [[ -f "$_BL_BENCH_FLAG" ]] && $SKIP_COMPLETED; then
        _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
        continue
    fi
    if is_job_active "$_BL_BENCH_JOB"; then
        _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
        continue
    fi
    _BL_BENCH_CMD="${_OPENAI_EXPORT}cd $WORK_DIR && $PYTHON $EVAL_SINGLE_SCRIPT \
        --baseline \
        --model_type $MERGE_MODEL_TYPE \
        --vlm_path $MODEL_PATH \
        --output_dir $_BASELINE_DIR \
        --eval_vlmevalkit \
        --vlmevalkit_benchmarks $_bench \
        ${VLMEVALKIT_DIR:+--vlmevalkit_dir $VLMEVALKIT_DIR}"
    if $LOCAL; then
        eval "$_BL_BENCH_CMD" 2>&1 | tee "$STEP_LOG_DIR/${_BL_BENCH_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$_BL_BENCH_JOB" \
            -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
            -R "rusage[mem=49152] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP_LOG_DIR/${_BL_BENCH_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP_LOG_DIR/${_BL_BENCH_JOB}${LOG_SUFFIX}.err" \
            "$_BL_BENCH_CMD"
        _EVAL_SUBMITTED=$((_EVAL_SUBMITTED + 1))
    fi
done
echo "    → baseline: 7 jobs (POPE + CHAIR + ${#VLMEVAL_BENCHMARKS[@]} VLMEvalKit)"

# ── Merged variant eval jobs ──────────────────────────────────

# Pre-compute expected model variant names from lambda sweep × mask flags
# (same naming convention as neuron_weight_merge.py Step D)
_METHOD="text_inject"
_VARIANTS=()
for _lam in $MERGE_LAMBDA_SWEEP; do
    _VARIANTS+=("${_METHOD}_lambda${_lam}")
    $MERGE_INCLUDE_UNIFORM   && _VARIANTS+=("${_METHOD}_uniform_lambda${_lam}")
    $MERGE_INCLUDE_MULTIMODAL && _VARIANTS+=("${_METHOD}_text_multi_lambda${_lam}")
    $MERGE_INCLUDE_VISUAL_ONLY && _VARIANTS+=("${_METHOD}_visual_only_lambda${_lam}")
    $MERGE_INCLUDE_VISUAL_MULTI && _VARIANTS+=("${_METHOD}_visual_multi_lambda${_lam}")
    $MERGE_INCLUDE_MULTIMODAL_ONLY && _VARIANTS+=("${_METHOD}_multimodal_only_lambda${_lam}")
    $MERGE_INCLUDE_RANDOM    && _VARIANTS+=("${_METHOD}_random_lambda${_lam}")
done

echo "  Expected variants: ${#_VARIANTS[@]}"

for _run_name in "${_VARIANTS[@]}"; do
    _model_dir="$MERGE_OUT_DIR/text_inject/${_run_name}/"
    _state_file="${_model_dir}model/merged_state_dict.pt"

        # ── A) POPE job (all 3 strategies, 1 GPU) ──────────────────
        if $MERGE_EVAL_POPE; then
        _POPE_JOB="13e_${SHORT_MODEL}_${_run_name}_pope"
        _POPE_FLAG="${_model_dir}pope_adversarial_done.flag"  # last strategy = all done

        if [[ -f "$_POPE_FLAG" ]] && $SKIP_COMPLETED; then
            _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
        elif is_job_active "$_POPE_JOB"; then
            _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
        else
            _POPE_CMD="cd $WORK_DIR && $PYTHON $EVAL_SINGLE_SCRIPT \
                --state_path $_state_file \
                --model_type $MERGE_MODEL_TYPE \
                --vlm_path $MODEL_PATH \
                --output_dir $_model_dir \
                --eval_pope \
                --pope_dir $POPE_DIR \
                --pope_img_dir $POPE_IMG_DIR"

            if $LOCAL; then
                eval "$_POPE_CMD" \
                    2>&1 | tee "$STEP_LOG_DIR/${_POPE_JOB}${LOG_SUFFIX}.log"
            else
                bsub -q "$QUEUE" -J "$_POPE_JOB" \
                    -w "done($JOB_NAME)" \
                    -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                    -R "rusage[mem=49152] order[-gpu_maxfactor]" \
                    -oo "$WORK_DIR/$STEP_LOG_DIR/${_POPE_JOB}${LOG_SUFFIX}.log" \
                    -eo "$WORK_DIR/$STEP_LOG_DIR/${_POPE_JOB}${LOG_SUFFIX}.err" \
                    "$_POPE_CMD"
                _EVAL_SUBMITTED=$((_EVAL_SUBMITTED + 1))
            fi
        fi
        fi  # MERGE_EVAL_POPE

        # ── B) CHAIR job (1 GPU) ──────────────────────────────────
        if $MERGE_EVAL_CHAIR; then
        _CHAIR_JOB="13e_${SHORT_MODEL}_${_run_name}_chair"
        _CHAIR_FLAG="${_model_dir}chair_seed7_done.flag"

        if [[ -f "$_CHAIR_FLAG" ]] && $SKIP_COMPLETED; then
            _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
        elif is_job_active "$_CHAIR_JOB"; then
            _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
        else
            _CHAIR_CMD="cd $WORK_DIR && $PYTHON $EVAL_SINGLE_SCRIPT \
                --state_path $_state_file \
                --model_type $MERGE_MODEL_TYPE \
                --vlm_path $MODEL_PATH \
                --output_dir $_model_dir \
                --eval_chair \
                --pope_img_dir $POPE_IMG_DIR \
                --coco_ann_dir ${COCO_ANN_DIR:-data/annotations} \
                --chair_n_images ${CHAIR_N_IMAGES:-500}"

            if $LOCAL; then
                eval "$_CHAIR_CMD" \
                    2>&1 | tee "$STEP_LOG_DIR/${_CHAIR_JOB}${LOG_SUFFIX}.log"
            else
                bsub -q "$QUEUE" -J "$_CHAIR_JOB" \
                    -w "done($JOB_NAME)" \
                    -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                    -R "rusage[mem=49152] order[-gpu_maxfactor]" \
                    -oo "$WORK_DIR/$STEP_LOG_DIR/${_CHAIR_JOB}${LOG_SUFFIX}.log" \
                    -eo "$WORK_DIR/$STEP_LOG_DIR/${_CHAIR_JOB}${LOG_SUFFIX}.err" \
                    "$_CHAIR_CMD"
                _EVAL_SUBMITTED=$((_EVAL_SUBMITTED + 1))
            fi
        fi
        fi  # MERGE_EVAL_CHAIR

        # ── C) VLMEvalKit jobs (1 per benchmark, 1 GPU each) ─────
        for _bench in "${VLMEVAL_BENCHMARKS[@]}"; do
            _BENCH_JOB="13e_${SHORT_MODEL}_${_run_name}_${_bench}"
            _BENCH_FLAG="${_model_dir}vlmeval_${_bench}_done.flag"

            if [[ -f "$_BENCH_FLAG" ]] && $SKIP_COMPLETED; then
                _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
                continue
            fi
            if is_job_active "$_BENCH_JOB"; then
                _EVAL_SKIPPED=$((_EVAL_SKIPPED + 1))
                continue
            fi

            _BENCH_CMD="${_OPENAI_EXPORT}cd $WORK_DIR && $PYTHON $EVAL_SINGLE_SCRIPT \
                --state_path $_state_file \
                --model_type $MERGE_MODEL_TYPE \
                --vlm_path $MODEL_PATH \
                --output_dir $_model_dir \
                --eval_vlmevalkit \
                --vlmevalkit_benchmarks $_bench \
                ${VLMEVALKIT_DIR:+--vlmevalkit_dir $VLMEVALKIT_DIR}"

            if $LOCAL; then
                eval "$_BENCH_CMD" \
                    2>&1 | tee "$STEP_LOG_DIR/${_BENCH_JOB}${LOG_SUFFIX}.log"
            else
                bsub -q "$QUEUE" -J "$_BENCH_JOB" \
                    -w "done($JOB_NAME)" \
                    -gpu "num=1:gmem=${GPU_GMEM_TIERS[0]}$($GPU_EXCLUSIVE && echo :mode=exclusive_process)" \
                    -R "rusage[mem=49152] order[-gpu_maxfactor]" \
                    -oo "$WORK_DIR/$STEP_LOG_DIR/${_BENCH_JOB}${LOG_SUFFIX}.log" \
                    -eo "$WORK_DIR/$STEP_LOG_DIR/${_BENCH_JOB}${LOG_SUFFIX}.err" \
                    "$_BENCH_CMD"
                _EVAL_SUBMITTED=$((_EVAL_SUBMITTED + 1))
            fi
        done

        echo "    → $_run_name: 7 parallel jobs (POPE + CHAIR + ${#VLMEVAL_BENCHMARKS[@]} VLMEvalKit)"
    done

_N_BENCHMARKS=$(( ${#VLMEVAL_BENCHMARKS[@]} + 2 ))  # +2 for POPE and CHAIR
echo ""
echo "  Phase B summary: ${_EVAL_SUBMITTED} eval jobs submitted, ${_EVAL_SKIPPED} skipped"
echo "  Layout: 1 baseline + ${#_VARIANTS[@]} merged variants × ${_N_BENCHMARKS} benchmarks each, 1 GPU per job"

fi  # end if ! $_TEST_MODE (Phase B)

fi  # end step 13 (text_inject)


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
#     MathVision_MINI       — competition-level math reasoning (primary for A, C)
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
VLMEVAL_BENCHMARKS="MathVista_MINI MathVerse_MINI MathVerse_MINI_Vision_Only MathVerse_MINI_Vision_Dominant MathVerse_MINI_Vision_Intensive MathVerse_MINI_Text_Lite MathVerse_MINI_Text_Dominant MathVision_MINI MM-Math DynaMath MMStar POPE HallusionBench MMMU_DEV_VAL MMMU_Pro MME ScienceQA_IMG LLaVABench A-OKVQA CHAIR"
# Per-method benchmarks matching original papers:
#   SNRF (Cui et al. 2602.19058): MathVista, MMMU_DEV_VAL, MMMU_PRO, MME, POPE, ScienceQA_IMG
#   SRF  (Ali et al. 2511.12220): POPE, LLaVABench, A-OKVQA, CHAIR
#   All methods now evaluated on unified superset above.
VLMEVAL_CONFIG="$WORK_DIR/modern_vlms/VLMEvalKit/vlmeval/config.py"

# ── Auto-detect lambda from step 22 if not manually set ─────────────────────
LAMBDA_JSON="$WORK_DIR/results/17-select-lambda/${MODE_DIR}/${MODEL_NAME}/lambda_summary.json"
if [[ -z "${MERGE_LAMBDA16:-}" ]] && [[ -f "$LAMBDA_JSON" ]]; then
    # Extract best_pmbt tag (e.g. "text_l0.85" or "tmulti_l0.9")
    BEST_TAG=$(python3 -c "import json; d=json.load(open('$LAMBDA_JSON')); print(d.get('best_pmbt',''))" 2>/dev/null)
    if [[ -n "$BEST_TAG" ]]; then
        # Extract lambda value from tag: "text_l0.85" → "0.85", "tmulti_l0.9" → "0.9"
        AUTO_LAMBDA=$(echo "$BEST_TAG" | grep -oP '(?<=_l)\d+\.\d+' || true)
        if [[ -n "$AUTO_LAMBDA" ]]; then
            COMPOSE_LAMBDA16="$AUTO_LAMBDA"
            echo "  [auto] Lambda16=$COMPOSE_LAMBDA16 (from step 22: $BEST_TAG)"
        else
            COMPOSE_LAMBDA16="0.1"
            echo "  [warn] Could not parse lambda from step 22 tag '$BEST_TAG', using default λ=0.9"
        fi
    else
        COMPOSE_LAMBDA16="0.1"
        echo "  [warn] No best_pmbt in $LAMBDA_JSON, using default λ=0.9"
    fi
else
    COMPOSE_LAMBDA16="${MERGE_LAMBDA16:-0.1}"
    if [[ -n "${MERGE_LAMBDA16:-}" ]]; then
        echo "  [manual] Lambda16=$COMPOSE_LAMBDA16 (from --merge-lambda16)"
    elif [[ ! -f "$LAMBDA_JSON" ]]; then
        echo "  [warn] No step 22 results found at $LAMBDA_JSON, using default λ=0.9"
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
            qwen2vl|qwen2vl-brv)
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
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${COMPOSE_LAMBDA16}/model"
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
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_text_multi_lambda${COMPOSE_LAMBDA16}/model"
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
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_visual_only_lambda${COMPOSE_LAMBDA16}/model"
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
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_visual_multi_lambda${COMPOSE_LAMBDA16}/model"
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
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_random_lambda${COMPOSE_LAMBDA16}/model"
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
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_multimodal_only_lambda${COMPOSE_LAMBDA16}/model"
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
            VLM_PATH="$WORK_DIR/results/14-snrf/${MODE_DIR}/${MODEL_NAME}/snrf_r16_b0p5"
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
            VLM_PATH="$WORK_DIR/results/14-snrf/${MODE_DIR}/${MODEL_NAME}/snrf_random_r16_b0p5"
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
            VLM_PATH="$WORK_DIR/results/15-srf/${MODE_DIR}/${MODEL_NAME}/srf_a0.5_m10"
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
            VLM_PATH="$WORK_DIR/results/15-srf/${MODE_DIR}/${MODEL_NAME}/srf_random_a0.5_m10"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: SRF random model not found at $VLM_PATH"
                echo "         Run step 24 with --include_random first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        uniform|brv)
            TAG="uniform_l${COMPOSE_LAMBDA16}"
            VLM_NAME="${MODEL_NAME}_${TAG}"
            VLM_PATH="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_uniform_lambda${COMPOSE_LAMBDA16}/model"
            if [[ ! -d "$VLM_PATH" ]]; then
                echo "  ERROR: uniform model not found at $VLM_PATH"
                echo "         Run step 16 with --merge-uniform first."
                continue
            fi
            convert_if_needed "$VLM_PATH" "$TAG"
            ;;
        25|compose_layer1|composed_layer1)
            # Find the composed Layer 1a+1c model
            COMP_MODEL=$(find results/18-compose-layer1/${MODE_DIR}/${MODEL_NAME}/ -name "pytorch_model.bin" -path "*/srf_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
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
# STEP 16 (tune_lambda): BRV-style lambda tuning on MathVista
#
#   Evaluates text_inject (and optionally other masks) at multiple
#   lambda values on MathVista_MINI only, then reports the best.
#   Follows BRV methodology: tune on MathVista, apply everywhere.
#
#   Models at each lambda already exist from step 16's lambda sweep.
#   This step converts + evaluates + summarizes.
#
#   Lambdas: --tune-lambdas "0.9 0.85 0.8 0.7"
#   Masks:   --tune-masks "16 text_multi"   (16=text_inject)
#
#   Run as:
#     bash run_pipeline.sh --step 16 --model-type llava-ov --gmem 40
#     bash run_pipeline.sh --step 16 --model-type qwen2vl --gmem 40 --tune-lambdas "0.9 0.85 0.8 0.7"
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "tune_lambda" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 16: BRV-style lambda tuning on MathVista"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/21-tune-lambda"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

TUNE_OUT_DIR="results/16-tune-lambda/${MODE_DIR}/${MODEL_NAME}"
VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
fi
VLMEVAL_CONFIG="$WORK_DIR/modern_vlms/VLMEvalKit/vlmeval/config.py"
TUNE_BENCHMARK="MathVista_MINI"

echo "  Lambdas: $TUNE_LAMBDAS"
echo "  Masks:   $TUNE_MASKS"
echo "  Benchmark: $TUNE_BENCHMARK (BRV methodology)"
echo "  Models from: results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/"
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
            qwen2vl|qwen2vl-brv)
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
        13|text_inject)   echo "text_inject_lambda${lam}/model" ;;
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
        13|text_inject)   echo "text_l${lam}" ;;
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
MERGE_BASE_DIR="$WORK_DIR/results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject"

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

fi  # end step 16 (tune_lambda)


# ═══════════════════════════════════════════════════════════════
# STEP 17 (select_lambda): Combine step 16 POPE + step 21 MathVista → pick best λ
#
#   Reads:
#     - results/13-weight-merge/<mode>/<model>/text_inject/merge_results.json (POPE)
#     - results/16-tune-lambda/<mode>/<model>/*/MathVista*.csv (MathVista)
#
#   Produces:
#     - results/17-select-lambda/<mode>/<model>/lambda_summary.json
#     - Console table with all λ × mask × {POPE, MathVista}
#
#   Run as:
#     bash run_pipeline.sh --step 17 --model-type llava-ov
#     bash run_pipeline.sh --step 17 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "select_lambda" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 17: Select best lambda (POPE from step 16 + MathVista from step 21)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STEP_LOG_DIR="${LOG_DIR}/22-select-lambda"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

SELECT_OUT_DIR="results/17-select-lambda/${MODE_DIR}/${MODEL_NAME}"
mkdir -p "$SELECT_OUT_DIR"

POPE_RESULTS="results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/merge_results.json"
TUNE_DIR="results/16-tune-lambda/${MODE_DIR}/${MODEL_NAME}"

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
# step 21 tags: text_l0.9, tmulti_l0.8, visonly_l0.7, uniform_l0.7
# step 16 names: text_inject_lambda0.9, text_inject_text_multi_lambda0.8, etc.
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

fi  # end step 17 (select_lambda)


# ═══════════════════════════════════════════════════════════════
# STEP 14 (snrf): SNRF + PMBT — Shared Neuron Low-Rank Fusion (Layer 1b)
#   Stage 1: profile shared neurons between math LLM + VLM backbone
#   Stage 2: SVD rank-16 injection at PMBT text ∩ shared neuron positions
#   Reference: Cui et al., "Do LLMs and VLMs Share Neurons?" (2026)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "snrf" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 14: SNRF + PMBT — Shared Neuron Low-Rank Fusion (Layer 1b)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/23-snrf"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

SNRF_OUT_DIR="results/14-snrf/${MODE_DIR}/${MODEL_NAME}"

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
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
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
JN14="14_${SHORT_MODEL}"
JOB_NAME="${JN14}_g${GMEM_TAG}"

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

fi  # end step 14 (snrf)


# ═══════════════════════════════════════════════════════════════
# STEP 15 (srf): PMBT-Guided SRF — Spectral Representation Filtering (Layer 1c)
#   Stage 1: profile hallucination modes from contrastive POPE activations
#   Stage 2: apply spectral filter to down_proj at visual neuron positions
#   Reference: Ali, Zoabi & Wolf, "Suppressing VLM Hallucinations with SRF" (2025)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "srf" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 15: PMBT-Guided SRF — Spectral Representation Filtering (Layer 1c)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/24-srf"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

SRF_OUT_DIR="results/15-srf/${MODE_DIR}/${MODEL_NAME}"

# Resolve PMBT label dir
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
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
JN15="15_${SHORT_MODEL}"
JOB_NAME="${JN15}_g${GMEM_TAG}"

if is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already active"
    SKIPPED=$((SKIPPED + 1))
else
    # Prefer contrastive POPE from step 10 if it exists
    _SRF_POPE="results/10-halluc_scores/${CLASSIFY_MODE_DIR}/${MODEL_NAME}/contrastive_pope.jsonl"
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

fi  # end step 15 (srf)


# ═══════════════════════════════════════════════════════════════
# STEP 18 (compose_layer1): Compose Layer 1a + 1c (or 1b + 1c)
#   Applies both the text-neuron weight merge (step 16 or 23) and
#   the visual-neuron spectral filter (step 24) to produce a single
#   model with both edits. Safe because masks are disjoint by construction.
#
#   Usage:
#     bash run_pipeline.sh --step 18 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "compose_layer1" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 18: Compose Layer 1a/1b + Layer 1c into single model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/25-compose-layer1"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

COMPOSE_L1_OUT_DIR="results/18-compose-layer1/${MODE_DIR}/${MODEL_NAME}"
mkdir -p "$COMPOSE_L1_OUT_DIR"

# ── Determine which Layer 1a/1b model to use ─────────────────
# Prefer step 22's best lambda selection for step 16 (Layer 1a)
BEST_LAMBDA_JSON="results/17-select-lambda/${MODE_DIR}/${MODEL_NAME}/lambda_summary.json"
if [[ -f "$BEST_LAMBDA_JSON" ]]; then
    BEST_TAG=$(python3 -c "import json; d=json.load(open('$BEST_LAMBDA_JSON')); print(d.get('best_brv', d.get('best', '')))" 2>/dev/null)
    _LAMBDA25=$(echo "$BEST_TAG" | grep -oP '(?<=_l)\d+\.\d+' || true)
    [[ -z "$_LAMBDA25" ]] && _LAMBDA25="0.1"
    L1A_MODEL="results/13-weight-merge/${MODE_DIR}/${MODEL_NAME}/text_inject/text_inject_lambda${_LAMBDA25}/model"
    echo "  Layer 1a model (step 16, λ=${_LAMBDA25} from step 22): $L1A_MODEL"
else
    # Fallback: use step 23 SNRF output
    L1A_MODEL=$(find results/14-snrf/${MODE_DIR}/${MODEL_NAME}/ -name "pytorch_model.bin" -path "*/snrf_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [[ -n "$L1A_MODEL" ]]; then
        echo "  Layer 1b model (step 23 SNRF): $L1A_MODEL"
    else
        echo "  ERROR: No Layer 1a/1b model found."
        echo "  Run step 16 + 22 (Layer 1a) or step 23 (Layer 1b) first."
        exit 1
    fi
fi

# ── Determine which Layer 1c SRF edits to use ────────────────
SRF_MODES_PT="results/15-srf/${MODE_DIR}/${MODEL_NAME}/hallucination_modes.pt"
if [[ ! -f "$SRF_MODES_PT" ]]; then
    echo "  ERROR: No SRF hallucination modes found at $SRF_MODES_PT"
    echo "  Run step 24 (SRF profile) first."
    exit 1
fi

# ── Resolve PMBT label dir ───────────────────────────────────
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
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
JN18="18_${SHORT_MODEL}"
JOB_NAME="${JN18}_g${GMEM_TAG}"

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
        --eigenvecs_dir results/15-srf/${MODE_DIR}/${MODEL_NAME} \
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

fi  # end step 18 (compose_layer1)


# ═══════════════════════════════════════════════════════════════
# STEP 21 (weight_diff_rank): Weight Diff Effective Rank Analysis
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
#     bash run_pipeline.sh --step 21 --model-type qwen2vl
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "weight_diff_rank" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 21: Weight Diff Effective Rank Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/29-weight-diff-rank"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

RANK_OUT_DIR="results/21-weight-diff-rank/${MODE_DIR}/${MODEL_NAME}"
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
    qwen2vl|qwen2vl-brv)
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
_PM_FULL="results/3-classify/full/$MODEL_NAME/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
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
CROSS_RESULT="results/21-weight-diff-rank/${MODE_DIR}/cross_model_rank_comparison.json"
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
        --base_dir results/21-weight-diff-rank/${MODE_DIR} \
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

fi  # end step 21 (weight_diff_rank)



# ═══════════════════════════════════════════════════════════════════
# STEP 11 (vit_analysis): VIT Weight Change × PMBT Label Correlation
# ═══════════════════════════════════════════════════════════════════
#
# Compares per-neuron weight changes between base LLM and VLM,
# then tests whether visually-responsive neurons (PMBT label = visual)
# are the neurons most modified during visual instruction tuning.
#
# No GPU required — loads state dicts on CPU only.
#
# Usage:
#     bash run_pipeline.sh --step 11 --model-type llava-ov
#     bash run_pipeline.sh --step 11 --model-type qwen2vl,internvl,llava-ov
#
if [[ "$STEP" == "vit_analysis" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 11: VIT Weight Change Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VIT_SCRIPT="code/vit_weight_analysis.py"
VIT_OUT_DIR="results/11-vit-analysis/${MODE_DIR}/${MODEL_NAME}"
VIT_LOG_DIR="${LOG_DIR}/11-vit-analysis"
mkdir -p "$WORK_DIR/$VIT_OUT_DIR" "$WORK_DIR/$VIT_LOG_DIR"

# ── Resolve base LLM path ──────────────────────────────────
VIT_BASE_LLM=""
if [[ -n "$MERGE_BASE_LLM_PATH" ]]; then
    VIT_BASE_LLM="$MERGE_BASE_LLM_PATH"
else
    case "$MODEL_TYPE" in
        qwen2vl|qwen2vl-brv)
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
VIT_LABEL_DIR="${LABELS_BASE}/${MODEL_NAME}/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"

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
JOB_NAME="11_vit_${SHORT_MODEL}"

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

fi  # end step 11 (vit_analysis)


# ═══════════════════════════════════════════════════════════════
# STEP 22 (mathverse_ablation): MathVerse dissociation via VLMEvalKit
#
#   Creates ablated checkpoints (100% of each PMBT category zeroed),
#   then evaluates with VLMEvalKit on MathVerse subtasks:
#     - MathVerse_MINI_Text_Dominant  (reasoning, minimal visual dep.)
#     - MathVerse_MINI_Vision_Only    (visual perception, maximal visual dep.)
#
#   Follows SNRF's protocol: ablate all target neurons, compare vs baseline.
#   Baseline results reused from step 19 if available.
#
#   Run as:
#     bash run_pipeline.sh --step 22 --model-type llava-llama3 --gmem 40
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "mathverse_ablation" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 22: MathVerse dissociation (PMBT ablation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Separate dirs for option A (inline) vs B (VLMEvalKit)
if $INLINE_EVAL; then
    STEP22_LOG="${LOG_DIR}/22a-inline-ablation"
    STEP22_OUT="results/22a-inline-ablation/${MODE_DIR}/${MODEL_NAME}"
elif $SCORE_LOCAL; then
    STEP22_LOG="${LOG_DIR}/22-mathverse-ablation"
    if [[ -d "$WORK_DIR/results/22a-inline-ablation/${MODE_DIR}/${MODEL_NAME}" ]]; then
        STEP22_OUT="results/22a-inline-ablation/${MODE_DIR}/${MODEL_NAME}"
    elif [[ -d "$WORK_DIR/results/22b-vlmevalkit-ablation/${MODE_DIR}/${MODEL_NAME}" ]]; then
        STEP22_OUT="results/22b-vlmevalkit-ablation/${MODE_DIR}/${MODEL_NAME}"
    elif [[ -d "$WORK_DIR/results/mathverse_vlmekit_ablation/${MODE_DIR}/${MODEL_NAME}" ]]; then
        STEP22_OUT="results/mathverse_vlmekit_ablation/${MODE_DIR}/${MODEL_NAME}"
        echo "  [score-local] Using legacy dir: $STEP22_OUT"
    else
        STEP22_OUT="results/22a-inline-ablation/${MODE_DIR}/${MODEL_NAME}"
    fi
else
    STEP22_LOG="${LOG_DIR}/22b-vlmevalkit-ablation"
    STEP22_OUT="results/22b-vlmevalkit-ablation/${MODE_DIR}/${MODEL_NAME}"
fi
mkdir -p "$STEP22_LOG" "$WORK_DIR/$STEP22_LOG"
mkdir -p "$WORK_DIR/$STEP22_OUT"

STEP22_SCRIPT="code/create_ablated_checkpoints.py"
STEP22_CATEGORIES="visual text multimodal"

# Resolve label dir (same logic as step 7)
if [[ -n "$OUTPUT_DIR_USER" ]]; then
    STEP22_LABEL_DIR="${OUTPUT_DIR_USER}/${MODEL_NAME}/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
else
    _PM_FULL="results/3-classify/full${RUN_SUFFIX}/${MODEL_NAME}/llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
    _PM_MODE="${OUTPUT_DIR}/${MODEL_NAME}/llm_permutation"
    if [[ -d "$WORK_DIR/$_PM_FULL" ]]; then
        STEP22_LABEL_DIR="$_PM_FULL"
    elif [[ -d "$WORK_DIR/$_PM_MODE" ]]; then
        STEP22_LABEL_DIR="$_PM_MODE"
    else
        echo "  ERROR: No PMBT labels found for $MODEL_NAME."
        echo "  Run steps 1-4 first."
        exit 1
    fi
fi
echo "  PMBT labels: $STEP22_LABEL_DIR"

# VLMEvalKit setup (reuse from step 19 pattern)
VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/.venv/bin/python"
if [[ "$MODEL_TYPE" == "internvl" ]]; then
    VLMEVAL_PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
fi
VLMEVAL_CONFIG="$WORK_DIR/modern_vlms/VLMEvalKit/vlmeval/config.py"
STEP22_BENCHMARKS="MathVerse_MINI_Text_Dominant MathVerse_MINI_Vision_Only"

# Export OPENAI_API_KEY for VLMEvalKit GPT scoring
if [[ -f "$WORK_DIR/modern_vlms/VLMEvalKit/.env" ]]; then
    _OPENAI_KEY=$(grep OPENAI_API_KEY "$WORK_DIR/modern_vlms/VLMEvalKit/.env" | head -1 | cut -d= -f2-)
    [[ -n "$_OPENAI_KEY" ]] && export OPENAI_API_KEY="$_OPENAI_KEY"
fi

echo "  Model:       $MODEL_NAME ($MODEL_TYPE)"
echo "  Categories:  $STEP22_CATEGORIES"
echo "  Benchmarks:  $STEP22_BENCHMARKS"
echo "  Rescore:     $RESCORE"
echo "  Score local: $SCORE_LOCAL"
echo "  Inline eval: $INLINE_EVAL"
echo "  Inline cond: $INLINE_CONDITIONS"
echo "  Inline hooks:$INLINE_HOOKS"
echo "  Inline bench:$INLINE_BENCHMARKS"
echo "  Inline limit:$INLINE_LIMIT"
echo "  Output:      $STEP22_OUT"

# ── Score-local mode: score/merge existing results ──
if $SCORE_LOCAL; then
    echo ""
    _INLINE_MODEL_TYPE="$MODEL_TYPE"
    [[ "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "qwen25vl-3b" ]] && _INLINE_MODEL_TYPE="qwen2vl"

    _FOUND_HOOKS=false
    for _HDIR in "$WORK_DIR/$STEP22_OUT"/gate "$WORK_DIR/$STEP22_OUT"/gate_up "$WORK_DIR/$STEP22_OUT"/attn; do
        [[ -d "$_HDIR" ]] || continue
        _HOOK=$(basename "$_HDIR")
        _PH5=$(find "$_HDIR" -name "phase3_*.json" ! -name "phase3_snrf_ablation.json" 2>/dev/null | wc -l)
        if [[ $_PH5 -gt 0 ]]; then
            _FOUND_HOOKS=true
            echo "  [score-local] Merging $_PH5 files for hook=$_HOOK..."
            $PYTHON "$WORK_DIR/code/equal_fraction_ablation.py" --phase 3 \
                --model_type "$_INLINE_MODEL_TYPE" --model_name "$MODEL_NAME" \
                --n_layers "$N_LAYERS" --label_dir "$STEP22_LABEL_DIR" \
                --taxonomy pmbt --output_dir "$_HDIR" --phase3_condition merge
            echo ""
        fi
    done

    if ! $_FOUND_HOOKS; then
        _PH5=$(find "$WORK_DIR/$STEP22_OUT" -maxdepth 1 -name "phase3_*.json" ! -name "phase3_snrf_ablation.json" 2>/dev/null | wc -l)
        if [[ $_PH5 -gt 0 ]]; then
            echo "  [score-local] Merging $_PH5 phase5 files..."
            $PYTHON "$WORK_DIR/code/equal_fraction_ablation.py" --phase 3 \
                --model_type "$_INLINE_MODEL_TYPE" --model_name "$MODEL_NAME" \
                --n_layers "$N_LAYERS" --label_dir "$STEP22_LABEL_DIR" \
                --taxonomy pmbt --output_dir "$WORK_DIR/$STEP22_OUT" --phase3_condition merge
        else
            echo "  [score-local] Running local letter-extraction scoring (VLMEvalKit results)..."
            $PYTHON "$WORK_DIR/code/score_mathverse_local.py" \
                --base_dir "$WORK_DIR/$STEP22_OUT" --output "$WORK_DIR/$STEP22_OUT/results_table_local.csv"
        fi
    fi

    echo ""
    echo "  Results in: $STEP22_OUT/"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Inline mode (option A): phase 5, parallel per hook × benchmark × condition ──
elif $INLINE_EVAL; then
    echo ""
    echo "  [inline] Phase 5 — parallel per hook × benchmark × condition"

    _INLINE_MODEL_TYPE="$MODEL_TYPE"
    [[ "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "qwen25vl-3b" ]] && _INLINE_MODEL_TYPE="qwen2vl"

    MATHVERSE_DIR="${WORK_DIR}/data/mathverse"

    # Determine conditions
    _ALL_CONDITIONS="baseline visual text multimodal random_visual_count random_text_count random_multimodal_count"
    if [[ "$INLINE_CONDITIONS" == "all" ]]; then
        _CONDITIONS="$_ALL_CONDITIONS"
    else
        _CONDITIONS="${INLINE_CONDITIONS//,/ }"
    fi

    # Determine benchmarks
    _BENCHMARKS="${INLINE_BENCHMARKS//,/ }"

    _GPU_MODE="num=1:gmem=${GPU_GMEM_TIERS[0]}"
    $GPU_EXCLUSIVE && _GPU_MODE="${_GPU_MODE}:mode=exclusive_process" || true

    # Parse hooks list
    IFS=',' read -ra _HOOKS <<< "$INLINE_HOOKS"

    for _HOOK in "${_HOOKS[@]}"; do
        echo ""
        echo "  ── Hook: $_HOOK ──"

        case "$_HOOK" in
            gate)    _HSUFFIX="";         _HSHORT="g"  ;;
            gate_up) _HSUFFIX="_gate_up"; _HSHORT="gu" ;;
            attn)    _HSUFFIX="_attn";    _HSHORT="at" ;;
            *)       _HSUFFIX="_${_HOOK}"; _HSHORT="${_HOOK:0:2}" ;;
        esac

        # Resolve label dir for this hook
        _HOOK_LABEL_DIR=""
        _PM_FULL="results/3-classify/full${RUN_SUFFIX}/${MODEL_NAME}/llm_permutation${_HSUFFIX}"
        _PM_MODE="${OUTPUT_DIR}/${MODEL_NAME}/llm_permutation${_HSUFFIX}"
        if [[ -n "$OUTPUT_DIR_USER" ]]; then
            _HOOK_LABEL_DIR="${OUTPUT_DIR_USER}/${MODEL_NAME}/llm_permutation${_HSUFFIX}"
        elif [[ -d "$WORK_DIR/$_PM_FULL" ]]; then
            _HOOK_LABEL_DIR="$_PM_FULL"
        elif [[ -d "$WORK_DIR/$_PM_MODE" ]]; then
            _HOOK_LABEL_DIR="$_PM_MODE"
        else
            echo "    WARNING: No labels for hook=$_HOOK ($MODEL_NAME)"
            echo "    Expected: $WORK_DIR/$_PM_FULL"
            echo "    Skipping. Run: bash code/run_pipeline.sh --model-type $MODEL_TYPE --step 3 --hook-point $_HOOK"
            continue
        fi
        echo "    Labels: $_HOOK_LABEL_DIR"

        # Output subdir per hook
        _HOOK_OUT="${STEP22_OUT}/${_HOOK}"
        mkdir -p "$WORK_DIR/$_HOOK_OUT"

        # Base command (no benchmark-specific args yet)
        _HOOK_PHASE5_BASE="$PYTHON $WORK_DIR/code/equal_fraction_ablation.py --phase 3 \
            --model_type $_INLINE_MODEL_TYPE \
            --model_path $MODEL_PATH \
            --model_name $MODEL_NAME \
            --n_layers $N_LAYERS \
            --label_dir $_HOOK_LABEL_DIR \
            --taxonomy pmbt \
            --output_dir $WORK_DIR/$_HOOK_OUT \
            --phase3_limit $INLINE_LIMIT"

        if [[ -f "$WORK_DIR/$_HOOK_OUT/phase3_snrf_ablation.json" ]]; then
            echo "    [skip] Combined results exist"
            continue
        fi

        echo "    Conditions: $_CONDITIONS"
        echo "    Benchmarks: $_BENCHMARKS"

        for _BENCH in $_BENCHMARKS; do
            # Build benchmark-specific command
            case "$_BENCH" in
                POPE)
                    _BENCH_ARGS="--pope_path ${WORK_DIR}/${POPE_PATH} --pope_img_dir ${WORK_DIR}/${POPE_IMG_DIR}"
                    _BSHORT="P" ;;
                MV_Text_Dominant)
                    _BENCH_ARGS="--mathverse_dir $MATHVERSE_DIR --mathverse_subtasks Text_Dominant"
                    _BSHORT="TD" ;;
                MV_Vision_Only)
                    _BENCH_ARGS="--mathverse_dir $MATHVERSE_DIR --mathverse_subtasks Vision_Only"
                    _BSHORT="VO" ;;
                TriviaQA)
                    _BENCH_ARGS="--triviaqa_path ${WORK_DIR}/${TRIVIAQA_PATH} --triviaqa_num ${TRIVIAQA_NUM} --text_only_benchmarks"
                    _BSHORT="TQ" ;;
                all)
                    _BENCH_ARGS="--pope_path ${WORK_DIR}/${POPE_PATH} --pope_img_dir ${WORK_DIR}/${POPE_IMG_DIR} --mathverse_dir $MATHVERSE_DIR --mathverse_subtasks Text_Dominant,Text_Lite,Vision_Intensive,Vision_Dominant,Vision_Only --triviaqa_path ${WORK_DIR}/${TRIVIAQA_PATH} --triviaqa_num ${TRIVIAQA_NUM} --text_only_benchmarks"
                    _BSHORT="all" ;;
                *)
                    echo "    WARNING: Unknown benchmark $_BENCH, skipping"
                    continue ;;
            esac
            _BENCH_CMD="$_HOOK_PHASE5_BASE $_BENCH_ARGS"
            [[ "$_BENCH" != "all" ]] && _BENCH_CMD="$_BENCH_CMD --phase3_benchmark $_BENCH"

            for _COND in $_CONDITIONS; do
                case "$_COND" in
                    baseline)                 _CSHORT="bas" ;;
                    visual)                   _CSHORT="vis" ;;
                    text)                     _CSHORT="tex" ;;
                    multimodal)               _CSHORT="mul" ;;
                    random_visual_count)      _CSHORT="rVis" ;;
                    random_text_count)        _CSHORT="rTex" ;;
                    random_multimodal_count)  _CSHORT="rMul" ;;
                    *)                        _CSHORT="${_COND:0:3}" ;;
                esac

                if [[ "$_BENCH" == "all" ]]; then
                    _JOB_NAME="22a_${SHORT_MODEL}_${_HSHORT}_${_CSHORT}"
                    _COND_FILE="$WORK_DIR/$_HOOK_OUT/phase3_${_COND}.json"
                else
                    _JOB_NAME="22a_${SHORT_MODEL}_${_HSHORT}_${_BSHORT}_${_CSHORT}"
                    _COND_FILE="$WORK_DIR/$_HOOK_OUT/phase3_${_COND}_${_BENCH}.json"
                fi

                if [[ -f "$_COND_FILE" ]]; then
                    echo "    [skip] $_JOB_NAME — done"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                _COND_CMD="$_BENCH_CMD --phase3_condition $_COND"

                if $LOCAL; then
                    echo "    [local] $_HOOK/$_BENCH/$_COND..."
                    (cd "$WORK_DIR" && eval "$_COND_CMD") \
                        2>&1 | tee "$STEP22_LOG/${_JOB_NAME}${LOG_SUFFIX}.log"
                else
                    echo "    → $_JOB_NAME ($_HOOK/$_BENCH/$_COND)"
                    bsub -J "$_JOB_NAME" \
                        -q "$QUEUE" \
                        -gpu "$_GPU_MODE" \
                        -R "rusage[mem=65536] order[-gpu_maxfactor]" \
                        -oo "$WORK_DIR/$STEP22_LOG/${_JOB_NAME}${LOG_SUFFIX}.log" \
                        -eo "$WORK_DIR/$STEP22_LOG/${_JOB_NAME}${LOG_SUFFIX}.err" \
                        -cwd "$WORK_DIR" \
                        "$_COND_CMD"
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
        done
    done

    echo ""
    echo "  Step 22 (inline) summary:"
    echo "    Hooks:      $INLINE_HOOKS"
    echo "    Conditions: $_CONDITIONS"
    echo "    Benchmarks: $_BENCHMARKS"
    echo "    Output:     $STEP22_OUT/{hook}/phase3_{condition}_{benchmark}.json"
    echo ""
    echo "    After all jobs finish:"
    echo "      bash code/run_pipeline.sh --model-type $MODEL_TYPE --step 22 --score-local"

# ── VLMEvalKit mode (option B): checkpoints + separate eval jobs ──
else

# ── Phase A: Create ablated checkpoints ──
# One checkpoint per category: zero ALL neurons of that type
CKPT_JOB="22_ckpt_${SHORT_MODEL}"
CKPT_DONE="$WORK_DIR/$STEP22_OUT/checkpoints_done.flag"

if $RESCORE; then
    echo "  [rescore] Skipping checkpoint creation (scoring only)"
elif [[ -f "$CKPT_DONE" ]]; then
    echo "  [skip] Ablated checkpoints already created"
else
    echo "  [A] Creating ablated checkpoints..."

    # Map model type for the checkpoint script
    _CKPT_MODEL_TYPE="$MODEL_TYPE"
    [[ "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "qwen25vl-3b" ]] && _CKPT_MODEL_TYPE="qwen2vl"

    CKPT_CMD="$PYTHON $WORK_DIR/$STEP22_SCRIPT \
        --model_type $_CKPT_MODEL_TYPE \
        --model_path $MODEL_PATH \
        --label_dir $STEP22_LABEL_DIR \
        --n_layers $N_LAYERS \
        --output_dir $STEP22_OUT \
        --categories $STEP22_CATEGORIES && \
        touch $CKPT_DONE"

    rm -f "$WORK_DIR/$STEP22_LOG/${CKPT_JOB}${LOG_SUFFIX}."{log,err}

    if $LOCAL; then
        (cd "$WORK_DIR" && eval "$CKPT_CMD") \
            2>&1 | tee "$STEP22_LOG/${CKPT_JOB}${LOG_SUFFIX}.log"
    else
        bsub -q "$QUEUE" -J "$CKPT_JOB" \
            -R "rusage[mem=65536] order[-gpu_maxfactor]" \
            -oo "$WORK_DIR/$STEP22_LOG/${CKPT_JOB}${LOG_SUFFIX}.log" \
            -eo "$WORK_DIR/$STEP22_LOG/${CKPT_JOB}${LOG_SUFFIX}.err" \
            "cd $WORK_DIR && $CKPT_CMD"
        SUBMITTED=$((SUBMITTED + 1))
        echo "  → submitted $CKPT_JOB"
    fi
fi

# ── Phase B: Register models in VLMEvalKit config ──
# Also register baseline if needed
register_model_step22() {
    local vlm_name="$1"
    local vlm_path="$2"
    if ! grep -q "\"$vlm_name\"" "$VLMEVAL_CONFIG" 2>/dev/null; then
        case "$MODEL_TYPE" in
            qwen2vl|qwen25vl-7b|qwen25vl-3b)
                ENTRY_CLASS="Qwen2VLChat"
                ENTRY_EXTRA=", min_pixels=1280*28*28, max_pixels=16384*28*28"
                ANCHOR='"Qwen2-VL-7B-Instruct": partial('
                ;;
            llava-ov)
                ENTRY_CLASS="LLaVA_OneVision_HF"
                ENTRY_EXTRA=""
                ANCHOR='"llava-onevision-qwen2-7b-ov-hf": partial('
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
            llava-liuhaotian)
                ENTRY_CLASS="LLaVA_Next"
                ENTRY_EXTRA=""
                ANCHOR='"llava_next_mistral_7b": partial('
                ;;
            *)
                echo "  WARNING: Unknown model type $MODEL_TYPE for registration"
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

# Register baseline
BASELINE_NAME="${MODEL_NAME}_baseline"
register_model_step22 "$BASELINE_NAME" "$MODEL_PATH"

# Register ablated models
for _CAT in $STEP22_CATEGORIES; do
    _ABLATED_NAME="${MODEL_NAME}_ablated_${_CAT}"
    _ABLATED_PATH="$WORK_DIR/$STEP22_OUT/ablated_${_CAT}"
    register_model_step22 "$_ABLATED_NAME" "$_ABLATED_PATH"
done

# Register random baseline models
for _CAT in $STEP22_CATEGORIES; do
    _RANDOM_NAME="${MODEL_NAME}_random_${_CAT}_count"
    _RANDOM_PATH="$WORK_DIR/$STEP22_OUT/random_${_CAT}_count"
    if [[ -d "$WORK_DIR/$STEP22_OUT/random_${_CAT}_count" ]]; then
        register_model_step22 "$_RANDOM_NAME" "$_RANDOM_PATH"
    fi
done

# ── Phase C: Submit VLMEvalKit eval jobs ──
# For each (baseline + 3 ablated + 3 random) × N benchmarks
# Build full target list: baseline + ablated categories + random categories
STEP22_EVAL_TARGETS="baseline"
for _CAT in $STEP22_CATEGORIES; do
    STEP22_EVAL_TARGETS="$STEP22_EVAL_TARGETS $_CAT"
done
for _CAT in $STEP22_CATEGORIES; do
    if [[ -d "$WORK_DIR/$STEP22_OUT/random_${_CAT}_count" ]]; then
        STEP22_EVAL_TARGETS="$STEP22_EVAL_TARGETS random_${_CAT}_count"
    fi
done
echo "  Eval targets: $STEP22_EVAL_TARGETS"

# If --rescore: delete done flags for cells missing _score.xlsx, add --reuse
if $RESCORE; then
    echo ""
    echo "  [rescore mode] Cleaning stale done flags and restoring backup files..."
    
    # Restore raw files from VLMEvalKit backup dirs
    find "$WORK_DIR/$STEP22_OUT" -path "*/bak_*/*.xlsx" 2>/dev/null | while read _f; do
        _tdir=$(dirname "$_f" | sed 's|/bak_[^/]*||')
        _fname=$(basename "$_f")
        if [[ ! -f "$_tdir/$_fname" ]]; then
            cp "$_f" "$_tdir/"
            echo "    Restored: $_fname"
        fi
    done
    find "$WORK_DIR/$STEP22_OUT" -path "*/bak_*/*.pkl" 2>/dev/null | while read _f; do
        _tdir=$(dirname "$_f" | sed 's|/bak_[^/]*||')
        _fname=$(basename "$_f")
        if [[ ! -f "$_tdir/$_fname" ]]; then
            cp "$_f" "$_tdir/"
        fi
    done
    
    # Delete done flags where _score.xlsx is missing
    for _T in $STEP22_EVAL_TARGETS; do
        for _B in $STEP22_BENCHMARKS; do
            _FLAG="$WORK_DIR/$STEP22_OUT/${_T}/vlmeval_${_B}_done.flag"
            [[ ! -f "$_FLAG" ]] && continue
            # Check if score file exists (search recursively)
            if [[ "$_B" == "POPE" ]]; then
                _HAS_RESULT=$(find "$WORK_DIR/$STEP22_OUT/${_T}/vlmevalkit_results" -name "*POPE.xlsx" ! -name "*auxmatch*" ! -path "*/bak_*" 2>/dev/null | head -1)
            else
                _HAS_RESULT=$(find "$WORK_DIR/$STEP22_OUT/${_T}/vlmevalkit_results" -name "*${_B}_gpt-4o-mini_score.xlsx" ! -path "*/bak_*" 2>/dev/null | head -1)
            fi
            if [[ -z "$_HAS_RESULT" ]]; then
                echo "    Removing stale flag: $(basename $_FLAG)"
                rm -f "$_FLAG"
            fi
        done
    done
    REUSE_FLAG="--reuse"
    echo "  [rescore mode] Will use --reuse (skip inference, score only)"
else
    REUSE_FLAG=""
fi

for _TARGET in $STEP22_EVAL_TARGETS; do
    if [[ "$_TARGET" == "baseline" ]]; then
        _EVAL_NAME="${MODEL_NAME}_baseline"
        _DEP=""
    elif [[ "$_TARGET" == random_* ]]; then
        _EVAL_NAME="${MODEL_NAME}_${_TARGET}"
        if [[ ! -f "$CKPT_DONE" ]]; then
            _DEP="done($CKPT_JOB)"
        else
            _DEP=""
        fi
    else
        _EVAL_NAME="${MODEL_NAME}_ablated_${_TARGET}"
        # Depend on checkpoint creation job
        if [[ ! -f "$CKPT_DONE" ]]; then
            _DEP="done($CKPT_JOB)"
        else
            _DEP=""
        fi
    fi

    for _BENCH in $STEP22_BENCHMARKS; do
        # Short target name for job naming (avoid collisions)
        case "$_TARGET" in
            baseline) _TSHORT="bas" ;;
            visual)   _TSHORT="vis" ;;
            text)     _TSHORT="tex" ;;
            multimodal) _TSHORT="mul" ;;
            random_visual_count)   _TSHORT="rVis" ;;
            random_text_count)     _TSHORT="rTex" ;;
            random_multimodal_count) _TSHORT="rMul" ;;
            *) _TSHORT="${_TARGET:0:4}" ;;
        esac
        # Short benchmark name
        case "$_BENCH" in
            MathVerse_MINI_Text_Dominant)    _BSHORT="TD" ;;
            MathVerse_MINI_Vision_Only)     _BSHORT="VO" ;;
            MathVerse_MINI_Vision_Dominant) _BSHORT="VD" ;;
            POPE) _BSHORT="POPE" ;;
            *) _BSHORT="${_BENCH##*_}" ;;
        esac
        JOB_NAME="22_${SHORT_MODEL}_${_TSHORT}_${_BSHORT}"

        # Check if already done (look for VLMEvalKit output)
        _RESULT_DIR="$STEP22_OUT/${_TARGET}/vlmevalkit_results"
        _DONE_FLAG="$STEP22_OUT/${_TARGET}/vlmeval_${_BENCH}_done.flag"
        if [[ -f "$WORK_DIR/$_DONE_FLAG" ]]; then
            echo "  [skip] $JOB_NAME — already done"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        EVAL_CMD="$VLMEVAL_PYTHON $WORK_DIR/modern_vlms/VLMEvalKit/run.py \
            --data $_BENCH \
            --model $_EVAL_NAME \
            --work-dir $STEP22_OUT/$_TARGET/vlmevalkit_results \
            $REUSE_FLAG \
            --verbose && \
            touch $WORK_DIR/$_DONE_FLAG"

        echo "  → $JOB_NAME ($_EVAL_NAME × $_BENCH)"
        rm -f "$WORK_DIR/$STEP22_LOG/${JOB_NAME}${LOG_SUFFIX}."{log,err}

        if $LOCAL; then
            (cd "$WORK_DIR" && source modern_vlms/VLMEvalKit/.env 2>/dev/null; eval "$EVAL_CMD") \
                2>&1 | tee "$STEP22_LOG/${JOB_NAME}${LOG_SUFFIX}.log"
        else
            if $RESCORE; then
                # Scoring only — no GPU needed, just API calls
                bsub -q "$QUEUE" -J "$JOB_NAME" \
                    -R "rusage[mem=8192]" \
                    -oo "$WORK_DIR/$STEP22_LOG/${JOB_NAME}${LOG_SUFFIX}.log" \
                    -eo "$WORK_DIR/$STEP22_LOG/${JOB_NAME}${LOG_SUFFIX}.err" \
                    "cd $WORK_DIR && source modern_vlms/VLMEvalKit/.env 2>/dev/null; $EVAL_CMD"
            else
                _GPU_MODE="num=1:gmem=${GPU_GMEM_TIERS[0]}"
                $GPU_EXCLUSIVE && _GPU_MODE="${_GPU_MODE}:mode=exclusive_process" || true
                BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
                    -gpu "$_GPU_MODE" \
                    -R "rusage[mem=65536] order[-gpu_maxfactor]" \
                    -oo "$WORK_DIR/$STEP22_LOG/${JOB_NAME}${LOG_SUFFIX}.log" \
                    -eo "$WORK_DIR/$STEP22_LOG/${JOB_NAME}${LOG_SUFFIX}.err")
                [[ -n "$_DEP" ]] && BSUB_ARGS+=(-w "$_DEP")
                bsub "${BSUB_ARGS[@]}" "cd $WORK_DIR && source modern_vlms/VLMEvalKit/.env 2>/dev/null; $EVAL_CMD"
            fi
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

echo ""
echo "  Step 22 summary:"
echo "    Checkpoints: $STEP22_OUT/ablated_{visual,text,multimodal}/"
echo "    Eval results: $STEP22_OUT/{baseline,visual,text,multimodal}/vlmevalkit_results/"
echo "    Total eval jobs: $(( (3 + 1) * 2 )) (4 models × 2 benchmarks)"

fi  # end score-local / inline / vlmevalkit branch

fi  # end step 22 (mathverse_ablation)


# ═══════════════════════════════════════════════════════════════
# STEP 23 (compare_hooks): Gate-only vs Gate*Up activation correlation
#
#   Compares two possible FFN neuron hook points:
#     - SiLU(gate_proj(x))            — gate only (what PMBT uses, following Xu et al.)
#     - SiLU(gate_proj(x)) * up_proj(x) — full intermediate (Geva's conceptual equivalent)
#
#   Computes per-neuron Spearman rank correlation across token positions.
#   If correlation is high (>0.95), the hook choice barely matters.
#   If low (<0.85), PMBT labels may differ depending on which hook is used.
#
#   Run as:
#     bash run_pipeline.sh --step 23 --model-type llava-llama3 [--local]
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "compare_hooks" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 23: Compare hook points (gate-only vs gate*up)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP23_LOG="${LOG_DIR}/23-compare-hooks"
mkdir -p "$STEP23_LOG" "$WORK_DIR/$STEP23_LOG"

STEP23_OUT="results/23-compare-hooks/${MODE_DIR}"
mkdir -p "$WORK_DIR/$STEP23_OUT"

STEP23_SCRIPT="code/compare_hook_points.py"
STEP23_RESULT="$WORK_DIR/$STEP23_OUT/hook_comparison_${MODEL_NAME}.json"

# Number of images — more in full mode
if [[ "$MODE_DIR" == "test" ]]; then
    STEP23_N_IMAGES=5
else
    STEP23_N_IMAGES=50
fi

# COCO images path
STEP23_COCO_DIR="${WORK_DIR}/data/val2014"

echo "  Model:       $MODEL_NAME ($MODEL_TYPE)"
echo "  N images:    $STEP23_N_IMAGES"
echo "  N layers:    $N_LAYERS"
echo "  Output:      $STEP23_RESULT"

# Check if already done
if [[ -f "$STEP23_RESULT" ]]; then
    echo "  [skip] Hook comparison already exists: $STEP23_RESULT"
    echo "  Delete the file to re-run."
else
# Map qwen25vl variants to qwen2vl for the Python script
    _STEP23_MODEL_TYPE="$MODEL_TYPE"
    [[ "$MODEL_TYPE" == "qwen25vl-7b" || "$MODEL_TYPE" == "qwen25vl-3b" ]] && _STEP23_MODEL_TYPE="qwen2vl"

    STEP23_CMD="$PYTHON $WORK_DIR/$STEP23_SCRIPT \
        --model_type ${_STEP23_MODEL_TYPE} \
        --model_path $MODEL_PATH \
        --n_layers $N_LAYERS \
        --n_images $STEP23_N_IMAGES \
        --coco_dir $STEP23_COCO_DIR \
        --output $STEP23_RESULT"

    if $LOCAL; then
        echo "  [local] Running hook comparison..."
        eval "$STEP23_CMD" 2>&1 | tee "$STEP23_LOG/23_hook_${SHORT_MODEL}.log"
    else
        _GMEM="${GPU_GMEM_TIERS[0]}"
        _GPU_STR="num=1:gmem=${_GMEM}$($GPU_EXCLUSIVE && echo ':mode=exclusive_process')"
        _JOB_NAME="23_hook_${SHORT_MODEL}"

        echo "  [submit] $QUEUE  gpu=$_GPU_STR  job=$_JOB_NAME"
        bsub -J "$_JOB_NAME" \
            -q "$QUEUE" \
            -gpu "$_GPU_STR" \
            -R "$GPU_RES_BASE" \
            -oo "$WORK_DIR/$STEP23_LOG/23_hook_${SHORT_MODEL}.log" \
            -eo "$WORK_DIR/$STEP23_LOG/23_hook_${SHORT_MODEL}.err" \
            -cwd "$WORK_DIR" \
            "$STEP23_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
fi

echo ""
echo "  Step 23 summary:"
echo "    Result: $STEP23_RESULT"
echo "    After completion, check the 'interpretation' field in the JSON."

fi  # end step 23 (compare_hooks)

# ═══════════════════════════════════════════════════════════════
# STEP 24 (ranked_ablation): D-ranked sweep ablation (Phase 4)
#
#   Ranks neurons within each PMBT category by D, norm, D×norm,
#   or D-then-norm, ablates top-K for a sweep of K values,
#   and compares against matched-count random baselines.
#
#   Benchmarks: MathVerse Text_Dominant + Vision_Only (default).
#
#   Run as:
#     bash run_pipeline.sh --step 24 --model-type llava-llama3 --long-desc
#     bash run_pipeline.sh --step 24 --model-type llava-llama3 --long-desc --p4-ranking D
#     bash run_pipeline.sh --step 24 --model-type llava-llama3 --long-desc --p4-limit 10  # smoke test
#     bash run_pipeline.sh --step 24 --model-type llava-llama3 --long-desc --mode test    # test labels
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "ranked_ablation" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 24: D-ranked sweep ablation (Phase 4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STEP24_SCRIPT="code/equal_fraction_ablation.py"
STEP24_LOG="${LOG_DIR}/24-ranked-ablation${_LOG_HOOK_SUFFIX}"
mkdir -p "$STEP24_LOG" "$WORK_DIR/$STEP24_LOG"

STEP24_OUT="results/24-ranked-ablation/${CLASSIFY_MODE_DIR}/${MODEL_NAME}"
mkdir -p "$WORK_DIR/$STEP24_OUT"

# Label directory — use the PMBT permutation labels
# Match step 25's resolution: try canonical path, then auto-detect via find
_STEP24_HOOK_DIR="llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
STEP24_LABEL_DIR="$OUTPUT_DIR/$MODEL_NAME/${_STEP24_HOOK_DIR}"
if ! $P4_BASELINE_ONLY && [[ ! -f "$WORK_DIR/$STEP24_LABEL_DIR/neuron_labels_permutation_all.json" ]]; then
    # Auto-detect: search for matching pattern (handles parameter suffixes)
    _STEP24_LABEL_FILE=$(find "$WORK_DIR/$OUTPUT_DIR/$MODEL_NAME" -path "*/${_STEP24_HOOK_DIR}/neuron_labels_permutation_all.json" 2>/dev/null | head -1)
    if [[ -n "$_STEP24_LABEL_FILE" ]]; then
        STEP24_LABEL_DIR="${_STEP24_LABEL_FILE%/neuron_labels_permutation_all.json}"
        STEP24_LABEL_DIR="${STEP24_LABEL_DIR#$WORK_DIR/}"
        echo "  [auto] Label dir resolved via find: $STEP24_LABEL_DIR"
    fi
fi

echo "  Model:       $MODEL_NAME ($MODEL_TYPE)"
echo "  Hook point:  $HOOK_POINT"
echo "  Ranking:     $P4_RANKING"
echo "  Sweep fracs: $P4_SWEEP_FRACS"
echo "  Random trials: $P4_RANDOM_TRIALS"
echo "  Categories:  $P4_CATEGORIES"
echo "  Benchmarks:  $P4_BENCHMARKS"
echo "  Label dir:   $STEP24_LABEL_DIR"
echo "  Output:      $STEP24_OUT"
echo "  MathVerse:   $MATHVERSE_DIR"
$IMPORTANCE_WEIGHT && echo "  IW suffix:   _iw (importance-weighted labels)"
$P4_BASELINE_ONLY && echo "  Mode:        BASELINE ONLY (no ablation)"
[[ "$P4_LIMIT" -gt 0 ]] && echo "  Sample limit: $P4_LIMIT (smoke test)"

# Final guard: if still missing after auto-detect, exit
if ! $P4_BASELINE_ONLY && [[ ! -f "$WORK_DIR/$STEP24_LABEL_DIR/neuron_labels_permutation_all.json" ]]; then
    echo "  ERROR: PMBT labels not found at $STEP24_LABEL_DIR/neuron_labels_permutation_all.json"
    echo "  Run step 3 + step 4 first to generate PMBT labels."
    exit 1
fi

# Common args for all step 24 jobs
_P4_COMMON="$PYTHON $WORK_DIR/$STEP24_SCRIPT \
    --phase 1 \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --n_layers $N_LAYERS \
    --hook_point $HOOK_POINT \
    --taxonomy pmbt \
    --ranking $P4_RANKING \
    --sweep_fracs $P4_SWEEP_FRACS \
    --n_random_trials $P4_RANDOM_TRIALS \
    --mathverse_dir $WORK_DIR/$MATHVERSE_DIR \
    --mathverse_subtasks Text_Dominant,Vision_Only \
    --pope_dir $WORK_DIR/$POPE_DIR \
    --pope_img_dir $WORK_DIR/$POPE_IMG_DIR \
    --triviaqa_path $WORK_DIR/$TRIVIAQA_PATH \
    --triviaqa_num $TRIVIAQA_NUM \
    ${P4_LIMIT:+--sample_limit $P4_LIMIT} \
    --output_dir $WORK_DIR/$STEP24_OUT"

# Short benchmark names for job IDs
_bench_short() {
    case "$1" in
        POPE)               echo "PO" ;;
        MV_Text_Dominant)   echo "TD" ;;
        MV_Vision_Only)     echo "VO" ;;
        TriviaQA)           echo "TQ" ;;
        *)                  echo "${1:0:2}" ;;
    esac
}

IFS=',' read -ra _P4_BENCHES <<< "$P4_BENCHMARKS"
IFS=',' read -ra _P4_CATS <<< "$P4_CATEGORIES"

if $P4_BASELINE_ONLY; then
    # ── Baseline: VLMEvalKit (matches BRV exactly) ──
    _ve_dataset() {
        case "$1" in
            MV_Text_Dominant)   echo "MathVerse_MINI_Text_Dominant" ;;
            MV_Vision_Only)     echo "MathVerse_MINI_Vision_Only" ;;
            MV_Text_Lite)       echo "MathVerse_MINI_Text_Lite" ;;
            MV_Vision_Dominant) echo "MathVerse_MINI_Vision_Dominant" ;;
            MV_Vision_Intensive) echo "MathVerse_MINI_Vision_Intensive" ;;
            POPE)               echo "POPE" ;;
            *)                  echo "" ;;
        esac
    }

    _VE_MODEL="UNKNOWN"
    [[ "$MODEL_TYPE" == "llava-llama3" ]] && _VE_MODEL="llava_next_llama3"
    [[ "$MODEL_TYPE" == "idefics2" ]] && _VE_MODEL="idefics2_8b"
    [[ "$MODEL_TYPE" == "qwen2vl" ]] && _VE_MODEL="Qwen2-VL-7B-Instruct"
    [[ "$MODEL_TYPE" == "qwen25vl-7b" ]] && _VE_MODEL="qwen2.5-vl-7b_baseline"
    [[ "$MODEL_TYPE" == "internvl" ]] && _VE_MODEL="internvl2.5-8b_baseline"

    _VE_PYTHON="$WORK_DIR/modern_vlms/VLMEvalKit_brv/.venv/bin/python"
    [[ "$MODEL_TYPE" == "internvl" ]] && _VE_PYTHON="$WORK_DIR/modern_vlms/intervl_env/.venv_internvl/bin/python"
    _VE_WORKDIR="$STEP24_OUT/vlmeval_baseline"
    mkdir -p "$WORK_DIR/$_VE_WORKDIR"

    echo ""
    echo "  ── VLMEvalKit baseline (BRV-identical) ──"
    echo "  Model: $_VE_MODEL"
    echo "  Judge: $VLMEVAL_JUDGE"

    for _P4_BN in "${_P4_BENCHES[@]}"; do
        _VE_D=$(_ve_dataset "$_P4_BN")
        [[ -z "$_VE_D" ]] && continue

        _BN_SHORT=$(_bench_short "$_P4_BN")
        _VE_JOB="24_${SHORT_MODEL}_bl_${_BN_SHORT}"

        _VE_EXISTING=$(find "$WORK_DIR/$_VE_WORKDIR" -name "*${_VE_D}*_score*" 2>/dev/null | head -1 || true)
        if [[ -n "$_VE_EXISTING" ]] && ! is_job_active "$_VE_JOB"; then
            echo "  [skip] $_VE_JOB — already done: $(basename "$_VE_EXISTING")"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # POPE only needs yes/no — cap generation to avoid 60h runs on damaged models
        _VE_EXTRA_ENV=""
        [[ "$_VE_D" == "POPE" ]] && _VE_EXTRA_ENV="export VLMEVAL_MAX_NEW_TOKENS=10 && "

        _VE_CMD="cd $WORK_DIR && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ${_VE_EXTRA_ENV}export OPENAI_API_KEY=\$(cat $VLMEVAL_DIR/.env 2>/dev/null | grep OPENAI_API_KEY | cut -d= -f2); $_VE_PYTHON $VLMEVAL_DIR/run.py --data $_VE_D --model $_VE_MODEL --judge $VLMEVAL_JUDGE --reuse --work-dir $WORK_DIR/$_VE_WORKDIR"

        if $LOCAL; then
            echo "  [local] VLMEvalKit $_VE_D..."
            eval "$_VE_CMD" 2>&1 | tee "$STEP24_LOG/${_VE_JOB}${LOG_SUFFIX}.log"
        else
            _GMEM="${GPU_GMEM_TIERS[0]}"
            _GPU_STR="num=1:gmem=${_GMEM}$($GPU_EXCLUSIVE && echo ':mode=exclusive_process')"
            echo "  [submit] $QUEUE  gpu=$_GPU_STR  job=$_VE_JOB  ($_VE_D)"
            bsub -J "$_VE_JOB" \
                -q "$QUEUE" -gpu "$_GPU_STR" -R "$GPU_RES_BASE" \
                -oo "$WORK_DIR/$STEP24_LOG/${_VE_JOB}${LOG_SUFFIX}.log" \
                -eo "$WORK_DIR/$STEP24_LOG/${_VE_JOB}${LOG_SUFFIX}.err" \
                -cwd "$WORK_DIR" \
                "$_VE_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done

else
    # ── Ablation: save checkpoint + VLMEvalKit for each config ──
    _R1_SCRIPT="$WORK_DIR/code/run1_ablation.py"
    _R1_OUT="$WORK_DIR/$STEP24_OUT/run1"
    mkdir -p "$_R1_OUT"

    IFS=',' read -ra _P4_FRACS <<< "$P4_SWEEP_FRACS"

    _frac_short() {
        printf "f%03.0f" "$(echo "$1 * 100" | bc)"
    }

    _ve_dataset() {
        case "$1" in
            MV_Text_Dominant)   echo "MathVerse_MINI_Text_Dominant" ;;
            MV_Vision_Only)     echo "MathVerse_MINI_Vision_Only" ;;
            POPE)               echo "POPE" ;;
            TriviaQA)           echo "TriviaQA" ;;
            *) echo "" ;;
        esac
    }

    echo ""
    echo "  ── Ablation (save checkpoint + VLMEvalKit) ──"

    for _P4_CAT in "${_P4_CATS[@]}"; do
        for _P4_BN in "${_P4_BENCHES[@]}"; do
            _VE_D=$(_ve_dataset "$_P4_BN")
            [[ -z "$_VE_D" ]] && continue

            _BN_SHORT=$(_bench_short "$_P4_BN")

            for _P4_FRAC in "${_P4_FRACS[@]}"; do
                _FR_SHORT=$(_frac_short "$_P4_FRAC")

                # Trial types: "ranked" + "0" through "N-1"
                _TRIAL_TYPES=("ranked")
                for (( _ti=0; _ti<$P4_RANDOM_TRIALS; _ti++ )); do
                    _TRIAL_TYPES+=("$_ti")
                done

                for _P4_TRIAL in "${_TRIAL_TYPES[@]}"; do
                    if [[ "$_P4_TRIAL" == "ranked" ]]; then
                        _TR_SHORT="rk"
                        _TRIAL_TAG="ranked"
                    else
                        _TR_SHORT="r${_P4_TRIAL}"
                        _TRIAL_TAG="r${_P4_TRIAL}"
                    fi

                    # Short ranking tag for job name
                    case "$P4_RANKING" in
                        D) _RK_SHORT="D" ;;
                        norm) _RK_SHORT="N" ;;
                        D_x_norm) _RK_SHORT="DN" ;;
                        *) _RK_SHORT="${P4_RANKING:0:2}" ;;
                    esac
                    _R1_JOB="24_${SHORT_MODEL}${_HOOK_SHORT}_${_RK_SHORT}_${_P4_CAT:0:3}_${_BN_SHORT}_${_FR_SHORT}_${_TR_SHORT}"
                    _HOOK_TAG="$HOOK_POINT"
                    if $P4_COMBINED_ATTN && [[ "$HOOK_POINT" != "attn" ]]; then
                        _R1_JOB="${_R1_JOB}_attn"
                        _HOOK_TAG="${HOOK_POINT}_attn"
                    fi
                    _R1_RESULT="$_R1_OUT/run1_${P4_RANKING}_${_HOOK_TAG}_${_P4_CAT}_${_VE_D}_f${_P4_FRAC}_${_TRIAL_TAG}.json"

                    if [[ -s "$_R1_RESULT" ]] && ! is_job_active "$_R1_JOB"; then
                        echo "  [skip] $_R1_JOB — already done"
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    _R1_CMD="$PYTHON $_R1_SCRIPT \
                        --model_type $MODEL_TYPE \
                        --model_path $MODEL_PATH \
                        --model_name $MODEL_NAME \
                        --n_layers $N_LAYERS \
                        --hook_point $HOOK_POINT \
                        --label_dir $WORK_DIR/$STEP24_LABEL_DIR \
                        --ranking $P4_RANKING \
                        --category $_P4_CAT \
                        --fraction $_P4_FRAC \
                        --trial_idx $_P4_TRIAL \
                        --benchmark $_VE_D \
                        --vlmeval_dir $WORK_DIR/$VLMEVAL_DIR \
                        --vlmeval_python $WORK_DIR/modern_vlms/VLMEvalKit_brv/.venv/bin/python \
                        --judge $VLMEVAL_JUDGE \
                        --output_dir $_R1_OUT"

                    # Note: TriviaQA now routes through VLMEvalKit (same as POPE/MathVerse)
                    # via the custom TriviaQA dataset class in VLMEvalKit_brv/vlmeval/dataset/triviaqa.py.
                    # No special arg needed — the class loads from ~/LMUData/TriviaQA.tsv.

                    # Combined mode: also ablate attention heads in same category
                    if $P4_COMBINED_ATTN && [[ "$HOOK_POINT" != "attn" ]]; then
                        _ATTN_LABEL_DIR="$OUTPUT_DIR/$MODEL_NAME/llm_permutation_attn${_GEN_DIR_SUFFIX}"
                        if [[ ! -d "$WORK_DIR/$_ATTN_LABEL_DIR" ]]; then
                            echo "  ERROR: Attention label dir not found: $_ATTN_LABEL_DIR"
                            echo "  Run step 3 with --hook-point attn first."
                            exit 1
                        fi
                        _R1_CMD="$_R1_CMD --label_dir_attn $WORK_DIR/$_ATTN_LABEL_DIR"
                    fi

                    if $LOCAL; then
                        echo "  [local] ${_P4_CAT}/${_P4_BN}/f${_P4_FRAC}/${_P4_TRIAL}..."
                        eval "$_R1_CMD" 2>&1 | tee "$STEP24_LOG/${_R1_JOB}${LOG_SUFFIX}.log"
                    else
                        _GMEM="${GPU_GMEM_TIERS[0]}"
                        _GPU_STR="num=1:gmem=${_GMEM}$($GPU_EXCLUSIVE && echo ':mode=exclusive_process')"
                        echo "  [submit] $QUEUE  gpu=$_GPU_STR  job=$_R1_JOB  cat=$_P4_CAT  bench=$_P4_BN  frac=$_P4_FRAC  trial=$_P4_TRIAL"
                        bsub -J "$_R1_JOB" \
                            -q "$QUEUE" -gpu "$_GPU_STR" -R "$GPU_RES_BASE" \
                            -oo "$WORK_DIR/$STEP24_LOG/${_R1_JOB}${LOG_SUFFIX}.log" \
                            -eo "$WORK_DIR/$STEP24_LOG/${_R1_JOB}${LOG_SUFFIX}.err" \
                            -cwd "$WORK_DIR" \
                            "$_R1_CMD"
                        SUBMITTED=$((SUBMITTED + 1))
                    fi
                done
            done
        done
    done
fi  # end baseline vs ablation

echo ""
echo "  Step 24 summary:"
if $P4_BASELINE_ONLY; then
    echo "    Baseline: ${#_P4_BENCHES[@]} VLMEvalKit jobs ($VLMEVAL_JUDGE judge)"
    echo "    Results: $STEP24_OUT/vlmeval_baseline/"
else
    _N_TRIALS=$(( P4_RANDOM_TRIALS + 1 ))
    _N_JOBS=$(( ${#_P4_CATS[@]} * ${#_P4_BENCHES[@]} * ${#_P4_FRACS[@]} * _N_TRIALS ))
    echo "    Ablation: $_N_JOBS jobs (${#_P4_CATS[@]} cat × ${#_P4_BENCHES[@]} bench × ${#_P4_FRACS[@]} frac × $_N_TRIALS trials)"
    echo "    Results: $STEP24_OUT/run1/"
    echo "    Cost: ~\$$((  _N_JOBS / 10 )) GPT scoring"
fi
echo "    Ranking: $P4_RANKING | Hook: $HOOK_POINT | Fracs: $P4_SWEEP_FRACS"

fi  # end step 24 (ranked_ablation)

# ═══════════════════════════════════════════════════════════════
# STEP 25: Weight merging (BRV-compatible + PMBT-guided)
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "weight_merge" ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 25: Weight Merging ($P5_MODE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

_P5_ROOT_OUT="$WORK_DIR/results/25-merge/$MODEL_NAME"
_P5_ROOT_LOG="$WORK_DIR/logs/25-merge/$MODEL_NAME"

# Judge suffix for dirs/job names (empty for default gpt-4o-mini)
_JUDGE_SUFFIX=""
[[ "$P5_JUDGE" != "gpt-4o-mini" ]] && _JUDGE_SUFFIX="_$(echo "$P5_JUDGE" | sed 's/gpt-//;s/-//g')"

# Unique short names for job IDs (avoids collisions)
_bench_short() {
    case "$1" in
        MathVista_MINI)                   echo "mvi" ;;
        MathVerse_MINI_Text_Dominant)     echo "mvtd" ;;
        MathVerse_MINI_Text_Lite)         echo "mvtl" ;;
        MathVerse_MINI_Vision_Intensive)  echo "mvvi" ;;
        MathVerse_MINI_Vision_Dominant)   echo "mvvd" ;;
        MathVerse_MINI_Vision_Only)       echo "mvvo" ;;
        MMStar)                           echo "mms" ;;
        DynaMath)                         echo "dyn" ;;
        MathVision_MINI)                  echo "mvis" ;;
        MM-Math)                          echo "mmm" ;;
        *)                                echo "${1:0:4}" ;;
    esac
}

# Auto-set math LLM path per model
if [[ -z "$MERGE_MATH_LLM_PATH" ]]; then
    case "$MODEL_TYPE" in
        llava-llama3)
            MERGE_MATH_LLM_PATH="$WORK_DIR/modern_vlms/pretrained/dart-math-llama3-8b-prop2diff"
            echo "  [auto] math LLM: dart-math-llama3-8b-prop2diff (BRV main donor)"
            ;;
        idefics2)
            MERGE_MATH_LLM_PATH="$WORK_DIR/modern_vlms/pretrained/MAmmoTH-7B-Mistral"
            echo "  [auto] math LLM: MAmmoTH-7B-Mistral (BRV Table 3 best for Idefics2, local)"
            ;;
        qwen2vl)
            MERGE_MATH_LLM_PATH="Qwen/Qwen2-Math-7B"
            echo "  [auto] math LLM: Qwen2-Math-7B (BRV Appendix E)"
            ;;
    esac
fi

# Short name for math LLM (used in dirs and job names)
_MATH_LLM_SHORT=$(basename "$MERGE_MATH_LLM_PATH" | sed \
    -e 's/dart-math-.*prop2diff/dart-prop/' \
    -e 's/dart-math-.*uniform/dart-uni/' \
    -e 's/MAmmoTH-7B-Mistral/mammoth1/' \
    -e 's/MAmmoTH2-7B/mammoth2/' \
    -e 's/MAmmoTH2-8B/mammoth2/' \
    -e 's/Qwen2-Math-7B/qwen2-math/' \
    -e 's/MetaMath-Mistral-7B/metamath/' \
    -e 's/DeepSeek-R1-Distill-Llama-8B/r1-distill/' \
    -e 's/Magpie-Align.*SFT.*/magpie/')
echo "  Math LLM short: $_MATH_LLM_SHORT"

# Add math LLM subdirectory
_P5_ROOT_OUT="$_P5_ROOT_OUT/$_MATH_LLM_SHORT"
_P5_ROOT_LOG="$_P5_ROOT_LOG/$_MATH_LLM_SHORT"

# Mode/alpha directory name
if [[ "$P5_MODE" == "base" || "$P5_MODE" == "uniform" ]]; then
    _MODE_DIR="uniform_a${P5_ALPHA}"
elif [[ "$P5_MODE" == "pmbt" ]]; then
    _SCOPE_SUFFIX=""
    [[ "$P5_PMBT_SCOPE" != "both" ]] && _SCOPE_SUFFIX="_${P5_PMBT_SCOPE}only"
    $P5_KV_MERGE && _SCOPE_SUFFIX="${_SCOPE_SUFFIX}_kv"
    [[ -n "$P5_ALPHA_OTHER" ]] && _SCOPE_SUFFIX="${_SCOPE_SUFFIX}_o${P5_ALPHA_OTHER}"
    [[ "$HOOK_POINT" != "gate_up" ]] && _SCOPE_SUFFIX="${_SCOPE_SUFFIX}_h${HOOK_POINT}"
    $P5_MIXED_HOOK && _SCOPE_SUFFIX="${_SCOPE_SUFFIX}_mixed"
    # MLP projection filter — e.g., "down" → _downproj, "gate,up" → _gateup
    if [[ "$P5_MLP_PROJS" != "gate,up,down" ]]; then
        _projs_tag=$(echo "$P5_MLP_PROJS" | sed 's/,//g')
        _SCOPE_SUFFIX="${_SCOPE_SUFFIX}_${_projs_tag}proj"
    fi
    _MODE_DIR="pmbt_t${P5_ALPHA_TEXT}_v${P5_ALPHA_VISUAL}_m${P5_ALPHA_MULTI}${_SCOPE_SUFFIX}"
fi

# Baseline gets its own directory (shared across modes)
_BL_OUT="$_P5_ROOT_OUT/baseline"
_BL_LOG="$_P5_ROOT_LOG/baseline"

# Merge + eval go under mode/alpha directory
_P5_OUT="$_P5_ROOT_OUT/$_MODE_DIR"
_P5_LOG="$_P5_ROOT_LOG/$_MODE_DIR"

# Separate log subdirs per judge when non-default
if [[ -n "$_JUDGE_SUFFIX" ]]; then
    _BL_LOG="${_BL_LOG}${_JUDGE_SUFFIX}"
    _P5_LOG="${_P5_LOG}${_JUDGE_SUFFIX}"
fi
mkdir -p "$_BL_OUT" "$_BL_LOG" "$_P5_OUT" "$_P5_LOG"

# Mode tag for job names — unique per mode + alpha combination
if [[ "$P5_MODE" == "base" || "$P5_MODE" == "uniform" ]]; then
    # uniform_a0.9 → uni9, uniform_a0.85 → uni85
    _alpha_short=$(echo "$P5_ALPHA" | sed 's/0\.//;s/\.//g')
    _P5_JOB_TAG="uni${_alpha_short}"
elif [[ "$P5_MODE" == "pmbt" ]]; then
    # pmbt t0.5 v1.0 m0.9 → pt5v10m9
    _at=$(echo "$P5_ALPHA_TEXT" | sed 's/0\.//;s/\.//g')
    _av=$(echo "$P5_ALPHA_VISUAL" | sed 's/0\.//;s/\.//g')
    _am=$(echo "$P5_ALPHA_MULTI" | sed 's/0\.//;s/\.//g')
    _scope_tag=""
    [[ "$P5_PMBT_SCOPE" != "both" ]] && _scope_tag="_${P5_PMBT_SCOPE}o"
    $P5_KV_MERGE && _scope_tag="${_scope_tag}_kv"
    [[ -n "$P5_ALPHA_OTHER" ]] && _scope_tag="${_scope_tag}_o$(echo $P5_ALPHA_OTHER | sed 's/0\.//;s/\.//g')"
    [[ "$HOOK_POINT" != "gate_up" ]] && _scope_tag="${_scope_tag}_h${_HOOK_SHORT}"
    $P5_MIXED_HOOK && _scope_tag="${_scope_tag}_mx"
    # MLP projection filter — e.g., "down" → _dp, "gate,up" → _gup
    if [[ "$P5_MLP_PROJS" != "gate,up,down" ]]; then
        _projs_short=$(echo "$P5_MLP_PROJS" | sed 's/gate/g/g;s/up/u/g;s/down/d/g;s/,//g')
        _scope_tag="${_scope_tag}_${_projs_short}p"
    fi
    _P5_JOB_TAG="pt${_at}v${_av}m${_am}${_scope_tag}"
fi

# Very short math LLM tag for job names
_MLM_TAG=$(echo "$_MATH_LLM_SHORT" | sed \
    -e 's/dart-prop/dp/' \
    -e 's/dart-uni/du/' \
    -e 's/mammoth1/m1/' \
    -e 's/mammoth2/m2/' \
    -e 's/qwen2-math/qm/' \
    -e 's/metamath/mm/' \
    -e 's/r1-distill/r1/' \
    -e 's/magpie/mg/')

# VLMEvalKit model names
case "$MODEL_TYPE" in
    llava-llama3) _VE_MODEL="llava_next_llama3" ;;
    idefics2)     _VE_MODEL="idefics2_8b" ;;
    qwen2vl)      _VE_MODEL="Qwen2-VL-7B-Instruct" ;;
esac

# Read OpenAI API key
_OPENAI_KEY=$(cat "$WORK_DIR/$VLMEVAL_DIR/.env" 2>/dev/null | grep OPENAI_API_KEY | cut -d= -f2)
_VE_PYTHON="$WORK_DIR/modern_vlms/VLMEvalKit_brv/.venv/bin/python"
_VE_DIR="$WORK_DIR/modern_vlms/VLMEvalKit_brv"

echo "  Model:     $MODEL_TYPE ($MODEL_NAME)"
echo "  VLM:       $MODEL_PATH"
echo "  Math LLM:  $MERGE_MATH_LLM_PATH"
echo "  Mode:      $P5_MODE"
echo "  Alpha:     $P5_ALPHA"
[[ -n "$P5_ALPHA_TEXT" ]] && echo "  Alpha text:    $P5_ALPHA_TEXT"
[[ -n "$P5_ALPHA_VISUAL" ]] && echo "  Alpha visual:  $P5_ALPHA_VISUAL"
[[ -n "$P5_ALPHA_MULTI" ]] && echo "  Alpha multi:   $P5_ALPHA_MULTI"
[[ "$P5_MODE" == "pmbt" ]] && echo "  PMBT scope:   $P5_PMBT_SCOPE"
[[ "$P5_MODE" == "pmbt" && "$P5_MLP_PROJS" != "gate,up,down" ]] && echo "  MLP projs:    $P5_MLP_PROJS (others stay at VLM baseline)"
[[ "$P5_MODE" == "pmbt" ]] && $P5_KV_MERGE && echo "  KV merge:     enabled (GQA majority vote)"
[[ "$P5_MODE" == "pmbt" && -n "$P5_ALPHA_OTHER" ]] && echo "  Alpha other:  $P5_ALPHA_OTHER (layernorms, embeddings)"
[[ "$P5_MODE" == "pmbt" ]] && $P5_MIXED_HOOK && echo "  Mixed-hook:   gate_proj→gate labels, up/down_proj→gate_up labels"
echo "  Judge:     $P5_JUDGE"
echo "  Output:    $_P5_OUT"
echo ""

IFS=',' read -ra _P5_BENCHES <<< "$P5_BENCHMARKS"

# ── Phase A: Baseline evaluation (optional) ──
if $P5_BASELINE; then
    echo "  ── Baseline evaluation ──"
    for _BN in "${_P5_BENCHES[@]}"; do
        _SHORT=$(_bench_short "$_BN")
        _JOB="25_${SHORT_MODEL}_${_MLM_TAG}_bl_${_SHORT}${_JUDGE_SUFFIX}"
        _SCORE=$(find "$_BL_OUT" -name "*${_BN}*_score.csv" 2>/dev/null | head -1 || true)
        if [[ -n "$_SCORE" ]]; then
            echo "  [skip] $_JOB — score exists"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if is_job_active "$_JOB"; then
            echo "  [skip] $_JOB — already running"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        # MMStar is multi-choice → exact_matching, not GPT judge
        _BN_JUDGE="$P5_JUDGE"
        [[ "$_BN" == "MMStar" || "$_BN" == "MME" || "$_BN" == "POPE" || "$_BN" == "HallusionBench" ]] && _BN_JUDGE="exact_matching"
        # TriviaQA: MMStar judge check doesn't apply; use default P5_JUDGE
        # TriviaQA goes through VLMEvalKit like all other benchmarks via the custom
        # TriviaQA class in VLMEvalKit_brv. Alias-matching runs locally (no API).
        _CMD="cd $WORK_DIR && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export OPENAI_API_KEY=$_OPENAI_KEY; $_VE_PYTHON $_VE_DIR/run.py --data $_BN --model $_VE_MODEL --judge $_BN_JUDGE --reuse --work-dir $_BL_OUT"
        if $LOCAL; then
            echo "  [local] Baseline on $_BN"
            eval "$_CMD"
        else
            echo "  [submit] $_JOB — baseline on $_BN"
            _GPU_STR="num=1:gmem=${GPU_GMEM_TIERS[0]}:j_exclusive=yes"
            bsub -J "$_JOB" -q "$QUEUE" -gpu "$_GPU_STR" -R "$GPU_RES_BASE" \
                -oo "$_BL_LOG/${_JOB}.log" -eo "$_BL_LOG/${_JOB}.err" \
                "$_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    echo ""
fi

# ── Phase B: Merge ──
if [[ "$P5_MODE" == "base" || "$P5_MODE" == "uniform" ]]; then
    _MERGE_PTH="$_P5_OUT/merged_model_${P5_ALPHA}.pth"
    _MERGE_TAG="uniform"
elif [[ "$P5_MODE" == "pmbt" ]]; then
    _MERGE_PTH="$_P5_OUT/merged_model_pmbt_t${P5_ALPHA_TEXT}_v${P5_ALPHA_VISUAL}_m${P5_ALPHA_MULTI}.pth"
    _MERGE_TAG="pmbt"
fi

echo "  ── Merging ($P5_MODE) ──"
if [[ -f "$_MERGE_PTH" ]]; then
    echo "  [skip] Merge exists: $_MERGE_PTH"
else
    _JOB="25_${SHORT_MODEL}_${_MLM_TAG}_merge_${_P5_JOB_TAG}"

    if is_job_active "$_JOB"; then
        echo "  [skip] $_JOB — already running"
        SKIPPED=$((SKIPPED + 1))
    else
        _MERGE_CMD="$PYTHON $WORK_DIR/$P5_MERGE_SCRIPT \
            --model1_path $MODEL_PATH \
            --model2_path $MERGE_MATH_LLM_PATH \
            --output_dir $_P5_OUT \
            --alpha $P5_ALPHA \
            --mode $P5_MODE"

        if [[ "$P5_MODE" == "pmbt" ]]; then
            # PMBT labels — try default path, then auto-detect
            _PMBT_BASE="$WORK_DIR/results/3-classify/full/$MODEL_NAME"

            # MLP labels use hook-specific dir:
            #   gate:    llm_permutation${_GEN_DIR_SUFFIX}/
            #   gate_up: llm_permutation_gate_up${_GEN_DIR_SUFFIX}/
            if $P5_MIXED_HOOK; then
                # Mixed-hook: main labels are always gate_up
                _MLP_HOOK_DIR="llm_permutation_gate_up${_GEN_DIR_SUFFIX}"
            else
                _MLP_HOOK_DIR="llm_permutation${_HOOK_SUFFIX}${_GEN_DIR_SUFFIX}"
            fi

            # MLP labels (needed for scope=mlp or scope=both)
            if [[ "$P5_PMBT_SCOPE" == "mlp" || "$P5_PMBT_SCOPE" == "both" ]]; then
                _PMBT_LABELS="${_PMBT_BASE}/${_MLP_HOOK_DIR}/neuron_labels_permutation_all.json"
                if [[ ! -f "$_PMBT_LABELS" ]]; then
                    # Auto-detect: search for matching pattern
                    _PMBT_LABELS=$(find "$_PMBT_BASE" -path "*/${_MLP_HOOK_DIR}/neuron_labels_permutation_all.json" 2>/dev/null | head -1)
                    if [[ -z "$_PMBT_LABELS" ]]; then
                        echo "  ERROR: No PMBT MLP labels found at ${_PMBT_BASE}/${_MLP_HOOK_DIR}/"
                        echo "  Run step 3 (classify) with --hook-point $HOOK_POINT first."
                        exit 1
                    fi
                fi
                echo "  [OK] MLP labels (${HOOK_POINT}): $_PMBT_LABELS"
                _MERGE_CMD="$_MERGE_CMD \
                    --pmbt_labels $_PMBT_LABELS"

                # Mixed-hook: additionally pass gate-only labels for gate_proj
                if $P5_MIXED_HOOK; then
                    _GATE_DIR="llm_permutation${_GEN_DIR_SUFFIX}"
                    _PMBT_GATE="${_PMBT_BASE}/${_GATE_DIR}/neuron_labels_permutation_all.json"
                    if [[ ! -f "$_PMBT_GATE" ]]; then
                        _PMBT_GATE=$(find "$_PMBT_BASE" -path "*/${_GATE_DIR}/neuron_labels_permutation_all.json" 2>/dev/null | head -1)
                    fi
                    if [[ -n "$_PMBT_GATE" && -f "$_PMBT_GATE" ]]; then
                        echo "  [OK] Gate labels (mixed-hook): $_PMBT_GATE"
                        _MERGE_CMD="$_MERGE_CMD --pmbt_labels_gate $_PMBT_GATE"
                    else
                        echo "  ERROR: Mixed-hook requires gate labels at ${_PMBT_BASE}/${_GATE_DIR}/"
                        echo "  Run step 3 (classify) with --hook-point gate first."
                        exit 1
                    fi
                fi
            fi

            _MERGE_CMD="$_MERGE_CMD \
                --alpha_text $P5_ALPHA_TEXT \
                --alpha_visual $P5_ALPHA_VISUAL \
                --alpha_multimodal $P5_ALPHA_MULTI"

            # Alpha for non-modality weights (layernorms, embeddings)
            if [[ -n "$P5_ALPHA_OTHER" ]]; then
                _MERGE_CMD="$_MERGE_CMD --alpha_other $P5_ALPHA_OTHER"
            fi

            # Attention labels (needed for scope=attn or scope=both)
            if [[ "$P5_PMBT_SCOPE" == "attn" || "$P5_PMBT_SCOPE" == "both" ]]; then
                _PMBT_ATTN="${_PMBT_BASE}/llm_permutation_attn${_GEN_DIR_SUFFIX}/neuron_labels_permutation_all.json"
                if [[ ! -f "$_PMBT_ATTN" ]]; then
                    _PMBT_ATTN=$(find "$_PMBT_BASE" -path "*/llm_permutation_attn*/neuron_labels_permutation_all.json" 2>/dev/null | head -1 || true)
                fi
                if [[ -n "$_PMBT_ATTN" && -f "$_PMBT_ATTN" ]]; then
                    _MERGE_CMD="$_MERGE_CMD --pmbt_labels_attn $_PMBT_ATTN"
                    echo "  [OK] Attention labels: $_PMBT_ATTN"
                else
                    echo "  [warn] No attn labels found — skipping attention selective merge"
                fi
            else
                echo "  [info] Scope=$P5_PMBT_SCOPE — skipping attention labels"
            fi

            # KV merge (GQA majority vote)
            if $P5_KV_MERGE; then
                _MERGE_CMD="$_MERGE_CMD --kv_merge"
                echo "  [OK] KV merge enabled (GQA majority vote from Q head labels)"
            fi

            # MLP projection filter (default gate,up,down merges all)
            if [[ "$P5_MLP_PROJS" != "gate,up,down" ]]; then
                _MERGE_CMD="$_MERGE_CMD --mlp_projs $P5_MLP_PROJS"
                echo "  [OK] MLP projs restricted to: $P5_MLP_PROJS"
            fi

            # Merge scope (mlp-only, attn-only, both)
            if [[ "$P5_PMBT_SCOPE" != "both" ]]; then
                _MERGE_CMD="$_MERGE_CMD --merge_scope $P5_PMBT_SCOPE"
                echo "  [OK] Merge scope: $P5_PMBT_SCOPE only"
            fi
        fi

        if $LOCAL; then
            echo "  [local] Merging..."
            eval "$_MERGE_CMD"
        else
            echo "  [submit] $_JOB"
            _GPU_STR="num=1:gmem=${GPU_GMEM_TIERS[0]}:j_exclusive=yes"
            bsub -J "$_JOB" -q "$QUEUE" -gpu "$_GPU_STR" \
                -R "rusage[mem=65536] order[-gpu_maxfactor]" \
                -oo "$_P5_LOG/${_JOB}.log" -eo "$_P5_LOG/${_JOB}.err" \
                "$_MERGE_CMD"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    fi
fi
echo ""

# ── Phase C: Evaluate merged model ──
if $P5_EVAL; then
    echo "  ── Evaluating merged model ──"
    _EVAL_DIR="$_P5_OUT/eval"

    if [[ ! -f "$_MERGE_PTH" ]]; then
        echo "  [wait] Merge not complete. Run eval after merge finishes:"
        echo "    bash $0 --step 25 --model-type $MODEL_TYPE --p5-mode $P5_MODE --p5-alpha $P5_ALPHA --eval-only --merge-pth $_MERGE_PTH"
    else
        for _BN in "${_P5_BENCHES[@]}"; do
            _SHORT=$(_bench_short "$_BN")
            _JOB="25_${SHORT_MODEL}_${_MLM_TAG}_${_P5_JOB_TAG}_${_SHORT}${_JUDGE_SUFFIX}"
            _SCORE=$(find "$_EVAL_DIR" -name "*${_BN}*_score.csv" 2>/dev/null | head -1 || true)
            if [[ -n "$_SCORE" ]]; then
                echo "  [skip] $_JOB — score exists"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            if is_job_active "$_JOB"; then
                echo "  [skip] $_JOB — already running"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            # MMStar is multi-choice → exact_matching, not GPT judge
            _BN_JUDGE="$P5_JUDGE"
            [[ "$_BN" == "MMStar" || "$_BN" == "MME" || "$_BN" == "POPE" || "$_BN" == "HallusionBench" ]] && _BN_JUDGE="exact_matching"
            # TriviaQA goes through VLMEvalKit like all other benchmarks.
            _CMD="cd $WORK_DIR && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export OPENAI_API_KEY=$_OPENAI_KEY; $_VE_PYTHON $_VE_DIR/run.py --data $_BN --model $_VE_MODEL --merge_model $_MERGE_PTH --judge $_BN_JUDGE --reuse --work-dir $_EVAL_DIR"
            if $LOCAL; then
                echo "  [local] Eval $_MERGE_TAG on $_BN"
                eval "$_CMD"
            else
                echo "  [submit] $_JOB — $_MERGE_TAG on $_BN"
                _GPU_STR="num=1:gmem=${GPU_GMEM_TIERS[0]}:j_exclusive=yes"
                bsub -J "$_JOB" -q "$QUEUE" -gpu "$_GPU_STR" -R "$GPU_RES_BASE" \
                    -oo "$_P5_LOG/${_JOB}.log" -eo "$_P5_LOG/${_JOB}.err" \
                    "$_CMD"
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
    fi
fi

echo ""

# ── Phase D: Compute BRV-style splits (local, no GPU) ──
_SPLITS_SCRIPT="$WORK_DIR/code/compute_brv_splits.py"
if [[ -f "$_SPLITS_SCRIPT" ]]; then
    # Check if baseline or eval have results
    _HAS_BL=false
    _HAS_EVAL=false
    find "$_BL_OUT" -name "*_score.csv" 2>/dev/null | grep -q . && _HAS_BL=true
    find "$_P5_OUT/eval" -name "*_score.csv" 2>/dev/null | grep -q . && _HAS_EVAL=true

    if $_HAS_BL || $_HAS_EVAL; then
        echo "  ── Computing BRV-style splits ──"
        _SPLITS_ARGS="--model-name $MODEL_NAME --output $_P5_ROOT_OUT/brv_splits_summary.json"
        if $_HAS_BL; then
            _SPLITS_ARGS="$_SPLITS_ARGS --eval-dir $_BL_OUT"
        fi
        if $_HAS_EVAL; then
            _SPLITS_ARGS="$_SPLITS_ARGS --eval-dir $_P5_OUT/eval"
        fi
        $PYTHON "$_SPLITS_SCRIPT" $_SPLITS_ARGS
        echo ""
    else
        echo "  [skip] No eval results yet — run splits after eval jobs complete"
    fi
fi

echo ""
echo "  Results: $_P5_OUT/"
echo "  Merge:   $_MERGE_PTH"

fi  # end step 25 (weight_merge)
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
echo "  Text-inject dir:    results/13-weight-merge/${MODE_DIR}/$MODEL_NAME/text_inject/"
echo "  Benchmark results:  results/19-evaluate/${MODE_DIR}/$MODEL_NAME/"
echo "  Eval summary:       results/20-summary/${MODE_DIR}/$MODEL_NAME/"
echo "  SNRF (Layer 1b):    results/14-snrf/${MODE_DIR}/$MODEL_NAME/"
echo "  SRF  (Layer 1c):    results/15-srf/${MODE_DIR}/$MODEL_NAME/"
echo "  VIT analysis:       results/11-vit-analysis/${MODE_DIR}/$MODEL_NAME/"
echo "  Composed (1a+1c):   results/18-compose-layer1/${MODE_DIR}/$MODEL_NAME/"
echo "  MathVerse (A):  results/22a-inline-ablation/${MODE_DIR}/$MODEL_NAME/"
echo "  MathVerse (B):  results/22b-vlmevalkit-ablation/${MODE_DIR}/$MODEL_NAME/"
echo "  Hook comparison:   results/23-compare-hooks/${MODE_DIR}/"
echo "  Weight merge:      results/25-merge/$MODEL_NAME/"
echo "═══════════════════════════════════════════════════════════"
exit 0