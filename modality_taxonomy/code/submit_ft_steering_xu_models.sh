#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# FT-vs-PMBT steering comparison on Xu's 3 models for ECCV 2026 rebuttal.
# Submits 3 jobs (one per model) using --label_source xu (Xu et al. FT
# fixed-threshold labels) at the same matched top-N count as the existing
# PMBT steering jobs (5% of total neurons), alpha=0.5, target=visual.
#
# Each job mirrors the existing PMBT steering submission for the same
# model but points --labels_dir at the FT directory and sets
# --label_source xu.
#
# Run from $WORK_DIR = /home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail
WORK_DIR=/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
cd "$WORK_DIR"

# Common steering args (mirroring the existing PMBT jobs)
ALPHA=0.5
TARGET=visual
RANKING=d
QUEUE=waic-long
GMEM=80G
MEM_MB=98304

# Per-model configuration: model_type, model_name, top_n, venv
declare -A CFG=(
    [llava15]="liuhaotian|liuhaotian/llava-v1.5-7b|17613|.venv|llava-1.5-7b"
    [llavaov]="llava-ov|lmms-lab/llava-onevision-qwen2-7b-ov|26521|modern_vlms/.venv|llava-onevision-7b"
    [internvl]="internvl|OpenGVLab/InternVL2_5-8B|22938|intervl_env/.venv_internvl|internvl2.5-8b"
)

# Tags map model→ short jobname suffix
declare -A TAG=(
    [llava15]="l"
    [llavaov]="lo"
    [internvl]="int"
)

for KEY in llava15 llavaov internvl; do
    IFS='|' read -r MTYPE MNAME TOPN VENV MDIR <<< "${CFG[$KEY]}"
    JOBNAME="11_${TAG[$KEY]}_5pct_ft"
    LABELS_DIR="results/3-classify/full/$MDIR/llm_fixed_threshold"
    OUT_DIR="results/11-steering/full/$MDIR/d_ft"
    LOG_DIR="logs/full/$MDIR/11-steering"
    mkdir -p "$LOG_DIR" "$OUT_DIR"

    # Sanity: verify labels_dir exists with at least one layer dir
    if ! ls "$LABELS_DIR"/*layers.0* >/dev/null 2>&1; then
        echo "  ⚠ $KEY: no layer dirs in $LABELS_DIR — skipping"
        continue
    fi

    echo "→ Submitting $JOBNAME ($MDIR, FT, top_n=$TOPN)"

    bsub -q $QUEUE -J "$JOBNAME" \
         -gpu "num=1:gmem=$GMEM" \
         -R "rusage[mem=$MEM_MB]" \
         -oo "$LOG_DIR/$JOBNAME.log" \
         -eo "$LOG_DIR/$JOBNAME.err" \
         "cd $WORK_DIR && $VENV/bin/python code/neuron_ablation_validate.py \
            --model_type $MTYPE \
            --model_name $MNAME \
            --labels_dir $LABELS_DIR \
            --label_source xu \
            --output_dir $OUT_DIR \
            --ranking_method $RANKING \
            --ablation_top_n $TOPN \
            --alpha $ALPHA \
            --target_neurons $TARGET"
done

echo ""
echo "Submitted. Track with:  bjobs -w | grep _5pct_ft"
