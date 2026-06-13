#!/bin/bash
# Build the BRV "uniform" (non-selective linear-interp) merge of LLaVA-1.5-7b + MetaMath.
# Every weight -> alpha*W_vlm + (1-alpha)*W_math (embeddings/lm_head excluded by merge_pmbt.py).
# This is the non-selective baseline to compare against the PMBT/FT selective merges.
#
# Output: <OUT_DIR>/merged_model_<ALPHA>.pth  (mode=base names by alpha)
set -euo pipefail

REPO_ROOT="/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy"
OUT_DIR="${REPO_ROOT}/results/25-merge/llava-1.5-7b/MetaMath-7B-V1.0/uniform_a0.9"
ALPHA="0.9"
MODEL1="llava-hf/llava-1.5-7b-hf"
MODEL2="meta-math/MetaMath-7B-V1.0"

# Modern venv (same interpreter run_pipeline.sh uses to build the selective merges).
MERGE_PY="${REPO_ROOT}/modern_vlms/.venv/bin/python"

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
# Env hygiene: keep the legacy .venv site-packages from shadowing the modern interpreter.
unset VIRTUAL_ENV
export PYTHONPATH="$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v '\.venv' | paste -sd: -)"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

echo "[merge] uniform base merge, alpha=${ALPHA}"
echo "[merge] out=${OUT_DIR}/merged_model_${ALPHA}.pth"
echo "[merge] started at $(date -Iseconds)"

"${MERGE_PY}" code/merge_pmbt.py \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --output_dir "${OUT_DIR}" \
    --alpha "${ALPHA}" \
    --mode base

echo "[merge] finished at $(date -Iseconds)"
