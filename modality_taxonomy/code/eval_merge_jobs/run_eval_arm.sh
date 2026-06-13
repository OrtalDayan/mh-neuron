#!/bin/bash
# Per-arm × benchmark VLMEvalKit launcher.
#
# Args:
#   $1 = BENCHMARK   (e.g. MathVista_MINI)
#   $2 = MERGE_PTH   (absolute path to merged_model_*.pth, or "NONE" for the
#                     unmerged baseline — runs stock LLaVA-1.5 with no --merge_model)
#   $3 = JUDGE       (gpt-4o-mini | exact_matching)
#   $4 = WORK_DIR    (absolute path; per-arm "<arm>/eval")
set -euo pipefail

BENCH="$1"
MERGE_PTH="$2"
JUDGE="$3"
WORK_DIR="$4"

REPO_ROOT="/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy"
VE_PY="${REPO_ROOT}/modern_vlms/VLMEvalKit_brv/.venv/bin/python"
RUN_PY="${REPO_ROOT}/modern_vlms/VLMEvalKit_brv/run.py"
ENV_FILE="${REPO_ROOT}/modern_vlms/VLMEvalKit/.env"

# Source the OpenAI key inside the job — never inline it (would land in bjobs/logs).
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC2046
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
else
    echo "[warn] missing ${ENV_FILE}; OPENAI_API_KEY not set" >&2
fi

mkdir -p "${WORK_DIR}"

# Same env-hygiene as run_pipeline.sh when invoking the modern interpreter.
unset VIRTUAL_ENV
export PYTHONPATH="$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v '\.venv' | paste -sd: -)"

cd "${REPO_ROOT}/modern_vlms/VLMEvalKit_brv"

echo "[eval] bench=${BENCH} judge=${JUDGE}"
echo "[eval] merge=${MERGE_PTH}"
echo "[eval] work-dir=${WORK_DIR}"
echo "[eval] started at $(date -Iseconds)"

# Baseline arm: no --merge_model → evaluate stock LLaVA-1.5 as the no-merge reference.
MERGE_ARGS=(--merge_model "${MERGE_PTH}")
if [[ "${MERGE_PTH}" == "NONE" || "${MERGE_PTH}" == "none" || "${MERGE_PTH}" == "baseline" ]]; then
    MERGE_ARGS=()
    echo "[eval] no-merge baseline: running stock LLaVA-1.5 (no --merge_model)"
fi

"${VE_PY}" "${RUN_PY}" \
    --data "${BENCH}" \
    --model llava_v1.5_7b_hf \
    ${MERGE_ARGS[@]+"${MERGE_ARGS[@]}"} \
    --judge "${JUDGE}" \
    --reuse \
    --work-dir "${WORK_DIR}"

echo "[eval] finished at $(date -Iseconds)"
