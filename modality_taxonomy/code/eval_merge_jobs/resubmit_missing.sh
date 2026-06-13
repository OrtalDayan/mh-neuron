#!/bin/bash
# Resubmit only the (arm × benchmark) cells whose *_score.csv is missing.
# Idempotent — safe to re-run; skips active jobs and skips already-scored cells.
set -euo pipefail

REPO=/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
WRAPPER="${REPO}/code/eval_merge_jobs/run_eval_arm.sh"
RESULTS_BASE="${REPO}/results/25-merge/llava-1.5-7b/MetaMath-7B-V1.0"
LOG_BASE="${REPO}/logs/eval_merge"

declare -A ARMS=(
  [gate_pmbt]="hook_gate_pmbt_t0.9_v1.0_m1.0"
  [gateup_pmbt]="hook_gateup_pmbt_t0.9_v1.0_m1.0"
  [gate_ft]="hook_gate_ft_t0.9_v1.0_m1.0"
)

declare -A JUDGES=(
  [MathVista_MINI]=gpt-4o-mini
  [MathVerse_MINI_Text_Dominant]=gpt-4o-mini
  [MathVerse_MINI_Text_Lite]=gpt-4o-mini
  [MathVerse_MINI_Vision_Intensive]=gpt-4o-mini
  [MathVerse_MINI_Vision_Dominant]=gpt-4o-mini
  [MathVerse_MINI_Vision_Only]=gpt-4o-mini
  [MMStar]=exact_matching
  [DynaMath]=gpt-4o-mini
  [MathVision_MINI]=gpt-4o-mini
  [MM-Math]=gpt-4o-mini
  [POPE]=exact_matching
)

BENCHES=(
  MathVista_MINI
  MathVerse_MINI_Text_Dominant
  MathVerse_MINI_Text_Lite
  MathVerse_MINI_Vision_Intensive
  MathVerse_MINI_Vision_Dominant
  MathVerse_MINI_Vision_Only
  MMStar
  DynaMath
  MathVision_MINI
  MM-Math
  POPE
)
ARM_ORDER=(gate_pmbt gateup_pmbt gate_ft)

submitted=0
already_done=0
already_active=0
missing_pth=0
for arm in "${ARM_ORDER[@]}"; do
  arm_suffix="${ARMS[$arm]}"
  arm_dir="${RESULTS_BASE}/${arm_suffix}"
  pth="${arm_dir}/merged_model_pmbt_t0.9_v1.0_m1.0.pth"
  work="${arm_dir}/eval"

  if [[ ! -f "${pth}" ]]; then
    echo "[missing-pth] ${pth}"
    missing_pth=$((missing_pth+1))
    continue
  fi

  for bench in "${BENCHES[@]}"; do
    judge="${JUDGES[$bench]}"
    # Skip if a headline score file for this benchmark already exists under the arm's
    # eval dir. Naming varies: most -> *_score*.csv, MMStar -> *_acc.csv,
    # MM-Math -> *_score.json.
    existing=$(find "${work}" \( -name "*${bench}*_score*.csv" -o -name "*${bench}*_acc.csv" -o -name "*${bench}*_score.json" \) 2>/dev/null | head -1)
    if [[ -n "${existing}" ]]; then
      already_done=$((already_done+1))
      continue
    fi

    job_name="eval_${arm}_${bench}"
    if bjobs -J "${job_name}" 2>/dev/null | grep -qE 'PEND|RUN'; then
      echo "[active] ${job_name}"
      already_active=$((already_active+1))
      continue
    fi

    log_dir="${LOG_BASE}/hook_${arm}/${bench}"
    mkdir -p "${log_dir}"

    echo "[submit] ${job_name}"
    bsub -J "${job_name}" \
         -n 1 -R "rusage[mem=65536]" -gpu "num=1:gmem=20G:mode=exclusive_process" -q waic-risk \
         -o "${log_dir}/out.%J.log" -e "${log_dir}/err.%J.log" \
         bash "${WRAPPER}" "${bench}" "${pth}" "${judge}" "${work}" \
         > /dev/null
    submitted=$((submitted+1))
  done
done

echo "---"
echo "submitted=${submitted}  already_done=${already_done}  already_active=${already_active}  missing_pth=${missing_pth}"
