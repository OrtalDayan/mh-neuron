#!/bin/bash
# Submit (arm × benchmark) bsub jobs for all 4 merged arms on the BRV suite.
#
# Usage:
#   bash submit_all.sh              # submit all 4 arms × 11 benchmarks
#   bash submit_all.sh --skip-validated   # skip the gate_pmbt × MathVista_MINI cell
#                                          (already submitted as the validation run)
set -euo pipefail

REPO=/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy
WRAPPER="${REPO}/code/eval_merge_jobs/run_eval_arm.sh"
RESULTS_BASE="${REPO}/results/25-merge/llava-1.5-7b/MetaMath-7B-V1.0"
LOG_BASE="${REPO}/logs/eval_merge"

SKIP_VALIDATED=0
[[ "${1:-}" == "--skip-validated" ]] && SKIP_VALIDATED=1

# Arm short_name → arm directory suffix
declare -A ARMS=(
  [gate_pmbt]="hook_gate_pmbt_t0.9_v1.0_m1.0"
  [gateup_pmbt]="hook_gateup_pmbt_t0.9_v1.0_m1.0"
  [gate_ft]="hook_gate_ft_t0.9_v1.0_m1.0"
)

# Benchmark → judge
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

# Ordered list (for deterministic submission)
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
skipped=0
for arm in "${ARM_ORDER[@]}"; do
  arm_suffix="${ARMS[$arm]}"
  arm_dir="${RESULTS_BASE}/${arm_suffix}"
  pth="${arm_dir}/merged_model_pmbt_t0.9_v1.0_m1.0.pth"
  work="${arm_dir}/eval"

  if [[ ! -f "${pth}" ]]; then
    echo "[skip] missing checkpoint: ${pth}"
    continue
  fi

  for bench in "${BENCHES[@]}"; do
    judge="${JUDGES[$bench]}"
    if [[ $SKIP_VALIDATED -eq 1 && "$arm" == "gate_pmbt" && "$bench" == "MathVista_MINI" ]]; then
      echo "[skip-validated] gate_pmbt × MathVista_MINI"
      skipped=$((skipped+1))
      continue
    fi

    log_dir="${LOG_BASE}/hook_${arm}/${bench}"
    mkdir -p "${log_dir}"

    job_name="eval_${arm}_${bench}"
    # Skip if a same-named job is currently PEND/RUN
    if bjobs -J "${job_name}" 2>/dev/null | grep -qE 'PEND|RUN'; then
      echo "[skip-active] ${job_name}"
      skipped=$((skipped+1))
      continue
    fi

    echo "[submit] ${job_name}"
    bsub -J "${job_name}" \
         -n 1 -R "rusage[mem=65536]" -gpu "num=1:gmem=20G" -q waic-risk \
         -o "${log_dir}/out.%J.log" -e "${log_dir}/err.%J.log" \
         bash "${WRAPPER}" "${bench}" "${pth}" "${judge}" "${work}"
    submitted=$((submitted+1))
  done
done

echo "---"
echo "submitted=${submitted}  skipped=${skipped}"
