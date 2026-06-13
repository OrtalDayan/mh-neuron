#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# W3 dissociation experiment: single-point 5% D-ranked steering across 7 VLMs.
#
# For each model, runs 8 conditions × 4 benchmarks at one alpha value with
# top_n = 5% of the LLM's intermediate_size. Output:
#   results/11-steering/full/<model_name>/d/perm/alpha_<a>/ablation_summary.json
#
# Usage:
#   bash submit_w3_steering_all.sh                  # default queue=waic-long
#   QUEUE=waic-risk bash submit_w3_steering_all.sh  # override queue
#   SUFFIX=_5pct_v3 bash submit_w3_steering_all.sh  # override run-suffix
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd /home/projects/bagon/ortalda/mh-neuron/modality_taxonomy

QUEUE="${QUEUE:-waic-long}"           # which LSF queue to submit on
SUFFIX="${SUFFIX:-_5pct}"             # job-name + output-dir suffix (avoids collisions)
SHARDS="${SHARDS:-1}"                 # 1 = single GPU, no sharding (each model = 1 job)

# Per-model: model_type | model_name | model_path | alpha | top_n | shell_run_suffix
# top_n = round(0.05 × LLM_intermediate_size × n_layers)
declare -a JOBS=(
    "llava-liuhaotian|llava-1.5-7b|llava-hf/llava-1.5-7b-hf|0.5|17613|${SUFFIX}"
    "llava-ov|llava-onevision-7b|(default)|0.5|26521|${SUFFIX}"
    "internvl|internvl2.5-8b|modern_vlms/pretrained/InternVL2_5-8B|0.75|22938|${SUFFIX}"
    "qwen2vl|qwen2.5-vl-7b|modern_vlms/pretrained/Qwen2.5-VL-7B-Instruct|0.5|26521|${SUFFIX}_q25"
    "qwen2vl|qwen2-vl-7b|modern_vlms/pretrained/Qwen2-VL-7B-Instruct|0.5|26521|${SUFFIX}_q2"
    "llava-llama3|llava-next-llama3-8b|llava-hf/llama3-llava-next-8b-hf|0.5|22938|${SUFFIX}_v2"
    "idefics2|idefics2-8b|modern_vlms/pretrained/idefics2-8b|0.5|22938|${SUFFIX}"
)

echo "────────────────────────────────────────────────────────────"
echo "  W3 steering submission (queue: $QUEUE, shards: $SHARDS)"
echo "────────────────────────────────────────────────────────────"

for spec in "${JOBS[@]}"; do
    IFS='|' read -r mtype mname mpath alpha topn rsuffix <<< "$spec"

    # Build optional --model-name / --model-path overrides
    extra=""
    [[ "$mname" != "(default)" ]] && extra="$extra --model-name $mname"
    [[ "$mpath" != "(default)" ]] && extra="$extra --model-path $mpath"

    echo ""
    echo "→ $mname  (alpha=$alpha  top_n=$topn  suffix=$rsuffix)"

    bash code/run_pipeline_steering.sh \
        --step steering \
        --model-type "$mtype" \
        $extra \
        --steering-alphas "$alpha" \
        --halluc-score-method d \
        --steering-shards "$SHARDS" \
        --no-ablation-curve \
        --ablation-top-n "$topn" \
        --run-suffix "$rsuffix" \
        --queue "$QUEUE"
done

echo ""
echo "────────────────────────────────────────────────────────────"
echo "  All submitted. Monitor with:"
echo "    bjobs -q $QUEUE -w | grep '_5pct'"
echo "    /tmp/monitor_steering.sh"
echo "────────────────────────────────────────────────────────────"
