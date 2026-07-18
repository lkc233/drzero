#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/load_run_namespace.sh"

iteration="${1:-1}"
hop_ratio="${2:-4321}"
model_name="${MODEL_NAME:-qwen3-4b-instruct-2507}"
experiment="solver_iter${iteration}_ratio${hop_ratio}_grpo_group5_${model_name}${DRZERO_RUN_SUFFIX}"
experiment_dir="./checkpoints/dr-zero/${experiment}"

checkpoint_dir="$(find "$experiment_dir" -maxdepth 1 -type d -name 'global_step_*' \
    | sort -V | tail -n 1)"
if [ -z "$checkpoint_dir" ]; then
    echo "ERROR: no solver checkpoint found under $experiment_dir" >&2
    exit 1
fi

target_dir="${experiment_dir}/solver_iter${iteration}_hf"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${checkpoint_dir}/actor" \
    --target_dir "$target_dir"

echo "Converted $checkpoint_dir -> $target_dir"
