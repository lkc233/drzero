#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
source .venv/bin/activate
source scripts/load_deployment_config.sh

iteration="${1:?usage: evaluate_solver.sh ITERATION [HOP_RATIO]}"
hop_ratio="${2:-4321}"
model_name="${MODEL_NAME:-qwen3-4b-instruct-2507}"
experiment="solver_iter${iteration}_ratio${hop_ratio}_grpo_group5_${model_name}"
solver_dir="./checkpoints/dr-zero/${experiment}/solver_iter${iteration}_hf"
test_data="${SOLVER_TEST_DATA:-./data/test_sampled.parquet}"
output_dir="${SOLVER_TEST_OUTPUT_DIR:-./data/${experiment}_full_test}"

for required in "$solver_dir" "$test_data"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required Solver evaluation input is missing: $required" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="$SOLVER_GPU_DEVICES"
gpu_count="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
tp_size="${SOLVER_TEST_TP_SIZE:-2}"
if (( gpu_count % tp_size != 0 )); then
    echo "ERROR: Solver evaluation GPU count ($gpu_count) must be divisible by TP size ($tp_size)" >&2
    exit 2
fi

python -m verl.trainer.main_ppo \
    --config-path="$root/config" \
    --config-name=search_multiturn_grpo \
    data.train_files="$test_data" \
    data.val_files="$test_data" \
    data.val_batch_size="${SOLVER_TEST_BATCH_SIZE:-256}" \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="$solver_dir" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$tp_size" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$root/config/search_tool_config.yaml" \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name=dr-zero \
    trainer.experiment_name="${experiment}_full_test" \
    trainer.n_gpus_per_node="$gpu_count" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.validation_data_dir="$output_dir"
