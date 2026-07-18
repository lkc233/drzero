# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail
set -x
source "$(dirname "${BASH_SOURCE[0]}")/scripts/init_deployment.sh" judge
source "$(dirname "${BASH_SOURCE[0]}")/scripts/load_run_namespace.sh"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_DEVICES:-2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Keep :8000 alive: it is the local Qwen3.6 judge/updater service.
existing_server_pids="$(lsof -t -i :8001 2>/dev/null || true)"
if [ -n "$existing_server_pids" ]; then kill -9 $existing_server_pids; fi

cur_iter=3
prev_iter=2

tp=1
dp=6
gpus=6
batch_per_gpu=2
rollout_memory_utilization=0.25

solver_algorithm=grpo
solver_grpo_group_size=5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=Qwen/Qwen3-4B-Instruct-2507
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

CONFIG_PATH="./config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
VAL_DATA="./data/test.parquet"

SOLVER_NAME="solver_iter${prev_iter}_hf"
CHALLENGER_NAME="challenger_iter${cur_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}${DRZERO_RUN_SUFFIX}"
SOLVER_PATH="./checkpoints/dr-zero/solver_iter${prev_iter}_ratio${hop_ratio}_${solver_algorithm}_group${solver_grpo_group_size}_${model_name}${DRZERO_RUN_SUFFIX}/${SOLVER_NAME}"
RESUME_PATH="./checkpoints/dr-zero/challenger_iter${prev_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}${DRZERO_RUN_SUFFIX}/global_step_100"
STATE="${DRZERO_ITERATION_ROOT}/iter_${cur_iter}/state.json"
export DRZERO_ITERATION_STATE="$STATE"


source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"

if [ ! -f "$STATE" ]; then
    python -m verl.iteration.cli init-state \
        --state "$STATE" --iteration "$cur_iter" --proposer "$model" --solver "$SOLVER_NAME"
fi

python -m sglang.launch_server \
    --model=$SOLVER_PATH \
    --served-model-name=$SOLVER_NAME \
    --port=8001 \
    --tool-call-parser=qwen25 \
    --mem-fraction-static=${rollout_memory_utilization} \
    --dp-size=${dp} \
    --tp-size=${tp} \
    --log-level=error &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
bash "$(dirname "${BASH_SOURCE[0]}")/scripts/wait_for_model_server.sh" \
    "http://127.0.0.1:8001/v1/models" "$SOLVER_NAME" "$SERVER_PID"

python -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name="search_multiturn_grpo" \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA  \
    data.train_batch_size=240 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=240 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${grpo_group_size} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    reward_model.reward_manager=batch \
    custom_reward_function.name=compute_challenger_score_batch \
    custom_reward_function.path=verl/custom_reward/reward_function.py \
    custom_reward_function.reward_kwargs.model_name=$SOLVER_NAME \
    custom_reward_function.reward_kwargs.base_url="http://127.0.0.1:8001" \
    custom_reward_function.reward_kwargs.reward_rollout_n=${reward_group_size} \
    iteration.state_path="$STATE" \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$CHALLENGER_NAME \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path=$RESUME_PATH \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 $@
