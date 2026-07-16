# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail
set -x
source "$(dirname "${BASH_SOURCE[0]}")/scripts/init_deployment.sh" judge

# --- Environment (ported from drzero_v0: fixes Triton/flashinfer compilation) ---
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# GPUs 0-1 host Qwen3.6 and the retriever; training uses GPUs 2-7.
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_DEVICES:-2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"

source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
# Put the active env's CUDA libs on the linker path (conda first, uv .venv fallback).
export LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
# The venv uses system python3.10 without dev headers; Triton/sglang JIT-compile at
# runtime and need Python.h. Borrow ABI-compatible 3.10 headers from miniforge.
if [ ! -f "/usr/include/python3.10/Python.h" ]; then
    export CPATH="/home/luokc/miniforge3/include/python3.10:${CPATH:-}"
fi

# The local retriever is expected at 127.0.0.1:8020 by default; override it with
# DRZERO_RETRIEVER_URL when the retriever runs on another host.
# Port 8000 belongs to the local Qwen3.6 judge/updater service; keep it alive.
# Port 8001 is only the round-start solver used for reward rollouts.
existing_server_pids="$(lsof -t -i :8001 2>/dev/null || true)"
if [ -n "$existing_server_pids" ]; then kill -9 $existing_server_pids; fi

tp=2
dp=3
gpus=6
batch_per_gpu=2
rollout_memory_utilization=0.25

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

# Log to logs/ like drzero_v0
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/iter1_challenger_ratio${hop_ratio}_$(date +%Y%m%d_%H%M%S).log"

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=Qwen/Qwen3-4B-Instruct-2507
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

# Hydra resolves a relative --config-path against verl/trainer/, so use an absolute path.
CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
VAL_DATA="./data/test.parquet"


# --- Dr.Zero iteration state (frozen skills/rubrics) for the rubric reward ---
# Rubric reward uses the standalone local Qwen3.6 service configured by meta_model.
STATE="./iterations/iter_1/state.json"
export DRZERO_ITERATION_STATE="$STATE"
if [ ! -f "$STATE" ]; then
    python -m verl.iteration.cli init-state \
        --state "$STATE" --iteration 1 --proposer "$model" --solver "$model"
fi

python -m sglang.launch_server \
    --model=${model} \
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
    "http://127.0.0.1:8001/v1/models" "$model" "$SERVER_PID"

echo "Logging to: $LOG_FILE"

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
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
    custom_reward_function.reward_kwargs.model_name=${model} \
    custom_reward_function.reward_kwargs.base_url="http://127.0.0.1:8001" \
    custom_reward_function.reward_kwargs.reward_rollout_n=${reward_group_size} \
    iteration.state_path="$STATE" \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name="challenger_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}" \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 $@ > "$LOG_FILE" 2>&1
