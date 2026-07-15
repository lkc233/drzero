# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

# --- Environment (ported from drzero_v0: fixes Triton/flashinfer compilation) ---
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
export LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
# The venv uses system python3.10 without dev headers; Triton/sglang JIT-compile at
# runtime and need Python.h. Borrow ABI-compatible 3.10 headers from miniforge.
if [ ! -f "/usr/include/python3.10/Python.h" ]; then
    export CPATH="/home/luokc/miniforge3/include/python3.10:${CPATH}"
fi

# Retriever is remote (config/search_tool_config.yaml); no local retrieval server.
kill -9 $(lsof -t -i :8001) 2>/dev/null;

tp=2
dp=4
gpus=8
batch_per_gpu=2
rollout_memory_utilization=0.5

challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo
grpo_group_size=5
model=Qwen/Qwen3-4B-Instruct-2507
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

# Hydra resolves a relative --config-path against verl/trainer/, so use an absolute path.
CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter1_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"

TRAIN_DATA="./data/zero_${CHALLENGER_NAME}.parquet"
VAL_DATA_DIR="./data/${SOLVER_NAME}"
VAL_DATA="./data/test_sampled.parquet"
if [ ! -f "$VAL_DATA" ]; then
    VAL_DATA="./data/test.parquet"
fi

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$SOLVER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=False \
    trainer.validation_data_dir=${VAL_DATA_DIR} \
    trainer.total_epochs=1 $@