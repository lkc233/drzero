# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail
set -x
source "$(dirname "${BASH_SOURCE[0]}")/scripts/init_deployment.sh" all

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_DEVICES:-2,3,4,5,6,7}"

# Keep :8000 alive: it is the local Qwen3.6 judge/updater service.

tp=1
dp=6
gpus=6
sample_size=5
rollout_memory_utilization=0.8

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=Qwen/Qwen3-4B-Instruct-2507
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

cur_iter=3
prev_iter=2
solver_algorithm=grpo
solver_grpo_group_size=5
SOLVER_NAME="solver_iter${prev_iter}_hf"
SOLVER_PATH="./checkpoints/dr-zero/solver_iter${prev_iter}_ratio${hop_ratio}_${solver_algorithm}_group${solver_grpo_group_size}_${model_name}/${SOLVER_NAME}"
STATE="./iterations/iter_${cur_iter}/state.json"
export DRZERO_ITERATION_STATE="$STATE"

challenger_step=150
data_partition=3

EXP_DIR="checkpoints/dr-zero"
MODEL_PATH="challenger_iter3_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
CKPT_PATH="${EXP_DIR}/${MODEL_PATH}/global_step_${challenger_step}"

CONFIG_PATH="./config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
TRAIN_DATA_OUT="./data/zero_${MODEL_PATH}.parquet"


source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"

python -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    +ckpt_path=$CKPT_PATH \
    data.prompt_key=prompt \
    +data.path=$TRAIN_DATA \
    +data.partition=$data_partition \
    +data.batch_size=512 \
    +data.output_path=$TRAIN_DATA_OUT \
    iteration.state_path="$STATE" \
    verify.solver_model.base_url="http://127.0.0.1:8001" \
    verify.solver_model.model_name="$SOLVER_NAME" \
    verify.local_server.enabled=true \
    verify.local_server.model_path="$SOLVER_PATH" \
    verify.local_server.gpu_devices="${VERIFY_GPU_DEVICE:-2}" \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${sample_size} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=1536 \
    actor_rollout_ref.rollout.response_length=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${gpus} $@
