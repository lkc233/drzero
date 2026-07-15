# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_DEVICES:-4,5,6,7}"

# Keep :8000 alive: it is the local Qwen3.6 judge/updater service.

tp=1
dp=4
gpus=4
sample_size=5
rollout_memory_utilization=0.8

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

challenger_step=100
data_partition=2

EXP_DIR="checkpoints/dr-zero"
MODEL_PATH="challenger_iter2_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
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
