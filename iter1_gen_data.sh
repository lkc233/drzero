# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x
source "$(dirname "${BASH_SOURCE[0]}")/scripts/init_deployment.sh" all

# --- Environment (ported from drzero_v0: fixes Triton/flashinfer compilation) ---
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export HYDRA_FULL_ERROR=1

source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
export LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
# The venv uses system python3.10 without dev headers; Triton/sglang JIT-compile at
# runtime and need Python.h. Borrow ABI-compatible 3.10 headers from miniforge.
if [ ! -f "/usr/include/python3.10/Python.h" ]; then
    export CPATH="/home/luokc/miniforge3/include/python3.10:${CPATH}"
fi

# Retriever uses 127.0.0.1:8020 by default. Port 8001 serves the round-start
# solver; port 8000 remains the local Qwen3.6 judge/updater service.
kill -9 $(lsof -t -i :8001) 2>/dev/null;

tp=1
dp=6
gpus=6            # GPUs 2-7 are all used for generation; verify starts afterward.
sample_size=5
rollout_memory_utilization=0.8

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/iter1_gen_data_ratio${hop_ratio}_$(date +%Y%m%d_%H%M%S).log"

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=Qwen/Qwen3-4B-Instruct-2507
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

# Frozen iteration state produced by iter1_challenger.sh; used for skills injection
# during generation and for verify (round-start solver).
STATE="./iterations/iter_1/state.json"
export DRZERO_ITERATION_STATE="$STATE"

challenger_step=50
data_partition=1

EXP_DIR="checkpoints/dr-zero"
MODEL_PATH="challenger_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
CKPT_PATH="${EXP_DIR}/${MODEL_PATH}/global_step_${challenger_step}"
MERGED_MODEL_PATH="${CKPT_PATH}/merged_hf"
MERGE_COMPLETE="${MERGED_MODEL_PATH}/.merge_complete"
SOURCE_FINGERPRINT="$(
    find "${CKPT_PATH}/actor" -maxdepth 1 -type f \
        \( -name 'model_world_size_*.pt' -o -name 'fsdp_config.json' \) \
        -printf '%f:%s:%T@\n' |
        sort |
        sha256sum |
        cut -d' ' -f1
)"
CACHED_FINGERPRINT="$(sed -n '1p' "$MERGE_COMPLETE" 2>/dev/null)"

# Hydra resolves a relative --config-path against verl/trainer/, so use an absolute path.
CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
TRAIN_DATA_OUT="./data/zero_${MODEL_PATH}.parquet"

# Generation uses six workers while the proposer checkpoint was saved with eight
# FSDP shards. Merge once to a world-size-independent HF checkpoint so the rollout
# workers initialize from the trained proposer without attempting an incompatible
# 8-way -> 6-way sharded checkpoint load.
if [ "$CACHED_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
    if ! python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${CKPT_PATH}/actor" \
        --target_dir "$MERGED_MODEL_PATH"; then
        echo "Failed to merge proposer checkpoint: $CKPT_PATH" >&2
        exit 1
    fi
    printf '%s\n' "$SOURCE_FINGERPRINT" > "$MERGE_COMPLETE"
fi

echo "Logging to: $LOG_FILE"

CUDA_VISIBLE_DEVICES="${GENERATION_GPU_DEVICES:-2,3,4,5,6,7}" python -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    +ckpt_path=null \
    data.prompt_key=prompt \
    +data.path=$TRAIN_DATA \
    +data.partition=$data_partition \
    +data.batch_size=512 \
    +data.output_path=$TRAIN_DATA_OUT \
    iteration.state_path="$STATE" \
    verify.solver_model.base_url="http://127.0.0.1:8001" \
    verify.solver_model.model_name=${model} \
    verify.local_server.enabled=true \
    verify.local_server.model_path=${model} \
    verify.local_server.gpu_devices="${VERIFY_GPU_DEVICE:-2}" \
    actor_rollout_ref.model.path=$MERGED_MODEL_PATH \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${sample_size} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${gpus} $@ > "$LOG_FILE" 2>&1
