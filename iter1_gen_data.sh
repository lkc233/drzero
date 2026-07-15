# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

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

# Retriever is remote (config/search_tool_config.yaml). Reuse the co-located verify
# server on :8001 (GPU 7); do NOT kill :8000 (standalone judge from serve_vllm.sh).
kill -9 $(lsof -t -i :8001) 2>/dev/null;

tp=1
dp=8
gpus=7            # GPUs 0-6 for generation; GPU 7 reserved for the verify 4B server
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

# Frozen iteration state produced by iter1_challenger.sh; used for skills injection
# during generation and for verify (round-start solver + judge).
STATE="./iterations/iter_1/state.json"
export DRZERO_ITERATION_STATE="$STATE"

challenger_step=50
data_partition=1

EXP_DIR="checkpoints/dr-zero"
MODEL_PATH="challenger_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
CKPT_PATH="${EXP_DIR}/${MODEL_PATH}/global_step_${challenger_step}"

# Hydra resolves a relative --config-path against verl/trainer/, so use an absolute path.
CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
TRAIN_DATA_OUT="./data/zero_${MODEL_PATH}.parquet"


# Serve the untrained Qwen3-4B on :8001 for verify (solver answer samples + judge),
# pinned to GPU 7 so it does not contend with generation on GPUs 0-6.
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
    --model=${model} \
    --port=8001 \
    --tool-call-parser=qwen25 \
    --mem-fraction-static=0.8 \
    --tp-size=1 \
    --log-level=error &

sleep 30

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m verl.trainer.main_generation \
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
    verify.solver_model.model_name=${model} \
    meta_model.base_url="http://127.0.0.1:8001" \
    meta_model.model_name=${model} \
    meta_model.api_key_env=null \
    actor_rollout_ref.model.path=$model \
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
    trainer.n_gpus_per_node=${gpus} $@
