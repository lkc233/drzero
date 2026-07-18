# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -eo pipefail
set -x
source "$(dirname "${BASH_SOURCE[0]}")/scripts/init_deployment.sh" all
source "$(dirname "${BASH_SOURCE[0]}")/scripts/load_run_namespace.sh"

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

# Generation and verify run
# sequentially so each phase can independently use all eight local GPUs. Port 8000
# remains reserved for the local Qwen3.6 judge/updater service.
generation_tp=1
verify_tp=1
verify_dp=8
gpus=8
sample_size=5
rollout_memory_utilization=0.8

VERIFY_SERVER_PID=""

stop_verify_server() {
    if [ -n "$VERIFY_SERVER_PID" ] && kill -0 "$VERIFY_SERVER_PID" 2>/dev/null; then
        kill "$VERIFY_SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! kill -0 "$VERIFY_SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$VERIFY_SERVER_PID" 2>/dev/null; then
            kill -9 "$VERIFY_SERVER_PID" 2>/dev/null || true
        fi
        wait "$VERIFY_SERVER_PID" 2>/dev/null || true
    fi
    VERIFY_SERVER_PID=""
}

clear_verify_port() {
    local pids
    pids="$(lsof -t -i :8001 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null || true
        sleep 1
    fi
}

wait_for_verify_server() {
    local attempt
    for attempt in $(seq 1 120); do
        if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$VERIFY_SERVER_PID" 2>/dev/null; then
            echo "Verify server exited before becoming ready" >&2
            return 1
        fi
        sleep 5
    done
    echo "Timed out waiting for verify server on port 8001" >&2
    return 1
}

trap stop_verify_server EXIT INT TERM
clear_verify_port

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

LOG_DIR="$DRZERO_LOG_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/iteration_1_generate_data_detail.log"

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=Qwen/Qwen3-4B-Instruct-2507
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

# Frozen iteration state produced by iter1_challenger.sh; used for skills injection
# during generation and for verify (round-start solver).
STATE="${DRZERO_ITERATION_ROOT}/iter_1/state.json"
export DRZERO_ITERATION_STATE="$STATE"

challenger_step=50
data_partition=1

EXP_DIR="checkpoints/dr-zero"
MODEL_PATH="challenger_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}${DRZERO_RUN_SUFFIX}"
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

# Merge once to a world-size-independent HF checkpoint so the generation process
# can start and exit independently from the later verify server.
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

COMMON_OVERRIDES=(
    +ckpt_path=null
    data.prompt_key=prompt
    +data.path="$TRAIN_DATA"
    +data.partition="$data_partition"
    +data.batch_size=512
    +data.output_path="$TRAIN_DATA_OUT"
    iteration.state_path="$STATE"
    verify.solver_model.base_url=http://127.0.0.1:8001
    verify.solver_model.model_name="$model"
    actor_rollout_ref.model.path="$MERGED_MODEL_PATH"
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.n="$sample_size"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.prompt_length=2048
    actor_rollout_ref.rollout.response_length=3072
    actor_rollout_ref.rollout.max_num_batched_tokens=65536
    actor_rollout_ref.rollout.tensor_model_parallel_size="$generation_tp"
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_memory_utilization"
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
    trainer.nnodes=1
    trainer.n_gpus_per_node="$gpus"
)

# Phase 1: eight rollout workers generate candidates. main_generation persists the
# candidate snapshot, shuts Ray down, and exits before verify starts.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name=search_multiturn_grpo \
    "${COMMON_OVERRIDES[@]}" \
    "$@" \
    data.phase=generate > "$LOG_FILE" 2>&1

# Phase 2: replicate the 4B verify model once per GPU. This SGLang version
# implements round_robin for its DP controller; shortest_queue is only a stub that
# raises NotImplementedError on the first request.
clear_verify_port
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
    --model="$model" \
    --port=8001 \
    --mem-fraction-static=0.8 \
    --tp-size="$verify_tp" \
    --dp-size="$verify_dp" \
    --load-balance-method=round_robin \
    --log-level=error >> "$LOG_FILE" 2>&1 &
VERIFY_SERVER_PID=$!
wait_for_verify_server

# The verify-only process does not initialize Ray or the proposer checkpoint. It
# restores candidates and runs bounded concurrent group verification.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name=search_multiturn_grpo \
    "${COMMON_OVERRIDES[@]}" \
    "$@" \
    data.phase=verify \
    data.resume_candidates=true >> "$LOG_FILE" 2>&1

stop_verify_server
