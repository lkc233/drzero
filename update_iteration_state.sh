#!/usr/bin/env bash
set -euo pipefail

iteration="${1:?usage: update_iteration_state.sh ITERATION [HOP_RATIO]}"
hop_ratio="${2:-4321}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
source .venv/bin/activate
source scripts/load_deployment_config.sh

model_name="${MODEL_NAME:-qwen3-4b-instruct-2507}"
solver_name="solver_iter${iteration}_hf"
solver_dir="./checkpoints/dr-zero/solver_iter${iteration}_ratio${hop_ratio}_grpo_group5_${model_name}/${solver_name}"
challenger_name="challenger_iter${iteration}_ratio${hop_ratio}_grpo_batch_group1-5_${model_name}"
data_prefix="./data/zero_${challenger_name}"
state="./iterations/iter_${iteration}/state.json"
next_state="./iterations/iter_$((iteration + 1))/state.json"
artifact_dir="./iterations/iter_${iteration}/update"
mkdir -p "$artifact_dir"

candidates="${data_prefix}.candidates.jsonl"
keepout="${data_prefix}_keepout.parquet"
generation_summary="${data_prefix}_generation_summary.json"
keepout_results="${artifact_dir}/keepout_results.jsonl"
keepout_summary="${artifact_dir}/keepout_summary.json"
analysis="${artifact_dir}/trajectory_analysis.json"
skills="${artifact_dir}/skills.json"
rubrics="${artifact_dir}/rubrics.json"

for required in "$state" "$solver_dir" "$candidates" "$keepout" "$generation_summary"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: required iteration artifact is missing: $required" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_DEVICES:-2,3,4,5,6,7}"
export DRZERO_ITERATION_STATE="$state"
export DRZERO_SOLVER_AFTER="$solver_name"
export DRZERO_META_MODEL="${DRZERO_META_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export DRZERO_UPDATER_MODEL="${DRZERO_UPDATER_MODEL:-$DRZERO_META_MODEL}"

python -m verl.iteration.cli record-solver-after --state "$state" --solver "$solver_name"

old_server_pids="$(lsof -t -i :8001 2>/dev/null || true)"
if [ -n "$old_server_pids" ]; then kill -9 $old_server_pids; fi
python -m sglang.launch_server \
    --model="$solver_dir" \
    --served-model-name="$solver_name" \
    --port=8001 \
    --tool-call-parser=qwen25 \
    --mem-fraction-static="${UPDATE_SOLVER_MEM_FRACTION:-0.8}" \
    --dp-size=3 \
    --tp-size=2 \
    --log-level=error >"$artifact_dir/solver_server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
scripts/wait_for_model_server.sh "http://127.0.0.1:8001/v1/models" "$solver_name" "$server_pid"
scripts/check_deployment_services.sh all

python -m verl.iteration.cli keepout-eval \
    --state "$state" --config config/search_multiturn_grpo.yaml \
    --keepout "$keepout" --output "$keepout_results" --summary "$keepout_summary"
python -m verl.iteration.cli analyze \
    --state "$state" --config config/search_multiturn_grpo.yaml \
    --results "$keepout_results" --output "$analysis"
python -m verl.iteration.cli update-skills \
    --state "$state" --config config/search_multiturn_grpo.yaml \
    --candidates "$candidates" --analysis "$analysis" \
    --generation-summary "$generation_summary" --keepout-summary "$keepout_summary" \
    --output "$skills"
python -m verl.iteration.cli update-rubrics \
    --state "$state" --config config/search_multiturn_grpo.yaml \
    --candidates "$candidates" --analysis "$analysis" \
    --generation-summary "$generation_summary" --keepout-summary "$keepout_summary" \
    --skills "$skills" --output "$rubrics"
python -m verl.iteration.cli advance-state \
    --state "$state" --next-state "$next_state" \
    --skills "$skills" --rubrics "$rubrics" \
    --proposer "$challenger_name" --solver "$solver_name"

echo "Iteration $iteration state updated: $next_state"
