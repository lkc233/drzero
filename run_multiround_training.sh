#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/load_deployment_config.sh"

hop_ratio="${HOP_RATIO:-4321}"
rounds="${ROUNDS:-3}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Ray's local dashboard/runtime-env agents communicate over loopback and the
# node address.  Bypass the cluster HTTP proxy for those local control-plane
# requests; otherwise actor creation is sent to the compliance gateway.
local_hosts="127.0.0.1,localhost,::1,$(hostname),$(hostname -i | tr ' ' ',')"
export NO_PROXY="${local_hosts},${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

scripts/check_deployment_services.sh all

judge_stopped_for_solver=false

start_local_judge() {
    tmux new-session -d -s qwen36 \
        "cd '$SCRIPT_DIR' && set -o pipefail; GPU_DEVICES=${JUDGE_GPU_DEVICES:-0,1} JUDGE_PORT=${JUDGE_PORT:-8000} JUDGE_MODEL=${DRZERO_META_MODEL} JUDGE_TP_SIZE=${JUDGE_TP_SIZE:-2} bash setup_qwen36_judge.sh 2>&1 | tee -a logs/qwen36.log"
    scripts/check_deployment_services.sh judge
    judge_stopped_for_solver=false
}

stop_local_judge_for_solver() {
    if [[ "$MANAGE_JUDGE" != "true" ]]; then
        return
    fi
    if ! tmux has-session -t qwen36 2>/dev/null; then
        echo "ERROR: managed Qwen3.6 tmux session is missing; cannot safely release GPUs for Solver" >&2
        return 1
    fi
    tmux kill-session -t qwen36
    judge_stopped_for_solver=true
    for _ in $(seq 1 30); do
        if ! lsof -t -i ":${JUDGE_PORT:-8000}" >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done
    echo "ERROR: Qwen3.6 did not release port ${JUDGE_PORT:-8000} within 30 seconds" >&2
    return 1
}

restore_local_judge() {
    if [[ "$judge_stopped_for_solver" == "true" ]]; then
        start_local_judge
    fi
}

for iteration in $(seq 1 "$rounds"); do
    echo "[$(date -Is)] Starting iteration $iteration/$rounds"
    bash "iter${iteration}_challenger.sh" "$hop_ratio"
    bash "iter${iteration}_gen_data.sh" "$hop_ratio"
    trap restore_local_judge EXIT
    stop_local_judge_for_solver
    bash "iter${iteration}_solver.sh" "$hop_ratio"
    restore_local_judge
    trap - EXIT
    bash convert.sh "$iteration" "$hop_ratio"
    # Training, generation/verify, and update are intentionally serial. Solver
    # temporarily owns all eight GPUs; the other stages use GPUs 2-7.
    bash update_iteration_state.sh "$iteration" "$hop_ratio"
    echo "[$(date -Is)] Completed iteration $iteration/$rounds"
done
