#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/load_deployment_config.sh"
source "$SCRIPT_DIR/scripts/load_run_namespace.sh"

hop_ratio="${HOP_RATIO:-4321}"
rounds="${ROUNDS:-3}"
start_iteration="${START_ITERATION:-1}"
start_stage="${START_STAGE:-challenger}"
mkdir -p "$DRZERO_LOG_DIR"
timing_log="${TRAINING_TIMING_LOG:-$DRZERO_LOG_DIR/training_timing.tsv}"
mkdir -p "$(dirname "$timing_log")"
if [[ ! -e "$timing_log" ]]; then
    printf 'iteration\tstage\tstarted_at\tfinished_at\telapsed_seconds\tstatus\n' > "$timing_log"
fi
export WANDB_MODE="${WANDB_MODE:-offline}"
export TZ="${TRAINING_TIMEZONE:-Asia/Shanghai}"
# Ray's local dashboard/runtime-env agents communicate over loopback and the
# node address.  Bypass the cluster HTTP proxy for those local control-plane
# requests; otherwise actor creation is sent to the compliance gateway.
local_hosts="127.0.0.1,localhost,::1,$(hostname),$(hostname -i | tr ' ' ',')"
export NO_PROXY="${local_hosts},${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

if (( start_iteration < 1 || start_iteration > rounds )); then
    echo "ERROR: START_ITERATION must be between 1 and ROUNDS ($rounds)" >&2
    exit 2
fi

scripts/check_deployment_services.sh all
if [[ "$start_stage" != "challenger" && "$start_stage" != "solver" ]]; then
    echo "ERROR: START_STAGE must be 'challenger' or 'solver'" >&2
    exit 2
fi

judge_stopped_for_solver=false

run_timed_stage() {
    local iteration="$1"
    local stage="$2"
    shift 2
    local started_at started_epoch finished_at finished_epoch elapsed_seconds status stage_log
    started_at="$(date -Is)"
    started_epoch="$(date +%s)"
    echo "[$started_at] Starting iteration $iteration stage $stage"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$iteration" "$stage" "$started_at" "" "0" "RUNNING" \
        >> "$timing_log"
    stage_log="$DRZERO_LOG_DIR/iteration_${iteration}_${stage}.log"
    status=0
    "$@" > >(tee -a "$stage_log") 2>&1 || status=$?
    finished_at="$(date -Is)"
    finished_epoch="$(date +%s)"
    elapsed_seconds=$((finished_epoch - started_epoch))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$iteration" "$stage" "$started_at" "$finished_at" "$elapsed_seconds" "$status" \
        >> "$timing_log"
    echo "[$finished_at] Finished iteration $iteration stage $stage in ${elapsed_seconds}s (status=$status)"
    return "$status"
}

start_local_judge() {
    tmux new-session -d -s qwen36 \
        "cd '$SCRIPT_DIR' && set -o pipefail; GPU_DEVICES=${JUDGE_GPU_DEVICES:-0,1} JUDGE_PORT=${JUDGE_PORT:-8000} JUDGE_MODEL=${DRZERO_META_MODEL} JUDGE_TP_SIZE=${JUDGE_TP_SIZE:-2} bash setup_qwen36_judge.sh 2>&1 | tee -a '$DRZERO_LOG_DIR/judge.log'"
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
    judge_pids="$(lsof -t -i ":${JUDGE_PORT:-8000}" 2>/dev/null || true)"
    if [[ -n "$judge_pids" ]]; then
        kill $judge_pids 2>/dev/null || true
    fi
    for _ in $(seq 1 30); do
        if ! lsof -t -i ":${JUDGE_PORT:-8000}" >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done
    judge_pids="$(lsof -t -i ":${JUDGE_PORT:-8000}" 2>/dev/null || true)"
    if [[ -n "$judge_pids" ]]; then
        kill -9 $judge_pids 2>/dev/null || true
    fi
    if lsof -t -i ":${JUDGE_PORT:-8000}" >/dev/null 2>&1; then
        echo "ERROR: Qwen3.6 did not release port ${JUDGE_PORT:-8000}" >&2
        return 1
    fi
}

restore_local_judge() {
    if [[ "$judge_stopped_for_solver" == "true" ]]; then
        start_local_judge
    fi
}

for iteration in $(seq "$start_iteration" "$rounds"); do
    echo "[$(date -Is)] Starting iteration $iteration/$rounds"
    if [[ "$iteration" != "$start_iteration" || "$start_stage" == "challenger" ]]; then
        run_timed_stage "$iteration" challenger bash "iter${iteration}_challenger.sh" "$hop_ratio"
        run_timed_stage "$iteration" data_generation bash "iter${iteration}_gen_data.sh" "$hop_ratio"
    else
        echo "[$(date -Is)] Resuming iteration $iteration at Solver"
    fi
    trap restore_local_judge EXIT
    stop_local_judge_for_solver
    run_timed_stage "$iteration" solver bash "iter${iteration}_solver.sh" "$hop_ratio"
    run_timed_stage "$iteration" convert bash convert.sh "$iteration" "$hop_ratio"
    run_timed_stage "$iteration" full_test bash evaluate_solver.sh "$iteration" "$hop_ratio"
    restore_local_judge
    trap - EXIT
    # Training, generation/verify, and update are intentionally serial. Solver
    # and full-test evaluation temporarily own all eight GPUs; the other stages
    # use GPUs 2-7.
    run_timed_stage "$iteration" update_state bash update_iteration_state.sh "$iteration" "$hop_ratio"
    echo "[$(date -Is)] Completed iteration $iteration/$rounds"
    start_stage=challenger
done
