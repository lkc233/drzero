#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
# shellcheck disable=SC1091
source "$root/scripts/load_deployment_config.sh"

sync_tmux_environment() {
    tmux start-server
    local name value secret_name
    local names=(
        DRZERO_DEPLOY_CONFIG DRZERO_RESOLVED_DEPLOY_CONFIG
        MANAGE_RETRIEVER MANAGE_JUDGE
        DRZERO_RETRIEVER_URL
        DRZERO_META_BASE_URL DRZERO_META_MODEL DRZERO_META_API_KEY_ENV
        DRZERO_UPDATER_BASE_URL DRZERO_UPDATER_MODEL DRZERO_UPDATER_API_KEY_ENV
        TRAIN_GPU_DEVICES GENERATION_GPU_DEVICES
        RETRIEVER_GPU_DEVICES RETRIEVER_TYPE RETRIEVER_PORT FAISS_USE_GPU
        JUDGE_GPU_DEVICES JUDGE_PORT JUDGE_TP_SIZE WANDB_MODE
    )
    for name in "${names[@]}"; do
        if [[ -n "${!name:-}" ]]; then
            tmux set-environment -g "$name" "${!name}"
        else
            tmux set-environment -gu "$name" 2>/dev/null || true
        fi
    done
    for name in DRZERO_META_API_KEY_ENV DRZERO_UPDATER_API_KEY_ENV; do
        secret_name="${!name:-}"
        if [[ -n "$secret_name" && -n "${!secret_name:-}" ]]; then
            value="${!secret_name}"
            tmux set-environment -g "$secret_name" "$value"
        fi
    done
}

sync_tmux_environment

start_session() {
    local name="$1"
    shift
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "tmux session '$name' already exists; leaving it running"
        return
    fi
    tmux new-session -d -s "$name" "cd '$root' && bash -lc '$*'"
}

mkdir -p logs
if [[ "$MANAGE_RETRIEVER" == "true" ]]; then
    start_session retriever \
        "set -o pipefail; GPU_DEVICES=${RETRIEVER_GPU_DEVICES:-0,1} RETRIEVER_TYPE=${RETRIEVER_TYPE:-e5_flat} RETRIEVER_PORT=${RETRIEVER_PORT:-8020} FAISS_USE_GPU=${FAISS_USE_GPU:-1} bash setup_retriever.sh --launch-only 2>&1 | tee -a logs/retriever.log"
else
    echo "Retriever is externally managed: $DRZERO_RETRIEVER_URL"
fi
scripts/check_deployment_services.sh retriever

if [[ "$MANAGE_JUDGE" == "true" ]]; then
    start_session qwen36 \
        "set -o pipefail; GPU_DEVICES=${JUDGE_GPU_DEVICES:-0,1} JUDGE_PORT=${JUDGE_PORT:-8000} JUDGE_MODEL=${DRZERO_META_MODEL} JUDGE_TP_SIZE=${JUDGE_TP_SIZE:-2} bash setup_qwen36_judge.sh 2>&1 | tee -a logs/qwen36.log"
else
    echo "Judge/updater is externally managed: $DRZERO_META_BASE_URL"
fi
scripts/check_deployment_services.sh judge

training_config_prefix=""
if [[ -n "${DRZERO_RESOLVED_DEPLOY_CONFIG:-}" ]]; then
    training_config_prefix="DRZERO_DEPLOY_CONFIG='$DRZERO_RESOLVED_DEPLOY_CONFIG' "
fi
start_session training \
    "set -o pipefail; ${training_config_prefix}bash run_multiround_training.sh 2>&1 | tee -a logs/multiround_training.log"

tmux ls
