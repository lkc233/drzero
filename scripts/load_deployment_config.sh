#!/usr/bin/env bash
# Source this file from any pipeline entry point. An explicitly selected missing
# profile is an error; the default missing deploy/current.env keeps legacy local
# defaults so existing installations remain runnable.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source scripts/load_deployment_config.sh; do not execute it" >&2
    exit 2
fi

_drzero_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_drzero_default_config="$_drzero_root/deploy/current.env"
_drzero_config="${DRZERO_DEPLOY_CONFIG:-$_drzero_default_config}"
if [[ -f "$_drzero_config" ]]; then
    export DRZERO_RESOLVED_DEPLOY_CONFIG="$(cd -- "$(dirname -- "$_drzero_config")" && pwd)/$(basename -- "$_drzero_config")"
    set -a
    # shellcheck disable=SC1090
    source "$_drzero_config"
    set +a
elif [[ -n "${DRZERO_DEPLOY_CONFIG:-}" ]]; then
    echo "ERROR: deployment config does not exist: $_drzero_config" >&2
    return 1
fi

export MANAGE_RETRIEVER="${MANAGE_RETRIEVER:-true}"
export MANAGE_JUDGE="${MANAGE_JUDGE:-true}"
export DRZERO_RETRIEVER_URL="${DRZERO_RETRIEVER_URL:-http://127.0.0.1:8020/retrieve}"
export DRZERO_META_BASE_URL="${DRZERO_META_BASE_URL:-http://127.0.0.1:8000}"
export DRZERO_META_MODEL="${DRZERO_META_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export DRZERO_UPDATER_BASE_URL="${DRZERO_UPDATER_BASE_URL:-$DRZERO_META_BASE_URL}"
export DRZERO_UPDATER_MODEL="${DRZERO_UPDATER_MODEL:-$DRZERO_META_MODEL}"
export TRAIN_GPU_DEVICES="${TRAIN_GPU_DEVICES:-2,3,4,5,6,7}"
export SOLVER_GPU_DEVICES="${SOLVER_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export GENERATION_GPU_DEVICES="${GENERATION_GPU_DEVICES:-$TRAIN_GPU_DEVICES}"

unset _drzero_root _drzero_default_config _drzero_config
