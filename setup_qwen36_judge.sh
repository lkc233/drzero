#!/usr/bin/env bash
#
# Create an isolated uv environment and optionally serve Qwen3.6-35B-A3B as
# the OpenAI-compatible Dr. Zero judge and state updater.
#
# Usage:
#   bash setup_qwen36_judge.sh --no-launch
#   bash setup_qwen36_judge.sh --check
#   bash setup_qwen36_judge.sh
#   GPU_DEVICES=0,1 JUDGE_TP_SIZE=2 bash setup_qwen36_judge.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/environments/qwen36-judge"
JUDGE_VENV_DIR="${JUDGE_VENV_DIR:-$SCRIPT_DIR/.venv-qwen36-judge}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRIPT_DIR/.cache/uv/python}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/huggingface}"
# SGLang warms up through its own localhost HTTP endpoint. Cluster-wide proxy
# variables must not route that request through the compliance gateway.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
JUDGE_HOST="${JUDGE_HOST:-0.0.0.0}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-2}"
JUDGE_CONTEXT_LENGTH="${JUDGE_CONTEXT_LENGTH:-32768}"
JUDGE_MEM_FRACTION="${JUDGE_MEM_FRACTION:-0.80}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
DO_LAUNCH=1
CHECK_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=0 ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ "$CHECK_ONLY" -eq 1 ]; then
    MODELS_URL="http://127.0.0.1:${JUDGE_PORT}/v1/models"
    if ! curl --fail --silent --show-error "$MODELS_URL" | grep -Fq "\"id\":\"$JUDGE_MODEL\""; then
        echo "ERROR: local Qwen3.6 judge/updater is unavailable or does not serve $JUDGE_MODEL at $MODELS_URL" >&2
        exit 1
    fi
    echo "[qwen36-judge] Local judge/updater is ready: $JUDGE_MODEL"
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "[qwen36-judge] Syncing isolated environment at $JUDGE_VENV_DIR"
UV_PROJECT_ENVIRONMENT="$JUDGE_VENV_DIR" uv sync \
    --project "$PROJECT_DIR" \
    --locked \
    --python 3.12

echo "[qwen36-judge] Verifying SGLang installation"
"$JUDGE_VENV_DIR/bin/python" -c 'import sglang; print(f"SGLang {sglang.__version__}")'

if [ "$DO_LAUNCH" -eq 0 ]; then
    echo "[qwen36-judge] Environment ready; launch skipped"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
# SGLang's warmup calls the server through the node's IPv6 address rather than
# 127.0.0.1 on this cluster.  Dependencies and model weights are already local
# after uv sync/model resolution, so keep the serving process fully off proxy.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
echo "[qwen36-judge] Serving $JUDGE_MODEL on $JUDGE_HOST:$JUDGE_PORT"
exec "$JUDGE_VENV_DIR/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --host "$JUDGE_HOST" \
    --port "$JUDGE_PORT" \
    --tp-size "$JUDGE_TP_SIZE" \
    --context-length "$JUDGE_CONTEXT_LENGTH" \
    --mem-fraction-static "$JUDGE_MEM_FRACTION"
