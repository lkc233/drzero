#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

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
start_session retriever \
    "set -o pipefail; GPU_DEVICES=0,1 RETRIEVER_TYPE=e5_flat RETRIEVER_PORT=8020 FAISS_USE_GPU=1 bash setup_retriever.sh --launch-only 2>&1 | tee -a logs/retriever.log"
scripts/wait_for_model_server.sh "http://127.0.0.1:8020/docs" "" 0

start_session qwen36 \
    "set -o pipefail; GPU_DEVICES=0,1 JUDGE_TP_SIZE=2 bash setup_qwen36_judge.sh 2>&1 | tee -a logs/qwen36.log"
scripts/wait_for_model_server.sh \
    "http://127.0.0.1:8000/v1/models" "Qwen/Qwen3.6-35B-A3B" 0

start_session training \
    "set -o pipefail; TRAIN_GPU_DEVICES=2,3,4,5,6,7 GENERATION_GPU_DEVICES=2,3,4,5,6,7 bash run_multiround_training.sh 2>&1 | tee -a logs/multiround_training.log"

tmux ls
