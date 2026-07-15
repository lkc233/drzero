#!/usr/bin/env bash
#
# setup_retriever.sh — One-click Dr. Zero local retriever setup for a fresh server.
#
# This script covers every stage needed to bring up a local retriever that a
# training job on another machine can call over HTTP:
#   1. Create a uv-managed virtual environment (torch + faiss-gpu + pyserini ...)
#   2. Download the wiki-18 index + corpus from HuggingFace
#   3. Prepare the data (concatenate index shards, decompress corpus)
#   4. Launch the local retrieval server (FastAPI, port 8000 by default)
#
# It is idempotent: finished stages are detected and skipped on re-run.
#
# Usage:
#   bash setup_retriever.sh                # run all stages, then launch
#   RETRIEVER_TYPE=bm25 bash setup_retriever.sh
#   GPU_DEVICES=0,1 RETRIEVER_PORT=8000 bash setup_retriever.sh
#   bash setup_retriever.sh --no-launch    # set everything up but do not launch
#   bash setup_retriever.sh --launch-only  # only (re)launch the server
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration (override any of these via environment variables)
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVER_VENV_DIR="${RETRIEVER_VENV_DIR:-$SCRIPT_DIR/.venv-retriever}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/retriever}"

# Keep tool/model caches out of the ephemeral home directory.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRIPT_DIR/.cache/uv/python}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/huggingface}"

# Retriever config: e5_flat (GPU, accurate), e5_hnsw (CPU, fast), or bm25 (CPU, sparse)
RETRIEVER_TYPE="${RETRIEVER_TYPE:-e5_flat}"
FAISS_USE_GPU="${FAISS_USE_GPU:-auto}"          # auto, 1 (force GPU), or 0 (CPU index)

# Server config
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
TOPK="${TOPK:-3}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"                # used by e5_flat only
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"

# Stage control flags
DO_LAUNCH=1
LAUNCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-launch)   DO_LAUNCH=0 ;;
        --launch-only) LAUNCH_ONLY=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^#//'
            exit 0 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

log() { echo -e "\n\033[1;32m[setup_retriever]\033[0m $*"; }
info() { echo -e "\033[1;34m[setup_retriever]\033[0m $*"; }

faiss_gpu_supported() {
    "$RETRIEVER_VENV_DIR/bin/python" - <<'PY'
import sys
import faiss
import numpy as np
import torch

if not torch.cuda.is_available():
    sys.exit(1)

# Exercise the same float16, all-GPU cloning path used by DenseRetriever.
vectors = np.arange(4096, dtype=np.float32).reshape(32, 128)
index = faiss.IndexFlatL2(128)
index.add(vectors)
options = faiss.GpuMultipleClonerOptions()
options.useFloat16 = True
options.shard = True
gpu_index = faiss.index_cpu_to_all_gpus(index, co=options)
_, neighbors = gpu_index.search(vectors[:1], 1)
sys.exit(0 if neighbors[0, 0] == 0 else 1)
PY
}

faiss_cuda_archs() {
    "$RETRIEVER_VENV_DIR/bin/python" - <<'PY'
import torch

architectures = sorted({
    major * 10 + minor
    for major, minor in (
        torch.cuda.get_device_capability(device)
        for device in range(torch.cuda.device_count())
    )
})
print(";".join(map(str, architectures)))
PY
}

# --------------------------------------------------------------------------- #
# Locate uv
# --------------------------------------------------------------------------- #
init_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/ first." >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------- #
# Stage 1: Create the retriever virtual environment and install dependencies
# --------------------------------------------------------------------------- #
stage_env() {
    if [ -x "$RETRIEVER_VENV_DIR/bin/python" ]; then
        log "Retriever virtual environment already exists at $RETRIEVER_VENV_DIR."
    else
        log "Creating isolated retriever environment at $RETRIEVER_VENV_DIR (python=3.10) ..."
        uv venv --python 3.10 "$RETRIEVER_VENV_DIR"
    fi

    log "Installing retriever dependencies via uv (PyTorch 2.4.0, CUDA 12.1) ..."
    uv pip install --python "$RETRIEVER_VENV_DIR/bin/python" \
        --index https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        transformers datasets pyserini uvicorn fastapi huggingface_hub

    local cuda_archs
    cuda_archs="$(faiss_cuda_archs)"
    if [[ ";$cuda_archs;" == *";90;"* ]]; then
        if faiss_gpu_supported 2>/dev/null; then
            log "Existing FAISS build already passes the GPU kernel test."
        else
            log "Building FAISS GPU from source for CUDA architectures $cuda_archs ..."
            RETRIEVER_VENV_DIR="$RETRIEVER_VENV_DIR" \
            FAISS_CUDA_ARCHS="$cuda_archs" \
            FAISS_BUILD_CACHE_DIR="$SCRIPT_DIR/.cache/faiss-build" \
                bash "$SCRIPT_DIR/scripts/build_faiss_gpu.sh"
        fi
    else
        uv pip install --python "$RETRIEVER_VENV_DIR/bin/python" \
            --index https://download.pytorch.org/whl/cu121 \
            torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
            faiss-gpu-cu12==1.9.0.post1
    fi

    log "Verifying FAISS installation ..."
    "$RETRIEVER_VENV_DIR/bin/python" -c 'import faiss; print(f"FAISS {faiss.__version__} loaded successfully")'
}

# --------------------------------------------------------------------------- #
# Stage 2 + 3: Download + prepare index/corpus
# --------------------------------------------------------------------------- #
stage_data() {
    mkdir -p "$DATA_DIR"

    local corpus_file="$DATA_DIR/wiki-18.jsonl"

    case "$RETRIEVER_TYPE" in
        e5_flat)
            local index_file="$DATA_DIR/e5_Flat.index"
            if [ -f "$index_file" ]; then
                log "Index $index_file already exists — skipping download."
            else
                log "Downloading e5 flat index from HuggingFace ..."
                "$RETRIEVER_VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/download.py" --save_path "$DATA_DIR"
                log "Concatenating index shards -> $index_file ..."
                cat "$DATA_DIR"/part_* > "$index_file"
            fi
            ;;
        e5_hnsw)
            local index_file="$DATA_DIR/e5_HNSW64.index"
            if [ -f "$index_file" ]; then
                log "Index $index_file already exists — skipping download."
            else
                log "Downloading e5 HNSW64 index from HuggingFace ..."
                "$RETRIEVER_VENV_DIR/bin/hf" download \
                    PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir "$DATA_DIR"
                cat "$DATA_DIR"/part_* > "$index_file"
            fi
            ;;
        bm25)
            local index_file="$DATA_DIR/bm25"
            if [ -d "$index_file" ]; then
                log "BM25 index $index_file already exists — skipping download."
            else
                log "Downloading BM25 index + corpus from HuggingFace ..."
                "$RETRIEVER_VENV_DIR/bin/hf" download \
                    PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir "$DATA_DIR"
            fi
            ;;
        *)
            echo "ERROR: unknown RETRIEVER_TYPE '$RETRIEVER_TYPE' (use e5_flat, e5_hnsw or bm25)." >&2
            exit 1
            ;;
    esac

    if [ ! -f "$corpus_file" ] && [ ! -f "$corpus_file.gz" ]; then
        log "Downloading wiki-18 corpus from HuggingFace ..."
        "$RETRIEVER_VENV_DIR/bin/hf" download \
            PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz --repo-type dataset --local-dir "$DATA_DIR"
    fi

    # Decompress corpus if needed
    if [ -f "$corpus_file.gz" ] && [ ! -f "$corpus_file" ]; then
        log "Decompressing corpus $corpus_file.gz ..."
        gzip -d "$corpus_file.gz"
    fi
    if [ ! -f "$corpus_file" ]; then
        echo "ERROR: corpus file $corpus_file not found after download." >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------- #
# Stage 4: Launch the retrieval server
# --------------------------------------------------------------------------- #
stage_launch() {
    local server="$SCRIPT_DIR/search/retrieval_server.py"
    local corpus_file="$DATA_DIR/wiki-18.jsonl"

    log "Launching retriever ($RETRIEVER_TYPE) on 0.0.0.0:$RETRIEVER_PORT (topk=$TOPK) ..."
    info "Remote training must point its search URL to: http://<this-server-ip>:$RETRIEVER_PORT/retrieve"
    info "Ensure port $RETRIEVER_PORT is open in the firewall between the two servers."

    case "$RETRIEVER_TYPE" in
        e5_flat)
            export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
            local faiss_gpu_args=()
            case "$FAISS_USE_GPU" in
                1|true)
                    faiss_gpu_args=(--faiss_gpu)
                    ;;
                0|false)
                    info "Using the FAISS index on CPU (FAISS_USE_GPU=$FAISS_USE_GPU)."
                    ;;
                auto)
                    if faiss_gpu_supported; then
                        faiss_gpu_args=(--faiss_gpu)
                    else
                        info "FAISS GPU kernels do not support this GPU; using the index on CPU."
                    fi
                    ;;
                *)
                    echo "ERROR: FAISS_USE_GPU must be auto, 1, or 0 (got '$FAISS_USE_GPU')." >&2
                    exit 1
                    ;;
            esac
            "$RETRIEVER_VENV_DIR/bin/python" "$server" \
                --index_path "$DATA_DIR/e5_Flat.index" \
                --corpus_path "$corpus_file" \
                --topk "$TOPK" \
                --port "$RETRIEVER_PORT" \
                --retriever_name e5 \
                --retriever_model "$RETRIEVER_MODEL" \
                "${faiss_gpu_args[@]}"
            ;;
        e5_hnsw)
            "$RETRIEVER_VENV_DIR/bin/python" "$server" \
                --index_path "$DATA_DIR/e5_HNSW64.index" \
                --corpus_path "$corpus_file" \
                --topk "$TOPK" \
                --port "$RETRIEVER_PORT" \
                --retriever_name e5 \
                --retriever_model "$RETRIEVER_MODEL"
            ;;
        bm25)
            "$RETRIEVER_VENV_DIR/bin/python" "$server" \
                --index_path "$DATA_DIR/bm25" \
                --corpus_path "$corpus_file" \
                --topk "$TOPK" \
                --port "$RETRIEVER_PORT" \
                --retriever_name bm25
            ;;
    esac
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
init_uv

if [ "$LAUNCH_ONLY" -eq 1 ]; then
    stage_launch
    exit 0
fi

stage_env
stage_data

if [ "$DO_LAUNCH" -eq 1 ]; then
    stage_launch
else
    log "Setup complete. To launch later run: bash $0 --launch-only"
fi
