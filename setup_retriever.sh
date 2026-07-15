#!/usr/bin/env bash
#
# setup_retriever.sh — One-click Search-R1 retriever setup for a fresh server.
#
# This script covers every stage needed to bring up a local retriever that a
# training job on another machine can call over HTTP:
#   1. Clone the Search-R1 repository
#   2. Create the `retriever` conda environment (torch + faiss-gpu + pyserini ...)
#   3. Download the wiki-18 index + corpus from HuggingFace
#   4. Prepare the data (concatenate index shards, decompress corpus)
#   5. Launch the local retrieval server (FastAPI, port 8020 by default)
#
# It is idempotent: finished stages are detected and skipped on re-run.
#
# Usage:
#   bash setup_retriever.sh                # run all stages, then launch
#   RETRIEVER_TYPE=bm25 bash setup_retriever.sh
#   GPU_DEVICES=0,1 PORT=8020 bash setup_retriever.sh
#   bash setup_retriever.sh --no-launch    # set everything up but do not launch
#   bash setup_retriever.sh --launch-only  # only (re)launch the server
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration (override any of these via environment variables)
# --------------------------------------------------------------------------- #
REPO_URL="${REPO_URL:-https://github.com/PeterGriffinJin/Search-R1.git}"
REPO_DIR="${REPO_DIR:-$HOME/Search-R1}"          # where to clone Search-R1
CONDA_ENV="${CONDA_ENV:-retriever}"              # conda env name
DATA_DIR="${DATA_DIR:-$REPO_DIR/retriever/data}" # where to store index/corpus

# Retriever config: e5_flat (GPU, accurate), e5_hnsw (CPU, fast), or bm25 (CPU, sparse)
RETRIEVER_TYPE="${RETRIEVER_TYPE:-e5_flat}"

# Server config
# NOTE: retrieval_server.py hardcodes host=0.0.0.0 and port=8020. PORT below is
# only used for messages/checks; to actually change it, edit the uvicorn.run(...)
# line at the bottom of search_r1/search/retrieval_server.py.
PORT="${PORT:-8020}"
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
warn() { echo -e "\033[1;33m[setup_retriever][warn]\033[0m $*"; }

# --------------------------------------------------------------------------- #
# Locate conda and define a helper to run commands inside the retriever env
# --------------------------------------------------------------------------- #
init_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        # Try common install locations
        for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
            if [ -f "$c/etc/profile.d/conda.sh" ]; then
                # shellcheck disable=SC1091
                source "$c/etc/profile.d/conda.sh"
                break
            fi
        done
    else
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found. Please install Miniconda/Anaconda first." >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------- #
# Stage 1: Clone Search-R1
# --------------------------------------------------------------------------- #
stage_clone() {
    if [ -d "$REPO_DIR/.git" ] || [ -f "$REPO_DIR/setup.py" ]; then
        log "Repo already present at $REPO_DIR — skipping clone."
    else
        log "Cloning Search-R1 into $REPO_DIR ..."
        git clone "$REPO_URL" "$REPO_DIR"
    fi
}

# --------------------------------------------------------------------------- #
# Stage 2: Create the retriever conda environment
# --------------------------------------------------------------------------- #
stage_env() {
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        log "Conda env '$CONDA_ENV' already exists — skipping creation."
        return
    fi
    log "Creating conda env '$CONDA_ENV' (python=3.10) ..."
    conda create -y -n "$CONDA_ENV" python=3.10

    log "Installing torch + faiss-gpu (via conda) and python deps (via pip) ..."
    conda install -y -n "$CONDA_ENV" \
        pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 \
        -c pytorch -c nvidia
    conda install -y -n "$CONDA_ENV" -c pytorch -c nvidia faiss-gpu=1.8.0
    conda run -n "$CONDA_ENV" pip install \
        transformers datasets pyserini uvicorn fastapi "huggingface_hub[cli]"
}

# --------------------------------------------------------------------------- #
# Stage 3 + 4: Download + prepare index/corpus
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
                log "Downloading e5 flat index + corpus from HuggingFace ..."
                conda run -n "$CONDA_ENV" python "$REPO_DIR/scripts/download.py" --save_path "$DATA_DIR"
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
                conda run -n "$CONDA_ENV" huggingface-cli download \
                    PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir "$DATA_DIR"
                cat "$DATA_DIR"/part_* > "$index_file"
                # HNSW download does not include the corpus; grab it from the e5-index repo
                if [ ! -f "$corpus_file" ] && [ ! -f "$corpus_file.gz" ]; then
                    conda run -n "$CONDA_ENV" huggingface-cli download \
                        PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz --repo-type dataset --local-dir "$DATA_DIR"
                fi
            fi
            ;;
        bm25)
            local index_file="$DATA_DIR/bm25"
            if [ -d "$index_file" ]; then
                log "BM25 index $index_file already exists — skipping download."
            else
                log "Downloading BM25 index + corpus from HuggingFace ..."
                conda run -n "$CONDA_ENV" huggingface-cli download \
                    PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir "$DATA_DIR"
            fi
            ;;
        *)
            echo "ERROR: unknown RETRIEVER_TYPE '$RETRIEVER_TYPE' (use e5_flat, e5_hnsw or bm25)." >&2
            exit 1
            ;;
    esac

    # Decompress corpus if needed
    if [ -f "$corpus_file.gz" ] && [ ! -f "$corpus_file" ]; then
        log "Decompressing corpus $corpus_file.gz ..."
        gzip -d "$corpus_file.gz"
    fi
    if [ ! -f "$corpus_file" ]; then
        warn "Corpus file $corpus_file not found — check the download step."
    fi
}

# --------------------------------------------------------------------------- #
# Stage 5: Launch the retrieval server
# --------------------------------------------------------------------------- #
stage_launch() {
    local server="$REPO_DIR/search_r1/search/retrieval_server.py"
    local corpus_file="$DATA_DIR/wiki-18.jsonl"

    log "Launching retriever ($RETRIEVER_TYPE) on 0.0.0.0:8020 (topk=$TOPK) ..."
    warn "Remote training must point its search URL to: http://<this-server-ip>:8020/retrieve"
    warn "Ensure port 8020 is open in the firewall between the two servers."
    if [ "$PORT" != "8020" ]; then
        warn "PORT=$PORT requested but retrieval_server.py is hardcoded to 8020; edit the uvicorn.run(...) line to change it."
    fi

    case "$RETRIEVER_TYPE" in
        e5_flat)
            export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
            conda run --no-capture-output -n "$CONDA_ENV" python "$server" \
                --index_path "$DATA_DIR/e5_Flat.index" \
                --corpus_path "$corpus_file" \
                --topk "$TOPK" \
                --retriever_name e5 \
                --retriever_model "$RETRIEVER_MODEL" \
                --faiss_gpu
            ;;
        e5_hnsw)
            conda run --no-capture-output -n "$CONDA_ENV" python "$server" \
                --index_path "$DATA_DIR/e5_HNSW64.index" \
                --corpus_path "$corpus_file" \
                --topk "$TOPK" \
                --retriever_name e5 \
                --retriever_model "$RETRIEVER_MODEL"
            ;;
        bm25)
            conda run --no-capture-output -n "$CONDA_ENV" python "$server" \
                --index_path "$DATA_DIR/bm25" \
                --corpus_path "$corpus_file" \
                --topk "$TOPK" \
                --retriever_name bm25
            ;;
    esac
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
init_conda

if [ "$LAUNCH_ONLY" -eq 1 ]; then
    stage_launch
    exit 0
fi

stage_clone
stage_env
stage_data

if [ "$DO_LAUNCH" -eq 1 ]; then
    stage_launch
else
    log "Setup complete. To launch later run: bash $0 --launch-only"
fi
