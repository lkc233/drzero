#!/usr/bin/env bash
# Build and install a self-contained FAISS GPU wheel for the requested CUDA architectures.
set -euo pipefail

: "${RETRIEVER_VENV_DIR:?RETRIEVER_VENV_DIR is required}"
: "${FAISS_CUDA_ARCHS:?FAISS_CUDA_ARCHS is required (for example: 90)}"

FAISS_VERSION="${FAISS_VERSION:-1.9.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BUILD_JOBS="${FAISS_BUILD_JOBS:-16}"
CACHE_ROOT="${FAISS_BUILD_CACHE_DIR:-$(pwd)/.cache/faiss-build}"
SOURCE_DIR="$CACHE_ROOT/faiss-$FAISS_VERSION"
ARCH_SLUG="${FAISS_CUDA_ARCHS//;/_}"
BUILD_DIR="$SOURCE_DIR/build-sm$ARCH_SLUG"
PYTHON="$RETRIEVER_VENV_DIR/bin/python"

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: CUDA compiler not found at $CUDA_HOME/bin/nvcc." >&2
    exit 1
fi

uv pip install --python "$PYTHON" cmake ninja swig setuptools wheel

if [ ! -d "$SOURCE_DIR/.git" ]; then
    rm -rf "$SOURCE_DIR"
    git clone --depth 1 --branch "v$FAISS_VERSION" \
        https://github.com/facebookresearch/faiss.git "$SOURCE_DIR"
fi

export PATH="$RETRIEVER_VENV_DIR/bin:$CUDA_HOME/bin:$PATH"
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_C_API=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_OPT_LEVEL=avx2 \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES="$FAISS_CUDA_ARCHS" \
    -DPython_EXECUTABLE="$PYTHON" \
    -DBLA_VENDOR=Generic

cmake --build "$BUILD_DIR" --target swigfaiss -j "$BUILD_JOBS"

pushd "$BUILD_DIR/faiss/python" >/dev/null
rm -rf build dist faiss faiss.egg-info
"$PYTHON" setup.py bdist_wheel
wheel_path="$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"
popd >/dev/null

uv pip uninstall --python "$PYTHON" faiss-gpu-cu12 faiss >/dev/null 2>&1 || true
uv pip install --python "$PYTHON" --no-deps --force-reinstall \
    "$BUILD_DIR/faiss/python/$wheel_path"

