#!/usr/bin/env bash
# Build and install a FAISS GPU wheel with its FAISS core statically linked.
set -euo pipefail

: "${RETRIEVER_VENV_DIR:?RETRIEVER_VENV_DIR is required}"
: "${FAISS_CUDA_ARCHS:?FAISS_CUDA_ARCHS is required (for example: 90)}"

FAISS_VERSION="${FAISS_VERSION:-1.9.0}"
FAISS_SOURCE_COMMIT="${FAISS_SOURCE_COMMIT:-d243e628880676332263347817b3fe7f474b8b5b}"
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

cuda_major="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p')"
if [ "$cuda_major" != "12" ]; then
    echo "ERROR: FAISS build requires a CUDA 12.x Toolkit (found major version '$cuda_major')." >&2
    exit 1
fi

uv pip install --python "$PYTHON" \
    cmake==4.4.0 ninja==1.13.0 swig==4.4.1 setuptools==83.0.0 wheel==0.47.0

if [ ! -d "$SOURCE_DIR/.git" ] || \
   [ "$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)" != "$FAISS_SOURCE_COMMIT" ]; then
    rm -rf "$SOURCE_DIR"
    git clone --depth 1 --branch "v$FAISS_VERSION" \
        https://github.com/facebookresearch/faiss.git "$SOURCE_DIR"
fi
if [ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$FAISS_SOURCE_COMMIT" ]; then
    echo "ERROR: FAISS v$FAISS_VERSION did not resolve to expected commit $FAISS_SOURCE_COMMIT." >&2
    exit 1
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
if [ -z "$wheel_path" ]; then
    echo "ERROR: FAISS wheel was not produced." >&2
    exit 1
fi
popd >/dev/null

uv pip uninstall --python "$PYTHON" faiss-gpu-cu12 faiss >/dev/null 2>&1 || true
uv pip install --python "$PYTHON" --no-deps --force-reinstall \
    "$BUILD_DIR/faiss/python/$wheel_path"
