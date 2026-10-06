#!/usr/bin/env bash
# build.sh — reproduce the exact llama.cpp build behind the numbers in README.md.
#
# No fork, no patches: this is a clean checkout of upstream ggml-org/llama.cpp at a
# pinned commit. If you want a newer build, re-tune -ngl afterwards — buffer sizes and
# VRAM self-fit change between releases.
#
# Usage:
#   ./build.sh                 # clone/pull into ~/src/llama.cpp and build
#   SRC=/path/to/llama.cpp ./build.sh
set -euo pipefail

# Upstream commit the measurements were taken on. `git describe` reports 0.5.0-dev.
COMMIT=254b177307b14d055bbe4c050c9c441a1e06d8a4
REPO=https://github.com/ggml-org/llama.cpp.git

SRC="${SRC:-$HOME/src/llama.cpp}"
BUILD="${BUILD:-$SRC/build-cuda}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

# --- toolchain checks -------------------------------------------------------
command -v cmake  >/dev/null || { echo "build.sh: cmake not found"  >&2; exit 1; }
command -v nvcc   >/dev/null || command -v /opt/cuda/bin/nvcc >/dev/null || {
  echo "build.sh: nvcc not found — install the CUDA toolkit (12.4+ tested: 13.4)" >&2
  exit 1
}

# --- source -----------------------------------------------------------------
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone "$REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet "$COMMIT"

# --- configure + build ------------------------------------------------------
# Flags below match the CMakeCache of the measured build:
#   CMAKE_BUILD_TYPE=Release   GGML_NATIVE=ON        GGML_CUDA=ON
#   GGML_CUDA_GRAPHS=ON        GGML_CUDA_FA=ON       GGML_OPENMP=ON
#   GGML_CUDA_COMPRESSION_MODE=size                  LLAMA_CURL=ON
#   GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16
#   GGML_BLAS=OFF              (BLAS is not on the decode path for this model)
#
# GGML_NATIVE=ON implies AVX-512 where the CPU has it; no explicit -march needed.
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" \
  -DGGML_CUDA_COMPRESSION_MODE=size \
  -DGGML_OPENMP=ON \
  -DGGML_BLAS=OFF \
  -DLLAMA_CURL=ON

cmake --build "$BUILD" --target llama-server -j "$JOBS"

echo
echo "built: $BUILD/bin/llama-server"
"$BUILD/bin/llama-server" --version || true
echo
echo "next:  MODEL=/path/to/model.gguf LLAMA_SERVER=$BUILD/bin/llama-server ./serve.sh"
