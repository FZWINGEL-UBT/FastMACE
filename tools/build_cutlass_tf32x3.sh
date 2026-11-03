#!/usr/bin/env bash
# Builds fastmace/kernels/cutlass/tf32x3_sm80.cubin from tf32x3_sm80.cu (offline; FastMACE ships the cubin).
# Needs nvcc (CUDA 13; the local PC's WSL has /usr/local/cuda-13.3) and CUTLASS v4.8.1's headers
# (git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git, commit 2c85f497):
#   CUTLASS=~/src/cutlass-4.8.1 bash tools/build_cutlass_tf32x3.sh
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
src=$here/fastmace/kernels/cutlass
cutlass=${CUTLASS:?set CUTLASS to a CUTLASS v4.8.1 checkout}
nvcc=${NVCC:-nvcc}
"$nvcc" -cubin -arch=sm_80 -O3 -std=c++17 -DNDEBUG --expt-relaxed-constexpr -I"$cutlass/include" \
  -o "$src/tf32x3_sm80.cubin" "$src/tf32x3_sm80.cu"
{
  echo "nvcc: $("$nvcc" --version | tail -n 2 | tr '\n' ' ')"
  echo "cutlass: $(git -C "$cutlass" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "sha256: $(sha256sum "$src/tf32x3_sm80.cubin" | cut -d' ' -f1)"
} > "$src/tf32x3_sm80.build.txt"
cat "$src/tf32x3_sm80.build.txt"
