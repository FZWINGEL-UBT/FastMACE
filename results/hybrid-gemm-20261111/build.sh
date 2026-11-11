#!/usr/bin/env bash
# Builds the hybrid GEMM lab cubins (sm_80 for the A100; sm_120 for a local 5060 Ti smoke): CUTLASS v4.8.1 headers.
#   CUTLASS=~/src/cutlass-4.8.1 bash results/hybrid-gemm-20261111/build.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cutlass=${CUTLASS:-$HOME/src/cutlass-4.8.1}
nvcc=${NVCC:-/usr/local/cuda-13.3/bin/nvcc}
for sm in 80 120; do
  "$nvcc" -cubin -arch=sm_$sm -O3 -std=c++17 -DNDEBUG --expt-relaxed-constexpr -I"$cutlass/include" \
    -o "$here/hybrid_sm$sm.cubin" "$here/hybrid_sm80.cu"
done
"$nvcc" --version | tail -n 2
sha256sum "$here"/hybrid_sm*.cubin
