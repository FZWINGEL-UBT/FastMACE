#!/usr/bin/env bash
# Builds tools/bench/idea33/libidea33_wgmma.so (idea 33 timing bound; a lab library, shipped in the branch because
# the Modal job image has no nvcc). Needs nvcc 13.x and CUTLASS v4.8.1 (commit 2c85f497, the checkout that built
# fastmace/kernels/cutlass/tf32x3_sm*.cubin):
#   CUTLASS=~/src/cutlass-4.8.1 NVCC=/usr/local/cuda-13.3/bin/nvcc bash tools/bench/idea33/build.sh
# The CUDA runtime is linked statically; libstdc++ and libgcc too, so the library loads on the Modal image's
# Debian glibc (the script prints the GLIBC symbol versions it needs).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cutlass=${CUTLASS:?set CUTLASS to a CUTLASS v4.8.1 checkout}
nvcc=${NVCC:-nvcc}
"$nvcc" -shared -Xcompiler -fPIC -arch=sm_90a -O3 -std=c++17 -DNDEBUG --expt-relaxed-constexpr \
  -I"$cutlass/include" -I"$cutlass/tools/util/include" -cudart static \
  -Xcompiler -static-libstdc++,-static-libgcc \
  -o "$here/libidea33_wgmma.so" "$here/wgmma_tf32_gemm.cu"
{
  echo "nvcc: $("$nvcc" --version | tail -n 2 | tr '\n' ' ')"
  echo "cutlass: $(git -C "$cutlass" rev-parse HEAD 2>/dev/null || echo unknown) ($(git -C "$cutlass" describe --tags 2>/dev/null || echo v4.8.1))"
  echo "arch: sm_90a"
  echo "glibc symbols: $(objdump -T "$here/libidea33_wgmma.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tr '\n' ' ')"
  echo "sha256: $(sha256sum "$here/libidea33_wgmma.so" | cut -d' ' -f1)"
} > "$here/build.txt"
cat "$here/build.txt"
