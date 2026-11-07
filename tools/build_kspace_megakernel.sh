#!/usr/bin/env bash
# Builds the cubins of fastmace/kernels/cuda/kspace_megakernel.cu (offline; FastMACE ships them): sm_80 (A100),
# sm_89 (RTX 4000 Ada), sm_90 (H100, H200), sm_120 (RTX 5060 Ti). Needs nvcc (CUDA 13; the local PC's WSL has
# /usr/local/cuda-13.3). No fast math: the phases use the precise sincosf.
#   bash tools/build_kspace_megakernel.sh
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
src=$here/fastmace/kernels/cuda
nvcc=${NVCC:-nvcc}
for sm in 80 89 90 120; do
  "$nvcc" -cubin -arch=sm_$sm -O3 -std=c++17 -DNDEBUG -Xptxas -v \
    -o "$src/kspace_megakernel_sm$sm.cubin" "$src/kspace_megakernel.cu" 2> "$src/.ptxas_sm$sm.log"
done
{
  echo "nvcc: $("$nvcc" --version | tail -n 2 | tr '\n' ' ')"
  for sm in 80 89 90 120; do
    echo "sha256 sm_$sm: $(sha256sum "$src/kspace_megakernel_sm$sm.cubin" | cut -d' ' -f1)"
  done
  echo "ptxas sm_120 (registers, spills per entry):"
  grep -E "Compiling entry|registers|spill" "$src/.ptxas_sm120.log" | sed 's/^ptxas info *: //'
} > "$src/kspace_megakernel.build.txt"
rm -f "$src"/.ptxas_sm*.log
cat "$src/kspace_megakernel.build.txt"
