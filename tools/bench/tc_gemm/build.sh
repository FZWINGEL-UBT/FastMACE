#!/usr/bin/env bash
# Build libfm_tc_gemm.so (tools/bench/tc_gemm/cutlass_tc.cu, five variants) against a pinned CUTLASS checkout.
#   bash build.sh CUTLASS_DIR OUT_DIR [NVCC]
# SASS for sm_80 (A100) plus sm_86, sm_89 and sm_120, which only the correctness tests use (3xTF32 cannot beat
# SIMT FP32 there: opt-T01 §2). The CUDA runtime is linked statically, so the library needs no toolkit at run time.
set -euo pipefail
CUTLASS=$1 OUT=$2 NVCC=${3:-nvcc}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
python3 "$HERE/gen_variants.py" "$CUTLASS/include" "$OUT/fast_f32_variants.h"
ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
pids=()
for v in 0 1 2 3 4; do
  "$NVCC" $ARCH -O3 -std=c++17 -DNDEBUG -DFM_VARIANT=$v -Xcompiler -fPIC -I"$CUTLASS/include" -I"$OUT" \
    -c "$HERE/cutlass_tc.cu" -o "$OUT/cutlass_tc_v$v.o" > "$OUT/build_v$v.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ $status -eq 0 ] || { tail -30 "$OUT"/build_v*.log; exit 1; }
"$NVCC" $ARCH -shared -o "$OUT/libfm_tc_gemm.so" "$OUT"/cutlass_tc_v*.o
"$NVCC" --version | tail -1 > "$OUT/nvcc.txt"
git -C "$CUTLASS" rev-parse HEAD > "$OUT/cutlass_commit.txt" 2>/dev/null || true
echo "built $OUT/libfm_tc_gemm.so"
