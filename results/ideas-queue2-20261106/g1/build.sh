#!/bin/bash
# G1 ceiling: tf32x3 cubins with B's split removed (B's raw bits in both fragments; wrong outputs), built from
# fastmace/kernels/cutlass/tf32x3_sm80.cu with a patched copy of mma_tensor_op_fast_f32.h before CUTLASS's include.
set -euo pipefail
CUT=$HOME/src/cutlass-4.8.1
SRC=/mnt/d/Margraf/fm-queue/fastmace/kernels/cutlass
W=$HOME/fm-runs/g1-build
rm -rf "$W"; mkdir -p "$W/patch/cutlass/gemm/warp"
H=cutlass/gemm/warp/mma_tensor_op_fast_f32.h
cp "$CUT/include/$H" "$W/patch/$H"
python3 - "$W/patch/$H" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "      convert_B(B, ptr_dst_B[0], ptr_dst_B[1]);\n"
assert s.count(old) == 1, s.count(old)
new = ("      (void)convert_B;  // G1 ceiling: no B split (B's raw bits as both parts; wrong outputs)\n"
       "      ptr_dst_B[0] = *reinterpret_cast<Array<typename ArchMmaOperator::ElementB, FragmentB::kElements> const *>(&B);\n"
       "      ptr_dst_B[1] = ptr_dst_B[0];\n")
open(p, "w").write(s.replace(old, new))
EOF
NVCC=/usr/local/cuda-13.3/bin/nvcc
for sm in 80 90; do
  "$NVCC" -cubin -arch=sm_$sm -O3 -std=c++17 -DNDEBUG --expt-relaxed-constexpr -I"$CUT/include" \
    -o "$W/plain_sm$sm.cubin" "$SRC/tf32x3_sm80.cu"
  "$NVCC" -cubin -arch=sm_$sm -O3 -std=c++17 -DNDEBUG --expt-relaxed-constexpr -I"$W/patch" -I"$CUT/include" \
    -o "$W/ceiling_sm$sm.cubin" "$SRC/tf32x3_sm80.cu"
done
sha256sum "$W"/*.cubin "$SRC"/tf32x3_sm80.cubin "$SRC"/tf32x3_sm90.cubin
for f in plain_sm90 ceiling_sm90; do
  echo "$f: $(/usr/local/cuda-13.3/bin/cuobjdump -sass "$W/$f.cubin" | grep -A100000 'fm_tf32x3_rr_64x64' | grep -cE 'FADD|LOP3') FADD+LOP3 (rr_64x64 onward)"
done
