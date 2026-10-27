#!/bin/bash
# Count SASS opcodes of the forward kernels (FP64 pipe vs FP32 pipe vs conversions), sm_120 cubins from render_check.py.
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
for f in native_f64.cubin dd_forward_cvt.cubin dd_forward_bits.cubin; do
  fn=oeq_projected_forward; [ "$f" != native_f64.cubin ] && fn=oeq_projected_forward_dd
  echo "== $f ($fn)"
  $CUOBJDUMP -sass -fun $fn "$f" | grep -oE '^\s+/\*[0-9a-f]+\*/\s+(@!?U?P[0-9T] )?[A-Z0-9]+(\.[A-Z0-9_]+)*' \
    | awk '{print $NF}' | sed -E 's/^@!?U?P[0-9T]//' | awk -F. '{op=$1; if (op=="F2F"||op=="I2F"||op=="F2I") op=$0; c[op]++} END {for (k in c) print c[k], k}' \
    | sort -rn | grep -E ' (D[A-Z]+|F[A-Z0-9]+|I2F.*|F2F.*|MUFU.*)$' | head -20
done
