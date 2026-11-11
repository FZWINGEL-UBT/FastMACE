#!/usr/bin/env bash
# Offline sm_80 SASS census of the hybrid and 3xTF32 reference entries: registers, spills, tensor and FP32 ops.
here=$(cd "$(dirname "$0")" && pwd)
cuobj=/usr/local/cuda-13.3/bin/cuobjdump
$cuobj -res-usage "$here/hybrid_sm80.cubin" 2>&1 | grep -E "Function|REG" | paste - - | sed -E 's/ +/ /g'
tmp=$(mktemp -d)  # the SASS dumps are scratch
$cuobj -sass "$here/hybrid_sm80.cubin" > "$tmp/all.sass"
for f in fm_hyb_rr_64x64 fm_ref_rr_64x64; do
  awk -v f="$f" '/Function : /{on = ($3 == f)} on' "$tmp/all.sass" > "$tmp/$f.sass"
  printf "%s total %s " "$f" "$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$tmp/$f.sass")"
  for op in HMMA.1688.F32.TF32 HMMA.16816.F32 FFMA FADD FMUL F2FP LOP3 LDS LDSM LDGSTS LDL STL; do
    printf "%s %s " "$op" "$(grep -cE "[[:space:]]$op[ .;]" "$tmp/$f.sass")"
  done
  echo
done
