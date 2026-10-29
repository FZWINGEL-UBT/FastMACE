#!/usr/bin/env bash
# The fp32 ideas' microbenchmarks (tools/bench/fp32_ideas_micro.py) into $OUT (default .): every mode plainly,
# then the GEMM shapes again with cuBLAS's BF16x9 FP32 emulation requested (idea 6), each in its own process.
set -u
OUT=${OUT:-.}
PY=${PY:-python}
here=$(dirname "$0")
$PY "$here/fp32_ideas_micro.py" --mode bw chain gemmchunk strassen emu > "$OUT/micro.jsonl" 2> "$OUT/micro.err"
echo "plain exit $?"
CUBLAS_EMULATE_SINGLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager \
  $PY "$here/fp32_ideas_micro.py" --mode emu > "$OUT/micro-emu.jsonl" 2> "$OUT/micro-emu.err"
echo "emulated exit $?"
