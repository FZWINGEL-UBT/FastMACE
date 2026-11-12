#!/bin/bash
# Colab H100 (sm_90, the H200's architecture) kernel lab of the H200 w' instruction-tuning variants (12 Nov 2026).
# Run from the source root by tools/colab/fastmace_colab.py ($OUT set). The H100 runs the H200's plan through
# tools/bench/gpu_alias (FASTMACE_GPU_ALIAS), so the captured w' is the H200's source.
ckpt=$FASTMACE_MODELS/mace-omat-0-medium.model
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv > $OUT/gpu.csv
cat $OUT/gpu.csv
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
export PYTHONPATH=$PWD/tools/bench/gpu_alias${PYTHONPATH:+:$PYTHONPATH}
export FASTMACE_GPU_ALIAS="$name=NVIDIA H200"
python tools/omat0_kernel_lab.py --checkpoint $ckpt --output $OUT/lab --kernels wprime --rounds 21 \
  --variants base h2-skip h2-hoist h2-skip+hoist h2-vec2 h2-skip+vec2 h2-vec4 h2-skip+vec4 \
  > $OUT/lab.log 2>&1
echo "lab $?"
grep -E "^wprime|plan|Error|error" $OUT/lab.log | tail -40
python - <<'PY'
import os, re
from pathlib import Path
out = Path(os.environ["OUT"])
mine = Path("results/wprime-h200-20261111/sources/wprime-h200.cu").read_text()
lab = (out / "lab" / "wprime.cu").read_text() if (out / "lab" / "wprime.cu").exists() else ""
strip = lambda t: [l for l in t.splitlines() if l.strip()]
print("captured w' == offline H200 render (blank lines aside):", strip(mine) == strip(lab))
PY
