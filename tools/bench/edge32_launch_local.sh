#!/usr/bin/env bash
# WSL: edge32_local.sh inside a local_gpu reservation.  bash edge32_launch_local.sh REPEAT
set -u
OUT=$HOME/fm-ada-capacity/edge32/${1:-10x10x10}
mkdir -p "$OUT"
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "int32 edge_index falsifier (off/on, bitwise)" -- \
  bash /mnt/d/Margraf/FastMACE-ada-capacity/tools/bench/edge32_local.sh "$OUT" "${1:-10x10x10}" > "$OUT/launch.log" 2>&1
echo "launch exit=$?" >> "$OUT/launch.log"
