#!/usr/bin/env bash
# WSL: stage 0 on the 5060 Ti inside a local_gpu reservation.
#   bash launch_local0.sh REPEAT
set -u
OUT=$HOME/fm-ada-capacity/arena/stage0-${1:-10x10x10}
mkdir -p "$OUT"
cd /mnt/d/Margraf/FastMACE-ada-capacity
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "capture arena stage 0 (shim, toy, two MD records)" -- \
  bash /mnt/d/Margraf/FastMACE-ada-capacity/tools/bench/arena/local0.sh "$OUT" "${1:-10x10x10}" > "$OUT/launch.log" 2>&1
echo "launch exit=$?" >> "$OUT/launch.log"
