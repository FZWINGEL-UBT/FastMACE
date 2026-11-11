#!/usr/bin/env bash
# WSL: edge32_local.sh inside a local_gpu reservation.
#   bash edge32_launch_local.sh REPEAT [rep]   (rep: the reproducible-plan sides only, into REPEAT-rep)
set -u
REPEAT=${1:-10x10x10}
ONLY=0; SUFFIX=""
[ "${2:-}" = rep ] && { ONLY=1; SUFFIX=-rep; }
OUT=$HOME/fm-ada-capacity/edge32/$REPEAT$SUFFIX
mkdir -p "$OUT"
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "int32 edge_index falsifier (off/on, bitwise)" -- \
  env ONLY_REPRODUCIBLE=$ONLY bash /mnt/d/Margraf/FastMACE-ada-capacity/tools/bench/edge32_local.sh "$OUT" "$REPEAT" > "$OUT/launch.log" 2>&1
echo "launch exit=$?" >> "$OUT/launch.log"
