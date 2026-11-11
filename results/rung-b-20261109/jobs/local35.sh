#!/usr/bin/env bash
# WSL: SrTiO3 35^3 on the 5060 Ti on cd56800c (src3, already unpacked), under a local_gpu reservation.
set -u
S=$HOME/fm-ada-capacity/rungb/src3
R=$HOME/fm-ada-capacity/rungb/points3b
mkdir -p "$R"
cp /mnt/d/Margraf/FastMACE-ada-capacity/results/rung-b-20261109/jobs/points_local.sh "$R/"
cd /mnt/d/Margraf/FastMACE-ada-capacity
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "rung-b item 2: 5060 Ti SrTiO3 35^3 on cd56800c" -- \
  env FM="$S" bash "$R/points_local.sh" "$R" SrTiO3:35x35x35 > "$R/launch.log" 2>&1
echo "LOCAL35 exit=$?" >> "$R/run.log"
