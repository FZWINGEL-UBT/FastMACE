#!/usr/bin/env bash
# WSL: pin 1522aef2 (the two-size probe; tar made on Windows) to ~/fm-ada-capacity/rungb/src4 and check the 5060 Ti
# limits again under a local_gpu reservation: SrTiO3 35^3, 36^3 and NaCl 33^3, 34^3.
#   bash launch_local4.sh TAR
set -u
S=$HOME/fm-ada-capacity/rungb/src4
R=$HOME/fm-ada-capacity/rungb/points4
rm -rf "$S"; mkdir -p "$S" "$R"
tar -x -C "$S" -f "$1"
echo 1522aef2 > "$S/.source_commit"
cp /mnt/d/Margraf/FastMACE-ada-capacity/results/rung-b-20261109/jobs/points_local.sh "$R/"
cd /mnt/d/Margraf/FastMACE-ada-capacity
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "rung-b item 2: 5060 Ti limits on 1522aef2" -- \
  env FM="$S" bash "$R/points_local.sh" "$R" SrTiO3:35x35x35,36x36x36 NaCl:33x33x33,34x34x34 > "$R/launch.log" 2>&1
echo "LOCAL4 exit=$?" >> "$R/run.log"
