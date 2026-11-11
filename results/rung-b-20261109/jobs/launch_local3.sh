#!/usr/bin/env bash
# WSL: pin cd56800c to ~/fm-ada-capacity/rungb/src3 and scan the 5060 Ti limits under a local_gpu reservation.
set -u
W=/mnt/d/Margraf/FastMACE-ada-capacity
S=$HOME/fm-ada-capacity/rungb/src3
R=$HOME/fm-ada-capacity/rungb/points3
rm -rf "$S"; mkdir -p "$S" "$R"
tar -x -C "$S" -f "$1"  # git archive of cd56800c made on Windows (WSL git cannot read this worktree)
echo cd56800c > "$S/.source_commit"
cp "$W/results/rung-b-20261109/jobs/points_local.sh" "$R/"
cd "$W"
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "rung-b item 2: 5060 Ti capacity scan on cd56800c" -- \
  env FM="$S" bash "$R/points_local.sh" "$R" SrTiO3:36x36x36,37x37x37,38x38x38,39x39x39 NaCl:33x33x33,34x34x34,35x35x35,36x36x36 \
  > "$R/launch.log" 2>&1
echo "LOCAL3 exit=$?" >> "$R/run.log"
