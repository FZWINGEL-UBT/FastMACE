#!/bin/bash
# 6-T13: verdict lines of the trajectory comparisons.
cd ~/tmp/6-T13/traj
for f in *-compare.log; do
  echo "== $f"
  grep -E '"passed"|"bitwise"|"seed|true|false' "$f" | tr -s ' ' | head -8
done
