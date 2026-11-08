#!/usr/bin/env bash
# leto: after the new-default scan, the old/new peak-memory comparison (Benchmark_loup and the old maxima).
J=/mnt/homes/zwingel/fm-ada-capacity
while pgrep -f "[j]obs3/chain-leto7" > /dev/null; do sleep 60; done
QUIET=2 bash $J/jobs/queue.sh true
ROOT=$J/runs/memcmp-ada bash $J/jobs3/memcmp.sh $J/src-1cd369e4 SrTiO3:10x10x10 SrTiO3:24x24x24 NaCl:25x25x25
