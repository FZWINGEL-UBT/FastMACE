#!/usr/bin/env bash
# usage: ncu_stall.sh OUT STRUCTURE CONFIG
cd ~/tmp/opt8
source ./env.sh
M=gpu__time_duration.sum
for r in long_scoreboard short_scoreboard mio_throttle math_pipe_throttle wait not_selected selected barrier membar lg_throttle dispatch_stall no_instruction branch_resolving tex_throttle drain sleeping misc imc_miss; do
  M=$M,smsp__warp_issue_stalled_${r}_per_warp_active.pct
done
/usr/local/cuda-13.3/bin/ncu --profile-from-start off --cache-control all --clock-control none -k regex:fused_ \
  --metrics $M $P ncu_fused.py ops-mh1-r2scan-$2.pt "$3" > $1 2>&1
echo done
