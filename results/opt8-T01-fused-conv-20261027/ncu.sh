#!/usr/bin/env bash
# usage: ncu.sh OUT STRUCTURE CONFIG
cd ~/tmp/opt8
source ./env.sh
/usr/local/cuda-13.3/bin/ncu --profile-from-start off --cache-control all --clock-control none -k regex:fused_ \
  --section SpeedOfLight --section Occupancy --section WarpStateStats --section ComputeWorkloadAnalysis \
  --section InstructionStats --section MemoryWorkloadAnalysis \
  $P ncu_fused.py ops-mh1-r2scan-$2.pt "$3" > $1 2>&1
echo done
