#!/bin/bash
# The trial's job.sh on an RTX 4000 Ada workstation: shared-home environments and models (read only), run dir under
# /mnt/homes/zwingel/t215/runs/<host>. Stops if the GPU has a process.  bash ws_run.sh [STAGE]
B=/mnt/homes/zwingel/t215
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
[ -n "$gpu" ] && { echo "GPU busy [$gpu]"; exit 3; }
export RUN=$B/runs/$(hostname) ENVROOT=/mnt/homes/zwingel/environments MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  STRUCTURES=/mnt/homes/zwingel/ResearchModule PROFILE_C=0 STAGE=${1:-all} CUDA_CACHE_DISABLE=1
mkdir -p $RUN && rm -rf $RUN/src && ln -s $B/src-b7c19b19 $RUN/src && rm -rf $RUN/jobs && ln -s $B/jobs $RUN/jobs
exec bash $B/jobs/job.sh
