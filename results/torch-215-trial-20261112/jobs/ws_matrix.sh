#!/bin/bash
# matrix.sh on an RTX 4000 Ada workstation (12 November 2026): waits for this trial's earlier job on the host to end,
# then stops if any process is on the GPU (the only thing that blocks a workstation), installs the staged job scripts
# (jobs.new -> jobs; never while a job of ours runs) and runs the CUDA library matrix.  nohup bash ws_matrix.sh &
B=/mnt/homes/zwingel/t215
while pgrep -u "$USER" -f "t215/cublas.sh|t215/jobs/job.sh" > /dev/null; do sleep 20; done
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
[ -n "$gpu" ] && { echo "GPU busy [$gpu]"; exit 3; }
[ -d $B/jobs.new ] && cp $B/jobs.new/* $B/jobs/
export RUN=$B/runs/$(hostname) NV=$B/census ENVROOT=/mnt/homes/zwingel/environments \
  MODELS=/mnt/homes/zwingel/ResearchModule/Models STRUCTURES=/mnt/homes/zwingel/ResearchModule CUDA_CACHE_DISABLE=1
exec bash $B/jobs/matrix.sh
