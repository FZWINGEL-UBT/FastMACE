#!/bin/bash
# Python 3.11 alone (variant p11) against production (a) and Triton 3.9 (t9), 12 November 2026: run after the full
# matrix on this machine ends. Own run dir <base>-p11 (matrix/ inside) with src, jobs and the warm caches linked from
# the main run, so no running script changes. Two forward + reversed passes per structure (REPS=2), then profiles.
#   workstation: nohup bash p11_after.sh ws &      local: bash p11_after.sh local  (inside a local_gpu reservation)
set -u
mode=$1
if [ "$mode" = ws ]; then
  B=/mnt/homes/zwingel/t215; base=$B/runs/$(hostname)
  while pgrep -u "$USER" -f "ws_matrix.sh|t215/jobs/matrix.sh" > /dev/null; do sleep 30; done
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
  [ -n "$gpu" ] && { echo "GPU busy [$gpu]"; exit 3; }
  export NV=$B/census ENVROOT=/mnt/homes/zwingel/environments MODELS=/mnt/homes/zwingel/ResearchModule/Models \
    STRUCTURES=/mnt/homes/zwingel/ResearchModule
else
  base=$HOME/fm-t215; export NV=$HOME/fm-t215/census CARD_LIMIT_MIB=14000
fi
R=$base-p11; mkdir -p $R
for d in src jobs cache; do [ -e $R/$d ] || ln -s $(readlink -f $base/$d) $R/$d; done
export RUN=$R CUDA_CACHE_DISABLE=1 REPS=2 VARIANTS="a:a:- p11:p11:- t9:t9:-"
exec bash $R/jobs/matrix.sh
