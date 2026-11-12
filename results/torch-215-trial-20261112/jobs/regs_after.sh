#!/bin/bash
# w' register-budget A/B (matrix2.sh) after this machine's p11 run (12 November 2026). Own run dir <base>-regs with
# src, jobs and caches linked from the main run; REPS=2 (forward + reversed, twice), then profiles.
#   workstation: nohup bash regs_after.sh ws &      local: bash regs_after.sh local  (inside a local_gpu reservation)
set -u
mode=$1
if [ "$mode" = ws ]; then
  B=/mnt/homes/zwingel/t215; base=$B/runs/$(hostname)
  sleep 60; while pgrep -u "$USER" -f "ws_matrix.sh|p11_after.sh|/jobs/matrix.sh|/jobs/matrix2.sh" > /dev/null; do sleep 30; done
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
  [ -n "$gpu" ] && { echo "GPU busy [$gpu]"; exit 3; }
  export NV=$B/census ENVROOT=/mnt/homes/zwingel/environments MODELS=/mnt/homes/zwingel/ResearchModule/Models \
    STRUCTURES=/mnt/homes/zwingel/ResearchModule
  export VARIANTS=${VARIANTS:-"a:a:-:- r96:a:-:96 r128:a:-:128 n5:a:13.5.37:- n5r128:a:13.5.37:128"}
else
  base=$HOME/fm-t215; export NV=$HOME/fm-t215/census CARD_LIMIT_MIB=14000
  export VARIANTS=${VARIANTS:-"a:a:-:- r0:a:-:0 r128:a:-:128 n5:a:13.5.37:- n5r128:a:13.5.37:128 b0:b:13.0.88:-"}
fi
R=$base-regs; mkdir -p $R
for d in src jobs cache; do [ -e $R/$d ] || ln -s $(readlink -f $base/$d) $R/$d; done
export RUN=$R CUDA_CACHE_DISABLE=1 REPS=2
exec bash $R/jobs/matrix2.sh
