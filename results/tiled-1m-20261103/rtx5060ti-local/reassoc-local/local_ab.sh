#!/bin/bash
# Local RTX 5060 Ti: whole-system MH-1 step time, main a7159afd against e2903496 (alternating processes).
OUT=~/fm-runs/reassoc-local; mkdir -p $OUT
ENV=~/environments/mace_cueq_oeq; site=$ENV/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib PYTHONDONTWRITEBYTECODE=1
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule FASTMACE_CACHE=~/fm-cache TORCHINDUCTOR_CACHE_DIR=~/fm-cache/inductor TRITON_CACHE_DIR=~/fm-cache/triton TMPDIR=~/tmp
for p in base-a new-a new-b base-b; do
  case $p in base-*) S=~/fm-src/a7159afd;; new-*) S=~/fm-src/e2903496;; esac
  rm -rf $OUT/$p
  (cd $S && PYTHONPATH=$S $ENV/bin/python tools/ab_step_time.py --sides fastmace --checkpoint ~/models/mace-mh-1.model \
     --structures NaNbO3 NbCl5 --warmup 300 --block 300 --blocks 4 --output $OUT/$p > $OUT/$p.log 2>&1)
  echo "$p exit $?"
done
