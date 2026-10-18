#!/bin/bash
# usage: repro_local.sh <side> <tag> <args...>   (local PC, WSL; fresh compiler caches per run)
W=/mnt/d/Margraf/FastMACE/.claude/worktrees/1a-t13-927920
O=~/fastmace-work/1a-T13b; mkdir -p $O
E=~/environments/mace_cueq_oeq
B=~/fastmace-baseline/1a-T18
side=$1; shift; tag=$1; shift
C=$O/cache-$tag; rm -rf $C; mkdir -p $C/tmp
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=~/models MACE_BENCH_CHECKPOINT=~/models/mace-mh-1.model
export FASTMACE_CACHE=$C/fastmace TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/mace
export XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp
cd $O
if [ "$side" = fastmace ]; then
  PYTHONPATH=$W $E/bin/python $O/repro_bucket.py fastmace $B "$@" > $O/$tag.json 2> $O/$tag.err
elif [ "$side" = ab ]; then
  PYTHONPATH=$W bash $B/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh $E $C/oeq     $E/bin/python $O/repro_ab.py $B "$@" > $O/$tag.json 2> $O/$tag.err
else
  bash $B/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh $E $C/oeq \
    $E/bin/python $O/repro_bucket.py baseline $B "$@" > $O/$tag.json 2> $O/$tag.err
fi
echo "EXIT $?" >> $O/$tag.err
