#!/bin/bash
# usage: probe.sh <tag> <pytest args...>   (one process, -n 0)
ROOT=/mnt/homes/zwingel/fastmace-2-T14; TAG=$1; shift
mkdir -p $ROOT/probe $ROOT/basetemp $ROOT/tmpdir/probe
cd ${SRC:-$ROOT/src}
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${SRC:-$ROOT/src}:$ROOT/plugins \
  FASTMACE_CACHE=$ROOT/cache-warm TMPDIR=$ROOT/tmpdir/probe FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/20261017 \
  nice -n 10 /mnt/homes/zwingel/environments/mace_cueq_latest/bin/python -m pytest -p no:xdist -p no:cacheprovider -p t14_order -rf -q \
  --basetemp=$ROOT/basetemp/probe-$TAG "$@" > $ROOT/probe/$TAG.log 2>&1
echo "$TAG exit $? $(grep -E 'Max abs' $ROOT/probe/$TAG.log | head -1) $(tail -1 $ROOT/probe/$TAG.log)"
