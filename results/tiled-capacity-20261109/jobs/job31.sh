#!/bin/bash
# tiled capacity job 31: census at 1.25M and 5M atoms (f16 state, no kept lists): what of the host memory scales with N
# wavefront on, host state aliased: float32 store with kept lists, float16 store without kept lists
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
prof() { name=$1; shift
  echo "== hostprof-$name $(date +%T) $(free -g | awk '/Mem/{print $7}') GiB available"
  python tools/bench/tiled_host_profile.py --structure SrTiO3 --model omat0-medium --repeat ${REP}x${REP}x${REP} --mode auto \
    --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --ledger --tag=-$name "$@" \
    --output $OUT/hostprof-$name > $OUT/hostprof-$name.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/hostprof-$name.log | cut -c1-1500; }

export HOSTPROF_CENSUS=1
REP=63 prof r63 --state-dtype float16 --list-reuse off
REP=100 prof r100 --state-dtype float16 --list-reuse off
echo done
