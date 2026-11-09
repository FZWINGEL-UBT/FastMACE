#!/bin/bash
# tiled-speed job 1: baseline profile of main d5de5bf7 on the local RTX 5060 Ti
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB:-job1}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
B="python tools/bench/tiled_capacity.py --model omat0-medium --structure SrTiO3 --tile-list --species-index --mode auto"
run() { name=$1; shift; echo "== $name $(date +%T)"; $B "$@" --output $OUT/$name > $OUT/$name.log 2>&1; echo "exit=$? $(date +%T)"; tail -c 600 $OUT/$name.log; echo; }
run ws29k --repeat 18x18x18 --calls 3 --sweep-times --compare compiled
run dev1m --repeat 58x58x58 --calls 3 --sweep-times
run host1m --repeat 58x58x58 --calls 3 --sweep-times --host-state --alias-state --wavefront off
run wave1m --repeat 58x58x58 --calls 3 --sweep-times --host-state --alias-state --wavefront on
run prof1m --repeat 58x58x58 --calls 3 --sweep-profile
echo done
