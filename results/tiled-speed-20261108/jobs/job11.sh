#!/bin/bash
# tiled-speed job 11: repeat of the after-profiles (job 8's host-state call was 12.9 s, scatter1 4.1 s: contention?)
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
B="python tools/bench/tiled_capacity.py --model omat0-medium --structure SrTiO3 --tile-list --species-index --mode auto"
run() { name=$1; shift; echo "== $name $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"; $B "$@" --output $OUT/$name > $OUT/$name.log 2>&1; echo "exit=$? $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"; }
run dev1m --repeat 58x58x58 --calls 4 --sweep-times
run host1m --repeat 58x58x58 --calls 4 --sweep-times --host-state --alias-state --wavefront off
run wave1m --repeat 58x58x58 --calls 4 --sweep-times --host-state --alias-state --wavefront on
echo done
