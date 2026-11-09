#!/bin/bash
# tiled gap job 18 (455facf2): where the wavefront call's time goes (host method times, GPU busy time)
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
cp /mnt/c/Users/fzwin/AppData/Local/Temp/claude/06a4a8e8fc832e51/1a150e36-92ea-4347-8da9-78c469ccacf9/scratchpad/jobs/wprof.py $OUT/
echo "== wprof $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nproc)"
python $OUT/wprof.py 58x58x58 $OUT/wprof-58.json > $OUT/wprof.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/wprof.log | cut -c1-1500
echo done
