#!/bin/bash
# collect.sh DEST: copy the run records (A/B JSONs, profiles, gate metadata, job scripts) into the results folder
set -eu
DEST=$1
R=/home/fzwin/fm-runs/tiled-speed-20261108
S=/mnt/c/Users/fzwin/AppData/Local/Temp/claude/06a4a8e8fc832e51/1a150e36-92ea-4347-8da9-78c469ccacf9/scratchpad/jobs
mkdir -p $DEST/jobs $DEST/ab $DEST/profiles $DEST/gate
cp $S/job*.sh $S/kprof*.py $S/cmp.py $S/summ.py $DEST/jobs/ 2>/dev/null || true
rm -f $DEST/jobs/launch.sh
for j in $R/job*/; do
  n=$(basename $j)
  [ "$n" = "job3-try1" ] && continue
  if [ -d $j/ab ]; then mkdir -p $DEST/ab/$n; cp $j/ab/*.json $DEST/ab/$n/; fi
  for p in dev1m host1m wave1m ws29k prof1m; do
    if [ -d $j/$p ]; then mkdir -p $DEST/profiles/$n/$p; cp $j/$p/*.json $DEST/profiles/$n/$p/ 2>/dev/null || true; fi
  done
  for k in $j/kprof*.json; do [ -e "$k" ] && mkdir -p $DEST/profiles/$n && cp $k $DEST/profiles/$n/; done
  if [ -e $j/gate/metadata.json ]; then mkdir -p $DEST/gate/$n; cp $j/gate/metadata.json $DEST/gate/$n/; fi
  for g in gputest.log; do [ -e $j/$g ] && mkdir -p $DEST/gputest && cp $j/$g $DEST/gputest/$n-gputest.log; done
done
du -sh $DEST/*
