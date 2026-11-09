#!/usr/bin/env bash
# Holds leto's claim (runs/ada-claim, which tiled-speed's suite queue watches) while chain-ada2 can still run on leto:
# released at once if murphy claims chain-ada2, else when the memory comparison is done. Claims kept as *.done.
R=/mnt/homes/zwingel/fm-ada-capacity/runs
until [ -f $R/ada-claim2/host ]; do sleep 60; done
if [ "$(cat $R/ada-claim2/host)" = leto ]; then
  until [ -d $R/memcmp-ada.stopped ] && grep -q DONE $R/memcmp-ada/run.log 2>/dev/null; do sleep 60; done
fi
sleep 5
mv $R/ada-claim $R/ada-claim.done; echo "$(date -u +%FT%TZ) released ada-claim (chain-ada2 on $(cat $R/ada-claim2/host))"
until grep -q DONE $R/memcmp-ada/run.log 2>/dev/null && [ -d $R/memcmp-ada.stopped ]; do sleep 60; done
mv $R/ada-claim2 $R/ada-claim2.done; echo "$(date -u +%FT%TZ) released ada-claim2"
