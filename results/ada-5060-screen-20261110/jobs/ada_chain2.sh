#!/usr/bin/env bash
# Second Ada job (murphy, after murphy-A2): the POLAR-1 megakernel cells, then SPLIT_COLUMNS on OFF23-large (and
# OFF24 as a no-split control). Each behind queue.sh's quiet check; it waits for murphy-A2 to finish first.
B=/mnt/homes/zwingel/fm-ada5060
until grep -qE "DONE|STOP" $B/runs/murphy-A2/run.log 2>/dev/null; do sleep 120; done
mkdir -p $B/runs/murphy-polar $B/runs/murphy-off
bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/murphy-polar bash $B/jobs/polar.sh $B/src-7d18028d mk-S mk-M mk-L
bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/murphy-off bash $B/jobs/off.sh $B/src-7d18028d off23-large:MACE-OFF23_large.model off24-medium:MACE-OFF24_medium.model
