#!/usr/bin/env bash
# Rerun of the Ada packs whose first launch failed on the missing SrTiO3.cif (10 Nov, 16:14 UTC):
#   bash ada_rerun.sh WAIT_RUN NAME PACK...   (waits for runs/WAIT_RUN to log DONE or STOP, then ab2.sh behind queue.sh)
B=/mnt/homes/zwingel/fm-ada5060
wait_run=$1 name=$2; shift 2
until grep -qE "DONE|STOP" $B/runs/$wait_run/run.log 2>/dev/null; do sleep 120; done
mkdir -p $B/runs/$name
bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/$name bash $B/jobs/ab2.sh $B/src-7d18028d "$@"
