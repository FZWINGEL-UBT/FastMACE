#!/usr/bin/env bash
# leto, after leto-A1A3: pack A4 (ab2.sh) behind queue.sh.
B=/mnt/homes/zwingel/fm-ada5060
until grep -qE "DONE|STOP" $B/runs/leto-A1A3/run.log 2>/dev/null; do sleep 120; done
mkdir -p $B/runs/leto-A4
bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/leto-A4 bash $B/jobs/ab2.sh $B/src-7d18028d A4
