#!/usr/bin/env bash
# murphy: after the POLAR-1 megakernel gates (gate.sh writes no DONE), mark the run done, then memcheck_chain.sh.
B=/mnt/homes/zwingel/fm-ada5060
until grep -q "gate mk-L exit" $B/runs/murphy-polar-gate/run.log 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) murphy DONE" >> $B/runs/murphy-polar-gate/run.log
bash $B/jobs/memcheck2.sh murphy-polar-gate
