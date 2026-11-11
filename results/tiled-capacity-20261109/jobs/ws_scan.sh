#!/bin/bash
# ws_scan.sh SHA REPEAT...: tiled capacity scan on this workstation (10 November 2026), upward from the 149M record:
# OMAT-0 medium, SrTiO3, --mode auto, tile lists, aliased host state, species index, wavefront on,
# state_dtype auto, two calls. Before each size: none of our other jobs (capacity, profiles, pytest) and run.sh's
# idle check (no compute process on the GPU, no other user's busy process). Stops at the first failure.
set -u
SHA=$1; shift
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-scan-$(hostname).log
for rep in "$@"; do
  ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[f]m-memprofile|[c]apacity_scan|[l]aunch_|[p]ytest|[t]iled_capacity')
  claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
  if [ "$ours" != 0 ] || [ "$claims" != 0 ]; then echo "$(date -Is) $rep: our jobs ($ours) or claims ($claims) here: stop" >> $LOG; break; fi
  echo "$(date -Is) start $rep" >> $LOG
  bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-cap-$SHA-$rep tools/bench/tiled_capacity.py --structure SrTiO3 \
    --model omat0-medium --repeat ${rep}x${rep}x${rep} --mode auto --tile-list --host-state --alias-state \
    --species-index --wavefront on --calls 2 --state-dtype auto > /dev/null 2>&1
  code=$?
  last=$(grep -v "^exit" /mnt/homes/zwingel/fm-tiled/runs/$(hostname)-cap-$SHA-$rep/run.log 2>/dev/null | tail -1 | cut -c1-400)
  echo "$(date -Is) end $rep exit=$code $last" >> $LOG
  { [ $code = 0 ] && echo "$last" | grep -q '"phase": "done"'; } || break
done
echo "$(date -Is) scan over" >> $LOG
