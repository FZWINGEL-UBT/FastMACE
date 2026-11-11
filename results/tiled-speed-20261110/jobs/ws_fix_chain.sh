#!/bin/bash
# ws_fix_chain.sh SHA: the pinned kept-lists fix on murphy (10 November 2026): the full CPU suite (ws_suite.sh's
# checks), then the host-memory probe's "all on" point again (SrTiO3 150^3, ws_gate_probe.sh's settings and checks).
set -u
SHA=$1
R=/mnt/homes/zwingel/fm-tiled-speed
LOG=$R/ws-fix-chain-$(hostname).log
until out=$(bash $R/ws_suite.sh $SHA) && echo "$out" | grep -q started; do echo "$(date -Is) $out" >> $LOG; sleep 120; done
echo "$(date -Is) suite started" >> $LOG
until grep -q '^exit=' $R/suite-$SHA.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) suite $(tail -2 $R/suite-$SHA.log | tr '\n' ' ')" >> $LOG
export SRC=$R/src-$SHA
M=fastmace.model.tiled
ON="$M:TILE_PREFETCH_DEVICE=True $M:TILE_VIRIAL=True $M:TILE_GRAPH_CACHE=0.25 $M:TILE_FILTER32=True $M:TILE_EPILOGUE=True $M:TILE_PINNED_SUPERSET=True $M:TILE_GRAPH_CACHE_WAVE=True $M:TILE_HOST_WORKER=True $M:TILE_COPY_STREAM=True"
for attempt in $(seq 1 360); do
  ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[f]m-memprofile|[c]apacity_scan|[l]aunch_|[p]ytest|[t]iled_capacity|[t]iled_ab|[f]orce_stress_gate')
  claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
  if [ "$ours" = 0 ] && [ "$claims" = 0 ]; then
    echo "$(date -Is) start probe-on" >> $LOG
    bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-$SHA-probe-on $R/jobs/with_constants.py $ON -- \
      tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 150x150x150 --mode auto \
      --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto > /dev/null 2>&1
    code=$?
    [ $code != 3 ] && { echo "$(date -Is) end probe-on exit=$code" >> $LOG; break; }
    echo "$(date -Is) probe-on: not idle, waiting" >> $LOG
  fi
  sleep 120
done
echo "$(date -Is) fix chain over" >> $LOG
