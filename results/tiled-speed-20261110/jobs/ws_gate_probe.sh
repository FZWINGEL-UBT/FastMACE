#!/bin/bash
# ws_gate_probe.sh SHA: on this workstation's RTX 4000 Ada (10 November 2026, conductor review of 673880c3):
# 1. the 3x float32 force/stress gate (tools/force_stress_gate.py, tiled R = 48, compiled) with the nine
#    TILED_SPEED_GPUS switches all off and all on (decision 66: per family for a rounding change);
# 2. the host-memory probe: the capacity settings of ws_scan.sh at SrTiO3 150^3 (16,875,000 atoms, kept lists
#    above TILE_PINNED_SUPERSET_BYTES), all on, each host-side switch off alone, all off; the record's host peaks.
# Before each run: none of our other jobs here, no Ada claims, and run.sh's idle check (exit 3: wait and retry).
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
J=/mnt/homes/zwingel/fm-tiled-speed/jobs
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-gate-probe-$(hostname).log
M=fastmace.model.tiled
OFF="$M:TILE_PREFETCH_DEVICE=False $M:TILE_VIRIAL=False $M:TILE_GRAPH_CACHE=0.0 $M:TILE_FILTER32=False $M:TILE_EPILOGUE=False $M:TILE_PINNED_SUPERSET=False $M:TILE_GRAPH_CACHE_WAVE=False $M:TILE_HOST_WORKER=False $M:TILE_COPY_STREAM=False"
ON="$M:TILE_PREFETCH_DEVICE=True $M:TILE_VIRIAL=True $M:TILE_GRAPH_CACHE=0.25 $M:TILE_FILTER32=True $M:TILE_EPILOGUE=True $M:TILE_PINNED_SUPERSET=True $M:TILE_GRAPH_CACHE_WAVE=True $M:TILE_HOST_WORKER=True $M:TILE_COPY_STREAM=True"
run() { name=$1; shift
  for attempt in $(seq 1 360); do
    ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[f]m-memprofile|[c]apacity_scan|[l]aunch_|[p]ytest|[t]iled_capacity|[t]iled_ab|[f]orce_stress_gate')
    claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
    if [ "$ours" = 0 ] && [ "$claims" = 0 ]; then
      echo "$(date -Is) start $name" >> $LOG
      bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-$SHA-$name "$@" > /dev/null 2>&1
      code=$?
      if [ $code != 3 ]; then
        echo "$(date -Is) end $name exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/$(hostname)-$SHA-$name/run.log | tail -1 | cut -c1-300)" >> $LOG
        return
      fi
      echo "$(date -Is) $name: not idle, waiting" >> $LOG
    fi
    sleep 120
  done
  echo "$(date -Is) $name: gave up waiting" >> $LOG
}
G="--engine fastmace --model omat0-medium --checkpoint /mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model
   --structures-root /mnt/homes/zwingel/ResearchModule --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5
   --tiled-receivers 48 --tiled-mode compiled"
export OUTSUB=gate
run gate-off tools/force_stress_gate.py $G --production-constant $OFF
run gate-on tools/force_stress_gate.py $G --production-constant $ON
unset OUTSUB
C="-- tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 150x150x150 --mode auto
   --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto"
run probe-on $J/with_constants.py $ON $C
run probe-nopinned $J/with_constants.py $ON $M:TILE_PINNED_SUPERSET=False $C
run probe-noworker $J/with_constants.py $ON $M:TILE_HOST_WORKER=False $C
run probe-nocopy $J/with_constants.py $ON $M:TILE_COPY_STREAM=False $C
run probe-off $J/with_constants.py $OFF $C
echo "$(date -Is) gate-probe over" >> $LOG
