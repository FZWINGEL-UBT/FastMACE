#!/bin/bash
# ws_gate_cuts.sh SHA: the 3x float32 force/stress gate (tools/force_stress_gate.py, tiled R = 48, compiled; SrTiO3
# 3x3x3, NaCl, NbCl5) on the resident cuts' tree, this workstation's RTX 4000 Ada. run.sh's idle check (exit 3: wait).
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA OUTSUB=gate
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-gate-cuts-$(hostname).log
G="--engine fastmace --model omat0-medium --checkpoint /mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model
   --structures-root /mnt/homes/zwingel/ResearchModule --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5
   --tiled-receivers 48 --tiled-mode compiled"
for attempt in $(seq 1 180); do
  bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-$SHA-gate tools/force_stress_gate.py $G > /dev/null 2>&1
  code=$?
  [ $code != 3 ] && break
  echo "$(date -Is) gate: not idle, waiting" >> $LOG
  sleep 120
done
echo "$(date -Is) end gate exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/$(hostname)-$SHA-gate/run.log | tail -1 | cut -c1-300)" >> $LOG
