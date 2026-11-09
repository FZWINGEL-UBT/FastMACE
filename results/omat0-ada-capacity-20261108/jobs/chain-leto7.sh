#!/usr/bin/env bash
# leto: OMAT-0 medium capacity with the new Ada defaults (main d3d459ef: adoptions + rescue-rung fixes), from the old maxima.
# Waits for the gate-A chain to finish first, then for QUIET minutes of idle.
J=/mnt/homes/zwingel/fm-ada-capacity
while pgrep -f "[j]obs2/chain-leto6" > /dev/null; do sleep 60; done
QUIET=2 bash $J/jobs/queue.sh true
ROOT=$J/runs/scan-ada-new bash $J/jobs3/scan.sh $J/src-d3d459ef SrTiO3:69120 NaCl:125000
