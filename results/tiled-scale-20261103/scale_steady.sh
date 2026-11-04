#!/bin/bash
# 20M SrTiO3 at R 8,192, three calls (the third one steady, no compile): main's grid (--grid-fit off), then
# fit_grid, the same source (branch fastmace/tile-grid-fit 5a162b6c); leto, RTX 4000 Ada.
W=/mnt/homes/zwingel/fm-tiled
A="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --mode auto --tile-list --species-index
  --no-budget --sweep-times --repeat 159x159x159 --calls 3 --receivers 8192 --host-state --alias-state"
# shellcheck disable=SC2086
SRC=$W/src-gridfit-5a162b6c bash $W/run.sh leto-steady-20m-base $A --grid-fit off > /dev/null || exit $?
# shellcheck disable=SC2086
SRC=$W/src-gridfit-5a162b6c bash $W/run.sh leto-steady-20m-fit $A --grid-fit on > /dev/null
echo "exit $?"
