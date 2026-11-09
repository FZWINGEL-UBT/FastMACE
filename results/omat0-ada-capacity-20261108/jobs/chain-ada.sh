#!/usr/bin/env bash
# Queued on both RTX 4000 Ada workstations (leto, murphy): the first to be idle QUIET minutes claims the run (mkdir on
# the shared home is atomic) and does the capacity scan, then the old/new memory comparison; the other one exits.
J=/mnt/homes/zwingel/fm-ada-capacity
QUIET=2 bash $J/jobs/queue.sh true
if ! mkdir $J/runs/ada-claim 2>/dev/null; then echo "$(date -u +%FT%TZ) $(hostname): claimed by $(cat $J/runs/ada-claim/host 2>/dev/null), exiting"; exit 0; fi
hostname > $J/runs/ada-claim/host
echo "$(date -u +%FT%TZ) $(hostname): claimed the Ada scan"
ROOT=$J/runs/scan-ada-new bash $J/jobs3/scan.sh $J/src-d3d459ef SrTiO3:69120 NaCl:125000
ROOT=$J/runs/memcmp-ada bash $J/jobs3/memcmp.sh $J/src-d3d459ef SrTiO3:10x10x10 SrTiO3:24x24x24 NaCl:25x25x25
