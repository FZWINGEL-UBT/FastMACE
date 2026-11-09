#!/usr/bin/env bash
# Queued on leto and murphy; the first idle one claims (runs/ada-claim2) the boundary reruns, then the memory comparison.
J=/mnt/homes/zwingel/fm-ada-capacity
QUIET=2 bash $J/jobs/queue.sh true
if ! mkdir $J/runs/ada-claim2 2>/dev/null; then echo "$(date -u +%FT%TZ) $(hostname): claimed by $(cat $J/runs/ada-claim2/host 2>/dev/null), exiting"; exit 0; fi
hostname > $J/runs/ada-claim2/host
echo "$(date -u +%FT%TZ) $(hostname): claimed"
ROOT=$J/runs/edge-ada bash $J/jobs3/edge.sh $J/src-d3d459ef SrTiO3:32x32x32 NaCl:34x34x34
rm -rf $J/runs/memcmp-ada.stopped; mv $J/runs/memcmp-ada $J/runs/memcmp-ada.stopped 2>/dev/null
ROOT=$J/runs/memcmp-ada bash $J/jobs3/memcmp.sh $J/src-d3d459ef SrTiO3:10x10x10 SrTiO3:24x24x24 NaCl:25x25x25
