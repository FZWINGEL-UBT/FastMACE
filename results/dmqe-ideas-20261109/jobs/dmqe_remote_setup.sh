#!/usr/bin/env bash
set -eu
run=/mnt/homes/zwingel/fm-dmqe-20261109
mkdir -p "$run/src" "$run/tmp" "$run/cache" "$run/results"
printf 'RUN %s\n' "$run"
ls -l /mnt/homes/zwingel/Models/mace-omat-0-medium.model /mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model /mnt/homes/zwingel/FastMACE/.source_commit 2>/dev/null || true
ps -u sovic -o pid,stat,time,etime,comm
sleep 2
ps -u sovic -o pid,stat,time,etime,comm
exit
