#!/usr/bin/env bash
# Detached start on a workstation: bash launch.sh LOGFILE CMD...  (runs CMD behind queue.sh)
J=/mnt/homes/zwingel/fm-ada-capacity
log=$1; shift
mkdir -p "$(dirname "$log")"
nohup setsid bash $J/jobs/queue.sh "$@" > "$log" 2>&1 < /dev/null &
echo "launched $! -> $log"
