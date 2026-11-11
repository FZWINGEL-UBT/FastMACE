#!/usr/bin/env bash
# Detached start on a workstation: bash launch.sh NAME PACK...  (ab.sh behind queue.sh; logs under runs/NAME)
B=/mnt/homes/zwingel/fm-ada5060
name=$1; shift
mkdir -p $B/runs/$name
ROOT=$B/runs/$name nohup setsid bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/$name bash $B/jobs/ab.sh $B/src-7d18028d "$@" > $B/runs/$name/launch.out 2>&1 < /dev/null &
echo "launched $(hostname) $! -> $B/runs/$name"
