#!/bin/bash
# copy local jobs 51-56 records into the worktree's results
set -eu
R=/home/fzwin/fm-runs/tiled-speed-20261108
D=/mnt/d/Margraf/FastMACE-tiled-speed/results/tiled-speed-20261111/idea11-local
mkdir -p "$D"
for j in 57 58 59 60 61 62 63; do
  mkdir -p "$D/job$j"
  cp -r "$R/job$j/." "$D/job$j/"
  cp "$R/job$j.log" "$D/job$j/run.log"
done
find "$D" -type f -size +2M
du -sh "$D"
