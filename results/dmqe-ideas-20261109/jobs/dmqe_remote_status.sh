#!/usr/bin/env bash
set -u
run=/mnt/homes/zwingel/fm-dmqe-20261109
hostname
date -u +%FT%TZ
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
for name in murphy-force-v2 leto-outer-v3 leto-force leto-geometry; do
  printf '\n%s\n' "$name"
  test ! -f "$run/results/$name.log" || tail -n 12 "$run/results/$name.log"
done
exit
