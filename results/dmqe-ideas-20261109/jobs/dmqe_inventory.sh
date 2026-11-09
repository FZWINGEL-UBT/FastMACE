#!/usr/bin/env bash
set -u
cd /mnt/d/Margraf/fm-dmqe-ideas
printf 'LOCAL\n'
date -u +%FT%TZ
free -h
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv
python3 tools/ci/local_gpu.py status
for item in "$HOME/environments/mace_cueq_latest/bin/python" "$HOME/environments/mace_cueq_oeq/bin/python" "$HOME/models/mace-omat-0-medium.model"; do
  test ! -e "$item" || ls -ld "$item"
done
command -v ncu || true
for host in murphy leto coraline; do
  printf '\nWORKSTATION %s\n' "$host"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" bash -s < tools/local/dmqe_remote_inventory.sh
done
