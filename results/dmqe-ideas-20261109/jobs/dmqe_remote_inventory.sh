#!/usr/bin/env bash
set -u
hostname
date -u +%FT%TZ
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -eo user,pid,pcpu,etime,comm --sort=-pcpu | head -n 28
for item in /mnt/homes/zwingel/environments/mace_cueq_latest/bin/python /mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python /mnt/homes/zwingel/Models/mace-omat-0-medium.model /mnt/homes/zwingel/FastMACE/.git; do
  test ! -e "$item" || ls -ld "$item"
done
