#!/usr/bin/env bash
# leto (RTX 4000 Ada): the memory profiles, each case alone on the GPU. Before each case: wait until no other
# user has a process above 20 % CPU and no process is on the GPU, then two quiet minutes. Our own CPU suites
# (zwingel, CUDA_VISIBLE_DEVICES empty) may run beside it: the profile is not timing-sensitive.
#   ROOT=... FM=... bash leto.sh CASE:REPEAT...
set -u
B=/mnt/homes/zwingel/fm-ada-capacity
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
export PY=$E/bin/python CACHE=$B/cache-$(hostname) FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule CUDA_CACHE_DISABLE=1
busy() {
  local gpu users
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  users=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$users" ] && { echo "gpu=[$gpu] users=[$users]"; return 0; }
  return 1
}
wait_idle() {
  local quiet=0 last=
  while [ $quiet -lt 4 ]; do
    if why=$(busy); then
      quiet=0; [ "${why//[0-9]/}" != "$last" ] && echo "$(date -u +%FT%TZ) waiting: $why" >> "$ROOT/run.log"; last=${why//[0-9]/}
    else quiet=$((quiet + 1)); fi
    sleep 30
  done
  echo "$(date -u +%FT%TZ) idle: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> "$ROOT/run.log"
}
export -f busy wait_idle
export ROOT WAIT_IDLE=wait_idle
exec bash "$(dirname "$0")/profile.sh" "$ROOT" "$@"
