#!/usr/bin/env bash
# Waits until leto has no pytest run, no other GPU process and no busy CPU (QUIET minutes quiet, default 10), then runs the given command.
#   bash queue.sh CMD...
set -u
quiet=0
while [ $quiet -lt ${QUIET:-10} ]; do
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20' | wc -l)
  suite=$(pgrep -f "[p]ytest" | wc -l)
  if [ -z "$gpu" ] && [ "$busy" = 0 ] && [ "$suite" = 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) queue: quiet, starting $*"
exec "$@"
