#!/usr/bin/env bash
# Waits until this workstation has no pytest run, no GPU process and no other user's busy CPU for QUIET minutes
# (default 2), then runs the command.  bash queue.sh CMD...
set -u
quiet=0
while [ $quiet -lt ${QUIET:-2} ]; do
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20' | wc -l)
  suite=$(pgrep -f "[p]ytest" | wc -l)
  if [ -z "$gpu" ] && [ "$busy" = 0 ] && [ "$suite" = 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) $(hostname) queue: quiet, starting $*"
exec "$@"
