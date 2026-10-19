#!/usr/bin/env bash
# Run a file of jobs one after another on murphy, detached: queue_murphy.sh JOBFILE
# Each line: VAR=value ... RUN_NAME MODE SCRIPT ARGS...   (variables for run_murphy.sh: SRC, CACHE,
# MACE_COMPILE_CACHE_DIR; then run_murphy.sh's arguments)
# run_murphy.sh checks that the GPU has no compute process before each job; the queue stops at the
# first busy GPU or failed job. Log: <jobfile>.log
set -uo pipefail
jobs=$1
here=$(cd "$(dirname "$0")" && pwd)
(
  while read -r line; do
    case $line in ''|\#*) continue ;; esac
    read -ra words <<< "$line"
    vars=()
    while [ ${#words[@]} -gt 0 ] && [[ ${words[0]} == [A-Z_]*=* ]]; do vars+=("${words[0]}"); words=("${words[@]:1}"); done
    echo "$(date -u +%FT%TZ) start: $line"
    env FOREGROUND=1 "${vars[@]}" bash "$here/run_murphy.sh" "${words[@]}" || { echo "$(date -u +%FT%TZ) stopped: exit $?"; exit 1; }
    echo "$(date -u +%FT%TZ) done"
  done < "$jobs"
  echo "$(date -u +%FT%TZ) queue complete"
) > "$jobs.log" 2>&1 < /dev/null &
disown
echo "queue started; log $jobs.log"
