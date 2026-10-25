#!/usr/bin/env bash
# 3b-T10: run one pytest command so that stopping it keeps its results (phase-3.md, open finding 7).
#
#   bash tools/acceptance/stoppable_pytest.sh OUTDIR PYTHON [pytest arguments...]
#
# Runs PYTHON -m pytest -p resultlog --result-log OUTDIR/results.jsonl --junitxml OUTDIR/junit.xml ARGS as a child,
# with tools/acceptance on PYTHONPATH, and writes the child's PID to OUTDIR/pytest.pid. INT, TERM and HUP to this
# script are forwarded to pytest as SIGINT; pytest then ends the session and writes junit.xml. After GRACE_S
# (default 120) seconds a pytest still running is killed; results.jsonl still holds every finished test.
# To stop a run: kill -INT "$(cat OUTDIR/pytest.pid)" (or signal this script); never kill the reservation first.
# Bash cannot trap a signal that was ignored when it started: under `nohup` (HUP) or as a background job of a
# non-interactive shell (INT) only TERM reaches this script; signal pytest's pid instead (its plugin re-arms INT).
set -uo pipefail
out=${1:?OUTDIR}; python=${2:?PYTHON}; shift 2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
grace=${GRACE_S:-120}
mkdir -p "$out"
export PYTHONPATH=$here${PYTHONPATH:+:$PYTHONPATH}
"$python" -m pytest -p resultlog --result-log "$out/results.jsonl" --junitxml "$out/junit.xml" "$@" &
child=$!
echo "$child" > "$out/pytest.pid"
stop() {
  echo "stoppable_pytest: $1, forwarding SIGINT to pytest $child" >&2
  kill -INT "$child" 2>/dev/null
  for _ in $(seq "$grace"); do
    kill -0 "$child" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$child" 2>/dev/null; then
    echo "stoppable_pytest: pytest $child still running after ${grace}s, killing it" >&2
    kill -KILL "$child" 2>/dev/null
  fi
  wait "$child"; status=$?
  exit "$status"
}
trap 'stop SIGINT' INT
trap 'stop SIGTERM' TERM
trap 'stop SIGHUP' HUP
wait "$child"
