#!/usr/bin/env bash
# Keep a self-hosted GitHub Actions runner running without root (plan task 2-T18; docs/development.md, "GPU CI").
#
#   keepalive.sh RUNNER_DIR          start run.sh unless it already runs (murphy: from cron every 10 minutes)
#   keepalive.sh --loop RUNNER_DIR   start it and restart it whenever it exits (the PC: from the Windows log-on)
#
# One instance per runner directory, through flock on RUNNER_DIR/.keepalive.lock. HOME, TMPDIR and the
# caches of the runner process point into RUNNER_DIR, so on a workstation nothing it writes leaves
# /mnt/homes/zwingel. Pause: create RUNNER_DIR/PAUSED, then kill the process group named in
# RUNNER_DIR/.keepalive.pid (never a broad pkill); while PAUSED exists nothing starts. Resume: remove PAUSED.
set -u
# Lead a process group of our own, so that stopping kills this runner and its jobs and nothing else.
# (setsid execs without a fork when the caller does not lead a group, so wsl.exe keeps its foreground.)
if [ "$(cut -d' ' -f5 /proc/$$/stat)" != "$$" ]; then
    exec setsid "$0" "$@"
fi
loop=0
if [ "${1:-}" = "--loop" ]; then
    loop=1
    shift
fi
dir=${1:?usage: keepalive.sh [--loop] RUNNER_DIR}
dir=$(cd "$dir" && pwd) || exit 2
[ -x "$dir/run.sh" ] || { echo "keepalive: $dir/run.sh not found (runner not unpacked?)" >&2; exit 2; }
[ -e "$dir/PAUSED" ] && exit 0
[ -f "$dir/.runner" ] || { echo "keepalive: $dir is not registered (no .runner file)" >&2; exit 2; }

exec 9>"$dir/.keepalive.lock"
flock --nonblock 9 || exit 0  # already running
echo "$$" > "$dir/.keepalive.pid"

mkdir -p "$dir/home" "$dir/tmp" "$dir/_diag"
export HOME="$dir/home" TMPDIR="$dir/tmp" XDG_CACHE_HOME="$dir/home/.cache"
log="$dir/_diag/keepalive.log"
while [ ! -e "$dir/PAUSED" ]; do
    echo "$(date -u +%FT%TZ) keepalive: starting run.sh" >> "$log"
    "$dir/run.sh" >> "$log" 2>&1
    status=$?
    echo "$(date -u +%FT%TZ) keepalive: run.sh exited with $status" >> "$log"
    [ "$loop" = 1 ] || break
    sleep 60
done
