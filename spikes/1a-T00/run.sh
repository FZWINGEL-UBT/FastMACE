#!/usr/bin/env bash
# 1a-T00 spike: run probe.py in one mode with a fresh cache, detached from the calling shell.
# Usage: bash run.sh MODE RUN_ID [PROBE_ARG ...]
#   MODE: script | bare | inproc | inproc-nopreload | reexec | launched
# Results: ~/fastmace-spike-cache/RUN_ID/{out/result.json,out/forces.npy,log.txt}
set -euo pipefail
mode=$1
run=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
environment=$HOME/environments/mace_cueq_oeq
python=$environment/bin/python
cache=$HOME/fastmace-spike-cache/$run
baseline=$HOME/fastmace-baseline/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh
test ! -e "$cache"  # every attempt gets a fresh cache, so the OEq extension is really rebuilt
mkdir -p "$cache"
probe=("$python" "$here/probe.py" --mode "$mode" --cache "$cache" --out "$cache/out" "$@")
case $mode in
    script) command=(bash "$baseline" "$environment" "$cache" "${probe[@]}") ;;
    launched) command=(env PYTHONPATH="$repo" "$python" -m fastmace.kernels._oeq_runtime "$cache" -- "${probe[@]}") ;;
    *) command=("${probe[@]}") ;;
esac
printf '%q ' "${command[@]}" > "$cache/command.txt"
cd "$cache"
setsid nohup "${command[@]}" > "$cache/log.txt" 2>&1 < /dev/null &
echo "started $mode as $run (pid $!)"
