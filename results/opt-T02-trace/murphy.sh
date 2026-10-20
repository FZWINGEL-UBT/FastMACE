#!/usr/bin/env bash
# The 2-T20a main pass with OPT-T02 stage A on (gated_rows=on), NaCl and NbCl5, on murphy. Start detached:
#   setsid nohup bash murphy.sh SRC RUN_ID < /dev/null > /dev/null 2>&1 &
# Before each process: no compute process on the GPU and the 1-minute load below LOAD_MAX (waits up to 30
# minutes). Files only under /mnt/homes/zwingel.
set -uo pipefail
src=$1; id=$2
work=/mnt/homes/zwingel/fastmace-work/opt-T02-trace
run=$work/runs/$id
cache=/mnt/homes/zwingel/fastmace-cache/opt-T02-trace-murphy
oeq=/mnt/homes/zwingel/environments/mace_cueq_oeq
baseline=/mnt/homes/zwingel/fastmace-baseline/1a-T18
models=/mnt/homes/zwingel/ResearchModule/Models
LOAD_MAX=${LOAD_MAX:-6}
log=$run/run.log
mkdir -p "$run/tmp" "$cache"
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export FASTMACE_CACHE=$cache FASTMACE_MODELS=$models FASTMACE_STRUCTURES=$baseline
export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton MACE_COMPILE_CACHE=$cache/compile
export XDG_CACHE_HOME=$cache/xdg TMPDIR=$run/tmp TMP=$run/tmp TEMP=$run/tmp TORCHINDUCTOR_COMPILE_THREADS=4
site=$oeq/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
launcher=(bash "$baseline/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh" "$oeq"
          "$cache/oeq-launcher" "$oeq/bin/python")
idle() {
  local name=$1 waited=0 pids load
  while true; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    load=$(cut -d' ' -f1 /proc/loadavg)
    if [ -z "$pids" ] && awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then break; fi
    if [ $waited -ge 1800 ]; then note "$name NOT IDLE after ${waited}s: pids=[$pids] load=$load"; return 1; fi
    sleep 30; waited=$((waited + 30))
  done
  note "$name idle (waited ${waited}s) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) top=[$(ps -eo pcpu,user,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | paste -sd';' -)]"
}
note "source commit $(cat "$src/.source_commit") host $(hostname)"
for structure in NaCl NbCl5; do
  idle "$structure" || exit 3
  note "start $structure"
  "${launcher[@]}" "$src/tools/step_breakdown.py" --output "$run/$structure" --checkpoint "$models/mace-mh-1.model" \
    --model mh1-r2scan --backend hybrid --temperature-K 1000 --structure "$structure" --rewrites gated_rows=on \
    > "$run/$structure.log" 2>&1 < /dev/null
  note "end $structure exit=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
done
note done
