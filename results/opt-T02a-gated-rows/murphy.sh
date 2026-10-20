#!/usr/bin/env bash
# OPT-T02 stage A (gated_rows) on murphy, one job at a time. Start detached:
#   setsid nohup bash murphy.sh SRC RUN_ID < /dev/null > /dev/null 2>&1 &
# 1. gpu-cueq64: plain cuEq float64, gated_rows off against on, bitwise (NaCl, NbCl5; NVT and NPT stress)
# 2. gpu-hybrid32: the same with the hybrid float32 backend (recorded)
# 3. gate: tools/force_stress_gate.py, FastMACE minimal planner with --rewrites gated_rows=on (NaCl, NbCl5)
# 4. ab-off-first, ab-on-first: same-process A/B, 12 blocks of 300 steps, ABAB and ABBA
# Before every job: no compute process on the GPU and the 1-minute load below LOAD_MAX (waits up to 30
# minutes, then stops); the idle record goes into the run log. Files only under /mnt/homes/zwingel.
# ONLY="gpu-cueq64 gpu-cueq64-npt" (environment) runs only the named jobs.
set -uo pipefail
src=$1; id=$2
work=/mnt/homes/zwingel/fastmace-work/opt-T02a
run=$work/runs/$id
cache=/mnt/homes/zwingel/fastmace-cache/opt-T02a-murphy
oeq=/mnt/homes/zwingel/environments/mace_cueq_oeq
latest=/mnt/homes/zwingel/environments/mace_cueq_latest
baseline=/mnt/homes/zwingel/fastmace-baseline/1a-T18
models=/mnt/homes/zwingel/ResearchModule/Models
checkpoint=$models/mace-mh-1.model
LOAD_MAX=${LOAD_MAX:-6}
log=$run/run.log
here=$src/results/opt-T02a-gated-rows
mkdir -p "$run/tmp" "$cache"
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export FASTMACE_CACHE=$cache FASTMACE_MODELS=$models FASTMACE_STRUCTURES=$baseline
export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton MACE_COMPILE_CACHE=$cache/compile
export XDG_CACHE_HOME=$cache/xdg TMPDIR=$run/tmp TMP=$run/tmp TEMP=$run/tmp TORCHINDUCTOR_COMPILE_THREADS=4
libs() { local site=$1/lib/python3.10/site-packages; echo "$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib"; }
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

job() {  # name env_dir command...; ONLY="name ..." runs only those jobs
  local name=$1 env=$2; shift 2
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return 0; fi
  idle "$name" || exit 3
  note "start $name"
  LD_LIBRARY_PATH=$(libs "$env") "$@" > "$run/$name.log" 2>&1 < /dev/null
  local rc=$?
  note "end $name exit=$rc load=$(cut -d' ' -f1-3 /proc/loadavg)"
  return $rc
}

note "source commit $(cat "$src/.source_commit") host $(hostname)"
job gpu-cueq64 "$latest" "$latest/bin/python" "$here/gpu_check.py" --output "$run/gpu-cueq64" \
  --checkpoint "$checkpoint" --backend cueq
job gpu-cueq64-npt "$latest" "$latest/bin/python" "$here/gpu_check.py" --output "$run/gpu-cueq64-npt" \
  --checkpoint "$checkpoint" --backend cueq --stress
job gpu-hybrid32 "$oeq" "${launcher[@]}" "$here/gpu_check.py" --output "$run/gpu-hybrid32" \
  --checkpoint "$checkpoint" --backend hybrid
job gate "$oeq" "$oeq/bin/python" "$src/tools/force_stress_gate.py" --output "$run/gate" --engine fastmace \
  --checkpoint "$checkpoint" --model mh1-r2scan --structures NaCl NbCl5 --structures-root "$baseline" \
  --rewrites gated_rows=on --reference-command "$latest/bin/python" \
  --production-command "${launcher[*]}"
job ab-off-first "$oeq" "${launcher[@]}" "$here/ab_gated_rows.py" --output "$run/ab-off-first" \
  --checkpoint "$checkpoint" --order off on --pattern abab
job ab-on-first "$oeq" "${launcher[@]}" "$here/ab_gated_rows.py" --output "$run/ab-on-first" \
  --checkpoint "$checkpoint" --order on off --pattern abba
note done
