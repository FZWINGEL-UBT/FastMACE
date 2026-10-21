#!/usr/bin/env bash
# OPT-T08 (species_tables, fold_readout_tail, fold_product_linear) on one GPU, one job at a time.
# murphy:  setsid nohup bash run.sh SRC RUN_ID murphy < /dev/null > /dev/null 2>&1 &
# local:   setsid -f python3 tools/ci/local_gpu.py run --purpose "OPT-T08 A/B" -- bash run.sh SRC RUN_ID local
# 1. gpu-cueq64: plain cuEq float64, first force evaluation, default against all three records (round-off)
# 2. gpu-hybrid32: the same with the hybrid float32 backend
# 3. gate: tools/force_stress_gate.py, FastMACE minimal planner with the three records on (NaCl, NbCl5)
# 4. ab-off-first, ab-on-first: same-process A/B of all three, 12 blocks of 300 steps, ABAB and ABBA
# 5. ab-c1, ab-c1c3, ab-c2: attribution, off first, ABAB
# Before every job: no compute process on the GPU (murphy) and the 1-minute load below LOAD_MAX (waits up
# to 30 minutes, then stops); the idle record goes into the run log. ONLY="job ..." runs only those jobs.
set -uo pipefail
src=$1; id=$2; machine=${3:-murphy}
if [ "$machine" = murphy ]; then
  root=/mnt/homes/zwingel; env_root=$root/environments; models=$root/ResearchModule/Models
  baseline=$root/fastmace-baseline/1a-T18
else
  root=$HOME; env_root=$HOME/environments; models=$HOME/models; baseline=$HOME/fastmace-baseline/1a-T18
fi
work=$root/fastmace-work/opt-T08
run=$work/runs/$id
cache=$root/fastmace-cache/opt-T08-$machine
oeq=$env_root/mace_cueq_oeq
latest=$env_root/mace_cueq_latest
checkpoint=$models/mace-mh-1.model
LOAD_MAX=${LOAD_MAX:-6}
log=$run/run.log
here=$src/results/opt-T08-linear-audit
all=(species_tables=on fold_readout_tail=on fold_product_linear=on)
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
    pids=$( [ "$machine" = murphy ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, - )
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

note "source commit $(cat "$src/.source_commit") host $(hostname) machine $machine"
job gpu-cueq64 "$latest" "$latest/bin/python" "$here/gpu_check.py" --output "$run/gpu-cueq64"   --checkpoint "$checkpoint" --backend cueq --rewrites "${all[@]}"
job gpu-hybrid32 "$oeq" "${launcher[@]}" "$here/gpu_check.py" --output "$run/gpu-hybrid32"   --checkpoint "$checkpoint" --backend hybrid --rewrites "${all[@]}"
job gate "$oeq" "$oeq/bin/python" "$src/tools/force_stress_gate.py" --output "$run/gate" --engine fastmace   --checkpoint "$checkpoint" --model mh1-r2scan --structures NaCl NbCl5 --structures-root "$baseline"   --rewrites "${all[@]}" --reference-command "$latest/bin/python"   --production-command "${launcher[*]}"
job ab-off-first "$oeq" "${launcher[@]}" "$here/ab_species.py" --output "$run/ab-off-first"   --checkpoint "$checkpoint" --order off on --pattern abab --rewrites "${all[@]}"
job ab-on-first "$oeq" "${launcher[@]}" "$here/ab_species.py" --output "$run/ab-on-first"   --checkpoint "$checkpoint" --order on off --pattern abba --rewrites "${all[@]}"
job ab-c1 "$oeq" "${launcher[@]}" "$here/ab_species.py" --output "$run/ab-c1"   --checkpoint "$checkpoint" --order off on --pattern abab --rewrites species_tables=on
job ab-c1c3 "$oeq" "${launcher[@]}" "$here/ab_species.py" --output "$run/ab-c1c3"   --checkpoint "$checkpoint" --order off on --pattern abab --rewrites species_tables=on fold_product_linear=on
job ab-c2 "$oeq" "${launcher[@]}" "$here/ab_species.py" --output "$run/ab-c2"   --checkpoint "$checkpoint" --order off on --pattern abab --rewrites fold_readout_tail=on
note done
