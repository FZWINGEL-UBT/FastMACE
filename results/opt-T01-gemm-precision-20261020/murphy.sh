#!/usr/bin/env bash
# OPT-T01 bench and 2-T20a step breakdown on murphy (RTX 4000 Ada, sm_89). Start detached:
#   setsid nohup bash murphy.sh SRC RUN_ID < /dev/null > /dev/null 2>&1 &
# SRC: the source copy under /mnt/homes/zwingel/fastmace-work/opt-T01 (with .source_commit). One job at a
# time: the GEMM precision bench, then the step breakdown of NaCl and NbCl5 (main pass only: murphy's
# Nsight Compute and Nsight Systems are 2021.3, older than sm_89 and CUDA 13). Before every job: no compute
# process on the GPU from anyone, and the host's 1-minute load below LOAD_MAX (waits up to 30 minutes, then
# stops). The idle record of each job (GPU, processes, load, top CPU users) goes into the run log.
# Everything (outputs, caches, TMPDIR) under /mnt/homes/zwingel (ResearchModule AGENTS.md).
set -uo pipefail
src=$1; id=$2
work=/mnt/homes/zwingel/fastmace-work/opt-T01
run=$work/runs/$id
cache=/mnt/homes/zwingel/fastmace-cache/opt-T01-murphy
env_dir=/mnt/homes/zwingel/environments/mace_cueq_oeq
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
site=$env_dir/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
launcher=(bash "$baseline/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh" "$env_dir"
          "$cache/oeq-launcher" "$env_dir/bin/python")

idle() {  # name: wait for an empty GPU and a quiet host, record both
  local name=$1 waited=0 pids load
  while true; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    load=$(cut -d' ' -f1 /proc/loadavg)
    if [ -z "$pids" ] && awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then break; fi
    if [ $waited -ge 1800 ]; then
      note "$name NOT IDLE after ${waited}s: pids=[$pids] users=[$(for p in ${pids//,/ }; do ps -o user= -p "$p"; done | paste -sd, -)] load=$load"
      return 1
    fi
    sleep 30; waited=$((waited + 30))
  done
  note "$name idle (waited ${waited}s) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) top=[$(ps -eo pcpu,user,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | paste -sd';' -)]"
}

job() {  # name command...
  local name=$1; shift
  idle "$name" || exit 3
  note "start $name"
  "$@" > "$run/$name.log" 2>&1 < /dev/null
  local rc=$?
  note "end $name exit=$rc load=$(cut -d' ' -f1-3 /proc/loadavg)"
  return $rc
}

note "source commit $(cat "$src/.source_commit") host $(hostname)"
job bench "${launcher[@]}" "$src/tools/gemm_precision_bench.py" --output "$run/bench" \
  --gemms "$src/results/2-T20a-step-breakdown-20261020/NaCl/gemms.csv" \
  "$src/results/2-T20a-step-breakdown-20261020/NbCl5/gemms.csv"
for structure in NaCl NbCl5; do
  job "$structure" "${launcher[@]}" "$src/tools/step_breakdown.py" --output "$run/$structure" \
    --checkpoint "$models/mace-mh-1.model" --model mh1-r2scan --backend hybrid --temperature-K 1000 \
    --structure "$structure"
done
note done
