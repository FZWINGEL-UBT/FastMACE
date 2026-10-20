#!/usr/bin/env bash
# OPT-T02 stage B (segment_gemm) on murphy, one job at a time. Start detached:
#   setsid nohup bash murphy.sh SRC RUN_ID < /dev/null > /dev/null 2>&1 &
# 1. gpu-test: tests/gpu/test_segment_gemm_cuda.py (untimed)
# 2. bench-rule: tools/segment_gemm_bench.py --rule, the shape rule's own configuration (timed, short)
# 3. gate: tools/force_stress_gate.py --rewrites gated_rows=on segment_gemm=on (NaCl, NbCl5)
# 4. ab-ab, ab-ba: tools/ab_rewrites.py, a = gated_rows=on, b = gated_rows=on segment_gemm=on, 12 blocks,
#    build order a-b with ABAB and b-a with ABBA (timed)
# Before every job: no compute process on the GPU and the 1-minute load below LOAD_MAX (waits up to 30
# minutes, then stops). ONLY="name ..." runs only those jobs. Files only under /mnt/homes/zwingel.
set -uo pipefail
src=$1; id=$2
work=/mnt/homes/zwingel/fastmace-work/opt-T02b
run=$work/runs/$id
cache=/mnt/homes/zwingel/fastmace-cache/opt-T02b-murphy
oeq=/mnt/homes/zwingel/environments/mace_cueq_oeq
latest=/mnt/homes/zwingel/environments/mace_cueq_latest
baseline=/mnt/homes/zwingel/fastmace-baseline/1a-T18
models=/mnt/homes/zwingel/ResearchModule/Models
checkpoint=$models/mace-mh-1.model
LOAD_MAX=${LOAD_MAX:-6}
log=$run/run.log
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
job() {  # name env_dir command...
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
job gpu-test "$oeq" "${launcher[@]}" -m pytest -p no:cacheprovider -q -rf "$src/tests/gpu/test_segment_gemm_cuda.py" \
  --basetemp="$run/pytest"
job bench-rule "$oeq" "${launcher[@]}" "$src/tools/segment_gemm_bench.py" --output "$run/bench-rule" --rule
job gate "$oeq" "$oeq/bin/python" "$src/tools/force_stress_gate.py" --output "$run/gate" --engine fastmace \
  --checkpoint "$checkpoint" --model mh1-r2scan --structures NaCl NbCl5 --structures-root "$baseline" \
  --rewrites gated_rows=on segment_gemm=on --reference-command "$latest/bin/python" \
  --production-command "${launcher[*]}"
job ab-ab "$oeq" "${launcher[@]}" "$src/tools/ab_rewrites.py" --output "$run/ab-ab" --checkpoint "$checkpoint" \
  --a gated_rows=on --b gated_rows=on segment_gemm=on --order ab --pattern abab
job ab-ba "$oeq" "${launcher[@]}" "$src/tools/ab_rewrites.py" --output "$run/ab-ba" --checkpoint "$checkpoint" \
  --a gated_rows=on --b gated_rows=on segment_gemm=on --order ba --pattern abba
note done
