#!/usr/bin/env bash
# TF32 screen (owner, 29 October 2026) on one shared workstation GPU (murphy: RTX 4000 Ada, sm_89). MH-1 hybrid
# float32 default plan; side a = IEEE FP32 (production), side b = PyTorch's TF32 matmul mode. Phases:
#   speed  same-process A/Bs (tools/ab_rewrites.py --b-tf32, 12 blocks of 300 steps), both build orders,
#          NaCl NaNbO3 NbCl5
#   control  NaCl A/B with side b's graphs dropped and re-captured in IEEE (the TF32 side's method, no TF32)
#   gemms  per-GEMM breakdown: tools/step_breakdown.py main pass with and without --tf32 (in-model GEMM times),
#          then tools/gemm_precision_bench.py on the FP32 pass's shapes (cuBLAS IEEE / TF32 / Triton tf32x3)
#   acc    force/stress gate with --production-tf32 and an IEEE control; NVE drift 20 ps at 1 fs, NaCl NbCl5,
#          with and without --tf32
# Run from a scratch folder under /mnt/homes/zwingel holding a git archive of the source (FM); every job first
# checks that no other user's job runs and the GPU has no compute process, and stops the sequence otherwise.
#   bash ada.sh ROOT FM PHASE...
set -u -o pipefail
ROOT=$1 FM=$2; shift 2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
REF=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton MACE_COMPILE_CACHE=$ROOT/cache/compile XDG_CACHE_HOME=$ROOT/cache/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {  # other users' processes using CPU (a job), or any GPU compute process
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
run() {  # name command...
  local name=$1 why; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  local status=$?
  note "end $name exit=$status"
  return $status
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) phases=$*"
for phase in "$@"; do case $phase in
speed)
  for s in NaCl NaNbO3 NbCl5; do for order in ab ba; do
    if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
    run "speed-$s-$order" $E/bin/python "$FM/tools/ab_rewrites.py" --output "$ROOT/speed-$s-$order" --checkpoint "$CK" \
      --model mh1-r2scan --structures "$s" --b-tf32 --order "$order" --pattern "$pattern"
  done; done ;;
control)
  for order in ab ba; do
    if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
    run "control-NaCl-$order" $E/bin/python "$FM/tools/ab_rewrites.py" --output "$ROOT/control-NaCl-$order"       --checkpoint "$CK" --model mh1-r2scan --structures NaCl --b-recapture --order "$order" --pattern "$pattern"
  done ;;
gemms)
  for s in NaCl NaNbO3 NbCl5; do
    run "sb-fp32-$s" $E/bin/python "$FM/tools/step_breakdown.py" --output "$ROOT/sb-fp32/$s" --checkpoint "$CK" \
      --structure "$s" --temperature-K 300 --steps 300
    run "sb-tf32-$s" $E/bin/python "$FM/tools/step_breakdown.py" --output "$ROOT/sb-tf32/$s" --checkpoint "$CK" \
      --structure "$s" --temperature-K 300 --steps 300 --tf32
  done
  run gemm-bench $E/bin/python "$FM/tools/gemm_precision_bench.py" --output "$ROOT/gemm-bench" \
    --gemms "$ROOT/sb-fp32/NaCl/gemms.csv" "$ROOT/sb-fp32/NaNbO3/gemms.csv" "$ROOT/sb-fp32/NbCl5/gemms.csv" ;;
acc)
  run gate-tf32 $E/bin/python "$FM/tools/force_stress_gate.py" --output "$ROOT/gate-tf32" --engine fastmace \
    --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" --production-tf32 \
    --reference-command "$REF" --production-command "$E/bin/python"
  run gate-ieee $E/bin/python "$FM/tools/force_stress_gate.py" --output "$ROOT/gate-ieee" --engine fastmace \
    --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
    --reference-command "$REF" --production-command "$E/bin/python"
  for s in NaCl NbCl5; do
    run "nve-tf32-$s" $E/bin/python "$FM/tools/bench/nve_drift.py" --structure "$s" --output "$ROOT/nve-tf32" --tf32
    run "nve-ieee-$s" $E/bin/python "$FM/tools/bench/nve_drift.py" --structure "$s" --output "$ROOT/nve-ieee"
  done ;;
*) note "unknown phase $phase"; exit 2 ;;
esac; done
note DONE
