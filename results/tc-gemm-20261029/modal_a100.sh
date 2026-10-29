#!/usr/bin/env bash
# Tensor-core GEMM screen (owner, 29 October 2026) on a Modal A100, run from the source root by
# tools/modal/fastmace_modal.py::job with a tarball that holds the source and tc_gemm_lib/libfm_tc_gemm.so
# (tools/bench/tc_gemm/build.sh, SASS sm_80):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min N --tarball T \
#     --command "bash results/tc-gemm-20261029/modal_a100.sh PHASE [VARIANT]" --out DIR
# Phases:
#   screen   check.py, then tc_gemm_screen.py (every mode) on NaCl NaNbO3 NbCl5, then overlap traces of the default
#            and of tf32x3_gemms on the CUTLASS variants rz3 and rn3 (NaNbO3, NbCl5)
#   ab V     same-process whole-step A/Bs (ab_tc.py over ab_rewrites.py, 12 blocks of 300 steps): a = the default
#            plan, b = tf32x3_gemms=on with CUTLASS variant V; both build orders; NaNbO3 NbCl5 NaCl
#   acc V    the force/stress gate (b's configuration) and NVE drift (20 ps, 1 fs, 300 K) on NaCl and NbCl5, with
#            and without it
# A step starts only before LIMIT seconds (default 80 % of the job's own timeout argument), so the job returns what
# finished.
set -u -o pipefail
phase=$1 variant=${2:-rn3} limit=${LIMIT:-1500}
start=$(date +%s)
CK=$FASTMACE_MODELS/mace-mh-1.model
export FASTMACE_SCREEN_TC_GEMM_LIB=$PWD/tc_gemm_lib/libfm_tc_gemm.so
W=tools/bench/tc_gemm
log=$OUT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
run() {  # name command...
  local name=$1; shift
  if [ $(( $(date +%s) - start )) -ge "$limit" ]; then note "skip $name (time)"; return 0; fi
  note "begin $name"
  "$@" > "$OUT/$name.log" 2>&1 < /dev/null
  local status=$?
  note "end $name exit=$status"
  return $status
}
note "fastmace=$(cat .source_commit) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) phase=$phase variant=$variant"
case $phase in
screen)
  run check python $W/check.py || { note "STOP: check failed"; exit 1; }
  run screen python $W/tc_gemm_screen.py --output "$OUT/screen" --provider modal
  for s in NaNbO3 NbCl5; do
    run "overlap-default-$s" python $W/overlap_trace.py --output "$OUT/overlap-default" --checkpoint "$CK" --structure "$s"
    for v in rz3 rn3; do
      run "overlap-$v-$s" env FASTMACE_SCREEN_TC_GEMM=cutlass-$v python $W/run_with_backend.py $W/overlap_trace.py \
        --output "$OUT/overlap-$v" --checkpoint "$CK" --structure "$s" --rewrites tf32x3_gemms=on grouped_launch=off
    done
  done
  rm -f "$OUT"/overlap-*/*.trace.json ;;
ab)
  for s in NaNbO3 NbCl5 NaCl; do for order in ab ba; do
    if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
    run "ab-$variant-$s-$order" python $W/ab_tc.py --b-backend "$variant" -- --output "$OUT/ab-$variant-$s-$order" \
      --checkpoint "$CK" --model mh1-r2scan --structures "$s" --b tf32x3_gemms=on grouped_launch=off \
      --order "$order" --pattern "$pattern"
  done; done ;;
acc)
  run "gate-$variant" env FASTMACE_SCREEN_TC_GEMM=cutlass-$variant python tools/force_stress_gate.py \
    --output "$OUT/gate-$variant" --engine fastmace --checkpoint "$CK" --model mh1-r2scan \
    --structures NaCl NbCl5 NaNbO3 --structures-root "$FASTMACE_STRUCTURES" --rewrites tf32x3_gemms=on grouped_launch=off \
    --production-command "python $W/run_with_backend.py"
  run gate-default python tools/force_stress_gate.py --output "$OUT/gate-default" --engine fastmace --checkpoint "$CK" \
    --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$FASTMACE_STRUCTURES"
  for s in NaCl NbCl5; do
    run "nve-$variant-$s" env FASTMACE_SCREEN_TC_GEMM=cutlass-$variant python $W/run_with_backend.py \
      tools/bench/nve_drift.py --structure "$s" --output "$OUT/nve-$variant" --rewrites tf32x3_gemms=on grouped_launch=off
    run "nve-default-$s" python tools/bench/nve_drift.py --structure "$s" --output "$OUT/nve-default"
  done ;;
*) note "unknown phase $phase"; exit 2 ;;
esac
note DONE
