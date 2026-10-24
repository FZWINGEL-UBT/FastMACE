#!/usr/bin/env bash
# OPT4-T03 stage 1 (tf32x3_gemms accuracy) on the local RTX 5060 Ti (sm_120, WSL2). Untimed. One job per
# reservation, launched as a background task of the worker's session:
#   python3 tools/ci/local_gpu.py run --wait-minutes 600 --purpose "opt4-T03 JOB (<15 min)" -- \
#     env FM=<pinned source copy> bash results/opt4-T03/local.sh ROOT JOB
# FM is a pinned copy of the source (git archive into ~/fastmace-opt4/src-t03-<sha>, with .source_commit).
# Jobs:
#   gpu-test       tests/gpu/test_tf32x3_gemm_cuda.py (mace_cueq_latest: the OEq environment has no pytest)
#   accuracy       tools/tf32x3_accuracy.py, MH-1, NaCl and NbCl5 3x3x3 (real operands of one force call)
#   gate-NaCl      tools/force_stress_gate.py --rewrites tf32x3_gemms=on, NaCl 3x3x3
#   gate-NbCl5     the same on NbCl5 3x3x3
#   gate0-NaCl     the gate with the default plan (the same seeds), for the comparison
#   gate0-NbCl5
#   omol-accuracy  tools/tf32x3_accuracy.py, OMOL-0 XL, NaCl 3x3x3
#   ogate-NaCl     the gate on OMOL-0 XL, NaCl 3x3x3, tf32x3; ogate0-NaCl the same with the default plan
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
REF=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/opt4-T03
CK=$HOME/models/mace-mh-1.model
OMOL=$HOME/models/MACE-omol-0-extra-large-1024.model
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
run() {  # name command...
  local name=$1; shift
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  if [ -n "$pids" ]; then note "$name NOT IDLE pids=[$pids]"; exit 3; fi
  note "start $name load=$(cut -d' ' -f1-3 /proc/loadavg) avail_mb=$(free -m | awk '/^Mem:/{print $7}')"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
gate() {  # name structure rewrites...
  local name=$1 structure=$2; shift 2
  run "$name" "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/$name" --engine fastmace \
    --checkpoint "${GATE_CK:-$CK}" --model "${GATE_MODEL:-mh1-r2scan}" --structures "$structure" --structures-root "$B" \
    --reference-command "$REF" --production-command "$PY" "$@"
}
note "fastmace=$(cat "$FM/.source_commit") source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gpu-test) run $job "$REF" -m pytest -p no:cacheprovider -q -rf "$FM/tests/gpu/test_tf32x3_gemm_cuda.py" \
                --basetemp="$ROOT/pytest" ;;
    accuracy) run $job "$PY" "$FM/tools/tf32x3_accuracy.py" --output "$ROOT/accuracy" --structures NaCl NbCl5 ;;
    gate-NaCl) gate $job NaCl --rewrites tf32x3_gemms=on ;;
    gate-NbCl5) gate $job NbCl5 --rewrites tf32x3_gemms=on ;;
    gate0-NaCl) gate $job NaCl ;;
    gate0-NbCl5) gate $job NbCl5 ;;
    omol-accuracy) run $job "$PY" "$FM/tools/tf32x3_accuracy.py" --output "$ROOT/omol-accuracy" --model omol0-xl \
                     --structures NaCl ;;
    ogate-NaCl) GATE_CK=$OMOL GATE_MODEL=omol0-xl gate $job NaCl --rewrites tf32x3_gemms=on ;;
    ogate0-NaCl) GATE_CK=$OMOL GATE_MODEL=omol0-xl gate $job NaCl ;;
    *) note "unknown job $job" ;;
  esac
done
note done
