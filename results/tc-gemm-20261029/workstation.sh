#!/usr/bin/env bash
# Tensor-core GEMM screen (owner, 29 October 2026) on one shared workstation GPU (trillian: RTX A4000, sm_86; an
# RTX 4000 Ada only when one is free). NAS hybrid environment mace_cueq_oeq. Phases:
#   check   tools/bench/tc_gemm/check.py: the CUTLASS library against float64, graph replay, the patched record
#   screen  tools/bench/tc_gemm/tc_gemm_screen.py on NaCl, NaNbO3, NbCl5 (every mode; idea 3's prepacked FP16 check)
# Run from a scratch folder under /mnt/homes/zwingel holding a git archive of the source (FM) and the library (LIB);
# every job first checks that no other user's job runs and the GPU has no compute process, and stops otherwise.
#   bash workstation.sh ROOT FM LIB PHASE...
set -u -o pipefail
ROOT=$1 FM=$2 LIB=$3; shift 3
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor FASTMACE_SCREEN_TC_GEMM_LIB=$LIB \
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
host=$(hostname)
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) phases=$*"
for phase in "$@"; do case $phase in
check)
  run check $E/bin/python "$FM/tools/bench/tc_gemm/check.py" || exit 1 ;;
screen)
  run screen $E/bin/python "$FM/tools/bench/tc_gemm/tc_gemm_screen.py" --output "$ROOT/screen" --provider "$host" ;;
*) note "unknown phase $phase"; exit 2 ;;
esac; done
note DONE
