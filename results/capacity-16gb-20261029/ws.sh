#!/usr/bin/env bash
# Capacity points on a shared Linux workstation (RTX 4000 Ada 20 GB), for comparison with WSL2 (CAP-T03):
#   bash ws.sh ROOT FM JOB...   with JOB = point:<structure>:<precision>:<NXxNYxNZ>, run from a git archive (FM) under
# /mnt/homes/zwingel. Every job first checks that no other user's job runs and the GPU has no compute process
# (ResearchModule AGENTS.md, "Shared workstations") and stops the sequence otherwise. TRACE / TAG / ENGINE as local.sh.
set -u -o pipefail
ROOT=$1 FM=$2; shift 2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton MACE_COMPILE_CACHE=$ROOT/cache/compile XDG_CACHE_HOME=$ROOT/cache/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for job in "$@"; do
  IFS=: read -r kind structure precision extra <<< "$job"
  if why=$(others); then note "STOP before $job: $why"; exit 3; fi
  opts=("--tag=${TAG:-}"); [ -n "${ENGINE:-}" ] && opts+=(--engine $ENGINE); [ -n "${TRACE:-}" ] && opts+=(--trace)
  note "begin $job"
  $E/bin/python "$FM/tools/bench/capacity_scan.py" point --structure "$structure" --precision "$precision" \
    --output "$ROOT/points" --repeat "$extra" "${opts[@]}" >> "$ROOT/jobs.log" 2>&1 < /dev/null
  note "end $job exit=$?"
done
