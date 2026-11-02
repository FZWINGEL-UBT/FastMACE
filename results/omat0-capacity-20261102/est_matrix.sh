#!/usr/bin/env bash
# (omat0-capacity: estimate-band's est_matrix.sh for OMAT-0 medium and its three stages; source as the first argument)
# Rule 12's estimate against the measured peak, per memory-rescue stage: capacity_scan.py point (30 steps after 10),
# every rescue record set explicitly, so the planner keeps the stage and records its estimate for it.
#   est_matrix.sh SOURCE_DIR ROOT STRUCTURE REPEAT...
# A stage that runs out of memory at one size is skipped at the larger ones. Idle check before every job.
set -u
B=/mnt/homes/zwingel/fastmace-capacity
FM=$1 ROOT=$2 STRUCTURE=$3; shift 3
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=$B/omat0-cache-$(hostname)
mkdir -p "$ROOT/tmp" "$ROOT/points"
log=$ROOT/run.log
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
echo "$(date -u +%FT%TZ) $(hostname) fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $log
BASE=""
declare -A STAGES=(
  [s0-default]="chunked_conv_backward=off recompute_conv_weights=off"
  [s1-chunked]="chunked_conv_backward=on recompute_conv_weights=off"
  [s2-recompute]="chunked_conv_backward=on recompute_conv_weights=on"
)
for stage in s2-recompute s1-chunked s0-default; do
  for repeat in "$@"; do
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
    if [ -n "$gpu$busy" ]; then echo "$(date -u +%FT%TZ) STOP gpu=[$gpu] busy=[$busy]" >> $log; exit 3; fi
    out="$ROOT/point-$stage-$repeat.log"
    "$E/bin/python" "$FM/tools/bench/capacity_scan.py" point --structure "$STRUCTURE" --model omat0-medium \
      --precision float32 --repeat "$repeat" --steps 30 --warmup 10 --output "$ROOT/points" --tag "$stage" \
      --rewrites $BASE ${STAGES[$stage]} --engine memory_probe=off > "$out" 2>&1 < /dev/null
    last=$(tail -n 1 "$out" | cut -c1-200)
    echo "$(date -u +%FT%TZ) end $stage $repeat $last" >> $log
    case "$last" in *": ok,"*) ;; *) break ;; esac
  done
done
echo "$(date -u +%FT%TZ) DONE" >> $log
