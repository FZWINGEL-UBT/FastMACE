#!/usr/bin/env bash
# Rule 12's estimate against the measured peak, per memory-rescue stage: capacity_scan.py point (30 steps after 10),
# every rescue record set explicitly, so the planner keeps the stage and records its estimate for it.
#   est_matrix.sh ROOT STRUCTURE REPEAT...
# A stage that runs out of memory at one size is skipped at the larger ones. Idle check before every job.
set -u
B=/mnt/homes/zwingel/fastmace-capacity
ROOT=$1 STRUCTURE=$2; shift 2
FM=$B/src-rl
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=$ROOT/cache
mkdir -p "$ROOT/tmp" "$ROOT/points"
log=$ROOT/run.log
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
echo "$(date -u +%FT%TZ) $(hostname) fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $log
BASE="wprime_backward=on"
declare -A STAGES=(
  [s0-default]="recompute_conv_weights=off recompute_first_table=off recompute_node_segments=off recompute_node_region=off gate_epilogue=on"
  [s1-conv]="recompute_conv_weights=on recompute_first_table=off recompute_node_segments=off recompute_node_region=off gate_epilogue=on"
  [s2-table]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=off recompute_node_region=off gate_epilogue=on"
  [s3-segments]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=off gate_epilogue=on"
  [s4-geoff]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=off gate_epilogue=off"
  [s5-region]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=on gate_epilogue=off"
  [s6-region-ge]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=on gate_epilogue=on"
)
for stage in s6-region-ge s5-region s4-geoff s3-segments s2-table s1-conv s0-default; do
  for repeat in "$@"; do
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
    if [ -n "$gpu$busy" ]; then echo "$(date -u +%FT%TZ) STOP gpu=[$gpu] busy=[$busy]" >> $log; exit 3; fi
    out="$ROOT/point-$stage-$repeat.log"
    "$E/bin/python" "$FM/tools/bench/capacity_scan.py" point --structure "$STRUCTURE" --model mh1-r2scan \
      --precision float32 --repeat "$repeat" --steps 30 --warmup 10 --output "$ROOT/points" --tag "$stage" \
      --rewrites $BASE ${STAGES[$stage]} > "$out" 2>&1 < /dev/null
    last=$(tail -n 1 "$out" | cut -c1-200)
    echo "$(date -u +%FT%TZ) end $stage $repeat $last" >> $log
    case "$last" in *": ok,"*) ;; *) break ;; esac
  done
done
echo "$(date -u +%FT%TZ) DONE" >> $log
