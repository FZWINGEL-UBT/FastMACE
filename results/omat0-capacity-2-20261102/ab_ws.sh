#!/usr/bin/env bash
# (omat0-capacity-2: AB_EXTRA passes more ab_rewrites.py arguments, such as --a-constant)
# Same-process A/Bs of OMAT-0 medium (omat0-medium, hybrid float32): side a the default plan, side b with
# RewriteConfig fields set (tools/ab_rewrites.py, 12 blocks of 300 steps), both build orders, per structure.
# Idle check before every job.
#   ab_ws.sh SOURCE_DIR ROOT "FIELD=VALUE ..." STRUCTURE_SPEC...
set -u
B=/mnt/homes/zwingel/fastmace-capacity
FM=$1 ROOT=$2 SIDE_B=$3; shift 3
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=$B/omat0-cache-$(hostname)
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model
mkdir -p "$ROOT/tmp" "$C"
log=$ROOT/run.log
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
echo "$(date -u +%FT%TZ) $(hostname) fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) a=[${SIDE_A:-}] b=[$SIDE_B] extra=[${AB_EXTRA:-}]" >> $log
for spec in "$@"; do
  for order in ab ba; do
    if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
    if [ -n "$gpu$busy" ]; then echo "$(date -u +%FT%TZ) STOP gpu=[$gpu] busy=[$busy]" >> $log; exit 3; fi
    name=$(basename "${spec%%@*}" .cif)-${spec##*@}-$order
    "$E/bin/python" "$FM/tools/ab_rewrites.py" --output "$ROOT/$name" --checkpoint "$CK" --model omat0-medium \
      --structures "$spec" ${SIDE_A:+--a $SIDE_A} --b $SIDE_B ${AB_EXTRA:-} --order "$order" --pattern "$pattern" > "$ROOT/$name.log" 2>&1 < /dev/null
    echo "$(date -u +%FT%TZ) end $name exit=$? $(grep -E 'b/a|ratio' "$ROOT/$name.log" | tail -n 2 | tr '\n' ' ' | cut -c1-250)" >> $log
  done
done
echo "$(date -u +%FT%TZ) DONE" >> $log
