#!/usr/bin/env bash
# The memory ledger (tools/bench/memory_ledger.py) of OMAT-0 medium's default plan on a workstation, one job per
# STRUCTURE:REPEAT, idle check before each.
#   [REWRITES="FIELD=VALUE ..."] ledger_ws.sh SOURCE_DIR ROOT STRUCTURE:REPEAT...
set -u
B=/mnt/homes/zwingel/fastmace-capacity
FM=$1 ROOT=$2; shift 2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=$B/omat0-cache-$(hostname)
mkdir -p "$ROOT/tmp" "$C"
log=$ROOT/run.log
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
echo "$(date -u +%FT%TZ) $(hostname) fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $log
for job in "$@"; do
  structure=${job%%:*} repeat=${job#*:}
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  if [ -n "$gpu$busy" ]; then echo "$(date -u +%FT%TZ) STOP gpu=[$gpu] busy=[$busy]" >> $log; exit 3; fi
  "$E/bin/python" "$FM/tools/bench/memory_ledger.py" --structure "$structure" --model omat0-medium \
    --precision float32 --repeat "$repeat" --output "$ROOT" ${REWRITES:+--rewrites $REWRITES}     > "$ROOT/ledger-$structure-$repeat.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $job exit=$?" >> $log
done
echo "$(date -u +%FT%TZ) DONE" >> $log
