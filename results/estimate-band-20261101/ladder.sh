#!/usr/bin/env bash
# leeloo (RTX 4000 Ada, 20 GB): the default plan (no record forced) on the cells of estimate-refit's ladder (NbCl5)
# and on SrTiO3 above its default limit, so rule 12 plans the rescue rungs itself. capacity_scan.py point, 60 steps
# after 20. Idle check before every job.
#   ladder.sh SOURCE_DIR ROOT
set -u
B=/mnt/homes/zwingel/fastmace-capacity
FM=$1 ROOT=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=$B/region-norm-leeloo/cache
mkdir -p "$ROOT/tmp"
log=$ROOT/run.log
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
echo "$(date -u +%FT%TZ) $(hostname) fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $log
for cell in NbCl5:6x6x28 NbCl5:6x6x33 NbCl5:6x6x40 NbCl5:6x6x47 SrTiO3:21x21x21 SrTiO3:25x25x25 SrTiO3:28x28x28; do
  structure=${cell%%:*} repeat=${cell#*:}
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  if [ -n "$gpu$busy" ]; then echo "$(date -u +%FT%TZ) STOP gpu=[$gpu] busy=[$busy]" >> $log; exit 3; fi
  "$E/bin/python" "$FM/tools/bench/capacity_scan.py" point --structure $structure --model mh1-r2scan --precision float32 \
    --repeat $repeat --output "$ROOT/points" --tag default --steps 60 --warmup 20 > "$ROOT/point-$structure-$repeat.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $cell exit=$? $(tail -n 1 "$ROOT/point-$structure-$repeat.log" | cut -c1-240)" >> $log
done
echo "$(date -u +%FT%TZ) DONE" >> $log
