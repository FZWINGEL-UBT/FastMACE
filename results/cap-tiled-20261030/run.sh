#!/bin/bash
# Tiled capacity jobs on murphy (RTX 4000 Ada 20 GB): run.sh NAME SCRIPT [ARGS...]
# Runs SCRIPT (relative to the source copy) with the hybrid NAS environment, after the idle check: no compute
# process on the GPU and no other user's busy process (root, system accounts and zwingel ignored). The check's
# output is kept with the run. Everything under /mnt/homes/zwingel/fm-tiled (outputs, caches, TMPDIR).
set -u
W=/mnt/homes/zwingel/fm-tiled
SRC=$W/src
RUN=$W/runs/$1
shift
mkdir -p "$RUN/tmp" "$W/cache"
ENV=/mnt/homes/zwingel/environments/mace_cueq_oeq
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
busy=$(ps -eo user:20,pcpu,etime,comm --sort=-pcpu | awk 'NR > 1 && $1 != "root" && $1 != "zwingel" && $1 !~ /^systemd/ && $1 !~ /^_/ && $2 > 5')
{
  date -u +%FT%TZ
  nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv,noheader
  for pid in $pids; do echo "compute pid $pid user $(ps -o user= -p "$pid")"; done
  ps -eo user:20,pcpu,etime,comm --sort=-pcpu | head -8
  echo "source $(cat "$SRC/.source_commit")"
} > "$RUN/idle_check.txt"
if [ -n "$pids" ] || [ -n "$busy" ]; then
  echo "NOT IDLE: pids=[$pids] busy=[$busy]" | tee -a "$RUN/idle_check.txt" >&2
  exit 3
fi
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=$(cat "$SRC/.source_commit")
export FASTMACE_CACHE=$W/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export FASTMACE_STRUCTURES=/mnt/homes/zwingel/fastmace-baseline/1a-T18
export TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TRITON_CACHE_DIR=$W/cache/triton MACE_COMPILE_CACHE=$W/cache/compile
export XDG_CACHE_HOME=$W/cache/xdg TMPDIR=$RUN/tmp TMP=$RUN/tmp TEMP=$RUN/tmp TORCHINDUCTOR_COMPILE_THREADS=4
site=$ENV/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$SRC"
script=$1
shift
# OUTSUB: a fresh subdirectory as the output (force_stress_gate.py refuses an existing one).
"$ENV/bin/python" "$script" "$@" --output "$RUN${OUTSUB:+/$OUTSUB}" > "$RUN/run.log" 2>&1 < /dev/null
code=$?
echo "exit $code" >> "$RUN/run.log"
tail -c 3000 "$RUN/run.log"
exit $code
