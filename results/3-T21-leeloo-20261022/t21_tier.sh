#!/bin/bash
# 3-T21 Ada leg on leeloo: one GPU test tier of the cd43463 snapshot, after idle checks, ending by 15:58 UTC.
#   bash t21_tier.sh NAME 'MARKER EXPRESSION'
set -u
name=$1; marker=$2
R=/mnt/homes/zwingel/fastmace-runs/3-T21-leeloo-20261022
S=/mnt/homes/zwingel/fastmace-t21-cd43463
C=/mnt/homes/zwingel/fastmace-cache/3-T21-leeloo
E=/mnt/homes/zwingel/environments/mace_cueq_latest
SP=$E/lib/python3.10/site-packages
mkdir -p $C/tmp $C/xdg $C/fastmace $C/inductor $C/triton $R/$name
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg FASTMACE_CACHE=$C/fastmace
export TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
export FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export FASTMACE_HYBRID_PYTHON=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
export FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/1a-T18
export FASTMACE_RESULTS=$R/$name/results
echo "== $name $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) source $(cat $S/.source_commit) $(date -u +%FT%TZ)"
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$pids" ]; then for p in $pids; do ps -o user=,pid=,cmd= -p $p; done; echo "GPU busy, not starting"; exit 3; fi
others=$(ps -eo uid=,user=,pcpu=,etime=,comm= | awk '$1>=1000 && $2!="zwingel" && $3>5')
if [ -n "$others" ]; then echo "$others"; echo "another user active, not starting"; exit 4; fi
left=$(( $(date -u -d "today 15:58" +%s) - $(date -u +%s) ))
if [ $left -lt 600 ]; then echo "too close to the 16:00 stop, not starting"; exit 5; fi
echo "== deadline in ${left}s"
cd $S
timeout -s INT -k 60 $left $E/bin/python -m pytest tests/gpu -m "$marker" -p no:cacheprovider \
  --basetemp=$C/pytest-$name -rfEs --durations=60 --junitxml=$R/$name/junit.xml
echo "== exit $? $(date -u +%FT%TZ)"
