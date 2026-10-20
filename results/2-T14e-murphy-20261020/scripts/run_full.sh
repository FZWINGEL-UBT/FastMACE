#!/bin/bash
# 2-T14e on murphy: a GPU tier with the timing hook. usage: run_full.sh <tag> <pytest args...>
H=/mnt/homes/zwingel; ROOT=$H/fastmace-2-T14e; SRC=$ROOT/src; ENV=$H/environments/mace_cueq_latest; CACHE=$ROOT/cache
TAG=$1; shift
mkdir -p $ROOT/logs $ROOT/tmp
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$H/ResearchModule/Models \
  FASTMACE_BASELINE=$H/fastmace-baseline/1a-T18 FASTMACE_STRUCTURES=$H/fastmace-baseline/1a-T18 \
  FASTMACE_HYBRID_PYTHON=$H/environments/mace_cueq_oeq/bin/python \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile \
  XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 \
  T14E_LOG=$ROOT/logs/$TAG.events.jsonl \
  LD_LIBRARY_PATH=$ENV/lib/python3.10/site-packages/nvidia/cu13/lib:$ENV/lib/python3.10/site-packages/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd $SRC
{ echo "start $(date -Is) commit $(cat .source_commit)"; nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do ps -o user=,pid=,args= -p $p | cut -c1-100; done; echo "gpu apps listed"; } > $ROOT/logs/$TAG.log
$ENV/bin/python -m pytest -p no:cacheprovider -rfs --durations=0 --junitxml=$ROOT/logs/$TAG.junit.xml --basetemp=$ROOT/bt-$TAG "$@" >> $ROOT/logs/$TAG.log 2>&1
echo "exit $? $(date -Is)" >> $ROOT/logs/$TAG.log
{ nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do ps -o user=,pid=,args= -p $p | cut -c1-100; done; echo "gpu apps at end"; } >> $ROOT/logs/$TAG.log
