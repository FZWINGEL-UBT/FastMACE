#!/bin/bash
# usage: run_batch.sh <run> <seed|default> <cold|warm> [<run> <seed> <mode>]...
set -u
ROOT=/mnt/homes/zwingel/fastmace-2-T14
E=/mnt/homes/zwingel/environments/mace_cueq_latest
SRC=${SRC:-$ROOT/src}; cd $SRC
while [ $# -ge 3 ]; do
  RUN=$1; SEED=$2; MODE=$3; shift 3
  [ -e $ROOT/STOP ] && { echo "$(date -Is) stop file, skipping run $RUN" >> $ROOT/batches.log; break; }
  if [ $MODE = cold ]; then CACHE=$ROOT/cache-cold-$RUN; rm -rf $CACHE; else CACHE=$ROOT/cache-warm; fi
  ORDER=(); [ $SEED != default ] && ORDER=(--t14-seed $SEED)
  mkdir -p $ROOT/tmpdir/$RUN $ROOT/logs $ROOT/junit $ROOT/basetemp
  LOG=$ROOT/logs/run-$RUN.log
  { echo "run $RUN seed $SEED mode $MODE cache $CACHE start $(date -Is) commit $(cat .source_commit)"; uptime;
    ps -eo user,pcpu,args --sort=-pcpu | awk '$2>50' | head -20; } > $LOG
  START=$(date +%s)
  set -o pipefail
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC:$ROOT/plugins \
    FASTMACE_CACHE=$CACHE TMPDIR=$ROOT/tmpdir/$RUN FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
    FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/20261017 \
    nice -n 10 $E/bin/python -m pytest -n 8 -rf -p no:cacheprovider -p t14_order "${ORDER[@]}" \
    --junitxml=$ROOT/junit/run-$RUN.xml --basetemp=$ROOT/basetemp/run-$RUN tests/cpu >> $LOG 2>&1
  CODE=$?
  { echo "exit $CODE end $(date -Is) seconds $(( $(date +%s) - START ))"; uptime;
    ps -eo user,pcpu,args --sort=-pcpu | awk '$2>50' | grep -v fastmace-2-T14 | head -20; } >> $LOG
  echo "$(date -Is) run $RUN seed $SEED $MODE exit $CODE $(tail -n 20 $LOG | grep -E '(passed|failed)' | tail -1)" >> $ROOT/batches.log
done
echo "$(date -Is) batch done" >> $ROOT/batches.log
