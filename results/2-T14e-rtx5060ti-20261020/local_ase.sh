#!/bin/bash
# 2-T14e on the local RTX 5060 Ti: the 4,000-step Langevin ASE-equivalence test, then BAOAB (new energy limit).
ROOT=$HOME/fastmace-2-T14e
SRC=/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t14-08a9fe
mkdir -p $ROOT/logs $ROOT/results $ROOT/tmp
cd $SRC
export TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 \
  FASTMACE_MODELS=$HOME/models FASTMACE_BASELINE=$HOME/fastmace-baseline/1a-T18 \
  FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python FASTMACE_RESULTS=$ROOT/results
for k in langevin baoab; do
  echo "$(date -Is) start $k $(git -c safe.directory='*' rev-parse --short HEAD) $(git -c safe.directory='*' status --short | tr '\n' ' ')" >> $ROOT/runs.log
  ~/environments/mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -s -rf --durations=0 -m acceptance -k $k \
    --junitxml=$ROOT/logs/$k.junit.xml --basetemp=$ROOT/bt-$k tests/gpu/test_ase_equivalence_gpu.py > $ROOT/logs/$k.log 2>&1
  echo "$(date -Is) $k exit $? $(tail -1 $ROOT/logs/$k.log)" >> $ROOT/runs.log
done
echo "$(date -Is) done" >> $ROOT/runs.log
