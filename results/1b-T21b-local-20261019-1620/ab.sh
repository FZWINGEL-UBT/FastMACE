#!/bin/bash
# 1b-T21b: Simulation vs fastmace.ase.Langevin, 300-step blocks, before (main) and after (this branch).
set -euo pipefail
ROOT=~/fastmace-acceptance/1b-T21b-local-$(date +%Y%m%d-%H%M)
mkdir -p $ROOT/before-src $ROOT/tmp ~/fastmace-cache/acceptance
git -C /mnt/d/Margraf/FastMACE -c safe.directory='*' archive origin/main | tar -x -C $ROOT/before-src
AFTER_REV=c38bb5612f; mkdir -p $ROOT/after-src; git -C /mnt/d/Margraf/FastMACE -c safe.directory="*" archive $AFTER_REV | tar -x -C $ROOT/after-src; AFTER=$ROOT/after-src
C=~/fastmace-cache/acceptance
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=~/models FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 \
  TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
PY=~/environments/mace_cueq_oeq/bin/python
for side in before after before-mntd after-mntd; do
  case $side in before) SRC=$ROOT/before-src;; after) SRC=$AFTER;; before-mntd) SRC=/mnt/d/Margraf/FastMACE;; after-mntd) SRC=/mnt/d/Margraf/FastMACE/.claude/worktrees/1b-t21-ab9016;; esac
  echo "== $side $(git -C /mnt/d/Margraf/FastMACE -c safe.directory='*' rev-parse --short origin/main) / $(git -C /mnt/d/Margraf/FastMACE -c safe.directory='*' rev-parse --short $AFTER_REV)"
  PYTHONPATH=$SRC $PY $SRC/tools/ab_frontends.py --output $ROOT/$side --sides simulation ase --model mh1-r2scan \
    --structure NaCl --warmup 300 --block 300 --blocks 12 2>&1 | grep -v -i warn | tail -3
done
echo "done $ROOT"
