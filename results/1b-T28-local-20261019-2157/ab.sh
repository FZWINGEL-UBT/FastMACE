#!/bin/bash
# 1b-T28: Simulation (and the ASE shim) in 300-step blocks from /mnt/d, before (main checkout) and after (T28 worktree).
set -euo pipefail
ROOT=~/fastmace-acceptance/1b-T28-local-$(date +%Y%m%d-%H%M)
mkdir -p $ROOT/tmp ~/fastmace-cache/acceptance
C=~/fastmace-cache/acceptance
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=~/models FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 \
  TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
PY=~/environments/mace_cueq_oeq/bin/python
for side in before after; do
  case $side in before) SRC=/mnt/d/Margraf/FastMACE;; after) SRC=/mnt/d/Margraf/FastMACE/.claude/worktrees/1b-t21-ab9016;; esac
  echo "== $side $SRC"
  PYTHONPATH=$SRC $PY $SRC/tools/ab_frontends.py --output $ROOT/$side --sides simulation ase --model mh1-r2scan \
    --structure NaCl --warmup 300 --block 300 --blocks 12 2>&1 | grep -v -i warn | tail -2
done
echo "done $ROOT"
