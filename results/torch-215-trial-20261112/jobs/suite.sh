#!/usr/bin/env bash
# PyTorch 2.15 trial: the full CPU suite (GPU hidden, -n 8, -m "not acceptance") on the b7c19b19 git-archive tree
# with mace_cueq_latest_torch215 (Python 3.11.16, torch 2.15.0+cu132). Same recipe as fm-ada5060/jobs/suite.sh.
#   bash suite.sh  (on a workstation whose GPU has no process)
B=/mnt/homes/zwingel/t215
ROOT=$B/runs/suite-$(hostname); mkdir -p $ROOT/tmp
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" | tee -a "$log"; }
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
[ -n "$gpu" ] && { note "STOP: GPU busy [$gpu]"; exit 3; }
cd $B/src-b7c19b19 && export PYTHONPATH=$PWD TMPDIR=$ROOT/tmp XDG_CACHE_HOME=$ROOT/xdg FASTMACE_CACHE=$ROOT/cache
note "suite start fastmace=$(cat .source_commit) torch 2.15 env"
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="*" CUDA_VISIBLE_DEVICES= \
  /mnt/homes/zwingel/environments/mace_cueq_latest_torch215/bin/python -m pytest -p no:cacheprovider -n 8 \
  -m "not acceptance" --basetemp=$ROOT/pytest-base tests/cpu -q -rfE > $ROOT/suite.log 2>&1 < /dev/null
note "end suite exit=$? $(tail -1 $ROOT/suite.log)"
