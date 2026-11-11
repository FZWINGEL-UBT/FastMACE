#!/usr/bin/env bash
# The full CPU suite on a tree (GPU hidden, -n 8 -m "not acceptance", mace_cueq_latest), with an explicit --basetemp
# on the shared home (pytest refuses its default base under TMPDIR there: "not owned by the current user", NFS ids), and git's
# safe.directory=* for the colab driver tests' fresh repositories (git config exits 128 there otherwise).
#   bash suite.sh TREE RUNNAME
B=/mnt/homes/zwingel/fm-ada5060
export HOST=ada ROOT=$B/runs/$2
source $B/jobs/common.sh
if why=$(others); then note "STOP before suite: $why"; exit 3; fi
cd $B/$1 && export PYTHONPATH=$B/$1
rm -rf $ROOT/pytest-base
note "suite start fastmace=$(cat .source_commit)"
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="*" CUDA_VISIBLE_DEVICES= /mnt/homes/zwingel/environments/mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -n 8 \
  -m "not acceptance" --basetemp=$ROOT/pytest-base tests/cpu -q > $ROOT/suite.log 2>&1 < /dev/null
note "end suite exit=$? $(tail -1 $ROOT/suite.log)"
