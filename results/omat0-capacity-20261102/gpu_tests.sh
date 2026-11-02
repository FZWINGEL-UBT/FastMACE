#!/usr/bin/env bash
# The GPU tests of the generic records on a workstation (pytest in mace_cueq_latest, the hybrid children in
# mace_cueq_oeq), idle check first.
#   gpu_tests.sh SOURCE_DIR ROOT PYTEST_ARGS...
set -u
B=/mnt/homes/zwingel/fastmace-capacity
FM=$1 ROOT=$2; shift 2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
PY=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
C=$B/omat0-cache-$(hostname)
mkdir -p "$ROOT/tmp" "$C"
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
if [ -n "$gpu$busy" ]; then echo "STOP gpu=[$gpu] busy=[$busy]" > $ROOT/status; exit 3; fi
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton XDG_CACHE_HOME=$C/xdg TMPDIR=$ROOT/tmp \
  FASTMACE_HYBRID_PYTHON=$E/bin/python GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
cd $FM
PYTHONPATH=$FM $PY -m pytest -p no:cacheprovider -q --basetemp=$ROOT/bt -s "$@" > $ROOT/pytest.log 2>&1
echo "exit=$?" > $ROOT/status; tail -3 $ROOT/pytest.log >> $ROOT/status
