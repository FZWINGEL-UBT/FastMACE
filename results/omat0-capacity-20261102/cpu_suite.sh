#!/bin/bash
# omat0-capacity on a workstation's CPUs: the planner, memory, probe, calibration, report and provenance files (GPU hidden,
# NAS basetemp, -n 8)
set -u
B=/mnt/homes/zwingel/fastmace-capacity
SHA=$1
PY=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
SRC=$B/src-$SHA OUT=$B/cpu-$SHA; mkdir -p $OUT/tmp $OUT/cache
busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
if [ -n "$busy" ]; then echo "NOT IDLE busy=[$busy]" > $OUT/status; exit 3; fi
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 TMPDIR=$OUT/tmp FASTMACE_CACHE=$OUT/cache XDG_CACHE_HOME=$OUT/cache/xdg \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
cd $SRC
files=$(ls tests/cpu/test_layering.py tests/cpu/test_memory*.py tests/cpu/test_recompute*.py tests/cpu/test_planner*.py \
  tests/cpu/test_minimal_planner.py tests/cpu/test_fallback*.py tests/cpu/test_simulation*.py tests/cpu/test_rules*.py \
  tests/cpu/test_config.py tests/cpu/test_docs_options.py tests/cpu/test_report*.py tests/cpu/test_plan*.py \
  tests/cpu/test_calibrate.py tests/cpu/test_metadata*.py tests/cpu/test_provenance*.py tests/cpu/test_subprocess*.py \
  tests/cpu/test_contracts_5.py tests/cpu/test_restart*.py tests/cpu/test_generic*.py tests/cpu/test_reproducible.py tests/cpu/test_capabilit*.py tests/cpu/test_rewrites*.py tests/cpu/test_registry*.py 2>/dev/null | sort -u)
PYTHONPATH=$SRC $PY -m pytest -p no:cacheprovider -q -n 8 --dist=load --maxschedchunk=1 --basetemp=$OUT/bt $files > $OUT/suite.log 2>&1
echo "suite exit=$?" > $OUT/status; tail -3 $OUT/suite.log >> $OUT/status
