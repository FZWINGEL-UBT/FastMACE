#!/usr/bin/env bash
# fastmace/spline-registry-fix (8583f6da, = merged tip on main d7dd7e23) on whichever Ada is quiet first:
# 1. the CPU registry test and the GPU multi-engine test (three engines of different sizes, one closed mid-way,
#    against each alone);
# 2. compute-sanitizer memcheck on that multi-engine layout, on the fix and on the timing source 7d18028d (before);
# 3. the full CPU suite (-n 8 -m "not acceptance", GPU hidden).
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-regfix
QUIET=3 bash $B/jobs/queue.sh true
mkdir $B/runs/regfix 2>/dev/null || { echo "claimed elsewhere"; exit 0; }
hostname > $B/runs/regfix/host
export HOST=ada ROOT=$B/runs/regfix
source $B/jobs/common.sh
CS="$B/compute-sanitizer/compute-sanitizer --tool memcheck --leak-check no --report-api-errors no --print-limit 50 --error-exitcode 99"
cd $FM && export PYTHONPATH=$FM
note "regfix start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m pytest -p no:cacheprovider -s -v tests/cpu/test_spline_forward_registry.py tests/gpu/test_spline_registry_engines_cuda.py \
  > $ROOT/tests.log 2>&1 < /dev/null
note "end tests exit=$? $(tail -1 $ROOT/tests.log)"
python -c "import sys; sys.path.insert(0, 'tests/gpu'); import test_spline_registry_engines_cuda as t; print(t.CHILD)" > $ROOT/child.py
for tree in src-regfix src-7d18028d; do
  if why=$(others); then note "STOP before memcheck $tree: $why"; exit 3; fi
  (cd $B/$tree && PYTHONPATH=$B/$tree $CS python $ROOT/child.py > $ROOT/memcheck-$tree.log 2>&1 < /dev/null)
  note "end memcheck $tree exit=$? $(grep -m5 -hE 'Invalid|ERROR SUMMARY|illegal|by thread' $ROOT/memcheck-$tree.log | tr '\n' ' ' | cut -c1-500)"
done
if why=$(others); then note "STOP before suite: $why"; exit 3; fi
CUDA_VISIBLE_DEVICES= python -m pytest -p no:cacheprovider -n 8 -m "not acceptance" tests/cpu -q > $ROOT/suite.log 2>&1 < /dev/null
note "end suite exit=$? $(tail -1 $ROOT/suite.log)"
note DONE
