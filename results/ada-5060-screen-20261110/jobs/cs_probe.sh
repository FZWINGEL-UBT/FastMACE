#!/usr/bin/env bash
# compute-sanitizer 2026.2 host crash ("double free or corruption" while loading OMAT-0 medium): which step triggers it.
B=/mnt/homes/zwingel/fm-ada5060
export HOST=ada ROOT=$B/runs/cs-probe
source $B/jobs/common.sh
cd $B/src-regfix && export PYTHONPATH=$B/src-regfix
CSB=$B/compute-sanitizer/compute-sanitizer
for tool in "--tool memcheck" ; do
  $CSB $tool python -c "import torch; print(torch.zeros(4, device='cuda').sum().item())" > $ROOT/p1.log 2>&1; echo "p1 torch exit=$? $(grep -c 'double free' $ROOT/p1.log) $(grep 'ERROR SUMMARY' $ROOT/p1.log)"
  $CSB $tool python -c "import fastmace as fm; m = fm.load('omat0-medium'); print('loaded')" > $ROOT/p2.log 2>&1; echo "p2 load exit=$? $(grep -c 'double free' $ROOT/p2.log) $(grep -E 'ERROR SUMMARY|loaded' $ROOT/p2.log | tr '\n' ' ')"
  $CSB $tool --check-api-memory-access no python -c "import fastmace as fm; m = fm.load('omat0-medium'); print('loaded')" > $ROOT/p3.log 2>&1; echo "p3 load no-api-check exit=$? $(grep -c 'double free' $ROOT/p3.log)"
  MALLOC_CHECK_=0 $CSB $tool python -c "import fastmace as fm; m = fm.load('omat0-medium'); print('loaded')" > $ROOT/p4.log 2>&1; echo "p4 MALLOC_CHECK_=0 exit=$? $(grep -c 'double free' $ROOT/p4.log)"
done
python -c "import fastmace as fm; m = fm.load('omat0-medium'); print('loaded')" > $ROOT/p0.log 2>&1; echo "p0 plain exit=$? $(grep -c 'double free' $ROOT/p0.log)"
