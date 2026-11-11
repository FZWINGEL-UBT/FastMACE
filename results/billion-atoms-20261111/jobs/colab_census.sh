#!/bin/bash
# Billion-atom plan, step 0 (11 November 2026): the device census of the capacity call at 20,480,000 atoms
# (SrTiO3 160^3) on a Colab A100, the capacity record's settings, two calls (eager, then compiled at the budget's
# cap). --ledger names the live tensors after the build and each call; --call-history 1 traces the compiled call
# and keeps its allocations live at its peak by source line. Run from the source root by
# tools/colab/fastmace_colab.py ($OUT set).
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $OUT/gpu.csv
free -g | tee $OUT/host.txt
export MALLOC_MMAP_THRESHOLD_=1048576
python tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 160x160x160 --mode auto \
  --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto \
  --ledger --call-history 1 --output $OUT/census > $OUT/census.log 2>&1
echo "census exit $?"
tail -c 1500 $OUT/census.log
