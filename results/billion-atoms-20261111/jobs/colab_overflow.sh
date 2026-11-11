#!/bin/bash
# Billion-atom plan, step 2 (11 November 2026): the whole-system [N, 3] paths past 2^31 elements on a Colab G4
# (96 GB): a 1M-atom dry run, then 750,000,000 atoms (tools/bench/overflow_probe.py). $OUT set by the driver.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $OUT/gpu.csv
python tools/bench/overflow_probe.py --atoms 1000000 --output $OUT/small > $OUT/small.log 2>&1
echo "small exit $?"; tail -12 $OUT/small.log
python tools/bench/overflow_probe.py --atoms 750000000 --output $OUT/big > $OUT/big.log 2>&1
echo "big exit $?"; tail -12 $OUT/big.log
