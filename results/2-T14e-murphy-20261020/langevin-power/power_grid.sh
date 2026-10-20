#!/bin/bash
# 2-T14e: the Langevin power grid on murphy's CPUs (no GPU), 6 runs at a time.
R=/mnt/homes/zwingel/fastmace-2-T14e
cd $R/power
rm -f results.jsonl errors.log DONE
one() {
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=$R/src PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
    /mnt/homes/zwingel/environments/mace_cueq_latest/bin/python langevin_power.py "$1" "$2" "$3" >> results.jsonl 2>> errors.log
}
export -f one
export R
xargs -P 6 -L1 bash -c 'one "$0" "$1" "$2"' < grid.txt
touch DONE
