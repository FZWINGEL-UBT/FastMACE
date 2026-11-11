#!/bin/bash
W=/mnt/d/Margraf/FastMACE-nested-radial
cd $W
export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=$(git -C $W rev-parse HEAD)
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "nested radial basis bound (K=32 lab)" --quiet --wait-minutes 90 -- \
  ~/environments/mace_cueq_latest/bin/python tools/first_interaction_lab.py --watchdog --only 'copy|k32' --output results/nested-radial-20261111/${OUTNAME:-bound-rtx5060ti}
echo "exit $?"
