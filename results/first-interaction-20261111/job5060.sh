#!/bin/bash
W=/mnt/d/Margraf/FastMACE-first-interaction
cd $W
export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=$(git -C /mnt/d/Margraf/FastMACE rev-parse main)
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "first-interaction falsifiers (C1/C2 lab)" --quiet --wait-minutes 90 -- \
  ~/environments/mace_cueq_latest/bin/python tools/first_interaction_lab.py --watchdog --output results/first-interaction-20261111/${OUTNAME:-rtx5060ti}
echo "exit $?"
