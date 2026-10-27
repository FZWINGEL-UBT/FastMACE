#!/bin/bash
# f64-T04 step 1: three configurations in one reservation (timing is the same kernel each time).
set -euo pipefail
cd /mnt/d/Margraf/fm-f64d/results/f64-T04-20261027
PY=$HOME/environments/mace_cueq_oeq/bin/python
export PYTHONDONTWRITEBYTECODE=1
$PY step1_df64_micro.py --pair-inputs --json step1_rtx5060ti_pair_inputs.json > /dev/null
$PY step1_df64_micro.py --a-low 0.5 --json step1_rtx5060ti_a05.json > /dev/null
$PY step1_df64_micro.py --a-low 0.5 --pair-inputs --json step1_rtx5060ti_a05_pair_inputs.json > /dev/null
echo STEP1-DONE
