#!/bin/bash
# 3-T02 GPU runs (local RTX 5060 Ti, WSL), inside one local_gpu.py reservation: the fast-tier test, then the
# superset cost. Launch, from the repository root in WSL:
#   setsid -f bash -c 'python3 tools/ci/local_gpu.py run --purpose "3-T02 superset" --wait-minutes 60 -- \
#       bash results/3-T02-rtx5060ti-20261022/run.sh > results/3-T02-rtx5060ti-20261022/reservation.log 2>&1'
set -u
cd "$(dirname "$0")/../.."
OUT=results/3-T02-rtx5060ti-20261022
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_MODELS=~/models \
       FASTMACE_HYBRID_PYTHON=~/environments/mace_cueq_oeq/bin/python FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18
~/environments/mace_cueq_latest/bin/python -m pytest -q -p no:cacheprovider tests/gpu/test_superset_gpu.py \
    --junitxml=$OUT/junit.xml > $OUT/test.log 2>&1
echo "test exit $?" >> $OUT/test.log
~/environments/mace_cueq_oeq/bin/python tools/superset_cost.py --checkpoint ~/models/mace-mh-1.model \
    --output $OUT/NaCl --pattern abba > $OUT/cost.log 2>&1
echo "cost exit $?" >> $OUT/cost.log
