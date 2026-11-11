#!/usr/bin/env bash
# leto, CPU: rewrite the golden plan texts (FASTMACE_UPDATE_GOLDEN=1) in the suite's tree, then check them.
set -u
D=/mnt/homes/zwingel/fm-memprofile/suite-rungb
L=/mnt/homes/zwingel/environments/mace_cueq_latest/bin
cd $D/src
export PYTHONPATH=$D/src TMPDIR=$D/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= GIT_CONFIG_GLOBAL=/mnt/homes/zwingel/fm-tiled/gitconfig-tests
FASTMACE_UPDATE_GOLDEN=1 $L/python -m pytest -q -p no:cacheprovider --basetemp=$D/pt-golden tests/cpu/test_plan_text.py > $D/golden-update.log 2>&1
$L/python -m pytest -q -p no:cacheprovider --basetemp=$D/pt-golden2 tests/cpu/test_plan_text.py > $D/golden-check.log 2>&1
echo "check exit=$? $(tail -n 1 $D/golden-check.log)"
