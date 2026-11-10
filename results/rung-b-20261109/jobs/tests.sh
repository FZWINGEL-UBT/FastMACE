#!/usr/bin/env bash
# leto, CPU only: the tests this change touches (mace_cueq_latest has pytest), beside our own GPU jobs.
#   bash tests.sh SRC OUT
set -u
FM=$1; OUT=$2
E=/mnt/homes/zwingel/environments/mace_cueq_latest
cd "$FM" && export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= TMPDIR=$OUT/tmp GIT_CONFIG_GLOBAL=/mnt/homes/zwingel/fm-tiled/gitconfig-tests
mkdir -p "$OUT/tmp"
"$E/bin/python" -m pytest -q -p no:cacheprovider --basetemp="$OUT/pt" tests/cpu/test_generic_chunked.py tests/cpu/test_memory_rescue_retry.py \
  tests/cpu/test_layering.py tests/cpu/test_memory_estimate.py tests/cpu/test_memory_fit.py \
  tests/cpu/test_recompute_first_table.py tests/cpu/test_rewrites_generic.py tests/cpu/test_reproducible.py \
  tests/cpu/test_config.py tests/cpu/test_report.py tests/cpu/test_recompute_node_segments.py tests/cpu/test_memory_probe.py \
  > "$OUT/tests.log" 2>&1 < /dev/null
echo "exit=$? $(tail -n 1 "$OUT/tests.log")" >> "$OUT/tests.log"
