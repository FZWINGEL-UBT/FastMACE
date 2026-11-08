#!/bin/bash
# Full CPU suite (merge tier) of fastmace/omat0-ada-capacity on trillian: ruff, then tests/cpu -n 8 -m "not acceptance".
#   bash suite.sh DIR   (DIR/src = git archive of the branch head)
set -u
D=$1; OUT=$D; cd $D/src
L=/mnt/homes/zwingel/environments/mace_cueq_latest/bin
echo "start $(date -u +%FT%TZ) $(hostname)" > "$OUT/summary.txt"
$L/ruff check fastmace tests tools > "$OUT/ruff.log" 2>&1; echo "ruff exit=$? $(tail -n 1 "$OUT/ruff.log")" >> "$OUT/summary.txt"
export PYTHONPATH=$D/src TMPDIR=$D/tmp TMP=$D/tmp TEMP=$D/tmp PYTHONDONTWRITEBYTECODE=1
GIT_CONFIG_GLOBAL=/mnt/homes/zwingel/fm-tiled/gitconfig-tests CUDA_VISIBLE_DEVICES= $L/python -m pytest -q -p no:cacheprovider \
  -n 8 -m "not acceptance" --basetemp="$OUT/pt-cpu" tests/cpu > "$OUT/pytest-cpu.log" 2>&1
echo "cpu exit=$? $(tail -n 1 "$OUT/pytest-cpu.log")" >> "$OUT/summary.txt"
grep -E "^FAILED|^ERROR" "$OUT/pytest-cpu.log" | head -30 >> "$OUT/summary.txt"
echo "done $(date -u +%FT%TZ)" >> "$OUT/summary.txt"
