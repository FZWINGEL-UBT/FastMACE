#!/usr/bin/env bash
# WSL, 5060 Ti: the int32 edge_index falsifier, off then on, then the comparison, in one local_gpu reservation.
#   bash edge32_local.sh OUT REPEAT
set -u
OUT=$1; REPEAT=${2:-10x10x10}
W=/mnt/d/Margraf/FastMACE-ada-capacity
PY=$HOME/environments/mace_cueq_oeq/bin/python
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
CACHE=$HOME/fastmace-cache/ada-capacity-local
mkdir -p "$OUT/tmp"
export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$OUT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
S=$W/tools/bench/edge32_probe.py
cd "$OUT"
{
  if [ "${ONLY_REPRODUCIBLE:-0}" != 1 ]; then
  "$PY" "$S" run "$OUT" --side off --repeat "$REPEAT"; echo "off exit=$?"
  "$PY" "$S" run "$OUT" --side on --repeat "$REPEAT"; echo "on exit=$?"
  "$PY" "$S" run "$OUT" --side off --repeat "$REPEAT" --tag 2; echo "off2 exit=$?"
  "$PY" "$S" compare "$OUT/edge32-off.pt" "$OUT/edge32-on.pt"; echo "compare exit=$?"
  "$PY" "$S" compare "$OUT/edge32-off.pt" "$OUT/edge32-off2.pt"; echo "control exit=$?"
  fi
  "$PY" "$S" run "$OUT" --side off --repeat "$REPEAT" --reproducible --model mh1-r2scan --tag r; echo "off r exit=$?"
  "$PY" "$S" run "$OUT" --side on --repeat "$REPEAT" --reproducible --model mh1-r2scan --tag r; echo "on r exit=$?"
  "$PY" "$S" run "$OUT" --side off --repeat "$REPEAT" --reproducible --model mh1-r2scan --tag r2; echo "off r2 exit=$?"
  "$PY" "$S" compare "$OUT/edge32-offr.pt" "$OUT/edge32-onr.pt"; echo "reproducible compare exit=$?"
  "$PY" "$S" compare "$OUT/edge32-offr.pt" "$OUT/edge32-offr2.pt"; echo "reproducible control exit=$?"
} > "$OUT/edge32.log" 2>&1
echo "EDGE32 DONE" >> "$OUT/edge32.log"
