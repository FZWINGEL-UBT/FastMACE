#!/usr/bin/env bash
# WSL, stage 0 on the local RTX 5060 Ti: build the shim, the toy capture, then two MD records of SrTiO3 and their
# comparison, inside one local_gpu reservation. The source is the worktree itself (read only).
#   bash local0.sh OUT REPEAT
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
S=$W/tools/bench/arena/arena_stage0.py
cd "$OUT"
{
  "$PY" "$S" build "$OUT" && echo "build exit=0"
  [ "${SKIP_TOY:-0}" = 1 ] || { "$PY" "$S" toy "$OUT"; echo "toy exit=$?"; }
  "$PY" "$S" md "$OUT" --repeat "$REPEAT" --tag a; echo "md a exit=$?"
  "$PY" "$S" md "$OUT" --repeat "$REPEAT" --tag b; echo "md b exit=$?"
  "$PY" "$S" compare "$OUT/md-a.json" "$OUT/md-b.json"; echo "compare exit=$?"
  "$PY" "$S" md "$OUT" --repeat "$REPEAT" --tag c --no-split; echo "md c exit=$?"
  "$PY" "$S" md "$OUT" --repeat "$REPEAT" --tag d --no-split; echo "md d exit=$?"
  "$PY" "$S" compare "$OUT/md-c.json" "$OUT/md-d.json"; echo "compare no-split exit=$?"
} > "$OUT/stage0.log" 2>&1
echo "STAGE0 DONE" >> "$OUT/stage0.log"
