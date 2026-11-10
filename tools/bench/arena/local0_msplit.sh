#!/usr/bin/env bash
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
export PYTORCH_CUDA_ALLOC_CONF=${CONF:-max_split_size_mb:20}
S=$W/tools/bench/arena/arena_stage0.py
cd "$OUT"
{
  echo "conf=$PYTORCH_CUDA_ALLOC_CONF"
  "$PY" "$S" md "$OUT" --repeat "$REPEAT" --tag e; echo "md e exit=$?"
  "$PY" "$S" md "$OUT" --repeat "$REPEAT" --tag f; echo "md f exit=$?"
  "$PY" "$S" compare "$OUT/md-e.json" "$OUT/md-f.json"; echo "compare exit=$?"
} > "$OUT/stage0-msplit.log" 2>&1
echo "STAGE0 DONE" >> "$OUT/stage0-msplit.log"
