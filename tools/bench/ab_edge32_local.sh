#!/usr/bin/env bash
# WSL, RTX 5060 Ti, default allocator: the same A/B as ab_edge32_leto.sh, both orders, inside one local_gpu
# reservation (run through ab_edge32_launch_local.sh).
#   bash ab_edge32_local.sh OUT [REPEAT]
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
unset PYTORCH_CUDA_ALLOC_CONF
k=fastmace.neighbors.skin
cd "$W"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  "$PY" tools/ab_rewrites.py --output "$OUT/ab-$REPEAT-$order" --checkpoint "$HOME/models/mace-omat-0-medium.model" \
    --model omat0-medium --structures "$W/results/loup-a100-20261103/SrTiO3.cif@$REPEAT" --warmup 200 --block 256 \
    --blocks 6 --order "$order" --pattern "$pattern" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --a-constant $k:EDGE_INDEX_INT32=False --b-constant $k:EDGE_INDEX_INT32=True > "$OUT/ab-$REPEAT-$order.log" 2>&1 < /dev/null
  echo "$order exit=$?"
  grep -h -E 'b/a' "$OUT/ab-$REPEAT-$order.log" | sed "s/^/  $order /"
done > "$OUT/ab.log" 2>&1
echo "AB DONE" >> "$OUT/ab.log"
