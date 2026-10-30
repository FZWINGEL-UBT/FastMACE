#!/bin/bash
# Pair-search grid fix on murphy: the GPU neighbour tests, then the same-process A/B against main's screen.
# Idle check (run.sh's rule) before each job; everything under /mnt/homes/zwingel/fm-tiled.
set -u
W=/mnt/homes/zwingel/fm-tiled
SRC=$W/src
OUT=$W/runs/pair-search-grid
ENV=/mnt/homes/zwingel/environments/mace_cueq_oeq
mkdir -p "$OUT/tmp" "$W/cache"
idle() {
  local pids busy
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  busy=$(ps -eo user:20,pcpu,etime,comm --sort=-pcpu | awk 'NR > 1 && $1 != "root" && $1 != "zwingel" && $1 !~ /^systemd/ && $2 > 5')
  { date -u +%FT%TZ; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; ps -eo user:20,pcpu,etime,comm --sort=-pcpu | head -5; } >> "$OUT/idle_check.txt"
  if [ -n "$pids" ] || [ -n "$busy" ]; then echo "NOT IDLE: [$pids] [$busy]" | tee -a "$OUT/idle_check.txt"; exit 3; fi
}
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$W/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export FASTMACE_STRUCTURES=/mnt/homes/zwingel/fastmace-baseline/1a-T18 TRITON_CACHE_DIR=$W/cache/triton
export XDG_CACHE_HOME=$W/cache/xdg TMPDIR=$OUT/tmp TMP=$OUT/tmp TEMP=$OUT/tmp
site=$ENV/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$SRC"
idle
"$ENV/bin/python" -m pytest -p no:cacheprovider -q -W ignore --basetemp="$OUT/bt" tests/gpu/test_neighbors_gpu.py \
  tests/gpu/test_neighbors_lean_gpu.py tests/gpu/test_d3_list_gpu.py tests/gpu/test_async_rebuild_gpu.py \
  > "$OUT/gpu-tests.log" 2>&1
echo "gpu tests exit $?"; tail -3 "$OUT/gpu-tests.log"
idle
"$ENV/bin/python" tools/bench/pair_search_ab.py --baseline "$W/main-48af40c6" --repeat 4x4x8 6x6x20 --rounds 10 \
  --output "$OUT" > "$OUT/ab.log" 2>&1
echo "ab exit $?"; grep '^{' "$OUT/ab.log"
