#!/usr/bin/env bash
# Same-process MD-step A/B of fastmace.neighbors.skin.EDGE_INDEX_INT32 on the RTX 4000 Ada, both build orders: side
# a = False (int64, today), side b = True (int32). Benchmark_loup settings (SrTiO3 10x10x10, 1 fs, skin 0.5, warm-up
# 200, 6 blocks of 256 steps per side). The omat0-ada-capacity common.sh (environment; others(): no other user, no
# GPU job, no pytest; we wait, two quiet minutes before each order).
#   ROOT=<dir> bash ab_edge32_leto.sh SRC_DIR [REPEAT]
set -u -o pipefail
FM=$1; REPEAT=${2:-10x10x10}
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
k=fastmace.neighbors.skin
note "ab_edge32 start fastmace=$(cat .source_commit) repeat=$REPEAT $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  quiet=0; last=""
  while [ $quiet -lt 2 ]; do
    if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before $order: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
    sleep 60
  done
  note "begin $order"; t0=$(date +%s)
  python tools/ab_rewrites.py --output "$ROOT/ab-$REPEAT-$order" --checkpoint "$CK" --model omat0-medium \
    --structures "/mnt/homes/zwingel/fm-memprofile/SrTiO3.cif@$REPEAT" --warmup 200 --block 256 --blocks 6 --order "$order" \
    --pattern "$pattern" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --a-constant $k:EDGE_INDEX_INT32=False --b-constant $k:EDGE_INDEX_INT32=True > "$ROOT/ab-$REPEAT-$order.log" 2>&1 < /dev/null
  note "end $order exit=$? wall $(( $(date +%s) - t0 )) s"
  grep -h -E 'b/a' "$ROOT/ab-$REPEAT-$order.log" | sed "s/^/  $order /" >> "$log"
done
note "AB_EDGE32 DONE"
