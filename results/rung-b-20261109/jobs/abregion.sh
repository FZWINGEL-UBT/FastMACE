#!/usr/bin/env bash
# Same-process MD-step A/B of generic.modules.FIRST_LAYER_REGION on the RTX 4000 Ada, both build orders, with the
# rung it belongs to (generic_recompute_first_table, RewriteConfig.recompute_first_table) forced on on both sides:
# side a = False (the table alone under checkpoint), side b = True (the first interaction and product 0 under one
# checkpoint). Benchmark_loup settings (SrTiO3 10x10x10, 1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per
# side). Uses the omat0-ada-capacity common.sh (environment; others(): no other user, no GPU job, no pytest; we wait).
#   ROOT=<dir> bash abregion.sh SRC_DIR
# Ran on 58ad5c99, where FIRST_LAYER_REGION was a module constant; from the next commit the region is its own rung,
# RewriteConfig.recompute_first_layer (the same code path).
set -u -o pipefail
FM=$1; shift
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
g=fastmace.rewrites.generic.modules
note "abregion start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  quiet=0; last=""
  while [ $quiet -lt 2 ]; do
    if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before $order: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
    sleep 60
  done
  note "begin $order"; t0=$(date +%s)
  python tools/ab_rewrites.py --output "$ROOT/ab-$order" --checkpoint "$CK" --model omat0-medium \
    --structures "/mnt/homes/zwingel/fm-memprofile/SrTiO3.cif@10x10x10" --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a recompute_first_table=on \
    --b recompute_first_table=on --a-constant $g:FIRST_LAYER_REGION=False --b-constant $g:FIRST_LAYER_REGION=False \
    --pack "region|$g:FIRST_LAYER_REGION=True" > "$ROOT/ab-$order.log" 2>&1 < /dev/null
  note "end $order exit=$? wall $(( $(date +%s) - t0 )) s"
  grep -h -E 'b/a' "$ROOT/ab-$order.log" | sed "s/^/  $order /" >> "$log"
done
note "ABREGION DONE"
