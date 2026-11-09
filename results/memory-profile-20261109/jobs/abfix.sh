#!/usr/bin/env bash
# Same-process MD-step A/B of tabled_outer_sum.SHAPE_CARRIER (work removal: the recomputing backward's [E, K] shape
# carrier becomes a stride-0 view) on the RTX 4000 Ada, both build orders, Benchmark_loup settings (SrTiO3 10x10x10,
# 1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side); side a = False (the old floats), side b = True.
# Uses the omat0-ada-capacity common.sh (environment; others(): no other user, no GPU job, no pytest; we wait).
#   ROOT=<dir> bash abfix.sh SRC_DIR
set -u -o pipefail
FM=$1; shift
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
o=fastmace.kernels.triton.tabled_outer_sum
note "abfix start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  quiet=0; last=""
  while [ $quiet -lt 2 ]; do
    if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before $order: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
    sleep 60
  done
  note "begin $order"; t0=$(date +%s)
  python tools/ab_rewrites.py --output "$ROOT/ab-$order" --checkpoint "$CK" --model omat0-medium \
    --structures "$ST@10x10x10" --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $o:SHAPE_CARRIER=False \
    --b-constant $o:SHAPE_CARRIER=False --pack "shape|$o:SHAPE_CARRIER=True" > "$ROOT/ab-$order.log" 2>&1 < /dev/null
  note "end $order exit=$? wall $(( $(date +%s) - t0 )) s"
  grep -h -E 'b/a' "$ROOT/ab-$order.log" | sed "s/^/  $order /" >> "$log"
done
note "ABFIX DONE"
