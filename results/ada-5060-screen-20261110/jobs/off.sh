#!/usr/bin/env bash
# SPLIT_COLUMNS (skinny_linear; the folded radial layer's column-split input gradient) on the RTX 4000 Ada:
# MACE-OFF models on water_216, same-process MD-step A/B both orders (warm-up 500, 16 blocks of 1,000 steps, as
# off-polar-alex), b = SPLIT_COLUMNS True.  HOST=ada ROOT=<dir> bash off.sh SRC_DIR MODEL:CKPT...
set -u -o pipefail
FM=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
SC=fastmace.kernels.triton.skinny_linear:SPLIT_COLUMNS
W=$STRUCT/Structures/Organic/water_216.extxyz
[ -f "$W" ] || W=$(find "$STRUCT" -name water_216.extxyz | head -1)
note "off start fastmace=$(cat .source_commit) water=$W"
for job in "$@"; do
  model=${job%%:*} ck=$MODELS/${job#*:}
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $model $order: $why"; exit 3; fi
    o=$ROOT/off; mkdir -p "$o"
    note "begin $model $order"; t0=$(date +%s)
    python tools/ab_rewrites.py --output "$o/ab-$model-$order" --checkpoint "$ck" --model "$model" \
      --structures "$W@1x1x1" --warmup 500 --block 1000 --blocks 16 --order "$order" --pattern "$pattern" \
      --a-constant $SC=False --b-constant $SC=True > "$o/ab-$model-$order.log" 2>&1 < /dev/null
    note "end $model $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$o/ab-$model-$order.log" | sed "s/^/  $model $order /" | tee -a "$log"
  done
done
note DONE
