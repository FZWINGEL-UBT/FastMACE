#!/usr/bin/env bash
# POLAR-1 cells of defaults-by-gpu.md that were never measured on these GPUs; same-process MD-step A/Bs, both build
# orders, NaCl and NbCl5 (tools/ab_rewrites.py; warm-up 300, 8 blocks of 200 steps, as polar-reuse).
#   HOST=ada|local ROOT=<dir> bash polar.sh SRC_DIR SET...
# 5060 Ti: pair-M, pair-L (wprime_backward + fused_folded_forward, POLAR_DECISION_66_WPRIME_GPUS), epi-S, epi-M, epi-L
#   (gate_epilogue, POLAR_DECISION_66_GATE_EPILOGUE_GPUS).
# Ada: mk-S, mk-M, mk-L (longrange.polar MEGAKERNEL, the k-space field call as one CUDA megakernel; NbCl5 only, NaCl is
#   under MEGAKERNEL_MIN_WORK).
set -u -o pipefail
FM=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
MK=fastmace.longrange.polar:MEGAKERNEL
note "polar start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for set in "$@"; do
  size=${set##*-}; model=polar1-$(echo "$size" | tr SML sml); ck=$MODELS/MACE-POLAR-1-$size.model; structs=(NaCl NbCl5)
  case $set in
    pair-*) SIDE=(--b wprime_backward=on fused_folded_forward=on) ;;
    epi-*) SIDE=(--b gate_epilogue=on) ;;
    mk-*) SIDE=(--a-constant $MK=False --b-constant $MK=True); structs=(NbCl5) ;;
    *) note "unknown set $set"; exit 2 ;;
  esac
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $set $order: $why"; exit 3; fi
    o=$ROOT/polar; mkdir -p "$o"
    note "begin $set $order"; t0=$(date +%s)
    python tools/ab_rewrites.py --output "$o/ab-$set-$order" --checkpoint "$ck" --model "$model" \
      --structures "${structs[@]}" --warmup 300 --block 200 --blocks 8 --order "$order" --pattern "$pattern" \
      "${SIDE[@]}" > "$o/ab-$set-$order.log" 2>&1 < /dev/null
    note "end $set $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$o/ab-$set-$order.log" | sed "s/^/  $set $order /" | tee -a "$log"
  done
done
note DONE
