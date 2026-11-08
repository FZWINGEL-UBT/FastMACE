#!/usr/bin/env bash
# Same-process MD-step A/B packs of OMAT-0 medium on the RTX 4000 Ada (tools/ab_rewrites.py --pack), both build
# orders, Benchmark_loup settings (SrTiO3 10x10x10, 1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side).
# Packs of at most four entries: order ba holds every entry's engine and side a on the 20 GB card at once.
#   ROOT=<dir> bash ab.sh SRC_DIR PACKNAME...
# pack1a-c (first pass, superseded): the A100's ten + JOINED_LAYOUT without the Ada's prerequisites (inert: README).
# s1a-c: stage 1 against the Ada defaults, with HORNER and tf32x3_gemms where the groups need them.
# n2a-b: leave one out on N = the spline group + the readout tail (side a = N, side b = N with one switch off).
# a2a-c: leave one out on A = N + tf32x3_gemms and the compose group.
set -u -o pipefail
FM=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
note "start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for pk in "$@"; do
  AC=""
  case $pk in
    pack1a) PACK="spw|$s:SPLINE_W=True;spr|$s:SPLINE_W=True $s:R128=True;cskip|$c:COMPOSE_SKIP=True" ;;
    pack1b) PACK="cueq|$q:CUEQ_PROJECTED=True;tail|$t:GENERIC_FOLD_READOUT_TAIL=True;all|$ALL" ;;
    pack2a) AC="--a-constant $ALL --b-constant $ALL"
            PACK="no-r128|$s:R128=False $s:STAGE_EDGES=False;no-se|$s:STAGE_EDGES=False;no-vec|$w:SPLINE_VECTOR=2;no-ssh|$w:STAGE_SH=False" ;;
    pack2c) AC="--a-constant $ALL --b-constant $ALL"
            PACK="no-joined|$c:JOINED_LAYOUT=False" ;;
    pack2b) AC="--a-constant $ALL --b-constant $ALL"
            PACK="no-cfast|$c:COMPOSE_FAST=False;no-cplan|$c:COUNT_PLAN=False;no-cueq|$q:CUEQ_PROJECTED=False;no-tail|$t:GENERIC_FOLD_READOUT_TAIL=False" ;;
    s1a) PACK="hsp|$H $s:SPLINE_W=True;hspr|$H $s:SPLINE_W=True $s:R128=True;spline|$SPL" ;;
    s1b) PACK="t3|$T3;t3c|$CMP;t3ck|$CMP $KC;tail|$TAIL" ;;
    s1c) PACK="all|$SPL $CMP $TAIL;allk|$SPL $CMP $TAIL $KC;nt3|$SPL $TAIL" ;;
    n2a) AC="--a-constant $SPL $TAIL --b-constant $SPL $TAIL"
         PACK="no-spw|$s:SPLINE_W=False $w:HORNER=False;no-r128|$s:R128=False $s:STAGE_EDGES=False;no-se|$s:STAGE_EDGES=False" ;;
    n2b) AC="--a-constant $SPL $TAIL --b-constant $SPL $TAIL"
         PACK="no-vec|$w:SPLINE_VECTOR=2;no-ssh|$w:STAGE_SH=False;no-tail|$t:GENERIC_FOLD_READOUT_TAIL=False" ;;
    a2a) AC="--a $T3 --b $T3 --a-constant ${SPL} ${CMP#"$T3 "} $TAIL --b-constant ${SPL} ${CMP#"$T3 "} $TAIL"
         PACK="no-t3|tf32x3_gemms=off;no-cskip|$c:COMPOSE_SKIP=False;no-cfast|$c:COMPOSE_FAST=False;no-cplan|$c:COUNT_PLAN=False" ;;
    a2b) AC="--a $T3 --b $T3 --a-constant ${SPL} ${CMP#"$T3 "} $TAIL --b-constant ${SPL} ${CMP#"$T3 "} $TAIL"
         PACK="no-joined|$c:JOINED_LAYOUT=False;no-cueq|$q:CUEQ_PROJECTED=False;no-tail|$t:GENERIC_FOLD_READOUT_TAIL=False;no-spw|$s:SPLINE_W=False $w:HORNER=False" ;;
    a2c) AC="--a $T3 --b $T3 --a-constant ${SPL} ${CMP#"$T3 "} $TAIL --b-constant ${SPL} ${CMP#"$T3 "} $TAIL"
         PACK="no-r128|$s:R128=False $s:STAGE_EDGES=False;no-se|$s:STAGE_EDGES=False;no-vec|$w:SPLINE_VECTOR=2;no-ssh|$w:STAGE_SH=False" ;;
    *) note "unknown pack $pk"; exit 2 ;;
  esac
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $pk $order: $why"; exit 3; fi
    o=$ROOT/$pk; mkdir -p "$o"
    note "begin $pk $order"; t0=$(date +%s)
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$o/ab-$order" --checkpoint "$CK" --model omat0-medium \
      --structures "$ST@${CELL:-10x10x10}" --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
      --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 $AC --pack "$PACK" > "$o/ab-$order.log" 2>&1 < /dev/null
    note "end $pk $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$o/ab-$order.log" | sed "s/^/  $pk $order /" | tee -a "$log"
  done
done
note DONE
