#!/usr/bin/env bash
# Same-process MD-step A/B packs of OMAT-0 medium (tools/ab_rewrites.py --pack), both build orders, Benchmark_loup
# settings (1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side), SrTiO3 10x10x10 (5,000 atoms) and NbCl5
# (1,944 atoms). b = the entry on top of side a (today's defaults unless the pack sets a base). Packs of at most four.
#   HOST=ada|local ROOT=<dir> bash ab2.sh SRC_DIR PACKNAME...
# ab2.sh = ab.sh plus packs L7-L9 and A4 (added after docs-refresh's open list; a copy, so running jobs keep their file).
# L* packs: RTX 5060 Ti; A* packs: RTX 4000 Ada (inside common.sh's idle check).
set -u -o pipefail
FM=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
SPL="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True"   # the 5060 Ti's spline stack (HORNER is on there)
CMP="$c:COMPOSE_SKIP=True $c:COMPOSE_FAST=True"           # with tf32x3_gemms=on
note "start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for pk in "$@"; do
  AC=""
  case $pk in
    L1) PACK="spl|$SPL;t3c|$T3 $CMP;splt3c|$SPL $T3 $CMP;tail|$t:GENERIC_FOLD_READOUT_TAIL=True" ;;
    L2) PACK="t3|$T3;fold|fold_product_linear=on;sbl|$k:SORT_BY_LENGTH=True;gsum|$e:GRAPH_SUM=True" ;;
    L3) PACK="bfly|$w:BUTTERFLY=True $w:REDUX=False;dparts|$f:DEGREE_PARTS=True;power|fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD=True;rsum|fastmace.kernels.triton.receiver_sum:RECEIVER_SUM=True" ;;
    L4) PACK="ssh|$w:STAGE_SH=True;sdout|$w:STAGE_DOUT=True;falpha|$w:FOLD_ALPHA=True" ;;
    L5) AC="--a-constant $SPL --b-constant $SPL"
        PACK="p8|$s:PACKED8=True;bo|$s:BLOCK_ORDER=True;pf|$s:PREFETCH_AHEAD=2;sbl|$k:SORT_BY_LENGTH=True" ;;
    L6) AC="--a $T3 --b $T3 --a-constant $SPL $CMP --b-constant $SPL $CMP"
        PACK="scg|$g:SC_GENERATED=True;cplan|$c:COUNT_PLAN=True;cueq|$q:CUEQ_PROJECTED=True;tail|$t:GENERIC_FOLD_READOUT_TAIL=True" ;;
    A1) PACK="p8|$s:PACKED8=True;sbl|$k:SORT_BY_LENGTH=True;scg|$g:SC_GENERATED=True;bfly|$w:BUTTERFLY=True" ;;
    A2) PACK="bo|$s:BLOCK_ORDER=True;pf|$s:PREFETCH_AHEAD=2;sdout|$w:STAGE_DOUT=True;falpha|$w:FOLD_ALPHA=True" ;;
    A3) PACK="power|fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD=True;rsum|fastmace.kernels.triton.receiver_sum:RECEIVER_SUM=True;gsum|$e:GRAPH_SUM=True;dparts|$f:DEGREE_PARTS=True" ;;
    L7) PACK="sym|fastmace.rewrites.radial:SYMMETRIC_PAIRS=True;scalar|$w:SCALAR_PATHS=True;warps8|fastmace.kernels.triton.radial_table:WARPS=8" ;;
    L8) AC="--a-constant $SPL --b-constant $SPL"
        PACK="vec4|$w:SPLINE_VECTOR=4;wse|$w:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True" ;;
    L9) AC="--a $T3 --b $T3 --a-constant $SPL $CMP --b-constant $SPL $CMP"
        PACK="joined|$c:JOINED_LAYOUT=True" ;;
    A4) PACK="scalar|$w:SCALAR_PATHS=True;mr80|$w:MAX_REGISTERS=80;warps8|fastmace.kernels.triton.radial_table:WARPS=8;wse|$w:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True" ;;
    X*) eval "PACK=\"\$PACK_$pk\"; AC=\"\${AC_$pk:-}\"" ;;   # ad hoc packs from the environment (stage 2)
    *) note "unknown pack $pk"; exit 2 ;;
  esac
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $pk $order: $why"; exit 3; fi
    o=$ROOT/$pk; mkdir -p "$o"
    note "begin $pk $order"; t0=$(date +%s)
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$o/ab-$order" --checkpoint "$CK" --model omat0-medium \
      --structures "$ST@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
      --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 $AC --pack "$PACK" > "$o/ab-$order.log" 2>&1 < /dev/null
    note "end $pk $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$o/ab-$order.log" | sed "s/^/  $pk $order /" | tee -a "$log"
  done
done
note DONE
