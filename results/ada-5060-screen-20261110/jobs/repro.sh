#!/usr/bin/env bash
# The X4 illegal memory access (murphy, 10 Nov 22:3x UTC): single-engine Benchmark_loup runs (tools/bench/loup_md.py,
# SrTiO3 10x10x10, Langevin, 3,000 steps) with CUDA_LAUNCH_BLOCKING=1, the stage-2 set C and C with one switch off;
# each config in its own process. Exit code and the first error lines per config go to run.log.
#   HOST=ada ROOT=<dir> bash repro.sh SRC_DIR NAME...   (names below)
set -u -o pipefail
FM=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
P=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD R=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
C="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $w:BUTTERFLY=True $s:BLOCK_ORDER=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $P=True $R=True $e:GRAPH_SUM=True $w:SCALAR_PATHS=True"
for name in "$@"; do
  case $name in
    C) X="" ;; no-power) X="$P=False" ;; no-rsum) X="$R=False" ;; no-gsum) X="$e:GRAPH_SUM=False" ;;
    no-scalar) X="$w:SCALAR_PATHS=False" ;; *) X="${!name}" ;;
  esac
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name ${STEPS:-3000} steps launch_blocking=${CLB:-1}"
  # shellcheck disable=SC2086
  CUDA_LAUNCH_BLOCKING=${CLB:-1} ${WRAP:-} python tools/bench/loup_md.py --structure "$ST@${CELL:-10x10x10}" --checkpoint "$CK" \
    --ensemble langevin --warmup 5 --steps ${STEPS:-3000} --constant $C $X --output "$ROOT/$name" > "$ROOT/$name.log" 2>&1 < /dev/null
  rc=$?
  note "end $name exit=$rc $(grep -m3 -hE 'illegal|Error|error:|ERROR SUMMARY|Invalid' "$ROOT/$name.log" | tr '\n' ' ' | cut -c1-400)"
done
note DONE
