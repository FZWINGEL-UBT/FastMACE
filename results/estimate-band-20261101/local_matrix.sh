#!/usr/bin/env bash
# The rescue matrix (est_matrix.sh's stages) on the local RTX 5060 Ti 16 GB (WSL2), through
# results/capacity-16gb-20261029/local.sh (its idle check before every point), inside a reservation:
#   python3 tools/ci/local_gpu.py run --purpose "..." -- env FM=<pinned source copy> \
#     bash results/estimate-band-20261101/local_matrix.sh ROOT STRUCTURE REPEAT...
# A stage whose point fails at one size skips the larger ones.
set -u
ROOT=$1 STRUCTURE=$2; shift 2
FM=${FM:?set FM to a pinned source copy}
HERE=$(dirname "$0")
BASE="wprime_backward=on"
declare -A STAGES=(
  [s0-default]="recompute_conv_weights=off recompute_first_table=off recompute_node_segments=off recompute_node_region=off gate_epilogue=on"
  [s1-conv]="recompute_conv_weights=on recompute_first_table=off recompute_node_segments=off recompute_node_region=off gate_epilogue=on"
  [s2-table]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=off recompute_node_region=off gate_epilogue=on"
  [s3-segments]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=off gate_epilogue=on"
  [s4-geoff]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=off gate_epilogue=off"
  [s5-region]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=on gate_epilogue=off"
  [s6-region-ge]="recompute_conv_weights=on recompute_first_table=on recompute_node_segments=on recompute_node_region=on gate_epilogue=on"
)
for stage in s6-region-ge s5-region s4-geoff s3-segments s2-table s1-conv s0-default; do
  for repeat in "$@"; do
    TAG=$stage REWRITES="$BASE ${STAGES[$stage]}" bash "$HERE/../capacity-16gb-20261029/local.sh" "$ROOT" \
      "point:$STRUCTURE:float32:$repeat"
    status=$?
    [ $status -eq 3 ] && exit 3
    last=$(grep "point $STRUCTURE float32" "$ROOT/jobs.log" | tail -n 1)
    case "$last" in *": ok,"*) ;; *) echo "$(date -u +%FT%TZ) stop $stage at $repeat" >> "$ROOT/run.log"; break ;; esac
  done
done
echo "$(date -u +%FT%TZ) DONE" >> "$ROOT/run.log"
