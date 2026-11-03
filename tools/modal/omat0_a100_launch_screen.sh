#!/usr/bin/env bash
# OMAT-0 medium launch-setting screen on the A100 (owner, 3 November 2026: "Yes, screen them"). The A100 profile
# (results/omat0-a100-profile-20261103) put 65 % of the Benchmark_loup force call (SrTiO3 5,000 atoms) in two
# kernels whose launch rules were tuned on MH-1: the fused forward (6.9 ms) and the wprime backward (5.1 ms).
# Same-process MD step A/Bs, default plan on both sides, b with one module constant (or switch) changed:
# - fused_forward: BLOCKS_PER_SM 1 and 4 (default 2), RECEIVERS_PER_CHUNK 1 and 4 (default 2), W_TILE_REGISTERS
#   16 / 24 / 80 (parts 5 / 4 / 1; default 40: parts 2);
# - runtime.NUM_THREADS 64 / 256 / 512 (default 128): the streaming kernels' block size, the wprime backward's grid;
# - recompute_conv_weights=on: w rebuilt in the backward instead of stored (1.9 GB written and read per call).
# Screen pass: one order (abab), 6 blocks of 200 steps; SCREEN_BOTH=1 runs both orders, 8 blocks (winners).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_launch_screen.sh" --out ~/fm-runs/omat0-a100-launch-screen
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
ff=fastmace.kernels.oeq_streaming.fused_forward
rt=fastmace.kernels.oeq_streaming.runtime
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$? $(grep -h -E "b/a|Error" "$out/$name.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
}
ab() {  # name b-arguments...
  local name=$1; shift
  local orders=ab blocks=6
  [ "${SCREEN_BOTH:-}" = 1 ] && orders="ab ba" && blocks=8
  for order in $orders; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" \
      --model omat0-medium --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 200 --blocks "$blocks" --order "$order" --pattern "$pattern" "$@"
  done
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
for spec in ${SCREEN:-ff-bps1 ff-bps4 ff-rpc1 ff-rpc4 ff-wtr16 ff-wtr24 ff-wtr80 rt-thr64 rt-thr256 rt-thr512 recompute}; do
  case $spec in
    ff-bps*) ab "$spec" --b-constant "$ff:BLOCKS_PER_SM=${spec#ff-bps}" ;;
    ff-rpc*) ab "$spec" --b-constant "$ff:RECEIVERS_PER_CHUNK=${spec#ff-rpc}" ;;
    ff-wtr*) ab "$spec" --b-constant "$ff:W_TILE_REGISTERS=${spec#ff-wtr}" ;;
    rt-thr*) ab "$spec" --b-constant "$rt:NUM_THREADS=${spec#rt-thr}" ;;
    recompute) ab "$spec" --b recompute_conv_weights=on ;;
  esac
done
note done
grep -h -E "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-200
