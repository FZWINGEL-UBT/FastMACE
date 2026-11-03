#!/usr/bin/env bash
# OMAT-0 medium launch-setting screen on the A100 (owner, 3 November 2026: "Yes, screen them"). The A100 profile
# (results/omat0-a100-profile-20261103) put 65 % of the Benchmark_loup force call (SrTiO3 5,000 atoms) in two
# kernels whose launch rules were tuned on MH-1: the fused forward (6.9 ms) and the wprime backward (5.1 ms).
# Same-process MD step A/Bs, default plan on both sides, b with one module constant (or switch) changed:
# - fused_forward: BLOCKS_PER_SM 1 and 4 (default 2), RECEIVERS_PER_CHUNK 1 and 4 (default 2), W_TILE_REGISTERS
#   16 / 24 / 80 (parts 5 / 4 / 1; default 40: parts 2);
# - runtime.NUM_THREADS 64 / 256 / 512 (default 128): the streaming kernels' block size, the wprime backward's grid;
# - recompute_conv_weights=on: w rebuilt in the backward instead of stored (1.9 GB written and read per call).
# Kernel-lab follow-ups (3 November 2026, tools/omat0_kernel_lab.py), all bitwise or certified:
# - slopes2 / slopes4: wprime_backward.SLOPE_VECTOR (q-major derivative table, float2/float4 term loads);
# - streamw: wprime_backward.STREAM_W (the stored w read with __ldcs); sympairs: radial.SYMMETRIC_PAIRS;
# - degN: radial.DEGREE N (default 10; fewer terms per edge, more pieces, the same fit tolerance and certificate);
# - eos-fcN / eos-beN: edge_outer_sum FORWARD_CHUNK / BACKWARD_BLOCK_EDGES (default 32);
# - skipzero: fused_forward.SKIP_ZERO_COLUMN (the k-loop skips the folded weight's all-zero bias column);
# - stcs: fused_forward.STREAM_STORE_W (w stored with __stcs); bundle-all: degree 6, slopes2, skipzero and stcs;
# - dN-slopes2, dN-slopes4, dN-streamw, dN-skipzero, dN-stcs: the switch on top of degree N (a: degree N too);
#   bundle: degree 6 and slopes2; bundle-skipzero: and skipzero; allN: degree N, slopes2, skipzero and stcs; allnsN:
#   the same without slopes2 (streamw is 12.7 % slower in the RTX 4000 Ada lab, 3.4 % faster on the RTX 5060 Ti:
#   screened alone).
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
wp=fastmace.kernels.oeq_streaming.wprime_backward
zc=fastmace.kernels.oeq_streaming.fused_forward:SKIP_ZERO_COLUMN=True
ffs=fastmace.kernels.oeq_streaming.fused_forward:STREAM_STORE_W=True
rd=fastmace.rewrites.radial
eos=fastmace.kernels.triton.edge_outer_sum
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
note "$(hostname) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
for spec in ${SCREEN:-ff-bps1 ff-bps4 ff-rpc1 ff-rpc4 ff-wtr16 ff-wtr24 ff-wtr80 rt-thr64 rt-thr256 rt-thr512 recompute}; do
  case $spec in
    ff-bps*) ab "$spec" --b-constant "$ff:BLOCKS_PER_SM=${spec#ff-bps}" ;;
    ff-rpc*) ab "$spec" --b-constant "$ff:RECEIVERS_PER_CHUNK=${spec#ff-rpc}" ;;
    ff-wtr*) ab "$spec" --b-constant "$ff:W_TILE_REGISTERS=${spec#ff-wtr}" ;;
    ff-batch*) ab "$spec" --b-constant "$ff:BATCH=${spec#ff-batch}" ;;  # BATCH 4: all 10 OMAT-0 paths in one part
    rt-thr*) ab "$spec" --b-constant "$rt:NUM_THREADS=${spec#rt-thr}" ;;
    recompute) ab "$spec" --b recompute_conv_weights=on ;;
    strided) ab "$spec" --a-constant fastmace.kernels.triton.edge_outer_sum:STRIDED_ROWS=False ;;  # b: in place
    fullrows) ab "$spec" --b-constant fastmace.rewrites.generic.modules:FIRST_FULL_ROWS=True ;;
    colblock64) ab "$spec" --b-constant fastmace.kernels.triton.radial_table:COLUMN_BLOCK=64 ;;
    deg*) ab "$spec" --b-constant "$rd:DEGREE=${spec#deg}" ;;  # fewer terms, more pieces (same tolerance)
    slopes*) ab "$spec" --b-constant "$wp:SLOPE_VECTOR=${spec#slopes}" ;;
    streamw) ab "$spec" --b-constant "$wp:STREAM_W=True" ;;
    sympairs) ab "$spec" --b-constant "$rd:SYMMETRIC_PAIRS=True" ;;
    eos-fc*) ab "$spec" --b-constant "$eos:FORWARD_CHUNK=${spec#eos-fc}" ;;
    eos-be*) ab "$spec" --b-constant "$eos:BACKWARD_BLOCK_EDGES=${spec#eos-be}" ;;
    d[0-9]*-*) n=${spec%%-*}; n=${n#d}  # dN-switch: the switch on top of degree N (a: degree N too)
      case ${spec#*-} in
        slopes2) sw="$wp:SLOPE_VECTOR=2" ;; slopes4) sw="$wp:SLOPE_VECTOR=4" ;; streamw) sw="$wp:STREAM_W=True" ;;
        skipzero) sw=$zc ;; stcs) sw=$ffs ;; *) note "unknown case $spec"; continue ;;
      esac
      ab "$spec" --a-constant "$rd:DEGREE=$n" --b-constant "$rd:DEGREE=$n" "$sw" ;;
    bundle) ab "$spec" --b-constant "$rd:DEGREE=6" "$wp:SLOPE_VECTOR=2" ;;
    skipzero) ab "$spec" --b-constant "$zc" ;;
    bundle-skipzero) ab "$spec" --b-constant "$rd:DEGREE=6" "$wp:SLOPE_VECTOR=2" "$zc" ;;
    stcs) ab "$spec" --b-constant "$ffs" ;;
    bundle-all) ab "$spec" --b-constant "$rd:DEGREE=6" "$wp:SLOPE_VECTOR=2" "$zc" "$ffs" ;;
    all[0-9]*) ab "$spec" --b-constant "$rd:DEGREE=${spec#all}" "$wp:SLOPE_VECTOR=2" "$zc" "$ffs" ;;  # all at degree N
    allns[0-9]*) ab "$spec" --b-constant "$rd:DEGREE=${spec#allns}" "$zc" "$ffs" ;;  # the same without slopes2
    dz[0-9]*) ab "$spec" --b-constant "$rd:DEGREE=${spec#dz}" "$zc" ;;  # degree N and skipzero
  esac
done
note done
grep -h -E "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-200
