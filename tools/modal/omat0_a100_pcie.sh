#!/usr/bin/env bash
# The A100 80GB PCIe on Benchmark_loup (4 November 2026). Modal's A100-80GB can be either card; the recheck job got
# the PCIe one, where decision 66 lists only the A100-SXM4-80GB for tf32x3_gemms, the generic wprime backward with
# the fused folded forward (chunked_conv_backward their carrier) and the skipped zero column: 4.61 us/atom/step there
# against 3.17 on the SXM4 card. MargrafGroup's Benchmark_loup ran on the PCIe card. This job measures those switches
# there, same-process MD step A/Bs (omat0_a100_launch_screen.sh conventions), then the benchmark protocol.
# Exits 3 at once on any other GPU (rerun until Modal assigns the PCIe card).
# 1. pcie-all in both orders: b = tf32x3_gemms, wprime_backward, fused_folded_forward, chunked_conv_backward on and
#    SKIP_ZERO_COLUMN; then single parts in one order: pcie-tf32x3, pcie-wprime (the wprime set), pcie-skip (the
#    skip on top of the wprime set on both sides).
# 2. tools/bench/loup_md.py (5 warm-up and 20 timed steps, 3 processes, NVE and Langevin) with pcie-all.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_pcie.sh" --out ~/fm-runs/omat0-a100-pcie
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100 80GB PCIe" ] || { echo "not the PCIe card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
zc=fastmace.kernels.oeq_streaming.fused_forward:SKIP_ZERO_COLUMN=True
wps=(wprime_backward=on fused_folded_forward=on chunked_conv_backward=on)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit)"
ab() {  # name orders blocks b-arguments...
  local name=$1 orders=$2 blocks=$3; shift 3
  for order in $orders; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$name-$order"
    python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
      --blocks "$blocks" --order "$order" --pattern "$pattern" "$@" > "$out/ab-$name-$order.log" 2>&1 < /dev/null
    note "end ab-$name-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-$name-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
}
ab pcie-all "ab ba" 8 --b tf32x3_gemms=on "${wps[@]}" --b-constant "$zc"
ab pcie-tf32x3 ab 6 --b tf32x3_gemms=on
ab pcie-wprime ab 6 --b "${wps[@]}"
ab pcie-skip ab 6 --a "${wps[@]}" --b "${wps[@]}" --b-constant "$zc"
for ens in nve langevin; do
  for t in 1 2 3; do
    note "start loup-pcie-all-$ens-$t"
    python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" \
      --rewrites tf32x3_gemms=on "${wps[@]}" --constant "$zc" \
      --output "$out/loup/pcie-all-$ens-$t.json" > "$out/loup-pcie-all-$ens-$t.log" 2>&1 < /dev/null
    note "end loup-pcie-all-$ens-$t exit=$? $(tail -n 1 "$out/loup-pcie-all-$ens-$t.log" | cut -c1-300)"
  done
done
note done
grep -h -E "end (ab|loup)" "$log" | cut -c1-260
