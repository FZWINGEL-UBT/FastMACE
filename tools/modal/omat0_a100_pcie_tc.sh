#!/usr/bin/env bash
# The tensor-core forward on the A100 80GB PCIe (4 November 2026; MargrafGroup's Benchmark_loup ran on that card).
# On top of the SXM4's decision-66 switches (tf32x3_gemms, the generic wprime set, SKIP_ZERO_COLUMN; owner decision
# for the PCIe card pending), with fused_forward.TENSOR_CORES and TC_SPLIT round (tc) or mask (tcmask):
# 1. MD step A/B tcmask against the switches alone, both orders, 8 blocks (omat0_a100_pcie.sh conventions);
# 2. tools/bench/loup_md.py (5 warm-up and 20 timed steps), NVE and Langevin, 2 processes each, alternating
#    switches / tc / tcmask.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_pcie_tc.sh" --out ~/fm-runs/omat0-a100-pcie-tc
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100 80GB PCIe" ] || { echo "not the PCIe card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
ff=fastmace.kernels.oeq_streaming.fused_forward
zc=$ff:SKIP_ZERO_COLUMN=True
sw=(tf32x3_gemms=on wprime_backward=on fused_folded_forward=on chunked_conv_backward=on)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-tcmask-$order"
  python tools/ab_rewrites.py --output "$out/ab-tcmask-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
    --blocks 8 --order "$order" --pattern "$pattern" --a "${sw[@]}" --b "${sw[@]}" --a-constant "$zc" \
    --b-constant "$zc" "$ff:TENSOR_CORES=True" "$ff:TC_SPLIT='mask'" > "$out/ab-tcmask-$order.log" 2>&1 < /dev/null
  note "end ab-tcmask-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-tcmask-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
done
mkdir -p "$out/loup"
for t in 1 2; do
  for ens in nve langevin; do
    for set in switches tc tcmask; do
      c=("$zc")
      [ "$set" = tc ] && c+=("$ff:TENSOR_CORES=True")
      [ "$set" = tcmask ] && c+=("$ff:TENSOR_CORES=True" "$ff:TC_SPLIT='mask'")
      note "start loup-$set-$ens-$t"
      python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" --rewrites "${sw[@]}" \
        --constant "${c[@]}" --output "$out/loup/$set-$ens-$t.json" > "$out/loup-$set-$ens-$t.log" 2>&1 < /dev/null
      note "end loup-$set-$ens-$t exit=$? $(tail -n 1 "$out/loup-$set-$ens-$t.log" | cut -c1-300)"
    done
  done
done
note done
grep -h -E "end (ab|loup)" "$log" | cut -c1-260
