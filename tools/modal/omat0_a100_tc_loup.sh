#!/usr/bin/env bash
# Benchmark_loup on the A100-SXM4-80GB with the fused forward's a.W product on tensor cores (4 November 2026):
# the default plan, tc (fused_forward.TENSOR_CORES, TC_SPLIT round) and tcmask (TC_SPLIT mask), each under the
# benchmark's protocol (tools/bench/loup_md.py: 5 warm-up and 20 timed steps), NVE and Langevin, 2 fresh processes
# each, alternating the sets. Exits 3 on any other card (Modal's A100-80GB can be the PCIe card).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_tc_loup.sh" --out ~/fm-runs/omat0-a100-tc-loup
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
ff=fastmace.kernels.oeq_streaming.fused_forward
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)"
mkdir -p "$out/loup"
for t in 1 2; do
  for ens in nve langevin; do
    for set in default tc tcmask; do
      opt=()
      [ "$set" = tc ] && opt=(--constant "$ff:TENSOR_CORES=True")
      [ "$set" = tcmask ] && opt=(--constant "$ff:TENSOR_CORES=True" "$ff:TC_SPLIT='mask'")
      note "start loup-$set-$ens-$t"
      python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" "${opt[@]}" \
        --output "$out/loup/$set-$ens-$t.json" > "$out/loup-$set-$ens-$t.log" 2>&1 < /dev/null
      note "end loup-$set-$ens-$t exit=$? $(tail -n 1 "$out/loup-$set-$ens-$t.log" | cut -c1-300)"
    done
  done
done
note done
grep -h -E "end loup" "$log" | cut -c1-260
