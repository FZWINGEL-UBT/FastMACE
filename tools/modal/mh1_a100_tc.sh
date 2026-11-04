#!/usr/bin/env bash
# MH-1 with the tensor-core fused forward on the A100-SXM4-80GB (4 November 2026). The owner made
# fused_forward.TENSOR_CORES (TC_SPLIT mask) the A100 default after OMAT-0's Benchmark_loup; the switch sits in the
# kernel MH-1 uses too, so this measures MH-1: MD step A/B, default plan against TENSOR_CORES with the mask split,
# both build orders, 8 blocks, on NaCl, NaNbO3 and NbCl5. Exits 3 on any other card.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/mh1_a100_tc.sh" --out ~/fm-runs/mh1-a100-tc
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
mh1=$FASTMACE_MODELS/mace-mh-1.model
ff=fastmace.kernels.oeq_streaming.fused_forward
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-mh1-tcmask-$order"
  python tools/ab_rewrites.py --output "$out/ab-mh1-tcmask-$order" --checkpoint "$mh1" --model mh1-r2scan \
    --structures NaCl NaNbO3 NbCl5 --timestep-fs 1 --warmup 200 --block 200 --blocks 8 --order "$order" \
    --pattern "$pattern" --a-constant "$ff:TENSOR_CORES=False" \
    --b-constant "$ff:TENSOR_CORES=True" "$ff:TC_SPLIT='mask'" > "$out/ab-mh1-tcmask-$order.log" 2>&1 < /dev/null
  note "end ab-mh1-tcmask-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-mh1-tcmask-$order.log" | tr '\n' ' ' | cut -c1-500)"
done
note done
grep -h "end ab" "$log"
