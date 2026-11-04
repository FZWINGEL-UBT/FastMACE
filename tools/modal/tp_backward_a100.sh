#!/usr/bin/env bash
# A100-SXM4-80GB: the factored tensor-product backward (wprime_backward.FACTORED) and mirror species pairs
# (radial.SYMMETRIC_PAIRS), 4 November 2026. The kernel lab on the default plan's wprime kernel (base, factored,
# factored + unordered pairs, the channel-pair kernel, the one-slab and weight-gradient-only bounds), then the
# Benchmark_loup MD step A/B in both orders, 8 blocks: b = FACTORED, b = FACTORED + SYMMETRIC_PAIRS.
# Exits 3 at once on any other GPU.
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 45 --commit <sha> \
#     --command "bash tools/modal/tp_backward_a100.sh" --out ~/fm-runs/tp-backward-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
wp=fastmace.kernels.oeq_streaming.wprime_backward
rd=fastmace.rewrites.radial
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime \
  --variants base factored factored+unordered unordered pair FF-one-slab F-tp-wg-only --rounds 21 \
  --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
note "lab exit=$? $(grep -h -E "^wprime " "$out/lab.log" | cut -c1-90 | tr '\n' ';')"
ab() {  # name b-constants...
  local name=$1; shift
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
      --blocks 8 --order "$order" --pattern "$pattern" --b-constant "$@" > "$out/ab-$name-$order.log" 2>&1 < /dev/null
    note "ab-$name-$order exit=$? $(grep -h b/a "$out/ab-$name-$order.log" | tail -n 1)"
  done
}
ab factored "$wp:FACTORED=True"
ab factored-sym "$wp:FACTORED=True" "$rd:SYMMETRIC_PAIRS=True"
note done
