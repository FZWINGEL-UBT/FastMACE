#!/usr/bin/env bash
# A100-SXM4-80GB: the factored TP backward with each path's coefficient magnitude applied once
# (wprime_backward.FACTORED_SCALE), 4 November 2026. The wprime GPU test, the kernel lab (base, factored,
# factored-scaled, the channel-pair kernel), then the Benchmark_loup MD step A/B in both orders, 8 blocks:
# a = FACTORED, b = FACTORED + FACTORED_SCALE. Exits 3 at once on any other GPU.
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/tp_backward_scale_a100.sh" --out ~/fm-runs/tp-backward-scale-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
wp=fastmace.kernels.oeq_streaming.wprime_backward
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
python -m pytest -q -p no:cacheprovider -s tests/gpu/test_wprime_backward_cuda.py > "$out/gpu-wprime.log" 2>&1 < /dev/null
note "gpu-wprime exit=$? $(tail -n 1 "$out/gpu-wprime.log")"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime \
  --variants base factored factored-scaled pair --rounds 21 --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
note "lab exit=$? $(grep -h -E "^wprime " "$out/lab.log" | cut -c1-90 | tr '\n' ';')"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output "$out/ab-scale-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
    --blocks 8 --order "$order" --pattern "$pattern" --a-constant "$wp:FACTORED=True" \
    --b-constant "$wp:FACTORED=True" "$wp:FACTORED_SCALE=True" > "$out/ab-scale-$order.log" 2>&1 < /dev/null
  note "ab-scale-$order exit=$? $(grep -h b/a "$out/ab-scale-$order.log" | tail -n 1)"
done
note done
