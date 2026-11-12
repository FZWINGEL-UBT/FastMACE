#!/usr/bin/env bash
# packed8-auto (12 November 2026): PACKED8 at the MD step on cold structures, where the census
# (slab_census.json) says the SPLINE_W forward touches few table slabs. Same-process A/B (tools/ab_rewrites.py),
# both build orders, OMAT-0 medium, Benchmark_loup settings (1 fs, skin 0.5, warm-up 200, blocks of 256 steps),
# Langevin at TEMPERATURE_K from the ideal lattice. Side a = today's defaults (PACKED8 on for PACKED8_GPUS),
# side b = PACKED8=False, so b/a < 1 means the float4 table is faster (an auto-choice would pay there).
# Run from a source root (PYTHONPATH set, FASTMACE_MODELS / FASTMACE_STRUCTURES exported):
#   OUT=<dir> [TEMPS="1 50 300"] [STRUCTURES=...] [BLOCKS=8] [PY=python] bash results/packed8-auto-20261112/cold_ab.sh
set -uo pipefail
out=${OUT:?set OUT}
py=${PY:-python}
mkdir -p "$out"
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
b="fastmace.kernels.oeq_streaming.spline_forward:PACKED8=False"
note "$(hostname) $gpu; source $(cat .source_commit 2>/dev/null || git rev-parse --short HEAD 2>/dev/null); b: $b"
for temp in ${TEMPS:-1 50 300}; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    o=$out/ab-${temp}K-$order
    note "start ${temp}K $order"
    # shellcheck disable=SC2086
    "$py" tools/ab_rewrites.py --output "$o" --checkpoint "$ck" --model omat0-medium \
      --structures ${STRUCTURES:-"$st@10x10x10" NbCl5} --timestep-fs 1 --temperature-K "$temp" \
      --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 256 --blocks "${BLOCKS:-8}" \
      --order "$order" --pattern "$pattern" --b-constant "$b" > "$o.log" 2>&1 < /dev/null
    note "end ${temp}K $order exit=$? $(grep -h -E "b/a|Error" "$o.log" | tail -n 3 | tr '\n' ' ' | cut -c1-400)"
  done
done
note done
