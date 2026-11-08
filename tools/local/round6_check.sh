#!/usr/bin/env bash
# Round 6 (8 November 2026) on the local RTX 5060 Ti: the H200 spline path forced on (SPLINE_W, R128, the forward's
# STAGE_EDGES, STAGE_DOUT, STAGE_SH, SPLINE_VECTOR 4), then per new switch tools/spline_w_check.py against the same
# build without it (forces on a copy rattled by 0.05 A next to the build's own spread; registers and local bytes).
# Usage (through the reservation): python3 tools/ci/local_gpu.py run -- bash tools/local/round6_check.sh OUT
set -uo pipefail
out=${1:?output directory}
mkdir -p "$out"
py=${PYTHON:-$HOME/environments/mace_cueq_latest/bin/python}
ck=${CHECKPOINT:-$HOME/models/mace-omat-0-medium.model}
st=${STRUCTURE:-results/loup-a100-20261103/SrTiO3.cif@6x6x6}
s=fastmace.kernels.oeq_streaming.spline_forward
w=fastmace.kernels.oeq_streaming.wprime_backward
base="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:SPLINE_VECTOR=4"
declare -A on=([wse]="$w:STAGE_EDGES=True" [pf2]="$s:PREFETCH_AHEAD=2")
declare -A off=([wse]="$w:STAGE_EDGES=False" [pf2]="$s:PREFETCH_AHEAD=0")
for name in wse pf2; do
  PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 "$py" tools/spline_w_check.py --checkpoint "$ck" --structure "$st" \
    --output "$out/check-$name.json" --constant $base ${on[$name]} --compare-constant $base ${off[$name]} \
    --rattle 0.05 --sources "$out/sources-$name" > "$out/check-$name.log" 2>&1 < /dev/null
  echo "$name exit=$? $(grep -hoE '"(forward_registers|forward_local_bytes|registers|local_bytes|max_abs|same_build_max_abs|energy_diff|same_build_energy_diff)": [^,]+' "$out/check-$name.log" | tr -d '\n ' | cut -c1-400)"
done
