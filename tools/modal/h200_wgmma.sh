#!/usr/bin/env bash
# H200: the fused forward's a.W product on Hopper wgmma (fused_forward.WGMMA; round 5, 5 November 2026). Steps:
# - the fused forward's GPU test (the wgmma block runs on sm_90 only; TEST=0 skips it);
# - Benchmark_loup Langevin with WGMMA on (LOUP fresh processes, default 2);
# - MD-step A/Bs on SrTiO3 10x10x10, both build orders, side b with WGMMA on, then one per extra entry of AB
#   ("name|MODULE:NAME=VALUE ..." separated by ";"; each with WGMMA on on both sides);
# - the force/stress gate with WGMMA on (GATE=1; tools/bench/constants_hook).
# PIPE=1 adds WGMMA_PIPELINED=True everywhere WGMMA is on (the GPU test too, through the constants hook).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/h200_wgmma.sh" --out ~/fm-runs/h200-wgmma
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
on=fastmace.kernels.oeq_streaming.fused_forward:WGMMA=True
pipe=fastmace.kernels.oeq_streaming.fused_forward:WGMMA_PIPELINED=True
hook=()
if [ "${PIPE:-0}" = 1 ]; then
  on="$on $pipe"
  hook=(env PYTHONPATH=$PWD/tools/bench/constants_hook:${PYTHONPATH:-} FASTMACE_CONSTANTS=$pipe)
fi
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(tail -n 1 "$out/$name.log" | cut -c1-300)"
  return $code
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
if [ "${TEST:-1}" = 1 ]; then
  step test "${hook[@]}" python -m pytest -q -s -p no:cacheprovider tests/gpu/test_fused_folded_forward_cuda.py || { note "test failed: stop"; exit 0; }
  grep -h '"wgmma"' "$out/test.log" | head -n 2 | cut -c1-600 >> "$log"
fi
for t in $(seq 1 "${LOUP:-2}"); do
  step "loup-$t" python tools/bench/loup_md.py --structure "$st@10x10x10" --checkpoint "$ck" --ensemble langevin \
    --constant $on --output "$out/loup/langevin-$t.json"
done
ab() {  # name, order, extra ab_rewrites arguments
  local name=$1 order=$2; shift 2
  step "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" \
    --model omat0-medium --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --order "$order" --pattern abba "$@"
  grep -h "b/a" "$out/ab-$name-$order.log" | tail -n 1 >> "$log"
}
if [ "${ABMAIN:-1}" = 1 ]; then
  for order in ab ba; do ab wgmma "$order" --b-constant $on; done
fi
IFS=';' read -r -a abs <<< "${AB-}"
for entry in "${abs[@]}"; do
  [ -n "$entry" ] || continue
  name=${entry%%|*}
  read -r -a bc <<< "${entry#*|}"
  for order in ab ba; do ab "$name" "$order" --a-constant $on --b-constant $on "${bc[@]}"; done
done
if [ "${GATE:-1}" = 1 ]; then
  PYTHONPATH=$PWD/tools/bench/constants_hook:${PYTHONPATH:-} FASTMACE_CONSTANTS="${on// /;}" \
    step gate python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 "$st@3x3x3"
  grep -h -iE "verdict|pass|fail" "$out/gate.log" | tail -n 6 >> "$log"
fi
note done
