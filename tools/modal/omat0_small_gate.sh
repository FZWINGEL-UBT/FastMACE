#!/usr/bin/env bash
# OMAT-0 small's own H200 defaults (8 November 2026): the label omat0-small takes medium's 3xTF32 GEMMs and generic
# w' on the H200 (decision 66 tables) and keeps GENERIC_FOLD_READOUT_TAIL off. No constants: the plan's defaults.
#   1. tools/force_stress_gate.py (3x rule, selfcheck.RATIO) on SrTiO3 3x3x3, NaCl, NbCl5, as medium's H200 gates;
#   2. tools/step_breakdown.py on Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5): about 2.3 ms expected.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-small.model}
st=results/loup-a100-20261103/SrTiO3.cif
note "omat0_small_gate on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

note "start gate"
python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-small \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  > "$out/gate.log" 2>&1 < /dev/null
note "end gate exit=$?"
grep -hE "verdict|PASS|FAIL|ratio" "$out/gate.log" | tail -n 20 | sed 's/^/  /' >> "$log"

note "start breakdown"
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-small \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$? $(grep -hoE '"(wall_ms_per_step|force_graph_ms_per_replay)": [0-9.]+' "$out/breakdown.log" | tr '\n' ' ')"
rm -f "$out"/breakdown/trace.json
note done
