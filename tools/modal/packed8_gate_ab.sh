#!/usr/bin/env bash
# PACKED8 (spline_forward, packed8v2 of results/packed-table-lab-20261109) on the H200, 9 November 2026:
# 1. the force/stress gate (tools/modal/const_gate.sh: the default plan, then PACKED8=True);
# 2. MD-step A/Bs (tools/modal/round6_ab256.sh, both build orders, OMAT-0 medium, SrTiO3 10x10x10): 300 K (the
#    crystal, where the kernel lab gave 1.004) and TEMPERATURE_HOT (default 1500 K; disordered, lab 0.858 at 0.3 A).
# Environment: OUT (the job sets it), PARTS (default "gate ab300 abhot"), GPU_MATCH (default H200; A100 for the
# A100 run of 9 November 2026, after step_limiters showed its forward L2-bound).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
echo "$gpu" | grep -q "${GPU_MATCH:-H200}" || { note "WRONG-GPU $gpu"; exit 3; }
note "packed8_gate_ab on $gpu; source $(cat .source_commit 2>/dev/null)"
c="fastmace.kernels.oeq_streaming.spline_forward:PACKED8=True"
for part in ${PARTS:-gate ab300 abhot}; do
  case $part in
    gate)
      mkdir -p "$out/gate"
      OUT=$out/gate ALIAS="$gpu" NAME=packed8 CONST="$c" bash tools/modal/const_gate.sh > "$out/gate.log" 2>&1 < /dev/null
      note "end gate exit=$? $(grep -h 'gate-' "$out/gate/run.log" | cut -c1-300 | tr '\n' ' ')" ;;
    ab300|abhot)
      t=300; [ "$part" = abhot ] && t=${TEMPERATURE_HOT:-1500}
      mkdir -p "$out/$part"
      OUT=$out/$part NAME=packed8-$t B_CONSTANTS="$c" TEMPERATURE_K=$t \
        STRUCTURES=results/loup-a100-20261103/SrTiO3.cif@10x10x10 bash tools/modal/round6_ab256.sh \
        > "$out/$part.log" 2>&1 < /dev/null
      note "end $part exit=$? $(grep -h 'end ab' "$out/$part/run.log" | cut -c1-300 | tr '\n' ' ')" ;;
  esac
done
note done
