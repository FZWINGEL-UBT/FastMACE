#!/usr/bin/env bash
# H200 ideas plan idea 31 (6 November 2026; kernels/cutlass/linear_div.LINEAR_DIV, kernels/cutlass/first_div.FIRST_DIV):
# on one GPU,
# 1. the GPU tests of the epilogue GEMMs against today's compiled path (bitwise output and gradients, OMAT-0 shapes);
# 2. tools/bench/idea31_epilogue.py: OMAT-0 medium force calls and 20 MD steps on SrTiO3 10x10x10, default (a),
#    a second default build (a2: the run-to-run floor) and side b (CONST), compared bit for bit;
# 3. the MD-step A/B in both build orders (hopper_sweep.sh's AB).
# CONST (side b's module constants, space-separated), NAME (the A/B's name) and GPU_TESTS choose the switch; the
# defaults are the first family's (LINEAR_DIV=True).
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 45 --commit <sha> \
#     --command "ALIAS='NVIDIA H200' bash tools/modal/idea31_check.sh" --out ~/fm-runs/idea31-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "${ALIAS:?set ALIAS}" ] || { echo "WRONG-GPU: $gpu"; exit 3; }
const=${CONST:-fastmace.kernels.cutlass.linear_div:LINEAR_DIV=True}
name=${NAME:-lineardiv}
tests=${GPU_TESTS:-tests/gpu/test_linear_div_cuda.py tests/gpu/test_cutlass_tf32x3_tiles_cuda.py}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
# shellcheck disable=SC2086
python -m pytest -p no:cacheprovider -q -rs $tests > "$out/tests.log" 2>&1 < /dev/null
note "tests exit=$? $(tail -n 1 "$out/tests.log")"
python tools/bench/idea31_epilogue.py --output "$out/bitwise" --checkpoint "$FASTMACE_MODELS/mace-omat-0-medium.model" \
  --structure "results/loup-a100-20261103/SrTiO3.cif@10x10x10" --configs 2 --steps 20 --b-constant $const \
  > "$out/bitwise.log" 2>&1 < /dev/null
note "bitwise exit=$?"
grep -E "^side |vs a:|^  " "$out/bitwise.log" >> "$log"
TESTS=0 GATE=0 LOUP=0 FORWARD_VARIANTS= WPRIME_VARIANTS= AB="$name|$const" bash tools/modal/hopper_sweep.sh
note done
