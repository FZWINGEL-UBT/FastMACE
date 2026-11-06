#!/usr/bin/env bash
# wprime_backward.STAGE_DOUT (H200 ideas plan idea 38, 6 November 2026): the w' GPU test (STAGE_DOUT cases bitwise),
# the kernel lab (base, the descriptor-tile and the chunk form, the locality bound), then the MD-step A/B in both
# build orders (hopper_sweep.sh's AB). STAGE_DOUT is bitwise in dsh and dL/dr (dx: the atomics' order), so no gate.
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 30 --commit <sha> \
#     --command "ALIAS='NVIDIA H200' bash tools/modal/stage_dout_check.sh" --out ~/fm-runs/stage-dout-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "${ALIAS:?set ALIAS}" ] || { echo "WRONG-GPU: $gpu"; exit 3; }
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
python -m pytest -p no:cacheprovider -q -rs tests/gpu/test_wprime_backward_cuda.py > "$out/gputest.log" 2>&1 < /dev/null
note "gputest exit=$? $(tail -n 1 "$out/gputest.log" | cut -c1-200)"
TESTS=0 GATE=0 LOUP=0 FORWARD_VARIANTS= WPRIME_VARIANTS="${WPRIME_VARIANTS-base stage-t32 stage-c16 stage-c32 FF-dout-row0}" \
  AB="stagedout|fastmace.kernels.oeq_streaming.wprime_backward:STAGE_DOUT=True" bash tools/modal/hopper_sweep.sh
note done
