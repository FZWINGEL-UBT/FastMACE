#!/usr/bin/env bash
# Two cheap levers from the Codex report v4 (9 November 2026), as MD-step A/Bs (tools/modal/round6_ab256.sh, SrTiO3
# 10x10x10, both build orders) against the GPU's default plan:
# - nvrtc: the NVRTC-compiled kernels (w', the SPLINE_W forward, the streaming kernels) built with the newest NVRTC
#   wheel (NVRTC_VERSION, default 13.5.37) instead of the environment's 13.0.88 (runtime:NVRTC, compiled in a child
#   process; kernel ids include the path, so both sides' cubins coexist);
# - spill80 / spill64: w' capped at 80 / 64 registers with CUDA 13's shared-memory spilling (wprime_backward:
#   SMEM_SPILL). The plain cap at 80 spills to local memory and was 1.255x slower (ideas-r14 item 36).
# Rounding changes only (the compiler's FMA contraction / scheduling); a faster variant still needs the gate.
# Environment: OUT (the job sets it), VARIANTS (default "nvrtc spill80 spill64"), NVRTC_LIB, PY.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
r=fastmace.kernels.oeq_streaming.runtime
w=fastmace.kernels.oeq_streaming.wprime_backward
lib=${NVRTC_LIB:-}  # a local libnvrtc.so.13 skips the download
for v in ${VARIANTS:-nvrtc spill80 spill64}; do
  case $v in
    nvrtc)
      if [ -z "$lib" ]; then
        mkdir -p /tmp/nvrtc-new && pip download --no-deps "nvidia-cuda-nvrtc==${NVRTC_VERSION:-13.5.37}" -d /tmp/nvrtc-new -q \
          >> "$log" 2>&1 && (cd /tmp/nvrtc-new && python -m zipfile -e ./*.whl x)
        lib=/tmp/nvrtc-new/x/nvidia/cu13/lib/libnvrtc.so.13
        note "nvrtc $(ls -la $lib 2>&1)"
      fi
      bc="$r:NVRTC=\"$lib\"" ;;
    spill80) bc="$w:SMEM_SPILL=True $w:MAX_REGISTERS=80" ;;
    spill64) bc="$w:SMEM_SPILL=True $w:MAX_REGISTERS=64" ;;
    cap80) bc="$w:MAX_REGISTERS=80" ;;
    *) note "unknown variant $v"; continue ;;
  esac
  mkdir -p "$out/$v"
  OUT=$out/$v NAME=$v B_CONSTANTS="$bc" STRUCTURES=${STRUCTURES:-results/loup-a100-20261103/SrTiO3.cif@10x10x10} \
    bash ${AB_SCRIPT:-tools/modal/round6_ab256.sh} > "$out/$v.log" 2>&1 < /dev/null
  note "end $v exit=$? $(grep -h 'end ab' "$out/$v/run.log" | cut -c1-300 | tr '\n' ' ')"
done
note done
