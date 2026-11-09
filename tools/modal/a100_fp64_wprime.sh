#!/usr/bin/env bash
# Round 2 idea 1, third gate (9 November 2026): the A100's w' with its two 21-term paths on the FP64 pipe.
# Gates passed: the FFMA+DFMA microbenchmark (2:1 mix 93.2 FMA results per clock per SM against 63.7 for FP32 alone)
# and Nsight on today's A100 w' (FMA pipe 85.8 % active, math-pipe throttle 1.93 per issue). Variants of the kernel
# lab: fp64-paths (real float <-> double conversions per use; rounding, checked against base) and F-fp64-free (the
# conversions replaced by register reinterpretations: the upper bound of a design that stages doubles; wrong values).
# By the ledger's rule a lab variant must save at least 141 us at kernel level on the A100.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 25 --tarball <slim tar> \
#     --command "bash tools/modal/a100_fp64_wprime.sh" --out <dir>
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 > "$out/gpu.txt"
grep -q A100 "$out/gpu.txt" || { note "WRONG-GPU $(cat "$out/gpu.txt")"; exit 3; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --variants base fp64-paths F-fp64-free \
  --rounds 21 --ptxas-archs 8.0 --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
note "end lab exit=$?"
grep -h -E "^wprime |ptxas|registers|spill" "$out/lab.log" | cut -c1-240 >> "$log"
note done
