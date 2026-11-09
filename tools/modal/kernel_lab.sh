#!/usr/bin/env bash
# tools/omat0_kernel_lab.py on a Modal GPU (OMAT-0 medium, Benchmark_loup SrTiO3 10x10x10 capture), for bounds and
# variant screens. Environment: OUT (the job sets it), KERNELS (default wprime), VARIANTS (default "base"; base is
# always first), ROUNDS (default 21), CONSTANTS (lab --constant words), LAB_ARGS (further lab arguments).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 15 --tarball <tar> \
#     --command "VARIANTS='F-no-reduce' bash tools/modal/kernel_lab.sh" --out ~/fm-runs/lab
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
note "kernel_lab on $gpu; source $(cat .source_commit 2>/dev/null); kernels ${KERNELS:-wprime}; variants ${VARIANTS:-base}"
read -r -a variants <<< "${VARIANTS:-}"
read -r -a constants <<< "${CONSTANTS:-}"
read -r -a extra <<< "${LAB_ARGS:-}"
cmd=(python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels ${KERNELS:-wprime} --variants base "${variants[@]}"
     --rounds "${ROUNDS:-21}" --output "$out/lab")
[ ${#constants[@]} -eq 0 ] || cmd+=(--constant "${constants[@]}")
[ ${#extra[@]} -eq 0 ] || cmd+=("${extra[@]}")
"${cmd[@]}" > "$out/lab.log" 2>&1 < /dev/null
note "end lab exit=$?"
grep -h -E "^(wprime|forward|[a-z_]+) +[A-Za-z0-9-]+ +[0-9.]+ us|ptxas" "$out/lab.log" | cut -c1-260 >> "$log"
note done
