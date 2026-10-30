#!/usr/bin/env bash
# segment_gemm tile order (docs/validation/segment-gemm-order.md) on the Modal A100 (sm_80, 40 MB L2): the bitwise
# GPU tests, the kernel probe at 9,216 and 51,840 atoms, and same-process A/Bs (tools/ab_rewrites.py, a = TILE_ORDER
# "rows", b = the default "auto", both build orders), MH-1 hybrid float32 NVT 300 K. As
# results/segment-gemm-order-20261030/ada.sh.
# Usage (from WSL, Modal client venv; the guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100 --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/sgo_a100.sh" --out ~/fastmace-runs/segment-gemm-order-20261030/modal-A100
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-mh-1.model
export FASTMACE_HYBRID_PYTHON=$(command -v python)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
run() {  # name command...
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader); source $(cat .source_commit)"
run gputest python -m pytest -q -p no:cacheprovider tests/gpu/test_segment_gemm_cuda.py tests/gpu/test_tf32x3_gemm_cuda.py
for n in 9216 51840; do run "probe-$n" python results/segment-gemm-order-20261030/gemm_order_probe.py "$n"; done
for cell in nbcl5big nbcl5 nanbo3 nacl nanbo3big; do
  extra=()
  case $cell in
    nacl) structure=NaCl ;;
    nbcl5) structure=NbCl5 ;;
    nanbo3) structure=NaNbO3 ;;
    nbcl5big) structure=NbCl5@4x4x8; extra=(--warmup 100 --block 100 --blocks 8) ;;
    nanbo3big) structure=NaNbO3@6x6x6; extra=(--warmup 100 --block 100 --blocks 8) ;;
  esac
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    run "ab-$cell-$order" python tools/ab_rewrites.py --output "$out/ab-$cell-$order" --checkpoint "$ck" \
      --model mh1-r2scan --structures "$structure" --a-constant "fastmace.kernels.triton.tile_rule:TILE_ORDER='rows'" \
      --order "$order" --pattern "$pattern" --prepare "${extra[@]}"
  done
done
note done
