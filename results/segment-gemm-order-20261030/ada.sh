#!/usr/bin/env bash
# segment_gemm tile order on Ada (leto or murphy: RTX 4000 Ada, sm_89, 48 MB L2): the bitwise GPU tests, the
# kernel probe (gemm_order_probe.py beside this script, at 9,216 and 51,840 atoms) and same-process A/Bs
# (tools/ab_rewrites.py), a = TILE_ORDER "rows" (the order before), b = the default ("auto"), both build orders, MH-1
# hybrid float32 NVT 300 K. Run from a scratch folder under /mnt/homes/zwingel holding a git archive of the source
# (FM); every job first checks that no other user's job runs and the GPU has no compute process (AGENTS.md, "Shared
# workstations"), and stops the sequence otherwise. Template: results/opt6-T05-radial-tables-20261027/ada.sh.
#   bash ada.sh ROOT FM
set -u -o pipefail
ROOT=$1 FM=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
REF=/mnt/homes/zwingel/environments/mace_cueq_latest
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton MACE_COMPILE_CACHE=$ROOT/cache/compile XDG_CACHE_HOME=$ROOT/cache/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {  # other users' processes using CPU (a job), or any GPU compute process
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
step() {  # name command...
  local name=$1; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
(cd "$FM" && step gputest "$REF/bin/python" -m pytest -q -p no:cacheprovider tests/gpu/test_segment_gemm_cuda.py \
   tests/gpu/test_tf32x3_gemm_cuda.py)
for n in 9216 51840; do step "probe-$n" "$E/bin/python" "$FM/results/segment-gemm-order-20261030/gemm_order_probe.py" "$n"; done
for cell in nbcl5 nanbo3 nacl nbcl5big nanbo3big; do
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
    step "ab-$cell-$order" "$E/bin/python" "$FM/tools/ab_rewrites.py" --output "$ROOT/ab-$cell-$order" \
      --checkpoint "$CK" --model mh1-r2scan --structures "$structure" \
      --a-constant "fastmace.kernels.triton.tile_rule:TILE_ORDER='rows'" \
      --order "$order" --pattern "$pattern" --prepare "${extra[@]}"
  done
done
note DONE
