#!/usr/bin/env bash
# The 128-row tile rerun (docs/validation/ab-constants-audit-20261030.md): segment_gemm's Triton products with 128 rows
# and 8 warps against 64 and 4, with the fixed A/B tool (constants set before every step), on the old row order
# ("rows": main before the tile order) and on the tile order's rule ("auto"). MH-1 hybrid float32 NVT 300 K.
#   bash tiles128.sh ROOT FM ENV CHECKPOINT STRUCTURES_ROOT
# ENV is the hybrid environment's prefix; set the caches and LD_LIBRARY_PATH as ada.sh or local.sh do before.
set -u -o pipefail
ROOT=$1 FM=$2 E=$3 CK=$4 S=$5
export PYTHONPATH=$FM FASTMACE_STRUCTURES=$S
TR=fastmace.kernels.triton.tile_rule
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {  # other users' processes using CPU, or any GPU compute process
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $1 != "fzwin" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
step() {
  local name=$1; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for base in rows auto; do
  for cell in nbcl5big nbcl5 nanbo3; do
    extra=()
    case $cell in
      nbcl5) structure=NbCl5 ;;
      nanbo3) structure=NaNbO3 ;;
      nbcl5big) structure=NbCl5@4x4x8; extra=(--warmup 100 --block 100 --blocks 8) ;;
    esac
    for order in ab ba; do
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      step "t128-$base-$cell-$order" "$E/bin/python" "$FM/tools/ab_rewrites.py" --output "$ROOT/t128-$base-$cell-$order" \
        --checkpoint "$CK" --model mh1-r2scan --structures "$structure" \
        --a-constant "$TR:TILE_ORDER='$base'" \
        --b-constant "$TR:TILE_ORDER='$base'" "$TR:TRITON_BLOCK_M=128" "$TR:TRITON_NUM_WARPS=8" \
        --order "$order" --pattern "$pattern" --prepare "${extra[@]}"
    done
  done
done
note DONE
