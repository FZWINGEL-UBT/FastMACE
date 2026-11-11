#!/usr/bin/env bash
# Force/stress gate (tools/force_stress_gate.py, its default 3x float32 rule) of a candidate set on this GPU.
#   HOST=ada|local ROOT=<dir> bash gate.sh SRC_DIR NAME MODEL CKPT_FILE "STRUCTURES" "REPEAT" CONSTANT... [-- REWRITE...]
set -u -o pipefail
FM=$1 name=$2 model=$3 ckfile=$4 structs=$5 repeat=$6; shift 6
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
con=(); rw=()
while [ $# -gt 0 ]; do [ "$1" = "--" ] && { shift; rw=("$@"); break; }; con+=("$1"); shift; done
out=$ROOT/gate-$name; mkdir -p "$out"
if why=$(others); then note "STOP before gate $name: $why"; exit 3; fi
note "gate $name start fastmace=$(cat .source_commit) model=$model constants=${con[*]} rewrites=${rw[*]}"
# shellcheck disable=SC2086
python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model "$model" --checkpoint "$MODELS/$ckfile" \
  --structures-root "$FASTMACE_STRUCTURES" --structures $structs --repeat $repeat \
  ${con:+--production-constant "${con[@]}"} ${rw:+--rewrites "${rw[@]}"} > "$out/gate.log" 2>&1 < /dev/null
note "gate $name exit=$? $(grep -hiE 'verdict|passed|failed|ratio' "$out/gate.log" | tail -n 8 | tr '\n' ';' | cut -c1-600)"
