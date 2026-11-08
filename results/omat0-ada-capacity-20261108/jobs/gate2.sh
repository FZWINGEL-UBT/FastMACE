#!/usr/bin/env bash
# Force/stress gate (tools/force_stress_gate.py, 3x float32 limit) of the Ada candidate plans: GATE=A (the adopted set:
# the spline group + readout tail + tf32x3_gemms, COMPOSE_SKIP and COMPOSE_FAST; COUNT_PLAN, JOINED_LAYOUT and
# CUEQ_PROJECTED stay off, stage 2) or GATE=N (the spline group + readout tail). First
# tools/spline_w_check.py (spline and compose in use; forces against the Ada default plan, rattled SrTiO3 10x10x10).
#   ROOT=<dir> bash gate2.sh SRC_DIR A|N
set -u -o pipefail
FM=$1; GATE=$2
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
# The comparison build: every candidate constant back to its Ada default (False; SPLINE_VECTOR 2), plan auto.
read -r -a OFF <<< "$(echo "$SPL ${CMP#"$T3 "} $TAIL" | sed 's/=True/=False/g; s/SPLINE_VECTOR=4/SPLINE_VECTOR=2/')"
case $GATE in
  A) read -r -a CON <<< "$SPL $c:COMPOSE_SKIP=True $c:COMPOSE_FAST=True $TAIL"; RW=(--rewrites "$T3"); REQ=(--require-compose) ;;
  N) read -r -a CON <<< "$SPL $TAIL"; RW=(); REQ=() ;;
  *) note "GATE?"; exit 2 ;;
esac
out=$ROOT/gate-$GATE; mkdir -p "$out"
if why=$(others); then note "STOP before gate $GATE: $why"; exit 3; fi
note "gate $GATE start fastmace=$(cat .source_commit)"
python tools/spline_w_check.py --checkpoint "$CK" --structure "$ST@10x10x10" --output $out/check.json \
  --constant "${CON[@]}" "${RW[@]}" "${REQ[@]}" --rattle 0.05 --compare-constant "${OFF[@]}" > $out/check.log 2>&1 < /dev/null
note "check exit=$? $(grep -hoE '"(spline_in_use|compose_skip_in_use|joined_layout_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' $out/check.log | tr '\n' ' ' | cut -c1-300)"
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint "$CK" \
  --structures-root "$FASTMACE_STRUCTURES" --structures "$ST@3x3x3" NaCl NbCl5 --production-constant "${CON[@]}" \
  "${RW[@]}" > $out/gate.log 2>&1 < /dev/null
note "gate $GATE exit=$? $(python $(dirname "${BASH_SOURCE[0]}")/gatesum.py $out/gate.log | tr '\n' ';')"
