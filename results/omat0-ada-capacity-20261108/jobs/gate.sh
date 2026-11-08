#!/usr/bin/env bash
# Force/stress gate (tools/force_stress_gate.py, 3x float32 limit) on the RTX 4000 Ada: GATE=default (the Ada plan) or
# a set of switches (all = the ten). With a set it first runs tools/spline_w_check.py (spline and compose in use).
#   ROOT=<dir> bash gate.sh SRC_DIR default|all|"MODULE:NAME=VALUE ..."
set -u -o pipefail
FM=$1; GATE=$2
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
tag=$GATE; case $GATE in default) set -- ;; all) set -- $ALL ;; *) tag=custom; set -- $GATE ;; esac
out=$ROOT/gate-$tag; mkdir -p "$out"
if why=$(others); then note "STOP before gate $tag: $why"; exit 3; fi
note "gate $tag start fastmace=$(cat .source_commit)"
if [ $# -gt 0 ]; then
  python tools/spline_w_check.py --checkpoint "$CK" --structure "$ST@10x10x10" --output $out/check.json --constant "$@" \
    --require-compose --rattle 0.05 --compare-constant $s:SPLINE_W=False $c:COMPOSE_SKIP=False $q:CUEQ_PROJECTED=False $t:GENERIC_FOLD_READOUT_TAIL=False > $out/check.log 2>&1 < /dev/null
  note "check exit=$? $(grep -hoE '"(spline_in_use|compose_skip_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' $out/check.log | tr '\n' ' ' | cut -c1-300)"
  pc=(--production-constant "$@")
else
  pc=()
fi
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint "$CK" \
  --structures-root "$FASTMACE_STRUCTURES" --structures "$ST@3x3x3" NaCl NbCl5 "${pc[@]}" > $out/gate.log 2>&1 < /dev/null
note "gate $tag exit=$? $(grep -h -iE 'verdict' $out/gate.log | tail -n 1 | cut -c1-200)"
