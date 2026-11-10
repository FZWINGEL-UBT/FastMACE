#!/usr/bin/env bash
# Force/stress gate (tools/force_stress_gate.py, 3x float32 limit, decision 66) of generic_recompute_first_layer:
# the Ada default plan with recompute_first_table and recompute_first_layer forced on, SrTiO3 3x3x3 and NaCl.
#   ROOT=<dir> bash gate3.sh SRC_DIR
set -u -o pipefail
FM=$1
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
out=$ROOT/gate; mkdir -p "$out"
quiet=0; last=""
while [ $quiet -lt 2 ]; do
  if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before gate: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
  sleep 60
done
note "gate start fastmace=$(cat .source_commit)"
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint "$CK" \
  --structures-root "$FASTMACE_STRUCTURES" --structures "/mnt/homes/zwingel/fm-memprofile/SrTiO3.cif@3x3x3" NaCl \
  --rewrites recompute_first_table=on recompute_first_layer=on > $out/gate.log 2>&1 < /dev/null
note "gate exit=$? $(python /mnt/homes/zwingel/fm-ada-capacity/jobs2/gatesum.py $out/gate.log 2>/dev/null | tr '\n' ';' | cut -c1-600)"
note "GATE DONE"
