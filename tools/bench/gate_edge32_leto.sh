#!/usr/bin/env bash
# Force/stress gate (tools/force_stress_gate.py, 3x float32 limit, decision 66) of
# fastmace.neighbors.skin.EDGE_INDEX_INT32=True on the RTX 4000 Ada: the default plan, SrTiO3 3x3x3 and NaCl, OMAT-0
# medium. The int32 edge index is a rounding change (reproducible MH-1 plan: off/off bitwise, off/on 3.8e-5 eV/A).
# The omat0-ada-capacity common.sh (environment, others(); two quiet minutes before).
#   ROOT=<dir> bash gate_edge32_leto.sh SRC_DIR
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
  --production-constant fastmace.neighbors.skin:EDGE_INDEX_INT32=True > $out/gate.log 2>&1 < /dev/null
note "gate exit=$? $(python /mnt/homes/zwingel/fm-ada-capacity/jobs2/gatesum.py $out/gate.log 2>/dev/null | tr '\n' ';' | cut -c1-600)"
note "GATE DONE"
