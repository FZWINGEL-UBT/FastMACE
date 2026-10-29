#!/bin/bash
# f64-T04 step 2: record float64 operands (NbCl5, NaNbO3; MH-1, 3x3x3) and time the df64 forward against native.
# Run inside the local GPU reservation from the WSL side; operands go to ~/tmp/f64T04 (not committed, ~0.5 GB).
set -euo pipefail
HERE=/mnt/d/Margraf/fm-f64d/results/f64-T04-20261027
WORK=$HOME/tmp/f64T04
mkdir -p "$WORK"
cd "$WORK"
PY=$HOME/environments/mace_cueq_oeq/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/d/Margraf/fm-f64d FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq
for s in NbCl5 NaNbO3; do
  [ -s "ops64-mh1-r2scan-$s.pt" ] || TORCHDYNAMO_DISABLE=1 $PY "$HERE/record_f64.py" mh1-r2scan "$s" "$WORK" 2>&1 \
    | grep -v Warning | tee -a "$HERE/step2_record.txt"
done
$PY "$HERE/step2_dd_forward.py" "$HERE/step2_rtx5060ti.json" ops64-mh1-r2scan-NbCl5.pt ops64-mh1-r2scan-NaNbO3.pt \
  2>&1 | grep -v Warning | tee "$HERE/step2_rtx5060ti.txt"
