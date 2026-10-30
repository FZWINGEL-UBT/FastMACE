#!/bin/bash
# Tiled capacity path, accuracy on murphy: the tiled call against the default compiled evaluation plan
# (tiled_capacity.py --compare compiled), then the force/stress gate with the tiled path as production and the
# default production path, NaCl, NbCl5, NaNbO3 3x3x3. Each job goes through run.sh (idle check first).
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
REF=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
PY=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
for spec in "NbCl5 4x4x8" "NbCl5 5x5x12" "NaCl 8x8x8" "NaNbO3 6x6x6"; do
  set -- $spec
  bash $R cmp-$1-$2 tools/bench/tiled_capacity.py --structure $1 --repeat $2 --receivers 4096 --compare compiled \
    --rattle 0.05 --strain 0.02 > /dev/null 2>&1
  echo "cmp $1 $2 exit $?"
done
bash $R gate-tiled tools/force_stress_gate.py --engine fastmace --checkpoint $CK --model mh1-r2scan \
  --structures NaCl NbCl5 NaNbO3 --structures-root $B --tiled-receivers 128 \
  --reference-command $REF --production-command $PY > /dev/null 2>&1
echo "gate-tiled exit $?"
bash $R gate-default tools/force_stress_gate.py --engine fastmace --checkpoint $CK --model mh1-r2scan \
  --structures NaCl NbCl5 NaNbO3 --structures-root $B \
  --reference-command $REF --production-command $PY > /dev/null 2>&1
echo "gate-default exit $?"
