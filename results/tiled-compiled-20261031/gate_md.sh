#!/bin/bash
# Compiled tile bodies on murphy: the force/stress gate (NaCl, NbCl5, NaNbO3 3x3x3, R = 128) and the NVE checks
# (NbCl5 9,216 and NaCl 13,824 atoms, 300 steps, tiled against the compiled whole-system call). run.sh's check.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
REF=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
PY=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
OUTSUB=gate bash $R tc-gate tools/force_stress_gate.py --engine fastmace --checkpoint $CK --model mh1-r2scan \
  --structures NaCl NbCl5 NaNbO3 --structures-root $B --tiled-receivers 128 --tiled-mode compiled \
  --reference-command $REF --production-command $PY > /dev/null 2>&1
echo "gate exit $?"
bash $R tc-md-NbCl5-4x4x8 tools/bench/tiled_md.py --structure NbCl5 --repeat 4x4x8 --mode compiled --path both \
  --steps 300 > /dev/null 2>&1
echo "md NbCl5 exit $?"
bash $R tc-md-NaCl-12x12x12 tools/bench/tiled_md.py --structure NaCl --repeat 12x12x12 --mode compiled --path both \
  --steps 300 > /dev/null 2>&1
echo "md NaCl exit $?"
