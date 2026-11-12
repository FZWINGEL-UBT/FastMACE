#!/bin/bash
# NVE driver smoke on the 5060 Ti: SrTiO3 4x4x4 (320 atoms), 1 ps Langevin, 1 ps NVE of a and c.
source $(dirname $0)/local_common.sh
step eq 900 $PY tools/nve_energy.py equilibrate --config a --checkpoint $CK --cif $CIF --repeat 4x4x4 --temperature-K 300 \
  --ps 1 --state $OUT/state.npz
for c in a c; do
  step nve-$c 900 $PY tools/nve_energy.py nve --config $c --checkpoint $CK --start $OUT/state.npz --temperature-K 300 \
    --ps 1 --every 100 --output $OUT/nve --tag $c
done
note DONE
