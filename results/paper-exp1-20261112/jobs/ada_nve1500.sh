#!/bin/bash
# Ada: 20 ps Langevin at 1,500 K from Maxwell-Boltzmann momenta (configuration a, saved), then 50 ps NVE for a and b.
source $(dirname $0)/ada_common.sh
note "ada_nve1500 $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
step eq-1500K $PY tools/nve_energy.py equilibrate --config a --checkpoint $CK --cif $CIF --repeat 10x10x10 \
  --temperature-K 1500 --ps 20 --state $OUT/state-1500K.npz
for c in a b; do
  step nve-$c-1500K $PY tools/nve_energy.py nve --config $c --checkpoint $CK --start $OUT/state-1500K.npz \
    --temperature-K 1500 --ps 50 --output $OUT/nve --tag $c-1500K
done
note DONE
