#!/bin/bash
# Ada replicate at 1,500 K (leto's first (a) drifted -4.8e-5 against b 0.0 and the A100's +6e-6): (a) again from the
# same saved state (STATE1, reproducibility of one trajectory), then a second Langevin seed (seed 2) with a and b.
source $(dirname $0)/ada_common.sh
note "ada_nve1500_rep $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
step nve-a-1500K-again $PY tools/nve_energy.py nve --config a --checkpoint $CK --start $STATE1 --temperature-K 1500 \
  --ps 50 --output $OUT/nve --tag a-1500K-again
step eq-1500K-seed2 $PY tools/nve_energy.py equilibrate --config a --checkpoint $CK --cif $CIF --repeat 10x10x10 \
  --temperature-K 1500 --ps 20 --seed 2 --state $OUT/state-1500K-seed2.npz
for c in a b; do
  step nve-$c-1500K-seed2 $PY tools/nve_energy.py nve --config $c --checkpoint $CK --start $OUT/state-1500K-seed2.npz \
    --temperature-K 1500 --ps 50 --output $OUT/nve --tag $c-1500K-seed2
done
note DONE
