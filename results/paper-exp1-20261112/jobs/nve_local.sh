#!/bin/bash
# Measurement 4, first pass on the RTX 5060 Ti: 20 ps Langevin at 300 K (configuration a, saved), then 100 ps NVE at
# 1 fs for configurations a and b from the same saved state; SrTiO3 10x10x10 (5,000 atoms).
source $(dirname $0)/local_common.sh
note "nve_local $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
NVE_PS=${NVE_PS:-100}; EQ_PS=${EQ_PS:-20}; MS=${MS:-25}   # MS: expected ms/step for the 3x wall limit
lim() { echo $(( 3 * ($1 * MS + 600) )); }
[ -e $OUT/state-300K.npz ] || step eq-300K $(lim $(( EQ_PS * 1000 ))) $PY tools/nve_energy.py equilibrate --config a \
  --checkpoint $CK --cif $CIF --repeat 10x10x10 --temperature-K 300 --ps $EQ_PS --state $OUT/state-300K.npz
for c in ${CONFIGS:-a b}; do
  step nve-$c-300K $(lim $(( NVE_PS * 1000 ))) $PY tools/nve_energy.py nve --config $c --checkpoint $CK \
    --start $OUT/state-300K.npz --temperature-K 300 --ps $NVE_PS --output $OUT/nve --tag $c-300K
done
note DONE
