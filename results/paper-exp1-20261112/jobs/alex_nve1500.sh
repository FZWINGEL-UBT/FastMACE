#!/bin/bash -l
# Measurement 4 at 1,500 K on Alex A100: 20 ps Langevin at 1,500 K from Maxwell-Boltzmann momenta (configuration a,
# saved), then 50 ps NVE at 1 fs for a, b, c from that state; SrTiO3 10x10x10.
#SBATCH --job-name=fm-pexp1-nve1500
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
source $SLURM_SUBMIT_DIR/alex_common.sh
run eq-1500K python tools/nve_energy.py equilibrate --config a --checkpoint $CK --cif $CIF --repeat 10x10x10 \
  --temperature-K 1500 --ps 20 --state $LOUT/state-1500K.npz || exit 1
for c in a b c; do
  run nve-$c-1500K python tools/nve_energy.py nve --config $c --checkpoint $CK --start $LOUT/state-1500K.npz \
    --temperature-K 1500 --ps 50 --output $LOUT/nve --tag $c-1500K
done
note DONE
