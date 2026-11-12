#!/bin/bash -l
# Measurement 4 at 300 K on Alex A100: 20 ps Langevin (configuration a, saved), then 100 ps NVE at 1 fs for a, b, c
# from that state; SrTiO3 10x10x10. sbatch --export=SRC_TAR=... from $M.
#SBATCH --job-name=fm-pexp1-nve300
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
source $SLURM_SUBMIT_DIR/alex_common.sh
run eq-300K python tools/nve_energy.py equilibrate --config a --checkpoint $CK --cif $CIF --repeat 10x10x10 \
  --temperature-K 300 --ps 20 --state $LOUT/state-300K.npz || exit 1
for c in a b c; do
  run nve-$c-300K python tools/nve_energy.py nve --config $c --checkpoint $CK --start $LOUT/state-300K.npz \
    --temperature-K 300 --ps 100 --output $LOUT/nve --tag $c-300K
done
note DONE
