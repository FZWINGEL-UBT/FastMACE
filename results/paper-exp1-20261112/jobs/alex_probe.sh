#!/bin/bash -l
# Measurements 1-3 on Alex A100 (the A100's own kernels: packed forward is on there too).
#SBATCH --job-name=fm-pexp1-probe
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
source $SLURM_SUBMIT_DIR/alex_common.sh
for c in a b c; do
  K="--knots-from $LOUT/probe/probe-a.json"; [ $c = a ] && K=""
  run probe-$c python tools/consistency_probe.py --config $c --output $LOUT/probe --checkpoint $CK --cif $CIF $K
done
note DONE
