#!/bin/bash
# usage: repro.sh <side> <args...>   (leto, NAS environments)
W=/mnt/homes/zwingel/fastmace-work/1a-T13b
O=$W-out
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
SP=$E/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1
export FASTMACE_CACHE=/mnt/homes/zwingel/fastmace-cache/fastmace
export XDG_CACHE_HOME=/mnt/homes/zwingel/fastmace-cache/xdg
export TRITON_CACHE_DIR=/mnt/homes/zwingel/fastmace-cache/triton
export TMPDIR=/mnt/homes/zwingel/fastmace-cache/tmp TMP=/mnt/homes/zwingel/fastmace-cache/tmp TEMP=/mnt/homes/zwingel/fastmace-cache/tmp
export MPLCONFIGDIR=/mnt/homes/zwingel/fastmace-cache/matplotlib
export FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export MACE_BENCH_CHECKPOINT=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
side=$1; shift; tag=$1; shift
cd $O
if [ "$side" = fastmace ]; then
  PYTHONPATH=$W $E/bin/python $O/repro_bucket.py fastmace $B "$@" > $O/$tag.json 2> $O/$tag.err
elif [ "$side" = ab ]; then
  export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  PYTHONPATH=$W bash $B/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh $E /mnt/homes/zwingel/fastmace-cache/oeq-baseline-1a-T13b     $E/bin/python $O/repro_ab.py $B "$@" > $O/$tag.json 2> $O/$tag.err
else
  export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  bash $B/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh $E /mnt/homes/zwingel/fastmace-cache/oeq-baseline-1a-T13b \
    $E/bin/python $O/repro_bucket.py baseline $B "$@" > $O/$tag.json 2> $O/$tag.err
fi
echo "EXIT $?" >> $O/$tag.err
