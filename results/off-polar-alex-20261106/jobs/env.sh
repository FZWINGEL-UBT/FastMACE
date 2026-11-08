# Sourced by every fm-offpolar job on Alex.
B=/home/atuin/v130cb/v130cb18/fm-offpolar
module purge >/dev/null 2>&1
source $B/env/bin/activate
SITE=$B/env/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export FASTMACE_CACHE=$B/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$B/baseline
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=${TMPDIR:-$B/tmp}
export PYTHONPATH=$B/src
export ORG=$B/baseline/Structures/Organic
mkdir -p $B/cache $B/runs
cd $B/src
