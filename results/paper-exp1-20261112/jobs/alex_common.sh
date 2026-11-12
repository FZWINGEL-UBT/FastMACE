# Sourced by the Alex (A100-SXM4-80GB) job files of paper experiment 1. Code tree, env and caches on node-local
# $TMPDIR (atuin is over its group file quota); only records are copied back to $M/runs/<job>.
#   sbatch --export=NONE ... alex_<name>.sh   (submitted from $M; SRC_TAR and ENV_TAR set below)
unset SLURM_EXPORT_ENV
set -uo pipefail
M=/home/atuin/v130cb/v130cb18/fm-paper-exp1
B=/home/atuin/v130cb/v130cb18/fm-a100omat
SRC_TAR=${SRC_TAR:?set SRC_TAR}
ENV_TAR=${ENV_TAR:-/home/atuin/v130cb/v130cb18/fm-alexsetup/env-py310.tar}
module purge >/dev/null 2>&1
L=$TMPDIR/fm; mkdir -p $L/src $L/cache $L/tmp
OUT=$M/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}; mkdir -p $OUT
LOUT=$L/out; mkdir -p $LOUT
note() { echo "$(TZ=Europe/Berlin date '+%F %T %Z') $*" | tee -a $LOUT/run.log; }
tar xf $SRC_TAR -C $L/src
t0=$(date +%s); tar xf $ENV_TAR -C $L; note "env extracted in $(( $(date +%s) - t0 )) s"
ENVDIR=$(ls -d $L/env* | head -1)
site=$(echo $ENVDIR/lib/python3.*/site-packages)
export PATH=$ENVDIR/bin:/usr/bin:/bin VIRTUAL_ENV=$ENVDIR PYTHONPATH=$L/src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$L/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$B/baseline TMPDIR=$L/tmp FASTMACE_SOURCE_COMMIT=$(cat $L/src/.source_commit)
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
CK=$B/models/mace-omat-0-medium.model
CIF=$L/src/results/loup-a100-20261103/SrTiO3.cif
cd $L/src
note "host $(hostname) job $SLURM_JOB_ID src $FASTMACE_SOURCE_COMMIT $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
sync_out() { cp -r $LOUT/. $OUT/; }
trap sync_out EXIT
run() { name=$1; shift; note "== $name"; t0=$(date +%s); "$@" > $LOUT/$name.log 2>&1; code=$?
  note "end $name exit=$code took=$(( $(date +%s) - t0 )) s"; tail -2 $LOUT/$name.log | cut -c1-300 | tee -a $LOUT/run.log
  sync_out; return $code; }
