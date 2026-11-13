# Sourced by the Alex (A100-SXM4-80GB) job files of paper experiment 1 (route of the first-interaction lane, 12 Nov
# 2026): the installed env fm-omat0small/env (torch 2.14.1+cu130, mace-torch 0.3.16, cuEquivariance 0.12.0,
# OpenEquivariance 0.7.0), a slim source tar (tools/jobs/slim_tar.py) extracted to node-local $TMPDIR, every cache on
# $TMPDIR (removed at the end), records copied to $M/runs/<job name>-<job id>. The job files sit beside the tar in $M
# (the slim tar drops results/).
unset SLURM_EXPORT_ENV
set -uo pipefail
A=/home/atuin/v130cb/v130cb18
M=$A/fm-paper-exp1
E=$A/fm-omat0small
SRC_TAR=${SRC_TAR:?set SRC_TAR}
module purge >/dev/null 2>&1
L=$TMPDIR/fm; mkdir -p $L/src $L/cache $L/tmp $L/out
OUT=$M/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}; mkdir -p $OUT
LOUT=$L/out
note() { echo "$(TZ=Europe/Berlin date '+%F %T %Z') $*" | tee -a $LOUT/run.log; }
tar xf $SRC_TAR -C $L/src
site=$(echo $E/env/lib/python3.*/site-packages)
export PATH=$E/env/bin:/usr/bin:/bin VIRTUAL_ENV=$E/env PYTHONPATH=$L/src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$L/cache TRITON_CACHE_DIR=$L/cache/triton TORCHINDUCTOR_CACHE_DIR=$L/cache/inductor \
  XDG_CACHE_HOME=$L/cache/xdg MACE_COMPILE_CACHE=$L/cache/compile FASTMACE_MODELS=$A/fm-arena/models \
  FASTMACE_STRUCTURES=$E/baseline TMPDIR=$L/tmp FASTMACE_SOURCE_COMMIT=$(cat $L/src/.source_commit)
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
CK=$A/fm-arena/models/mace-omat-0-medium.model
CIF=$L/src/results/loup-a100-20261103/SrTiO3.cif
cd $L/src
note "host $(hostname) job $SLURM_JOB_ID src $FASTMACE_SOURCE_COMMIT $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader)"
sync_out() { cp -r $LOUT/. $OUT/; }
finish() { sync_out; rm -rf $L; }
trap finish EXIT
# run NAME CMD...: one step, its log on $TMPDIR, records synced after each step
run() { name=$1; shift; note "== $name"; t0=$(date +%s); "$@" > $LOUT/$name.log 2>&1; code=$?
  note "end $name exit=$code took=$(( $(date +%s) - t0 )) s"; tail -2 $LOUT/$name.log | cut -c1-300 | tee -a $LOUT/run.log
  sync_out; return $code; }
