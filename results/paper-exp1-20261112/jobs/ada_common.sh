# Sourced by the Ada workstation (RTX 4000 Ada: leto / murphy / leeloo) job files of paper experiment 1. Shared NFS home:
# files only under /mnt/homes/zwingel. Before every step: the idle check (another user's GPU process or busy CPU, a
# pytest run, or the benchmark-loup lane on this host stops the job).
set -u
H=/mnt/homes/zwingel; P=$H/fm-paper-exp1
SRC=${SRC:?set SRC}; OUT=${OUT:?set OUT}
mkdir -p $OUT $P/tmp-$(hostname)
E=$H/environments/mace_cueq_oeq; SITE=$E/lib/python3.10/site-packages; C=$P/cache-$(hostname)
export PATH=$E/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$H/ResearchModule/Models \
  FASTMACE_STRUCTURES=$H/ResearchModule FASTMACE_CACHE=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg TMPDIR=$P/tmp-$(hostname) TORCHINDUCTOR_COMPILE_THREADS=4 \
  CUDA_CACHE_DISABLE=1 FASTMACE_HYBRID_PYTHON=$E/bin/python FASTMACE_SOURCE_COMMIT=$(cat $SRC/.source_commit) \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
CK=$H/ResearchModule/Models/mace-omat-0-medium.model
CIF=$SRC/results/loup-a100-20261103/SrTiO3.cif
cd $SRC
note() { echo "$(date '+%F %T %Z') $(hostname) $*" | tee -a $OUT/run.log; }
others() {
  local gpu busy suite loup
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  suite=$(pgrep -f "[p]ytest" | paste -sd, -)
  loup=$(pgrep -f "[b]enchmark-loup" | paste -sd, -)
  [ -n "$gpu$busy$suite$loup" ] && { echo "gpu=[$gpu] busy=[$busy] pytest=[$suite] loup=[$loup]"; return 0; }
  return 1
}
step() { name=$1; shift
  if why=$(others); then note "STOP before $name: not idle: $why"; exit 3; fi
  note "== $name"; t0=$(date +%s)
  "$@" > $OUT/$name.log 2>&1; code=$?
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s"
  [ $code != 0 ] && { tail -20 $OUT/$name.log | cut -c1-300 | tee -a $OUT/run.log; note "STOP after $name"; exit 1; }
  return 0; }
PY=python
