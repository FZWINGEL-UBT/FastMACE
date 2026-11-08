# Sourced by the omat0-ada-capacity job files (8 November 2026, leto / murphy, RTX 4000 Ada): environment, idle
# check and the ten H200/A100 switches plus JOINED_LAYOUT (H200, 73e421ea) as MODULE:NAME=VALUE (results/a100-omat0-best-20261108/jobs/common.sh's set).
s=fastmace.kernels.oeq_streaming.spline_forward
w=fastmace.kernels.oeq_streaming.wprime_backward
c=fastmace.kernels.triton.compose_skip
q=fastmace.rewrites.cueq_projection
t=fastmace.rewrites.species
ALL="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $w:SPLINE_VECTOR=4 $w:STAGE_SH=True $c:COMPOSE_SKIP=True $c:COMPOSE_FAST=True $c:COUNT_PLAN=True $q:CUEQ_PROJECTED=True $t:GENERIC_FOLD_READOUT_TAIL=True $c:JOINED_LAYOUT=True"
# The Ada needs two prerequisites the H200 and A100 already have on (probes, README): HORNER for the spline (its w'
# kernel reads the spline only with TABLE_W + HORNER), and the tf32x3_gemms rewrite, which is where COMPOSE_SKIP and
# CUEQ_PROJECTED are routed (rewrites/precision.py). KC lets the CUTLASS kernels load on sm_89 (CUTLASS_CAPABILITIES).
H="$w:HORNER=True"
T3="tf32x3_gemms=on"
KC="fastmace.kernels.triton.tf32x3_gemm:CUTLASS_CAPABILITIES=((8,0),(8,9),(9,0))"
SPL="$H $s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $w:SPLINE_VECTOR=4 $w:STAGE_SH=True"
CMP="$T3 $c:COMPOSE_SKIP=True $c:COMPOSE_FAST=True $c:COUNT_PLAN=True $c:JOINED_LAYOUT=True $q:CUEQ_PROJECTED=True"
TAIL="$t:GENERIC_FOLD_READOUT_TAIL=True"
B=/mnt/homes/zwingel/fm-ada-capacity
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
SITE=$E/lib/python3.10/site-packages
ROOT=${ROOT:?set ROOT}
C=$B/cache-$(hostname)
mkdir -p "$ROOT/tmp" "$C"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model
ST=results/loup-a100-20261103/SrTiO3.cif
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" | tee -a "$log"; }
others() {  # someone else on the GPU, a busy CPU or a pytest run
  local gpu busy suite
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  suite=$(pgrep -f "[p]ytest" | paste -sd, -)
  [ -n "$gpu$busy$suite" ] && { echo "gpu=[$gpu] busy=[$busy] pytest=[$suite]"; return 0; }
  return 1
}
