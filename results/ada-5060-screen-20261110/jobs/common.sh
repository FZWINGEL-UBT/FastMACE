# Sourced by the ada-5060-screen job files (10 November 2026): environment, idle check and the switch shorthands
# (MODULE:NAME=VALUE for tools/ab_rewrites.py --a/--b-constant and --pack).
# HOST=ada (leto / murphy / leeloo, shared home) or HOST=local (the RTX 5060 Ti in WSL2, inside tools/ci/local_gpu.py run).
s=fastmace.kernels.oeq_streaming.spline_forward
w=fastmace.kernels.oeq_streaming.wprime_backward
f=fastmace.kernels.oeq_streaming.fused_forward
c=fastmace.kernels.triton.compose_skip
q=fastmace.rewrites.cueq_projection
g=fastmace.rewrites.sc_generated
t=fastmace.rewrites.species
k=fastmace.neighbors.skin
e=fastmace.model.energy
r=fastmace.kernels.cutlass.segment_rows
KC="fastmace.kernels.triton.tf32x3_gemm:CUTLASS_CAPABILITIES=((8,0),(8,9),(9,0))"
T3="tf32x3_gemms=on"
ROOT=${ROOT:?set ROOT}
if [ "${HOST:-ada}" = local ]; then
  E=$HOME/environments/mace_cueq_oeq
  C=$HOME/fastmace-cache/ada5060-local
  MODELS=$HOME/models
  STRUCT=$HOME/fastmace-baseline/1a-T18
else
  E=/mnt/homes/zwingel/environments/mace_cueq_oeq
  C=/mnt/homes/zwingel/fm-ada5060/cache-$(hostname)
  MODELS=/mnt/homes/zwingel/ResearchModule/Models
  STRUCT=/mnt/homes/zwingel/ResearchModule
fi
SITE=$E/lib/python3.10/site-packages
mkdir -p "$ROOT/tmp" "$C"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=$MODELS \
  FASTMACE_STRUCTURES=$STRUCT TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
CK=$MODELS/mace-omat-0-medium.model
ST=results/loup-a100-20261103/SrTiO3.cif
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" | tee -a "$log"; }
others() {  # workstations: someone else on the GPU, another user's busy CPU, or a pytest run
  [ "${HOST:-ada}" = local ] && return 1
  local gpu busy suite
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  suite=$(pgrep -f "[p]ytest" | paste -sd, -)
  [ -n "$gpu$busy$suite" ] && { echo "gpu=[$gpu] busy=[$busy] pytest=[$suite]"; return 0; }
  return 1
}
