#!/usr/bin/env bash
# 3-T14 on a workstation (leeloo): run.sh's commands with the NAS hybrid environment, every cache
# under /mnt/homes/zwingel (AGENTS.md: nothing on local disks), after an idle check of the GPU.
#
#   FASTMACE_COMMIT=<sha> bash results/3-T14-protocol/workstation.sh <run.sh arguments>
#
# Exits 3 without starting when any process is on the GPU.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
SP=$E/lib/python3.10/site-packages
cache=/mnt/homes/zwingel/fastmace-cache/3-T14-leeloo
mkdir -p "$cache"/{tmp,xdg,fastmace,inductor,triton}
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib
export FASTMACE_HYBRID_PYTHON=$E/bin/python FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export TMPDIR=$cache/tmp XDG_CACHE_HOME=$cache/xdg FASTMACE_CACHE=$cache/fastmace
export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton

pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$pids" ]; then
    for p in $pids; do ps -o user=,pid=,cmd= -p "$p" || true; done
    echo "GPU busy, not starting" >&2
    exit 3
fi
echo "== $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) $(date -u +%FT%TZ)"
bash "$here/run.sh" "$@"
echo "== done $(date -u +%FT%TZ)"
