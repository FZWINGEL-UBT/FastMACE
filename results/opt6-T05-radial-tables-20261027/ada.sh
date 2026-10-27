#!/usr/bin/env bash
# Owner decision 62 (27 October 2026): radial_tables default; the Ada check on a shared workstation (leto, murphy,
# leeloo: RTX 4000 Ada, sm_89). Same-process A/B (tools/ab_rewrites.py), a = radial_tables=off, b = on, both build
# orders, 12 blocks of 300 steps, MH-1 hybrid float32 NVT 300 K. Run from a scratch folder under /mnt/homes/zwingel
# holding a git archive of the source (FM); every job first checks that no other user's job runs and the GPU has no
# compute process (AGENTS.md, "Shared workstations"), and stops the sequence otherwise.
#   bash ada.sh ROOT FM
set -u -o pipefail
ROOT=$1 FM=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton MACE_COMPILE_CACHE=$ROOT/cache/compile XDG_CACHE_HOME=$ROOT/cache/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {  # other users' processes using CPU (a job), or any GPU compute process
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
for s in NaCl NbCl5 NaNbO3; do for order in ab ba; do
  if why=$(others); then note "STOP before $s-$order: $why"; exit 3; fi
  note "begin $s-$order"
  $E/bin/python $FM/tools/ab_rewrites.py --output "$ROOT/$s-$order" --checkpoint "$CK" --structures "$s" \
    --a radial_tables=off --b radial_tables=on --order "$order" > "$ROOT/$s-$order.log" 2>&1
  note "end $s-$order exit=$?"
done; done
note DONE
