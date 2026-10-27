#!/usr/bin/env bash
# OPT8-T02 (fused_folded_forward) Ada check on a shared workstation (murphy: RTX 4000 Ada, sm_89). GPU unit test,
# force/stress gate (MH-1, NaCl NbCl5 NaNbO3), then same-process A/Bs (tools/ab_rewrites.py, 12 blocks of 300 steps,
# a = fused_folded_forward=off, b = on, both build orders) on NaNbO3, NbCl5, NaCl. Run from a scratch folder under
# /mnt/homes/zwingel holding a git archive of the source (FM); every job first checks that no other user's job runs
# and the GPU has no compute process (AGENTS.md, "Shared workstations"), and stops the sequence otherwise.
#   bash ada.sh ROOT FM
set -u -o pipefail
ROOT=$1 FM=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
REF=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
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
run() {  # name command...
  local name=$1 why; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
run check $E/bin/python -m pytest -p no:cacheprovider -q -rf -s "$FM/tests/gpu/test_fused_folded_forward_cuda.py"
run gate $E/bin/python "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace --checkpoint "$CK" \
  --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" --rewrites fused_folded_forward=on \
  --reference-command "$REF" --production-command "$E/bin/python"
for s in NaNbO3 NbCl5 NaCl; do for order in ab ba; do
  if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
  run "$s-$order" $E/bin/python "$FM/tools/ab_rewrites.py" --output "$ROOT/$s-$order" --checkpoint "$CK" \
    --model mh1-r2scan --structures "$s" --a fused_folded_forward=off --b fused_folded_forward=on \
    --order "$order" --pattern "$pattern"
done; done
note DONE
