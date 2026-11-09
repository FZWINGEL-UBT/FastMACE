#!/usr/bin/env bash
# leto: the new GPU test against the old source (expected to fail) and the whole test file on the fixed source, with
# the NAS mace_cueq_latest Python (it has pytest), alone on the GPU (idle wait as leto.sh).
#   ROOT=... OLD=<source> NEW=<source> bash gputest.sh
set -u
E=/mnt/homes/zwingel/environments/mace_cueq_latest
C=/mnt/homes/zwingel/fm-memprofile/cache-latest-$(hostname)
mkdir -p "$ROOT/tmp" "$C"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton XDG_CACHE_HOME=$C/xdg TMPDIR=$ROOT/tmp CUDA_CACHE_DISABLE=1
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
busy() {
  local gpu users
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  users=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$users" ] && { echo "gpu=[$gpu] users=[$users]"; return 0; }
  return 1
}
quiet=0; while [ $quiet -lt 4 ]; do if busy > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi; sleep 30; done
note "idle (gputest)"
T=tests/gpu/test_round2_kernels_cuda.py
(cd "$OLD" && PYTHONPATH=$OLD python -m pytest -p no:cacheprovider -q "$NEW/$T" -k recomputing --basetemp "$ROOT/tmp/pt-old") > "$ROOT/test-old.log" 2>&1 < /dev/null
note "test on old source exit=$? $(tail -n 1 $ROOT/test-old.log)"
(cd "$NEW" && PYTHONPATH=$NEW python -m pytest -p no:cacheprovider -q "$T" --basetemp "$ROOT/tmp/pt-new") > "$ROOT/test-new.log" 2>&1 < /dev/null
note "test file on new source exit=$? $(tail -n 1 $ROOT/test-new.log)"
note "GPUTEST DONE"
