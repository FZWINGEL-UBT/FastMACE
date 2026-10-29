#!/usr/bin/env bash
# OPT9-T05 (wprime_backward) on a Modal A100, run from the source root by tools/modal/fastmace_modal.py::job:
#   modal run tools/modal/fastmace_modal.py::job --gpu A100 --timeout-min 25 --commit <sha> \
#     --command "bash results/opt9-T03-wprime-backward-20261029/modal.sh" --out <dir> ...
# The GPU test (tests/gpu/test_wprime_backward_cuda.py), then same-process A/Bs (tools/ab_rewrites.py, 12 blocks of
# 300 steps, a = wprime_backward=off, b = on, both build orders) on NaNbO3, NbCl5, NaCl. An A/B starts only within
# the first 19 minutes, so the job ends inside its 25-minute timeout and returns what finished.
set -u -o pipefail
start=$(date +%s)
CK=$FASTMACE_MODELS/mace-mh-1.model
log=$OUT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "fastmace=$(cat .source_commit) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
python -m pytest -p no:cacheprovider -q -rf -s tests/gpu/test_wprime_backward_cuda.py > "$OUT/check.log" 2>&1
status=$?
note "end check exit=$status"
[ $status -eq 0 ] || { note "STOP: the GPU test failed"; exit 1; }
for s in NaNbO3 NbCl5 NaCl; do for order in ab ba; do
  if [ $(( $(date +%s) - start )) -ge 1140 ]; then note "skip $s-$order (time)"; continue; fi
  if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
  note "begin $s-$order"
  python tools/ab_rewrites.py --output "$OUT/$s-$order" --checkpoint "$CK" --model mh1-r2scan --structures "$s" \
    --a wprime_backward=off --b wprime_backward=on --order "$order" --pattern "$pattern" > "$OUT/$s-$order.log" 2>&1
  note "end $s-$order exit=$?"
done; done
note DONE
