#!/bin/bash
# Smoke: the probe on the 135-atom cell only, 2 directions, short line and loop, configurations a, b, c.
source $(dirname $0)/local_common.sh
note "smoke $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
for c in a b c; do
  K=""; [ $c = c ] && K="--knots-from $OUT/probe/probe-a.json"
  step smoke-$c 900 $PY tools/consistency_probe.py --config $c --output $OUT/probe --checkpoint $CK --cif $CIF \
    --repeats 3x3x3 --directions 2 --line-A 0.002 --loop-steps 30 --candidates 50 $K
done
note DONE
