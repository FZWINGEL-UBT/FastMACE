#!/bin/bash
# Measurements 1-3 (directional derivatives, knot crossings, closed loop) for configurations a, b, c on one GPU.
# c reuses a's line (atom, direction, knots). Each configuration in its own process (constants set before the build).
source $(dirname $0)/local_common.sh
note "probe $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
LIMIT=${LIMIT:-3600}
for c in a b c; do
  K="--knots-from $OUT/probe/probe-a.json"; [ $c = a ] && K=""
  step probe-$c $LIMIT $PY tools/consistency_probe.py --config $c --output $OUT/probe --checkpoint $CK --cif $CIF $K
done
note DONE
