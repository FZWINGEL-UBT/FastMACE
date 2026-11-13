#!/bin/bash
# Ada: 100 ps NVE at 1 fs for a and b from the 5060 Ti's saved 300 K state (same file, SHA-256 dcf267b3...), then the
# paired speed A/B (speed.sh, both orders). SRC=<tree> OUT=<dir> STATE=<npz> bash ada_nve300.sh
source $(dirname $0)/ada_common.sh
note "ada_nve300 $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
for c in a b; do
  step nve-$c-300K $PY tools/nve_energy.py nve --config $c --checkpoint $CK --start $STATE --temperature-K 300 --ps 100 \
    --output $OUT/nve --tag $c-300K
done
mkdir -p $OUT/speed
RUNNER=ada_speed; ada_speed() { name=$1; shift 2; step $name "$@"; }
source $(dirname $0)/speed.sh
note DONE
