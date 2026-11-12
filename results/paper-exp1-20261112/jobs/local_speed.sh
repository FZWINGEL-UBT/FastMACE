#!/bin/bash
# Measurement 5 on the RTX 5060 Ti: speed.sh inside the local guards (each order its own step, 3,600 s limit).
source $(dirname $0)/local_common.sh
note "speed $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src $FASTMACE_SOURCE_COMMIT"
mkdir -p $OUT/speed
source $(dirname $0)/speed.sh
note DONE
