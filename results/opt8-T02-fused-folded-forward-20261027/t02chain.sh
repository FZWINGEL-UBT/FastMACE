#!/usr/bin/env bash
# OPT8-T02 gates on the local GPU, one short reservation per group (FIFO with the conductor).
source ~/tmp/opt8/env.sh
FM=$HOME/fm-src/opt8-T02-9eb1ab1e
ROOT=$HOME/tmp/opt8/t02
R="env PYTHONPATH=/mnt/d/Margraf/FastMACE $P /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet"
for group in "gate" "drift-nacl-off drift-nacl-on" "drift-nbcl5-off drift-nbcl5-on" "nacl-ab nacl-ba" \
             "nbcl5-ab nbcl5-ba" "nanbo3-ab nanbo3-ba" "polar-nbcl5-ab omol-nbcl5-ab"; do
  $R --purpose "fm-opt8 T02 $group" -- env FM=$FM bash ~/tmp/opt8/t02local.sh $ROOT $group
  echo "group done: $group $(date -u +%T)"
done
echo CHAIN DONE
