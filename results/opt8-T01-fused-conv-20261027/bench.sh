#!/usr/bin/env bash
# usage: bench.sh OUT.jsonl "VARIANTS" "FUSED_CONFIG" BLOCKS ROUNDS STRUCTURES...
cd ~/tmp/opt8
source ./env.sh
out=$1; variants=$2; cfg=$3; blocks=$4; rounds=$5; shift 5
files=""
for s in "$@"; do files="$files ops-mh1-r2scan-$s.pt"; done
$P /mnt/d/Margraf/fm-opt8/tools/bench/opt8_fused_conv.py $files --output $out --variants $variants \
   --fused-config "$cfg" --blocks $blocks --rounds $rounds 2>&1 | grep -v -E "Warning|pynvml|warn\(|^\s*$|import"
