#!/usr/bin/env bash
# usage: sweep.sh OUT.jsonl STRUCTURES -- configs from sweep_configs.txt (one per line)
cd ~/tmp/opt8
out=$1; shift
while read -r cfg; do
  [ -z "$cfg" ] && continue
  echo "== $cfg"
  bash bench.sh $out fused "$cfg" 5 1 "$@" | python3 summarize.py
done < sweep_configs.txt
