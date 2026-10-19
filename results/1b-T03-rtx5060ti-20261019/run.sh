#!/usr/bin/env bash
# 1b-T03 GPU runs on the local RTX 5060 Ti (WSL), conductor slot of 19/20 October 2026.
# Launch: setsid nohup bash results/1b-T03-rtx5060ti-20261019/run.sh > <dir>/run.log 2>&1 &
set -u
ROOT=/mnt/d/Margraf/FastMACE/.claude/worktrees/1b-t03-e792e5
OUT=$ROOT/results/1b-T03-rtx5060ti-20261019
cd "$ROOT"
export PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_CACHE=$HOME/fastmace-cache
CUEQ=$HOME/environments/mace_cueq_latest/bin/python
OEQ=$HOME/environments/mace_cueq_oeq/bin/python
(while true; do echo "$(date +%T) $(free -m | awk '/Mem:/ {print "used", $3, "available", $7}') $(cat /proc/loadavg)"; sleep 10; done) > "$OUT/free.log" &
MEMLOG=$!
trap 'kill $MEMLOG' EXIT

run() { echo "== $(date +%T) $*"; "$@"; echo "== exit $?"; }

for case in polar1-s:NbCl5 omol0-xl:NbCl5 mh1-r2scan:NbCl5 mh1-r2scan:NaCl; do
    model=${case%%:*}; structure=${case##*:}
    run "$CUEQ" tools/energy64_gpu.py energy --side reference --model "$model" --structure "$structure" \
        --output "$OUT/$model-$structure-reference.json"
    run "$OEQ" tools/energy64_gpu.py energy --side production --model "$model" --structure "$structure" \
        --output "$OUT/$model-$structure-float64sum.json"
    run "$OEQ" tools/energy64_gpu.py energy --side production --float64-energy off --model "$model" \
        --structure "$structure" --output "$OUT/$model-$structure-1asum.json"
done
run "$CUEQ" -m pytest -q -p no:cacheprovider -s tests/gpu/test_energy_gpu.py
run "$OEQ" tools/energy64_gpu.py ab --model mh1-r2scan --structure NaCl --output "$OUT/ab-NaCl"
echo "== done $(date +%T)"
