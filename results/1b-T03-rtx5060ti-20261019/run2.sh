#!/usr/bin/env bash
# 1b-T03, second part: the float64 references of POLAR-1-S and OMOL-0 XL do not fit NbCl5 3x3x3
# (1944 atoms) in 16 GB, so all three sides again on NbCl5 2x2x2 (576 atoms); the GPU test; and the
# step-time A/B in both build orders (the first A/B, 02:52-02:55, overlapped another CPU suite and
# is discarded). The POLAR-1-S 2x2x2 reference finished in the interrupted first attempt (02:56).
set -u
ROOT=/mnt/d/Margraf/FastMACE/.claude/worktrees/1b-t03-e792e5
OUT=$ROOT/results/1b-T03-rtx5060ti-20261019
cd "$ROOT"
export PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_CACHE=$HOME/fastmace-cache
CUEQ=$HOME/environments/mace_cueq_latest/bin/python
OEQ=$HOME/environments/mace_cueq_oeq/bin/python
NBCL5=$FASTMACE_STRUCTURES/Structures/Benchmark/R2SCAN/NbCl5.cif@2x2x2
(while true; do echo "$(date +%T) $(free -m | awk '/Mem:/ {print "used", $3, "available", $7}') $(cat /proc/loadavg)"; sleep 10; done) > "$OUT/free2.log" &
MEMLOG=$!
trap 'kill $MEMLOG' EXIT

run() { echo "== $(date +%T) $*"; "$@"; echo "== exit $?"; }

for model in polar1-s omol0-xl; do
    [ -s "$OUT/$model-NbCl5-2x2x2-reference.json" ] || \
        run "$CUEQ" tools/energy64_gpu.py energy --side reference --model "$model" --structure "$NBCL5" \
            --output "$OUT/$model-NbCl5-2x2x2-reference.json"
    run "$OEQ" tools/energy64_gpu.py energy --side production --model "$model" --structure "$NBCL5" \
        --output "$OUT/$model-NbCl5-2x2x2-float64sum.json"
    run "$OEQ" tools/energy64_gpu.py energy --side production --float64-energy off --model "$model" \
        --structure "$NBCL5" --output "$OUT/$model-NbCl5-2x2x2-1asum.json"
done
run "$CUEQ" -m pytest -q -p no:cacheprovider -s tests/gpu/test_energy_gpu.py
rm -rf "$OUT/ab-NaCl-overlapped" && mv "$OUT/ab-NaCl" "$OUT/ab-NaCl-overlapped" 2>/dev/null
run "$OEQ" tools/energy64_gpu.py ab --model mh1-r2scan --structure NaCl --first float64_sum --output "$OUT/ab-NaCl-f64-first"
run "$OEQ" tools/energy64_gpu.py ab --model mh1-r2scan --structure NaCl --first float32_sum_1a --output "$OUT/ab-NaCl-1a-first"
echo "== done $(date +%T)"
