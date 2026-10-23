#!/usr/bin/env bash
# Run FastMACE's GEMM precision bench (OPT-T01) on a Colab A100, same shapes as 20 October (2-T20a breakdown).
set -u
S=fm-tf32x3-a100
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/colab-runs/opt-T01-a100-$(date -u +%Y%m%dT%H%M%SZ)
BD=$HOME/fastmace-acceptance/2-T20a-breakdown-20261020-1715
mkdir -p "$OUT"
flt() { grep -v "new version\|colab update\|silence"; }
git -C /mnt/d/Margraf/FastMACE -c core.autocrlf=false archive --format=tar origin/main > "$OUT/fastmace.tar"
git -C /mnt/d/Margraf/FastMACE rev-parse origin/main > "$OUT/commit.txt"
colab usage 2>&1 | flt | head -1 > "$OUT/cu-before.txt"
colab new --gpu A100 -s $S 2>&1 | flt
trap 'colab stop -s $S 2>&1 | flt' EXIT
colab exec -s $S -f "$HERE/mkdirs.py" 2>&1 | flt
colab upload -s $S "$OUT/fastmace.tar" /content/tf32x3/fastmace.tar 2>&1 | flt
colab upload -s $S "$BD/NaCl/gemms.csv" /content/tf32x3/g/NaCl/gemms.csv 2>&1 | flt
colab upload -s $S "$BD/NbCl5/gemms.csv" /content/tf32x3/g/NbCl5/gemms.csv 2>&1 | flt
colab exec -s $S -f "$HERE/remote_run.py" --timeout 3600 2>&1 | flt | tee "$OUT/remote.txt"
colab download -s $S /content/tf32x3/result.tgz "$OUT/result.tgz" 2>&1 | flt
tar -xzf "$OUT/result.tgz" -C "$OUT" && echo "results in $OUT"
colab usage 2>&1 | flt | head -1 > "$OUT/cu-after.txt"
