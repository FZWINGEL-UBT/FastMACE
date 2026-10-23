#!/bin/bash
# drive.sh GPU (A100|G4|L4): one Colab VM, setup, probe, download, stop.
set -u
GPU=$1; g=$(echo "$GPU" | tr A-Z a-z)
R=$HOME/colab-runs/opt-T01-bf16x6-20261023T185355Z
D=$R/$g; S=fm-bf16x6-$g
mkdir -p "$D"; cd "$R"
colab usage > "$D/cu-before.txt" 2>&1
echo "== new $S $(date -u +%FT%TZ)"
colab new --gpu "$GPU" -s "$S" > "$D/new.log" 2>&1 || { echo "new failed"; cat "$D/new.log"; exit 1; }
colab exec -s "$S" -f "$R/mkdirs.py" --timeout 60
colab upload -s "$S" "$R/probe_bf16x6.py" /content/bf16x6/probe_bf16x6.py
echo "== setup $(date -u +%FT%TZ)"
colab exec -s "$S" -f "$R/remote_setup.py" --timeout 1500 > "$D/setup.txt" 2>&1; cat "$D/setup.txt" | tail -5
echo "== run $(date -u +%FT%TZ)"
colab exec -s "$S" -f "$R/remote_run.py" --timeout 3000 > "$D/remote.txt" 2>&1; tail -40 "$D/remote.txt"
colab download -s "$S" /content/bf16x6/result.tgz "$D/result.tgz" && tar -xzf "$D/result.tgz" -C "$D"
echo "== stop $(date -u +%FT%TZ)"
colab stop -s "$S"
colab usage > "$D/cu-after.txt" 2>&1
cat "$D/cu-after.txt"
