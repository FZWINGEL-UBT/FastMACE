#!/bin/bash
# Waits for the local matrix unit, then holds one local_gpu reservation for p11_after.sh (systemd user unit).
while systemctl --user is-active -q t215-matrix; do sleep 30; done
cd /mnt/d/Margraf/FastMACE
exec python3 tools/ci/local_gpu.py run --quiet --purpose "Python 3.11 vs 3.10 vs Triton 3.9 (torch 2.15 trial, ~30 min, torch-215 session)" -- bash $HOME/fm-t215/jobs/p11_after.sh local
