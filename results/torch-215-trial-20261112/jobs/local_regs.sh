#!/bin/bash
# Waits for the local matrix and p11 units, then holds one local_gpu reservation for regs_after.sh (systemd user unit).
sleep 30; while systemctl --user is-active -q t215-matrix || systemctl --user is-active -q t215-p11; do sleep 30; done
cd /mnt/d/Margraf/FastMACE
exec python3 tools/ci/local_gpu.py run --quiet --purpose "w' register budget A/B under NVRTC 13.0/13.5 (torch 2.15 trial, ~45 min, torch-215 session)" -- bash $HOME/fm-t215/jobs/regs_after.sh local
