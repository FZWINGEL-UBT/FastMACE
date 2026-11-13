#!/bin/bash
# Holds one local_gpu reservation for job.sh (quiet), as a systemd user unit (the WSL launch that survives).
cd /mnt/d/Margraf/FastMACE
exec python3 tools/ci/local_gpu.py run --quiet --purpose "PyTorch 2.15 trial: compat, A/B 2.14.1 vs 2.15, profiles, gate (~2 h, torch-215 session)" -- env STAGE=${STAGE:-all} bash $HOME/fm-t215/jobs/job.sh
