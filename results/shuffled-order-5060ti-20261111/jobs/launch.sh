#!/bin/bash
# Holds one local_gpu reservation for job.sh (quiet), as a systemd user unit (the WSL launch that survives).
cd /mnt/d/Margraf/FastMACE
exec python3 tools/ci/local_gpu.py run --quiet --purpose "B4 shuffled vs ordered SrTiO3 24^3 (ada-5060-screen session)" -- bash $HOME/fm-b4/job.sh
