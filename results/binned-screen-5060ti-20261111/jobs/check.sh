#!/bin/bash
# Read-only state check before the BINNED_SCREEN run (11 November 2026).
export PATH=/usr/lib/wsl/lib:$PATH
echo "== reservation"; python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py status 2>&1 | head -20
echo "== gpu"; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head
echo "== env"; ls -d $HOME/environments/mace_cueq_oeq/bin/python $HOME/models/mace-omat-0-medium.model 2>&1
echo "== ram"; free -g | head -2
echo "== units"; systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null | head
echo "== runs"; ls $HOME/fm-runs 2>/dev/null | tail -8
