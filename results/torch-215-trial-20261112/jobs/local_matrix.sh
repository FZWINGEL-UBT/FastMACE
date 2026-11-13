#!/bin/bash
# Holds one local_gpu reservation for matrix.sh on the RTX 5060 Ti (12 November 2026), as a systemd user unit.
# c4/c8 are cuBLAS overlays on production (py-c4o/c8o.sh: no WSL disk for clones). Card guard 14 GiB of 16.
cd /mnt/d/Margraf/FastMACE
export RUN=$HOME/fm-t215 NV=$HOME/fm-t215/census CARD_LIMIT_MIB=14000 CUDA_CACHE_DISABLE=1 \
  VARIANTS="a:a:- c4:c4o:- c8:c8o:- t9:t9:- n2:a:13.2.86 n4:a:13.4.92 n5:a:13.5.37 b:b:- b0:b:13.0.88"
exec python3 tools/ci/local_gpu.py run --quiet --purpose "CUDA library matrix (torch 2.15 trial): 9 variants x SrTiO3/NbCl5 + profiles (~1.5 h, torch-215 session)" -- bash $HOME/fm-t215/jobs/matrix.sh
