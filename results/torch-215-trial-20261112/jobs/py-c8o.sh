#!/bin/bash
# interpreter of variant c8 where no cloned env fits (local WSL disk): production mace_cueq_oeq with nvidia-cublas
# 13.8.2.12 from an overlay directory first on LD_LIBRARY_PATH and on PYTHONPATH (torch preloads its CUDA libraries from the first
# nvidia/cu13/lib on sys.path, so LD_LIBRARY_PATH alone loses; checked in /proc/self/maps)
E=${ENVROOT:-$HOME/environments}/mace_cueq_oeq
O=${OVERLAY:-$HOME/fm-t215/overlay}/cublas-13.8.2.12/nvidia/cu13/lib
SITE=$E/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$O:$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:/usr/lib/wsl/lib${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}
export PYTHONPATH=${O%/nvidia/cu13/lib}${PYTHONPATH:+:$PYTHONPATH}
export FASTMACE_HYBRID_PYTHON=$E/bin/python
exec $E/bin/python "$@"
