#!/bin/bash
# interpreter of environment c8: ~/environments/mace_cueq_oeq, with its own CUDA libraries first
E=${ENVROOT:-$HOME/environments}/mace_cueq_oeq_cublas138
SITE=$E/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:/usr/lib/wsl/lib${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}
export FASTMACE_HYBRID_PYTHON=$E/bin/python
exec $E/bin/python "$@"
