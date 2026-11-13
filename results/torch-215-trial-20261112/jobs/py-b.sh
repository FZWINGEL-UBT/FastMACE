#!/bin/bash
# interpreter of environment b: ~/environments/mace_cueq_oeq_torch215, with its own CUDA libraries first
E=${ENVROOT:-$HOME/environments}/mace_cueq_oeq_torch215
SITE=$E/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:/usr/lib/wsl/lib${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}
export FASTMACE_HYBRID_PYTHON=$E/bin/python
exec $E/bin/python "$@"
