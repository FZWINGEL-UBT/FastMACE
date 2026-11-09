#!/usr/bin/env bash
set -u
cd /mnt/d/Margraf/fm-dmqe-ideas
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
"$HOME/environments/mace_cueq_latest/bin/python" -c 'import fastmace,inspect; import mace.modules.radial as radial; import torch; print("torch",torch.__version__); print(inspect.getsource(radial.ZBLBasis))'
exit
