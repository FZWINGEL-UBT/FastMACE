#!/bin/bash
# Local 5060 Ti smoke of GHOIST (compile, launch, MD runs): SPLINE_W/R128/STAGE_EDGES forced on both sides.
cd /mnt/d/Margraf/fm-a100-colab
export PYTHONPATH=$PWD
M=fastmace.kernels.oeq_streaming.spline_forward
PY=~/environments/mace_cueq_oeq/bin/python
OUT=${1:-/home/fzwin/fastmace-runs/a100-colab-ideas-20261111/smoke-ghoist}
rm -rf "$OUT"
$PY tools/ab_rewrites.py --output "$OUT" --checkpoint ~/models/mace-omat-0-medium.model --model omat0-medium \
  --structures results/loup-a100-20261103/SrTiO3.cif@6x6x6 --timestep-fs 1 --warmup 50 --block 64 --blocks 2 \
  --a-constant $M:SPLINE_W=True $M:R128=True $M:STAGE_EDGES=True \
  --b-constant $M:SPLINE_W=True $M:R128=True $M:STAGE_EDGES=True $M:GHOIST=True
