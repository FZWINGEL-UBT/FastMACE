#!/bin/bash
# 1a-T17 GPU runs: main.py --engine gpu (research, through oeq_no_nvcc.sh) and --engine fastmace.
# usage: t17_gpu.sh <engine> <run-name> [main.py args...]
set -u
ENGINE=$1; NAME=$2; shift 2
E=/mnt/d/Margraf/ResearchModule-wt/1a-T17/Experiments/Langevin_Thermostat
OEQ_ENV=$HOME/environments/mace_cueq_oeq
ROOT=$HOME/runs/1a-T17-20261018
mkdir -p "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MACE_COMPILE_CACHE=$HOME/compile-cache/1a-T17-$ENGINE
CKPT=$HOME/models/mace-mh-1.model
if [ "$ENGINE" = gpu ]; then
  bash "$E/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh" "$OEQ_ENV" "$HOME/compile-cache/1a-T17-oeq" "$OEQ_ENV/bin/python" \
    "$E/main.py" --engine gpu --checkpoint "$CKPT" "$@"
else
  PYTHONPATH=/mnt/d/Margraf/FastMACE "$OEQ_ENV/bin/python" "$E/main.py" --engine fastmace --checkpoint "$CKPT" "$@"
fi
