#!/bin/bash
# tf32x3 GEMM pipeline stages screen on an RTX 4000 Ada (conductor, 12 November 2026): production env a (torch 2.14.1,
# Triton 3.8), production plan, SrTiO3 10^3 and NbCl5 3^3; stages_lab.py replays every recorded Triton GEMM launch
# with num_stages 3 and 2 in one process (ABBA rounds).
set -u
B=/mnt/homes/zwingel/t215; R=$B/runs/$(hostname)-stages; mkdir -p $R/tmp
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -); [ -n "$gpu" ] && { echo "GPU busy [$gpu]"; exit 3; }
S=$B/src-b7c19b19; cd $S
export ENVROOT=/mnt/homes/zwingel/environments PYTHONPATH=$S FM_SRC=$S FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TMPDIR=$R/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  FASTMACE_SOURCE_COMMIT=b7c19b19 REFUSE_FALLBACK=1 MEMORY_FRACTION=0.9 CUDA_CACHE_DISABLE=1 STAGES_OUT=$R ROUNDS=20
for st in SrTiO3:10x10x10 NbCl5:3x3x3; do s=${st%%:*}; c=$R/cache-$s; mkdir -p $c
  FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile XDG_CACHE_HOME=$c/xdg \
  STAGES_TAG=$s timeout 1800 bash $B/jobs/py-a.sh $B/jobs.stages/stages_lab.py point --structure $s --repeat ${st#*:} \
    --model omat0-medium --precision float32 --output $R/points/$s --warmup 100 --steps 100 --timestep-fs 1 > $R/stages-$s.log 2>&1
  echo "$s exit=$? $(grep -a '^point\|^stages' $R/stages-$s.log | cut -c1-900)"
done
