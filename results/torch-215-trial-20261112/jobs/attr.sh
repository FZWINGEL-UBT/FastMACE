#!/bin/bash
# Step attribution on an RTX 4000 Ada workstation (conductor, 12 November 2026): production OMAT-0 medium float32 plan
# under the torch 2.15 profiling env (c: CUDA-graph annotations need 2.15 + cupti-python), SrTiO3 10^3 and NbCl5 3^3.
# Each run: fresh Inductor cache with comment_origin (the generated code names each fused kernel's FX/ATen origin and
# source lines), 200 warmup + 200 timed steps, then PROFILE_STEPS annotated steps (Cuspy trace + every node's stack).
set -u
B=/mnt/homes/zwingel/t215; R=$B/runs/$(hostname)-attr; mkdir -p $R/tmp
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -); [ -n "$gpu" ] && { echo "GPU busy [$gpu]"; exit 3; }
S=$B/src-b7c19b19; cd $S
export ENVROOT=/mnt/homes/zwingel/environments PYTHONPATH=$S FM_SRC=$S FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TMPDIR=$R/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  FASTMACE_SOURCE_COMMIT=b7c19b19 REFUSE_FALLBACK=1 MEMORY_FRACTION=0.9 CUDA_CACHE_DISABLE=1 PYSTACKS=1 PROFILE_OUT=$R/profile \
  PROFILE_STEPS=${PROFILE_STEPS:-5}
for st in SrTiO3:10x10x10 NbCl5:3x3x3; do s=${st%%:*}; c=$R/cache-$s; rm -rf $c; mkdir -p $c
  FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile XDG_CACHE_HOME=$c/xdg \
  PROFILE_TAG=attr-$s timeout 1500 bash $B/jobs/py-c.sh $B/jobs/profile_point.py point --structure $s --repeat ${st#*:} \
    --model omat0-medium --precision float32 --output $R/points/$s --warmup 200 --steps 200 --timestep-fs 1 \
    --constant "torch._inductor.config:comment_origin=True" > $R/attr-$s.log 2>&1
  echo "$s exit=$? $(grep -a '^point\|^profile' $R/attr-$s.log | cut -c1-200)"
done
