#!/usr/bin/env bash
# 4-T13 item 3: each fastmace.ase MD class against Simulation and fm.md, alternating blocks in one process
# (4-T10's run 2/3 set-up: the ASE side first, 12 blocks of 300 after 300, MH-1 NaCl 3x3x3). Run under
# local_gpu.py run --quiet. Usage: bash item3.sh SRC OUT
set -uo pipefail
src=${1:?src}; out=${2:?out}
mkdir -p "$out"
cd "$src" || exit 2
cache=$HOME/fastmace-cache/4-T13
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_MODELS=$HOME/models
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 FASTMACE_COMMIT=$(cat "$src/.source_commit")
export TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
python=$HOME/environments/mace_cueq_oeq/bin/python
for cls in Langevin VelocityVerlet Bussi NoseHooverChainNVT IsotropicMTKNPT MTKNPT NVTBerendsen NPTBerendsen \
           Inhomogeneous_NPTBerendsen; do
  echo "$(date -u +%FT%TZ) start $cls"
  "$python" tools/ab_frontends.py --integrator "$cls" --sides ase simulation md --warmup 300 --block 300 \
    --blocks 12 --output "$out/$cls" > "$out/$cls.log" 2>&1
  echo "$(date -u +%FT%TZ) end $cls exit=$?"
done
