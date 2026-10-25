#!/usr/bin/env bash
# 6-T13 item 6b: bitwise trajectories of the freeze (side 6) against fastmace-5 (side 5), as phase 5 item 4
# (tools/acceptance/phase5_local.sh, trajectories-cpu and trajectories-reproducible).
#
#   bash tools/acceptance/phase6_trajectories.sh SRC cpu|reproducible
#
# SRC: git archive of the freeze with .source_commit. Run the reproducible part under the GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "6-T13 trajectories" -- bash ... reproducible
set -euo pipefail
src=$(cd "${1:?SRC}" && pwd); part=${2:?cpu|reproducible}
root=$HOME/tmp/6-T13/traj
cache=$HOME/fastmace-cache/6-T13
hybrid=$HOME/environments/mace_cueq_oeq/bin/python
latest=$HOME/environments/mace_cueq_latest/bin/python
checkpoint=$HOME/models/mace-mh-1.model
versions=$HOME/tmp/6-T13/versions
base=d304834cf018a55c7b2e0f7f2945d0988cb38569
mkdir -p "$root" "$cache"/{tmp,fastmace,inductor,triton}
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_SOURCE_COMMIT=$(cat "$src/.source_commit") FASTMACE_COMMIT=$(cat "$src/.source_commit")
[ -d "$versions/fastmace_5" ] || (cd "$src" && "$hybrid" -m fastmace.bench.versions export --ref fastmace-5 \
  --name fastmace_5 --destination "$versions" --repository "${FASTMACE_REPOSITORY:-/mnt/d/Margraf/FastMACE}")
run() { local name=$1 status=0; shift; echo "$(date -u +%FT%TZ) start $name"
        (cd "$src" && "$@") > "$root/$name.log" 2>&1 < /dev/null || status=$?
        echo "$(date -u +%FT%TZ) end $name exit=$status"; }
case $part in
  cpu)
    tiny=$root/tiny-mh1-11-17.model
    [ -f "$tiny" ] || (cd "$src" && env CUDA_VISIBLE_DEVICES= PYTHONPATH="$src:$src/tests" "$hybrid" -c "
import sys, torch
from _support.tiny_models import FAMILIES, tiny_mace_model
torch.save(tiny_mace_model([11, 17], **FAMILIES['mh1']), sys.argv[1])" "$tiny")
    for ensemble in nvt npt; do
      for dispersion in none d3bj; do
        for side in 5 6; do
          run traj-cpu-$ensemble-$dispersion-$side env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$hybrid" \
            tools/trajectory_versions.py run --against 5 --side $side --expected-commit "$base" --renamed "$versions" \
            --device cpu --backend auto --ensemble $ensemble --dispersion $dispersion \
            --output "$root/traj-cpu-$ensemble-$dispersion-$side" --model "$tiny" --checkpoint "$tiny" \
            --structure NaCl@2x2x2 --steps 100 --every 10 --seeds 20261023 --positions
        done
        run traj-cpu-$ensemble-$dispersion-compare "$hybrid" tools/trajectory_versions.py compare \
          --release "$root/traj-cpu-$ensemble-$dispersion-5" --current "$root/traj-cpu-$ensemble-$dispersion-6" --bitwise
      done
    done ;;
  reproducible)
    for ensemble in nvt npt; do
      for side in 5 6; do
        run traj-repro-$ensemble-$side "$latest" tools/trajectory_versions.py run --against 5 --side $side \
          --expected-commit "$base" --renamed "$versions" --reproducible --backend cueq --ensemble $ensemble \
          --output "$root/traj-repro-$ensemble-$side" --checkpoint "$checkpoint" --structure NaCl --steps 500 \
          --every 50 --seeds 20261023 --positions
      done
      run traj-repro-$ensemble-compare "$latest" tools/trajectory_versions.py compare \
        --release "$root/traj-repro-$ensemble-5" --current "$root/traj-repro-$ensemble-6" --bitwise
    done ;;
  *) echo "unknown part $part" >&2; exit 2 ;;
esac
