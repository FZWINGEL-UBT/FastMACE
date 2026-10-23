#!/usr/bin/env bash
# 4-T07 on the local RTX 5060 Ti (WSL): variable-cell relaxations, tools/bench/optimizer_frames.py with
# --cell-filter frechet, ASE on fm.calc against the GPU, one local GPU reservation (<= 20 min) per job, in
# sequence. Usage: tools/validation/4_t07_local.sh RUN_ID [JOB...] (an ase-* job before its gpu-* job).
# Results: results/RUN_ID/<job>.json, .npz and .log. docs/validation/4-T07.md has the findings.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
run=${1:?usage: 4_t07_local.sh RUN_ID [JOB...]}
shift
out=$repo/results/$run
mkdir -p "$out"
export PYTHONPATH=$repo FASTMACE_MODELS=$HOME/models PYTHONDONTWRITEBYTECODE=1
py=$HOME/environments/mace_cueq_oeq/bin/python
MH1="--checkpoint $HOME/models/mace-mh-1.model"
POLAR="--checkpoint $HOME/models/MACE-POLAR-1-S.model --model polar1-s --head none"
CELL="--cell-filter frechet --modes graph --skins 0.6"
A="NaCl NaNbCl6 NaTaCl6 NaNbO3 NaTaO3"
B="NbCl3O TaCl3O NbCl2O NbCl5 TaCl5"
# job: "<side> <optimizer> <model> <structures> [extra]"; the gpu job compares against ase-<rest of its name>.
declare -A JOBS=(
  [ase-lbfgs-a]="ase lbfgs MH1 $A"
  [gpu-lbfgs-a]="gpu lbfgs MH1 $A"
  [ase-lbfgs-b]="ase lbfgs MH1 $B"
  [gpu-lbfgs-b]="gpu lbfgs MH1 $B"
  [ase-lbfgs-d3]="ase lbfgs MH1 NbCl5 --dispersion d3bj"
  [gpu-lbfgs-d3]="gpu lbfgs MH1 NbCl5 --dispersion d3bj"
  [ase-fire]="ase fire MH1 NaCl"
  [gpu-fire]="gpu fire MH1 NaCl"
  [ase-polar]="ase lbfgs POLAR NbCl5"
  [gpu-polar]="gpu lbfgs POLAR NbCl5"
)
ORDER="ase-lbfgs-a gpu-lbfgs-a ase-lbfgs-b gpu-lbfgs-b ase-lbfgs-d3 gpu-lbfgs-d3 ase-fire gpu-fire ase-polar gpu-polar"
for job in ${@:-$ORDER}; do
  spec=${JOBS[$job]:?unknown job $job}
  read -r side optimizer model rest <<<"$spec"
  structures=${rest%%--*}
  extra=""
  [[ $rest == *--* ]] && extra="--${rest#*--}"
  models=$MH1
  [[ $model == POLAR ]] && models=$POLAR
  compare=""
  [[ $side == gpu ]] && compare="--compare $out/ase-${job#gpu-}.json"
  echo "== $job: $side $optimizer $model [$structures] $extra $(date -Is)"
  python3 tools/ci/local_gpu.py run --wait-minutes 600 --purpose "4-T07 $job (<20 min)" -- \
    timeout 1150 "$py" tools/bench/optimizer_frames.py --side "$side" --optimizer "$optimizer" $models $CELL \
    --structures $structures $extra $compare --output "$out/$job" >"$out/$job.log" 2>&1
  echo "   exit $? $(date -Is)"
  tail -n 12 "$out/$job.log"
done
