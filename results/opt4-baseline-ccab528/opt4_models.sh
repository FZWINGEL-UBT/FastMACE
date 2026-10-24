#!/usr/bin/env bash
# opt-4 baseline, other model families: tools/step_breakdown.py main pass on NaCl 3x3x3, main ccab528,
# float32, local RTX 5060 Ti. Run inside a local_gpu.py reservation.
# Usage: opt4_models.sh LABEL CHECKPOINT ENV BACKEND
set -uo pipefail
label=$1 ckpt=$HOME/models/$2 env=$3 backend=$4
src=$HOME/fastmace-opt4/src-ccab528
run=$HOME/fastmace-opt4/breakdown-ccab528/models/$label-NaCl
export PYTHONPATH=$src FASTMACE_MODELS=$HOME/models
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
py=$HOME/environments/$env/bin/python
cd "$src"
mkdir -p "$(dirname "$run")"
echo "$(date -u +%FT%TZ) main $label $backend"
timeout 420 "$py" tools/step_breakdown.py --output "$run" --checkpoint "$ckpt" --model "$label" --backend "$backend" \
  --structure NaCl --steps 500 > "$run.log" 2>&1 < /dev/null
echo "$(date -u +%FT%TZ) exit=$? $label: $(tail -2 "$run.log" | tr '\n' ' ' | cut -c1-300)"
