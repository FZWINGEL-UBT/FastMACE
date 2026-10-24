#!/usr/bin/env bash
# opt-4 baseline: tools/step_breakdown.py main pass + nsys pass for one structure, main ccab528,
# MH-1 r2SCAN hybrid float32, 3x3x3, local RTX 5060 Ti. Run inside a local_gpu.py reservation.
# Usage: opt4_breakdown.sh STRUCTURE
set -uo pipefail
s=$1
src=$HOME/fastmace-opt4/src-ccab528
run=$HOME/fastmace-opt4/breakdown-ccab528/$s
export PYTHONPATH=$src FASTMACE_MODELS=$HOME/models
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
py=$HOME/environments/mace_cueq_oeq/bin/python
ckpt=$HOME/models/mace-mh-1.matpes_r2scan.model
cd "$src"
mkdir -p "$(dirname "$run")"
echo "$(date -u +%FT%TZ) main $s"
timeout 600 "$py" tools/step_breakdown.py --output "$run" --checkpoint "$ckpt" --structure "$s" \
  > "$run.main.log" 2>&1 < /dev/null || { echo "main failed: $(tail -3 "$run.main.log")"; exit 1; }
echo "$(date -u +%FT%TZ) nsys $s"
timeout 400 nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --force-overwrite true \
  -o "$run/nsys" "$py" tools/step_breakdown.py --pass nsys --output "$run" --checkpoint "$ckpt" --structure "$s" \
  > "$run.nsys.log" 2>&1 < /dev/null || { echo "nsys failed: $(tail -3 "$run.nsys.log")"; exit 0; }
nsys stats --report cuda_gpu_trace --format csv --force-export true -o "$run/nsys" "$run/nsys.nsys-rep" \
  >> "$run.nsys.log" 2>&1
"$py" tools/step_breakdown.py --output "$run" --summarize-nsys "$run/nsys_cuda_gpu_trace.csv" >> "$run.nsys.log" 2>&1
echo "$(date -u +%FT%TZ) done $s"
