#!/usr/bin/env bash
# fp32 profile (owner request, 28 October 2026) on Modal: MH-1 float32 default plan, NaCl, NaNbO3 and NbCl5 3x3x3,
# 1 fs, 1000 K. Modal's image has no Nsight tools, so this is the torch.profiler part of
# results/fp32-profile-20261028/run.sh: step_breakdown's main pass, fp32_parts, and fp32_setup warm (the volume's
# per-GPU cache) and cold (fresh caches under /tmp) on NaCl. Every record is labelled with the GPU and "Modal".
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 45 --commit <sha> \
#     --command "bash tools/modal/fp32_profile.sh" --out ~/fastmace-runs/fp32-profile-20261028/modal-L4
set -uo pipefail
out=${OUT:?the job sets OUT}
ckpt=$FASTMACE_MODELS/mace-mh-1.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
run() {  # name command...
  local name=$1; shift
  note "start $name gpu=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader)"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit) host $(hostname)"
for s in NaCl NaNbO3 NbCl5; do
  run "$s" python tools/step_breakdown.py --output "$out/$s" --checkpoint "$ckpt" --model mh1-r2scan --backend hybrid \
    --temperature-K 1000 --timestep-fs 1 --structure "$s"
  run "$s-parts" python tools/bench/fp32_parts.py --output "$out/$s/parts" --checkpoint "$ckpt" --structure "$s" \
    --temperature-K 1000 --timestep-fs 1
  rm -f "$out/$s/trace.json" "$out/$s/parts/trace.json"  # large; the tables keep what the report uses
done
for s in NaCl NaNbO3 NbCl5; do
  run "setup-$s-warm" python tools/bench/fp32_setup.py --output "$out/setup-$s-warm" --structure "$s" --label warm
done
cold=/tmp/fp32-cold-cache
mkdir -p "$cold"
run setup-NaCl-cold env FASTMACE_CACHE=$cold TORCHINDUCTOR_CACHE_DIR=$cold/inductor TRITON_CACHE_DIR=$cold/triton \
  MACE_COMPILE_CACHE=$cold/compile XDG_CACHE_HOME=$cold/xdg \
  python tools/bench/fp32_setup.py --output "$out/setup-NaCl-cold" --structure NaCl --label cold
rm -rf "$out"/setup-*/run/*.traj "$out"/setup-*/run/trajectory* 2>/dev/null
note done
