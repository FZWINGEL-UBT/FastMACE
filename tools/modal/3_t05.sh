#!/usr/bin/env bash
# 3-T05 (rule 11) skin A/B on Modal: tools/ab_skin.py same-process A/Bs, fixed 0.3 A against rule 11's skin,
# 12 blocks of 300 steps, both build orders, NaCl and NbCl5, MH-1 hybrid float32 NVT at 1 fs. Every record is
# labelled "Modal <GPU>". Usage (from WSL, Modal client venv; budget $2, guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 50 --commit <sha> \
#     --command "bash tools/modal/3_t05.sh" --out ~/fastmace-runs/3-t05/modal-L4
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
ab() {  # name temperature structure order pattern
  note "start $1 gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  python tools/ab_skin.py --output "$out/$1" --checkpoint "$checkpoint" --structures "$3" --temperature-K "$2" \
    --timestep-fs 1.0 --order "$4" --pattern "$5" > "$out/$1.log" 2>&1 < /dev/null
  note "end $1 exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader); source $(cat .source_commit) host $(hostname)"
for structure in NaCl NbCl5; do
  ab skin-1500K-$structure-ab 1500 $structure ab abab
  ab skin-1500K-$structure-ba 1500 $structure ba abba
done
for structure in NaCl NbCl5; do
  ab skin-1000K-$structure-ab 1000 $structure ab abab
done
note done
grep -h "b/a" "$out"/*.log 2>/dev/null | cut -c1-200
