#!/usr/bin/env bash
# Symmetrix-XL under LAMMPS with the Benchmark_loup protocol on a Modal GPU (5 November 2026). Runs in the
# Symmetrix-XL image (fastmace_modal.py::sx_job, FASTMACE_MODAL_SX=1). Inputs in the volume:
#   /data/symmetrix/lmp-<arch>/bin/lmp   LAMMPS + Symmetrix-XL, built locally (symmetrix_build.py lammps, MPI off)
#   /data/symmetrix/mace-omat-0-medium.json   the Symmetrix export of MACE-OMAT-0 medium
# Steps: the JIT device artifact for this GPU (symmetrix_prepare_jit_device_artifact), then tools/bench/sx_loup.py:
# NVE and Langevin, 3 fresh processes each (5 + 20 steps), and one longer Langevin run (100 + 300 steps).
# Usage (from WSL, Modal client venv):
#   FASTMACE_MODAL_SX=1 modal run tools/modal/fastmace_modal.py::sx_job --gpu H200 --timeout-min 25 \
#     --commit <sha> --command "bash tools/modal/symmetrix_loup.sh" --out ~/fm-runs/sx-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
arch=${SX_ARCH:-sm90}
lmp=/data/symmetrix/lmp-$arch/bin/lmp
model=/data/symmetrix/mace-omat-0-medium.json
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
pip freeze 2>/dev/null | grep -i symmetrix >> "$log"
cu13=$(python -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13', 'lib'))")
shim=/tmp/sx-shim; mkdir -p "$shim"
ln -sf "$cu13/libnvrtc.so.13" "$shim/libnvrtc.so"  # Symmetrix-XL dlopens an unversioned libnvrtc.so
export LD_LIBRARY_PATH=$shim:$cu13:${LD_LIBRARY_PATH:-} SYMMETRIX_JIT_CACHE=/tmp/sx-jit
chmod +x "$lmp"
ldd "$lmp" > "$out/ldd.txt" 2>&1
grep -q "not found" "$out/ldd.txt" && note "missing libraries: $(grep 'not found' "$out/ldd.txt" | tr '\n' ' ')"
symmetrix_prepare_jit_device_artifact --model "$model" --precision float32 --lammps-arguments \
  > "$out/jit_arguments.txt" 2> "$out/jit-prepare.log"
note "jit exit=$? $(cut -c1-300 "$out/jit_arguments.txt")"
jit=$(cat "$out/jit_arguments.txt")
run() {
  local name=$1; shift
  python tools/bench/sx_loup.py --lmp "$lmp" --model-json "$model" --jit-arguments "$jit" --structure "$st" \
    --workdir "$out/work-$name" --output "$out/sx/$name.json" "$@" > "$out/$name.log" 2>&1
  note "$name exit=$? $(tail -n 1 "$out/$name.log" | cut -c1-400)"
}
for ens in nve langevin; do
  for t in 1 2 3; do run "$ens-$t" --ensemble "$ens"; done
done
run langevin-long --ensemble langevin --warmup 100 --steps 300
note done
