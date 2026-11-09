#!/usr/bin/env bash
# Symmetrix-XL 0.1.2 under LAMMPS with MACE-OMAT-0 small, Benchmark_loup protocol, on a Modal H200 (9 November 2026;
# owner: "one symmetrix xl run using omat 0 small on h200"). As tools/modal/symmetrix_profile.sh without Nsight:
# Symmetrix-XL image (fastmace_modal.py::sx_job, FASTMACE_MODAL_SX=1, FASTMACE_SX_VERSION=0.1.2). Inputs in the volume:
#   $SX_DIR/lmp-<arch>/bin/lmp           LAMMPS + Symmetrix-XL 0.1.2 (symmetrix_build.py lammps, MPI off)
#   $SX_DIR/mace-omat-0-small.json       symmetrix_extract_mace 0.1.2 of mace-omat-0-small.model (SHA-256 0abfde07...),
#                                        defaults (universal compact radial, 256 spline points, head omat_pbe)
# Steps: the JIT device artifact; tools/bench/sx_loup.py NVE and Langevin, 3 fresh processes each (5 + 20 steps), and
# one Langevin run of 100 + 300 steps.
set -uo pipefail
out=${OUT:?the job sets OUT}
arch=${SX_ARCH:-sm90}
dir=${SX_DIR:-/data/symmetrix-0.1.2}
lmp=$dir/lmp-$arch/bin/lmp
model=$dir/${SX_MODEL:-mace-omat-0-small.json}
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null); model $model $(sha256sum "$model" | cut -c1-16)"
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H200; then note "WRONG-GPU"; exit 3; fi
pip freeze 2>/dev/null | grep -i symmetrix >> "$log"
cu13=$(python -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13', 'lib'))")
shim=/tmp/sx-shim; mkdir -p "$shim"
# Symmetrix-XL dlopens unversioned libnvrtc.so and libcudart.so (the wheels ship only the .so.13 names).
for lib in nvrtc cudart cublas cublasLt; do
  [ -e "$cu13/lib$lib.so.13" ] && ln -sf "$cu13/lib$lib.so.13" "$shim/lib$lib.so"
done
# 0.1.2 refuses a JIT cache whose ancestors are world-writable without the sticky bit (Modal's /tmp is).
mkdir -p -m 700 /root/sx-jit
export LD_LIBRARY_PATH=$shim:$cu13:${LD_LIBRARY_PATH:-} SYMMETRIX_JIT_CACHE=/root/sx-jit
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
rm -rf "$out"/work-*/*.dump 2>/dev/null
note done
