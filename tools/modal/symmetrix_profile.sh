#!/usr/bin/env bash
# Symmetrix-XL kernel profile under LAMMPS, Benchmark_loup protocol (6 November 2026; owner: "go ahead and run it").
# Runs in the Symmetrix-XL image (fastmace_modal.py::sx_job, FASTMACE_MODAL_SX=1, FASTMACE_SX_VERSION). Inputs in the
# volume, under SX_DIR (default /data/symmetrix-0.1.2):
#   $SX_DIR/lmp-<arch>/bin/lmp           LAMMPS + Symmetrix-XL, built locally (symmetrix_build.py lammps, MPI off)
#   $SX_DIR/mace-omat-0-medium.json      the Symmetrix export of MACE-OMAT-0 medium (that release's format)
# Steps: the JIT device artifact; the speed protocol (tools/bench/sx_loup.py, NVE and Langevin, TIMED fresh processes
# each, 5 + 20 steps); then two NVE runs under Nsight Systems (CUDA trace) with 20 and 120 timed steps. The per-step
# GPU time of every kernel is the difference of the two runs' kernel sums over 100 steps (setup, JIT, warm-up and
# the first neighbour build cancel).
# Usage (from WSL, Modal client venv):
#   FASTMACE_MODAL_SX=1 FASTMACE_SX_VERSION=0.1.2 modal run tools/modal/fastmace_modal.py::sx_job --gpu H200 \
#     --timeout-min 25 --commit <sha> --command "bash tools/modal/symmetrix_profile.sh" --out ~/fm-runs/sx012-profile
set -uo pipefail
out=${OUT:?the job sets OUT}
arch=${SX_ARCH:-sm90}
dir=${SX_DIR:-/data/symmetrix-0.1.2}
lmp=$dir/lmp-$arch/bin/lmp
model=$dir/mace-omat-0-medium.json
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
pip freeze 2>/dev/null | grep -i symmetrix >> "$log"
nsys --version >> "$log" 2>&1
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
  for t in $(seq 1 "${TIMED:-3}"); do run "$ens-$t" --ensemble "$ens"; done
done
mkdir -p "$out/nsys"
for n in 20 120; do
  run "nsys-$n" --ensemble nve --warmup 5 --steps "$n" --wrap "nsys profile -t cuda --force-overwrite true -o $out/nsys/steps$n"
  for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum; do
    nsys stats --report "$r" --format csv --output "$out/nsys/steps$n" "$out/nsys/steps$n.nsys-rep" \
      > "$out/nsys/stats-$n-$r.log" 2>&1
  done
done
rm -f "$out"/nsys/*.sqlite
ls -la "$out/nsys" >> "$log"
note done
