# Symmetrix-XL on the H200, Benchmark_loup (5 November 2026)

Modal NVIDIA H200 (sm_90, driver 610.57.04, CUDA 13.3 UMD), profile `second`, source 16e9f947.
Symmetrix-XL 0.1.1 (`symmetrix-xl` and `symmetrix-xl-cuda13-sm90` wheels) in `sx_image`
(`tools/modal/fastmace_modal.py::sx_job`, Ubuntu 24.04, Python 3.12). Its LAMMPS plugin was built locally in WSL
(`symmetrix_build.py lammps --backend cuda --arch sm90 --mpi off`, Symmetrix-XL source aee9f98 = v0.1.1-15, CUDA
13.3) and uploaded to the volume with `put`; CUDA runtime, cuBLAS and NVRTC come from the wheels' `nvidia/cu13`.
MACE-OMAT-0 medium (Symmetrix JSON export), float32 (`symmetrix/mace/float32/kk`, `streamed_edges direct profile
speed`, JIT device artifact from `symmetrix_prepare_jit_device_artifact`, `jit_device_blocks_per_compute_unit 8`).

Protocol (`tools/bench/sx_loup.py`, the counterpart of `tools/bench/loup_md.py`): SrTiO3 10x10x10 (5,000 atoms),
from rest, skin 0.5 Å checked every step, 1 fs, NVE or `fix langevin` 300 K with damp 0.5 ps (friction 0.002 per fs);
`run 5` untimed, then `run 20 pre no`, LAMMPS's "Loop time"; three fresh processes each. Job script
`tools/modal/symmetrix_loup.sh`.

| Run | Symmetrix-XL ms per step | µs per atom and step |
| --- | --- | --- |
| NVE (3 processes) | 14.63 / 14.60 / 14.58 | 2.92 |
| Langevin (3 processes) | 14.84 / 14.88 / 14.86 | 2.97 |
| Langevin 100 + 300 steps (heats to 171 K, neighbour rebuilds) | 15.69 | 3.14 |

**Against FastMACE on the same GPU type** (`../hopper-sweep-h200-20261105/`, Langevin, same protocol):

- FastMACE under the A100-SXM4 plan with the sm_90 CUTLASS cubin: 5.66 ms (1.13 µs), **2.63x faster** than
  Symmetrix-XL (14.86 ms).
- FastMACE's generic plan (what an un-aliased H200 runs today, `../omat0-h200-profile-0ab2bd9a/`): 13.12 ms
  (2.62 µs), 1.13x faster.

From the A100 PCIe (Symmetrix-XL 5.00 µs per atom and step) to the H200, Symmetrix-XL gains 1.68x; FastMACE's
A100 plan gains 1.93x from the A100-SXM4 (2.18 µs).

The first attempt (`failed-first-attempt/`, 08f9b3fc) stopped at the start of every run with "Failed to load
libcudart.so": Symmetrix-XL dlopens unversioned names and the wheels ship only `.so.13`. 16e9f947 adds shims for
libnvrtc, libcudart, libcublas and libcublasLt.

Cost: about $0.16 (failed attempt) + $0.36 (this run) + $0.01 (uploads), profile `second`.
