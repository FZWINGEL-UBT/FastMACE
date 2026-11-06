# Symmetrix-XL 0.1.2 profiled on the H200, and what FastMACE can take from it (6 November 2026)

Owner, 5–6 November: "go ahead and run it" (profile Symmetrix-XL kernel by kernel next to FastMACE), then "keep trying
to take insight from symmetrix-xl and try to improve fastmace on H200". Benchmark_loup throughout: MACE-OMAT-0 medium,
SrTiO3 10x10x10 (5,000 atoms, 373,760 edges), float32, 1 fs, skin 0.5 Å. Modal NVIDIA H200 (sm_90, driver
610.57.04, CUDA 13.3 UMD).

## 1. Symmetrix-XL 0.1.2 speed (`symmetrix-h200/`)

Symmetrix-XL 0.1.2 (released 4 November 2026; wheels `symmetrix-xl` and `symmetrix-xl-cuda13-sm90` 0.1.2) in
`sx_image` (`FASTMACE_SX_VERSION=0.1.2`), profile `third`, source `4525df3c`. LAMMPS plugin built locally in WSL from
the v0.1.2 source (`symmetrix_build.py lammps --backend cuda --arch sm90 --mpi off`), model exported with that
release's `symmetrix_extract_mace` (local venv `~/environments/symmetrix_xl_012`: Python 3.12, torch 2.14.1+cpu,
mace-torch 0.3.16). Job `tools/modal/symmetrix_profile.sh`; protocol `tools/bench/sx_loup.py` as for 0.1.1
(`../symmetrix-xl-h200-loup-20261105/`). 0.1.2 refuses a JIT cache below a world-writable, non-sticky directory, so
the cache is `/root/sx-jit` (mode 700).

| Run | ms per step | µs per atom and step |
| --- | --- | --- |
| NVE (3 processes) | 14.21 / 14.32 / 14.27 | 2.85 |
| Langevin (3 processes) | 14.65 / 14.52 / 14.55 | 2.91 |
| 0.1.1 on the H200, 5 November (Langevin) | 14.86 | 2.97 |
| **FastMACE `d599cc70`, Langevin** (`fastmace-h200-breakdown/loup/`) | **5.72 / 5.73** | **1.145** |

0.1.2 is 2 % faster than 0.1.1; FastMACE stays **2.54x faster** on the H200. The "5x faster next release" the owner
heard of is not 0.1.2.

## 2. Where Symmetrix-XL's time goes (Nsight Systems 2026.5.2)

Two NVE runs under `nsys profile -t cuda` with 20 and 120 timed steps (`symmetrix-h200/nsys/`); every kernel's
per-step time is the 120-step run's average (126 instances = 5 warm-up + 120 + 1 setup call; the 20-step run agrees).

| Kernel | µs per step | Share |
| --- | ---: | ---: |
| R0 coordinate reverse (`standard_r0::launch_coordinate_reverse_edge_owned`, interaction 1 backward, Kokkos) | 5,577 | 39 % |
| `symmetrix_factorized_reverse_fused_v2` (interaction 2 backward, JIT) | 3,876 | 27 % |
| `symmetrix_factorized_forward_v2` (interaction 2 forward, JIT) | **952** | 7 % |
| R0 forward (interaction 1, Kokkos) | 822 | 6 % |
| M1 reverse | 381 | 3 % |
| cuBLAS FP32 SIMT GEMMs (`sm80_xmma_gemm_f32f32…ffma`), all | ~400 | 3 % |
| A1 compute and reverse, M0, H1/H2, spherical harmonics, ZBL, rest | ~2,200 | 15 % |
| Total GPU time | ~14,200 | |

FastMACE's eager force call on the same GPU (`fastmace-h200-breakdown/`, 5.75 ms):

| FastMACE kernel | µs per call |
| --- | ---: |
| w′ backward (interaction 2 backward) | 1,826 |
| fused forward (interaction 2 forward, 3xTF32 tensor cores) | 1,623 |
| CUTLASS 3xTF32 atom GEMMs (29 launches) | 828 |
| tabled outer-sum backward / forward (interaction 1) | 218 / 118 |
| cuEq symmetric contraction | ~150 |
| radial table forward | 90 |

**Reading.** Symmetrix-XL's interaction 1 (6.4 ms) and interaction 2 backward (3.9 ms) are far slower than ours
(0.34 ms and 1.83 ms). These are the parts a faster next release can fix; we already have them. The one stage where
Symmetrix-XL is ahead is the **interaction 2 forward: 952 µs against our 1,623 µs (1.7x)**.

**How that forward works** (read in `symmetrix/source/symmetrix/jit_codegen.py`, v0.1.2): one thread per (receiver,
channel), grid-stride, 256-thread blocks. Each thread walks its receiver's CSR edges serially. It evaluates w per path
and channel from a cubic Hermite spline: 4 coefficient loads per value, 256 knots, per active species pair. The table
is about 47 MB. The CG terms are unrolled, the message is accumulated in registers, and there are no atomics. It uses
no tensor cores, `__ldg` or vector loads. FastMACE computes w = [h·c, c]·[W, b]ᵀ with a 3xTF32 tensor-core product
per batch of 8 edges.

## 3. Does w from a table make our forward faster on the H200? No (`forward-slab-edges-h200/`)

Kernel lab (`tools/omat0_kernel_lab.py --kernels forward`, 15 rounds), commit `918b21f3`, profile `third`. Every
slab variant is the FP32-SIMT forward without the w store, with w read from 6-term slabs of our value table
(`wprime_backward.TABLE_W`, 1,280 x 6 floats per slab) instead of the a·W product. They are timing bounds and give
wrong values. The new `simt-slab-edges` gives every edge its real slab (pair x piece, as the w′ kernel computes it
from the captured w′ call).

| Variant | µs | vs base |
| --- | ---: | ---: |
| base (production: 3xTF32 tensor cores) | 1,686 | 1.000 |
| simt-rerender (FP32 SIMT, rounding-class) | 2,394 | 1.420 |
| simt-slab3 (3 slabs, L1-resident: the unreachable best case) | 1,519 | 0.901 |
| simt-slab-real (5,340 hashed slabs, 164 MB) | 1,928 | 1.144 |
| **simt-slab-edges (each edge's real slab: 511 of 5,616 touched, 15.7 MB)** | **2,034** | **1.207** |

The real slabs fit the H200's 50 MB L2 easily, yet the forward is 21 % slower than production. Each edge reads its
whole slab (1,280 x 6 floats = 30 KB). That is 11.5 GB per call from L2, more than the tensor-core a·W product costs.
Symmetrix-XL's spline reads 4 values per w, not 6, and its thread layout differs, but even the L1 bound gains only
10 %. **Rejected: an SX-style w-from-table forward does not pay on the H200.** On the RTX 5060 Ti (local smoke test,
746 slabs, 22.9 MB) it was 1.238.

## 4. The w′ backward's reduction on the H200 (`wprime-reduction-h200/`)

Kernel lab `--kernels wprime`, commit `d599cc70`, profile `fourth`, with `BUTTERFLY=False` as the base (1,634 µs):

| Variant | vs base |
| --- | ---: |
| butterfly (the lab's transform of the base; bitwise) | 1.145 |
| F-no-reduce (no reduction at all: bound) | 0.964 |
| F-cart3 | 1.081 |
| F-redux16 / redux | 1.135 |
| F-cart3-redux | 1.156 |

Removing the reduction completely would save only 3.6 %, so the reduction is not where Symmetrix-XL's reverse
differs usefully (theirs, a Cartesian Jacobian before the reduction, is F-cart3: slower). The lab's butterfly is
14.5 % slower than no butterfly, which conflicts with the MD A/B that made `BUTTERFLY` an H200 default
(0.994 / 0.994). Section 5 rechecks it.

## 5. Butterfly recheck on the H200: off is faster (`butterfly-recheck-h200/`)

Commit `d599cc70`, profile `fourth`, one job: the production kernel lab's w′ base with `BUTTERFLY` forced on and off,
twice each, then the MD-step A/B `nobfly` in both build orders through `tools/modal/hopper_sweep.sh`
(`ALIAS="NVIDIA H200"`, Benchmark_loup, b = `BUTTERFLY=False`).

| Measurement | on | off | off / on |
| --- | ---: | ---: | ---: |
| w′ kernel, run 1 | 1,830.7 µs | 1,612.4 µs | 0.881 |
| w′ kernel, run 2 | 1,828.2 µs | 1,613.4 µs | 0.883 |
| MD step, order ab | 5.9073 ms | 5.7060 ms | **0.9659** |
| MD step, order ba | 5.8964 ms | 5.7062 ms | **0.9677** |

The 5 November A/B (`../h200-defaults-20261105/`, 0.994 / 0.994 for on) was taken at an earlier commit; which later w′ change
reversed it was not isolated. Under decision 66 (faster in both build orders on that exact GPU) **the H200 leaves `BUTTERFLY_GPUS`**
(this branch). The change is bitwise for dsh and dr and rounding-class for dx (max |d|/max |a| = 1.87e-07); both sides
are configurations that already passed the force gate. The RTX 5060 Ti keeps the butterfly.

## Cost

Modal: Symmetrix-XL runs (profile `third`) about $1.0 metered; FastMACE breakdown and reduction bounds (profile
`fourth`) about $0.45; slab-edges bound (`third`) $1.02 metered (estimate $0.31); butterfly recheck (`fourth`, 12 min wall)
$0.66 metered.
