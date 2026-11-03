# CUTLASS 3×TF32 GEMMs on the A100 (3 November 2026)

Owner request: "build it for a100", after `tf32x3_gemms` on Triton measured slower than IEEE on the A100 (NaCl 3.1 →
4.2 ms) while a standalone CUTLASS 3×TF32 benchmark ran at 40.8 TFLOPS against cuBLAS IEEE's 19.0 and Triton's 16.8
(`opt-T01-tensor-cores.md` §5). Owner decision the same day, after these results: **default on the
A100-SXM4-80GB for MH-1 and OMAT-0 medium** (decision 66, `planner.fixed.DECISION_66_TF32X3_GPUS`).

**Verdict:**
- The CUTLASS op is as accurate as Triton's tf32x3 and better than cuBLAS IEEE on the scaled metric; it is
  deterministic and replays bitwise in CUDA graphs.
- MD step on the A100-SXM4-80GB, both build orders: **MH-1 NaCl −12 %, NbCl5 −23 %, OMAT-0 SrTiO3 −5 %**.
- Force gates pass for MH-1 and OMAT-0. OMAT-0's energy error grows 4–11× the float32 reference's (still under
  the 10 meV floor); forces stay at IEEE's level.

## 1. What was built

- **`fastmace/kernels/cutlass/tf32x3_sm80.cu`:** two CUTLASS 4.8.1 (`2c85f497`) kernels, `DefaultGemm` with
  `OpMultiplyAddFastF32` (the big/small TF32 split, three tensor-core products), 64×64×16 threadblock, 32×32×16 warp,
  16×8×8 instruction, 4 stages, no split-K. A row-major; B row-major (`rr`) or column-major (`rc`, a transposed
  weight in the backward). `Gemm::Params` is built on the device, so the launch takes plain pointers and sizes.
- **`tf32x3_sm80.cubin`:** built offline with nvcc 13.3 (`tools/build_cutlass_tf32x3.sh`, record
  `tf32x3_sm80.build.txt`; 48 `HMMA.1688.F32.TF32` per k-step, 134 and 164 registers, no spills). CUTLASS's licence
  and notice ship beside it. No nvcc is needed at run time.
- **`fastmace/kernels/cutlass/tf32x3.py`:** loads the cubin through `cuda.bindings` (`cuModuleLoadData`,
  `cuLaunchKernel` on the current torch stream). `available` admits compute capability 8.0 only (an sm_80 cubin and a
  tile chosen on the A100). The launch makes the device's primary context current first (autograd's device threads
  may have none).
- **The backend rule** (`tf32x3_gemm._launch`): on a device where CUTLASS loads, a product runs in CUTLASS when
  `takes(m, k, n)` holds, else in cuBLAS IEEE, never in Triton. The rule reads the shape only (strides are made to
  fit by `fit`, copying where needed), so a product's arithmetic does not depend on its operands' layout.
  - `takes`: K and N multiples of 4, and either at least `MIN_TILES = 54` output tiles (half the A100's SMs) or
    K ≤ `LONG_K = 256` (§3).
- **`GatedRows` (MH-1):** on the CUTLASS backend it takes the block products through `tf32x3_mm` instead of its
  joined Triton GEMM (`precision.route` sets `tf32x3_backend`).
- **Planner** (decision 66): `tf32x3_gemms="auto"` is on for MH-1 labels and `omat0-medium` on the exact device name
  `NVIDIA A100-SXM4-80GB`, in float32 plans that are not reproducible. It turns `grouped_launch` off (both reroute
  the block products; the A100 measured no change from `grouped_launch`). An explicit `grouped_launch="on"` keeps
  it off. Every other GPU and model keeps IEEE.

## 2. Accuracy (Modal A100-SXM4-80GB)

`tools/bench/cutlass_tf32x3_check.py`, 42 atom-GEMM-like shapes (M 512–40,000; K, N 96–2,048; both B layouts),
error scaled by the output's |a||b| sum against float64:

| Backend | Worst scaled error |
| --- | --- |
| CUTLASS 3×TF32 | 3.2e-7 |
| Triton tf32x3 | 2.1e-7 |
| cuBLAS IEEE | 4.8e-7 |

Every case repeats bitwise and replays bitwise in a CUDA graph. `tools/tf32x3_accuracy.py` on MH-1's real operands:
max relative error 1.1e-6 (NaCl) and 1.1e-6 (NbCl5) against IEEE's 0.9e-6 and 1.5e-6.

Force gate (`tools/force_stress_gate.py`, NaCl and NbCl5, ratio to the float32 calculator's error against float64;
the limit is 2 with an absolute floor):

| Plan | NaCl energy / forces / stress | NbCl5 energy / forces / stress |
| --- | --- | --- |
| MH-1, tf32x3 | 2.07 / 0.91 / 1.28 | 0.30 / 1.04 / 0.37 |
| OMAT-0, IEEE control | 1.29 / 0.81 / 0.82 | 0.92 / 0.96 / 0.51 |
| OMAT-0, tf32x3 | 4.47 / 0.82 / 1.61 | 11.1 / 1.03 / 1.77 |

All pass. The energy ratios above 2 pass on the floor: OMAT-0 NbCl5's energy error is 1.24 meV for the cell
(IEEE 0.10 meV, floor 10 meV). The control shows the excess comes from tf32x3, not the model. Forces are unchanged.

## 3. Op speed and the tile rule

CUDA-graph timings (20 launches per graph, no host launch cost), CUTLASS over cuBLAS IEEE:

- Products with many output tiles: 0.54–0.69 at 5,000–40,000 rows for every K and N measured, 0.86 at 5,000 × 96 ×
  160; 1.6–3.4× faster than Triton everywhere.
- Few tiles: 512 rows × 128 or 160 columns (16–24 tiles) at K ≤ 256 0.91–1.00; at K 512 1.24–1.44 and at K 2,048
  1.94–2.27. Without split-K a long K on 16 tiles leaves most of the 108 SMs idle. Those products stay on cuBLAS
  (`MIN_TILES`, `LONG_K`). An MD step's products have atom rows, so many tiles; a weight gradient's K is the atom
  count, and it never runs in MD.

## 4. MD step A/Bs (Modal A100-SXM4-80GB, source `74012562`)

`tools/ab_rewrites.py`, same process, 1 fs, 200 warm-up steps, 8 blocks of 200, both build orders. Side a is IEEE
with `grouped_launch=off` (it conflicts with `tf32x3_gemms`).

| Workload | b | order ab | order ba |
| --- | --- | --- | --- |
| MH-1 NaCl 3×3×3 | tf32x3 (CUTLASS) | 2.20 → 1.95 ms, **0.887** | 2.22 → 1.95 ms, **0.882** |
| MH-1 NbCl5 3×3×3 | tf32x3 (CUTLASS) | 11.28 → 8.71 ms, **0.772** | 11.29 → 8.69 ms, **0.770** |
| OMAT-0 SrTiO3 5,000 atoms, skin 0.5 | tf32x3 (CUTLASS) | 22.82 → 21.65 ms, **0.949** | 22.88 → 21.66 ms, **0.947** |
| OMAT-0, a = the full default (grouped_launch on) | tf32x3 (CUTLASS) | 22.85 → 21.66 ms, **0.948** | 22.90 → 21.69 ms, **0.947** |
| OMAT-0, a = tf32x3 on Triton | tf32x3 on CUTLASS | 23.87 → 21.69 ms, **0.909** | 23.88 → 21.65 ms, **0.907** |

Every block pair valid. The first version (CUTLASS only where the layout fit, Triton elsewhere) measured MH-1 NaCl
1.18 and NbCl5 1.02: the Triton remainder and the joined Triton products in `GatedRows` cost more than CUTLASS
saved. Routing everything else to cuBLAS IEEE fixed that.

## 5. Records

`results/a100-cutlass-tf32x3-20261103/` (job logs, `check.json`, gates, A/B JSONs). Cost: $1.37 and $0.48 of Modal
credit (the second job: tile rule, GPU tests 8/8, the control gate).
