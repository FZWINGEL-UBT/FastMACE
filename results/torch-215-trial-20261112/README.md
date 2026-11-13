# PyTorch 2.15 trial and CUDA library matrix (12 November 2026)

loupMD (FastMACE `b7c19b19`) on PyTorch 2.15.0+cu132 against production 2.14.1+cu130. The trial then splits the
2.15 stack into its parts: Triton, cuBLAS, NVRTC and torch. The question is which part changes the speed, on which
GPU, and why. Model OMAT-0 medium, float32, the planner's default plan, fallbacks refused. Worktree branch
`fastmace/torch-215-trial`.

**Verdict: do not migrate to 2.15 now.**

- Its only speed gain is Triton 3.9's tf32x3 code, on GPUs where Triton runs those products (Ada: -2.9% SrTiO3,
  -6.9% NbCl5). The 5060 Ti does not gain.
- 2.15 also brings three costs:
  - NVRTC 13.2 slows the w' backward kernel by 4-7% on sm_89 and sm_120.
  - The `save_transcendentals` partitioner rewrite is disabled (source pin).
  - A crt packaging trap breaks OEq.

The cheaper route to the Ada gain is a Triton-3.8 form of the tf32x3 kernel or Triton 3.9 on its own; see §7.

## 1. Releases and environments

- PyTorch 2.15.0: released 28 October 2026 ([release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0),
  [PyPI](https://pypi.org/project/torch/2.15.0/)).
  - Python >= 3.11. The PyPI default is cu132; cu134 also exists and cu130 is dropped.
  - Triton 3.9.0 (cp311+ wheels only); driver >= 580.
- Production: torch 2.14.1+cu130, Triton 3.8.0, cuBLAS 13.1.1, NVRTC 13.0, Python 3.10.12.

| Variant | Environment | What differs from production |
| --- | --- | --- |
| a | `mace_cueq_oeq` | production |
| b | `mace_cueq_oeq_torch215` | torch 2.15.0+cu132 (Triton 3.9.0, cuBLAS 13.4.1, NVRTC 13.2), Python 3.11.16, + `nvidia-cuda-crt==13.2.86` |
| c4 / c8 | `mace_cueq_oeq_cublas134` / `_cublas138` | nvidia-cublas 13.4.1.3 / 13.8.2.12 |
| t9 | `mace_cueq_oeq_triton39` | Triton 3.9.0 (`--no-deps`), Python 3.11.16 (no cp310 wheel) |
| n2 / n4 / n5 | a | our NVRTC kernels compiled by NVRTC 13.2.86 / 13.4.92 / 13.5.37 (`runtime.NVRTC`, a child process) |
| b0 | b | our NVRTC kernels compiled by NVRTC 13.0.88 |

- Workstation envs are under `/mnt/homes/zwingel/environments`, each with a `.requirements.txt` freeze.
- The local 5060 Ti uses WSL `~/environments`; c4/c8 there are overlays of the cuBLAS libraries on production
  (`jobs/py-c4o.sh`, `jobs/py-c8o.sh`), for lack of WSL disk.
- Alex builds them on `$TMPDIR` (`jobs/alex_matrix.sh`).
- The library each variant really loads was checked in `/proc/self/maps` with `cublasGetProperty`.

## 2. Compatibility (`jobs/compat.py`)

All checks pass under 2.15 once `nvidia-cuda-crt` is installed. The checks: import, NVRTC compile and launch,
CUTLASS cubin load, torch.compile, CUDA-graph capture, OMAT-0 force call, and Simulation with fallbacks refused.

- **crt trap.** The cu132 wheels no longer pull in `nvidia-cuda-crt`, so OEq's JIT host extension fails to build
  (`crt/host_defines.h`).
  - The planner then falls back to the reference tier without saying so: 17-30 GiB RSS, an 11-16 GiB card, and
    NbCl5 fails with `_NoNeighbours`.
  - A plain `ForceField.evaluate` passes anyway; only Simulation planning checks the hybrid. The probe therefore
    builds a Simulation with `FallbackConfig(fallback="refuse")`.
- **save_transcendentals blocker** (conductor: note it, don't fix it now). The rewrite pins the source of PyTorch
  2.14.1's `min_cut_rematerialization_partition` (`fastmace/rewrites/inductor.py:PARTITIONER_SOURCES`).
  - Under 2.15 the hash differs, so the rewrite is refused.
  - It is the default only for float64 MH-1 on two GPUs (ledger `prec-save-transcendentals`), but a migration must
    re-port and re-pin it.
- CPU suite under 2.15 (coraline, b7c19b19): 5714 passed, 204 skipped.
- Gate (`tools/force_stress_gate.py`, SrTiO3 3x3x3 and NbCl5, 3x float32 bound against the 2.14.1 reference): b
  passes on the Ada.
  - SrTiO3 energy ratio 3.5 against a's own 4.3; the bound is the 0.01 eV floor.
  - Matrix gates (b, t9, c4, c8) on the A100: not measured (§5).

## 3. A/B: torch 2.15 against 2.14.1 (one process per run, ABAB then BABA, 200 warmup + 1000 steps)

| GPU | Structure | a ms/step | b ms/step | b/a ABAB | b/a BABA | reserved a -> b |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4000 Ada (leeloo) | SrTiO3 10^3 (5000) | 17.197 | 16.700 | 0.971 | 0.973 | 1026 -> 946 MiB |
| RTX 4000 Ada (leeloo) | NbCl5 3^3 (1944) | 3.759 | 3.500 | 0.930 | 0.931 | 536 -> 456 MiB |
| RTX 5060 Ti (local) | SrTiO3 10^3 | 16.072 | 16.065 | 1.000 | 1.011 | 1182 -> 1042 MiB |
| RTX 5060 Ti (local) | NbCl5 3^3 | 3.383 | 3.418 | 1.003 | 0.998 | 612 -> 472 MiB |

- Plans are identical (same plan-text hash).
- Cold build is unchanged: Ada 58.0 s against 57.5 s; 5060 Ti 48.5 s against 49.0 s.
- 2.15 reserves 80-140 MiB less, from the caching allocator, not the plan.

## 4. Where the time moves, and why (kernel attribution)

The profiles take 20 steps after the timed run (`jobs/profile_point.py`). Categories come from
`jobs/kernel_category.py`, checked against the Python stacks of an annotated capture (§6).

> Correction: an earlier reading credited cuBLAS 13.4 with the Ada gain. The kernel named `gemm` is our own Triton
> `tf32x3_gemm` / `species_gemm` (a bare function name), not cuBLAS. cuBLAS 13.4 alone (c4) changes nothing:
> SrTiO3 17.15 against 17.16 ms, NbCl5 3.76 against 3.76 ms (8 processes, adad dada).

Microseconds per MD step, SrTiO3 10^3:

| Category | Ada a | Ada b | Ada b-a | 5060 Ti a | 5060 Ti b | 5060 Ti b-a |
| --- | --- | --- | --- | --- | --- | --- |
| NVRTC (w' backward, spline forward, sc_generated) | 8351 | 8588 | +237 | 6646 | 7058 | +412 |
| our Triton (tf32x3/species gemm, tabled outer sum, integrator) | 5144 | 4437 | **-707** | 2364 | 2419 | +54 |
| Inductor | 2458 | 2456 | -2 | 3244 | 3405 | +161 |
| cuBLAS | 51 | 51 | 0 | 1354 | 1402 | +48 |
| CUTLASS | 298 | 306 | +8 | 1054 | 1101 | +46 |
| cuEquivariance | 369 | 361 | -8 | 789 | 922 | +133 |
| memset32 (2.15) / FillFunctor (2.14.1) zero fills | 2 / 199 | 209 / 1 | +10 | 1 / 195 | 197 / 1 | +2 |

**Triton 3.9 (Ada -704 us, from `gemm`).**

- Same tile (64x64x32, 4 warps), same 96 `mma.sync.m16n8k8.tf32` per K step. Triton 3.8.0 lowers the tf32x3
  `tl.dot` through two extra shared-memory buffers: the split operands go back to shared memory and are reloaded
  (TTGIR: 4 `local_alloc` against 2, 6 `local_load` against 2; PTX: 112 `cvt.rna.tf32` against 80).
- Triton 3.9.0 splits in registers from the loaded tile:

| | Triton 3.8 | Triton 3.9 |
| --- | --- | --- |
| Shared memory, 3 stages | 48 KiB | 32 KiB |
| Shared memory, 2 stages | 32 KiB | 16 KiB |
| Registers | 251-255 | 168-245 |
| Same launch | 109 us | 75 us |

- It helps only where Triton runs the tf32x3 products:
  - Ada: 22 launches per step.
  - A100 / H200: CUTLASS runs them (`CUTLASS_CAPABILITIES`).
  - 5060 Ti: the plan sends the big products to CUTLASS SIMT and MAGMA sgemm, and its 10 `gemm` launches use the
    small 2-stage tile, which has no extra buffer to remove.

**NVRTC 13.2 (Ada +229 us, 5060 Ti +329 us, `oeq_projected_backward_wprime`).**

- Same source, slower SASS on sm_89 and sm_120.
- On sm_80 and sm_90 the per-edge w' loop is the same for 13.0, 13.2, 13.4 and 13.5 (§5 census). The regression is
  specific to the consumer architectures.
- The matrix variants n2/n4/n5/b0 check the speed side (§5).

**Inductor and cuEq on the 5060 Ti (+161, +133 us).** cuEq picks different algorithms per process (autotune: e.g.
`algo4_2` against split `algo1_*`), which is noise-level. Inductor: see t9 in §5.

## 5. CUDA library matrix

One process per variant and structure, all variants forward then reversed (f1, r1). Each cell is the mean of the
two passes in ms per step, then the change against production `a`. Cross-process noise is about 3%, so only
changes that both passes agree on, and that the same-process A/B of §3 confirms, count. Table from
`jobs/matrix_table.py */matrix/run.log`.

| Variant | Ada SrTiO3 10³ | Ada NbCl5 | 5060 Ti SrTiO3 10³ | 5060 Ti NbCl5 |
| --- | --- | --- | --- | --- |
| a: production 2.14.1, cuBLAS 13.1.1, NVRTC 13.0, Triton 3.8 | 17.17 | 3.77 | 15.74 | 3.31 |
| c4: cuBLAS 13.4.1 | 17.18 (+0.0%) | 3.76 (-0.3%) | 15.81 (+0.4%) | 3.39 (+2.3%, passes 3.32 / 3.46) |
| c8: cuBLAS 13.8.2 | 17.18 (+0.1%) | 3.75 (-0.5%) | 15.62 (-0.8%) | 3.33 (+0.6%) |
| t9: Triton 3.9 on Python 3.11 | **16.54 (-3.7%)** | **3.46 (-8.1%)** | 15.63 (-0.7%) | 3.33 (+0.3%) |
| n2: our kernels by NVRTC 13.2 | 17.36 (+1.1%) | 3.76 (-0.1%) | 15.86 (+0.8%) | 3.37 (+1.5%) |
| n4: NVRTC 13.4 | 17.34 (+1.0%) | 3.76 (-0.1%) | 15.79 (+0.3%) | 3.38 (+2.0%) |
| n5: NVRTC 13.5 | 17.37 (+1.2%) | 3.76 (-0.1%) | 15.90 (+1.0%) | 3.38 (+2.1%) |
| b: torch 2.15 | 16.69 (-2.8%) | 3.50 (-7.2%) | 15.92 (+1.1%) | 3.38 (+2.1%) |
| b0: torch 2.15, our kernels by NVRTC 13.0 | 16.51 (-3.9%) | 3.46 (-8.1%) | failed | failed |

- **The whole 2.15 gain on the Ada is Triton 3.9:** t9 alone reaches it, and b0 (2.15 without the NVRTC change)
  equals t9. b minus b0 is the NVRTC 13.2 cost, +1% on SrTiO3, the same as n2 on production.
- **cuBLAS changes nothing** beyond noise on either GPU.
- **Every newer NVRTC is slower or equal on sm_89 and sm_120;** none is faster.
- **The 5060 Ti gains from nothing.** Its Triton GEMMs are not on the critical path (§4).
- **b0 on the 5060 Ti** failed in the NVRTC 13.0 child compile of the torch 2.15 environment (`runtime.py:90`
  `_compile_elsewhere`); not analysed, since the Ada b0 already answers the question.

**A100: not measured.** Alex job 4990792 (matrix, gates, census run and step attribution in one job) failed in its
environment setup on 12 November: the cuBLAS-variant build pinned a cuBLAS that torch 2.14.1 forbids. The fixed
script is `jobs/alex_all.sh` (1ab751a3). It is not resubmitted while the owner's hold on new Alex jobs stands
(12 November). On the A100 only the offline census below exists.

### Offline NVRTC census, sm_80 / sm_90 (`jobs/census_nvrtc.py`, leto, CPU only)

| Kernel / arch | NVRTC 13.0 | 13.2 | 13.4 | 13.5 |
| --- | --- | --- | --- | --- |
| w' backward, A100 render, sm_80, SASS instr per edge (regs) | 2053 (79) | 2053 | 2053 | 2053 |
| w' backward, A100 render, sm_90 | 2088 (80) | 2088 | 2088 | 2088 |
| w' backward, H200 render, sm_80 | 1976 (128) | 1976-1977 | 1976-1977 | 1976-1977 |
| w' backward, H200 render, sm_90 | 1990 (96) | 1990-1991 | 1990-1991 | 1990-1991 |
| spline forward, sm_80, per edge | 1679 | 1664 | 1644 | 1644 |
| spline forward, sm_90, per edge | 1717 | 1712 | 1695 | 1695 |

- No spills anywhere.
- On the A100 and H200, a newer NVRTC trims the spline forward by 2% of its loop and leaves w' as it is.

## 6. Profiler tool: CUDA-graph kernels to Python lines

`jobs/profile_point.py` with `PYSTACKS=1` captures the production CUDA graphs annotated:

- `torch.cuda.graph(..., enable_annotations=True, annotation_config={"record_py_stacks": True})`.
- Profiled with Cuspy: `ProfilerActivityConfig(profiler_configs=[CuspyConfig()])`.
- Joined on `(graph id << 32) | graph node id`.

Requirements:

- torch >= 2.15, `cupti-python` (CUPTI >= 13.3), single-threaded autograd
  (`torch.autograd.grad_mode.set_multithreading_enabled(False)`).
- Native Linux. Under WSL, CUPTI's clock source fails.

Sample from the Ada:

- `ada-leeloo/profile/trace-prof-c.json.gz` (Chrome trace) and `ada-leeloo/profile/pystacks-prof-c.json`.
- 100% of the 3200 graph kernels map to a Python line; that mapping found the `gemm` misattribution above.

## 6a. Time we don't own: CUDA-graph attribution of the production step (Ada)

Run setup:

- Machine and model: leeloo, RTX 4000 Ada, OMAT-0 medium float32, production plan.
- Code: FastMACE `b7c19b19`.
- Environment: torch 2.15.0+cu132 profiling env with cupti-python.
- Window: 5 annotated MD steps after the steady run.
- Tooling: `jobs/attr.sh` → `jobs/profile_point.py` (PYSTACKS=1, Inductor `comment_origin=True`) → `jobs/attribution.py`.

Attribution method:

- Each replayed graph kernel is joined to its capture stack, keyed by `(graph id << 32) | node id`.
- The site is the innermost model frame, plus its caller.
- For an Inductor kernel, the site is the first model line in the generated kernel's origin comment.

Records: `ada-leeloo/attribution/attribution-{SrTiO3,NbCl5}.json`. Sample Chrome traces:
`ada-leeloo/attribution/profile/trace-attr-*.json.gz` (open in Perfetto), with `pystacks-attr-*.json`.

Step totals (µs per step):

| | SrTiO3 10³ (5000 atoms) | NbCl5 (1944 atoms) |
| --- | --- | --- |
| GPU kernel time | 16,878 | 3,372 |
| Window wall time | 16,592 | 3,523 |
| Ours: NVRTC / our Triton / CUTLASS | 8,304 / 4,494 / 301 | 1,390 / 1,070 / 278 |
| Inductor | 2,473 (14.7%) | 382 (11.3%) |
| Eager ATen / cuEq / memset / copy / cuBLAS | 527 / 345 / 241 / 141 / 50 | 86 / 49 / 26 / 61 / 32 |
| **Not ours** (everything above that is not our kernels) | **3,778 (22.4%)** | **634 (18.8%)** |
| Idle GPU between kernels, steady (after the window's first ms) | 53 (0.3%), of which 35 inside graphs | 26 (0.75%), of which 12 inside graphs |

### Ranked: time we don't own, by source line (µs per step, Ada)

| # | Source line (site ← caller) | What runs there | SrTiO3 | NbCl5 |
| --- | --- | --- | --- | --- |
| 1 | `rewrites/generic/modules.py:211` ← `compile.py:330 lean_forward` | Inductor `cat_view_24` and three `clone/slice/transpose` kernels around `species_mm`: layout only | 563 (3.3%) | 80 (2.4%) |
| 2 | `rewrites/generic/modules.py:530` ← `lean_forward` | Inductor `add_cat_div_slice_view_28`, `cat_view_27`: a cat and a division | 481 (2.9%) | 64 (1.9%) |
| 3 | eager, outside the graphs | neighbour-list rebuild (index kernels, copies); 1 rebuild in the SrTiO3 window, none in NbCl5's | 443 (2.6%) | 0 |
| 4 | `kernels/triton/tabled_outer_sum.py:739` ← `compute_forces` | Inductor `add_gather_slice_backward..._transpose_view_12` (backward gather + transpose) | 358 (2.1%) | 19 |
| 5 | `rewrites/generic/modules.py:530` ← `compute_forces` | Inductor backward of row 2 | 313 (1.9%) | 65 (1.9%) |
| 6 | `mace/modules/utils.py:26 compute_forces` | cuEq backward (symmetric contraction) | 234 (1.4%) | 32 |
| 7 | `kernels/triton/tabled_outer_sum.py:600 tabled_outer_sum_forward` | graph Memset zero-fill of the outer-sum output | 192 (1.1%) | 17 (0.5%) |
| 8 | `kernels/triton/tf32x3_gemm.py:171` ← `compute_forces` | Inductor `clone/slice_backward` around the tf32x3 backward | 185 (1.1%) | 30 (0.9%) |
| 9 | `rewrites/engine.py:427` (forward and backward) | Inductor edge geometry, `index_put`/`masked_fill`, scatter_add | 261 (1.5%) | 53 (1.6%) |
| 10 | `kernels/triton/tabled_outer_sum.py:757` ← `compute_forces` | Inductor backward | 150 (0.9%) | 17 |
| 11 | `compile.py:330 lean_forward` ← `forces.py:1194 _split_outputs` | eager ATen, cuEq and cuBLAS inside the graph: the radix sort of `species_plan` (see below), the energies gemv | 225 (1.3%) | 78 (2.3%) |

### Per-step-constant work inside the graph (ours, but not needed every step)

- **What is captured:** `rewrites/mh1/modules.py:206 refresh_composed` → `:239 _compose_weights` →
  `kernels/triton/compose_skip.py:116 composed`. These are COMPOSE_SKIP's composed first-interaction weights.
- **When it runs:** `forces.py:1036` binds the table in the captured region, so the compose is inside the CUDA graph
  (graph 5 for SrTiO3, graph 11 for NbCl5).
- **Cost:** four float64 CUTLASS `d884gemm` per step at about 70–75 µs each. On the Ada float64 runs at 1/64 rate.
- **Why it repeats:** the inputs are the species table and the radial weights, both constant in MD. The graph still
  replays the compose every step.
- **NbCl5 extras:** the `copy_`s at `compose_skip.py:115–120`, the in-place refresh at `mh1/modules.py:218`, and
  eager elementwise kernels at `:237`. These are about 53 µs more.

| | SrTiO3 | NbCl5 |
| --- | --- | --- |
| Compose work replayed per step | 301 µs (1.8%) | 331 µs (9.8%) |

- **Bound:** the full cost listed above. It is a work removal: compose once per table version, outside the graph.
- **Not built** (per the request).
- **A100 and H200:** float64 runs on tensor cores at full rate there, so the share should be far smaller. The A100
  run will say.

### Ledger rows: measured bounds (Ada)

| Ledger row | Earlier bound | Measured here, SrTiO3 / NbCl5 | How |
| --- | --- | --- | --- |
| `species-only-kernels` | 1.1–1.3% (kernel names) | **126 µs (0.75%) / 57 µs (1.7%)** | `species_plan`'s stable argsort (`kernels/triton/species_gemm.py:249`, a radix sort; called from `mh1/modules.py:330 _composed` and `generic/modules.py:798` skip_tp; in the graph, every step): 112 / 49 µs. Plus Inductor argmax kernels (`_to_copy_argmax_10`, `argmax_index_index_add_11`) and the node_attrs × reference-energy gemv: 14 / 8 µs. All depend only on the species. |
| `i0-copies-zero-fill` | — | **pure layout: 849 µs (5.0%) / 128 µs (3.8%)**; + the Memset of row 7: 192 / 17 µs. Upper bound with fused compute: 1,949 (11.6%) / 314 (9.3%) | "Pure layout" counts Inductor kernels whose origin ops are only cat/view/clone/transpose/slice/zeros (rows 1–2 mostly). The upper bound adds fused kernels that also add or gather. |
| `launch-memsets` | — | **4 µs (0.02%) / 1 µs (0.03%)** | Eager fills and memsets outside the outer-sum Memset: nothing left. |
| `launch-combo-sizerule` | — | small Inductor kernels: **37 µs over 22 launches / 60 µs over 31 launches**; all kernels under 5 µs: 207 µs (118 per step) / 194 µs (104 per step) | Steady idle time: 53 / 26 µs per step, 35 / 12 µs of it inside graphs. Host-side gaps over 20 µs in steady state: 24.5 µs per step on SrTiO3, all in the eager neighbour-list rebuild (`aten::_unique2`, `remainder`, `to`); none on NbCl5. The large gaps in the first ms of the window are profiler start, not per step: the first `cudaMemcpyAsync` (aten::copy_, 386 / 372 µs on the host, a 120 kB / 47 kB DtoD positions copy) and the first `cudaGraphLaunch` (48–205 µs). `jobs/gaps_radix.py` lists them apart. |

What this means:

- **The species-only row is not closed:** 0.75% / 1.7%, mostly `species_plan`'s argsort replayed every step. COUNT_PLAN, the counting sort, is off on the Ada. The plan depends only on species, so it could be built once per species change.
- **Layout-only Inductor kernels are the largest block we don't own:** 3.8–5.0%. The `species_mm` boundary at
  `generic/modules.py:211` and the cat at `:530` account for most of it.
- **The second largest is the per-step compose:** 1.8% / 9.8% on the Ada, and the clearest work removal.
- **Small-kernel launch overhead and host syncs are small in steady state:** small Inductor kernels and idle time are under 0.4% / 0.8% of the step.

A100: not measured. Job 4990485 failed (CUPTI 13.2 too old for `record_py_stacks`), and its replacement 4990792
failed in environment setup (§5). The fixed one-job script `jobs/alex_all.sh` waits, unsubmitted, until the owner
lifts the hold on new Alex jobs. `jobs/layout_rows.py` then prints the A100 rows that step-layout-cleanup asked for.

## 6b. Falsifier: Triton 3.9's tf32x3 gain on Triton 3.8 by an explicit register split (Ada, closed)

Idea (conductor, owner-approved): split a and b in registers, then run three `tl.dot(..., "tf32")`.
- The split rounds hi to nearest-even TF32 by an integer add and mask:
  hi = (u + 0x0FFF + ((u >> 13) & 1)) & 0xFFFFE000, and lo = x − hi.
- The three products run lo·hi first, then hi·lo, then hi·hi.
- The aim was to stay on torch 2.14.1 + Triton 3.8 and drop the two extra shared-memory buffers of Triton 3.8's
  `tf32x3` lowering.

**Is the gain in these GEMMs?** Yes. All of Triton 3.9's gain on the Ada is in our `gemm` kernels (22 launches per
step on SrTiO3 10³; matrix profiles):

| | `gemm` kernels | Total kernel time |
| --- | --- | --- |
| Production (Triton 3.8) | 3324 µs | 17,326 µs |
| t9 (Triton 3.9) | 2607 µs | 16,609 µs |
| Difference | −717 µs | −717 µs |

The CUTLASS sm_89 backend tied Triton 3.8 (ledger `ada-omat0-best`), so Triton 3.9 beats it too.

**Falsifier.** Run: `jobs/split_lab.py` via `jobs/splitlab.sh`, on leto, CPU only (`triton.compile` for sm_89 and
sm_120); records in `splitlab-leto/`. Setup:
- the production tile: 64×64×32, 4 warps, 3 stages;
- registers and spills from `ptxas -v`.

| Triton | Kernel | Mode | Shared | `local_alloc` / `local_load` | Registers | Spill stores / loads |
| --- | --- | --- | --- | --- | --- | --- |
| 3.8.0 | tf32x3_gemm | tf32x3 (production) | 48 KiB | 4 / 6 | 255 | 8 / 4 |
| 3.8.0 | tf32x3_gemm | **register split** | **48 KiB** | **4 / 6** | 255 | 0 / 0 |
| 3.8.0 | species_gemm | tf32x3 (production) | 48 KiB | 4 / 6 | 255 | 60 / 60 |
| 3.8.0 | species_gemm | **register split** | **48 KiB** | **4 / 6** | 255 | **48 / 48** |
| 3.8.0 | either | one tf32 dot (floor) | 32 KiB | 2 / 10 | 162 / 254 | 0 |
| 3.9.0 | tf32x3_gemm | tf32x3 | 32 KiB | 2 / 2 | 255 | 0 |
| 3.9.0 | tf32x3_gemm | register split | 32 KiB | 2 / 2 | 190 | 0 |
| 3.9.0 | species_gemm | tf32x3 / register split | 32 KiB | 2 / 2 | 255 | 0 |

**Verdict: stop.** Both conditions fail on sm_89: shared memory stays at 48 KiB, and species_gemm spills.
- The extra buffers come from Triton 3.8's layout handling, not from its split. The split runs on the loaded tile
  (blocked layout). Each of the four parts then goes through shared memory to reach the MMA operand layout.
- Triton 3.9 keeps the conversion in registers, for its own `tf32x3` and for the explicit split alike.
- No PTX `cvt.rna.tf32` in the split, so the opt5-T01 double-rounding trap does not arise. The question is moot
  anyway.
- Correction: the lab compiled both kernels at 3 stages, but production `species_gemm` runs 2
  (`species_gemm.NUM_STAGES`). Only `tf32x3_gemm` uses tile_rule's 3. Its 3-stage rows show the lowering, not
  production's shared memory. The 2-stage question is screened in §6c.

The ways to get the 717 µs remain:
- Triton 3.9 itself (t9: torch 2.14.1 + Triton 3.9 on Python 3.11);
- a hand-written MMA kernel, such as `species_gemm`'s NVRTC tensor-core path.

## 6c. Screen: pipeline stages of the tf32x3 Triton GEMMs (Ada, closed)

Run setup:

- Machine: leto (RTX 4000 Ada).
- Environment: production a (torch 2.14.1, Triton 3.8), production plan, b7c19b19.
- Scripts: `jobs/stages_lab.py` via `jobs/stages.sh`; records in `stages-leto/`.

The lab:

- **Recording:** it records every Triton launch of both GEMMs in a real MD point. Calls outside the capture
  give the inputs; calls inside the capture give the launches per step: 12 `tf32x3_gemm` + 10 `species_gemm` = 22,
  the profile's count.
- **Replay:** after the point's runs, it replays one step's launches with `num_stages` 3 and 2. That is 20 rounds
  in ABBA order, the median of CUDA-event times.

SrTiO3 10³, µs per step of GEMM time:

| Kernel (production stages) | 3 stages | 2 stages | 2 / 3 (ABBA halves) |
| --- | --- | --- | --- |
| `tf32x3_gemm` (3) | 1962 | 2108 | 1.074 (1.073 / 1.069) |
| `species_gemm` (2) | 1277 | 1336 | 1.047 (1.044 / 1.047) |

Findings:

- **2 stages for `tf32x3_gemm`:** 7.4% slower. Stop.
- **3 stages for `species_gemm`, the reverse:** 4.5% faster, −60 µs per step. That is 1.8% of the GEMM time and
  0.35% of the step, under the triage bound of 1%. Recorded, not A/B'd.
- **NbCl5:** the lab's replay hit an illegal memory access, in the run and in its one retry. The failure is in
  the replay ("stages FAILED" comes before the point's steady error), most likely an eager-sample species plan that
  does not match the replayed grid. Production NbCl5 runs fine elsewhere.
- **leto:** this point ran at 29.9 ms per step against about 17 ms on leeloo, so leto's GPU is slower in general.
  The ratios are from one process and still hold.

## 7. Recommendations

1. **Stay on 2.14.1.**
2. **Ada and other GPUs where Triton runs tf32x3 (measured: t9 -3.7% SrTiO3 10³, -8.1% NbCl5, §5):**
   - The cheap route, an explicit register split on Triton 3.8, is closed (§6b: no shared-memory saving).
   - Fewer pipeline stages are closed too (§6c).
   - What remains is Triton 3.9 alone (needs Python 3.11 for production, plus its own gate and A/B) or a
     hand-written MMA kernel.
3. **Pin no cuBLAS:** no effect beyond noise on the Ada or the 5060 Ti.
4. **NVRTC stays 13.0 on consumer GPUs**: 13.2 is 4-7% slower on w' backward for sm_89 and sm_120.
5. **A later migration must:** add `nvidia-cuda-crt`, re-pin save_transcendentals, and keep NVRTC 13.0 for our
   kernels or fix the sm_89/sm_120 regression.
6. **Work removal found by the attribution (§6a, handed to step-layout-cleanup):** build `species_plan` once per
   species change, not every step; move the per-step compose out of the graph; remove the layout copies at
   `generic/modules.py:211`/`:530` and `tabled_outer_sum.py:739`/`:757`.
7. **A100: not measured.** Run the fixed one-job script (`jobs/alex_all.sh`) once the owner lifts the Alex hold.

## Files

- `jobs/`: all scripts.
  - `job.sh`: compat, cold, A/B, profiles, gate.
  - `matrix.sh`, `ws_matrix.sh`, `alex_matrix.sh`: the library matrix; `matrix_table.py`: the §5 table.
  - `alex_all.sh`: the one A100 job (attribution, then matrix, gates and census); not run.
  - `attribution.py`, `gaps_radix.py`, `layout_rows.py`: §6a; `split_lab.py`: §6b; `stages_lab.py`: §6c.
  - `profile_point.py`, `kernel_category.py`, `analyze.py`.
  - `census_nvrtc.py`, `nvrtc_census.sh`.
  - env builds: `cublas.sh`, `cublas138.sh`, `triton39.sh`, `local_build.sh`.
- `ada-leeloo/`, `rtx5060ti-local/`: summaries, run logs, points, profiles, compat JSON, gate metadata.
- `*/matrix/`: matrix records per machine (§5).
