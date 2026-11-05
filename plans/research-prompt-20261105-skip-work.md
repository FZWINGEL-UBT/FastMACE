# Research prompt: skipping work in FastMACE's OMAT-0 A100 MD step (5 November 2026)

Copy everything below the line into the research assistant. It has read access to the GitHub repository
`FZWINGEL-UBT/FastMACE` (private); branch `main` at `eb66bb2c` or later, plus `fastmace/round5` for the latest records.

---

You are helping to speed up FastMACE, a GPU molecular-dynamics engine for MACE machine-learning interatomic
potentials (repository `FZWINGEL-UBT/FastMACE`, branch `main`; the latest lab records are on `fastmace/round5`).
The target is **OMAT-0 medium on an NVIDIA A100-SXM4-80GB**, float32, Langevin NVT at 1 fs. The benchmark is SrTiO3
10x10x10: 5,000 atoms, 373,760 edges within the 6 Å cutoff, padded to 392,448. One MD step takes about 11.4-11.8 ms.

We have spent five rounds on kernel-level tuning. The two largest kernels are now near a hardware limit, and recent
scheduling ideas (producer/consumer warps, prefetching, register caps, more accumulators, FP16 splits) all lost. **We
think the next real gains must come from doing less work: new, exact (or FP32-class) ways to skip, reuse or avoid
computation.** Please look for those. Think from the physics and the model's algebra as much as from the GPU.

## Where the time goes now (A100 profile at `eb66bb2c`)

Records: `results/omat0-a100-profile-eb66bb2c/` (breakdown/ops.csv, categories.csv, stages.csv, sources/).

Per MD step (ms): force call 11.17; neighbour-list rebuild 1.36 amortised (11.35 ms per rebuild, 24 rebuilds in 200
steps, about 7 ms of it host-side Python per rebuild); noise upload 0.53; integrator 0.08; overlaps hide about 1.3.

Inside the force call (11.04 ms of kernels):

| us per call | Kernel | What it is |
|---|---|---|
| 3,167 | `fused_folded_forward` | Interaction 2 message: per edge, w = a(r) · W (3xTF32 mma.sync, K = 64 radial features, Q = 1,280 = 10 paths x 128 channels), then the uvu tensor product of x[sender] (128x(0e+1o+2e+3o)) with the edge's spherical harmonics (l <= 3), summed per receiver |
| 3,078 | `oeq_projected_backward_wprime` | Its backward: per edge, the gradients w.r.t. sh (16 components), r and x[sender]; w rebuilt from the radial table (TABLE_W), one warp per edge |
| 1,446 | CUTLASS 3xTF32 GEMMs (`mm_into`, 26 launches) | Atom-level linear layers, forward and backward |
| 387 | CUTLASS species GEMMs (`species_into`, `segment_rows`) | Per-species and per-irrep block linears |
| 454 + 388 | `tabled_outer_sum` backward / forward | Interaction 1's first convolution from radial tables |
| 393 | cuEquivariance `segmented_polynomial` | Symmetric contraction (correlation 3), forward and backward |
| 177 + 23 | `radial_table` forward / backward | Radial MLP evaluated from per-bin polynomial tables (degree 5) |
| about 1,000 | Inductor elementwise kernels | Gates, residual adds, readouts, slices |

Measured bounds on the forward (kernel lab, thermalised capture, `results/round5-20261105/runs/a100-forward-stage-bounds/`):
removing the whole projection 0.473 of the kernel, removing the tensor product 0.613, removing only the MMAs 0.664.
The projection (1.71 ms) and the tensor product (1.26 ms) run back to back. The projection is 373,760 x 64 x 1,280 x 2
x 3 passes = 184 GFLOP, at least 1.18 ms at the A100's 156 TFLOPS TF32 peak, so the tensor cores are near
saturation while it runs. The w' backward's end-of-edge reduction is only about 3 % of that kernel; the rest is the
per-edge recomputation and the dsh / dx products.

## Rules a proposal must respect

- Accuracy: forces, stress and energy against the float64 reference must stay within 2x the error of the float32
  MACE calculator (`tools/force_stress_gate.py`, SPEC §12). "FP32-class" is the bar; TF32 or FP16 alone is not.
  Hybrid runs need not be bitwise reproducible, but MD must stay physically identical (same model, same integrator,
  same 1 fs timestep; no larger timesteps, no multiple time stepping that changes the dynamics, no skipping force
  evaluations).
- General: it must work for any crystal and composition without per-structure tuning tables. Per-GPU defaults are
  allowed only where same-process MD A/Bs measured them faster in both build orders (SPEC decision 66).
- One GPU, one structure at a time (no multi-GPU, no batching of structures).
- Optimisations that change the model's arithmetic must be exact in exact arithmetic (reordering, factoring,
  reuse); approximations (low rank, coarser tables) only if they pass the gate with margin, and they are
  second choice.

## What has been tried (do not re-propose without a new angle)

Read `docs/optimisation-ledger.md` (index rows `omat0-*`, `conv-*`, `elt-*`, `model-*`) and
`plans/ideas-20261105-round2.md` (rounds 2-5, with falsifier numbers). In short:

- Forward: tensor-core 3xTF32 projection (default), FP32 SIMT on other GPUs, VEC_A / VEC_SH / TC_WFRAG layouts
  (default), pre-split W (1.362), 3xFP16 (1.197), tensor product on tensor cores (1.276), cp.async prefetch (1.25),
  fewer warps (1.11-1.16), producer/consumer warps (0.99-1.10 in a microbenchmark), split accumulators (1.31-1.50),
  path-specific precision (bound 0.964, rejected), structure-aware plan selection (owner: no per-crystal tuning).
- w' backward: butterfly and fixed-point reductions (5060 Ti only), register caps (A100: uncapped is fastest),
  Horner tables (default), stored-w streaming (rejected), channel splits (1.20-1.66), receiver-owned backward and
  path grouping (lost).
- Radial: per-bin polynomial tables (default), radial-manifold rank compression (refuted by
  `tools/radial_rank_lab.py`), table in the forward (bandwidth), table-w in the forward (rejected).
- Atom level: species GEMMs, segment_rows, head slice, one-graph sums, fused first interaction (defaults); cuEq
  symmetric contraction left as is.
- Mirror-edge sharing of w (edge i->j and j->i share r) was judged to need a memory round trip of about 1.9 GB per
  call; w is not stored on the A100 (STORE_W false).

## What we want from you

1. **New ways to skip work**, ranked by expected saving per MD step on the A100. Ideas in the direction we mean:
   - Exploiting that w depends only on r and the species pair (the radial tables are per ordered pair: 9 for
     SrTiO3, `[9, 896 bins, degree + 1, columns]`; check in the model whether the pair dependence is symmetric):
     exact sharing between mirror edges, or between edges with equal (pair, r), inside one kernel without a global
     round trip.
   - Reuse across MD steps: positions move about 0.01 Å per fs. Is any intermediate exactly or FP32-reproducibly
     reusable between steps (neighbour structure, sorted orders, radial bins, sh of slow edges), or can the
     neighbour-list rebuild (1.36 ms per step, mostly host time) be made much cheaper or rarer without changing the
     dynamics?
   - Algebraic restructuring of the uvu tensor product with l <= 3 (Clebsch-Gordan sparsity, factoring the
     projection W through the product, contracting the backward's 16 sh components earlier), with a FLOP count
     against today's.
   - Doing the backward with less recomputation (what the forward already had in registers), or fusing forward and
     backward so per-edge values are computed once.
   - Atom-level layers: removing GEMMs or elementwise passes by folding consecutive linears, gates or residuals.
2. For each idea: the mechanism; why it is exact or FP32-class; a FLOP or byte count against the profile numbers
   above giving an upper bound on the saving; the cheapest falsifier experiment that would refute it (we run these
   in `tools/omat0_kernel_lab.py` and `tools/ab_step_time.py` on Modal A100s); the files it would touch; and the
   risks (registers, shared memory, generality across structures and other MACE models).
3. Point out any measurement in our records that looks inconsistent or any bound we derived wrongly.

Start from `fastmace/kernels/oeq_streaming/fused_forward.py`, `fastmace/kernels/oeq_streaming/wprime_backward.py`,
`fastmace/kernels/triton/tabled_outer_sum.py`, `fastmace/kernels/triton/radial_table.py`,
`fastmace/model/forces.py` (neighbour list and rebuild), `fastmace/rewrites/` and `SPEC.md`. Prefer few, well-argued
ideas with honest bounds over a long list.
