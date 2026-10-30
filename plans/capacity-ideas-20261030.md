# Capacity and speed ideas, 30 October 2026

The conductor's assessment of the external idea lists the owner passed on during the capacity round. It records the
ideas that make sense for FastMACE, what each would need first, and which session has it. It ends with the ideas
that do not fit and why, so they are not proposed again without new evidence.

Context on main at the time:
- The leaner neighbour list (CAP-T09), the noise cap (CAP-T06) and the w rebuild (CAP-T07b) are merged.
- Whole-system MH-1 float32 limits on the RTX 5060 Ti: 51,840 NbCl5 and 49,248 NaCl atoms. With
  `recompute_node_segments` (CAP-T08, separate regions) they rise to 69,984 and 67,200.
- The tiled prototype (`fastmace.model.tiled`, `docs/design/tiled-backend.md`, `docs/validation/capacity-tiled.md`)
  fits 884,736 NbCl5 and 778,688 NaCl atoms on murphy's 20 GB, but runs eager, about 2.1x slower than main's step.

Owner scope (30 October 2026): no multi-GPU work for now; it is for later. Float32 only.

## Where FastMACE stands against Symmetrix-XL (conductor's assessment)

- **Speed at normal cell sizes, measured:**
  - The 14 October screen found Symmetrix-XL 1.9-2.5x slower than the hybrid default.
  - The 23 October LAMMPS rematch (MACE-OMAT-0 medium) found its native GPU path 1.1-2.5x slower per step than FastMACE's default.
- **Capacity:** the tiled path is the same class of design, with a flat 1.2-1.7 GiB workspace and 23-26 kB/atom.
  - Fits 884,736 NbCl5 atoms with MH-1 on 20 GB.
  - The paper's 11.24M atoms is MACE-OMAT-0 on an 80 GB A100, a much smaller per-atom state than MH-1's 16 kB/atom h1 and adjoint. Scaled crudely, MH-1 gives about 3.5M atoms on 80 GB, before freeing h1 (T2).
  - Expected to be in the same range as Symmetrix-XL on a matched model and GPU, perhaps somewhat below.
- **Speed at large sizes, the open point:** the tiled path is eager, about 2.1x main's step. With graph tile bodies (T1) the estimate is 1.3-1.5x main's per-atom cost, which would put it about even with Symmetrix-XL or a bit ahead. This is not measured.
- **To settle it:** a matched benchmark, MACE-OMAT-0 medium on the same structures and GPU, measuring largest cell and step time. It needs T1 and a tiled path for models whose first interaction is not species-only.

## Taken up

| # | Idea | Why | First check | Owner |
| --- | --- | --- | --- | --- |
| T1 | **One padded CUDA graph per tiled sweep**: tile slots padded to fixed receiver, halo and edge counts; padding edges beyond the cutoff (exact zero) or masked in the receiver scatter; each tile fed through static device buffers the graph reads, then replayed | The tiled path's 2x gap is eager execution, not tiling (tiling itself costs 40-45 %). One graph per sweep avoids a capture per tile shape | Sweep 2 on one padded tile, graph against eager, 16 iterations; target >= 1.8x | tiled-backend session |
| T1 result | Falsified (`docs/validation/capacity-tiled-t1t2.md`): the tile body is GPU-bound. Graph replay 1.00x eager on the largest NbCl5 tile, and padding to one max slot costs 0.53 → 0.75 s per call. Padded and graph modes stay opt-in | Next: compiled tile bodies with bucketed slots, a falsifier at ≥1.3x (ceiling about 1.45x, so about 1.5x main) |
| T2 | **Free h1 after sweep 2** (sweep 3 recomputes interaction 0 anyway) | The 16 kB/atom global state (h1 and its adjoint) is the tiled limit; exact; about 1.4x more atoms | Peak at the current limit with and without | tiled-backend session |
| T3 | **Host-staged h1 and dh1 in pinned memory**, double-buffered over PCIe behind tile compute | Workstations have 250 GB of RAM; per-tile copies are far shorter than tile compute. The dh1 halo contributions must be staged per tile and accumulated in a fixed order | After T1 and T2. The next limit is the neighbour list (about 3 kB/atom) plus positions, velocities and forces, so about 4-5M atoms on 16 GB, and the list build needs tiling too | later, tiled-backend session |
| C1 | **Symmetric contraction fused into the following GEMM** (the contracted product is produced in registers and consumed by the linear, never written) | Removes the [N,8192] product tensor (1.58 GiB per interaction at 51,840 atoms) on the normal path. That helps capacity as well as the 4-8 % speed claim. Earlier own-GEMM and combined-kernel attempts gave no speed, so expect less on speed. It replaces a cuEquivariance kernel; the backward still needs the products or a recompute (see `docs/design/opt3-T02-symmetric-contraction.md`) | Falsifier: DRAM traffic of that tensor as a share of step time, from the step profile, and its share of saved state with `recompute_node_segments` on | capacity worker, after CAP-T08 merges |

## Worth measuring before deciding

- **Lossless compression of h1/dh1 between sweeps.** The claimed 2.2-2.8x is optimistic; float32 activations
  usually compress 1.1-1.4x. Measure the ratio on a dumped h1 before writing an encoder. A lossy block-float
  variant changes precision and would need a `certified` record (SPEC 6.1), outside the float32-only focus.
- **Overlapping sweeps 2 and 3 across neighbouring tiles** on two streams. A 4,096-receiver tile (about 200k edges)
  likely fills the GPU already, so measure in-tile utilisation first. It is not bitwise: the atomic-add order
  changes. It doubles the tile workspace.
- **One generated kernel for the elementwise chain.** Inductor already fuses most of it, and earlier combo kernels
  gave nothing. About 300 kernels in a graph cost roughly 0.3-0.6 ms, which matters only for small cells. Profile
  the idle gaps on NaCl first.

From the earlier review (same day):
- **Combined gated-rows-plus-product recompute region:** measured. Separate regions won (1.095x against 1.24x,
  more atoms). See `docs/validation/capacity-T08-chunks.md`.
- **Atom tiling of the node-local recompute:** open. The combined region's recompute is the compiled peak.
- **Unused model outputs held through the backward:** screen running in the neighbour-list session.
- **Inductor `memory_planning`:** screen running in the neighbour-list session.

## Third list (same day, all claimed exact on the existing models)

| Idea | Assessment | Status |
| --- | --- | --- |
| **Edge-virial stress** (sum r_e ⊗ dE/dr_e in the edge backward epilogue) instead of the strain graph (`single_graph_strain`) | Sound for models whose energy depends on positions only through edge vectors; the tiled path already assembles stress this way. Rounding class, not bitwise as claimed. Gains only where stress is computed (NPT, cell relaxations). Models with cell-dependent terms (PolarMACE k-space, D3 on the cell) keep the strain path | Taken up (owner, 30 October): falsifier queued for the capacity worker after C1 |
| **Static liveness slab for graph activations** | The capture-time allocator from the pool study, reached via FX. FX does not see allocations inside external ops (OEq, cuEq, cuBLAS workspaces), which hold the largest blocks. Up to 1.47 GiB in the offline replay, but a process-wide allocator constraint. Less urgent now that the tiled path exists | Owner decision (SPEC 18), not started |
| **Morton or Hilbert tile order** in the tiled path | Tiles come from spatial blocks in index order. A tile's halo working set (about 4,096 + H atoms at 8 kB) is larger than the 32-48 MB L2, so cross-tile L2 reuse is small. The 15-22 % claim is unlikely, but it is cheap to try | Optional quick A/B inside T1 |
| **Packed shifts and CSR neighbour list** | The list is about 1 % of the whole-system peak since CAP-T09, so there is no capacity gain there. Shift values are bounded only for cells larger than the cutoff, so packing must handle small cells. Becomes relevant when the tiled path with host-staged h1 (T3) makes the list the limit | Revisit with T3 |
| **Co-scheduled irrep tiles in segment_gemm** | segment_gemm already launches one 2D grid per call. Whether irreps are separate calls, and whether a tail dip exists, needs a profile look before any work | Check the profile first |
| **Skipping the displacement check for K steps after a rebuild** | Langevin noise is unbounded, so there is no kinematic certainty. The check is already inside the captured step and its flag read is pipelined, so the saving is small. A wrong skip gives silently wrong forces | Not taken up |
| **Peer-to-peer multi-GPU tiling across workstations** | leto, murphy and trillian each have one GPU. Peer-to-peer copies need GPUs in one machine, and cross-machine exchange would be networking | Deferred: the owner wants no multi-GPU work for now (30 October 2026), "this is for later" |
| **cuMemMap growth for edge buffers on recapture** | The premise does not hold here. Idea 5 showed the limit failure is the larger capture's own peak, not two graphs resident at once, and expandable segments already map memory on demand | Not taken up |

## Fourth list (same day)

| Idea | Assessment | Status |
| --- | --- | --- |
| **Fused pair repulsion (ZBL)** | The 28 October profile shows pair_repulsion_fn at 59 forward and 58 backward launches. Per step that is 157 µs on NaCl (4.7 % of kernel time), 168 µs on NbCl5 (0.6 %) and 188 µs on NaNbO3 (0.9 %), so the claimed 2 % on NaNbO3 is high. First check whether the compiled default graph still runs it unfused; the 59 launches may be eager or a fusion break. Prefer one exact fused kernel computing ZBL analytically (rounding class) over a Chebyshev table. A table is a `certified` approximation, and folding it into the radial tables only works where `radial_tables` (opt-in) is on | Taken up (owner, 30 October): falsifier queued for the neighbour-list session |
| **Spherical-harmonic gradient on chip in wprime_backward** (write the edge force, not dsh[E,16]) | Sound: the l <= 3 harmonic gradient is a short closed form, rounding class. dsh is a transient of 64 B/edge, so the gain is bandwidth plus one backward kernel. Check the SH-backward kernel time in the profile first | Measure first: profile read queued for the neighbour-list session |
| **segment_gemm with D as constexpr** | Confirmed in `fastmace/kernels/triton/segment_gemm.py`: D (1, 3, 5, 7) is a runtime argument and `r // D`, `r % D` run per row tile. A constexpr D gives one compiled variant per irrep, a general rule. The joined products also run one launch per irrep (a loop over dims), which bears on the third list's co-scheduling idea. Cheap, exact | Taken up (owner, 30 October): A/B queued for the neighbour-list session |
| **Host-staged h1/dh1** | Same as T3 | Already taken up (after T1, T2) |
| **Receivers from row_ptr instead of a stored array** | 4 B/edge. The neighbour list is about 1 % of the peak, and a row_ptr search adds work in the edge-owned kernels | Not taken up for now; revisit with the packed list under T3 |

## Not taken up

| Idea | Reason |
| --- | --- |
| Multiple time stepping (RESPA by rank or cutoff) | MH-1 has no separable cheap pair term, and stiff parts of the force are not confined to one. A 2-4 fs outer step risks resonance. It changes the dynamics, and the owner ruled out larger timesteps (1 fs production protocol). Only as a separate protocol study if the owner revisits that |
| Reversible (RevNet) equivariant blocks | Needs a new architecture and retraining; the tiled path gives the capacity without it |
| Short cutoff (3.5 A) plus a PME or multipole tail | A different model needing training and validation, not an optimisation of an existing one |
| Directional JVP force stepping | The force rate along v is a Hessian-vector product, which still needs a reverse pass. Extrapolated forces are not conservative and drift |
| Spatial tiling with discarded halo forces | Done properly as the three-sweep tiled path. Discarding halo forces is wrong for two message-passing layers unless the halo is doubled |
| Wyckoff orbit folding | Thermal motion breaks the symmetry at the first MD step. Only exactly symmetric relaxations would gain, and they are not a bottleneck |
| Stochastic-rounding FP8 absorbed into the thermostat | Rounding noise depends on the force values, is correlated and non-Gaussian, so fluctuation-dissipation breaks. NVE and relaxations have no thermostat. E4M3's 3 mantissa bits are far below BF16, which already failed the gate. Outside the float32 focus |
| Distilled direct-force head | A different, non-conservative model; a curl-free projection on a grid does not recover MACE's forces |
| Planned segment_gemm workspace | Falsified at the tracing spike (`docs/validation/capacity-workspace.md`) |
| Capture-time allocator following the offline packing plan | Up to 1.47 GiB, but a process-wide allocator constraint. Needs a SPEC 18 decision; not started |
