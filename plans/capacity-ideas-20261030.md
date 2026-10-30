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
| C1 result | Stopped (`docs/validation/c1-contraction-fusion.md`). The [N,8192] tensor is the contraction's input Y. A forward epilogue fusion has a ceiling of 3.4 % and the backward is not buildable; removing Y needs a backward rebuild (+8-16 %), so it belongs to atom tiling. Side finding taken up: segment_gemm with a column-fastest tile order, 0.80-0.84x on l ≥ 1 products, bitwise, about 5 % of the step | First check | capacity worker: tile-order rewrite, then edge-virial |

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
| **Fused pair repulsion (ZBL)** | The 28 October profile shows pair_repulsion_fn at 59 forward and 58 backward launches. Per step that is 157 µs on NaCl (4.7 % of kernel time), 168 µs on NbCl5 (0.6 %) and 188 µs on NaNbO3 (0.9 %), so the claimed 2 % on NaNbO3 is high. First check whether the compiled default graph still runs it unfused; the 59 launches may be eager or a fusion break. Prefer one exact fused kernel computing ZBL analytically (rounding class) over a Chebyshev table. A table is a `certified` approximation, and folding it into the radial tables only works where `radial_tables` (opt-in) is on | Stopped: already fused in the compiled step (at most about 15 µs/step on NaCl, threshold 100) ([speed-small.md](../docs/validation/speed-small.md)) |
| **Spherical-harmonic gradient on chip in wprime_backward** (write the edge force, not dsh[E,16]) | Sound: the l <= 3 harmonic gradient is a short closed form, rounding class. dsh is a transient of 64 B/edge, so the gain is bandwidth plus one backward kernel. Check the SH-backward kernel time in the profile first | Stopped: the SH backward is one kernel of 1.2-8.1 µs per force call, under 0.1 % ([speed-small.md](../docs/validation/speed-small.md)) |
| **segment_gemm with D as constexpr** | Confirmed in `fastmace/kernels/triton/segment_gemm.py`: D (1, 3, 5, 7) is a runtime argument and `r // D`, `r % D` run per row tile. A constexpr D gives one compiled variant per irrep, a general rule. The joined products also run one launch per irrep (a loop over dims), which bears on the third list's co-scheduling idea. Cheap, exact | Measured, not adopted: exact, A/B neutral (0.992-1.004), more compiled variants; no tail gap between the per-irrep launches ([speed-small.md](../docs/validation/speed-small.md)) |
| **Host-staged h1/dh1** | Same as T3 | Already taken up (after T1, T2) |
| **Receivers from row_ptr instead of a stored array** | 4 B/edge. The neighbour list is about 1 % of the peak, and a row_ptr search adds work in the edge-owned kernels | Not taken up for now; revisit with the packed list under T3 |

## Fifth list (same day)

Lesson from the fourth list: `results/fp32-profile-20261028/*/parts/*.csv` time the eager tier (fp32_parts), not the compiled graph-tier step. Launch counts and µs there do not carry over to production; check the compiled step's kernels (nsys node trace plus Inductor output_code) first.

| Idea | Assessment | Status |
| --- | --- | --- |
| **One grouped GEMM for FirstInteractionReassociation** (4 per-l GEMMs plus cat into one kernel, forward and backward) | The 189 µs and 49 launches on NaCl are the eager tier's (`parts.csv`). The compiled step still runs per-l matmuls plus copies, at an unknown cost. Structurally it is a grouped GEMM over irrep row blocks with per-l weights, which is what `segment_gemm`'s joined products already do, so reuse that kernel rather than writing a new one. Rounding class, not bitwise (different accumulation order) | Measure first: compiled-step kernel time of this region on NaCl, NbCl5 and NaNbO3. Build with segment_gemm only if ≥60 µs/step on NaCl. Sent to the neighbour-list session (owner, 30 October) |
| **Loop inversion in gated_products** (K outer, m inner, weight tile loaded once per K block) | Confirmed in `fastmace/kernels/triton/gated_products.py`: the m loop is outer (line 53) and the K loop inner (55), so each weight tile is loaded D times. Inverting needs D accumulators of BLOCK_M x BLOCK_N (7 for l=3), so register pressure or spills may be why it is written this way. Could rescue `gate_epilogue` (OPT10-T01, rejected at -0.3 to -0.9 %) | Worth an A/B: inverted against current, both orders, NbCl5 and NaNbO3, including `gate_epilogue` on. Sent to the neighbour-list session |
| **Warp broadcast of sender and SH loads** in the OEq streaming forward | Loads of one address by all lanes of a warp are already served as a single broadcast transaction by the hardware. The gain is only issue slots, if the kernel is LSU-bound. It is vendored OEq code | Low priority: only if ncu shows the forward LSU-bound |
| **Coalesced 2D store in edge_outer_sum backward** | Valid micro-optimisation, exact. 20-30 µs claimed from the eager-tier profile | Cheap check: its kernel time in the compiled step, then a quick A/B if ≥10 µs. Sent to the neighbour-list session |
| **Displacement max fused into the integrator** | Already implemented: `fused_integrator.py` computes the squared displacement against the reference positions as an exact atomic max (lines 19, 81, 95) | Already done |

## Sixth list (same day)

| Idea | Assessment | Status |
| --- | --- | --- |
| **Sort-free pair placement** (count per receiver tile, prefix sum, write each tile's pairs into its own range instead of a global atomic append plus `torch.sort(key)`) | Confirmed: `skin.py` sorts a 64-bit key over every found pair (lines 371-376 and 464; also `d3_list.py:211`). At 2M+ atoms that is a transient of key + sorted copy + indices + CUB workspace, a capacity term for the tiled path once compact is lean (compact is the current limit at 2.3M NbCl5). The two-pass placement groups by receiver tile only; the list's order (the sort key) must still match exactly, so the within-tile order needs a defined rule or a small segmented sort. The "2.4 GB at 100M pairs" figure is not measured | Taken up together with the compact work in the tiled session: measure the rebuild peak with compact fixed first, then build if the sort is the next limit |
| **Density derivative in the wprime_backward epilogue** (fold `density_backward` and the `dr +` add into the kernel's single dr store) | Confirmed at `wprime_backward.py:424`; it runs only where the density rewrite is on. It saves one small kernel, one add and one [E] read and write. Rounding class (summation order changes) | Measure first: those kernels' time in the compiled step; low priority. Sent to the neighbour-list session (owner, 30 October) |
| **Single-pass fused compaction** (one Triton stream-compaction kernel instead of cumsums, where and three index_copy) | Valid, and it overlaps the compact memory work already in the tiled session. The speed gain is per rebuild (every few to tens of steps), so small. Its value is memory: no [C] destination and int64 temporaries | Folded into the tiled session's compact work |
| **Readout scale and shift folded into the readout weights** | Saves at most a couple of elementwise nodes, which Inductor likely fuses into neighbours already. Folding changes rounding (W·scale) and touches checkpoint semantics (per-head scale and shift, PolarMACE and OMOL variants) | Not taken up: negligible gain, accuracy and generality cost |
| **Cutoff multiplied in on load instead of storing augmented [E,65]** | `augmented` exists on the MH-1 path (`rewrites/mh1/modules.py`, edge_outer_sum input). The "13 GB at 2M atoms" assumes 25 edges/atom; NbCl5 has about 6.4 (58,880 edges at 9,216 atoms), and the tiled path holds it per tile only. On the whole-system path at the NbCl5 limit it is about 0.1 GB; NaCl (denser) is larger. Exact if the product is formed identically | Measure first: its bytes in the memory ledger at the NbCl5 and NaCl limits; build only if ≥0.3 GiB. Sent to the neighbour-list session |

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
