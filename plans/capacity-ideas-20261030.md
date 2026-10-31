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
| T2 result | Done differently and merged (218937d2): only linear_up(h1) and skip_tp(h1) and their adjoints are kept, and h1 is recomputed per tile in sweep 3. Exact, about 17 → 4.75 kB/atom. **Correction (30 October 2026):** the 1,728,000 and 2,000,376-atom NbCl5 points quoted in the merge message are invalid. The GPU pair screen's grid product overflowed 32 bits and Triton skipped the launch silently, giving zero edges and zero forces. The valid checked NbCl5 limit is 1,481,544 atoms; the true limit lies between that and about 2.2M (rescan with an edge-count check pending). At a valid 2,299,968-atom run the list holds 4.2 kB/atom and the global state 4.75 kB/atom, about equal next terms | Fix merged (939c3f6e): launch capped at 2^31-1 blocks with a receiver-tile loop, empty-screen brute-force check, audit (only the pair screen could overflow up to 10M atoms), scans mark points invalid without matching edges/atom. Rescan with the edge check next (tiled session) |
| T3 | **Host-staged h1 and dh1 in pinned memory**, double-buffered over PCIe behind tile compute | Workstations have 250 GB of RAM; per-tile copies are far shorter than tile compute. The dh1 halo contributions must be staged per tile and accumulated in a fixed order | After T1 and T2. The next limit is the neighbour list (about 3 kB/atom) plus positions, velocities and forces, so about 4-5M atoms on 16 GB, and the list build needs tiling too | Done (7a07cc7f, TileConfig.state="host", with 54a13ee5 slot-row species tables and 0b8eaf28 budget-driven R): valid tiled limits on murphy's 20 GB NbCl5 and NaCl 7.76M atoms (call GPU peak about 0.75 kB/atom, 35 GiB host, copies about 5 %). Next limit: the build (about 2.2-2.4 kB/atom: float64 one-hot E0, whole-system edge arrays, allocator cache); build follow-up taken up |
| C1 | **Symmetric contraction fused into the following GEMM** (the contracted product is produced in registers and consumed by the linear, never written) | Removes the [N,8192] product tensor (1.58 GiB per interaction at 51,840 atoms) on the normal path. That helps capacity as well as the 4-8 % speed claim. Earlier own-GEMM and combined-kernel attempts gave no speed, so expect less on speed. It replaces a cuEquivariance kernel; the backward still needs the products or a recompute (see `docs/design/opt3-T02-symmetric-contraction.md`) | Falsifier: DRAM traffic of that tensor as a share of step time, from the step profile, and its share of saved state with `recompute_node_segments` on | capacity worker, after CAP-T08 merges |
| C1 result | Stopped (`docs/validation/c1-contraction-fusion.md`). The [N,8192] tensor is the contraction's input Y. A forward epilogue fusion has a ceiling of 3.4 % and the backward is not buildable; removing Y needs a backward rebuild (+8-16 %), so it belongs to atom tiling. Side finding taken up: segment_gemm with a column-fastest tile order, 0.80-0.84x on l ≥ 1 products, bitwise, about 5 % of the step | First check | capacity worker: tile-order rewrite, then edge-virial |
| Tile-order result | Merged (bb208f2a), per-GPU default under decision 66: on only for "NVIDIA GeForce RTX 5060 Ti" and "NVIDIA RTX 4000 Ada Generation" (`tile_rule.COLUMN_ORDER_GPUS`), and only where the products do not fit in L2. Bitwise. MD step NbCl5 9,216 atoms 0.938-0.940 (5060 Ti), 0.923-0.925 (Ada); NaNbO3 8,640 atoms 0.955-0.959, 0.944-0.945; A100-SXM4-80GB neutral (0.999-1.002), so it keeps the row order (`docs/validation/segment-gemm-order.md`) | Done | — |
| Edge-virial result | Stopped (`docs/validation/edge-virial-falsifier.md`). Exact (float64 within 3e-13 meV/Å³ of the strain path; float32 error at or below the strain path's), but a force call with no stress at all is only 0.3-1.4 % faster than the strain path, with equal peak memory, so an edge virial cannot gain more | Done | — |
| A/B tool audit | `tools/ab_rewrites.py --a/--b-constant` set constants only around the build; graphs captured in warm-up (all of them before `--prepare`) could run the default. Fixed (d16d41c6). Audit (`docs/validation/ab-constants-audit-20261030.md`): tiles128 invalid, rerun running; MARK_ATOMS_DYNAMIC step-time claim at risk, recheck queued; all other constant A/Bs valid. The speed-def A/Bs use `ab_swap.py`, which sets the side per step (trace: 0 wrong-side calls) | Reruns | capacity worker |

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
| **Density derivative in the wprime_backward epilogue** (fold `density_backward` and the `dr +` add into the kernel's single dr store) | Confirmed at `wprime_backward.py:424`; it runs only where the density rewrite is on. It saves one small kernel, one add and one [E] read and write. Rounding class (summation order changes) | Stopped: density_backward is one 2-10 µs launch per step (0.05-0.08 %), and the add is already fused by Inductor; under the 10 µs bar |
| **Single-pass fused compaction** (one Triton stream-compaction kernel instead of cumsums, where and three index_copy) | Valid, and it overlaps the compact memory work already in the tiled session. The speed gain is per rebuild (every few to tens of steps), so small. Its value is memory: no [C] destination and int64 temporaries | Not needed: the chunked compact (tiled session) already removed the memory term; the fused kernel would mainly save time, with exactness risk from FMA contraction |
| **Readout scale and shift folded into the readout weights** | Saves at most a couple of elementwise nodes, which Inductor likely fuses into neighbours already. Folding changes rounding (W·scale) and touches checkpoint semantics (per-head scale and shift, PolarMACE and OMOL variants) | Not taken up: negligible gain, accuracy and generality cost |
| **Cutoff multiplied in on load instead of storing augmented [E,65]** | `augmented` exists on the MH-1 path (`rewrites/mh1/modules.py`, edge_outer_sum input). The "13 GB at 2M atoms" assumes 25 edges/atom; NbCl5 has about 6.4 (58,880 edges at 9,216 atoms), and the tiled path holds it per tile only. On the whole-system path at the NbCl5 limit it is about 0.1 GB; NaCl (denser) is larger. Exact if the product is formed identically | Measure first: its bytes in the memory ledger at the NbCl5 and NaCl limits; build only if ≥0.3 GiB. Result: 0.38 GiB (NbCl5 54,432) and 0.42 GiB (NaCl 51,984), saved by edge_outer_sum through the backward. The proposed layout saves nothing on the default plan, since `augmented` is the radial table's own output and no separate [E,64] exists. Follow-up taken up: rebuild `a` from the saved lengths in the backward as a memory-mode rewrite (decision 67), if it is at the peak and costs ≤3 % when on |

## Seventh list (same day)

| Idea | Assessment | Status |
| --- | --- | --- |
| **Per-tile neighbour storage** (build and keep the list per tile, or host-staged per tile, instead of one global list) | Right direction: the global list grows with N and is the next capacity term on the tiled path once compact and the sort are lean. The "15 MB" figure leaves out the build itself (the candidate search needs the neighbouring cells of each tile) and the halo edges each tile needs, so the GPU holds more than one tile's list. Exact if each tile's edges equal the global list's edges for its receivers, in the same order. This is the list-build-tiling part of T3 | Done (6a7e678e, TileConfig.neighbours="tiles"): edges bitwise the global list's; memory before the call 4.3 → 0.85 kB/atom; valid limits on 20 GB NbCl5 2.00M → 2.99M and NaCl 1.73M → 2.99M atoms. Next limit: the build's N-row identity species table (about 5.9 kB/atom); fix (a), slot-row table in the tiled path (bitwise), taken up, then host staging |
| **27-entry shift table plus a one-byte shift code** | Unit shifts are bounded to {-1,0,1} only when the cell is larger than cutoff plus skin in each direction; small cells need more, so the table size must come from the cell. The table entries computed with the same `times_3x3` are bitwise identical. Speed gain is per rebuild only; the real value is memory: 1 B instead of 12 B of int32 unit shifts per slot | Folded into the per-tile list work (packed list) |
| **Chebyshev coefficients in constant memory** | Does not fit: the coefficient table is P x 10 x 1280 floats (51 kB per pair), and constant memory is 64 kB in total. Neighbouring edges use different pairs and bins, and constant memory serialises divergent addresses, so it would be slower. Coefficients are already L1/L2-served | Not taken up |
| **Warp specialisation in wprime_backward** (producer and consumer warps) | High complexity. Only worth it if ncu shows the kernel neither memory- nor compute-bound but latency-bound with both pipes idle | Measure first (ncu), low priority |
| **In-place ballot compaction** | Same as the sixth list's fused compaction. The chunked compact already removed the memory term (one byte of mask per candidate), and a fused kernel would mostly save time while risking exactness through FMA contraction | Not needed after the chunked compact |

## Eighth list (31 October 2026)

None of these adds capacity except GPUDirect Storage, which conflicts with the workstation file rule. Two cheap
measurements settle ideas 1, 2, 4 and 5 before anything is built.

| Idea | Assessment | Status |
| --- | --- | --- |
| **L2-locked micro-tiles** (512-768-atom tiles sized to L2, pinned with the CUDA access-policy window) | The proposal's own budget has w at 61 MB per tile, above the 32 MB (5060 Ti) and 48 MB (Ada) L2; rebuilding w costs 8-16 % (C1). The receptive field (about 12 Å over two interactions) makes the halo several times a 22 Å tile, so most work becomes halo recompute, and small tiles were launch-bound in the tiled path. Cells that already fit in L2 gained nothing from the tile-order change, so residency alone is not the bottleneck; fixing one kernel that rereads beyond L2 was (segment_gemm, 5-8 %) | Not as proposed. Taken up as measurement M1: ncu L2 hit-rate audit of the top 10-15 kernels at about 9,000 atoms on sm_120 and Ada, to find other segment_gemm-like kernels; persistence windows may be one per-kernel fix |
| **GPU-resident MD loop with conditional graph nodes** | Feasible with CUDA 13, but rebuilds change edge counts (fixed edge-capacity buckets needed), and output and thermostat noise would have to move onto the device. The only gain is the host gap between steps, which matters for small cells, not production sizes | Measure first, M2: nsys GPU idle time between steps at about 1,000 and 9,000 atoms; stop if under 2 % of the step |
| **GPUDirect Storage tile streaming from NVMe** (50M atoms) | At 50M atoms the per-atom state and list are about 9 kB/atom (about 450 GB), read and written over three sweeps per step: minutes per step at 7-14 GB/s, not hidden behind tile compute. Workstation runs may write only under /mnt/homes/zwingel (no local disks), and WSL has no GDS. Host RAM staging (T3) is the next step | Not taken up; revisit only if a target needs more than about 25M atoms |
| **Morton-voxel atom order with TMA bulk copies** | A species/Morton order gave no gain in the research engine; supercells are already spatially ordered. Ada (sm_89) has no TMA. A receiver's senders do not form contiguous blocks even in Morton order | Not taken up; the gathers' cache behaviour is covered by M1 |
| **Pure-SIMT cooperative megakernel** (one kernel per force call, grid syncs) | Means rewriting the cuEquivariance, OpenEquivariance and Triton kernels and the backward into one kernel; a fused heterogeneous kernel was rejected in round 4. The ceiling is the inter-kernel gaps of 230-320 nodes, about 2-4 µs each, under 1 ms per step | Measure first, M2 (same trace: gap total inside the force graph); reuse the fp32-profile numbers if they already contain it |

M1 runs on sm_120 only: the workstations restrict GPU performance counters to admins (RmProfilingAdminOnly=1),
Modal's image has no Nsight tools, and the Colab account had no compute units on 31 October.

## Ninth list (31 October 2026)

| Idea | Assessment | Status |
| --- | --- | --- |
| **Group-ordered wprime_backward** (edges permuted by (pair, piece) so a block reuses one G′ table) | Well grounded. OPT9-T01 measured receiver-order 2.46 vs group-sorted 0.46 ms (Ada, NaNbO3) for this term in isolation and stopped only because the fused kernel was receiver-owned. If the built kernel is edge-owned (per-edge arrays, atomic dx), the order is free. dsh/dr stay bitwise per edge; dx is already an atomic sum, so it stays in the existing rounding class. The permutation must be rebuilt per step inside the graph (counting sort over a few hundred groups) | Taken up: falsifier on captured real operands (1000 K), both GPUs, as proposed (GO ≤ 0.80x NaNbO3 / ≤ 0.90x NbCl5 with sort ≤ 50 µs; STOP ≥ 0.92x everywhere). Capacity worker, after tiles128 |
| **Atom-chunked checkpointing of the combined node region** | The fix CAP-T08 named but did not build: the combined region saves 62 vs 97.5 kB/atom, but its whole-region recompute is the backward peak. Capacity is the priority, and it is a decision-67 memory mode (planned only when the plan does not fit). Risk: AOT keeps every chunk's output alive, or the chunked call is slower than the 1.24x combined region | Taken up: prototype and falsifier at NbCl5 21,600 atoms (GO peak ≤ 3.0 GiB and call ≤ 1.35x separate-on; STOP peak ≥ 3.3 GiB or ≥ 1.45x). Neighbour-list session, with the rebuild-`a` memory mode (same peak, same planner rule) |
| **Tiled tile size and slot bucket from the workspace budget** | Cheap screen of R ∈ {4096, 8192, 16384} x bucket ∈ {0.15, 0.05} on the compiled tiled call; padding and halo work are real. The rule must come from the memory model and stay under Dynamo's recompile cap; no per-structure table | Taken up: screen at NbCl5 21,600 and 51,840 atoms (GO ≤ 0.92x with workspace ≤ 2 GiB and ≤ 5 slot changes in 300 steps). Tiled session, after the rescan, alongside T3 |
| **dr fused into the interaction-0 edge_outer_sum backward** | Ceiling about 0.5-1 % of the step; item F (a store change in the same kernel) just measured neutral on both GPUs | Measure first: edge_outer_sum backward plus the interaction-0 table backward from the M2 trace; STOP below 100 µs/step |
| **Certified float64 tables for D3(BJ)** | Plausible (smooth 1-D and 2-D functions, the radial-table certificate machinery exists), but only for --dispersion d3bj runs, which are not the capacity or default path, and D3's reference status needs a ≤ 1e-10 certificate | Deferred: profile a D3-on step first when D3 production runs start |

Results (31 October 2026, merged 4dd46874 and 3ec374a2):
- Item 1, group-ordered wprime_backward: STOP. The G′ tables fit in L2 on both GPUs; NbCl5 1.29x (Ada) and 1.43x
  (sm_120), NaNbO3 0.80x and 1.02x (`docs/validation/wprime-group-order.md`).
- Item 3, tiled R x bucket: no GO. R=16,384 is about 0.90x but needs about 4.2 GiB of workspace; within 2 GiB the
  best is 0.97x; bucket 0.05 buys nothing (`docs/validation/tiled-screen.md`). Follow-up taken up: R from the memory
  budget (largest of 4096/8192/16384 that fits after the tiled state), so the limit is unchanged.
- Item 4: edge_outer_sum backward plus the interaction-0 table backward are 0.5-0.57 % of the step at 9k atoms. Not
  built.
- Eighth list M1 (sm_120): no kernel rereads beyond L2; nothing with a ceiling above about 1 %. M2: host gap
  0.24-0.8 % and in-graph gaps 0.06-0.8 % of the step, so the GPU-resident loop and the megakernel are dead
  (`docs/validation/l2-gap-audit.md`).
- tiles128 rerun: default on the RTX 4000 Ada only (0.986-0.999); RTX 5060 Ti 1.007-1.020 and A100-40GB 1.021 slower.

Note on the proposal's premises: gate epilogues are not dead after wprime_backward. With the K-outer kernel, gate_epilogue=on measured
1.2-4.3 % faster on sm_120 (speed-def E2, Ada pending).

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
