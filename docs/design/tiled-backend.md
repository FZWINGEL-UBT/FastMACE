# Tiled capacity path: a receiver-tiled, three-sweep force call (design note)

Status: design note and prototype, 30 October 2026. No library default and no SPEC section changes. Asked for by the
owner through the conductor, after an outside review pointed at Symmetrix-XL's tiled mode.

## 1. The idea and why it fits FastMACE now

**Where the memory goes today.** At the float32 MH-1 limit (NbCl5, 51,840 atoms, RTX 5060 Ti), the force call's graph
pool holds 11.5 GiB live, about 240 kB per atom ([capacity-ledger.md](../validation/capacity-ledger.md)). Almost all
of it is per-atom activations kept for the backward or alive in it:

- the symmetric contraction's inputs;
- the `segment_gemm` gated products;
- the convolution message;
- the gate activations;
- the linear inputs.

See [capacity-T08-inventory.md](../validation/capacity-T08-inventory.md). After CAP-T07 and CAP-T09 the per-edge state
is small. So every lever left for capacity is a per-atom lever:

- recompute_node_segments (CAP-T08 chunks, in another session) chunks the node-local segments. It reaches about
  +35 % atoms;
- but it leaves the convolution backward and both interactions' saved state whole-system.

**Symmetrix-XL's tiled mode** (arXiv 2610.01036, §5 and §7.2) bounds every such intermediate by a tile:

- **Tiles.** Receivers and their incident edges are partitioned into tiles of at most 8,192 receivers.
- **Three sweeps** reuse one workspace.
- **Global state.** Only the inter-layer node state H₁ and its adjoint stay graph-wide.

On MACE-OMAT-0 it reaches 11.2M atoms on an A100 80 GB, about 7 kB per atom, against 1.25M without tiles. That costs
about 4 % per atom-step at each mode's own limit.

**This note maps that schedule onto MH-1's float32 hybrid path.** It reuses FastMACE's kernels and rewritten modules,
and keeps the result accuracy-equivalent: float32 rounding against the current path, and the force/stress gate. It
does not try to match Symmetrix-XL's headline numbers: different model, different GPU, and MH-1's H₁ is 4× wider.

## 2. The schedule on MH-1

MH-1 (`mh1-r2scan`) is a two-interaction `ScaleShiftMACE` with these irreps (from [opt-T08-linear-audit.md](opt-T08-linear-audit.md)):

- node features 512x0e into interaction 0;
- **512x0e+512x1o (2,048 floats) between the interactions**;
- 512x0e after interaction 1.

The energy is

    E = Σ_i E0(z_i) + Σ_i s·(pair_i + R0(h1_i) + R1(h2_i)) + N·shift,

with `h1 = P0(I0(…))`, `h2 = P1(I1(h1, …), sc1(h1))`, readouts R0 (linear) and R1 (non-linear), and s the scale. The
model depends on the positions only through the edge vectors `v_e = x_r − x_s + shift_e`: radial basis, spherical
harmonics, envelope and the pair term.

### What a tile is

A tile T is:

- a set of receiver atoms (at most `R` of them), kept in increasing global index;
- the edges whose receiver is in T (at most `Ec` of them). The edges are receiver-sorted already (the streaming
  convolution needs them sorted), so they are one contiguous slice per receiver. Within a receiver they stay in the
  neighbour list's order (sender groups for `segmented_first_sum`);
- the halo: the senders of those edges outside T (at most `H` of them).

The tile is run as a small graph whose nodes are `[T, halo]` (local indices 0..|T|+|H|−1). Edges keep their receiver
local (0..|T|−1) and their sender local into the halo. This is exactly MACE's LAMMPS ghost layout. Pass the
interaction `lammps_natoms=(|T|, |H|)` and `first_layer=True`, so that `handle_lammps` is the identity. Then
`truncate_ghosts` cuts the message, the density and `sc` to the |T| receivers before `gated_rows`, the product and the
readout. So the rewritten interaction modules run unchanged on a tile. The node-local work on halo rows is only:

- `linear_up` (needed: it builds the halo's sender features);
- `skip_tp` (wasted on halo rows; small);
- interaction 0's species reassociation GEMM (per node; small).

### The three sweeps

| Sweep | Per tile, in this order | Global writes |
| --- | --- | --- |
| **1. Interaction 0 forward, no autograd** | `v_e` of the tile's edges; spherical harmonics, radial table/embedding; I0 (species-only senders through `FirstInteractionReassociation`: the halo needs its species only, no features); P0; R0 and the pair term on the |T| receivers | `h1[T]` (float32, 8 kB per atom); the receivers' interaction-0 node energies |
| **2. Interaction 1 forward and backward** | leaf `h1_loc = h1[[T, halo]]` and leaf `v_e`; I1 (linear_up on T ∪ halo, streaming convolution on the tile's edges); P1; R1 on T; backward of `s·Σ_T R1` to `h1_loc` and `v_e` | `ḡ[[T, halo]] += ∂E/∂h1_loc` (scatter-add: a halo atom belongs to other tiles too); edge forces of the tile (§4); the receivers' interaction-1 node energies |
| **3. Interaction 0 replay and backward** | leaf `v_e`; I0, P0 as in sweep 1, with autograd; backward of `s·Σ_T (R0 + pair) + ⟨ḡ[T], h1_T⟩` to `v_e` | edge forces of the tile; nothing else |

**Why sweep 3 is complete.**

- Interaction 0's senders are species-only: the rewritten I0 reads the sender species' row of a per-species table,
  not the sender's features. So `h1_i` depends on the positions only through the edges into `i`, and those edges are
  all in i's tile.
- After sweep 2, `ḡ` holds the full `∂E/∂h1` of every atom, because every tile whose halo or receivers contain the
  atom has added its part.
- Sweep 3's surrogate `⟨ḡ, h1⟩` therefore gives every remaining position derivative: the chain rule through h1,
  plus R0 and the pair term.

**The energy** is assembled from sweeps 1 and 2. The node energies are reduced into float64, as `fastmace.model.energy`
does: E0 from the exact float64 table, the interaction terms in float32 per node, summed in float64.

**Exactness.**

- **Equal to the stock call except for rounding.** Every operation is the stock one on a row subset. The tile's
  GEMMs see fewer rows, and the scatter order of the edge forces changes. Those are float32 rounding differences,
  like CAP-T08's chunks (3 × 10⁻⁶ eV/Å).
- **Not bitwise.** Not even against the eager whole-system call.
- **Sweep 3 replays sweep 1's h1.** With deterministic kernels the replay is bitwise sweep 1's h1. With atomics it is
  equal to rounding, and ḡ was formed against sweep 1's h1, so the mismatch is second order.

**Other models.**

- **A first interaction whose senders carry features**, without the reassociation: interaction 0 would need halo
  sender features, which are species-only too (`node_embedding` of the one-hot). It can be recomputed per tile.
- **Three or more interactions** need one more global H and adjoint per extra boundary, and one sweep per layer:
  forward 0..L−1 tiled, then backward L−1..0, each replaying its layer.
- This prototype is MH-1 and two-interaction `ScaleShiftMACE` only.

## 3. What stays global, and the tile workspace

### Global, per atom (float32 model, float64 positions)

| Tensor | Bytes per atom | Note |
| --- | ---: | --- |
| `h1` (512x0e+512x1o) | 8,192 | written in sweep 1, read in sweeps 2 and 3 |
| `ḡ = ∂E/∂h1` | 8,192 | zeroed, accumulated in sweep 2, read in sweep 3 |
| node energies (two float32 parts, float64 sum) | 16 | |
| forces (float32 accumulator, float64 out) | 36 | |
| positions, species, one-hot `node_attrs` (MH-1: 89 columns), E0, batch | 400 | already held today |
| neighbour list: skin candidates, compacted `edge_index`, shifts and the screen (CAP-T09) | about 3,200 | about 27 edges per atom at 6 Å, plus the skin; already held today |
| **total** | **about 20 kB** | against about 240 kB per atom in the force call today (ledger) |

The MD state comes on top in a simulation: velocities, the Langevin noise batch, now capped (CAP-T05), and the
integrator buffers. It is outside the force call and the same for both paths.

### The tile workspace

The workspace depends on the tile caps (R, H, Ec) only, never on N:

- one tile's sweep-2 autograd graph (interaction 1, its product and readout), about 200-300 kB per receiver in
  eager. That is the CAP-T08 inventory per atom, on R atoms instead of N;
- its per-edge part: radial basis, harmonics, vectors, about 400 B per edge with `recompute_conv_weights`, or
  5.5 kB per edge without;
- the halo rows' `linear_up`/`skip_tp` (about 10 kB per halo atom);
- the tile's index arrays: O(Ec + H) ints.

Halo size: a compact tile of R receivers at density ρ has a halo of about `ρ·((L+2rc)³ − L³)` atoms, with
`L = (R/ρ)^(1/3)`. For NbCl5 (ρ ≈ 0.037 Å⁻³, rc = 6 Å) and R = 8,192, that is about 0.7 R. So tiles must be
spatially compact.

**Tile formation, general for any cell.** Bin the atoms by their fractional coordinates on a k₁×k₂×k₃ grid, where
the kᵢ are chosen from the cell's lengths so that a block holds about R atoms. Then split any block above R or Ec
greedily. That is a rule on the cell geometry and the caps, not a per-structure table. The atom order of the
structure is never changed: a tile is a sorted index list, not an index range.

**The cap.** The prototype measures the workspace per receiver and per edge (§6) and fixes R from a workspace
budget: `R = budget / (a_node + ρ_e·b_edge + halo_ratio·c_halo)`. A production planner would take a, b and c from the
memory model's declared terms (rule 12's per-record terms, decision 67), not from a measurement table.

**The capacity bound.** On 16 GB at about 20 kB per atom global plus a 2-3 GiB workspace and 1.2 GiB outside the
allocator, the ceiling is about 550-600k MH-1 atoms. Allow the neighbour-list build's transients and the MD state on
top. Symmetrix-XL's 7 kB per atom for OMAT-0 (H₁ of 128x0e+128x1o, 2 kB) is consistent with the same accounting.

## 4. Forces and stress

**Forces.** In sweeps 2 and 3 the leaf is the tile's edge vectors `v_e` (`[Ec, 3]`), not the positions. That makes
the backward local: `g_e = ∂E/∂v_e` for the tile's edges, then

    F[r(e)] −= g_e,   F[s(e)] += g_e

by `index_add_` into the global float32 force buffer. Senders in other tiles, the halo, get their share by the same
scatter. Each edge belongs to exactly one tile (its receiver's), so every edge gradient is added once per sweep.
These are MACE's edge forces (`compute_edge_forces`), assembled tile by tile.

**Stress.** With strain, `v_e` is used as `v_e (I + sym(ε))` at ε = 0. That is the same linear map that MACE's
`get_symmetric_displacement` applies to positions and shifts, so it is equal in exact arithmetic. Then

    W = −Σ_e v_e ⊗ g_e   (symmetrised),   stress = W / V   (MACE's sign: σ = −(∂E/∂ε)/V),

accumulated per tile into a 3×3 float64 accumulator. Rounding: MACE's virial goes through the per-atom positions
(large absolute values that cancel). The edge form sums `v ⊗ g` over short vectors, so it is at least as accurate.

**Determinism.** `index_add_` on CUDA is atomic, so its order is not fixed. In a `reproducible=True` plan the force
scatter must instead be a sorted segment reduction: receivers are already sorted, and senders need a per-tile sort.
That is an extra cost, needed in that mode only.

## 5. CUDA graphs

The prototype is eager: it shows the memory curve and the accuracy first. For the graph tier, two options:

**(a) One graph per sweep body, replayed T times (recommended).**

- **Padded shapes.** Each tile is padded to fixed (R, H, Ec): padding receivers have no edges, padding halo rows are
  zero, padding edges point at a padding node beyond the cutoff.
- **Exact padding.** Padding edges already contribute exactly zero (`padding_exact`, the streaming kernels' edge
  count). Here the count is per tile, in a device buffer the body reads.
- **Tile descriptors.** The body reads the tile's receiver list, halo list and edge offsets from device buffers,
  indexed by a device tile counter, and advances the counter itself. So the host replays the three bodies T, T and T
  times with no copies in between.
- **Memory stays flat.** Every replay reuses one pool. Capture memory is one body's workspace, not T of them, and a
  graph per sweep is three graphs, whatever N is.
- **Rebuilds.** Tile descriptors are recomputed on the device after each neighbour-list compaction, as the edge
  count is today. If a tile exceeds its cap after a rebuild, that is an overflow: re-plan the tiles on the host and
  re-capture, like today's edge-bucket overflow.

**(b) One graph for the whole loop** (T bodies captured in sequence).

- Fewer host launches.
- But the graph's node count, its capture time and the executable's memory grow with T. With T ≈ 70 tiles at 560k
  atoms and about 300 kernels per body, that is about 60k nodes per sweep.
- Not recommended. The host-replay cost in (a) is about 3T graph launches per step, about 2 ms at T = 70, against a
  step of several seconds at that size.

**Compiled bodies.** Inductor can compile the per-tile body at its padded shape: one shape per sweep, so no dynamic
shapes. The body's autograd uses `torch.autograd.grad` on a leaf, which today's split-forces path already handles
outside the compiled frame.

## 6. How it plugs into the planner

**A separate execution mode, not a rewrite.**

- **What changes.** Tiling changes the call schedule: three sweeps, global buffers and a scatter. The model's
  arithmetic stays the same. The rewrite records stay as they are and are applied to the model inside each tile.
- **The plan.** `EngineConfig` would gain `execution: "whole" | "tiled"` and a `TileConfig(receivers, edges, halo)`,
  resolved by the planner. `MACEForceField` keeps its interface (`__call__`, `compact`, `rebuild`), and a
  `TiledForceField` (the prototype's `fastmace.model.tiled`) implements the same `ForceField` protocol. Integrators
  see only a `ForceField`, as SPEC §5 asks.
- **Rule** (proposed, needs the owner's decision under SPEC §18; not added to SPEC here):
  - after rule 12 and decision 67's memory rewrites, including recompute_node_segments once merged, the plan still
    does not fit the free memory;
  - then the planner chooses `execution="tiled"`, with the tile caps from the memory model's per-atom and per-edge
    terms and the free memory: `R = (free − global(N) − margin) / workspace_per_receiver`;
  - no per-structure or per-GPU table. As with decision 67, it is off whenever the whole-system plan fits.
- **The owner's choice to make:** automatic, as in the rule above, or explicit opt-in only (Symmetrix-XL: "the
  execution planner considers the tiled plan only when it is explicitly enabled"). The speed cost (§7) decides which
  is acceptable.
- **Energy-only calls** (MC, `energy_only`): sweeps 1 and 2 forward only, without autograd. The workspace is smaller.

## 7. Risks

1. **Halo overhead.**
   - **The work.** At R = 8,192 the halo is about 0.7 R. The halo rows cost linear_up and skip_tp work, and the
     wasted skip_tp rows can be dropped later by slicing before `skip_tp` in a tile-aware forward.
   - **Atom order.** Tiles that are not spatially compact (contiguous index ranges in a badly ordered structure) would
     blow up the halo, hence the spatial binning.
2. **Speed.**
   - **The extra work.** Sweep 3 replays interaction 0's forward, and the tiles run their GEMMs on fewer rows.
   - **The eager prototype** will be much slower than the compiled graph path. It measures memory and accuracy, not
     speed.
   - **A production version needs** compiled per-tile bodies and graph replay (§5). Expected cost: one more
     interaction-0 forward, about 10-20 % of a force call, plus the halo work.
3. **Rewrites that assume the whole graph.**
   - the streaming edge count (bound per tile here);
   - the reassociated first interaction's species table (bound once);
   - `segmented_first_sum`'s sender groups (order kept within receivers);
   - `padding_exact` (no padding in the eager prototype);
   - self-checks run on the whole model as today, before any tiling.
4. **Non-determinism** of the atomic force scatter (§4). Reproducible plans need a sorted reduction.
5. **The neighbour list stays global**, at about 3 kB per atom, and its build has its own transients (CAP-T09). At
   500k+ atoms the build, not the force call, may set the limit. The prototype will show which.
6. **Rounding class, not bitwise.** The gate (ratio rule against float64) must pass. Bitwise A/B against today's path
   is impossible.
7. **Scope.** MH-1 and two-interaction ScaleShiftMACE only. Out of scope:
   - PolarMACE (k-space, global terms);
   - D3, which has its own list and could stay global;
   - OMOL-0's joint embedding (per-graph, so cheap to add);
   - variable-cell NPT. The stress is in, but cell updates trigger tile re-planning.
8. **CUDA-graph capture with tile descriptors read on the device** is new code in the engine. Until it exists, the
   tiled path is a compiled-tier or eager-tier mode, with host-side tile planning per neighbour rebuild.

## 8. Prototype plan

These are steps 2 and 3 of the brief.

1. `fastmace/model/tiled.py`: `TiledForceField(field, receivers=R)` wraps an eager `MACEForceField` built from the
   evaluation plan (`compile_mode="none"`). Steps:
   - plan tiles on the device after each `rebuild`/`compact`;
   - run the three sweeps;
   - return `ForceOutput(energy, forces, stress)`.
2. `tools/bench/tiled_capacity.py`:
   - **falsifier:** peak allocated memory of one force call at fixed R, for NbCl5 9,216, 21,600, 51,840 atoms and
     beyond, until it fails;
   - **comparison** against the default evaluation plan (compiled), where it fits: energy, forces and stress.
   - **Stop rule:** if the peak above the global buffers grows by more than 20 % from 9,216 to 51,840 atoms, the
     workspace is not flat and the design fails.
3. Gate: `tools/force_stress_gate.py` with the tiled force field as the production path, on NaCl, NbCl5 and NaNbO3
   (3×3×3, rattled and strained).
4. Only if 2 and 3 hold: the largest NbCl5 and NaCl cells on murphy's RTX 4000 Ada (20 GB), and on the RTX 5060 Ti if
   a reservation is free; step time against main at sizes both run; a short MD check with neighbour-list rebuilds.
