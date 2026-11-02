# Tiled path above 100M atoms: wavefront schedule and device trims (design note, not built)

Status: a proposal for the owner (2 November 2026). The owner earlier set the tiled capacity work to stop at about
70M atoms and move to speed. Nothing here is built.

## Where the memory goes today

MACE-OMAT-0 medium, SrTiO3 at 64.9M atoms; ledger from `tiled_capacity.py --ledger`.

**Host, per atom**

| Item | Bytes per atom | Share |
| --- | ---: | ---: |
| x, interaction 1's input (float32 × 512; later its adjoint dE/dx, aliased) | 2,048 | 57 % |
| sc (float32 × 128; later dE/dsc) | 512 | 14 % |
| Neighbour lists: the kept lists or the per-call cache | 371-491 | 10-14 % |
| The released one-hot (gone with the trims of this branch) | 356 | 10 % |
| The rest | about 130 | 4 % |

- **The host limit:** leto (250 GiB) keeps 20 GB free, which caps the run near 65M atoms. With this branch's trims,
  about 75-85M.

**GPU, per atom**

| Item | Bytes per atom |
| --- | ---: |
| Whole-system arrays (two copies of the positions, the forces, duplicated int64 species and group indices, heads, E0, charges) | 152 |
| A whole-system-sized transient during the call, not yet identified | about 61 (compiled) / 47 (eager) |

- **Tile workspace:** about 0.5 MB per receiver.
- **The GPU bound with today's code** (20 GB RTX 4000 Ada): about 87M atoms at R = 4,096, and about 94M at R = 1,024.
  So above about 90M the GPU binds before the host does.

## Lever 4: the wavefront schedule (host)

**Today** the call runs three sweeps, each over all tiles in the same order:
1. Interaction 0 forward: x and sc.
2. Interaction 1 forward and backward: dE/dx and dE/dsc.
3. Interaction 0 again, with autograd.

So x and sc (and then their adjoints) are live for every atom at once.

**Proposal:** order the tiles along one axis in layers (slabs of the tile grid), and interleave the sweeps. Sweep 1
runs one tile layer ahead of sweep 2, and sweep 3 runs one layer behind it.
- **Why one layer is enough:**
  - A tile's halo reaches only the adjacent tiles. The halo is at most the 6 Å cutoff, and tiles are about 46 Å on
    a side at R = 8,192.
  - So sweep 2 on layer k needs only the x rows that sweep 1 wrote for layers k-1 to k+1.
  - An atom's dE/dx is final once sweep 2 has passed the last tile that holds it, at most layer k+1. Sweep 3 on layer
    k-1 then reads only final adjoints.
- **Memory:** x, sc and their adjoints live only for the atoms of about three layers. The existing alias window
  (which already keeps the open atoms' dE/dx) generalises to a row pool: a row is taken at sweep 1's write and
  returned after sweep 3's read.
  - The band fraction is 3 / (layers in the grid): 15 % at 20 layers. It falls as N grows, since the grid grows as
    N^(1/3).
  - Host per atom: about 2,560 × 0.15 + (lists 0-490) + about 130, so 0.5-1.0 kB.
- **Bound on leto (240 GB usable):** about 240-480M atoms by the host. Above about 90M the GPU binds first unless
  lever 5 comes with it.

**Exactness**
- **Per atom, x, sc and their adjoints are bitwise today's.** Each sweep still visits the tiles in the same order.
  Every atom's x and sc are written once, and its adjoint sums are added in the same tile order (the window does
  that today).
- **The forces and the strain gradient are summed in a different order.**
  - Today: all of sweep 2's edge forces, then all of sweep 3's.
  - Interleaved: per layer, both.
- **Bitwise is still reachable by giving each sweep its own accumulator:** one float32 [N, 3] force buffer and one
  float64 3×3 strain buffer per sweep, summed at the end.
  - That costs 12 B/atom more on the device.
  - The new order then differs from today's call at float32 rounding only.
  - Today's GPU call is not order-deterministic anyway: the force scatter is an atomic `index_add_`.
- The class is the tiled call's present one, "equal to the whole-system call up to float32 rounding". The CPU tests
  would compare against a reference with the same two accumulators, bitwise.

**Cost estimate**
- **No extra FLOPs:** every tile runs each sweep once, as now.
- **Compiled frames:**
  - Each sweep is already one compiled frame per padded slot. Interleaving alternates three resident frames instead
    of running one frame thousands of times.
  - Frame dispatch is about 0.1 ms. A tile-sweep at 64.9M takes about 58 ms (2,097 s over 3 × 12,000), so switching
    costs well under 1 %.
  - No recompilation: the shapes are the slots' shapes as now.
- **Host overlap:** the host writes of one sweep would overlap the next sweep's tile instead of the same sweep's.
  This is the part to measure. The pipelined host writes rely on the order of pending writes, which stays per
  sweep.
- **Where it runs worse:** small systems, where the band is most of the system. There the schedule saves nothing,
  and the plain three sweeps should stay; a switch at about 8 layers or more.
- **Work:** a new schedule in `TiledForceField._call` and the row pool in `_State`. The tile lists, slots and
  compiled frames are unchanged. Tests: the CPU bitwise comparison against the two-accumulator reference, eager
  and padded, with stress.

## Lever 5: device trims (GPU)

Every whole-system device array the call does not need, or holds twice, goes.

| Today | Bytes per atom | After |
| --- | ---: | --- |
| float32 positions in `field.positions` and `data["positions"]` beside the float64 input positions | 12 + 24 | the float64 positions only (tiles take their rows) |
| float32 forces: the output, and `data["forces"]` | 24 | one output buffer |
| int64 `atomic_numbers`, `_species_index`, `tiled._species`, `_sender_groups` | 32 | one int32 species index (4) |
| int64 `heads` and `arange` [N+1] | 16 | per-tile ranges, or int32 |
| float64 `fastmace_node_e0` and `charges` | 16 | E0 looked up per tile from the species index; charges only for polar models |
| float32 `charges`, `density_coefficients`, `_ones` | about 12 | per tile |
| The transient (about 61 B/atom compiled) | 61 | find it first (a memory snapshot of one call at 10M); likely a whole-system float32 or float64 copy of positions or forces |

- **After the trims:** about 45 B/atom resident (float64 positions 24, float32 forces 12, int32 species 4, float32
  node energies 4), plus the tile workspace.
- **GPU bound on the 20 GB Ada:** about (20.9 − 1.2 − 2) GB / 45 B, about 400M atoms. The host binds again.
- **Exactness:** bitwise. The same values are read from fewer copies, and E0 per tile is the same table lookup.
- **Step-time cost:** none expected. The per-tile lookups are gathers of a few thousand rows.
- **Further on:** the positions and forces can live on the host too, each tile copying its rows as x and sc do
  today. Then the device holds no per-atom state, at about +5 % call time from the extra copies, to be measured.

## Bounds together (leto: 250 GiB host, 20 GB RTX 4000 Ada)

| Stage | Host per atom | Host bound | GPU bound | Limit |
| --- | ---: | ---: | ---: | ---: |
| Today (main) | 3.7 kB | 65M | about 90M | 65M |
| This branch (levers 1-3) | 3.0-3.4 kB | about 75-85M | about 90M | about 75-85M |
| + lever 5 | same | same | about 400M | about 75-85M |
| + lever 4 | 0.5-1.0 kB | 240-480M | about 90M | about 90M |
| + levers 4 and 5 | 0.5-1.0 kB | 240-480M | about 400M | **about 240-400M** |

- **On an 80 GB A100 with levers 4 and 5:** the GPU bound is about 1.6G atoms. The host binds at 0.5-1.0 kB per atom
  (about 1 TB for 1G).
- **Time per call:** at 32 µs per atom (compiled, measured at 64.9M on the Ada: 2,097 s), 300M atoms take about
  2.7 hours per force call. Capacity at this scale is for single-point calls or very short runs, not MD.
