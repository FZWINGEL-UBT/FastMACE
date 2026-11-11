# One billion atoms on one GPU (plan, 11 November 2026)

Owner request (11 November 2026): plan a run that fits 10^9 atoms on one GPU; carry it out up to the go/no-go at
397.5M atoms (step 3); no whole Alex node for now. Colab (A100 40 GB, RTX PRO 6000 "G4" 96 GB) may be used; Modal
credits are short. Branch `fastmace/billion-atoms`, from `fastmace/idea11-frame` (`fc8d28f1`), so the tile-local
frame (idea 11) is in.

## Target

SrTiO3 585³ = 1,000,951,625 atoms, about 74.1 billion edges, OMAT-0 medium, float32 calls, the capacity record's
settings (`tiled_capacity.py --mode auto --tile-list --host-state --alias-state --species-index --wavefront on
--calls 2 --state-dtype auto`). The box is 2,284 Å, so the tile-local frame is on by size.

## Where today's record stops

The record is 397,535,000 atoms on one A100-SXM4-80GB (Alex 4973640, `docs/validation/tiled/tiled-host-state-dtype.md`,
`results/tiled-capacity-20261109/alex/cap-4973640/`): GPU peak 40.6 GiB, host peak 229.1 GB of the job's 231.5 GB
cgroup, compiled call 3,913 s (9.84 µs/atom).

**The device peak is before the tiles** (the record's `calls`, compiled call, bytes per atom of 397,535,000):

| Point | GB | B/atom |
| --- | ---: | ---: |
| after the build | 11.61 | 29.2 |
| after the neighbour setup (`_prepare_one_off`) | 21.15 | 53.2 |
| allocated before the call | 22.75 | 57.2 |
| peak before the first tile (`peak_before_tiles`) | 40.17 | 101.0 |
| the call's peak (`total_peak`) | 43.58 | 109.6 |

So about 44 B/atom is a transient between the call's start and its first tile (the tile plan and what the call
sets up for the wavefront), and the tiles' own workspace is 3.4 GB. Without that transient, 10^9 atoms would need
about 57 + 4 = 61 GB, inside 80 GiB with no other change. The 320M point has the same shape (53.5 -> 100.9 B/atom
before the tiles).

| Resource | Per atom today | At 10^9 atoms | One A100-80 |
| --- | ---: | ---: | --- |
| GPU | 102-110 B (40.6 GiB at 397.5M; the Ada: 111.6 B at 179.7M, almost all per-atom arrays) | 110 GB + about 4 GB fixed and tile workspace | does not fit: 80 GiB allows about 75 B/atom with margin |
| Host | 576 B (float16 state, no kept lists), 790 B (float32, kept lists) | 0.6-0.8 TB | one GPU's share (231.5 GB) does not; a whole node (2 TB) does |
| Time | 9.84 µs (compiled), the eager first call 2.9x | build about 30 min, first call about 8 h, compiled call about 2.7 h | fits 24 h |

Other GPUs: the H200 (141 GB, Modal) would hold today's 110 B/atom, but Modal does not document its largest memory
request, GPU functions cannot be made non-preemptible, and credits are short. Colab G4 (96 GB) is closer than the
A100 on the device, but its VM has 176 GB of RAM. So the run is an A100-80 on a whole Alex node, after the device
falls to about 70 B/atom.

## Steps

### 0. Device census at 20M atoms (Colab A100)

Which device tensors are live at the call's peak, by owner and source line, B/atom. `tiled_capacity.py --ledger`
already names the persistent tensors after the build and after each call; `--memory-history` traces the build only.
Add the call's trace (the peak above the call's start, by source line) and run SrTiO3 160³ (20,480,000 atoms) with
the record's settings on a Colab A100 (A100 family rewrites; 40 GB is ample at 20M).

Output: a table of B/atom by owner (positions float32/float64, forces, species, maps, tiles, the plan's sort, kept
lists' device parts, tile workspace), which fixes step 1's list.

**Result (11 November 2026; Colab A100-SXM4-40GB, `94d8035b`, 3.91 CU, records `results/billion-atoms-20261111/census-colab/`).**
SrTiO3 160³ (20,480,000 atoms), the record's settings; the compiled call at R 16,384 took 312.6 s. Peaks per atom at
20M carry about 30-50 B/atom of size-independent tables (spline tables, slots, tile workspace); the per-atom part:

| Live at the peak before the first tile | Source | B/atom |
| --- | --- | ---: |
| the call's float64 input positions (also the tile-local frame's rows) | the caller (`_prepare_one_off`), `tiled._g.local_frame` | 24 |
| float32 positions | `field.positions` | 12 |
| the call's forces | `tiled.py:2837` (`_globals`) | 12 |
| species index, sender groups, the whole-system receiver order (int32 each) | `forces.py:778-779`, `:1025` | 12 |
| cell binning of the tile lists: integer cells, bins (int16 [N, 3] each), order, starts | `tile_list.py:86-104` | 19 |
| the plan's order (int64; the tiles are views of it) | `tiled.py:861` | 8 |
| `local_of` (int32) | `tiled.py:2321` | 4 |
| node energies | `_globals` | 4 |
| host-state staging on the device | `tiled.py:523`, `:614` | about 5 |

That is about 100 B/atom, matching the 397.5M record's 101 before the tiles (the sorts' temporaries come before
the peak counter's reset and stay below it). At 10^9 atoms the A100-80 leaves about 79 GB beside the tile workspace,
so step 1 has to cut at least 22 B/atom, better 30.

**Step 1's list from the census** (ranked by B/atom; each exact):
1. **Float64 positions on the host** (24): the caller passes pinned host float64 positions; each slot gets its tile's
   float64 rows gathered on the host and copied with the slot's other buffers (`_fill`), and `_slot_local_parts`
   reads those rows instead of `positions64[nodes]`; the cell binning reads host chunks. The kept lists' float64
   distance test (`_source`, 24 B/atom when the lists are kept) takes the tile's rows too.
2. **The plan's order in int32** (4) and the tiles as int32 views, widened per tile where a kernel needs int64.
3. **Species index and sender groups in uint8** (6) where the model has at most 255 columns and groups; the
   whole-system receiver order (4) is not built for the tiled call (its kernels run per tile).
4. **The bins' int16 [N, 3] atom bins** (6) recomputed per tile from the float32 positions and the cell instead of
   stored (exact: the same float64 arithmetic on the same values), if 1-3 leave too little room.

### 1. Device per-atom arrays on a diet (idea 8 of `ideas-20261111-tiled.md`)

Target: at most 70 B/atom at the peak (10^9 atoms: 70 GB plus fixed, about 10 GB left on 80 GiB). First the
transient before the first tile (about 44 B/atom); then the persistent arrays only if needed. Candidates, to be
confirmed by the census:
- the plan's order without a whole-system device sort (counting placement in chunks, int32): about 12 B/atom at
  the call's start, the Ada's 196.5M failure;
- tiles as int32 index tensors: 8 to 4 B/atom;
- whole-system float64 positions off the device: up to 24 B/atom (each tile brings its own float64 rows since
  idea 11);
- whatever else the census shows above about 4 B/atom.

Each change: exact (bitwise where the order is unchanged), tests in `tests/cpu` (run on leto/murphy), the 3x gate on
the A100 (Alex, one GPU) and a same-process A/B at 975,560 atoms (not slower, both orders). Each gets a ledger row.

**In (11 November 2026, `073f1f37`; CPU tests 211/211 on leeloo; Alex 4984899: gate, A/B, probe):**
- `TILE_HOST_ROWS` (24 B/atom): `_prepare_one_off(host=True)` keeps the call's float64 positions on the host, a view
  of the atoms' own array (no host copy either). The float32 copy, the tile lists' binning and the size rule's
  largest coordinate go a chunk at a time; each tile's distance test gathers its own rows on the host (`unique`
  of its receivers and candidate senders, then `searchsorted`), and the tile-local frame's float64 rows are gathered
  per padded slot in `_fill` (`rows64`, fixed addresses for the captured bodies; padding rows atom 0's, as on the
  device). The same float64 values, so bitwise the device-positions call (CPU test, eager and padded, far from the
  origin and strained; the tile lists bitwise the global list). `tiled_capacity.py --host-positions`,
  `tiled_ab.py --b-host-positions`.
- uint8 species index, sender groups and their tables after `release_whole_system` where every column and group is
  below 256 (6 B/atom); `node_e0` widens the index where it looks up.
- No whole-system BLOCK_ORDER receiver order in the tiled call (4 B/atom): its kernels run per tile on fewer rows than
  N, so they launched with their rows' identity already; `_bind_streaming_count(block_order=False)`.

Projected at 397.5M: 101 - 34 = 67 B/atom before the tiles, about 75.6 with the tile workspace (28.0 GiB, the
go/no-go's line); at 10^9 about 71 GB of the A100's 85.9. Held back: the plan's order in int32 (4 B/atom; the tiles
are int32 views that every consumer would widen, and the kept lists look tiles up by address) and the bins' atom
bins (6), if step 3 lands above the line.

### 2. 32-bit index audit above 716M atoms

At 715,827,883 atoms 3 x atoms passes 2^31: any kernel or view that indexes flattened [N, 3] arrays (positions,
forces, rows, the host state's float32 x) with int32 offsets breaks first there; at 10^9 atoms, [N] int32 maps still
fit (atom indices below 2^31). The 397.5M record never crossed it.
- Audit: Triton kernels' offsets (`tl.program_id * BLOCK + arange` on flattened per-atom arrays), `int32` casts of
  atom-times-3 counts, NumPy/torch views of [N, 3] on the host (the host state, the alias window, the kept lists'
  reference positions), and ASE's supercell.
- GPU test (Colab G4, 96 GB): the whole-system per-atom paths a tile touches (gather of rows, scatter of forces,
  the host state's write and read) on a fabricated 750M-atom positions array (9 GB float32), against a float64
  reference on the touched rows.

### 3. Go/no-go at 397.5M atoms (Alex, one A100-80)

The record's run again, with steps 1-2 in. Pass: GPU peak at most 28 GiB (about 75 B/atom), valid (74 edges per
atom, energy -7.98955 eV/atom within the float16 state's 0.54 meV/atom if the host forces float16), compiled call
not slower than 9.84 µs/atom by more than 3 %. Fail: stop and re-census. The host's 231.5 GB stays the limit here; the
heap-trim rerun (Alex 4981047) shows the room.

### 4. The 10^9 run (not reserved)

One whole Alex A100-80 node (2 TB host), SrTiO3 585³, the record's settings, `MALLOC_MMAP_THRESHOLD_=1048576`,
faulthandler, 24 h. About 14 h x 8 GPUs, about 110 GPU-hours. Needs the owner's go (a whole node). The 519M point's
segmentation fault before the build (exit 139 ten minutes in, the stack in ASE's `cell.py` `__array__`, so
while the supercell was made on the host; `tiled-host-state-dtype.md`) is unexplained; a 703M point (520³) first would
separate it from the 10^9 question, at a second node-day.

**Before step 4 (documented 11 November 2026, owner: "only document for now"):**

*Where the 397.5M record's 4 h 38 min went* (Alex 4973640, `results/tiled-capacity-20261109/alex/cap-4973640`; the
job had one CPU, `nproc 1`, where Alex now gives 16 per A100):

| Phase | 397.5M | B/atom or us/atom | Scaled to 10^9 |
| --- | ---: | ---: | ---: |
| supercell (`tiled_capacity.supercell`: ASE `repeat` + `wrap`), plan, checks | about 10 min | | about 25 min |
| force-field build (`phases.build`) | 660 s, host peak 111 GB | 1.66 us/atom | about 28 min |
| first call, eager (mode auto measures its memory) | 11,468 s | 28.8 us/atom | about 8 h |
| second call, compiled | 3,913 s | 9.84 us/atom | about 2.7 h |

1. **Our own supercell builder** (the ASE step; built on branch `fastmace/fast-setup`: `tiled_capacity.supercell`
   chunked, `supercell_ase` the reference, bitwise at small sizes in `tests/cpu/test_tiled_supercell.py`). `Atoms.repeat` loops in Python over every unit-cell copy (about 200M
   passes at 585^3, a slice write each), and `wrap` makes whole-system fractional temporaries (several 24 GB float64
   arrays at 10^9); the 519M segmentation fault came out of this step (ASE `cell.py` `__array__`). Replacement, about
   50 lines in `tools/bench/tiled_capacity.py` (or `fastmace.io` if it is to serve users): allocate the float64
   positions once, fill them a chunk of cell copies at a time by broadcasting (copy positions plus lattice offsets),
   no `wrap` (a perfect repeat is inside the box; rattle and strain chunked where asked), narrow atomic numbers, and
   wrap the arrays in an `ase.Atoms` without a copy (the force field takes one). Check: bitwise ASE's positions,
   numbers and cell at small and medium sizes; then a CPU-only dry run of this step at 520^3 and 585^3 (no GPU), which
   also answers the 519M fault.
2. **The eager first call** is the largest cost (about 8 of the run's 11 h at 10^9). Mode auto runs a whole call
   eager to measure the tile workspace before compiling. Correction (11 November 2026, from the records' own
   numbers): the tiles' workspace (``tiles_peak - loop_base``, what ``_choose_mode`` and ``_choose_receivers`` read)
   is set by the largest tile, not by the atom count: eager at R 8,192, 526,080,512 bytes at 149M, 320M and 397.5M
   (Alex 4973640), 503,742,464 at 20.48M with a smaller largest tile (Alex 4985398); compiled at R 16,384, 1.074 GB
   at 397.5M and 1.055 GB at 20.48M. (The earlier "1.8 GiB / 3.4 GB" were peaks above the call's start, which
   include the call's per-atom buffers.) So the decision can come from the run's own largest tiles:
   ``TILE_FIRST_CALL_PROBE`` (branch `fastmace/fast-setup`, opt-in) runs the measuring call's eager sweeps on the 8
   largest tiles only, decides the mode and the cap from them, and runs the call again in the chosen mode. A/B on
   Alex: the probe's decision and workspace against the whole call's at 58^3 and 160^3
   (`results/billion-atoms-20261111/jobs/alex_setup.sh`). Also: the go/no-go (Alex 4985889) has 16 CPUs where the
   record had one, so it re-measures the eager call's host share for free.
3. **The force-field build** (660 s, host peak 111 GB, about 279 B/atom at its peak): profile it at 20M (cProfile and
   the host peak by step) before changing anything; the one-hot and species tables built on the host at construction
   are the first suspects.

Together, 1 and 2 would take about 8.5 of the 10^9 run's 11.5 h (the supercell's 25 min and most of the eager call's
8 h); 2 is the less certain one (a few tiles must decide as a whole call does). Order: 1 (also the 519M fault's
answer), the 16-CPU re-measurement, then 2, then 3's profile.

## Feasibility of 10^9 atoms (owner's question, 11 November 2026)

Still possible on paper; step 3's measured peak decides it.
- **Device (the hard limit):** about 101 B/atom before the tiles (census), less step 1's 34 (host positions 24, uint8
  maps 6, no whole-system order 4): about 67 B/atom, so about 67 GB of per-atom arrays and 3-4 GB of tile workspace
  at 10^9, about 71 GB of the A100-80's 85.9 GB (about 15 GB left). Step 3 measures it at 397.5M (projected
  28.0 GiB). If it lands above, the held-back cuts remain: the plan's order in int32 (4 B/atom), the bins' atom
  bins (6).
- **32-bit indices:** audited; the one real overflow (the tile lists' sort key) is fixed and the 750M probe is
  bitwise on every check. The 10^9 run is still the first to cross 716M end to end, beyond the probe's per-tile paths.
- **Host:** about 0.6-0.8 TB at 10^9; one A100's share of an Alex node (231.5 GB) does not hold it, a whole node
  (2 TB) does. Needs the owner's go for a whole node (step 4).
- **Time:** about 30 min of build, about 8 h of eager first call, about 2.7 h per compiled call, plus 2-4 % for host
  positions: inside 24 h (and much less with "Before step 4" 1-2).
- **Risks:** the 519M segmentation fault in ASE's supercell (answered by our own builder and a CPU-only dry run);
  ASE's slow `repeat` loop and whole-system `wrap` temporaries; the large receiver caps (B3) are no risk here, as the
  host rule (`TILE_LARGE_CAP_HOST_BYTES`) keeps the cap at the host limit.

## Status

| Step | State | Record |
| --- | --- | --- |
| 0 | done: about 100 B/atom live before the first tile, 24 of it float64 positions | `census-colab/` (Colab A100, 3.91 CU) |
| 1 | in (34 B/atom): gate 3x 12/12 (Alex 4984899); host rows 1.043 / 1.021 at 58^3 (a capacity option, not a default) | `073f1f37`, `step1-alex-4984899/` |
| 2 | audit done (one fix: the tile lists' sort key, `KEY_LIMIT`); Alex 4984768 at 750M: the gathers bitwise (compiled and eager `_tile_vectors`, compiled `_slot_local_vectors`), untouched rows zero, the flat view's tail right; `index_add_` differed by 3e-5 at 1M as at 750M (the atomics' order with random values), again with integer values in 4984899: every check bitwise at 1M and 750M | `jobs/alex_overflow.sh`, `step1-alex-4984899/probe-*` |
| 3 | running: Alex 4985152 (one A100-80, 430^3 with `--host-positions`) | `jobs/alex_go.sh` |
| 4 | not reserved (owner, 11 November 2026) | |
