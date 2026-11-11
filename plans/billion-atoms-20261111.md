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

## Status

| Step | State | Record |
| --- | --- | --- |
| 0 | planned | |
| 1 | planned | |
| 2 | planned | |
| 3 | planned | |
| 4 | not reserved (owner, 11 November 2026) | |
