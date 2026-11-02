# Tiled path above 78M atoms: wavefront schedule and device trims

Owner decision (2 November 2026): "Go ahead and try to go as big as possible now". This replaces the earlier stop
at about 70M atoms. The design is `docs/design/tiled-wavefront.md`; this report records what was built and
measured.

**Setup**
- GPUs: NVIDIA RTX 4000 Ada (20 GB) on leto and murphy, each with 251 GiB of host memory; the `mace_cueq_oeq` NAS
  environment.
- Model: `omat0-medium` (MACE-OMAT-0 medium), hybrid backend, float32 forces.
- Structure: cubic SrTiO3, perfect crystal (the scan's exact edge check applies).
- Tool: `tools/bench/tiled_capacity.py` with `--tile-list --host-state --alias-state --species-index`, mode
  `auto` (first call eager, second compiled when it fits). Runs under `/mnt/homes/zwingel/fm-tiled/runs/`.

## 1. What changed

**Host trims (levers 1-3, exact).** The released one-hot keeps only its shape; small index and position arrays
are trimmed; the kept neighbour lists are freed at the host floor. Validated at 78.1M SrTiO3 atoms on murphy
(below).

**Device trims (lever 5, exact unless noted).**
- After the species index replaces the one-hot, every per-atom buffer that only the whole-system call reads is a
  shape-only (meta) tensor; `batch` is a zero-stride view; E0 is looked up from the species index where it is
  read (`MACEForceField.release_whole_system`, `node_e0`).
- One shared head (a zero-stride view) instead of an int64 `[N + 1]` heads array, and an `arange` of the largest
  receiver cap instead of `[N + 1]`.
- The tile neighbour bins in the narrowest integer dtypes that hold their values (int16 bins and cell images,
  int32 order).
- int32 species columns and sender groups.
- The call hands out its force buffer and allocates a new one next call, instead of keeping one and returning a
  copy.
- `plan_tiles` computes the fractional coordinates per chunk, element-wise, with int32 block keys. Before, it
  formed the whole-system float64 coordinates with a float64 GEMM. Atoms on a block boundary may move to the
  neighbouring tile; that changes the summation order only (the tiled call's rounding class).

**Wavefront schedule (lever 4, `TileConfig.wavefront`).**
- Sweep 2 visits the tiles in their order. Sweep 1 runs on a tile just before the first sweep-2 tile that reads
  its x and sc; sweep 3 runs on a tile just after the last sweep-2 tile that adds to its receivers' dE/dx
  (`wavefront_schedule`).
- x, sc and their aliased adjoints live in a pool of rows of the atoms between their sweeps 1 and 3, not in
  `[N, width]` tensors. Each tile's lists are dropped after its sweep 3.
- A search pass before the sweeps records each atom's first and last tile and the tiles each tile reads. It keeps
  no lists, so each tile searches once more than without the schedule.
- **Exactness:** each atom's x, sc, dE/dx and dE/dsc are bitwise those of the plain sweeps (the same writes,
  dE/dx summed in the same tile order), so the node energies and the energy are bitwise. The forces and the
  stress are the same sums in another order (float32 rounding; the GPU scatter is atomic anyway).
- **"auto"** turns it on from 12 layers of the tile grid along its slowest axis and 2^20 atoms.
- **CPU tests** (leto, GPU hidden): the schedule's invariants on a periodic ring; the wavefront call against the
  plain three sweeps, eager and padded, immediate and deferred host writes, on 864 atoms in 10 grid layers:
  energy bitwise, forces and stress within 1e-13 (float64 tiny model).

## 2. Results

(filled in below as the runs finish)

**98.4M atoms on leto (SrTiO3 270×270×270, source `src-wave-0a72f3ff`):** valid, a new capacity record (the
earlier one was 78.1M).
- **Edges:** 7,282,710,000 edges in the cutoff, exactly the expected 74 per atom.
- **Forces:** largest force component 1.3 meV/Å on the perfect crystal.
- **Calls:**

| Call | R | Tiles | Time | GPU peak | Host peak | Wavefront pool |
| --- | --- | --- | --- | --- | --- | --- |
| eager | 4,096 | 32,894 | 3,232 s | 12.0 GiB | 92.4 GiB | 14.2 % of the atoms |
| compiled | 8,192 | 17,854 | 2,277 s | 13.9 GiB | 127.3 GiB | 18.2 % of the atoms |

- **Run:** `results/tiled-omat-20261102/runs/leto-w-omat-scan-SrTiO3-270x270x270/`.

122M on murphy (290×290×290): the eager call passed (4,054 s, GPU peak 14.4 GiB, host 110 GiB). The compiled
call is still running.
