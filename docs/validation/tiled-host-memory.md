# Tiled capacity path: host memory at the call

After [tiled-species-index.md](tiled-species-index.md), host memory set the tiled path's limit on the
workstations (250 GB each). The conductor's lever (1), building MACE's input batch without its neighbour list,
was to lower it. **Measuring first showed that the host peak is the call's, not the build's**, and that a third
of it was PyTorch's pinned-memory rounding. This change removes that rounding. Lever (1) is not needed for the
host ceiling now.

## The measurement

`tools/bench/tiled_capacity.py` now records the process's host memory (`VmRSS` now, `VmHWM` peak so far,
pinned pages included) at every phase and call.

**NbCl5 9,261,000 atoms** (murphy; tile lists, the state on the host, the species index):

| Point | Pinned state (before) | Pageable state (after) |
| --- | ---: | ---: |
| user and plan | 2.26 GiB | 2.26 GiB |
| build, peak so far (transient, MACE's batch with its neighbour list included) | 28.19 GiB (3.27 kB/atom) | 28.14 GiB (3.26 kB/atom) |
| after the build | 6.64 GiB | 6.25 GiB |
| **the call** | **79.52 GiB (9.22 kB/atom)** | **48.98 GiB (5.68 kB/atom)** |
| of it, the tiled state (x, sc and their adjoints, float32) | 41.95 GiB | 41.95 GiB |

**The cause.** With the state pinned, the call held 31 GiB more than the state and the field:

- PyTorch's pinned host allocator rounds every block up to a power of two, and pinned pages are all resident.
- x and dE/dx (19.0 GB each at 9.26M atoms) become 32 GiB each; sc and dE/dsc (3.6 GB each) become 4 GiB each.
- That is 72 GiB for 42 GiB of state. The waste depends on N, up to almost 2×.

## What changed

`fastmace/model/tiled.py`, `_State.allocate`: the host state is plain (pageable) CPU memory of the exact size.

- The tiles never copy from these tensors directly. A read gathers the tile's rows into a new tensor
  (`index_select`) before the copy to the device; a write copies the results to the host before the
  assignment or `index_add_`.
- So pinning bought nothing.
- The values and every operation are unchanged: no numerical change.

**Speed, pinned against pageable** (`results/tiled-host-batch-20261101/ab_pinned.py`; murphy, one process, the
same field twice, 6 alternating blocks, the order flipped every block):

| NbCl5 atoms | Pinned | Pageable | Pageable / pinned | max \|ΔF\| |
| ---: | ---: | ---: | ---: | ---: |
| 51,840 | 1.307 s | 1.301 s | 0.995 | 9.0e-7 eV/Å |
| 884,736 | 24.09 s | 23.88 s | 0.991 | 1.1e-6 eV/Å |

Pageable memory costs nothing; it is half a percent to one percent faster.

## Capacity

The run setup:

- murphy and leto, RTX 4000 Ada 20 GB, 250 GB host; `mace_cueq_oeq`;
- MH-1 float32 hybrid, eager tiles of R = 4,096, tile lists, the state on the host, the species index;
- FastMACE 4759e00e;
- runs `results/tiled-host-batch-20261101/scan.sh`.

The scan starts a size only if its predicted host peak, 5.8 kB per atom, fits the available memory less 20 GB.
Every point passed the exact capacity check: the initial edges equal ASE's count, and the largest force is
nonzero.

| Structure | Atoms | Host | GPU build peak | GPU call total peak | Host peak (the call) | Host build peak | Call |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| NbCl5 | 37,933,056 | murphy | 16.54 GiB | 9.94 GiB | 190.0 GiB (5.38 kB/atom) | 108.5 GiB | 952 s |
| NbCl5 | 40,001,688 | murphy | 17.43 GiB | 10.48 GiB | 199.8 GiB (5.36 kB/atom) | 114.2 GiB | 1,108 s |
| NaCl | 37,933,056 | leto | 16.53 GiB | 9.93 GiB | 190.1 GiB (5.38 kB/atom) | 124.7 GiB | 1,031 s |

**The largest valid structure on a 20 GB GPU with 250 GB of host memory rises from 32.16M to 40.0M atoms.**

- 40.0M is about the largest size the host guard allows on these workstations.
- The GPU build peak, 17.43 GiB, is 2 GiB under the card's 19.5 GiB.

## What limits next

Both limits are now close:

- **The GPU build peak, 0.45 kB per atom.** Mostly the one device copy of the one-hot species, which the
  construction's species check and rewrites read before it is released. That is lever (2), the build reading
  the one-hot on the host; it is next.
- **Host memory, 5.4 kB per atom at the call.** The tiled state is 4.86 kB of it (float32, exact; the owner's
  rule).
  - The build's host peak is lower, 2.9-3.3 kB per atom, MACE's neighbour list included. So lever (1) would not
    lower the ceiling.
  - Beyond about 42M atoms, a workstation with more host memory is needed.
