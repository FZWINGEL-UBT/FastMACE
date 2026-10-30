# Tiled capacity path: T2 (smaller global state) and T1 (one padded CUDA graph per sweep)

The owner's approved next steps T1 and T2 of [plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md),
through the conductor, after [capacity-tiled.md](capacity-tiled.md).

## T2: keep interaction 1's reads of h1, not h1

### What changed against the plan

**The plan's T2 would not lower the peak.**

- It was to free h1 after sweep 2, since sweep 3 recomputes interaction 0.
- But the peak is in sweep 2, where h1 (read for every tile's receivers and halo) and its adjoint (accumulated
  over the tiles) are both alive.
- The out-of-memory errors at 1.06M NbCl5 and 1.0M NaCl atoms were in allocating that adjoint at the start of
  sweep 2.

**What is done instead, exact as well.** Interaction 1 reads h1 only through two linear maps, `linear_up` and
`skip_tp`. With MH-1's default `fold_product_linear` both are `FoldedConsumer`s, and h1 is the contraction output
plus the one-hot species: about 2,100 columns.

- **Sweep 1** stores their outputs, `x = linear_up(h1)` and `sc = skip_tp(h1)`, together 608 columns.
- **Sweep 2** runs interaction 1 with its `linear_up` and `skip_tp` standing in for the stored values. It
  accumulates `dE/dx` over each tile and its halo, and writes `dE/dsc` of the receivers.
- **Sweep 3** recomputes h1 per tile and backpropagates `⟨dE/dx, linear_up(h1)⟩ + ⟨dE/dsc, skip_tp(h1)⟩` with interaction
  0's own energy.

The global state falls from about 17 kB to **4.75 kB per atom** (measured: x, sc and their adjoints). This is
`fastmace/model/tiled.py`; the CPU tests compare it against the whole-system call in float64, with the stock
model and with MH-1's rewrites.

### Accuracy

NbCl5 4×4×8, 9,216 atoms, rattled by up to 0.05 Å and strained by up to 2 %, R = 4,096:

| Against | ΔE (meV/atom) | max \|ΔF\| (eV/Å) | max \|Δσ\| (eV/Å³) |
| --- | ---: | ---: | ---: |
| the whole-system eager call | −2.5e-6 | 2.3e-5 | 5.6e-9 |
| the default compiled plan | −7.8e-7 | 2.3e-5 | 6.5e-9 |

That is the same as before T2: 2.3e-5 against the compiled plan
([capacity-tiled.md](capacity-tiled.md)).

### New limits on murphy (RTX 4000 Ada 20 GB)

**Tool:** `tiled_capacity.py point --calls 1`, R = 4,096, growing until a size fails
(`results/cap-tiled-t2-20261030/t2.sh`). Records: `results/cap-tiled-t2-20261030/runs/`.

| Cell | Atoms | Call peak (GiB) | Before (GiB) | **Total (GiB)** | Global state (GiB) | Neighbour build peak (GiB) | Call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NbCl5 16×16×48 | 884,736 | 5.57 | 3.65 | **9.22** (was 19.04) | 4.01 | 3.49 | 25.6 s |
| NbCl5 18×18×54 | 1,259,712 | 7.26 | 5.13 | 12.39 | 5.71 | 4.97 | 33.0 s |
| NbCl5 19×19×57 | 1,481,544 | 8.25 | 5.99 | 14.24 | 6.71 | 5.85 | 43.8 s |
| NbCl5 20×20×60 | 1,728,000 | 8.72 | 3.22 | 11.94 | 7.83 | 0.19 | 39.9 s |
| **NbCl5 21×21×63** | **2,000,376** | 10.05 | 3.68 | **13.73** | 9.06 | 0.22 | 40.2 s |
| NbCl5 22×22×66 | 2,299,968 | | | the neighbour build fails | | | |
| NaCl 50×50×50 | 1,000,000 | 6.25 | 5.53 | 11.79 | 4.53 | 6.50 | 29.0 s |
| **NaCl 54×54×54** | **1,259,712** | 7.48 | 6.91 | **14.39** | 5.71 | 8.19 | 32.5 s |
| NaCl 57×57×57 | 1,481,544 | | | out of memory in the neighbour list's compaction | | | |

- **NbCl5: from 884,736 to at least 2,000,376 atoms (2.3×), at 13.7 of 19.5 GiB.**
  - The force call no longer sets the limit; the neighbour list does.
  - At 2,299,968 atoms the GPU pair search fails at launch (`Triton Error [CUDA]: invalid argument`):
    `pair_search.screen_codes` launches a 2-D grid of `cdiv(N, BLOCK)` × `cdiv(N, BLOCK)` blocks. CUDA's y
    dimension stops at 65,535, which is 2.1M atoms with BLOCK = 32. A launch limit, not memory: a 1-D grid, or
    the y dimension split, would lift it.
  - Above 1.7M atoms the build's own peak fell to 0.2 GiB and the memory before the call to 3.2 GiB. The build path
    there is not looked at here.
- **NaCl: from 778,688 to 1,259,712 atoms (1.6×).** At 1,481,544 atoms the build runs out of memory in
  `SkinNeighbourList.compact`, which forms float64 vectors for every candidate pair: 8.2 GiB peak at 1.26M atoms.
  NaCl has 32 edges per atom within the cutoff, more candidates within the skin, and a denser cell than NbCl5.
- **Both next limits are the neighbour list**, which T3 already names as the next step ("the list build needs
  tiling too"). These two cases make it concrete:
  - the pair search's grid;
  - the compaction's whole-candidate float64 vectors.
- **Against main on the same GPU:** NbCl5 2.0M against 60,480-74,088 atoms (27-33×); NaCl 1.26M against
  64,000-74,088 (17-20×).

## T1: one padded CUDA graph per sweep

(in progress)
