# Tiled path: setup leftovers, the compiled MD limit, the A/B against c8cb1a7f, step times

This report covers the four items the conductor ordered after merging tiled-call-memory (85195b39).

**Setup.** MH-1 float32 hybrid (`mh1-r2scan`) with `mace_cueq_oeq`, on an NVIDIA RTX 4000 Ada (20 GB) on murphy
and leto. The tiled force field uses tile lists, the host state aliased, and the species index; the force field's
own list stays empty. Every job started only after an idle check, kept with each run.

## 1. The leftovers

**The tile plan's degree path** (the global-list source):
- The plan split each block with a per-atom Python loop. It now does one search per piece over the cumulative
  degrees (`_split_block`).
- The tiles are the same. `test_block_split_by_the_caps_is_the_per_atom_greedy` checks the split against the
  per-atom loop:
  - edge caps from 7 up to no cap;
  - degrees that include zeros and degrees above the edge cap;
  - blocks inside the array, of one atom, and empty.
- CPU tests on trillian at 853564ac: 161 passed, 91 skipped (`test_tiled.py`, `test_layering.py`,
  `test_neighbors.py`).

**`_slot_keys`: no change.**
- Its 0.74 s of host time at 884,736 atoms is mostly each tile's first neighbour search. Sweep 1 needs that search
  anyway, and the list cache keeps the result.
- The search kernels run on the device during that time. The gap profile charges less than 0.2 s of device idle
  time to the tile graphs and lists of the whole call (`tiled-call-memory.md`).
- So no change is worth making. The whole compiled call is 95 % device time.

## 2. The compiled repeated call at 70.96M (leto)

`tiled_capacity.py --mode auto --calls 3` with the default budget, at 05821103. Run:
`results/tiled-call-memory-20261101/runs/leto-auto3-NbCl5-69x69x207`.

| Call | Mode | R | Time | GPU peak | Host peak |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | eager (the first call measures its memory) | 4,096 | 1,747 s | 14.49 GiB | 213 GiB |
| 2 | compiled (4.78 GiB left at the eager peak) | 16,384 | 1,096 s | 17.73 GiB | 219 GiB |
| 3 | compiled | 16,384 | 1,034 s | 17.73 GiB | 220 GiB |

- The run is valid: the edge counts and the forces check pass.
- **The default, compiled with the budget, holds repeated calls at 70.96M** and leaves 1.8 GiB of the card free.

## 3. Same-process A/B: main against c8cb1a7f (murphy)

**What changed in between.** The two trees differ only in `tiled.py` and `tile_list.py`: the overlap, the
slimmed binning and plan, and the vectorised split.

**How the old code runs.** `ab_arms.py` arm `+old` loads c8cb1a7f's two files as modules of their own in the same
process.

**Arms.** Both compiled, with the budget (8192, 16384), as both trees default to. Medians come from alternating
blocks.

| NbCl5 atoms | Main | c8cb1a7f | Main / old | Forces gate |
| ---: | ---: | ---: | ---: | --- |
| 884,736 | 13.63 s | 15.65 s | **0.871** | forces within 1.4e-6 eV/Å, energy within 5e-8 meV/atom |
| 9,261,000 | 133.5 s | 160.0 s | **0.834** | forces within 1.4e-6 eV/Å, energy within 3e-8 meV/atom |

Runs: `results/tiled-leftovers-20261101/runs/murphy-abold-*`.

## 4. Step times (results/step-time-vs-atoms-20261031)

**Change to the plot.**
- The tiled NbCl5 rows are replaced by four current compiled points.
- The tiled NaCl rows stay, labelled in the plot as older eager force calls.
- `plot.py` was rerun.

| NbCl5 atoms | Seconds per step | What | Run |
| ---: | ---: | --- | --- |
| 884,736 | 14.13 | compiled MD, median of the steady steps (12 steps) | `murphy-md-compiled-*` |
| 9,261,000 | 135.5 | compiled MD, the steps without a slot change (6 steps) | `murphy-md2-compiled-*` |
| 40,001,688 | 628 | compiled repeated call (auto) | `murphy-auto3-NbCl5-57x57x171` |
| 70,957,944 | 1,034 | compiled repeated call (auto) | `leto-auto3-NbCl5-69x69x207` |

**Why repeated calls at 40M and 70.96M.** `tiled_md.py` keeps the positions, velocities and forces as float64 on
the device, 72+ bytes per atom. That does not fit beside the tiled call there. A repeated call is the MD step's
force call, without the integrator.

**Tool changes.**
- `tiled_md.py` now builds the tiled field with `call_rows = R`. Without it, the first build at 9.26M tried to
  allocate a 17.7 GiB whole-system species table.
- `tiled_md.py` records each step's wall time.

**A finding: slot growth during MD.**
- In MD the padded slots grow as atoms move: 5 changes in 12 steps at 884,736 atoms, and 9 in 6 steps at 9.26M.
- Each growth recompiles the slot: about 10 s at 884,736 atoms, and 35 s at 9.26M.
- So at 9.26M the steps alternate between 135 s and 170 s. NVE drift stays small: 0.10 and 0.06 meV/atom over the
  run.
- A larger headroom for the halo and edges on the first sizing, or growth by more than a quarter, would settle
  this sooner. That is a candidate for the conductor; nothing is changed here.
