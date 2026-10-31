# Tiled capacity path: compiled tile bodies

The conductor's next step of 31 October 2026, after T1's graph falsifier ([capacity-tiled-t1t2.md](capacity-tiled-t1t2.md)).
The tile body is GPU-bound, so the remaining lever was Inductor:

- compiled tile bodies at the padded slot's fixed shapes, with the force derivative outside the compiled frame;
- slots from a few edge buckets instead of one slot as large as the largest tile;
- falsifier first: sweep 2 on the largest tile, compiled against eager, target ≥ 1.3×.

## Verdict

**The falsifier passes at 1.63×.** The whole tiled call now costs 1.43× and 1.32× main's whole MD step at 21,600 and
51,840 NbCl5 atoms. The eager tiles cost 2.1× and 2.0×.

**The force/stress gate passes**, with the default production path's errors: NaCl, NbCl5 and NaNbO3, ratios
0.15-1.03. **300-step NVE runs with 47 and 55 neighbour rebuilds** follow the compiled whole-system call to float32
rounding, with identical drift.

## What is built (`fastmace/model/tiled.py`)

**`TileConfig(mode="compiled")`.** Every tile runs in a padded slot, as in mode "padded". Each sweep's forward,
up to its objective, is one `torch.compile` function:

- with `dynamic=False` and the model's Inductor options;
- sweeps 1 and 3 apart, without and with autograd;
- `torch.autograd.grad` runs outside the compiled frame, as the split-forces path does for OEq models. AOT
  autograd compiles the backward too.

**It needs a compiled force field** (the evaluation plan's default compile mode):

- Its model is `compile_model`'s simplified, traceable copy, and the tiles run that copy's modules (`_orig_mod`).
- An eager field's e3nn modules would differentiate to zero through their TorchScript parts under compilation, so
  mode "compiled" refuses an eager field.
- The other modes now also accept a compiled field, and run its modules uncompiled.

**Holder modules.** Interaction 1 reads the stored `x = linear_up(h1)` and `sc = skip_tp(h1)` through holder
modules, whose buffers a compiled body takes as inputs. They replace the closure stand-ins, which a compiled
function would bake in.

**Slots per edge bucket** (`TileConfig.bucket`, default 0.15):

- **The bucket.** A tile runs in the slot of its edge bucket, `ceil(edges / step)`. The step is set on the first
  call from the largest tile: 15 % of its edges, at least 256 and a multiple of 256.
- **The slot.** It holds one receiver row more than the largest of its tiles (the padding receiver), their halo
  with `headroom`, and its bucket's edge count.
- **Growth.** A slot grows only when a tile does not fit it; its graphs and compiled shapes then go.
- **Graphs** live with their slot.
- **What it fixes.** Small tiles no longer pay for the largest tile's slot: mode "padded" is back at eager speed,
  0.51 s at 21,600 atoms against 0.75 s with one slot.
- Dynamo's cache limit is raised to 64 specialisations, as `compile_model` raises it: one per slot and sweep.

**Tests.** `tests/cpu/test_tiled.py` adds:

- the padded call with one slot (`bucket=1`) and with buckets, against the whole-system call in float64;
- slot reuse, and growth with the same forces;
- the refusals (compiled mode on an eager field, a bucket of 0).

## Falsifier

**Tool:** `tools/bench/tiled_graphs.py compiled`, NbCl5 5×5×12 (21,600 atoms), murphy (RTX 4000 Ada 20 GB).

- **The tile.** The largest one: 4,096 receivers, 2,324 halo atoms, 107,690 edges, in a slot of (4,097, 2,620,
  114,688).
- **The comparison.** Sweep 2's body with its forward compiled against the same body uncompiled, on the same
  modules: 16 iterations after three warm-ups each, CUDA events.

| | Sweep 2 on the largest tile |
| --- | ---: |
| uncompiled | 43.39 ms |
| compiled | 26.50 ms |
| **speed-up** | **1.64×** (target ≥ 1.3×) |

The first run, on one slot (`d1545fd4`), gave 43.41 / 26.66 ms: 1.63×.

**The outputs agree to float32 rounding:**

- edge-vector gradients within 1.0e-6, against a largest value of 0.30;
- `dE/dx` within 9.7e-8 (largest 0.17), `dE/dsc` within 1.3e-7 (largest 0.076);
- the strain gradient within 3.1e-4 of 833, 4e-7 relative.

## Whole calls against main's MD step

**Tool:** `tools/bench/tiled_graphs.py call --modes padded compiled`, on the compiled force field, R = 4,096; four
calls each, the last three steady.

| NbCl5 | Atoms | Main, whole MD step (graph tier) | Tiled, eager tiles (eager field) | Tiled, padded, 2 slots | **Tiled, compiled, 2 slots** | Compiled / main |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5×5×12 | 21,600 | 0.243 s | 0.52-0.53 s | 0.50 s | **0.348 s** | **1.43×** |
| 6×6×20 | 51,840 | 0.605 s | 1.2 s | 1.17 s | **0.800 s** | **1.32×** |

- **Sources.** Main's steps are from [capacity-tiled.md](capacity-tiled.md) (`capacity_scan.py point`, 50 steps,
  same GPU). The eager tiles are from the same report.
- **First calls,** including compilation: 20.8 s at 21,600 atoms and 65.9 s at 51,840 atoms: three sweeps × two
  slots.
- **Forces,** compiled against padded on the same field: within 6.3e-6 eV/Å. Energy within 2e-5 meV/atom, stress
  within 3e-9 eV/Å³.
- **Peak above the call's start:** 1.08-1.09 GiB, against 1.26-1.29 GiB uncompiled.
- **The comparison is not like for like.** Main's step includes the integrator and the neighbour-list stages; the
  tiled figures are the force call alone. The ratios therefore slightly favour main.

## Force/stress gate (compiled tile bodies)

**The run.** `tools/force_stress_gate.py --engine fastmace --tiled-receivers 128 --tiled-mode compiled`:

- NaCl, NbCl5 and NaNbO3 3×3×3;
- 3 rattled hydrostatic and 3 rattled full-strain configurations each;
- the references in their own processes;
- the production child builds the compiled evaluation plan and the compiled tile bodies.

Record: `results/tiled-compiled-20261031/runs/tc-gate/gate/metadata.json`.

**Columns.** Each cell is the error against float64, then the ratio to the float32 yardstick's error.

| Structure | Energy (meV/atom) | Max force (eV/Å) | Max stress (eV/Å³) | Gate |
| --- | ---: | ---: | ---: | --- |
| NaCl | 9.9e-5 (1.00) | 3.7e-6 (0.95) | 4.4e-8 (1.03) | pass |
| NbCl5 | 1.2e-4 (0.15) | 8.3e-5 (1.03) | 3.2e-8 (0.32) | pass |
| NaNbO3 | 3.0e-4 (0.28) | 1.4e-4 (1.02) | 1.9e-7 (0.58) | pass |

These are the same errors as the default production path and the eager tiled path ([capacity-tiled.md](capacity-tiled.md)).

## NVE with neighbour rebuilds (compiled tile bodies)

**Tool:** `tools/bench/tiled_md.py --mode compiled --path both`.

- 300 velocity-Verlet steps of 2 fs, starting at 600 K.
- The tiled call and the compiled whole-system call of the same compiled force field, from the same start.

| Cell | Atoms | Rebuilds (tiled / whole) | Max \|ΔE_pot\| per atom | Final max \|Δx\| | Drift, tiled / whole (meV/atom) | Slot changes |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| NbCl5 4×4×8 | 9,216 | 47 / 47 | 8.8e-5 meV | 1.0e-5 Å | 0.0633 / 0.0634 | 5 |
| NaCl 12×12×12 | 13,824 | 55 / 55 | 1.8e-5 meV | 4.7e-6 Å | 0.0328 / 0.0328 | 4 |

**Slot sizing had to change for MD.** The first runs changed slots 36-39 times in 300 steps, and every change is a
new shape and a new compilation:

- **The cause.** Tiles are planned again every call from the moving positions, so a tile's receiver and halo
  counts move by a few atoms, and a slot sized exactly to its largest tile grew again and again (2.8 s per step).
- **Receivers** now get the headroom as well, up to the cap (no tile exceeds `TileConfig.receivers`) and the atom
  count.
- **Growth.** An exceeded size grows by at least a quarter.
- **Now 4-5 changes per run.** The runs above are short, so these compilations still dominate their average step
  (0.37 and 0.57 s); the steady call is the whole-call timing above.

## What changed

- `fastmace/model/tiled.py`:
  - `TileConfig.mode="compiled"` and `TileConfig.bucket`;
  - compiled forwards of the three sweeps;
  - holder modules;
  - slots per edge bucket, with graphs per slot;
  - slot headroom and geometric growth;
  - `slot_changes`.
- `tests/cpu/test_tiled.py`: slots per bucket and one slot, reuse and growth, the new refusals.
- `tools/bench/tiled_graphs.py compiled` (the falsifier); `call` builds the compiled field for the compiled mode.
- `tools/force_stress_gate.py --tiled-mode`, `tools/bench/tiled_md.py --mode`.
- `results/tiled-compiled-20261031/`: the job scripts and records.

## Checks

- Ruff is clean on every changed file.
- **CPU tests** on murphy (GPU hidden, `-n 8`, `mace_cueq_latest`): `test_tiled.py` and `test_layering.py`,
  **57 passed** (`17bfe618`).
- **GPU:** the gate and the NVE runs above.

## Setup

- **Machine:** murphy, RTX 4000 Ada Generation 20 GB (driver 595.99.02), NAS `mace_cueq_oeq` (torch 2.14.1+cu130).
- **Idle check** before every job (`results/cap-tiled-20261030/run.sh`).
- **Model:** MH-1 float32 hybrid.
- **FastMACE:** branch `fastmace/tiled-compiled`, rebased onto main `bacf6e07`. Sources per run are in the records:
  falsifier `d1545fd4` and `f5495aca`, calls `f5495aca`, gate `f8f82fef`, NVE `17bfe618`.
- **When:** 30-31 October 2026, night CET.

