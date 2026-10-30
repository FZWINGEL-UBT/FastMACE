# Tiled capacity path: compiled tile bodies

The conductor's next step of 31 October 2026, after T1's graph falsifier ([capacity-tiled-t1t2.md](capacity-tiled-t1t2.md)).
The tile body is GPU-bound, so the remaining lever was Inductor:

- compiled tile bodies at the padded slot's fixed shapes, with the force derivative outside the compiled frame;
- slots from a few edge buckets instead of one slot as large as the largest tile;
- falsifier first: sweep 2 on the largest tile, compiled against eager, target ≥ 1.3×.

## Verdict

**The falsifier passes at 1.63×.** The whole tiled call now costs 1.43× and 1.32× main's whole MD step at 21,600 and
51,840 NbCl5 atoms. The eager tiles cost 2.1× and 2.0×.

GATE AND NVE: PENDING.

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
