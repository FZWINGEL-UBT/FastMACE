# 1b-T03: float64 energy accumulation

Plan task 1b-T03 (SPEC §7.3 "Energy" and "Energy and forces are one
scalar", §6.2 `float64_energy`, §4.3 energy scale). The potential energy is
now accumulated in float64: the E0 table is an exact float64 copy of the
checkpoint's table, and every other term is computed in the force precision
and reduced into float64. Forces and stress still come from MACE's autograd
of its own scalar (design A, SPEC §7.3 sixth amendment), so they keep every
bit.

## What changed

- `fastmace/model/energy.py`: the float64 reduction. Source rewrites of MACE
  0.3.16's `MACE.forward` and `ScaleShiftMACE.forward`, pinned by sha256; an
  unqualified source refuses the model (`EnergyReductionError`). The
  rewritten forward returns the float64 total as `"energy"`, per-atom float64
  energies of the atom-resolved terms (`"node_energy_float64"`) and the rest
  (`"energy_unresolved_float64"`). MACE's `"node_energy"` is never used.
- `fastmace/longrange/polar.py`: the same for `PolarMACE.forward`
  (`FLOAT64_ENERGY_EDITS`, composable with the PR 1762 port), with the
  electrostatic per-k and self terms reduced into float64
  (`coulomb_energy_pbc(with_float64=True)`) and the field term in float64.
  Enabled per instance (`set_float64_energy`).
- `fastmace/model/forces.py`: `MACEForceField(float64_energy=True,
  atomic_energies=None)`. The per-atom float64 E0 goes into the batch
  (`fastmace_node_e0`); the float64 forward is bound to the force field's
  private copy, or called as a function on the calculator's own model (never
  changed). The compiled-model cache key includes the flag.
  `per_atom_energies(atoms)` returns the per-atom energies and the terms
  they omit. `info()` records `float64_energy`.
- `fastmace/model/loading.py`: `LoadedModel.atomic_energies`, the
  checkpoint's E0 table copied in float64 while MACE's calculator unpickles
  it (`capture_atomic_energies`), before it converts the model to float32.
- `fastmace/planner/fixed.py`: `build` passes
  `PrecisionConfig.float64_energy` (default `True`, owner decision D2), so
  `main.py --engine fastmace` writes float64-accumulated energies. The plan's
  `precision` records it.

Which terms are atom-resolved (`fastmace.model.energy.TERMS`):

| Class | Per-atom terms | Not atom-resolved |
| --- | --- | --- |
| `MACE`, `ScaleShiftMACE` | E0, pair, interaction, embedding readout (OMOL-0) | — |
| `PolarMACE` | E0, interaction, local electron energy | electrostatic (k-space and self), field |

## CPU tests

`tests/cpu/test_energy.py`, tiny models with synthetic all-electron E0
tables (O −2,048, Na −4,413, Cl −12,523, Fe −34,510 eV, none a float32
number):

- float64 models of every family (mh1, residual, density, off, omol with its
  embedding readout, polar with its local electron energy): the float64 sum
  equals MACE's stock float64 energy within 1e-12;
- OMOL's and PolarMACE's stock `node_energy` misses a term; the energy
  includes it;
- float32 models: the error against float64 equals the forward-pass error
  (the same model with E0 = 0), at least 20× below the stock float32 error
  (2× for the tiny PolarMACE, whose float32 SCF alone is off by 13 meV on 64
  atoms); the model's float32-rounded E0 table is off by exactly the E0
  rounding, so only the checkpoint's table removes it;
- forces and stress bitwise equal with and without the float64 sum: float32
  and float64, eager (mh1, omol, polar) and compiled (mh1, `slow`). The
  compiled comparison runs with one intra-op thread: with two, two
  compilations of the *same* model already differ by about 3e-18 eV/Å
  (parallel reductions in Inductor's CPU kernels), with or without the
  float64 sum; with one thread both pairs are bitwise equal;
- per-atom energies plus the unresolved part sum to the total (1e-13);
- `float64_energy=False` gives the stock float32 energy exactly;
- `load` keeps the checkpoint's float64 table for a float32 model;
  `FixedPlanner.build` passes the flag through.

The polar cases need graph_longrange, which `fastmace_cpu` lacks; they ran
in `mace_cueq_latest` on the CPU (`CUDA_VISIBLE_DEVICES=''`,
`PYTHONPATH=<checkout>:<checkout>/tests`), together with
`tests/cpu/test_polar.py` and `tests/cpu/test_forces.py`.

Two existing tests now build with `float64_energy=False`: the bitwise
comparison with the research engine's `MACEForces` (CPU and GPU), which has
no float64 sum, and the streaming-counter test, which patches the class
`forward` (the float64 forward is built from MACE's own source).

`tests/gpu/test_energy_gpu.py`: the float64 energy replays in a CUDA graph
as eager (to the order of the GPU's atomic float64 adds), eager and compiled,
float32 and float64.

## GPU

Pending the conductor's GPU slot (local RTX 5060 Ti). Commands, from the
repository root in WSL with `PYTHONPATH=$PWD`, `FASTMACE_MODELS` and
`FASTMACE_STRUCTURES` set:

```bash
# References (plain cuEq float64) and production (main.py's hybrid plan), one process each.
python tools/energy64_gpu.py energy --side reference --model polar1-s --structure NbCl5 --output results/1b-T03/polar1-s-NbCl5-ref.json
python tools/energy64_gpu.py energy --side production --model polar1-s --structure NbCl5 --output results/1b-T03/polar1-s-NbCl5-f64.json
python tools/energy64_gpu.py energy --side production --float64-energy off --model polar1-s --structure NbCl5 --output results/1b-T03/polar1-s-NbCl5-1a.json
# Step time, float64 sum against 1a's sum, 12 alternating blocks in one process.
python tools/energy64_gpu.py ab --model mh1-r2scan --structure NaCl --output results/1b-T03/ab-NaCl
```
