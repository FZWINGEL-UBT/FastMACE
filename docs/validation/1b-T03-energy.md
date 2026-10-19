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

Local PC, NVIDIA GeForce RTX 5060 Ti (16 GB, WSL2), 20 October 2026,
02:39–03:12, in the conductor's slot. References: `mace_cueq_latest`, plain
cuEquivariance float64, eager, no rewrites. Production: `mace_cueq_oeq`,
main.py's fixed plan for the hybrid backend (float32, compiled, CUDA graphs),
the energy of the engine's initial evaluation. FastMACE: branch
`fastmace/1b-T03`, engine code of `7fe4345` (the later commits change only
tests, `tools/energy64_gpu.py`'s `--first` option and this document).
Baseline for "1a sum": the same build with `float64_energy=False`, which
reproduces 1a's energy exactly (CPU test). Structures from the baseline
export (`research-engine-baseline-20261017`, `Structures/Benchmark/R2SCAN`),
rattled by 0.03 Å (seed 1). Checkpoints by label: `polar1-s` (sha256
e4495612…), `omol0-xl` (9b64b4fd…), `mh1-r2scan` (a522eb7f…). Raw JSON, logs,
scripts and the `free -m`/load log in `results/1b-T03-rtx5060ti-20261019/`
(the JSON's `fastmace_commit` is empty: git in WSL does not read this Windows
worktree; the commit is the one above). Runs launched with
`setsid nohup`, `TORCHINDUCTOR_COMPILE_THREADS=4`, at least 10.9 GB RAM
available throughout the second part.

### Energies: float32 production against float64

Error = production − float64 reference, per atom.

| Model | Structure | Atoms | 1a float32 sum (meV/atom) | float64 sum (meV/atom) |
| --- | --- | ---: | ---: | ---: |
| POLAR-1-S | NbCl5 2×2×2 | 576 | −63.9 | +0.0004 |
| OMOL-0 XL | NbCl5 2×2×2 | 576 | −45.5 | +0.0014 |
| MH-1 `matpes_r2scan` | NbCl5 3×3×3 | 1944 | +0.122 | +0.00014 |
| MH-1 `matpes_r2scan` | NaCl 3×3×3 | 216 | −0.0051 | +0.00002 |

On NbCl5 3×3×3 the float64 references of POLAR-1-S and OMOL-0 XL do not fit
in 16 GB (CUDA out of memory), so those two models are compared on 2×2×2,
where the reference fits; the plan asked for 3×3×3. On 3×3×3 only the two
production sums exist: POLAR-1-S 1a −20,790,678.0 eV against float64
−20,790,840.03 eV (83 meV/atom apart), OMOL-0 XL −20,790,748.0 against
−20,791,047.98 eV (154 meV/atom apart). Since the float64 sum agrees with
the reference to 0.001 meV/atom on 2×2×2, these differences are the error of
1a's float32 sum at 3×3×3; OMOL-0 XL's exceeds the 45–102 meV/atom of 1a's
gates (§4.3), because the float32 summation error grows with the total. The
reference's two sums (float64 and MACE's own, both in a float64 model) agree
to 1e-15 relative in every case.

**Owner decision D2 (MH-1 on `main.py`).** The production energy of MH-1
`matpes_r2scan` changes by **0.12 meV/atom on NbCl5 3×3×3** (0.236 eV in
total) and 0.005 meV/atom on NaCl 3×3×3, against the expectation of D2
(below 0.01 meV/atom). The change is the removal of 1a's float32 summation
error: the float64-summed energy is within 0.00014 meV/atom of the float64
reference. Forces, stress and trajectories do not change.

### GPU tests

`tests/gpu/test_energy_gpu.py`, `mace_cueq_latest`: 4 passed (float32 and
float64, eager and compiled). The float64 energy replays in a CUDA graph as
eager, to the order of the GPU's atomic float64 adds (1e-13 relative) in a
float64 model and to the float32 per-atom noise (4e-7 eV on 64 atoms) in a
float32 one; forces and stress with and without the float64 sum differ by no
more than one force field's own replay noise (not bitwise on the GPU, where
both sides use atomic scatters). Tiny float32 model with all-electron E0:
float64 sum 2.6–5.9 µeV from the float64 model, stock float32 sum 0.28 eV.

### Step time

MH-1 `matpes_r2scan` NaCl 3×3×3, hybrid NVT, two engines in one process
(float64 sum, 1a sum), 300 warm-up steps, 12 alternating blocks of 300 steps
per side (`abba`), one process per build order:

| Built first | float64 sum (ms) | 1a sum (ms) | 1a / float64 |
| --- | ---: | ---: | ---: |
| float64 sum | 3.5656 | 3.5766 | 1.0031 |
| 1a sum | 3.5662 | 3.5760 | 1.0027 |

The float64 sum is not slower; in both orders it is 0.3 % faster, inside the
block-pair spread on this GPU (0.990–1.015) and the 1.8 % cross-process
spread of NaCl in `1a-noise-floor.md`. A first A/B process (02:52–02:55)
overlapped a CPU suite of another task and is discarded
(`ab-NaCl-overlapped/`).

Commands (from the repository root in WSL, `PYTHONPATH=$PWD`,
`FASTMACE_MODELS=~/models`, `FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18`):
`results/1b-T03-rtx5060ti-20261019/run.sh` and `run2.sh`, for example

```bash
python tools/energy64_gpu.py energy --side reference --model polar1-s --structure ~/fastmace-baseline/1a-T18/Structures/Benchmark/R2SCAN/NbCl5.cif@2x2x2 --output out.json
python tools/energy64_gpu.py ab --model mh1-r2scan --structure NaCl --first float32_sum_1a --output ab-dir
```

## 1b-T03b: the research-engine comparisons in `test_engine_gpu.py`

murphy, RTX 4000 Ada (idle GPU), NAS `mace_cueq_latest` and `mace_cueq_oeq`,
20 October 2026, branch `fastmace/1b-T03b`; research engine from
`/mnt/homes/zwingel/fastmace-baseline/1a-T18` (tag
`research-engine-baseline-20261017`). Scripts and reports in
`results/1b-T03b-murphy/`.

- **Stage B's kernels.** Plain cuEq float64, FastMACE against FastMACE with
  and without the float64 sum: graphs A and C are identical, stage B has the
  same node count (235) but different kernels. The reduction is part of the
  compiled graph, so Inductor fuses and numbers its Triton kernels
  differently. The research-engine comparisons (kernel sequence and
  trajectory) therefore run FastMACE with `float64_energy=False`
  (`spec(float64_energy=False)`), as the other research-engine comparisons
  do; the trajectory limits are unchanged.
- **Hybrid capture crash.** Not the float64 sum: it occurs with
  `float64_energy=False` too, and without the eager engine before. The
  test's own warm-up check stored `t.clone()` of every registered tensor on
  the default stream before calling `engine._capture`. One of them is the
  force field's positions, a leaf that requires grad; the clone created the
  leaf's `AccumulateGrad` node on the default stream and kept it alive, and
  the hybrid's eager `autograd.grad` (OEq differentiates outside the
  compiled graph) refused the capture. The library's `registry.snapshot()`
  already detaches; the test now does too (`t.detach().clone()`). Plain cuEq
  differentiates inside the compiled graph and was not affected.
- **Results:** the three tests pass (`-m slow -k against_the_research_engine`,
  966 s). Plain cuEq NVT: FastMACE against the research engine 2.5e-13 Å
  after 500 steps (the engines' own repeat scatter 2.5–2.9e-13 Å); NPT
  7.1e-15 Å; hybrid 1.0e-7 Å against a research-engine run-to-run scatter of
  8.7e-8 Å; warm-up checks empty in every mode. With the fix, the hybrid with
  the float64 sum also passes its captures and warm-up check.
