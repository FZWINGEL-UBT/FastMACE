# 1b-T09: integrator qualification (ASE equivalence and the BAOAB correction)

Plan task 1b-T09, 19 October 2026. Branch `fastmace/1b-T09`; GPU run from
commit `fedfcde`. SPEC §12.1 "Integrator corrections", §12.2 "ASE
equivalence", owner decision D8. The results are also held as
`fastmace.reference.integrators.QUALIFICATION`, which 1b-T13 enters in the
reference record.

## What is qualified

The full-cell BAOAB has two variants:

- the **ASE variant** (`IntegratorConfig.ase_trace_drag_all_elements=True`)
  adds Tr(p_ε)/N_dof to all nine elements of the drag, as ASE 3.27 does. It
  must reproduce ASE's `LangevinBAOAB` exactly, which tests the port;
- the **production variant** (the default) drags with
  (p_ε + Tr(p_ε)/N_dof · I)/W (Quigley and Probert 2004, Eq. 5b). It must
  reproduce an independent reference with the same correction.

Hydrostatic cells have no off-diagonal drag, so both variants equal ASE there.

## The independent references (`fastmace/reference/integrators.py`)

- `CorrectedLangevinBAOAB`: ASE's `LangevinBAOAB` with only
  `_barostat_BAOAB_OU_gamma_mod`, the drag, replaced. It keeps ASE's noise
  stream, so trajectories compare exactly. `check_ase_source` refuses an ASE
  whose drag, Ornstein–Uhlenbeck step or step order differs from what the
  patch was checked against (SHA-256 of the method sources; the same in
  ASE 3.27.0, used on the workstations, and 3.29.0, in `fastmace_cpu`).
- `baoab_step`: one float64 full-cell step written from the equations of
  motion, with the noise passed in. I wrote it from the equation, not from
  `fastmace/integrators/baoab.py`. With `trace_all_elements=True` it
  reproduces ASE, which checks the step itself.

## CPU (float64, `tests/cpu/test_integrator_qualification.py`)

A tiny random MACE model (`tiny_mace_calculator([11, 17])`) on rattled NaCl
2×2×2 (64 atoms, N_dof = 192) gives forces and stress on both sides through
its ASE calculator, so only the integrators differ. Runs of 300 steps of 1 fs
heat from 0 to 300 K under 0.5 GPa (full cell: an anisotropic load of 0.5,
0.75 and 0.35 GPa). The barostat moves the cell by more than 0.2 % and
changes its shape.

| Comparison (cell, positions, velocities) | Largest difference | Limit |
| --- | ---: | ---: |
| ASE variant vs ASE `LangevinBAOAB`, hydrostatic | 1.8e-14 | 1e-9 |
| ASE variant vs ASE `LangevinBAOAB`, full cell | 1.1e-13 | 1e-9 |
| Production vs `CorrectedLangevinBAOAB`, full cell | 1.1e-13 | 1e-9 |
| Production vs ASE variant (negative control) | 4.3e-4 | > 1e-6 |
| `baoab_step` with ASE's drag vs ASE, 1 and 3 steps | 3.6e-15 | 1e-12 |
| `baoab_step` vs `CorrectedLangevinBAOAB`, 1 and 3 steps | 1.4e-17 (velocities) | 1e-12 |
| Production, 3 steps vs `baoab_step` | 1.6e-14 | 1e-12 |

The single steps start from a 300 K Maxwell–Boltzmann state with a barostat
momentum that has a large trace and off-diagonal elements (and, for the
production steps, from rest of the barostat), so the two drags differ there
by 3.6e-6 to 1.0e-5 in the velocities. The Lennard-Jones argon tests in
`tests/cpu/test_integrators.py` (1a-T12) stay as they are. 15 tests, 39 s
on one thread.

## GPU (§12.2, `tests/gpu/test_ase_equivalence_gpu.py`)

murphy, NVIDIA RTX 4000 Ada (driver 595.58.03), NAS environment
`mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0,
ASE 3.27.0). MH-1 (`mh1-r2scan`, head `matpes_r2scan`) on NaCl 3×3×3
(216 atoms), plain cuEq float64. FastMACE runs the engine that
`FixedPlanner(engine=EngineConfig(backend="cueq"))` plans and builds (mode
`pipelined`, record `single_graph_strain`). ASE runs MACE's
`MACECalculator` (cuEq, float64). Each run is a fresh process, same seed
20261019. The GPU was idle before the start (no compute process of anyone).
19 October 2026, 01:09–02:29 UTC; both tests passed (4772 s). JSON in
`results/1b-T09-murphy-20261019/`.

**Langevin**, 0 → 400 K in 1 ps and 400 → 0 K in 8 ps (18,000 steps of
0.5 fs, friction 0.02) from the 0 K lattice, the protocol of the research
engine's MD checks:

| Measure | Result | Limit |
| --- | ---: | ---: |
| Largest bin difference of the mean potential energy, 16 bins of 50 K (heating and cooling) | 3.8e-11 meV/atom | 0.01 meV/atom |
| Largest bin difference of the mean temperature | 3.5e-10 K | 0.1 K |
| Final temperature, ASE / FastMACE | 23.3422257904 / 23.3422257904 K | |
| Final positions | 1.0e-11 Å | |

The shared noise stream keeps the two runs together; NaCl stays nearly
harmonic, so they do not diverge.

**BAOAB, full cell**, 2,000 steps at 300 K and 1 GPa from a 300 K
Maxwell–Boltzmann start (volume −2.0 %):

| Pair | Positions (Å) | Cell (Å) | Potential energy (meV/atom) |
| --- | ---: | ---: | ---: |
| ASE variant vs ASE `LangevinBAOAB` | 4.8e-13 | 1.8e-13 | 1.1e-11 |
| Production vs `CorrectedLangevinBAOAB` | 4.7e-13 | 1.5e-13 | 1.1e-11 |
| Production vs ASE variant (the correction) | 6.1e-6 | 3.5e-8 | 9.9e-5 |
| `CorrectedLangevinBAOAB` vs ASE | 6.1e-6 | 3.5e-8 | 9.9e-5 |

Each engine follows its own reference about 10⁷ times more closely than the
two variants differ from each other, and the references differ by exactly as
much as the engine variants do.

## Commands

CPU (local, `fastmace_cpu`):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 0 tests/cpu/test_integrator_qualification.py
```

GPU (murphy, the environment block of `docs/validation/phase-0.md` with
`FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models`), from a copy of the
branch in `/mnt/homes/zwingel/fastmace-branches/1b-T09`:

```bash
$E/bin/python -m pytest -n 0 -p no:cacheprovider -s --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-1b-T09 tests/gpu/test_ase_equivalence_gpu.py
```

Both GPU tests are marked `slow` (1h20 together).

## Notes

- The per-bin temperature limit (0.1 K) and the BAOAB criterion (each pair
  at least 100 times closer than the variants) are this task's; SPEC §12.2
  gives the 0.01 meV/atom energy limit only.
- The tests read only the simulation state and outputs, not integrator
  internals, so they survive 1b-T05 moving `p_eps` and `force_eps` into the
  state.
