# 1b-T16: the gate on the user's structure

Plan task 1b-T16 (SPEC §12.3, §12.2 gate row, §9.3 rule 14, §7.6), 19 October 2026. Code:
`fastmace/planner/selfcheck.py`; FastMACE commit `fff5296` (branch `fastmace/1b-T16`, on main
`b296654`).

## What the gate does

The gate is opt-in (owner decision D1): `check="skip"` (default) records `skipped`;
`check="gate"` runs it; `check="allow_unavailable"` runs it and lets an unavailable gate pass,
recorded.

1. `references(plan, force_field, atoms, integrator, *, check, seed)`, before any production
   engine. It makes 2 rattled copies (Gaussian, 0.05 Å); NPT adds 2 strained copies (±2 %,
   isotropic for a hydrostatic cell), which are also rattled. On each copy it runs 1b-T13's float64
   reference and float32 yardstick (float64 energy accumulation), one calculator at a time, each
   freed before the next. Once a production engine exists in the process, the references run in a
   child process (`python -m fastmace.planner.selfcheck`). A reference that cannot run gives
   `GateUnavailable` under `check="gate"`, raised before the engine is built.
2. `production(engine, references, *, check)`. A registry snapshot is taken. Each copy is loaded
   into the state, its neighbour list rebuilt, and stages C and B are replayed through the engine
   (captured in graph mode). The energy, forces and stress are read, and the registry then
   restores the state. If the neighbour capacity grows on a copy, the list is rebuilt at the
   committed state and the graphs are captured again (this is recorded). The ratio rule is applied
   with the floors 0.01 meV/atom, 1e-5 eV/Å and 1e-7 eV/Å³. Non-finite values fail first. An NVT
   plan records "stress: not in plan". A failing gate raises `GateFailed`.

## CPU (tiny models, float64 production, eager stages)

`tests/cpu/test_selfcheck.py`, 46 tests: 46 passed in WSL `fastmace_cpu` (`-n 0`,
`OMP_NUM_THREADS=1`, 52 s) and on murphy in NAS `mace_cueq_latest` (CUDA hidden, 46 s).

- The ratio rule and the floors on synthetic numbers. NaN or inf in either path and in any
  quantity fails, in either order.
- NVT, NPT hydrostatic and NPT full cell pass. After the gate, every registered tensor is bitwise
  unchanged, and 8 steps after the gate are bitwise equal to 8 steps of an engine built without
  it (positions, velocities, cell, energy).
- A 1 meV offset on every Na atom's E0 in the production force field fails on energy alone
  (0.5 meV/atom). Forces and stress pass.
- An out-of-memory reference: `GateUnavailable` naming `check="allow_unavailable"`, before any
  engine was built; under `allow_unavailable` the record is `unavailable`.
- After a production engine exists, the references run in a child process and equal the
  in-process ones within 1e-12.
- A copy that grows the neighbour capacity (cell × 0.85): the gate passes and the later trajectory
  stays bitwise equal on the CPU, where edges are not padded.

## GPU: murphy (19 October 2026, 06:03–06:14 CEST)

| Item | Value |
| --- | --- |
| Machine, GPU | murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03; idle before the job (no compute process of anyone) |
| Environments | pytest in NAS `mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0); each case in a fresh child in NAS `mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`, OEq 0.7.0) |
| FastMACE | `fff5296`, `git archive` export at `/mnt/homes/zwingel/fastmace-t16`; models from `/mnt/homes/zwingel/ResearchModule/Models` |
| Plan | `MinimalPlanner` from `fm.load` (pristine) and a float32 `ForceField`: hybrid float32, graph tier |
| Structures | NaCl 3×3×3 (216 atoms), NbCl5 2×2×2 (288 atoms); gate seed 1 |
| Baseline | none: the comparison is with 1b-T13's references, not with the research engine |

```bash
$E/bin/python -m pytest -p no:cacheprovider -rs -v -s \
  --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-t16-gpu tests/gpu/test_selfcheck_gpu.py
```

Result: **7 passed** (694 s). JSON: `results/1b-T16-gate-murphy-20261019/cases.jsonl`.

Each cell gives the production error / yardstick error (limit):

| Case | Status | Energy, meV/atom | Force, eV/Å | Stress, eV/Å³ | Record |
| --- | --- | --- | --- | --- | --- |
| MH-1 r²SCAN, NVT | passed | 1.0e-4 / 5.3e-5 (0.01) | 2.7e-6 / 2.4e-6 (1e-5) | not in plan | `mh1/763155208cd1ef8c` |
| MH-1 r²SCAN, NPT full cell | passed | 1.2e-4 / 5.2e-5 (0.01) | 3.1e-6 / 3.1e-6 (1e-5) | 2.2e-8 / 3.1e-8 (1e-7) | `mh1/763155208cd1ef8c` |
| MP-0 medium | passed | 1.7e-4 / 1.3e-4 (0.01) | 2.3e-6 / 2.6e-6 (1e-5) | not in plan | `residual/8ee53b93c3df1f6f` |
| POLAR-1-S | passed | 3.3e-4 / 1.9e-4 (0.01) | 4.1e-6 / 3.7e-6 (1e-5) | not in plan | `polar/8e216bde0c3515e9` |
| OMOL-0 XL | passed | 2.1e-3 / 9.0e-4 (0.01) | 8.2e-4 / 6.8e-4 (1.35e-3) | not in plan | `omol/fb04d91bc2a91209` |
| MH-1, NbCl5 2×2×2 | passed | 1.6e-4 / 2.0e-4 (0.01) | 2.8e-5 / 3.0e-5 (6.1e-5) | not in plan | `mh1/763155208cd1ef8c` |
| MH-1, broken captured graph | **failed** (expected) | 1.0e-4 (passes) | 5.6e-4 / 2.4e-6 (1e-5) | not in plan | `mh1/763155208cd1ef8c` |

- **Energy:** with float64 accumulation on both sides, every production energy is within the
  0.01 meV/atom floor. The floor is the limit in every case (the largest ratio is 2.3, OMOL-0
  XL). In 1a, POLAR-1 and OMOL-0 production energies were 2.4–95 meV/atom off on the counting
  structures; here they are 3.3e-4 and 2.1e-3 meV/atom.
- **Broken captured graph:** a stand-in for a wrong kernel that only the captured stage B runs
  (forces × 1.001 during capture). It fails on forces (ratio 237). The same comparison through the
  eager stages passes (2.6e-6 eV/Å).
- **In-process against fresh-process references** (the second call ran in a child process after
  the engine was built), largest |Δ| over the copies:

  | Model | float64 reference: energy, forces, stress | float32 yardstick: energy, forces, stress |
  | --- | --- | --- |
  | MH-1 | 0, 6.9e-16, 4.1e-18 | 6.0e-6 eV, 3.2e-7, 2.8e-9 |
  | POLAR-1-S | 0, 3.4e-15, 1.0e-16 | 4.3e-6 eV, 1.7e-6, 1.3e-8 |
  | OMOL-0 XL | 0, 2.8e-13, 7.5e-16 | 2.3e-5 eV, 1.3e-4, 1.1e-6 |

  The float64 references agree within 1e-12. The yardstick agrees within each model's float32
  process scatter: 1e-5 eV/Å, and 2e-4 eV/Å for OMOL-0 XL (1a-T19: 1.2–1.8e-4).
- **Time** (seconds: references; engine build, which the gate does not add; production half):

  | Case | References | Build | Production half |
  | --- | --- | --- | --- |
  | MH-1 NaCl NVT | 15.4 | 49.1 | 32.3 |
  | MH-1 NaCl NPT full | 13.4 | 31.7 | 27.3 |
  | MP-0 NaCl | 14.8 | 26.4 | 17.1 |
  | POLAR-1-S NaCl | 9.7 | 52.8 | 20.3 |
  | OMOL-0 XL NaCl | 15.5 | 69.6 | 31.9 |
  | MH-1 NbCl5 2×2×2 | 13.4 | 38.9 | 26.3 |

  So the gate adds 27–47 s, within "seconds to a minute". Most of the production half is the
  first compilation and capture of the B graphs, done again after the capacity grew (below); the
  run's first steps would otherwise pay part of that. In the broken-capture case the kernels were
  already compiled and the capacity already grown by the eager pass, and the captured replay took
  0.06 s.

## Deviations and notes

- `GateUnavailable` is raised by `references()` (step 4 of `Simulation`'s constructor), not after
  the build: an unavailable reference is known before the engine is built.
- The speculative engine's steps run combined ACB graphs. The gate replays the separate captured
  stages C and B, the same force-field code, because replaying A would move the copy.
- **The neighbour capacity grew in every NaCl 3×3×3 case** (the broken-capture case had already
  grown it in its eager pass; NbCl5 did not grow). Unrattled NaCl has a shell at a·√5/2 = 6.306 Å,
  just outside the build radius r_max + skin = 6.3 Å. A 0.05 Å rattle pulls part of that shell
  inside, which is more than the 15 % headroom. The same happens at the run's first rebuild under
  thermal motion, so the gate only brings the growth forward. But it leaves the larger capacity in
  place (recorded in `GateRecord.notes`), and every graph is captured again. On the CPU, where
  edges are not padded, the trajectory after a grown gate stays bitwise equal (test). On the GPU
  the padded edge counts can differ from a run without the gate, so a gated GPU run is not
  bitwise equal to an ungated one. The hybrid is not bitwise run to run anyway, but plain cuEq
  float64 is equal within 1e-12 Å, which a gated run need not keep. Question for the conductor:
  accept this, or restore the original capacity after the gate (a re-allocation in
  `neighbors/skin.py`, which this task does not own).
