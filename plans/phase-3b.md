# Phase 3b: NPT for the chlorides

Status: **in progress**, 22 October 2026, written while phase 3's acceptance
closes (report merged at 7dfa54c; the GPU slow tier's appendix and the owner's
`fastmace-3` tag follow). The owner accepted the recommendations for D1–D4 the
same day ("Accept the recommendations for phase 3b's D1-D4 and prestart the
phase 3b workers"). They are SPEC §18 decisions 40–43, and the workers are
prestarted. Spec: [SPEC.md](../SPEC.md) §16 row "3b. NPT for the
chlorides", and the sections it cites:
- §11.2: the GPU D3 term (own half list, own skin, three passes with
  analytic derivatives, stress, cost target, parameter table, acceptance);
- §9.3: rules 10, 11 and 12 with D3;
- §10.6: restart with a D3 list;
- §12.1: D3's reference;
- §7.7: the `"d3bj"` expansion;
- §18: decisions 6, 21, 26, 36 and 39.

Why it matters: SPEC decision 6 puts the chlorides' NPT melt–quench in the
first production deliverable, so GPU D3 is on the critical path. Without D3,
NbCl5 swells by about 20 % at 300 K and evaporates at 600 K under NPT. With
D3 (torch-dftd, ASE float64) it holds 29.0 ± 0.4 Å³/atom at 300 K and
31.6 ± 0.6 at 600 K (ResearchModule `studies/npt/README.md`, 15 October).

## Scope

- **GPU D3 in the force field**, as §11.2 specifies:
  - BJ damping with the `r2scan` set, 40 bohr pair and CN cutoffs, ATM off,
    no smoothing;
  - its own half list (i < j, plus each self-image pair once) with its own
    skin, every periodic image within the cutoff, and no per-step
    compaction;
  - its own displacement and strain test;
  - three passes with analytic derivatives: CN; energy, forces and virial at
    fixed CN; the CN chain term;
  - stress, capturable with `where` masks;
  - pair terms in the force precision, reductions in float64.
- **D3 everywhere a force field goes:** `ForceField`, `Simulation`, the
  planner (rules 10–12), the gate and reference tier, restart, MC species
  events, `fm.md`, `fm.calc`, `fastmace.ase` and the CLI. The `d3-phase-3b`
  refusals go.
- **D3 in the BAOAB NPT path**, with a pressure setter for scheduled targets
  (§10.1).
- **A constant-pressure melt–quench protocol** as data (owner decision D1).
- **The PolarMACE hybrid stress precision investigation** carried over by
  decision 26 (owner decision D2).
- **ResearchModule `main.py --dispersion`** (owner decision D4).

## Out of scope

- **Barostat qualification and equilibrium volumes.** They are an
  experiment, not library work (SPEC §16 note). The library keeps its
  defaults and records them.
- **Variable-cell relaxation and quench** (GPU FIRE and LBFGS): phase 4.
  Protocols relax and quench at fixed cell.
- **PolarMACE + D3 and PolarMACE NPT qualification** (D2).
- **Judging physics.** Whether a crystal stays solid is model qualification,
  not library acceptance (decision 6).

## Owner decisions

**Decided (user, 22 October 2026): the recommendation of each**, as SPEC §18
decisions 40 (D1), 41 (D2), 42 (D3) and 43 (D4). The conductor's choices
follow in the next section.

**D1. The acceptance melt–quench protocol.**
- **Recommended:**
  - NbCl5 3×3×3 (1944 atoms), MH-1 `matpes_r2scan` with D3-BJ;
  - hydrostatic NPT at 0 GPa;
  - the anneal profile at 1 fs (0 → 1000 K in 20 ps, then 3 K/ps to
    250 K);
  - the library's barostat defaults (P_tau = 20 T_tau, mass factor 1);
  - a fixed-cell LBFGS relaxation first, and quench frames relaxed at their
    NPT cells with the cell fixed;
  - its outcome recorded, not judged (decision 6).
- **Alternatives:** (b) 0.1 GPa, to keep the molecular chloride condensed at
  1000 K; (c) a 600 K peak.
- The production crystal set and pressures are a later, research decision.

**D2. PolarMACE in 3b.**
- **Recommended:** investigate the hybrid polar stress precision only: the
  NbCl5 3×3×3 POLAR-1-S stress against finite differences, hybrid against
  plain (decision 26; plans phase-1a, 1b, 2). No qualification.
  PolarMACE + D3 stays refused in 3b, with that reason.

**D3. If D3 misses the 10 % cost target.**
- §11.2's target: amortised, builds included, about 2.9 ms of NbCl5's 29 ms
  step, and provisional.
- **Recommended:** measure on the production ramp at 1 fs. If it misses,
  record it and continue; optimisations become a follow-up, not a blocker.

**D4. ResearchModule `main.py`.**
- **Recommended:** `--dispersion off|d3bj` for NVT and NPT, with `off` as
  the default and recorded in `run.json`. Whether production NPT of the
  chlorides defaults to `d3bj` is your call after acceptance.

## Conductor's choices (the owner may override)

1. **Second reference:** simple-dftd3 through PyPI's `dftd3` 1.6.0. It is
   Grimme's maintained program and satisfies §11.2's "the `dftd3` program".
   - If its wheel doesn't install cleanly, it is built with cmake and
     gfortran (both installed in WSL).
   - `tad-dftd3` 0.7.0 (pip) was meant as an optional third cross-check,
     but it has no periodic boundaries. It is installed and not used
     (3b-T01).
   - All three go into local `mace_cueq_latest`, under the local package
     rule.
2. **Explicit numbers in `fm.D3`:** s6 = 1, s8 = 0.7898, a1 = 0.4948,
   a2 = 5.7308 (torch-dftd's `r2scan` BJ set, doi 10.1063/5.0041008), the
   cutoffs, ATM off, and torch-dftd's unit constants. They are part of the
   force-field identity, and both references are called with exactly these
   numbers. torch-dftd defaults to zero damping, so `bj` is passed
   explicitly.
3. **Precision:** pair terms in the plan's force precision; the CN, energy
   and virial reductions in float64. The record's `precision` field names
   the reduction precision. The gate compares float32 plans against the
   float64 reference tier.
4. **Rule 11:** the D3 skin gets a strain term from the barostat (the
   expected volume change per rebuild interval) on top of the displacement
   term.
   - At zero displacement a 0.3 Å skin allows only about 0.8 % linear
     strain; melt–quench volume swings reach about 16 %.
   - The skin is chosen on the production ramp (3b-T06).
5. **MC:** the per-species D3 tables (C6 reference, CN) are derived state in
   the registry, so species events refresh them. A swap test with D3 on
   covers it.
6. **Environments:** torch-dftd 0.5.1 goes into local `mace_cueq_oeq` too,
   for the gate's D3 reference in the hybrid env (dry run first). The NAS
   environments wait until the workstations are back and the owner allows
   NAS changes again.
7. **The independent comparison:**
   - GPU NPT + D3 against ASE `LangevinBAOAB` + MACE + torch-dftd, on NbCl5
     1×1×3 (216 atoms; ASE is slow at 3×3×3);
   - at 300 K and 600 K, 0 GPa, 1 fs, 3 seeds per engine;
   - compared on mean V, V fluctuation, mean P and PE.
8. **Acceptance hardware** (decision 39): items in full on the RTX 5060 Ti.
   The one Ada spot check runs on the Modal L4 (at most $3) unless the
   workstations are back first.

## Tasks

| Task | Size | GPU | After | Files (main ones) |
| --- | --- | --- | --- | --- |
| 3b-T01 Contracts and references | S–M | no | — | `longrange/d3.py`, `model/forcefield.py`, `reference/`, tests, env installs |
| 3b-T02 D3 half list and rebuild test | M | yes | T01 | `neighbors/d3_list.py`, `neighbors/adaptive.py`, `planner/minimal.py` (rules 11, 12) |
| 3b-T03 D3 kernel, three passes | L | yes | T01 | `longrange/d3_kernel.py` (Triton) |
| 3b-T04 Integration | L | yes | T02, T03 | `model/forcefield.py`, `engine/graph.py`, `planner/`, `simulation.py`, `quick.py`, `ase/md.py`, `cli/`, `reference/`, restart |
| 3b-T05 NPT with D3, pressure setter, independent comparison | M | yes | T04 | `integrators/baoab.py`, `simulation.py`, tools |
| 3b-T06 D3 cost on the production ramp | S–M | yes | T04 | tools, `docs/validation/3b-T06.md` |
| 3b-T07 NPT melt–quench protocol | M | yes | T04, T05 | `protocols/` (profile fields, data), `cli/run.py` |
| 3b-T08 PolarMACE hybrid stress precision | S–M | yes | T01 | tools, `docs/validation/3b-T08.md` |
| 3b-T09 ResearchModule `main.py --dispersion` | S | smoke | T04, T05 | ResearchModule `langevin/` (conductor's ResearchModule worktree) |
| 3b-T10 Acceptance runs and report | M | yes | all except T09 | `docs/validation/phase-3b.md` |

**3b-T01 Contracts and references.**
- `fm.D3` gets the explicit numbers (choice 2), an identity hash entry, and
  the `precision` semantics (choice 3).
- `ForceField(dispersion=...)` gets its signature and validation.
- A `DispersionTerm` interface: build, needs_rebuild(positions, cell),
  evaluate → energy, forces and virial, plus counters.
- Reference wrappers: torch-dftd (BJ explicit), simple-dftd3 and
  tad-dftd3, on the same parameters.
- Test fixtures: rattled and strained NaCl, NbCl5 and NaNbCl6.
- Environment installs for choices 1 and 6.
- Merges first; T02, T03 and T08 start at once on its contract.

**3b-T02 D3 half list.**
- The half list with self-images and every image within 40 bohr (21.17 Å),
  built on the GPU with no per-step compaction: the kernel masks r² ≤ r_c².
- Its own reference positions and cell, with the displacement and the
  deformation-aware strain test (as `skin.py`'s, with the D3 cutoff).
- Rule 11's D3 skin with the strain term, and rule 12's memory estimate
  (about 1.4–1.75 M pairs on NbCl5 3×3×3).
- Build counters and time; the rebuild at canonical restart points.
- **Tests:** against a brute-force enumeration on the CPU (pairs equal as
  sets, every image), and on the GPU.

**3b-T03 D3 kernel.**
- Triton, three passes:
  - pass 1, CN;
  - pass 2, energy plus direct forces and virial at fixed CN, accumulating
    dE/dCN;
  - pass 3, the CN chain term into forces and virial (about 0.0013 eV/Å if
    left out).
- `where` masks only, so it is capturable; the precision rules of choice 3.
- **Tests, on the rattled and strained cells:** float64 energies, forces and
  stress against torch-dftd **and** simple-dftd3, to their agreement with
  each other; float32 against float64 within the gate; capture and replay
  bitwise.

**3b-T04 Integration.**
- The D3 term in the force graph (captured with MACE's) and in the eager and
  compiled tiers.
- Planner rules 10–12 with D3. The gate's energy, force and stress parts
  include D3.
- The reference tier: the stock calculator plus torch-dftd as a
  `SumCalculator`, labelled non-stock (§12.1).
- Restart: the identity, and the rebuild at canonical points. MC species
  events refresh the D3 tables (choice 5).
- `fm.md`, `fm.calc`, `fastmace.ase` and the CLI accept `"d3bj"`. The
  `d3-phase-3b` refusals go, and reports and provenance label D3.
- **Tests:** the existing suites with D3 on, plus D3 off bitwise identical
  to before.

**3b-T05 NPT with D3 and the comparison.**
- The D3 virial in BAOAB's stress path, and a pressure setter (a device
  scalar fill; scheduled targets as §10.1 specifies).
- NPT tests with D3, and the independent comparison of choice 7, an
  acceptance item.

**3b-T06 D3 cost.**
- The amortised D3 cost, builds included, on NbCl5 3×3×3 along the
  production ramp at 1 fs (NVT and NPT), against the same run without D3.
- The skin is chosen there. The outcome follows D3.

**3b-T07 NPT melt–quench protocol.**
- `Profile` gets fields for the ensemble, pressure (schedule) and
  dispersion.
- The D1 protocol becomes data, with a short variant for smokes.
- Quench frames are relaxed at their NPT cells with the cell fixed.
- It runs end to end locally through `fastmace run`, with every stage
  recorded. The outcome is recorded, not judged.

**3b-T08 PolarMACE stress precision (D2).**
- POLAR-1-S hybrid and plain stress against central finite differences on
  NbCl5 3×3×3.
- It explains or bounds the 4.4e-4 eV/Å³ gap (plans phase-1a line 901) and
  states what NPT with PolarMACE would need. Investigation only.

**3b-T09 ResearchModule `--dispersion` (D4).**
- The flag, `run.json`, resume, CPU tests and a short GPU smoke.
- Docs: `protocol.md` and `configuration.md`, plus a line in AGENTS.md.

**3b-T10 Acceptance (decision 39).**
- §16 row 3b's items:
  1. D3 against both references;
  2. GPU NPT + D3 against the independent implementation;
  3. the amortised cost against 10 %;
  4. the protocol end to end.
- Plus a no-regression A/B against the `fastmace-3` tag, with D3 off bitwise
  identical on CPU float64 and in reproducible mode.
- One Ada spot check (choice 8). Timing repeats sized to the effect.

## Order and parallelism

```text
Wave 0   T01 contracts and references
Wave 1   T02 D3 list   T03 D3 kernel   T08 PolarMACE stress
Wave 2   T04 integration (T02 T03)
Wave 3   T05 NPT + comparison (T04)   T06 cost (T04)   T07 protocol (T04 T05)   T09 main.py (T04 T05)
Wave 4   T10 acceptance
```

Merge orders where files overlap:

| File | Editors, in merge order |
| --- | --- |
| `model/forcefield.py` | T01, T04 |
| `planner/minimal.py` | T02, T04 |
| `simulation.py` | T04, T05, T07 |
| `integrators/baoab.py` | T05 |
| `protocols/` | T07 |

## Rules carried over from phase 3

- **GPU tests:** a task that changes behaviour covered by GPU tests (slow or
  acceptance) runs the affected ones itself. Trials run on the CPU only, and
  in phase 3 seven such tests went stale.
- **Workers' GPU launches:** they run through their own worktree's
  `tools/ci/local_gpu.py run` as background tasks of their session. A
  refused launch waits for the owner. The owner's approval up front in each
  worker session, or the allow rule for a fixed local launcher, avoids the
  waits of 22 October.
- **Small, low-risk tasks** (T09) skip the prestart report, the separate
  validation document and GPU checks beyond a smoke.
- **Timing:** repeats are sized to the effect. Cross-process controls use
  the documented 3 % noise floor or are left out.
- **Workstations:** none until the owner says they're back. Then only when
  nobody else is on the GPU or running active jobs, with files only under
  `/mnt/homes/zwingel`.
- **Modal:** under the guard, at most $5 for the phase (the month stood at
  $7.86 on 22 October).

## Conductor answers to the prestart reports (22 October 2026)

- **3b-T01:** the draft contract is on its branch (9a59cbb). Additions:
  - `evaluate_energy()`, energy only, for MC energy fields;
  - torch-dftd 0.5.1 and `dftd3` 1.6.0 also go into local `fastmace_cpu`,
    because trial merges run the CPU suite there and the D3 CPU tests would
    otherwise skip.
- **3b-T02, the list:**
  - 12 B per pair, sorted canonically by (i, j, image). Padded slots are
    i = j = N, with a device count.
  - Per-pair integer images (skin.py's convention: positions unwrapped, the
    atoms' build offsets absorbed), times the current cell at each step.
  - The runtime test is skin.py's deformation-aware test with the D3
    cutoff. The skin is max(1.0 Å, 2·n·d + r_c·f/(1−f)), with n = 50
    provisional; T06 sets the numbers. It applies to every run with D3 on.
  - Rule 12 records the pair and memory estimate. It warns above 25 % of
    device memory and refuses only when the D3 buffers alone exceed free
    memory. It drops no rewrites.
  - Registry entries and wiring stay with T04.
  - The builder is device-agnostic, so the CPU tier gets the same list.
- **3b-T03, the kernel:**
  - One Triton program per atom over a CSR, with float64 register sums and
    no atomics. It is deterministic, and the sums over atoms use a fixed
    order.
  - The tables ship with FastMACE as package data, byte-identical to
    torch-dftd 0.5.1's `dftd3_params.npz` (sha256 in the identity), with
    torch-dftd's MIT notice. So the native path needs no `[d3]` extra
    (SPEC §4.1, §14). Whether the release may ship them is checked in
    phase 6.
  - Tolerances follow the 3b-T10 runbook's item 1.
  - New: a torch CPU tier (float64) for CPU devices and the CPU suite, which
    also serves as the kernel's oracle.
- **3b-T04, integration:**
  - The energy-only entry is used for MC energy fields and follow copies.
  - `needs_rebuild` is a device flag OR-ed into the fused stage.
    `build()` returns True when addresses changed, and the graphs are then
    recaptured.
  - D3 rebuilds stay on the main stream.
  - MC trial rebuilds cover the D3 list.
- **3b-T05, NPT:**
  - The pressure setter is approved, half-step lag included (documented, and
    the restart round trip covers a changed target).
  - CPU tests use torch-dftd as the D3 term until T03's torch tier merges.
  - The comparison uses **P_tau = 2 ps** on both engines (mass factor 1,
    friction 0.02), recorded as a test setting. The library default and D1's
    protocol keep 20 T_tau.
  - The ASE side runs MACE in float32 (cuEq) with torch-dftd in float64:
    about 4 h instead of 6.5 h, in jobs of 70 min or less.
- **3b-T06, cost:**
  - It sets rule 11's D3 numbers after T02 and T04 merge.
  - If NPT can't share a process, the cost is decomposed: in-process pass
    time plus counted rebuilds times build time.
  - Timing takes about 1.5 h.
- **3b-T07, protocol:**
  - `trajectory_every = 100`, `thermo_every = 10` (unlike main.py's
    10 fs frames).
  - Pressure schedules are reserved and refused, unless T05's setter lands
    first and wiring them is small.
  - `SOURCES.md` covers the NbCl5 CIF, and the D3 counters go into the
    output.
- **3b-T10, acceptance:** the runbook (f2bf00e) is accepted with three
  item 2 changes:
  - P_tau = 2 ps on both engines;
  - ASE with MACE in float32;
  - SE from blocks of at least twice the decorrelation time pooled over the
    seeds, with a jackknife for the V fluctuation.

  The pass rule stays at 3 combined SE, with one extension of 3 more seeds.
  The tool-side additions are written and CPU-tested before the freeze.

## Checkpoints for the owner

1. Before the go messages: D1–D4 above.
2. After T06: the D3 cost against the 10 % target, if it misses (D3).
3. After T10: the acceptance report, and the tag `fastmace-3b`.
