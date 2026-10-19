# Contracts of phase 1b: public objects and who builds what

Plan task 1b-T01, 19 October 2026. Spec: SPEC §5, §7.1, §7.2, §7.6, §9.2, §9.5,
§10.6, §11.2; owner decisions D1–D8 of `plans/phase-1b.md`. Every later 1b task
builds against the names and signatures here. `tests/cpu/test_contracts.py`,
`tests/cpu/test_import_time.py` (lazy names) and `tests/cpu/test_research_surface.py`
pin them.

## Public names

`import fastmace as fm` imports neither torch nor MACE (nor ASE). Each name is
resolved at first use (`fastmace/__init__.py`, `_PUBLIC`). The records a user builds
before choosing a model stay torch-free when resolved.

| Name | Module | Signature (fixed here) | Filled by |
| --- | --- | --- | --- |
| `fm.load` | `model/loading.py` | 1a's `load(source, head=None, ...)` until T08 makes it public | T08 |
| `fm.ForceField` | `model/forcefield.py` | `ForceField(model, atoms=None, *, precision="float32", rewrites="auto", species=None)`; `evaluate(atoms, stress=False) -> (e, f, s)` | T12 |
| `fm.Simulation` | `simulation.py` | `Simulation(atoms, force_field, *, species=None, integrator, workload=None, output=None, seed, check="skip", allow_naive_cueq=False)`; `plan`, `gate` (properties); `run(steps=None, schedule=None, mc=...)`; `save_restart(path)`; `Simulation.resume(directory, *, allow_plan_change=False, check="skip")`; `set_positions`, `set_momenta` | T17 |
| `fm.Workload` | `planner/workload.py` | §9.2, frozen, validated (complete) | T01 |
| `fm.Output` | `io/output.py` | `Output(directory, trajectory_every=20, thermo_every=20, restart_every=10_000)`, frozen (record complete; writer T06) | T06 |
| `fm.integrators` | `integrators/__init__.py` | `Langevin(*, timestep_fs=0.5, temperature_K, friction_per_fs, fixcm=True)`; `LangevinBAOAB(*, timestep_fs=0.5, temperature_K, friction_per_fs, pressure_GPa=0.0, external_stress_GPa=None, cell_mode="hydrostatic", barostat_tau_fs=None, barostat_mass_factor=1.0, disable_cell_langevin=False, fixcm=True)`; both `.to_config(*, seed) -> IntegratorConfig` (complete) | T05 (implementations) |
| `fm.protocols` | `protocols/` | `ramp`, `hold`, `+`, profiles | T07 |
| `fm.md` | `quick.py` | `md(structure, *, model, head, charge, spin_multiplicity, sha256, trusted, ensemble="NVT", temperature_K, schedule, pressure_GPa=0.0, cell_mode="hydrostatic", timestep_fs=0.5, steps, mc, min_mc_temperature_K, dispersion, repeat=(1, 1, 1), output, seed, check="skip", integrator) -> RunResult`; `RunResult(summary, plan, gate, simulation, metadata, output)` | T20 |
| `fm.calc` | `ase/calculator.py` | `calc(model, *, head=None, **options)`; `FastMACECalculator(model, *, head=None, rewrites="auto", precision="float32")` | T18 |
| `fm.D3` | `longrange/d3.py` | `D3(*, functional, damping="bj", pair_cutoff_bohr=40.0, cn_cutoff_bohr=40.0, three_body=False, smoothing="none", precision="float64", autoang=0.52917726, autoev=27.21138505)` (record complete; `"d3bj"` expansion T20, kernel 3b) | T20 |

Stubs raise `NotImplementedError` naming their task. `Simulation` validates `check`
first. `fm.mc` (phase 3) and `fastmace.ase.Langevin`/`LangevinBAOAB` (T21, not a
top-level name) are not listed.

## Who builds what

```text
fm.load ──> LoadedModel ──> fm.ForceField (pristine model, precision, superset, identity)
                                   │
 atoms, integrator record, Workload, Output, seed, check
                                   │
                                   v
 fm.Simulation ──> MinimalPlanner.plan ──> Plan (frozen, revisions)
      │                    │
      │  check="gate": selfcheck references (before any production engine)
      │                    v
      └──────────> MinimalPlanner.build ──> MACEForceField copy + SimulationState
                                            + Integrator + GraphEngine
                           │
      check="gate": selfcheck production half (captured replay, registry snapshot/restore)
fm.md, fastmace.ase.Langevin, fastmace md ──> build a Simulation; never an engine
fm.calc ──> MinimalPlanner (compiled mode, no workload) ──> MACEForceField.evaluate
```

Only the planner builds force-field copies, integrators and engines (SPEC §5).
`FixedPlanner` stays for `main.py`.

## Config fields added (the only config edit of 1b)

- `PrecisionConfig.float64_energy: bool = True`: §7.3. It has no effect until T03.
  `False` reproduces 1a's energy, for A/B against the research engine only (D2).
- `IntegratorConfig.temperature_K: float | None = None`: the starting target. `None`
  means the caller sets it through `Integrator.set_temperature`, as `main.py` does.
  The public records always set it. D7's default workload reads it.
- `IntegratorConfig.external_stress_GPa`: `None`, a float or six Voigt components
  (xx, yy, zz, yz, xz, xy), ASE `externalstress` sign. `None` means `-pressure_GPa`
  (`effective_external_stress_GPa()`). Both it and a nonzero `pressure_GPa` →
  `ValueError`.
- `IntegratorConfig.disable_cell_langevin: bool = False`, as ASE.

Each has a row in `switch-map.md`. The public records take friction per fs, and
`to_config` converts it to the config's inverse ASE time (`friction_per_fs / units.fs`).
For BAOAB, `T_tau = 1 / friction`, as in `main.py`.

## Runtime-state registry policies (record: `engine/registry.py`)

Each persistent buffer is a `RegisteredBuffer(name, owner, restart, recompute=None,
capture=True, sticky=False)`. `restart` is one of these:

- `save`: in the restart file;
- `reset`: recomputed at the canonical restart point by `recompute` (required);
- `derived`: recomputed from saved state, or overwritten before it is read.

`capture` means it is restored after warm-ups. `sticky` flags are cleared only by a
host decision. Policies for what phase 1a has (`GraphEngine.runtime_state`); T04
registers them and adds anything its version-counter test finds:

| Buffers | Restart | Capture, sticky |
| --- | --- | --- |
| `state.positions`, `velocities`, `cell`, `species`, `masses` | save | capture |
| `state.forces`, `energy`, `stress` (committed step) | save | capture |
| `rng` generator, drawn batch and cursor | save | capture (batch) |
| BAOAB `p_eps`, `force_eps` (move into the state, T05) | save | capture |
| Langevin `x_old`, `rnd_pos`, `rnd_vel`, `xi`, `eta`; BAOAB `xi`, `zeta` | derived (written before read) | capture |
| integrator `sqrt_kT` (target) | derived from the protocol position | capture |
| `neighbours.*` (candidates, edges, shifts, `count`, `reference_positions`, build cell h₀) | reset (`rebuild`) | capture |
| force-field `cell`, `volume`, `rcell` | derived from `state.cell` | capture |
| `polar.*` k-vector set and bounds | reset (rebuild from the cell) | capture |
| `polar.overflow` | reset | capture, **sticky** |
| `engine.flags`, `pipe_flags`, `spec_flags`, `rebuild_needed` | reset (quiesce) | capture, **sticky** (overflow bits) |
| `engine.mid.*`, pipeline floor and next bucket (`_pipe_floor`, `_pipe_edges`) | reset (quiesce) | capture |

## SimulationState fields 1b adds (T05)

T05 added them to `SimulationState`; `state.FIELDS_ADDED_IN_1B` keeps only `mc`:

- `charges` (N,) float64 or `None`; host `total_charge`, `spin_multiplicity`;
- `pbc` (3 bools); `constraints` (always empty in 1b);
- `p_eps`, `force_eps`: device float64, shape () hydrostatic or (3, 3), `None` for
  fixed-cell integrators;
- `protocol`: the T07 schedule position and target;
- `plan_revision: int`; `force_field_identity`: T05's frozen record with a stable
  digest.

`mc` stays for phase 3.

## Where the gate hooks into `Simulation` (D1)

The constructor runs these steps in order:

1. ForceField from a model, if needed;
2. default `Workload` (D7);
3. plan;
4. if `check` requests it, the gate's references, before any production engine
   exists;
5. build;
6. the gate's production half, then `GateFailed` or `GateUnavailable`.

A constructed `Simulation` is ready to step, and `sim.gate` is final. `resume` does
the same with its own `check`. Check modes (`simulation.CHECK_MODES`):

- `"skip"` (default): status `skipped`;
- `"gate"`;
- `"allow_unavailable"`: requests the gate; an unavailable gate is recorded and the
  run goes on.

`fm.md`, `fastmace.ase` and the CLI pass `check` through.

## Module placement (deviations from SPEC §5)

- `simulation.py` holds `Simulation`; §5 has no module for it.
- `planner/workload.py` holds `Workload`; §5 says `WorkloadSpec`.
- `planner/minimal.py` holds the minimal planner (T12), next to `fixed.py`.
- `model/forcefield.py` holds `fm.ForceField`; `forces.py` keeps 1a's `MACEForceField`.
- `reference/integrators.py` holds the BAOAB reference (T09).

## Decisions here that the conductor should check

1. `ForceField(model, atoms=None, ...)`: SPEC §7.1 passes `atoms` positionally, and
   the T12 plan omits it. Optional `atoms` fits both.
2. `Simulation`'s second argument is named `force_field` (a `ForceField` or a model).
   `species=` together with a `ForceField` will be refused, because the `ForceField`
   owns the superset.
3. `seed` is required in `Simulation`. `fm.md`'s default seed is T20's.
4. `mc=...` (Ellipsis) means "keep the attached move set"; `mc=None` detaches it.
5. Friction default: SPEC §7.7 says "Langevin friction 0.02/fs" for `fm.md`, but
   `main.py`'s `--friction 0.02` is in inverse ASE time units, about 0.002/fs. The
   records have no friction default; T20 needs the conductor's answer.
