# Models, force fields, dispersion and schedules

This page lists the model label table, the arguments of `fm.load` and
`fm.ForceField`, the `fm.D3` dispersion record, the PolarMACE options in
`PolarConfig`, and the temperature schedules of `fm.protocols`. For advice on
which model to use see [Models and licences](../guide/models.md); for D3 in
practice see [NPT and D3](../guide/npt-d3.md). The generated API pages are
[Models and force fields](../api/model.md), [Dispersion (D3)](../api/d3.md)
and [Protocols and schedules](../api/protocols.md).

## The label table

A **label** names one released checkpoint and, for a multi-head checkpoint,
one head. Each label's file has a fixed SHA-256; a file that does not match
is not loaded. `fastmace models` prints the table with file names and
hashes, the licence texts and the aliases; it reads no file and downloads
nothing.

| Label | Head | Model | Training functional | Licence | File |
| --- | --- | --- | --- | --- | --- |
| `mp0-medium` | — | MACE-MP-0 medium | PBE (PBE+U) | MIT | `2023-12-03-mace-128-L1_epoch-199.model` |
| `mp0-large` | — | MACE-MP-0 large | PBE (PBE+U) | MIT | `MACE_MPtrj_2022.9.model` |
| `mp0b3-medium` | `default` | MACE-MP-0b3 medium | PBE (PBE+U) | MIT | `mace-mp-0b3-medium.model` |
| `mpa0-medium` | `default` | MACE-MPA-0 medium | PBE (PBE+U) | MIT | `mace-mpa-0-medium.model` |
| `omat0-medium` | `default` | MACE-OMAT-0 medium | PBE (PBE+U) | ASL | `mace-omat-0-medium.model` |
| `matpes-pbe` | `default` | MACE-MatPES-PBE-0 | PBE (no U) | ASL | `MACE-matpes-pbe-omat-ft.model` |
| `matpes-r2scan` | `default` | MACE-MatPES-r2SCAN-0 | r2SCAN | ASL | `MACE-matpes-r2scan-omat-ft.model` |
| `mh1-omat` | `omat_pbe` | MACE-MH-1 | PBE (PBE+U), OMat24 | ASL | `mace-mh-1.model` |
| `mh1-mp` | `mp_pbe_refit_add` | MACE-MH-1 | PBE (PBE+U), MPTraj | ASL | `mace-mh-1.model` |
| `mh1-r2scan` | `matpes_r2scan` | MACE-MH-1 | r2SCAN | ASL | `mace-mh-1.model` |
| `polar1-s` | `Default` | MACE-POLAR-1 S (PolarMACE) | unknown | ASL | `MACE-POLAR-1-S.model` |
| `polar1-m` | `Default` | MACE-POLAR-1 M (PolarMACE) | unknown | ASL | `MACE-POLAR-1-M.model` |
| `polar1-l` | `Default` | MACE-POLAR-1 L (PolarMACE) | unknown | ASL | `MACE-POLAR-1-L.model` |
| `omol0-xl` | `omol` | MACE-OMOL-0 extra large | ωB97M-V (includes VV10 dispersion) | ASL | `MACE-omol-0-extra-large-1024.model` |
| `off23-small` | — | MACE-OFF23 small | ωB97M-D3(BJ) | ASL | `MACE-OFF23_small.model` |
| `off23-medium` | — | MACE-OFF23 medium | ωB97M-D3(BJ) | ASL | `MACE-OFF23_medium.model` |
| `off23-large` | — | MACE-OFF23 large | ωB97M-D3(BJ) | ASL | `MACE-OFF23_large.model` |
| `off23b-medium` | — | MACE-OFF23b medium | ωB97M-D3(BJ) | ASL | `MACE-OFF23b_medium.model` |
| `off24-medium` | — | MACE-OFF24 medium | ωB97M-D3(BJ) | ASL | `MACE-OFF24_medium.model` |

"—" means a single-head checkpoint whose head has no name. ASL is the
Academic Software License (non-commercial use only); MIT is the MIT License.
The weights' licence binds you independently of FastMACE's own. FastMACE
shows each checkpoint's licence once per process as a `ModelLicenceWarning`.

**Aliases.** `mh1` (the MACE-MH-1 checkpoint) and `omol0` (MACE-OMOL-0) take
`head=`. The checkpoint names (`mh-1`, `omol-0-xl`, and so on) work the same
way. A multi-head checkpoint without `head=` raises `ValueError` listing its
heads; there is no default head.

**MACE-MH-1 heads.**

| Head | Training data and functional | Own label |
| --- | --- | --- |
| `matpes_r2scan` | MatPES, r2SCAN | `mh1-r2scan` |
| `mp_pbe_refit_add` | MPTraj, PBE/PBE+U | `mh1-mp` |
| `omat_pbe` | OMat24, PBE/PBE+U | `mh1-omat` |
| `oc20_usemppbe` | OC20 subset, PBE | — |
| `spice_wB97M` | SPICE, ωB97M-D3(BJ) | — |
| `omol` | OMol25 subset, ωB97M-V | — |

The training functional decides what `dispersion="d3bj"` expands to (see
[D3](#fmd3)).

**Known issues.** The plan and the report warn, without refusing, when a
model meets a structure it is known to handle badly:

| Model | Condition | Warning |
| --- | --- | --- |
| `polar1-s` | structure contains Na, Nb and O | Fukui charge runaway on NaNbO3 |
| `polar1-s`, `polar1-m`, `polar1-l` | Na, Ta and O | huge forces and pressures on NaTaO3 |
| `omol0-xl` | Na and Cl | outside its training domain; collapses in NaCl MD |
| `omol0-xl` | O in an inorganic structure | outside its training domain on dense oxides |
| `off23-small`, `off23-medium`, `off23-large`, `off23b-medium` | variable cell | relaxes molecular crystals to 5.5-9.2 % below the experimental volume; OFF24-medium is closest |
| `off23-large` | 2,000 atoms or more | may need more than 16 GB in some engines |

## `fm.load`

```python
model = fm.load("mh1", head="matpes_r2scan")
model = fm.load("/data/my.model", sha256="9f65...")
```

`fm.load` returns the checkpoint as a **pristine model**: MACE's e3nn
modules in float64 on the CPU. Each plan converts its own copy to the
chosen backend, precision and device; the pristine model is never changed.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `source` | label, alias, checkpoint name, path or URL | required | The checkpoint to load. | First positional argument. |
| `head` | head name | `None` | The head of a multi-head checkpoint. | A label fixes its head; a different `head=` raises `ValueError`. |
| `sha256` | hex string | `None` | The expected SHA-256 of the file. | Must match the file. Lets a path or URL load. |
| `trusted` | bool | `False` | Load a file without a stated hash. | Also loads a label's file whose hash differs from the table. |
| `download` | bool | `False` | Fetch a missing file into the models cache. | The hash is checked before the file is kept. |
| `charge` | int | `None` | Total charge for a model with a total-charge embedding (MACE-OMOL-0). | OMOL-0's range is −100 to 100. Raises `ValueError` for a model without that embedding. |
| `spin_multiplicity` | int | `None` | Spin multiplicity for a model with a total-spin embedding (MACE-OMOL-0). | Raises `ValueError` for a model without that embedding. |
| `checkpoint` | path | `None` | An explicit file for a label. | The label's hash still applies. |

**Where a label's file is found**, in order:

1. `checkpoint=`, if given;
2. `$FASTMACE_MODELS/<file>`;
3. `<cache root>/models/<file>`, where the cache root is `$FASTMACE_CACHE`,
   else the user cache directory (`~/.cache/fastmace` on Linux);
4. with `download=True`, the label's official release URL, downloaded into
   `<cache root>/models/`.

Otherwise `FileNotFoundError` names the places searched.

**Trust.** MACE checkpoints are Python pickles, which run code when they are
loaded. The SHA-256 is computed before anything is unpickled:

| Source | Loads when |
| --- | --- |
| A label | the file's hash matches the table, or `trusted=True` |
| A path | `sha256=` matches, or `trusted=True` |
| A URL | `sha256=` is given (checked on download) or `trusted=True`; refused before any download otherwise. The URL needs `download=True` unless the file is already in the cache. |

`provenance()` of the result records the trust decision, the hash, the
source and the licence. FastMACE never ships or redistributes weights.

## `fm.ForceField`

```python
ff = fm.ForceField(model, atoms, precision="float32", species=["Na", "Nb", "Cl", "O"],
                   dispersion="d3bj")
```

The force field the user holds. It keeps the pristine model, the force
precision, the species superset and the dispersion record. It builds nothing
itself; each plan builds its own execution copy. `fm.md` and `Simulation`
build one for you when you pass a model.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `model` | an `fm.load` result, or a label | required | The model. | A path or URL needs `fm.load` first, for the trust decision. |
| `atoms` | ASE `Atoms`, or `None` | `None` | Its species are the default superset. | Without it, the first structure used fixes the superset. |
| `precision` | `"float32"`, `"float64"` | `"float32"` | Precision of the force evaluation. | Integration is always float64, and the energy is summed in float64. |
| `rewrites` | `"auto"` or a `RewriteConfig` | `"auto"` | Which exact rewrites the planner may apply. | Resolved per plan; see [The planner](../guide/planner.md). |
| `species` | symbols or atomic numbers | `None` | The species superset: every species the run may contain. | May be larger than the structure's (needed for MC). A species the model has no embedding for raises `ModelElementError`. |
| `dispersion` | `None`, `"d3bj"`, or an `fm.D3` | `None` | Add a D3 term to the model. | See [D3](#fmd3). Refused for PolarMACE. |

`ff.evaluate(atoms, stress=False)` returns `(energy, forces, stress)` as host
copies (eV, eV/Å, eV/Å³), through a one-off plan; the structure must be
periodic in all three directions. `ff.dispersion_warnings` lists warnings of
the dispersion record, for example a functional that differs from the
training one.

## `fm.D3`

A complete D3 dispersion correction with Becke-Johnson damping, computed on
the GPU inside the captured step. "D3(BJ)" alone does not identify a
correction, so every choice is a field, and the whole record is part of the
force field's identity (and of the restart check).

```python
fm.D3(functional="pbe")
fm.D3(functional="custom", s6=1.0, s8=0.8, a1=0.45, a2=5.0)
```

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `functional` | `"pbe"`, `"r2scan"`, or any name with explicit s6, s8, a1, a2 | required | The Becke-Johnson parameter set. | Must match the model's training functional; a mismatch is a warning. |
| `damping` | `"bj"` | `"bj"` | Becke-Johnson damping. | The only damping. |
| `pair_cutoff_bohr` | bohr, > 0 | 40.0 | Cutoff of the pair terms. | 40 bohr = 21.167 Å. As `mace_mp` and `torch-dftd`. |
| `cn_cutoff_bohr` | bohr, > 0 | 40.0 | Cutoff of the coordination numbers. | |
| `three_body` | bool | `False` | The Axilrod-Teller-Muto three-body term. | `True` is refused: it is not implemented. |
| `smoothing` | `"none"` | `"none"` | Cutoff smoothing. | None is applied. |
| `precision` | `"float64"` | `"float64"` | Precision of the coordination-number, energy and virial reductions. | The pair terms run in the force precision. |
| `autoang` | Å per bohr | 0.52917726 | Length unit constant. | `torch-dftd`'s value; differs from ASE's `Bohr` by about 1e-7 relative. |
| `autoev` | eV per hartree | 27.21138505 | Energy unit constant. | `torch-dftd`'s value. |
| `s6` | > 0 | from `functional` | Scale of the C6 term. | Give all four of s6, s8, a1, a2 or none. |
| `s8` | ≥ 0 | from `functional` | Scale of the C8 term. | |
| `a1` | ≥ 0 | from `functional` | BJ damping parameter. | |
| `a2` | bohr, > 0 | from `functional` | BJ damping parameter. | |

Built-in parameter sets (`torch-dftd` 0.5.1):

| `functional` | s6 | s8 | a1 | a2 |
| --- | --- | --- | --- | --- |
| `pbe` | 1.0 | 0.7875 | 0.4289 | 4.4407 |
| `r2scan` | 1.0 | 0.7898 | 0.4948 | 5.7308 |

**`dispersion="d3bj"`** expands to `fm.D3(functional=...)` with the head's
training functional from the label table: `pbe` for the PBE-trained models
and heads, `r2scan` for `matpes-r2scan` and `mh1-r2scan`. It raises
`ValueError` when:

- the training functional already contains dispersion: ωB97M-D3(BJ)
  (MACE-OFF, MH-1's `spice_wB97M`) or ωB97M-V (MACE-OMOL-0, MH-1's `omol`),
  since D3 would count it twice;
- the functional is unknown: MACE-POLAR-1, or any model loaded from a path
  or URL. Pass an explicit `fm.D3` then.

The expanded record is printed in the report and recorded in
`metadata.json`. D3 is experimental: the report carries a "D3: experimental:
not yet qualified" line.

## PolarMACE

MACE-POLAR-1 (`polar1-s`, `polar1-m`, `polar1-l`) adds long-range
electrostatics with charges from Fukui equilibration. It needs the `[polar]`
extra (graph-longrange 0.4.0). FastMACE replaces the parts of its long-range
code that synchronise with the host, so the force call can be captured in a
CUDA graph. The k-vectors are a superset fixed when the force field is
built; the k-vectors outside the reciprocal-space cutoff of the current cell
contribute exactly zero.

- The stress includes the long-range energy's dependence on the cell. MACE
  0.3.16 omits it; FastMACE has no switch back to the uncorrected stress.
- The structure's total charge is 0 and its spin multiplicity 1.
- Refused: D3 with PolarMACE; a cell filter in a relaxation (relax at fixed
  cell); the compiled plain cuEquivariance backend, which returns zero forces
  for PolarMACE. The hybrid backend is exact.
- PolarMACE NPT stress has an extra "experimental" report line: it was
  validated against finite differences on NaCl and NbCl5 3×3×3 only.

### `PolarConfig`

`fastmace.config.PolarConfig` holds PolarMACE's options; it has no effect on
other models. The defaults reproduce the released model. The two stabilising
options are MACE pull request 1762, open upstream; they change the model, so
they are opt-in only. Pass the record to the force field:

```python
from fastmace.config import PolarConfig

model = fm.load("polar1-s")
ff = fm.ForceField(model, atoms, polar=PolarConfig(fukui_regularisation=0.1, dipole_cap=(0.5, 2.0)))
sim = fm.Simulation(atoms, ff, integrator=..., seed=1)
```

!!! warning "Not tested"
    The PR 1762 options are wired through the planner, the gate's float64
    reference, `simulation.json` and resume, and they are part of the
    force-field identity. No PolarMACE run with them has been validated.
    The run report carries a "not tested" warning whenever they are set.

A non-default `PolarConfig` on a model that is not PolarMACE raises
`ValueError`. `fm.md` and the TOML files take no `PolarConfig` and run the
defaults.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `fukui_regularisation` | 0 ≤ R < 1 | 0.0 | Regularised Fukui normalisation: the weights are normalised by s / (s² + (R·Σ\|f\|)²). | 0 is the released model. |
| `dipole_cap` | (onset, ceiling) in e·Å, 0 < onset < ceiling, or `None` | `None` | Smooth cap on each atom's charge dipole. | `None` is the released model. |
| `kspace_headroom` | relative, ≥ 0, or `"auto"` | `"auto"` | How far the cell may deform before the fixed k-vector superset no longer covers the cutoff. | `"auto"`: 0.1 with a variable cell, 0 otherwise. A cell beyond it sets an overflow flag. |

## Schedules and profiles

`fm.protocols` builds temperature schedules. A schedule is independent of
the timestep: its segments are in ps. At a given timestep each segment is
rounded to whole steps (half up), and the run report records the rounded
durations and rates.

```python
anneal = fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3)
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", schedule=anneal, output="anneal/")
sim.run(schedule=fm.protocols.hold(300, ps=10))
```

### Segments

| Builder | Arguments | What it does | Notes |
| --- | --- | --- | --- |
| `ramp(start_K, end_K, *, ps=None, rate_K_per_ps=None)` | K, K; ps or K/ps | Linear ramp from `start_K` to `end_K`. | Exactly one of `ps` and `rate_K_per_ps`. The rate is signed and must point from start to end. |
| `hold(temperature_K, *, ps)` | K; ps | Constant target for `ps`. | |
| `a + b` | two schedules | Concatenate. | The second must start where the first ends: a temperature jump raises `ValueError`. |

Temperatures must be finite and at least 0 K; durations positive. A segment
that rounds to 0 steps raises `ValueError`.

**Target per step.** Step n of a segment uses the target at the step's end
point. On a ramp the target changes every step; on a hold it never changes.
The target reaches the thermostat before the step; it is not an event.

**Steps.** `fm.md(schedule=...)` without `steps=` runs the whole schedule.
`steps=` beyond what remains raises before any step. The plan's skin is
sized for the schedule's highest target, so `Simulation.run` refuses a
schedule above `Workload.max_temperature_K`.

### Profiles

`fm.protocols.PROFILES` holds four named profiles: a schedule with output
intervals and, for `anneal`, an initial relaxation and a quench.

| Profile | Schedule | Output interval (steps) | Initial relaxation | Quench |
| --- | --- | --- | --- | --- |
| `slow` | 0 → 400 K in 30 ps, 400 → 0 K in 240 ps | 100 | none | none |
| `fast` | 0 → 400 K in 3 ps, 400 → 0 K in 24 ps | 100 | none | none |
| `ultrafast` | 0 → 400 K in 1 ps, 400 → 0 K in 8 ps | 100 | none | none |
| `anneal` | 0 → 1000 K in 20 ps, 1000 → 250 K in 250 ps (3 K/ps) | 20 | fixed cell, fmax 0.01 eV/Å, `gpu-LBFGS` | 2 frames per 100 K bin from 500 K, fmax 0.01 eV/Å, `gpu-LBFGS`, at most 5000 steps |

In Python, pass the schedule: `fm.md(..., schedule=fm.protocols.PROFILES["anneal"].schedule)`.
In a TOML file for `fastmace run`, `schedule = "anneal"` names the profile,
or `[[schedule]]` tables give segments:

```toml
[[schedule]]
kind = "ramp"
start_K = 0.0
end_K = 1000.0
ps = 2.0

[[schedule]]
kind = "hold"
temperature_K = 1000.0
ps = 5.0
```

A ramp table has `start_K`, `end_K` and one of `ps` and `rate_K_per_ps`; a
hold table has `temperature_K` and `ps`. Unknown keys raise `ValueError`.

### Profile fields

`fm.protocols.load_profile(path)` reads a profile from a `.toml`, `.yaml` or
`.yml` file (YAML through `safe_load` only). A profile has `name`, `segments`
and these optional fields:

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `sample_interval` | steps, int ≥ 1 | 100 | Thermo sample interval. | |
| `trajectory_interval` | steps, int ≥ 1 | 100 | Trajectory frame interval. | |
| `relax_fmax` | eV/Å, ≥ 0 | 0.0 | Initial fixed-cell relaxation to this fmax. | 0 means none. |
| `relax_optimizer` | `gpu-LBFGS`, `gpu-FIRE`, `ase-LBFGS`, `ase-BFGS`, `LBFGS`, `BFGS` | `gpu-LBFGS` | Optimiser of the initial relaxation. | Bare `LBFGS` and `BFGS` mean ASE's. |
| `quench` | table with `min_K`, `bin_K`, `per_bin`, `fmax`, `optimizer`, `max_steps` | none | Quench frames: `per_bin` frames per `bin_K` bin of the target from `min_K`, relaxed to `fmax` eV/Å with `optimizer`, at most `max_steps` steps. | All six keys are required. Each frame is relaxed at its own cell, with the cell fixed. |
| `ensemble` | `"NVT"`, `"NPT"` | `"NVT"` | As `fm.md`. | |
| `pressure_GPa` | GPa | none (NVT); 0.0 (NPT) | NPT's constant hydrostatic target. | Pressure schedules are refused. |
| `dispersion` | `"none"`, `"d3bj"` | `"none"` | As `fm.md`. | |
