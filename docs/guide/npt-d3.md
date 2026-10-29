# NPT and D3 dispersion

!!! warning "Experimental"
    NPT and D3 carry an "experimental: not yet qualified" line in the run
    report until their qualification studies are recorded.

## NPT

`ensemble="NPT"` in `fm.md` (and `--ensemble NPT` on the command line) is
ASE 3.27's `LangevinBAOAB` barostat on the graph engine:

```python
run = fm.md("NbCl5.cif", model="mh1", head="matpes_r2scan", ensemble="NPT",
            temperature_K=300, pressure_GPa=0.0, cell_mode="hydrostatic",
            steps=200_000, repeat=(3, 3, 3), dispersion="d3bj", output="npt/", seed=1)
```

`cell_mode` is `"hydrostatic"` or `"full"`. For other barostats, pass an
explicit integrator:

| Record | ASE class |
| --- | --- |
| `fm.integrators.LangevinBAOAB(...)` | `LangevinBAOAB` |
| `fm.integrators.MTK(..., cell_mode="hydrostatic" or "full")` | `IsotropicMTKNPT`, `MTKNPT` |
| `fm.integrators.Berendsen(..., pressure_GPa=...)` | `NPTBerendsen`, `Inhomogeneous_NPTBerendsen` (equilibration only, non-canonical) |

**Cell divergence.** A vapour at low pressure expands without bound.
`LangevinBAOAB` stops the run with `CellDivergedError` (restart and metadata
written first) once the volume leaves `[V0/20, 20 V0]` or a cell vector is
longer than 1000 Å. Change the limits with `max_volume_ratio` and
`max_cell_length_A`.

**Stress convention.** The force field returns the potential (virial)
stress as a full 3x3 tensor in ASE's sign convention; the integrator adds the
kinetic part, as ASE's `get_stress(include_ideal_gas=True)` does.

## D3(BJ) dispersion

FastMACE has its own GPU D3(BJ) term. It runs inside the captured step in
MD, in relaxations and in quenches, and needs no extra package.

```python
run = fm.md(..., dispersion="d3bj")
```

`"d3bj"` is shorthand. FastMACE expands it to a complete `fm.D3` record
before planning: Becke-Johnson damping with the parameter set of the head's
training functional (`pbe` for PBE-trained models, `r2scan` for
`matpes_r2scan`), 40 bohr pair and coordination cutoffs, no three-body term,
float64. The expanded record is printed in the report and recorded in
`metadata.json`.

`"d3bj"` is **refused** when:

- the training reference already contains dispersion (MACE-OFF, OMOL-0,
  MH-1's `omol` head), since D3 would count it twice;
- the training functional is unknown (a model loaded from a path or URL).
  Pass an explicit record then:

```python
run = fm.md(..., model=model, dispersion=fm.D3(functional="pbe"))
```

PolarMACE (POLAR-1) with D3 is refused with any record.

An explicit `fm.D3` is used as given; a functional that differs from the
training functional is a report warning. In TOML, a `[dispersion]` table
holds the record's fields.

**When to use it.** The molecular chlorides need D3 in NPT: without it,
NbCl5 swelled by 20 % at 300 K and evaporated at 600 K in our tests.

See also: [Integrators and optimisers](../reference/integrators.md), [Models, D3, PolarMACE and schedules](../reference/models.md).
