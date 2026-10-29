# Models and licences

## Labels

`fm.load`, `fm.md`, `fm.calc` and the CLI take a model as a **label**, a
local **path** or a **URL**. List the labels with:

```bash
fastmace models
```

| Label | Head | Licence | Kind |
| --- | --- | --- | --- |
| `mp0-medium`, `mp0-large`, `mp0b3-medium`, `mpa0-medium` | | MIT | MACE-MP foundation models (PBE) |
| `omat0-medium`, `matpes-pbe`, `matpes-r2scan` | | ASL | OMAT and MatPES models |
| `mh1-omat`, `mh1-mp`, `mh1-r2scan` | fixed by the label | ASL | MACE-MH-1 with one head |
| `mh1` | `head=` required | ASL | MACE-MH-1, any of its six heads |
| `polar1-s`, `polar1-m`, `polar1-l` | | ASL | MACE-POLAR-1 (needs `[polar]`) |
| `omol0-xl` (`omol0`) | `omol` | ASL | MACE-OMOL-0, molecules; takes `charge=` and `spin_multiplicity=` |
| `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium` | | ASL | MACE-OFF, organic molecules and crystals |

`fastmace models` prints the file names and SHA-256 of each checkpoint.

## Heads

A multi-head checkpoint without `head=` raises `ValueError` listing its
heads. There is no default head. MH-1's heads are `matpes_r2scan`,
`mp_pbe_refit_add`, `spice_wB97M`, `oc20_usemppbe`, `omol` and `omat_pbe`.

## Where checkpoints come from

`fm.load(label)` looks for the file in `$FASTMACE_MODELS`, then in
`<cache root>/models`. With `download=True` it fetches the file from its
official URL and checks the SHA-256 before keeping it. FastMACE never ships
or redistributes model weights.

A file outside the label table needs `sha256=` (its expected hash) or
`trusted=True`, because loading a checkpoint runs pickled code.

## Licences

The weights' licence binds you independently of FastMACE's MIT licence.
Several foundation models, MACE-MH-1 among them, are under the **Academic
Software License** (ASL, non-commercial use only). FastMACE shows each
model's licence once per process as a `ModelLicenceWarning`, and the run
report has a `licence:` line read from the label table.

FastMACE's own licence and its dependencies' licences are on the
[Licences](../licences.md) page.

## Which model for what

- Inorganic crystals: MH-1 (`mh1-r2scan` or `mh1` with a head), OMAT-0 or
  MP-0. MH-1 runs at its fastest in FastMACE through the MH-1 rewrites.
- Long-range electrostatics: MACE-POLAR-1. POLAR-1-S showed a Fukui charge
  runaway on NaNbO3, and all POLAR sizes failed on NaTaO3 in our tests.
- Molecules: OMOL-0 (charge and spin) or MACE-OFF (organic elements only:
  H, C, N, O, F, P, S, Cl, Br, I).
- Dispersion: see [NPT and D3](npt-d3.md). MACE-OFF, OMOL-0 and MH-1's
  `omol` head already contain dispersion, so `dispersion="d3bj"` is refused
  for them.

See also: [Models, D3, PolarMACE and schedules](../reference/models.md), [Environment and cache](../reference/environment.md).
