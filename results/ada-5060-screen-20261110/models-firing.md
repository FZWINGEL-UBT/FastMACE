# Which Ada-set switches fire for POLAR-1 and MACE-OFF (code reading, 11 November 2026)

RTX 4000 Ada, float32, default plan at edfc0115 / 7219cac7. The A/B JSONs record only `RewriteConfig` fields
(`plan.describe`), not module constants, so this table comes from the code paths.

| Switch | polar1-m | polar1-l | off23-medium | Why |
| --- | --- | --- | --- | --- |
| spline `PACKED8` | – | – | – | read only with a spline attached (`attach_generic_wprime`, OMAT-0 class); POLAR's `attach_wprime` has none |
| `SORT_BY_LENGTH` | fires | fires | fires | neighbour list, every plan with `sort_receivers`; the model off-list is A100-only (OFF23-small, -large) |
| `SC_GENERATED`, `SC_FACTORED` | – | – | – | need `compose_skip` (generic density interaction) and tf32x3 routing; tf32x3 is on the Ada for OMAT-0 medium only |
| w′ `BUTTERFLY` | fires | fires (if `butterfly_supported`) | – | POLAR gets w′ on the Ada (`POLAR_DECISION_66_WPRIME_GPUS`); OFF23-medium gets no w′ |
| w′ `STAGE_DOUT` | fires | fires (if the row fits 48 KB) | – | the polar1-m off-list was A100-only before this screen |
| w′ `STAGE_SH` | fires | fires | – | float32, 32-lane cohorts |
| w′ `STAGE_EDGES` | – | – | – | needs the spline (`self.spline is not None`) |
| tabled `POWER_BACKWARD` | – | – | – | generic reassociated first interaction only; OFF's stock first block and MH-1/POLAR's tabled path pass no `pcoef` |

Hence OFF23-medium's A/B (1.0002 / 0.9992) measures `SORT_BY_LENGTH` alone, and POLAR-1-M's leave-one-out
(`jobs/ada_polar_loo.sh`) has four live entries (`no-sbl`, `no-bfly`, `no-sdout`, `no-ssh`); the other five are
controls that should read 1.000 within noise.
