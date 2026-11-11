# A100 receiver caps for host-state calls (B3, 11 November 2026)

Plan B3 of the tiled round 2: do larger receiver caps pay on the A100 for the host-state and wavefront calls?
OMAT-0 medium, SrTiO3, `--mode auto`, aliased host state, one A100-SXM4-80GB on Alex.

## Speed (Alex 4984555, source `f8ea447e`, `tools/bench/tiled_ab.py --b-receivers`, same process, both orders)

Side a R 16,384 fixed, side b the larger cap fixed; each side its own `TiledForceField`. b/a (median):

| Call | Atoms | R 32,768 (ab / ba) | R 65,536 (ab / ba) |
| --- | ---: | --- | --- |
| host state | 975,560 (58^3) | 0.886 / 0.893 | 0.549 / 0.549 |
| wavefront | 975,560 | 0.675 / 0.671 | 0.410 / 0.409 |
| host state | 20,480,000 (160^3) | 0.732 / 0.724 | 0.597 / 0.599 |
| wavefront | 20,480,000 | 0.722 / 0.723 | 0.573 / 0.583 |

Energies within 1e-5 meV/atom, forces within 5e-6 eV/A (the tiles' order). Records: `rcap-alex-4984555/`.

## Memory (Alex 4985398, source `66afc6a7`, `tiled_capacity.py`, one process per cap, 160^3)

| Cap | Device total peak | Host peak |
| --- | ---: | ---: |
| 8,192 (the default budget's first call) | 2.83 GiB | 24.3 GB |
| 16,384 | 3.33 GiB | 28.0 GB |
| 32,768 | 4.20 GiB | 31.9 GB |
| 65,536 | 6.11 GiB | 43.3 GB |

With `TILE_LARGE_CAPS` the default budget took 65,536. The device fits every cap; the host grows about 0.33 MB per
receiver. The device budget alone would take 65,536 at the 397.5M record, whose host stood at 229 of the cgroup's
231.5 GB: a larger cap would lower the largest system that fits. Hence the size rule `TILE_LARGE_CAP_HOST_BYTES`: a
large cap only where the host keeps 0.35 MB per added receiver (plus 10 %) above `list_reuse_floor` (20 GiB). Records:
`capspeak-alex-4985398/`.

## Adopted

`tiled.TILE_LARGE_CAPS` None: on for `TILE_LARGE_CAP_GPUS` (the A100 family, decision 66), for host-state calls whose
`budget_receivers` were left at their default; an explicit budget, the device state and other GPUs are unchanged.
