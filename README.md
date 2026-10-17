# FastMACE

Fast molecular dynamics with pretrained MACE models on NVIDIA GPUs: exact
model rewrites, CUDA graphs and a planner that picks the fastest engine that
passes its checks on your structure.

**Status (18 October 2026):** phase 1a, extraction of the research engine:
14 of 20 tasks are merged. The model, kernels, rewrites, neighbour list,
PolarMACE path, output and validation tools exist and match the research
engine. The force field, integrators, graph engine and planner are not
written yet, so there is no MD run yet. The design is in [SPEC.md](SPEC.md),
and progress is in [plans/README.md](plans/README.md).

## Install (development)

```bash
pip install -e ".[dev]"
```

Extras: `[oeq]` (OpenEquivariance hybrid backend), `[polar]` (PolarMACE
long-range models), `[d3]` (torch-dftd D3 reference) and `[full]` (all
three). Python 3.10–3.12.

## Licence

MIT, see [LICENSE](LICENSE).
