# FastMACE

Fast molecular dynamics with pretrained MACE models on NVIDIA GPUs: exact
model rewrites, CUDA graphs and a planner that picks the fastest engine that
passes its checks on your structure.

**Status:** phase 0 (package skeleton). Nothing runs yet. The design is in
[SPEC.md](SPEC.md) and the phase plans are in [plans/](plans/).

## Install (development)

```bash
pip install -e ".[dev]"
```

Extras: `[oeq]` (OpenEquivariance hybrid backend), `[polar]` (PolarMACE
long-range models), `[d3]` (torch-dftd D3 reference) and `[full]` (all
three). Python 3.10–3.12.

## Licence

MIT, see [LICENSE](LICENSE).
