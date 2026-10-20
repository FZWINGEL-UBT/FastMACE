# FastMACE

Fast molecular dynamics with pretrained MACE models on NVIDIA GPUs: exact
model rewrites, CUDA graphs and a planner that picks the fastest engine that
passes its checks on your structure.

**Status (20 October 2026):** phases 0, 1a (extraction of the research
engine) and 1b (contracts) are done. Phase 2
(safety: version checks, validation gates, fallback, subprocess isolation,
durable restarts, reproducibility, provenance and GPU CI) is in its final
acceptance. The ResearchModule's `main.py` runs its MD through FastMACE by
default. The design is in [SPEC.md](SPEC.md), and progress is in
[plans/README.md](plans/README.md).

## Install (development)

```bash
pip install -e ".[dev]"
```

Extras: `[oeq]` (OpenEquivariance hybrid backend), `[polar]` (PolarMACE
long-range models), `[d3]` (torch-dftd D3 reference) and `[full]` (all
three). Python 3.10–3.12.

## Licence

MIT, see [LICENSE](LICENSE).
