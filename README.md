# FastMACE

**The fastest way we know to run one long MACE trajectory on one NVIDIA
GPU, exact to MACE within float32 rounding, checked on your structure.**

FastMACE runs molecular dynamics with pretrained MACE models (MH-1, MP-0,
OMAT-0, MatPES, MACE-POLAR-1, OMOL-0, MACE-OFF). It owns the whole MD loop:
CUDA graphs over each step, exact model rewrites, a cuEquivariance and
OpenEquivariance hybrid backend, and a planner that picks the fastest engine
that passes its checks on your structure.

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, steps=1_000_000, repeat=(3, 3, 3), output="run/", seed=1)
print(run.summary)
```

Documentation: [user guide and API reference](https://fzwingel-ubt.github.io/FastMACE/).
Start with [docs/quickstart.md](docs/quickstart.md), or build the site locally with `mkdocs build --strict`.

## Status (25 October 2026)

Version **1.0.0**. Phases 0 to 5 are accepted: extraction of the research engine,
contracts, safety (gates, fallback, durable restarts, provenance), MC and
the remaining integrators, D3, relaxations and the planner with runtime
adaptation. Phase 6's packaging, documentation, examples and clean-install acceptance checks passed,
including the prebuilt hybrid backend on Python 3.13. See the
[acceptance report](docs/validation/phase-6.md) for qualification limits and known findings.
The design is in [SPEC.md](SPEC.md), progress in
[plans/README.md](plans/README.md).

## Why FastMACE

Measured by us on an RTX 5060 Ti, float32, 216 to 1,944 atom cells
([TorchSim comparison](docs/validation/torchsim-comparison-20261025.md),
[positioning](docs/design/2026-10-positioning.md)):

- 3.8 to 10 times faster per step than TorchSim 0.6.2, and 1.1 to 2 times
  faster than Symmetrix-XL in LAMMPS; 3 to 22 times faster than MACE's ASE
  calculator. MH-1 on 216-atom NaCl: 2.90 ms per 1 fs step. The margin is
  largest on small cells; we claim nothing beyond the measured sizes and
  GPUs.
- **Exact and checked.** Every rewrite is exact to rounding and is checked
  against the original MACE module at setup. An optional gate checks the
  fast path on your own structure and falls back safely if it fails.
- **Species swaps at graph speed.** MC swaps and transmutations are
  transactional events inside the captured graph (experimental).
- **MH-1 and PolarMACE** at graph speed, with upstream bug fixes.
- **D3(BJ) on the GPU**, in MD, relaxations and quenches.
- **Complete restarts**: RNG state, exchanged species, schedule position,
  model hash, plan and versions.
- **`pip install`, no `nvcc`**, no LAMMPS build.

## Use something else when

- **You have many small independent structures** (screening, batched
  relaxations): use a batched engine such as
  [TorchSim](https://github.com/TorchSim/torch-sim) or
  [NVIDIA ALCHEMI](https://github.com/NVIDIA/nvalchemi-toolkit). FastMACE
  runs one structure at a time.
- **Your cell has more than about 100,000 atoms, or you need several GPUs**:
  use MACE's [LAMMPS ML-IAP](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)
  interface or [Symmetrix-XL](https://github.com/amphilagus/symmetrix-xl),
  which decompose the domain and use less memory per atom.
- **Your model is not MACE 0.3.x** (MatterSim, UMA, SevenNet, ORB, NequIP
  ...): TorchSim or ALCHEMI run many model families.
- **You run on AMD GPUs or CPUs**: Symmetrix runs there; FastMACE is
  NVIDIA only.
- **You need LAMMPS's ecosystem** (PLUMED, enhanced sampling, its many
  fixes): use LAMMPS.

## Install

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install cuequivariance-ops-torch-cu13
pip install "fastmace[full]"
fastmace doctor
```

FastMACE 1.0 supports torch 2.14.1+cu130 and Python 3.10 to 3.13 on Linux.
Extras: `[oeq]` (OpenEquivariance hybrid backend), `[polar]` (PolarMACE),
`[d3]` (torch-dftd reference), `[full]` (all three), `[dev]` and `[docs]`.
For development: `pip install -e ".[dev]"`.

## Licence

FastMACE is MIT, see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Model
weights keep their own licences; many MACE foundation models, MH-1 among
them, are under the Academic Software License (non-commercial). See
[docs/licences.md](docs/licences.md).
