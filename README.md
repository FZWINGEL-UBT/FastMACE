# FastMACE

FastMACE runs long molecular-dynamics trajectories with pretrained MACE
models (MH-1, MP-0, OMAT-0, MatPES, MACE-POLAR-1, OMOL-0, MACE-OFF) on one
NVIDIA GPU, exact to MACE within float32 rounding and checked on your
structure. It owns the whole MD loop: CUDA graphs over each step, exact
model rewrites, a cuEquivariance/OpenEquivariance hybrid backend, and a
planner that picks the fastest engine that passes its checks.

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, steps=1_000_000, repeat=(3, 3, 3), output="run/", seed=1)
print(run.summary)
```

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

FastMACE runs one structure at a time on one GPU. For batched screening,
very large cells, other model families or other hardware, see
[Use something else when](https://fzwingel-ubt.github.io/FastMACE/#use-something-else-when).

## Install

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install cuequivariance-ops-torch-cu13
pip install "fastmace[full]"
fastmace doctor
```

FastMACE 1.0 supports torch 2.14.1+cu130 and Python 3.10 to 3.13 on Linux.
See [docs/install.md](docs/install.md) for the extras and the supported
stack.

## Documentation

The [documentation site](https://fzwingel-ubt.github.io/FastMACE/) has the
user guide, the option reference and the API reference. Start with
[docs/quickstart.md](docs/quickstart.md). Runnable scripts are in
[examples/](examples/README.md). To build the site locally:
`pip install -e ".[docs]"`, then `mkdocs build --strict`.

Version 1.0.0; see [CHANGELOG.md](CHANGELOG.md).

## Licence

FastMACE is MIT, see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Model
weights keep their own licences; many MACE foundation models, MH-1 among
them, are under the Academic Software License (non-commercial). See
[docs/licences.md](docs/licences.md).
