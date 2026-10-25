# Upstream PR drafts (plan task 6-T10)

SPEC §6.3 ("Upstream where possible") and plans/phase-6.md owner decision **D5 (b)**: the conductor
drafts these pull requests, **the owner opens them**. Nothing here has been opened, forked or pushed
to an upstream repository.

| Folder | Upstream | Proposed PR | Target |
| --- | --- | --- | --- |
| [`mace-block-linear/`](mace-block-linear/README.md) | ACEsuit/mace | Block and grouped linear: cuEquivariance `Linear` as per-l GEMMs with frozen weights | `main` at `e37c858` (19 October 2026) |
| [`mace-reassociate-first/`](mace-reassociate-first/README.md) | ACEsuit/mace | Reassociated first interaction: per-(receiver, sender species) edge sums before the radial projection | `main` at `e37c858` |
| [`oeq-no-nvcc-build/`](oeq-no-nvcc-build/README.md) | PASSIONLab/OpenEquivariance | JIT host-extension build from the NVIDIA runtime wheels, without a CUDA toolkit or `nvcc` | `main` at `dc99790` (27 September 2026) |

Each folder has the PR title and description (`README.md`) and one self-contained `.patch` made with
`git diff` against the named upstream files. Apply with `git apply <patch>` from the upstream
repository root.

## How the patches were made and what was verified (25 October 2026)

The upstream repositories were not cloned. The files each patch touches were fetched from
`raw.githubusercontent.com` at the commits above into scratch git repositories, edited there, and
diffed. The new modules were then run against the **installed** packages on the local PC (WSL2,
RTX 5060 Ti), which are older than upstream main:

| Patch | Run against | Result |
| --- | --- | --- |
| MACE block linear | `fastmace_cpu` venv: MACE 0.3.16, cuEquivariance 0.13.0, torch 2.14.1+cpu; the new module loaded from the patch, `tests/unit/test_block_linear.py` | 13 passed (with the reassociation tests: 19 passed, `run_mace_tests.py`) |
| MACE reassociation | same venv; `tests/unit/test_reassociate.py` builds small float64 MACE models (Residual, DensityResidual, ResidualNonLinear first blocks) and compares energy and forces | 6 passed |
| OEq wheel build | `tests/wheel_cuda_test.py` in `mace_cueq_latest`: 3 passed, 1 skipped (the end-to-end test needs `OEQ_TEST_WHEEL_BUILD=1`). `check_wheel_build.py` in `mace_cueq_oeq` (OEq 0.7.0, torch 2.14.1+cu130): built OEq 0.7.0's host-extension sources with **only** the patch's flags, `CUDA_HOME`/`CUDA_PATH` unset and no `nvcc` on `PATH`, in 18–20 s; the library loaded and `ldd` resolved `cudart`, `nvrtc`, `cublas` to the `nvidia/cu13` wheel and `libcuda.so.1` to the driver | pass |

**Not verified:**

- the patches applied to and tested inside the upstream trees (MACE main has dropped Python 3.10;
  our environments are Python 3.10, so MACE main's test suite was not run). The MACE patches only add
  new files and do not touch existing ones, so they apply to `e37c858` by construction; the OEq patch
  was diffed against `dc99790`'s `extlib/__init__.py`, which differs from 0.7.0's (SYCL backend), and
  its edited branch was exercised only through `check_wheel_build.py`, not by importing the patched
  `openequivariance` package;
- GPU runs of the upstream-style modules. The speed and accuracy numbers in the READMEs are
  FastMACE's and the research engine's measurements of the same algorithms (cited per row), not of
  these drafts. The drafts are plain PyTorch, without FastMACE's Triton kernels, CUDA graphs or
  shape rules;
- CUDA 12 wheel layouts (`nvidia/cuda_runtime` and siblings) for the OEq patch: covered by code
  paths only, no CUDA 12 environment was built.

`run_mace_tests.py` is the small runner used for the MACE tests (it loads the new modules from the
patch files into an installed MACE 0.3.16).
