# Prebuilt OEq host-extension wheels: `fastmace-oeq-ext` (plan task 6-T04)

Owner decisions (plans/phase-6.md): **D3 (a)** torch 2.14.1+cu130 only; **D4 (a)** prebuilt wheels on PyPI as a
separate package `fastmace-oeq-ext`, Linux x86_64, CPython 3.10 to 3.13.

## What it is

OpenEquivariance 0.7.0 builds a small C++ host extension (`libtorch_tp_jit.so`, pybind11 plus `TORCH_LIBRARY`
ops) with `torch.utils.cpp_extension.load` at its first import, which needs g++ and ninja
([oeq-runtime.md](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/docs/design/oeq-runtime.md)). Its device kernels are compiled with NVRTC at run time, so
the host library is the only compiled part. `fastmace-oeq-ext` ships that library, built from OEq's
unmodified sources against torch 2.14.1+cu130, one wheel per CPython:

| Wheel | Tag | Contents |
| --- | --- | --- |
| `fastmace_oeq_ext-0.7.0.post1-cp310-cp310-manylinux_2_28_x86_64.whl` | cp310 | `fastmace_oeq_ext/{__init__.py, libtorch_tp_jit.so, manifest.json, LICENSE.openequivariance, BUILD_COMPILER.txt}` |
| `...-cp311-cp311-...` / `...-cp312-cp312-...` | cp311 / cp312 | the same |
| `...-cp313-cp313-...` | cp313 | the same; Python 3.13 support added by 6-T15 |

`dependencies = ["torch==2.14.1", "openequivariance==0.7.0"]`. Licence `MIT AND BSD-3-Clause`: the packaging
is FastMACE's MIT, the library OEq's BSD-3 (its licence ships in the wheel, as BSD-3 clause 2 requires for
binaries). The version `0.7.0.postN` follows OEq's; `N` counts rebuilds.

The library links `libc10/libtorch*/libtorch_python` (found through `$ORIGIN/../torch/lib`) and
`libcudart/libnvrtc/libcublas.so.13` (`$ORIGIN/../nvidia/cu13/lib`, the wheels torch already loads), and
`libcuda.so.1` from the driver. It does not link libpython (OEq's JIT build does, for its AOTI path only).

## How FastMACE picks it

`fastmace.kernels._oeq_runtime.prebuilt_extension()` reads `fastmace_oeq_ext/manifest.json` without
importing anything and accepts it only if torch (with its `+cu130` tag), OpenEquivariance and the CPython
tag match exactly, the platform is Linux x86_64 and the library exists. Then `prepare_in_process`
sets up the stand-in CUDA home as before (links only), and wraps `torch.utils.cpp_extension.load` so
that OEq's `load_jit_extension("libtorch_tp_jit", ...)` gets the prebuilt module; nothing is compiled
and no build cache is written. Otherwise it builds with the host toolchain, as before, and records why
the wheel was not used (`extension["prebuilt_skipped"]`).

- `FASTMACE_OEQ_PREBUILT=0`: never use the wheel. `=1`: refuse to build (the clean-install check uses it).
- `fastmace doctor` prints `host extension: prebuilt (fastmace-oeq-ext 0.7.0.post1, <path>); no compiler
  needed`, or `built with the host toolchain ...; prebuilt not used: <reason>`, and only then requires
  g++ and ninja.
- The run's patch record `oeq_runtime` (provenance, `metadata.json`) ends with `host extension: prebuilt
  (<path>)` or `host extension: built|reused|raced|in-process`.

`pip install "fastmace[oeq]"` pulls `fastmace-oeq-ext` with the marker
`sys_platform == 'linux' and platform_machine == 'x86_64' and python_version < '3.14'`. On another torch
pip refuses to resolve (`torch==2.14.1` pin); install `openequivariance` alone to use the toolchain build.

## Building

```bash
tools/release/oeq_wheels/build.sh ~/tmp/6-T04/wheels    # WSL or Linux; docker or Docker Desktop's docker.exe
```

`build.sh` streams this directory into `quay.io/pypa/manylinux_2_28_x86_64` (gcc-toolset 14) and runs
`build_in_container.sh`: pinned CUDA 13 header/library wheels (runtime 13.0.96, nvrtc 13.0.88, crt 13.0.88,
cccl 13.0.85, cublas 13.1.1.3, as torch 2.14.1+cu130 requires), torch 2.14.1+cu130 and pybind11 3.1.0 per
Python, the two OEq sources compiled with the flags of OEq's own JIT build, a stub `libcuda.so` for the link,
`strip`, then a `py3-none-any` wheel retagged `cpXY-cpXY-manylinux_2_28_x86_64`. Scratch lives in the docker
volume `fastmace-oeq-ext-work`, wheels come out through `fastmace-oeq-ext-out`; nothing touches the repository.
Docker Desktop's WSL integration was off on the PC, so `DOCKER=docker.exe` (the Windows client) was used.

Nothing is uploaded. Publishing is the owner's step: `twine check` then `twine upload` of the three wheels.

## Clean-install check

`tools/release/clean_install.sh WHEEL_DIR [PYTHON]` makes a fresh uv venv under `~/tmp/6-T04`, installs
torch 2.14.1+cu130, `cuequivariance-ops-torch-cu13` and `fastmace[oeq]` from the checkout with the local
wheels (`--find-links`), then shadows g++, c++, ninja and nvcc with sentinels that log and fail, sets
`FASTMACE_OEQ_PREBUILT=1`, runs `fastmace doctor` and a 20-step NaCl (64 atoms, MH-1) NVT smoke, and
requires the hybrid backend, a `prebuilt` host extension and an empty sentinel log. gcc (C) stays on PATH:
Triton compiles its CUDA launcher stub with a C compiler, a separate requirement (6-T03).
Run it under the local reservation: `python3 tools/ci/local_gpu.py run --purpose ... -- bash tools/release/clean_install.sh DIR`.

## Results

See the "6-T04" section of [docs/validation/6-T04.md](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/docs/validation/6-T04.md).
