# OpenEquivariance PR draft: build the JIT host extension from the NVIDIA runtime wheels (no toolkit, no `nvcc`)

**Upstream:** PASSIONLab/OpenEquivariance, base `main` at `dc9979099c65113adcc016977c5c60974f9ddafb`
(27 September 2026, after v0.7.0). **Patch:**
[`0001-wheel-cuda-host-build.patch`](0001-wheel-cuda-host-build.patch): edits
`openequivariance/openequivariance/_torch/extlib/__init__.py`, adds `.../extlib/wheel_cuda.py` and
`tests/wheel_cuda_test.py`. Not opened (owner opens it, decision D5 (b)).

## Title

JIT extension: build against the pip NVIDIA CUDA wheels when no CUDA toolkit is installed

## Description

### Motivation

The torch host extension `libtorch_tp_jit` has only C++ sources; its device kernels are compiled at run
time with NVRTC. Building it needs the CUDA runtime headers and link libraries, **not `nvcc`**. A
PyTorch install from PyPI already ships both in the NVIDIA wheels (`nvidia/cu13/{include,lib}` for
CUDA 13; `nvidia/cuda_runtime`, `cuda_nvrtc`, `cublas`, `cuda_cccl` for CUDA 12). But
`load_jit_extension()` calls `include_paths("cuda")` and `library_paths("cuda")`, which require
`torch.utils.cpp_extension.CUDA_HOME`. On machines without a toolkit (shared HPC workstations, cloud
images, Colab-like VMs, containers built from `python:` images) `CUDA_HOME` is `None`, the build fails,
and MACE silently runs without OEq. Where the precompiled extension does not apply (torch < 2.10,
`OEQ_JIT_EXTENSION=1`, HIP), users currently need a toolkit or a wrapper script that fakes one.

We ran OEq 0.7 on toolkit-less workstations for weeks through such a wrapper (a directory of symbolic
links standing in for `CUDA_HOME`, plus an `nvcc` sentinel that fails loudly), later replaced by
in-process preparation in FastMACE (`fastmace/kernels/_oeq_runtime.py`; findings in
`docs/design/oeq-runtime.md`). The sentinel was **never** invoked: no step of the build needs `nvcc`.
This PR removes the need for any such wrapper.

### Change

- new `extlib/wheel_cuda.py`: `wheel_build_flags()` finds the wheel directories in site-packages and
  returns explicit include directories (with CCCL) and linker arguments that link each library **by
  soname** (`-l:libcudart.so.13`, `-l:libnvrtc.so.13`, `-l:libcublas.so.13`, driver
  `-l:libcuda.so.1`) with an RPATH to the wheel. No `CUDA_HOME` and no unversioned `lib*.so` links are
  needed. The driver directory is searched in the usual places (`/usr/lib/wsl/lib`, multiarch,
  `/usr/lib64`, `/usr/lib64-nvidia`) or taken from `OEQ_CUDA_DRIVER_DIR`.
- `load_jit_extension()` (CUDA branch only): new variable `OEQ_CUDA_FROM_WHEELS` = `auto` (default:
  wheels only when `CUDA_HOME` is `None`, so **existing toolkit builds are unchanged**), `1` (always;
  error if the wheels are missing), `0` (never). With the wheels, the device include path is
  `include_paths("cpu")` plus the wheel's.
- HIP and SYCL branches untouched.

### Why linking by soname matters

On one of our workstations the libpython directory (`/usr/lib/x86_64-linux-gnu`) also holds a system
CUDA 11.5's development links (`libnvrtc.so`, `libcudart.so`). OEq passes `-L<libpython dir>` before the
CUDA directories, so `-lnvrtc` resolved to CUDA 11 and the extension mixed runtimes. `-l:libnvrtc.so.13`
plus `-L<wheel>` cannot pick a different major version.

### Evidence

- Local PC, WSL2, RTX 5060 Ti, torch 2.14.1+cu130, OEq 0.7.0, nvidia-cuda-runtime 13.0.96, g++ 13.3:
  `check_wheel_build.py` (in this folder) builds OEq 0.7.0's host sources with **only** the new flags,
  `CUDA_HOME`/`CUDA_PATH` unset, no `nvcc` on `PATH`: built in 18–20 s, loaded with
  `torch.ops.load_library`, and `ldd` shows `libcudart.so.13`, `libnvrtc.so.13`, `libcublas.so.13`
  (and `libcublasLt.so.13`) from `nvidia/cu13/lib` and `libcuda.so.1` from the driver.
- The equivalent symbolic-link layout (same headers, same libraries) has built and run OEq 0.7.0 with
  MACE on four workstations (RTX A4000, RTX 4000 Ada x2, RTX 5060 Ti) since 8 October 2026; forces of
  the in-process route agree with the wrapper-script route to 3.9e-7 eV/Å, the same scatter as two runs
  of the script itself (`docs/design/oeq-runtime.md`, run table).

### Exactness

Build-system change only; the compiled code is the same. No effect on kernels.

## Test plan

- [x] `tests/wheel_cuda_test.py` (CPU, no GPU): cu13 layout found and flags correct (CCCL include,
  sonames, one `-L` per directory, driver last); missing wheel and missing library return `None`.
  **Run: 3 passed, 1 skipped** (in torch 2.14.1+cu130, Python 3.10).
- [x] build check on a toolkit-less configuration (above), with OEq **0.7.0** sources and the flags
  this patch computes.
- [ ] `OEQ_TEST_WHEEL_BUILD=1 pytest tests/wheel_cuda_test.py` on a GPU host with the **patched**
  package (imports `openequivariance._torch.extlib` in a subprocess without `CUDA_HOME`). Not run: we did
  not install the patched upstream main.
- [ ] `verify_extension_build.yml`: add a job leg on a runner image without the CUDA toolkit.
- [ ] CUDA 12 wheels (`torch==2.x+cu126`): untested; the code path exists.
- [ ] then `tests/conv_test.py` / `batch_test.py` with the wheel-built extension.

## Risks

- **`auto` changes behaviour only where the build failed before** (`CUDA_HOME` is `None`); with a
  toolkit the old path runs. Users with a toolkit **and** wheels keep the toolkit (set `=1` to prefer the
  wheels).
- **Header/runtime skew:** the wheel's headers match the wheel's libraries by construction, which is
  safer than a system toolkit of another version, but a torch built for CUDA 12 next to a stray `cu13`
  wheel would pick cu13 first (layout order). Could be refined by preferring the major version of
  `torch.version.cuda`.
- **`-l:` syntax** is GNU ld / lld / gold; not macOS (not a CUDA target) or MSVC (Windows is not
  supported by the JIT path today).
- **Driver lookup** heuristics may miss unusual installs; `OEQ_CUDA_DRIVER_DIR` overrides.
- Interaction with the existing stale-lock wait (`_wait_for_torch_build_lock`): unchanged.
