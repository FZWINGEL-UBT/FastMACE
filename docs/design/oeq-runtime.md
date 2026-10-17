# OEq runtime without the launcher script (plan task 1a-T00)

Spike, 17 October 2026. Question from [plans/phase-1a.md](../../plans/phase-1a.md)
task 1a-T00 and SPEC §3.2 "Paths and environment" and §14 "No nvcc": which
settings of ResearchModule's `oeq_no_nvcc.sh` must exist before the process
starts, and which can FastMACE set in-process before OpenEquivariance (OEq) is
imported?

**Answer: none of the OEq settings has to exist before the process starts.**
Candidate (a), in-process preparation plus `ctypes` preloading, builds the OEq
host extension in a fresh cache without the script and gives the same forces
as the script. Candidates (b) and (c) work too, but they add nothing.
**Recommendation: (a).** The prototype is
[`fastmace/kernels/_oeq_runtime.py`](../../fastmace/kernels/_oeq_runtime.py).

## Setup

| Item | Value |
| --- | --- |
| Machine | local PC, WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2), Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 5060 Ti, driver 610.88; no other compute process |
| Environment | `~/environments/mace_cueq_oeq` (hybrid): torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0, OpenEquivariance 0.7.0, nvidia-cuda-runtime 13.0.96, nvidia-cuda-nvrtc 13.0.88; nothing installed |
| Host compiler | g++ 13.3.0 |
| Model | MH-1 `~/models/mace-mh-1.model` (SHA-256 a522eb7f…), head `matpes_r2scan` |
| Backend, precision | cuEquivariance/OEq hybrid, float32, TF32 off |
| Structure | NaCl rock salt a = 5.64 Å, cubic, 2×2×2 (64 atoms), rattled σ = 0.05 Å, seed 17 |
| Baseline | ResearchModule tag `research-engine-baseline-20261017` (d2faa9fd), `oeq_no_nvcc.sh` exported to `~/fastmace-baseline/` |
| FastMACE | branch `fastmace/1a-T00`, based on `main` 8d884ec |

Spike scripts are in [`spikes/1a-T00/`](../../spikes/1a-T00/):

- `probe.py` loads MH-1, converts it to the hybrid the way
  `common.load_calculator(..., enable_oeq=True)` does (the
  `_hybrid_keeps_last_layer_irreps` patch copied), evaluates forces, and
  records the evidence: environment, the libraries mapped in the process,
  `ldd` of the extension, the `build.ninja` flags and the nvcc sentinel log.
- `run.sh MODE RUN_ID` starts one probe detached, with a fresh cache
  `~/fastmace-spike-cache/RUN_ID`, so every attempt rebuilds the extension.
  `OEQ_EXTENSION_SEED` is never set.
- `compare.py` compares forces against the first script run.
- `loaded_libs.py` lists which CUDA libraries are mapped after each import stage.

Raw results are in
[`results/1a-T00-oeq-runtime-20261017/`](../../results/1a-T00-oeq-runtime-20261017/).

## Results

Each run built `libtorch_tp_jit.so` in its own fresh cache
(`BUILT_EXTENSION=True`, `USE_PRECOMPILED_EXTENSION=False`), and none called the
nvcc sentinel. Forces are compared with the first script run, at identical
positions.

| Run | How the runtime was prepared | OEq import (build) s | Wall s | max \|ΔF\| eV/Å | ΔE eV |
| --- | --- | ---: | ---: | ---: | ---: |
| t00-script-1 | `oeq_no_nvcc.sh` (reference) | 17.1 | 34.1 | 0 | 0 |
| t00-script-2 | `oeq_no_nvcc.sh` again | 13.8 | 25.8 | 2.6e-7 | 3.1e-5 |
| t00-inproc-2 | (a) in-process plus preload | 15.3 | 27.4 | 3.0e-7 | 6.1e-5 |
| t00-inproc-3 | (a) again | 14.8 | 26.9 | 3.0e-7 | 0 |
| t00-inproc-dcache-1 | (a) plus `CUDA_CACHE_PATH` set in-process | 14.9 | 27.4 | 3.2e-7 | 0 |
| t00-nopreload-1 | (a) without the preload | 15.0 | 27.1 | 3.9e-7 | 0 |
| t00-reexec-1 | (b) re-exec with `LD_LIBRARY_PATH` | 14.7 | 26.9 | 3.4e-7 | 0 |
| t00-launched-1 | (c) launcher `python -m fastmace.kernels._oeq_runtime` | 14.7 | 26.6 | 3.6e-7 | 3.1e-5 |
| t00-bare-1 | nothing (plain login environment) | 300.4 | 302.2 | no forces | — |
| t00-inproc-1 | (a), first version without the environment's `bin` on `PATH` | 0.4 | 2.0 | no forces | — |

|E| = 440.15 eV for 64 atoms. **Force agreement:** two runs under the script
differ by 2.6e-7 eV/Å. Every other route differs from the reference by at most
3.9e-7 eV/Å, which is the same run-to-run scatter (OEq and cuEq reduce with
atomics). Energies differ by 0 or one or two float32 ulps (3.05e-5 eV at
440 eV). The forces agree, but not bitwise, as between two runs of the script
itself.

The two failures show why two of the settings matter:

- **t00-bare-1** waited 300 s and then gave up with `RuntimeError("Timed out
  after 300 seconds waiting for another process to finish building the
  OpenEquivariance extension 'libtorch_tp_jit'. … delete
  ~/.cache/torch_extensions/py310_cu130/libtorch_tp_jit/lock")`. The lock file
  has been there since 8 October. MACE then runs without OEq. This is the
  OEq 0.7 lock-file bug that `TORCH_EXTENSIONS_DIR` works around.
- **t00-inproc-1** failed with `RuntimeError('Ninja is required to load C++
  extensions')`. ninja is in the environment's `bin/`, which the script puts on
  `PATH`. The probe was started by the environment's Python without activating
  the environment. Adding that directory to `PATH` in-process fixed it
  (t00-inproc-2), so torch's subprocess lookups use the in-process `PATH`.

## Per setting: needed before process start?

The loader setting is the only one that is read once at process start, and
FastMACE does not need it. Everything else is read later: by torch when it
builds the extension (during `import openequivariance`), by the subprocesses it
starts (ninja, g++), or by the CUDA driver at `cuInit`.

| Setting (script) | Read by, when | Needed before start? | Evidence |
| --- | --- | --- | --- |
| `LD_LIBRARY_PATH` (cu13 lib, cuequivariance_ops lib, driver dir) | the dynamic loader, at process start | **No, and not needed at all** with these wheels | `import torch` already maps `libcudart.so.13`, `libnvrtc.so.13`, `libcublas.so.13` and `libcublasLt.so.13` from `nvidia/cu13/lib` through torch's own RPATH (`loaded_libs.py`). The extension's `DT_NEEDED` sonames match libraries already loaded, so the loader reuses them. t00-nopreload-1 has no `LD_LIBRARY_PATH` and no preload, and maps only the wheel copies. `libcuda.so.1` comes from the standard driver path, which torch also loads at CUDA init. The cuequivariance_ops entry is not needed: every run without it worked. |
| `ctypes` preload (replaces `LD_LIBRARY_PATH`) | — | in-process, before OEq | Harmless and cheap. It keeps (a) correct even if a torch build loads NVRTC lazily. Without it, `ldd` resolves the extension's libraries through `ld.so.cache`: on this PC to the system CUDA 13.3 in `/usr/local/cuda`, and on the workstations to nothing. Kept in the prototype as a guard. |
| `PATH` (sentinel `bin` first, environment `bin`) | `subprocess` / `shutil.which` at build time | No | t00-inproc-1 → t00-inproc-2 (ninja found only after the in-process `PATH` change). The sentinel was never invoked in any run: the extension has only `.cpp` sources, and `CUDA_HOME` is set, so torch does not search for `nvcc`. The sentinel is a guard, kept. |
| `CUDA_HOME`, `CUDA_PATH` (stand-in toolkit of symbolic links) | `torch.utils.cpp_extension` **at its import** (module constant `CUDA_HOME`) | No, but before `torch.utils.cpp_extension` is imported | `build.ninja` of t00-inproc-2 has `-I<cache>/cuda-runtime/include` and `-L<cache>/cuda-runtime/lib64`. **`import mace.calculators` imports OEq and `torch.utils.cpp_extension`**. Without a preset, `CUDA_HOME` became `/usr/local/cuda` on this PC (the system toolkit), and on the workstations it is `None`. |
| `CUDACXX`, `CMAKE_CUDA_COMPILER`, `PYTORCH_NVCC` (sentinel) | build tools, if any reached for nvcc | No | set in-process; never invoked |
| `CPATH` (CCCL headers) | g++ subprocess | No | in-process value inherited by ninja/g++; build succeeded |
| `CXX` wrapper (only where a system CUDA has `lib*.so` links in the libpython dir; leto) | torch at build time | No | in the plan, conditional, as in the script; CPU test covers the condition. Not exercised on this PC (the system CUDA here is not in `/usr/lib/x86_64-linux-gnu`). |
| `OEQ_JIT_EXTENSION=1` | OEq at import | No, before OEq import | without the precompiled `.so`, OEq JIT-builds anyway (t00-bare-1); set for clarity |
| `MAX_JOBS=4` | torch at build time | No | in-process; the user's value is kept |
| `TORCH_EXTENSIONS_DIR` (fresh per process) | torch at build time (`_get_build_directory`) | No, before OEq import | every in-process run built in its own fresh `mkdtemp`. t00-bare-1 shows the failure without it. |
| `OEQ_NVCC_ATTEMPTS` | the sentinel | No | in-process |
| `CUDA_CACHE_PATH`, `CUDA_CACHE_DISABLE` | the CUDA driver at `cuInit` | No, before CUDA init | t00-inproc-dcache-1 set `CUDA_CACHE_PATH` in-process before `import torch`; the driver wrote 17 entries there. `import fastmace` already sets `CUDA_CACHE_DISABLE` on NFS in-process (`fastmace/_file_locks.py`, SPEC §14). |
| `XDG_CACHE_HOME`, `MACE_COMPILE_CACHE`, `TMPDIR`/`TMP`/`TEMP`, `PIP_CACHE_DIR`, `UV_CACHE_DIR` | Python libraries at use time | No | Cache placement belongs to `fastmace._cache` (`FASTMACE_CACHE`), not to the OEq runtime. `tempfile` caches its directory at first use, so `TMPDIR` must be set before then, or `tempfile.tempdir` reset. pip/uv caches matter only for installs. |
| `CMAKE_ARGS` (toolkit discovery off) | OEq's package build (`pip install`) | not a runtime setting | install-time only (SPEC §14 extras) |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | libgomp/MKL/OpenBLAS when loaded (`import torch`) | No, before `import torch` (or `torch.set_num_threads`) | not an OEq setting; belongs to engine configuration |
| `PYTHONDONTWRITEBYTECODE` | the interpreter at start | yes, as a variable; `sys.dont_write_bytecode = True` is the in-process equivalent | not an OEq setting |
| `OEQ_EXTENSION_SEED` | the script (copies a finished build) | — | not reproduced; see "Open" |

### Ordering constraint for (a)

In-process preparation has to happen before the first import of any of
`openequivariance`, `torch.utils.cpp_extension` or `mace.calculators` /
`mace.modules.wrapper_ops`, because importing MACE imports OEq, and OEq builds
at import. It also has to happen before CUDA initialization, for the driver
cache variables. `prepare_in_process` refuses to run when `openequivariance`
or `torch.utils.cpp_extension` is already imported, and names the module. The
module that was imported first decides the error message.

## Candidates

- **(a) In-process environment plus `ctypes` preloading: works.** Same build,
  same libraries, same forces as the script. No extra process, no change to
  how users start Python. One constraint: ordering (above).
- **(b) `prepare()` that re-executes the interpreter once: works**
  (t00-reexec-1), but it brings nothing: the only thing it adds over (a) is
  `LD_LIBRARY_PATH`, which is not needed. It also has costs. It loses
  in-memory state, and it breaks under Jupyter and embedded interpreters,
  `python -c`, debuggers and anything that already holds the GPU. It also
  depends on `sys.orig_argv` replaying the command line exactly.
- **(c) Launcher as a console entry point: works** (t00-launched-1). Same
  environment as (b). It brings back the thing SPEC §3.2 removes: the library
  only works through a wrapper.

## Recommendation

Adopt **(a)**. Whoever builds the hybrid model (the planner's engine factory in
1a) calls `prepare_in_process(cache_dir("oeq"))` once, before it imports MACE.
Because FastMACE loads models itself, it controls that order. If the user
imported MACE or OEq before FastMACE, it is too late: the planner must not run
the hybrid then. It falls back to plain cuEq with a stated reason (SPEC §9.4),
or raises in strict mode. It never silently runs a hybrid built against the
wrong `CUDA_HOME`. With (a), `main.py`'s `require_hybrid_wrapper` refusal
(`HYBRID_WRAPPER_VARIABLES`) can go in T17.

What the prototype carries over from the script: the stand-in CUDA home
(links to the wheel's `include/` and `lib*.so.13`, and `libcuda.so` → the
driver), the nvcc sentinel, `CUDA_HOME`/`CUDA_PATH`, the sentinel variables,
`PATH` (sentinel and environment `bin`), `CPATH`, the conditional `CXX`
wrapper, `OEQ_JIT_EXTENSION`, `MAX_JOBS`, a fresh `TORCH_EXTENSIONS_DIR` and
`OEQ_NVCC_ATTEMPTS`. It leaves the general cache and thread variables to
`fastmace._cache` and the engine config. It also leaves `CUDA_CACHE_DISABLE`
to `fastmace._file_locks`, which already does it at `import fastmace`.

Code rules: the prototype has no module-level mutable state. `plan()` is pure
(CPU tests in [`tests/cpu/test_oeq_runtime.py`](../../tests/cpu/test_oeq_runtime.py)).
It writes only under the cache directory it is given. `prepare_by_reexec` and
`main` are kept only as spike evidence and should be deleted when the
prototype is promoted.

## Open (not in this spike)

- **Rebuild per process.** A fresh extension directory per process repeats the
  host build every time: 13.8–17.1 s here, 23 s on coraline. The script's
  `OEQ_EXTENSION_SEED` avoided that by copying a finished build. With (a),
  FastMACE could instead keep one build per (OEq version, torch, compiler,
  `CUDA_HOME` contents) key under `cache_dir("oeq")`. It would remove the
  stale `lock` before importing OEq, under its own soft lock (SPEC §14 forbids
  kernel locks on NFS). This needs its own task. Prebuilt wheels (phase 6)
  remove the build altogether.
- **Workstations not run.** This spike ran only on the local PC. The claim
  that `LD_LIBRARY_PATH` is not needed rests on torch's RPATH, which is the
  same wheel layout on the NAS environments. Before T17, run the same probe on
  one workstation: coraline (A4000) and leto (the `CXX` wrapper case, CUDA 11.5
  development links).
- **This PC has a system CUDA 13.3 toolkit** in `/usr/local/cuda` (with
  `nvcc`, and in `ld.so.cache`). The workstations have none. The in-process
  `CUDA_HOME` and the preload make sure FastMACE never picks it up silently.
  A probe without them would still build here by accident. On the workstations
  it fails.
