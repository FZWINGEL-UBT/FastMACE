# Environment and cache

Every run option is a field of a config record. The few settings on this
page are process-wide, so they are environment variables. This page lists
each variable FastMACE reads or sets, the cache layout, the file-lock rules
and what `fastmace doctor` checks. Installation is on [Install](../install.md);
common problems are in the [FAQ](../faq.md).

## Variables FastMACE reads

| Variable | Values | Default | What it does |
| --- | --- | --- | --- |
| `FASTMACE_CACHE` | directory | the user cache directory (below) | the cache root; must lie outside the Python environment (`sys.prefix`), else `CacheLocationError` |
| `FASTMACE_MODELS` | directory | unset | searched for a label's checkpoint file before `<cache root>/models`; an empty file there is skipped |
| `FASTMACE_SOFT_FILE_LOCKS` | unset, `1`, `0` | unset | PyTorch's file locks; see [File locks on NFS](#file-locks-on-nfs). Any other value raises `ValueError` at `import fastmace` |
| `FASTMACE_OEQ_PREBUILT` | unset, `0`, `1` | unset | the prebuilt OpenEquivariance extension (`fastmace-oeq-ext`): unset uses it when its manifest matches the installed torch, OpenEquivariance and CPython, else builds with the host toolchain; `0` never uses it; `1` refuses the toolchain build, so a mismatch makes the hybrid backend unavailable |
| `OEQ_CUDA_DRIVER_LIBRARY` | path to `libcuda.so.1` | the usual system locations | the CUDA driver library linked into the OpenEquivariance build |
| `MAX_JOBS` | int | `4` | parallel compile jobs of the OpenEquivariance host build |
| `TORCH_EXTENSIONS_DIR` | directory | `$XDG_CACHE_HOME/torch_extensions` or `~/.cache/torch_extensions` | read by `fastmace doctor` to look for stale build locks; FastMACE sets its own value per process for the OEq build (below) |
| `CUDA_CACHE_PATH` | directory | `~/.nv/ComputeCache` | where the CUDA driver keeps its JIT cache; checked for NFS |
| `CUDA_CACHE_DISABLE` | `0`, `1` | unset | the CUDA driver's JIT cache; if you set it, FastMACE leaves it alone |
| `XDG_CACHE_HOME`, `HOME` | directory | `~/.cache` | the base of the default cache root on Linux |
| `LOCALAPPDATA` | directory | | the base of the default cache root on Windows |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch's syntax | unset | if you set it, FastMACE does not change it; see below |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | directory | under the cache root | if you set them, they win over the cache root |

## Variables FastMACE sets

FastMACE sets these in its own process (and so in its child processes).
A variable you set yourself is kept, except where the table says otherwise.

| Variable | Set to | When |
| --- | --- | --- |
| `CUDA_CACHE_DISABLE` | `1` | at `import fastmace`, when the CUDA driver cache lies on NFS and the variable is unset (not with `FASTMACE_SOFT_FILE_LOCKS=0`); always in `fastmace doctor`'s own process |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `<cache root>/inductor`, `<cache root>/triton` | before a plan's first compilation, where unset (a value equal to Inductor's own default `$TMPDIR/torchinductor_$USER` counts as unset) |
| `TORCHINDUCTOR_FX_GRAPH_CACHE`, `ENABLE_AOT_AUTOGRAD_CACHE`, `TORCHINDUCTOR_AUTOGRAD_CACHE` | `1` | with the two above, where unset: the compiled graphs and the compiled force derivative are cached |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | before the first CUDA allocation, when `EngineConfig.expandable_segments` resolves on (torch 2.14 or newer) and the variable is unset |
| `CUBLAS_EMULATE_DOUBLE_PRECISION`, `CUBLAS_EMULATION_STRATEGY` | `1`, `eager` | float64 plans with `EngineConfig.fp64_emulation` on, where unset |
| `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT` | the plan's mantissa bits | with the two above, when `fp64_emulation_mantissa_bits` resolves to a number, where unset |
| `CUBLAS_WORKSPACE_CONFIG`, `TORCHINDUCTOR_DETERMINISTIC`, `CUEQUIVARIANCE_OPS_U1D_ALGOS` | `:4096:8`, `1`, `1,1` | reproducible plans (`fm.Workload(reproducible=True)`), before CUDA starts; these overwrite |
| `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `CMAKE_CUDA_COMPILER`, `PYTORCH_NVCC`, `PATH`, `CPATH`, `CXX`, `OEQ_JIT_EXTENSION`, `OEQ_NVCC_ATTEMPTS`, `MAX_JOBS`, `TORCH_EXTENSIONS_DIR` | a stand-in CUDA home under `<cache root>/oeq` | once per process, before the first hybrid model load: OpenEquivariance builds its host extension without `nvcc` |
| `PYTHONPATH` | the directory holding this `fastmace` first | child processes (the gate's reference, the subprocess runner), so they import the same FastMACE; the runner's children also get `RunnerConfig.environment` |

`metadata.json` records the OEq build variables under
`software.oeq_extension` for hybrid plans, and the planner records the
allocator, emulation and reproducibility settings as plan decisions.

## Cache layout

The cache root is:

1. `$FASTMACE_CACHE`, if set (`~` is expanded);
2. otherwise the user cache directory: `$XDG_CACHE_HOME/fastmace` or
   `~/.cache/fastmace` on Linux, `~/Library/Caches/fastmace` on macOS,
   `%LOCALAPPDATA%\fastmace\Cache` on Windows.

It must not lie inside `sys.prefix`: environments may be read-only, shared
or rebuilt. `import fastmace` creates nothing; each directory is created
when it is first needed. The caches hold generated code and checked
downloads only; their keys cover everything that changes the code, so
results are the same with a cold or a warm cache. Deleting the cache root
is safe while no run is active: the next run rebuilds it.

```text
<cache root>/
├── triton/                  Triton kernels
├── inductor/                torch.compile (Inductor) graphs and the compiled force derivative
├── oeq/                     OpenEquivariance
│   ├── cuda-runtime/        stand-in CUDA home: links to the CUDA 13 runtime wheel and the driver
│   ├── extensions/<key>/    one finished host-extension build per build identity
│   ├── torch-extensions-*/  the per-process TORCH_EXTENSIONS_DIR, seeded from extensions/
│   └── nvcc-attempts.log    calls to the nvcc stand-in
├── models/                  downloaded checkpoints, each checked against its SHA-256 before it is kept
├── handoff/<run>/           subprocess hand-offs of runs without an output directory
└── gate-*/                  scratch of the gate's reference child, removed when it ends
```

A `compile/` kind is reserved; nothing writes it yet.

The OpenEquivariance host extension is built once per build identity
(Python and environment, torch, OpenEquivariance, the CUDA runtime wheel,
the compiler and the build variables; never shapes or models). The first
process builds it into a staging directory and renames it into place, so
the cache needs no lock and may lie on NFS. With the prebuilt
`fastmace-oeq-ext` wheel nothing is built.

### Model files

`fm.load(label)` looks for the label's released file name in this order:

1. `checkpoint=`, an explicit path;
2. `$FASTMACE_MODELS/<file>`;
3. `<cache root>/models/<file>`;
4. with `download=True`, the official URL, downloaded into
   `<cache root>/models/` and kept only if its SHA-256 matches the label
   table.

`fm.md`, `fastmace md` and `fastmace run` never download; fetch a model
once with `fm.load(label, download=True)`. A path is used as it is. A URL is
downloaded into `<cache root>/models/` and needs `sha256=` or
`trusted=True`.

## File locks on NFS

PyTorch locks its extension builds and its Inductor cache through
`filelock.FileLock`, which on Linux uses kernel locks (`fcntl.flock`). On an
NFS home directory every such lock is a request to the server, and a server
can stop accepting a client's lock requests until the mount is reset. The
CUDA driver also locks the `index` file of its JIT cache, hundreds of times
per run.

`import fastmace` therefore:

- makes `filelock.FileLock` the soft lock (`SoftFileLock`), which creates
  the lock file atomically (`O_CREAT | O_EXCL`) and sends no lock request;
- switches the CUDA driver's JIT cache off (`CUDA_CACHE_DISABLE=1`) when it
  lies on NFS and you have not set `CUDA_CACHE_DISABLE`. NFS is read from
  `/proc/self/mounts`, never with `stat`, which can hang on a broken mount.

PyTorch binds the lock class when it is imported. So **`import fastmace`
must come before `import torch`** (and before anything that imports torch,
such as MACE). Importing fastmace imports neither torch nor MACE.

| `FASTMACE_SOFT_FILE_LOCKS` | Effect |
| --- | --- |
| unset | soft locks when torch is not imported yet; if torch came first, kernel locks stay and `fastmace.runtime_info()` reports that the switch came late |
| `1` | soft locks; importing fastmace after torch raises `RuntimeError` |
| `0` | kernel locks, and the CUDA driver cache is left alone |

`fastmace.runtime_info()` returns the lock mode (`"soft"` or `"kernel"`),
how it came about, whether the switch came late, and what happened to the
CUDA driver cache. `metadata.json` records it under `runtime`.

Stale build locks: a killed torch-extension build leaves a `lock` file in
its build directory, and OpenEquivariance waits up to 300 s for it. FastMACE
gives each process a fresh extension directory, so a leftover lock elsewhere
does not stall a run. `fastmace doctor` lists locks older than one hour that
no process holds, with the command that removes them.

## `fastmace doctor`

`fastmace doctor` prints one line per check, in this order. A failed
required check prints `[FAIL]` and makes the exit code 1; a failed optional
check prints `[warn]`. Each failed item names its fix. `--no-hash` skips the
SHA-256 of the model files.

| Check | Required | `ok` when | When it fails |
| --- | --- | --- | --- |
| GPU and driver | no | `torch.cuda.is_available()`; shows the GPUs, the driver (from `nvidia-smi`) and torch's CUDA version | no CUDA GPU: MD needs an NVIDIA GPU |
| torch | yes | torch is installed | install a CUDA build of torch |
| MACE | yes | `mace-torch` is installed (looked up, not imported); notes a version other than 0.3.16, the qualified one | install `mace-torch==0.3.16` |
| cuEquivariance | yes | `cuequivariance`, `cuequivariance_torch` and the ops package `cuequivariance_ops_torch` are installed, and every shared library the ops wheels list is present | without the ops package MACE would use the naive kernels; install the ops wheel matching torch's CUDA |
| OpenEquivariance | no | installed; the prebuilt extension matches, or `g++`, `ninja`, the CUDA 13 runtime wheel and `libcuda.so.1` are found; shows whether the extension was built before; no stale build lock | the hybrid backend is unavailable (plain cuEq runs), or a stale lock is listed with its `rm` command |
| Triton | no | Triton is available | `pip install triton` |
| NVRTC | no | NVRTC is available (it compiles the streaming kernels) | install `cuda-bindings` matching torch's CUDA |
| graph-longrange | no | installed | needed for MACE-POLAR-1: `pip install 'fastmace[polar]'` |
| torch-dftd | no | installed | needed for the D3 reference: `pip install 'fastmace[d3]'` |
| cache root | yes | outside `sys.prefix` and writable (or its nearest existing parent is); shows where it comes from and whether it exists | set `FASTMACE_CACHE` to a writable directory outside the environment |
| file locks | yes | not (kernel locks with the cache root on NFS), and not (the CUDA driver cache on NFS and enabled) | `import fastmace` before torch and leave `FASTMACE_SOFT_FILE_LOCKS` unset or `1`; or set `CUDA_CACHE_DISABLE=1` or `CUDA_CACHE_PATH` to a local directory |
| model files | no | every file in `$FASTMACE_MODELS` and `<cache root>/models` that has a label-table name matches its SHA-256; files outside the table are listed | delete the file and fetch it again with `fm.load(label, download=True)` |

`doctor` changes nothing. It looks MACE and OpenEquivariance up without
importing them (importing either builds OEq's host extension), reads the
cuEquivariance ops wheels' metadata without importing them (the import
creates a Triton cache directory), creates no cache directory, and deletes
no lock. So an ops library that is present but does not load shows only at
the first model load. A check that crashes is reported as failed, with the
hint to run `fastmace --debug doctor`.

Without a GPU, the GPU check is a warning. On a CPU-only machine the exit
code is usually 1 all the same, because the cuEquivariance ops wheel is
missing.
