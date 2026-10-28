# Install

FastMACE is a Python library for Linux with an NVIDIA GPU. It needs Python
3.10, 3.11, 3.12 or 3.13 and a driver for CUDA 13. No CUDA compiler
(`nvcc`) and no LAMMPS build are needed.

## Steps

1. **Install PyTorch for CUDA 13.0.** FastMACE 1.0 supports one PyTorch
   build, torch 2.14.1+cu130:

    ```bash
    pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    ```

2. **Install the cuEquivariance ops wheel for your CUDA build.** It holds
   the fused kernels. It is not installed automatically, because the
   right wheel depends on your CUDA version (`-cu12` or `-cu13`):

    ```bash
    pip install cuequivariance-ops-torch-cu13
    ```

3. **Install FastMACE with its extras.** `[full]` is what most users want:

    ```bash
    pip install "fastmace[full]"
    ```

4. **Check the installation:**

    ```bash
    fastmace doctor
    ```

Then go to [FastMACE in five minutes](quickstart.md).

!!! note "Compilers"
    Kernels are compiled at run time with Triton and NVRTC, and the results
    are cached (see [Caches](faq.md#caches)). On Linux x86_64, the `[oeq]`
    and `[full]` extras install the prebuilt OpenEquivariance host
    extension for Python 3.10–3.13
    ([Prebuilt OEq extension](release/oeq-wheels.md)), so no C++ compiler
    or Ninja is needed.
    A C compiler such as gcc is still required by Triton's launcher stub,
    and the runtime compilation and the first warm-up still happen.

## Extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `[oeq]` | openequivariance | the cuEq/OEq hybrid backend, the fastest one |
| `[polar]` | graph-longrange 0.4.0 | PolarMACE (MACE-POLAR-1) long-range models |
| `[d3]` | torch-dftd 0.5.1 | the torch-dftd D3 reference; FastMACE's own GPU D3 needs nothing extra |
| `[full]` | the three above | most users |
| `[d3-reference]` | torch-dftd, simple-dftd3 | the D3 cross-check tests only |
| `[dev]` | pytest, pytest-xdist, ruff | development |
| `[docs]` | mkdocs-material, mkdocstrings | building this site |

!!! note "The base install"
    `pip install fastmace` without extras also works. It runs on the plain
    cuEquivariance backend. The run report tells you when a faster backend
    was not available and which package would change that.

## Supported stack

FastMACE 1.0 is qualified on this stack:

| Package | Version |
| --- | --- |
| torch | 2.14.1+cu130 |
| mace-torch | 0.3.16 |
| cuequivariance-torch | 0.12.0 |
| ASE | 3.27.0 |
| NumPy | 2.2.6 |
| openequivariance (`[oeq]`) | 0.7.0 |

!!! note "Other versions"
    Other versions may work, but they are not tested. An unknown MACE
    version runs with the rewrites the planner cannot verify switched off,
    and the run report says so.

## What `fastmace doctor` checks

`doctor` checks the GPU and driver, torch, MACE, the cuEquivariance ops and
whether their fused kernels load, OpenEquivariance and its build path,
Triton, NVRTC, graph-longrange, torch-dftd, the cache root, the file-lock
mode on NFS and the model files with their hashes. Each failed item names
the command or page that fixes it. `doctor` changes nothing. It exits 0
when everything is ready and 1 otherwise.

`fastmace models` lists the model labels (see
[Models and licences](guide/models.md)).

!!! note "Checking an install on a machine without a GPU"
    On a machine without a GPU, the GPU check only warns, but `doctor`
    usually exits 1 because the cuEquivariance ops wheel is missing. To check only that a fresh environment resolves `[full]` and that
    `doctor` runs, install CPU torch and accept exit code 0 or 1; exit code
    2 or a traceback is a failure. From a source checkout:

    ```bash
    python -m venv fastmace-check
    fastmace-check/bin/python -m pip install --upgrade pip
    fastmace-check/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fastmace-check/bin/python -m pip install ".[full]"
    fastmace-check/bin/python -m pip check
    fastmace-check/bin/fastmace doctor; test $? -le 1
    ```

## From a source checkout

```bash
pip install -e ".[dev]"
```

To build this documentation:

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

See also: [Environment and cache](reference/environment.md) for the
environment variables, the cache root and every `doctor` item.
