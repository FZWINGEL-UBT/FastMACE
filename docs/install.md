# Install

FastMACE is a Python library for Linux with an NVIDIA GPU. It needs Python
3.10, 3.11, 3.12 or 3.13.

## Supported stack

FastMACE 1.0 supports one PyTorch build (owner decision D3): **torch
2.14.1+cu130**. The qualified stack around it is:

| Package | Version |
| --- | --- |
| torch | 2.14.1+cu130 |
| mace-torch | 0.3.16 |
| cuequivariance-torch | 0.12.0 |
| ASE | 3.27.0 |
| NumPy | 2.2.6 |
| openequivariance (`[oeq]`) | 0.7.0 |

Other versions may work, but they are not tested. An unknown MACE version
runs with the rewrites the planner cannot verify switched off, and the run
report says so.

## Steps

1. Install PyTorch with CUDA 13.0 first:

    ```bash
    pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    ```

2. Install the cuEquivariance ops wheel that matches the CUDA build. It holds
   the fused kernels and is not a base dependency, because it depends on your
   CUDA version:

    ```bash
    pip install cuequivariance-ops-torch-cu13
    ```

3. Install FastMACE with the extras you need:

    ```bash
    pip install "fastmace[full]"
    ```

No CUDA compiler (`nvcc`) is needed. Kernels are compiled at run time with
Triton and NVRTC, and the results are cached (see [Caches](faq.md#caches)).
On Linux x86_64, the `[oeq]` and `[full]` extras install the prebuilt OEq
host extension for Python 3.10–3.13, so no C++ compiler or Ninja is needed.
A C compiler such as gcc is still required by Triton's launcher stub;
runtime compilation and the initial warm-up still occur.

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

The base install (`pip install fastmace`) runs on the plain cuEquivariance
backend. The run report tells you when a faster backend was not available
and which package would change that.

## Check the installation

```bash
fastmace doctor
```

`doctor` checks the GPU and driver, torch, MACE, the cuEquivariance ops and
whether their fused kernels load, OpenEquivariance and its build path,
Triton, NVRTC, graph-longrange, torch-dftd, the cache root, the file-lock
mode on NFS and the model files with their hashes. Each failed item names
the command or page that fixes it. `doctor` changes nothing. It exits 0 when
everything is ready and 1 otherwise; on a machine without a GPU it exits 1.

`fastmace models` lists the model labels (see [Models and licences](guide/models.md)).

## Build this documentation

```bash
pip install -e ".[docs]"
mkdocs build --strict
```
