# FAQ and troubleshooting

## First steps when something fails

1. Run `fastmace doctor`. It checks the environment and names the fix for
   each failed item.
2. Read the error message. Every `FastMACEError` says what happened, why,
   what to do next, and where the record is (`run.plan`, `run.gate`,
   `metadata.json`). The CLI prints it without a traceback; add `--debug`
   for the traceback.
3. Read the plan: `print(sim.plan)` (see [The planner](guide/planner.md)).

## Why is my run not on the graph engine?

The report's `plan:` line says which tier ran, and a warning says why. The
usual reasons: OpenEquivariance is not installed (`pip install
"fastmace[oeq]"`), the structure is not 3D periodic, or the model is outside
the rewrite registry. `fm.calc` never uses CUDA graphs; use `fm.md` or the
`fastmace.ase` integrators.

## Why is the first run slow to start?

Setup compiles the model with `torch.compile`, builds the OpenEquivariance
and Triton kernels and captures CUDA graphs. It takes tens of seconds the
first time. Later runs reuse the caches and start faster. The report's
`setup:` line shows the time.

## Caches

All caches (Triton, Inductor, OpenEquivariance, compiled kernels and
downloaded models) live under one root:

- `$FASTMACE_CACHE` if set;
- otherwise `$XDG_CACHE_HOME/fastmace` or `~/.cache/fastmace` on Linux.

The root must not lie inside the Python environment (`sys.prefix`); FastMACE
refuses that with `CacheLocationError`, because environments may be
read-only, shared or rebuilt. Importing fastmace creates nothing; an engine
creates a directory when it needs one. Deleting the cache root is safe: the
next run rebuilds it.

`$FASTMACE_MODELS` is a directory searched for checkpoints before the cache.

## NFS home directories and soft file locks

PyTorch locks its extension and Inductor caches with kernel file locks. On
an NFS home, each lock is a request to the server, and on our workstations
the NAS stopped accepting a client's lock requests altogether until the
mount was reset. So `import fastmace`:

- switches PyTorch's file locks to **soft locks** (a lock file created
  atomically, no lock request);
- switches off the CUDA driver's JIT cache when it lies on NFS and you have
  not set `CUDA_CACHE_DISABLE`.

This only works if fastmace is imported **before torch**. Put `import
fastmace` first in your script. `fastmace.runtime_info()` reports the lock
mode and whether the switch came too late.

`FASTMACE_SOFT_FILE_LOCKS` controls it:

| Value | Effect |
| --- | --- |
| unset | soft locks when possible; importing torch first keeps kernel locks |
| `1` | soft locks; importing fastmace after torch is an error |
| `0` | kernel locks; the CUDA driver cache is left alone |

`fastmace doctor` reports the lock mode on NFS.

## Out of memory

The planner estimates memory before it builds and turns off memory-hungry
rewrites if the estimate does not fit. If the run still fails, use a smaller
supercell or a GPU with more memory. FastMACE is a single-GPU engine with no
domain decomposition.

## My hybrid runs are not bitwise reproducible

That is expected: the hybrid backend uses atomic sums. Compare runs
statistically. See [Reproducibility](guide/restart.md#reproducibility).

## Can I run several structures at once?

Not in 1.0: FastMACE runs one structure at a time. For many small
independent structures, a batched engine such as TorchSim is the better
tool.

## Which GPUs are supported?

NVIDIA GPUs with a driver for CUDA 13. FastMACE has been measured on RTX
5060 Ti (sm_120), RTX 4000 Ada and L4 (sm_89) and A100 (sm_80). There are no
per-GPU tables; every rule is general.

## A model licence warning appears

FastMACE shows each model's licence once per process. Many MACE foundation
models are non-commercial (ASL). See [Models and licences](guide/models.md).
