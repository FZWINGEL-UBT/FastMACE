# Phase 0 validation

Phase 0 has no model code yet, so these runs check the package skeleton,
the import-time soft locks, the test infrastructure and the layering rules
on each target. No baseline comparison applies.

## Summary

| Run | Machine | GPU | Environment | FastMACE commit | Result |
| --- | --- | --- | --- | --- | --- |
| CPU suite | local PC, WSL2 | none (CPU) | `~/environments/fastmace_cpu` | phase 0 tasks | 87 passed, 7 skipped |
| GPU smoke (0-T04) | local PC, WSL2 | RTX 5060 Ti | local GPU environment | 0-T04 branch | passed |
| CPU CI (0-T06) | GitHub Actions | none | 7 cells, see `docs/development.md` | 0-T06 branch | 7/7 green |
| CPU suite | murphy | hidden (CPU) | NAS `mace_cueq_latest` | `8d884ec` | 92 passed |
| GPU tests | murphy | RTX 4000 Ada | NAS `mace_cueq_latest` | `8d884ec` | 2 passed |

## Local and CI results (phases 0-T02 to 0-T06)

- **CPU suite, local PC (WSL2 Ubuntu), `~/environments/fastmace_cpu`:**
  87 passed, 7 skipped.
- **GPU smoke, local PC, NVIDIA GeForce RTX 5060 Ti:** passed in plan task
  0-T04.
- **CPU CI:** run
  <https://github.com/FZWINGEL-UBT/FastMACE/actions/runs/39035008135>,
  7/7 cells green.

These figures are as reported by the tasks that produced them.

## NAS checkout on murphy, 17 October 2026 (plan task 0-T07)

- Machine: murphy (36 CPUs, 250 GB RAM).
- GPU: NVIDIA RTX 4000 Ada Generation, driver 595.58.03; idle before the
  runs (no compute processes, 2 MiB used, 0 % utilisation).
- Environment: `/mnt/homes/zwingel/environments/mace_cueq_latest/`,
  Python 3.10.12, torch 2.14.1+cu130 (CUDA 13.0), MACE 0.3.16,
  cuEquivariance 0.12.0, pytest 9.1.1, pytest-xdist 3.8.0.
- FastMACE: `/mnt/homes/zwingel/FastMACE`, copied from `main` at
  `8d884ecf36a597236e4fa58bcea734a7598aaf96` (`.source_commit`), 42 files,
  LF line endings (checked: no file contains CR).

Common environment of both runs, from `/mnt/homes/zwingel/FastMACE`
(noninteractive SSH, so `LD_LIBRARY_PATH` is set as in ResearchModule's
AGENTS.md):

```bash
E=/mnt/homes/zwingel/environments/mace_cueq_latest
SP=$E/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib
export PYTHONPATH=/mnt/homes/zwingel/FastMACE PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/mnt/homes/zwingel/fastmace-cache/tmp
export XDG_CACHE_HOME=/mnt/homes/zwingel/fastmace-cache/xdg
export FASTMACE_CACHE=/mnt/homes/zwingel/fastmace-cache/fastmace
```

### CPU suite

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 $E/bin/python -m pytest -n 24 -p no:cacheprovider \
  --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-cpu tests/cpu
```

Result: `92 passed, 878 warnings in 13.72s` (24 workers). The warnings
come from torch (`torch.jit.script` deprecation, TorchScript annotations)
and `graph_longrange` (NumPy 2 deprecations), not from FastMACE.
The local run's 7 skips do not occur here.

`--basetemp` is needed on the NAS: without it pytest stops with
`OSError: The temporary directory /mnt/homes/zwingel/fastmace-cache/tmp/pytest-of-zwingel is not owned by the current user`,
because NFS reports owner uid 1024 for files the login uid 1012 creates.
This is a command-line setting only; no code was changed.

### GPU tests

```bash
$E/bin/python -m pytest -p no:cacheprovider -rs -v \
  --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-gpu tests/gpu
```

Result: 2 passed (`test_cuda_is_available`,
`test_tiny_model_forward_on_gpu`), no skips. Warnings only from MACE's
`torch.load` of its constants and TorchScript.
