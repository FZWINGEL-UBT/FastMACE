# 2-T14: the flaky CPU test

Plan task 2-T14 (phase 2), 20 October 2026. In the conductor's trial of 1b-T20 and
1b-T25, 1 CPU test of 1964 failed once and was not captured. This task found it,
found its cause in two dependencies, and fixed it in `fastmace.model.loading`.

## Setup

| Item | Value |
| --- | --- |
| Machine | murphy, CPUs only (36 cores; the GPU was not used) |
| Environment | NAS `mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuequivariance 0.12.0, e3nn), `PYTHONPATH` checkout |
| Command | `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -m pytest -n 8 -rf -p no:cacheprovider --junitxml=… --basetemp=… tests/cpu`, `set -o pipefail` |
| Caches | own cache root under `/mnt/homes/zwingel/fastmace-2-T14/`; batch 1 cold (a new, empty cache per run), the confirmation runs warm |
| Files | logs, junit files and probe scripts in `/mnt/homes/zwingel/fastmace-2-T14/` (`logs/`, `junit/`, `probe/`, `plugins/`) |

No order plugin is installed in any environment; a small seeded order plugin outside the
repository was prepared but not needed, because the default order reproduced the failure.

## Batch 1: the failing test

FastMACE `main` ed39645, default order, cold caches.

| Run | Result | Time |
| --- | --- | ---: |
| 01 | 2145 passed, 5 skipped | 323 s |
| 02 | 2145 passed, 5 skipped | 308 s |
| 03 | 1 failed, 2144 passed | 315 s |
| 04 | 1 failed, 2144 passed | 313 s |

Both failures were `tests/cpu/test_forces.py::test_polar_force_field_enables_its_own_copy`:
1 force component of 192 differed from MACE's e3nn calculator by 5.41e-12 (relative 2.0e-12),
against `rtol=1e-12, atol=1e-12`. No other test failed.

## Cause

The test compares a tiny PolarMACE, converted to cuEquivariance in float64, with the same
model in e3nn. Alone, its largest force deviation is 2.66e-12. After any **float32**
e3nn-to-cuEq conversion earlier in the same process, it is 5.41e-12 (7.26e-12 outside
pytest), and the e3nn reference stays bitwise the same. Every CPU `Simulation` of a tiny
model does such a conversion: the minimal planner plans plain cuEq in float32 and
`loading.load` converts the model.

The chain:

1. MACE 0.3.16's `mace.tools.cg.O3_e3nn.clebsch_gordan` returns
   `e3nn.o3.wigner_3j(l1, l2, l3)` without a dtype, so in torch's default dtype. During a
   float32 conversion that is float32.
2. cuequivariance 0.12.0 keeps what it builds from these coefficients, keyed without their
   dtype:
   - the basis products in the module dict
     `cuequivariance.group_theory.irreps_array.reduced_tensor_product._cache_reduce_basis_product`
     (key: irreps, shapes and the bytes of the input bases);
   - the reduced bases and the symmetric-contraction descriptor in `functools` caches
     (`_main_cached_recursive`, `descriptors.symmetric_contractions._symmetric_contraction_core`, …).
3. A later float64 conversion with the same irreps takes these float32-rounded values. In the
   converted polar model, 48 of 338 state tensors differ from a fresh conversion:
   the symmetric-contraction weights by up to 1.7e-8 and the descriptor coefficients
   (1/√3, 1/3, …) by up to 5.8e-9. The reduced basis differs by up to 1.3e-8.

Only the first conversion of each irreps in a process sets what all later ones get. So the
result depends on which tests ran earlier in the same xdist worker, and the test failed in
2 of 4 runs.

Modules whose tests leave the state behind (each one, run before the polar test in one
process, makes it fail): `test_acceptance_1b`, `test_cli`, `test_ase_md`,
`test_provenance_git` (`test_a_simulation_builds_the_block_once_across_ten_runs`) and
`test_quick`. 1b-T27's worker found the pair
`test_acceptance_1b.py::test_two_simulation_sides_on_the_cpu` and the polar test
independently; its new test files change xdist's distribution, so the pair met in 3 of 5
runs there.

Ruled out on the way: the `_patches` registry (restoring every patch changes nothing),
`torch._dynamo.reset()`, TorchScript's profiling executor, thread counts, imports
(`cuequivariance_ops_torch` included), and environment variables. Clearing every
`functools` cache in the process (596) does not help while the dict is kept; clearing the
dict alone restored the values in the probe. The fix clears both, because the memoised
bases and descriptors are built from the dict's products too.

## Fix

`fastmace.model.loading._fresh_cueq_basis_cache()` clears the dict and the module-level
`functools` caches of the imported `cuequivariance.group_theory` modules before and after
each conversion:

- in `load`, around MACE's calculator build (plain cuEq, hybrid and e3nn, so also
  `load_pristine`);
- in `fastmace.model.compile.naive_cueq()`, the scope of every CPU cuEq conversion, MACE's
  converter called directly included.

Each conversion then depends only on its own dtype, as the first one of a process does.
Only modules already imported are touched; if the dict is missing (another cuequivariance),
the clearing of it is skipped with one debug log. Nothing is recorded in the provenance.

The fix changes no default result. A production run converts in a fresh process, and the
steps before its first conversion (`load_pristine`, `ForceField`, the planner) fill none of
these caches (checked). In production order, the tiny MH-1's float32 cuEq model hashes the
same before and after the fix (`59ca5f81…`). Only a float64 conversion that follows a
float32 one changes (`2feb0901…` before, the fresh value after).

The acceptance gates were not affected. The acceptance tools build their references in
fresh processes, and the in-library gate builds its references before the production
engine (1b-T16), so a reference conversion was always the first of its process.

## Tests

- `tests/cpu/test_cueq_basis_cache.py` (CPU, about 20 s):
  - a float64 cuEq load after a float32 one equals a fresh float64 load, bitwise (48
    tensors differed before the fix);
  - a pristine e3nn load after a float32 conversion equals a fresh one, bitwise. Pristine
    models keep their checkpoint's U matrices, so this guards, it did not fail before;
  - the 1b-T27 trigger pair passes in a fresh pytest process (failed at 5.41e-12 before);
  - `naive_cueq` clears the caches; a missing dict is skipped;
  - a canary for the upstream behaviour: without clearing, a float64 basis after a float32
    one is off by between 1e-10 and 1e-6.
- `test_polar_force_field_enables_its_own_copy` keeps its 1e-12 bound; it is what found this.
- `tests/cpu/test_loading.py`: the `clean_patches` fixture also hides the patches that their
  own module keeps (`_RECORDED_ELSEWHERE`). Before, any PolarMACE test earlier in the worker
  left `polar_longrange` in `_patches.applied()`, and three conv-fusion tests failed whether
  the polar test passed or not. `polar_longrange` stays permanent in `tests/conftest.py`,
  since it cannot be undone.

## Confirmation runs

murphy CPUs, default order, warm caches, same command as batch 1. The branch synced with
`main` 8c8a101 (2-T01 merged) is 4cdcfd1; 1b-T27 merged into a scratch copy of it only
(f364992), since T27 merges after this task.

| Run | Source | Result | Time |
| --- | --- | --- | ---: |
| c1 | 4cdcfd1 + 1b-T27 | 2252 passed, 5 skipped | 148 s |
| c2 | 4cdcfd1 + 1b-T27 | 2252 passed, 5 skipped | 147 s |
| c3 | 4cdcfd1 + 1b-T27 | 2252 passed, 5 skipped | 149 s |
| c4 | 4cdcfd1 + 1b-T27 | 2252 passed, 5 skipped | 160 s |
| c5 | 4cdcfd1 + 1b-T27 | 2252 passed, 5 skipped | 154 s |
| c6 | 4cdcfd1 | 2239 passed, 5 skipped | 145 s |

Without the fix, the same T27 combination failed in 3 of 5 runs (1b-T27's worker). No other
user's process ran on murphy during the runs. The local `fastmace_cpu` suite was not run,
because the local machine was reserved for 2-T12's timing.

## Upstream

- cuequivariance 0.12.0: `reduced_tensor_product._cache_reduce_basis_product` and the
  `functools` caches built on it are keyed without the dtype of the Clebsch-Gordan
  coefficients that an `Irrep` subclass supplies.
- MACE 0.3.16: `O3_e3nn.clebsch_gordan` returns e3nn's `wigner_3j` in torch's default dtype,
  so a float32 conversion feeds float32 coefficients into cuequivariance's float64 numpy code.

Phase 6 files the upstream issues.
