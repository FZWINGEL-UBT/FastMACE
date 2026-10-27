# Suite speed: the CPU merge tier

Owner request, 27 October 2026: make `tests/cpu` in the merge tier (`pytest -m "not acceptance"`) faster.
Physics-critical tests stay, and no test is weakened or skipped. At the owner's request ("yes please just
optimize it on murphy") the measurements ran on workstation murphy, not on the local PC.

**Result on murphy, `-n 16`, `OMP_NUM_THREADS=2`, warm caches: 483 s before and 350–363 s after, about
26 % less elapsed time (1.34–1.38× faster).** No test that passed before fails or is skipped after.

## Changes

The library (`fastmace/`) is unchanged.

1. **Thread limits with `-n` (`tests/conftest.py`).** The controller sets `OMP_NUM_THREADS=2`,
   `MKL_NUM_THREADS=2` and `TORCHINDUCTOR_COMPILE_THREADS=1` for the workers and their child processes,
   unless the caller already set them. With the load scheduler it also sets `--maxschedchunk=1`, the
   documented setting of `docs/validation/6-T16.md`.
   - Today's local conductor run (`-n 8`, no `OMP_NUM_THREADS`) ran every worker with a 24-thread OpenMP
     pool. It reached a load of about 130 on 24 threads and took about 100 minutes.
   - On murphy, which was already called with `OMP_NUM_THREADS=2`, this alone changed nothing (cfg1).
   - With one thread per worker the run was slower (457 s, cfg3), so the default stays at 2.
2. **Longest first (`tests/conftest.py`, `tools/test_order.py`, `tests/_support/data/cpu_durations.json`).**
   The xdist workers sort the test functions recorded as taking 2 s or more to the front, longest first.
   - In collection order, the 50–75 s tests (subprocess runner and worker, fallback searches) started in
     the last minutes and left the other workers idle.
   - The result: 483 → 423 s, with the run then about 5 % above the summed test time divided by 16.
   - The file only changes the order: a stale or missing file costs time, never a test.
     `tests/cpu/test_test_order.py` checks that every entry still names an existing test function.
   - Refresh the file after large test changes: `python tools/test_order.py <run>/junit.xml`.
3. **cuEquivariance's naive CPU kernels cached per process (`tests/_support/naive_cueq_cache.py`).**
   - **What was slow:** a profile of every test over 5 s put about 35 % of their time in cuEq conversions
     of tiny models. Each `Simulation` or planner build converts the model again: 1,331 loads in those
     tests. On the CPU each conversion builds its products with cuEq's naive method, and
     `opt_einsum_fx.optimize_einsums_full` took 83 % of that. `compile.private_copy` rebuilds the same
     kernels through cuEq's `__init__`, because the modules pickle bare.
   - **The cache:** the first result for each key is kept, and every caller gets a deep copy: its own
     modules and buffers, with the same graph and bitwise the same constants.
   - **The key:** the descriptor (cuEq compares the coefficients of its paths exactly), the coefficient
     dtypes, the device, the math dtype, the einsum flag and torch's default dtype. So one dtype never
     takes another dtype's kernels (the 2-T14 concern).
   - **Effect on murphy:**
     - a repeated tiny-model conversion: 0.79 → 0.25 s;
     - `private_copy`: 0.45 → 0.10 s;
     - the suite: 420 → 360 s.
   - **Checks:** `tests/cpu/test_naive_cueq_cache.py` checks private copies, separate dtype entries, and
     a whole conversion with and without the cache giving bitwise equal energy and forces.
   - **Scope:** tests only. `FASTMACE_TEST_NAIVE_CUEQ_CACHE=0` turns it off, and child processes do not
     inherit it.
4. **Tiny models built once per set of arguments (`tests/_support/tiny_models.py`).**
   - **The cache:** later calls get a deep copy of the first build (0.05 s instead of 0.4–0.7 s) and
     leave torch's default dtype and RNG state as a build does.
   - **Not cached:** arguments that hold arrays or tensors.
   - **Checks:** `tests/cpu/test_tiny_model_cache.py` checks the weights against a fresh build, the RNG
     state, and that a copy changed by one test changes no other model.
   - **Pickled bytes:** a cached copy pickles to different bytes than the first build, but two fresh
     builds already do that, so no test can depend on it.
   - **Effect:** 360 → 350 s.
5. **One packaging build (`tests/cpu/test_packaging.py`).**
   - **Why:** `test_wheel_contents` and `test_sdist_contents` shared a module fixture that ran
     `python -m build` in the source tree. Under xdist two workers ran it at once and raced on
     `*.egg-info` and `build/` (the conductor saw errors at `-n 4`).
   - **Change:** they are now one test, `test_wheel_and_sdist_contents`, with the same assertions and one
     build.

## Rejected after measurement

- **One OpenMP thread per worker:** 457 s against 423 s.
- **A bytecode cache (`PYTHONPYCACHEPREFIX`) for child processes:** no measurable change. A child's
  start-up is mostly MACE and cuEquivariance imports; `cuequivariance_ops_torch` alone takes 2.5 s in
  the NAS environment.
- **Merging the fresh-child comparisons:**
  - What they cost: the subprocess runner and worker tests (about 640 s of summed time) spend it in
    fresh child processes of 5–9 s each. The gate's reference children (about 95 per run, about 9 s
    each) do the same.
  - Why not merged: the comparison with fresh children is what those tests check. Merging them would mean
    relying on transitivity across different call sequences, which does not give the same coverage.

## Measurements

Machine: murphy, 36 cores, 250 GB RAM, idle (idle check before every run, CPU only,
`CUDA_VISIBLE_DEVICES=`). Environment: NAS `mace_cueq_latest` (torch 2.14.1+cu130, cuEquivariance 0.12.0,
MACE 0.3.16, pytest 9.1.1, xdist 3.8.0). Source: `git archive` of `aaeac537` with the branch's files on
top, in `/mnt/homes/zwingel/suite-speed-aaeac537/`, with every cache and `--basetemp` there.

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=$PWD TMPDIR=... FASTMACE_CACHE=... \
  XDG_CACHE_HOME=... TORCHINDUCTOR_CACHE_DIR=... TRITON_CACHE_DIR=... \
  .../mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -q -n 16 -m "not acceptance" tests/cpu \
  --basetemp=... --durations=60 --junitxml=...
```

| Run | Change | Wall (pytest) | Summed test time | Passed |
| --- | --- | ---: | ---: | ---: |
| before-warm | `aaeac537`, cold compile caches | 563.4 s | 6,609 s | 4,493 |
| **before** | `aaeac537`, warm caches | **483.1 s** | 5,892 s | 4,493 |
| cfg1 | 1 (thread limits, chunk 1), 5 | 482.9 s | 6,109 s | 4,493 |
| cfg2 | + 2 (longest first) | 423.4 s | 6,420 s | 4,493 |
| cfg3-omp1 | cfg2 with one thread per worker (rejected) | 457.1 s | — | 4,493 |
| after (commit `60bb747b`) | 1, 2, 5 and the order tests | 420.1 s | 6,374 s | 4,497 |
| cache1 | + 3 (naive cuEq kernels) | 360.0 s | 5,367 s | 4,500 |
| cache2 | + 4 (tiny models) | 349.7 s | 5,218 s | 4,500 |
| **final** | the branch, refreshed order file | **357.3 s** | 5,170 s | 4,502 |

Run-to-run noise is about ±10 s (cache2 and final have the same changes apart from the order file). The
summed test time rose under longest-first ordering, because the long tests now share the machine. Murphy's
`-n 16` floor for the final sum is 323 s.

Failures in every run, all from the environment:

- `tests/cpu/test_colab_driver.py`: 16 tests. They need a git checkout, and the NAS copy is none.
- `tests/cpu/test_packaging.py`: `build` cannot build the package in the NAS environment. There were two
  errors before (one fixture for two tests) and there is one failure after (the merged test).

Coverage: the junit files of `before` and `final` have the same skipped set (202). No test that passed in
`before` is missing or failing in `final`. The 9 new passes are the new tests of this change.

**Local PC** (WSL2, `fastmace_cpu`, `-n 8`, after commit `60bb747b`, the thread limits applied by the
conftest, no other pytest running): the 20 slowest files passed in 351 s (711 passed, 9 skipped). There is
no local full-suite measurement. The local suite was stopped at the owner's request, and fm-ideas was
running GPU benchmarks at the time of the final run.

### Slowest tests

Before (`before`, warm):

| # | s | Test |
| ---: | ---: | --- |
| 1 | 76.4 | `test_subprocess_worker::test_the_worker_equals_a_fresh_child_per_call` |
| 2 | 71.5 | `test_subprocess_worker::test_observer_mutations_through_the_worker_equal_fresh_children` |
| 3 | 64.3 | `test_swap_anneal_protocol::test_a_tiny_protocol_in_pieces_equals_one_go` |
| 4 | 62.6 | `test_subprocess_runner::test_a_species_host_write_reaches_the_next_child` |
| 5 | 60.1 | `test_ci_local_gpu::test_concurrent_waiters_run_in_arrival_order` (a stall, see below) |
| 6 | 57.4 | `test_subprocess_runner::test_mc_attaches_keeps_and_detaches_through_the_runner[fresh-children]` |
| 7 | 56.5 | `test_fallback_simulation::test_a_broken_pair_is_found_by_the_real_gate` |
| 8 | 53.0 | `test_fallback_simulation::test_a_broken_rewrite_is_found_by_the_real_gate` |
| 9 | 45.9 | `test_d3_kernel::test_triton_kernels_through_the_interpreter` |
| 10 | 42.5 | `test_fallback_mid_run_mc::test_a_move_set_attached_after_the_point_is_redone_with_its_streams` |

After (`cache2`). These tests now run at the same time at the start, so each takes slightly longer while
the run as a whole is shorter:

| # | s | Test |
| ---: | ---: | --- |
| 1 | 93.2 | `test_subprocess_worker::test_observer_mutations_through_the_worker_equal_fresh_children` |
| 2 | 79.4 | `test_subprocess_worker::test_the_worker_equals_a_fresh_child_per_call` |
| 3 | 70.5 | `test_subprocess_runner::test_a_species_host_write_reaches_the_next_child` |
| 4 | 66.5 | `test_subprocess_runner::test_mc_attaches_keeps_and_detaches_through_the_runner[fresh-children]` |
| 5 | 49.1 | `test_fallback_simulation::test_a_broken_pair_is_found_by_the_real_gate` |
| 6 | 46.9 | `test_d3_kernel::test_triton_kernels_through_the_interpreter` |
| 7 | 44.7 | `test_fallback_mid_run_mc::test_a_move_set_attached_after_the_point_is_redone_with_its_streams` |
| 8 | 44.6 | `test_combo_kernels::test_the_record_marks_the_model_and_nothing_else` |
| 9 | 44.5 | `test_fallback_simulation::test_a_broken_rewrite_is_found_by_the_real_gate` |
| 10 | 44.1 | `test_swap_anneal_protocol::test_a_tiny_protocol_in_pieces_equals_one_go` |

The junit files are in `results/suite-speed-20261027/` (gzip).

## Found on the way, not changed

- **`test_concurrent_waiters_run_in_arrival_order`** (`tools/ci/local_gpu.py`) took 60.05 s, its waiters'
  `wait_s`, in 3 of 9 full runs and 0.1 s in the others. It still passes. It did not reproduce in 12 runs
  alone or under a 100-test load. One waiter seems to miss its turn until it times out; this is worth a
  look with the queue's polling on NFS (`--basetemp` on the NAS).
- **Child processes do not get the caches.** A gate reference child (8.8 s, measured) loads the same model
  twice: 2.7 s of conversions, about 1 s of which the naive-kernel cache would save. Reaching the children
  needs a start-up hook (`sitecustomize` on `PYTHONPATH`). Murphy's system Python already has one, which
  the hook would have to chain to.
- **The library could cache the naive kernels itself**, inside `compile.naive_cueq()`. CPU production
  conversions would then cost about 0.25 s instead of 0.8 s. That is a change to `fastmace/model/` and
  needs its own review.
