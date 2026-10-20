# 2-T08: instance isolation, A → B → A (phase 2 acceptance 5)

Plan task 2-T08. Design and audit: [docs/design/isolation.md](../design/isolation.md).
Tests: `tests/cpu/test_isolation.py` (every change) and
`tests/gpu/test_isolation_gpu.py` (acceptance tier, released models).

## Result

A → B → A passes on the plain cuEq, MH-1 hybrid and POLAR-1-S paths. Every
interleaved evaluation equals a fresh force field's within the dtype's
self-check threshold (1e-10 float64, 1e-4 float32), and so does every
interleaved engine's force at its own positions. Repeating A's first
evaluation after B compiles nothing. No lease was needed. One shared state is
left open, a failing `single_graph_strain` self-check (finding 1 of the
audit); it is pinned by a CPU test and closed by 2-T08b.

## GPU run

- Machine: murphy, NVIDIA RTX 4000 Ada Generation (20 GB). The GPU was shared
  with other zwingel processes (untimed jobs of sibling tasks); no other
  user's process was on it.
- Environments: NAS `mace_cueq_latest` (torch 2.14.1+cu130, cuEquivariance
  0.12.0, MACE 0.3.16) for plain cuEq; `mace_cueq_oeq` as
  `FASTMACE_HYBRID_PYTHON` for both hybrids.
- Models: `mh1-r2scan` (MH-1, head `matpes_r2scan`) and `polar1-s`
  (POLAR-1-S), from `/mnt/homes/zwingel/ResearchModule/Models`.
- FastMACE commit d70307e, from a copy in `/mnt/homes/zwingel/fastmace-2-T08`
  with its own compile caches. No baseline: the comparison is against fresh
  FastMACE instances in the same process.
- `TORCHINDUCTOR_COMPILE_THREADS=4`; 20 October 2026, 04:50–04:59 CEST.
- JSON: `results/2-T08-isolation-murphy-20261020/`.

```bash
$E/bin/python -m pytest -n 0 -p no:cacheprovider -s -rf -m acceptance tests/gpu/test_isolation_gpu.py
```

Crystals: A = NaCl 3×3×3 (216 atoms), B = NbCl5 2×2×2 (288 atoms),
A2 = NaCl 2×2×2 (64 atoms, the same species as A). The fixed plans are
compiled, with captured graphs and pipelined steps. Engines ran 100 NVT
steps at 600 K, each engine stepped in turn with the others.

| Path | dtype | Compiled models shared | Max. evaluation error | Compiles in the repeated A | Engine force error (A, B, A2) | Position deviation from alone (A, B, A2), Å | Rebuilds (A, B, A2) | Time |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| MH-1 plain cuEq | float64 | A, B, A2 (one entry: no species-dependent record) | 1.1e-15 | 0 | 8.4e-16, 1.7e-15, 8.4e-16 | 9.9e-14, 2.4e-13, 3.6e-14 | 12, 13, 11 | 129 s |
| MH-1 hybrid | float32 | A with A2 (two entries; 2 streaming interactions each) | 6.7e-7 | 0 | 5.0e-7, 5.4e-7, 3.9e-7 | 3.5e-7, 6.6e-7, 1.9e-7 | 12, 13, 11 | 135 s |
| POLAR-1-S hybrid | float32 | none (polar copies are never cached) | 3.2e-6 | 0 | 2.2e-7, 1.8e-7, 2.0e-7 | 4.5e-7, 1.3e-6, 4.5e-7 | 12, 15, 10 | 259 s |

- **MH-1 hybrid:** A and A2 share one compiled model with its streaming
  interactions, so each call swaps in its own edge counter. Their edge counts
  differ, and both stay at float32 rounding.
- **POLAR-1-S:** building B and A2 reset dynamo (`torch._dynamo.reset()`),
  and the earlier field A recompiled at its next call: 21 graphs compiled.
  A's result after the recompile is within 3.2e-6 of its first value. This
  is "correct, slower" (conductor answer Q3).
- **Plain cuEq:** the trajectories are not bitwise equal to the engines run
  alone, because cuEquivariance's scatter is not deterministic. They are
  within 2.4e-13 Å after 100 steps, and the test's bound is 1e-8 Å.
- A2 used two edge sizes (a new bucket with a third graph), A and B one.

The first run (at b2c3b2b) failed only on the test's own sharing assertion:
it expected NaCl and NbCl5 never to share a model. Plain cuEq MH-1 has no
species-dependent record, so the cache key does not include the species and
they share one model, as the audit says. Its numbers matched the table. The
test now derives the expectation from the plan's records (d70307e).

## CPU

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 0 tests/cpu/test_isolation.py
```

Result: 9 passed, 1 skipped (PolarMACE: `fastmace_cpu` has no
graph_longrange; the GPU test covers POLAR-1-S), in 28 s. Covered there:
- no shared state tensor between two force fields on one model;
- A → B → A bitwise against fresh force fields, with the shared source model,
  private rewritten copies and one compiled model from the cache, with and
  without stress;
- a buffer swapped on compiled code (the streaming counter) is read as an
  input: the right count every call, no recompilation after the first round;
- two engines stepped in turn are bitwise equal to each engine run alone,
  across neighbour rebuilds;
- finding 1, pinned.

## 2-T08b: the strain path per model

Finding 1 of the audit is closed: each force field binds its strain path into
its own forward ([strain-per-model.md](../design/strain-per-model.md)), so a
failing `single_graph_strain` self-check or the reference calculators'
process patch no longer changes a force field built earlier.

All runs are at a3ecb2f (main f0bde74 merged in), 20 October 2026.

- **CPU, murphy's CPUs** (NAS `mace_cueq_latest`, `-n 8`,
  `FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/20261017`):
  - `tests/cpu`: 2454 passed, 8 skipped (for platform or checkout reasons).
  - 5 failed, and all 5 now pass:
    - 4 `test_output_chunks` failures were my runner's missing `PYTHONPATH`;
      the tests pass with it set;
    - `test_energy.py::test_compile_cache_key_includes_float64_energy`
      pinned "no instance forward without the float64 energy". Every
      private copy now binds its strain path, so the test now checks which
      forward's code is bound, and it passes.
  - The CPU PolarMACE isolation test runs there (graph_longrange installed)
    and passes.
- **GPU, murphy** (RTX 4000 Ada; the GPU was shared only with zwingel's
  untimed jobs):
  - full tier of `test_forces_gpu`, `test_polar_gpu`, `test_engine_gpu`,
    `test_tiny_engine_gpu` and `test_planner_gpu`: 38 passed;
  - `test_isolation_gpu`: 3 passed. POLAR-1-S again recompiled 21 graphs
    after the dynamo reset, and was within 1.9e-6 of its fresh value after
    it;
  - acceptance tier of the same files: 4 passed, 1 failed. The MH-1 NPT A/B
    against the research engine, which runs the strain path, passed. The
    failure is
    `test_mh1_plain_cueq_float64_against_the_research_engine`: the stage-B
    graph has 12 more kernels than the baseline's (249 vs 237). `main`
    14a5e20 fails it the same way on murphy (248 vs 236), so the gap is
    older than 2-T08b; it was reported to the conductor.
