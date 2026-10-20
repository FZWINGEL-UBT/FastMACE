# 2-T14e: reusing float64 reference results in the GPU test tiers

Task 2-T14e (owner's request, 20 October 2026): measure how much of the full and acceptance GPU
tiers goes into reference work that does not test FastMACE's current code, then propose how to reuse
it. This is a measurement and a proposal only: nothing in the library or the tests changed.

**Recommendation in short:**
- **Full tier:** no on-disk cache. Only about 2 % of it is cacheable reference work. Removing one
  duplicate (MACE's LBFGS minimum computed twice) recovers about half of that.
- **Acceptance tier:** about 48 % of its time is reference work that does not depend on FastMACE's
  code: ASE's integrators with MACE's calculator, and the research engine's runs. A test-only on-disk
  reference cache would roughly halve the tier at merge time. Phase acceptance keeps computing
  everything fresh. Build it only if acceptance runs at merges are frequent enough to pay for it.

## Setup

| Item | Value |
| --- | --- |
| Machine | murphy, RTX 4000 Ada, NAS `mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuequivariance 0.12.0); hybrid children in `mace_cueq_oeq` |
| Source | FastMACE `main` e24d48e, own copy and caches under `/mnt/homes/zwingel/fastmace-2-T14e/` |
| GPU | **shared**, untimed: processes of our own account ran throughout (2-T21's float64 anneals among them); no other user's process |
| Runs | full tier `-m "not acceptance" tests/gpu` (244 passed, 83 min); acceptance subset `-m acceptance` of `test_engine_gpu.py`, `test_forces_gpu.py`, `test_selfcheck_gpu.py` (7 passed, 72 min) |
| Raw files | [results/2-T14e-murphy-20261020/](../../results/2-T14e-murphy-20261020/README.md) |

Because the GPU was shared, the absolute times are inflated, the float64 parts most of all. The
proposal therefore uses **shares** (the reference part of a test as a fraction of that test's time).
For absolute tier times it uses the earlier, unshared-or-less-shared junit files: eight full-tier
trial runs on murphy on 20 October (median sum 37.9 min) and 1b-T27's acceptance run (103 min).

### How the time was attributed

A timing hook (`scripts/timing_hook_sitecustomize.py`) ran in a scratch copy as its `sitecustomize.py`.
It is not part of the repository: every test child process runs with the repository root on its path,
so it reached all of them. It wraps the reference entry points when they are imported, so the
"MACE imported first" checks see an unchanged process, and it logs per test (`PYTEST_CURRENT_TEST`,
inherited by children):

- `MACECalculator.calculate`: MACE's own calculator;
- `fastmace.planner.selfcheck.references`: the gate's references (they run in `-m` child processes);
- `fastmace.reference.calculators.load` and `ReferenceCalculator.evaluate`: the reference tier;
- each process's wall time, flagged when it imports the research engine (`langevin.gpu_langevin`).

`scripts/attribute.py` adds these up per test against the junit times. Nested calls are counted once.

## What counts as reference work

| Kind | Tests the current code? | Cacheable by a test-only cache |
| --- | --- | --- |
| MACE's own calculator (`MACECalculator`, float64 or float32) | no | yes |
| ASE's integrators with MACE's calculator (ASE equivalence) | no | yes |
| Research-engine runs (`FASTMACE_BASELINE`, the frozen tag) | no | yes |
| Gate references (`selfcheck.references`) | **yes**: the gate and its reference path are under test | no; the owner also decided against a self-check cache (checkpoint 2) |
| Reference-tier runs in `test_reference_tier_gpu.py` | **yes**: the reference tier is under test | no |
| FastMACE float64 runs used as comparisons (plain-cuEq trajectories, hand-built engines, float64 force fields) | **yes** | no, they must rerun |

## Full tier

Instrumented run (shared GPU), shares of the tier's total time:

| Kind | Seconds | Share |
| --- | ---: | ---: |
| Gate references | 329 | 6.6 % |
| MACE's own calculator | 80 | 1.6 % |
| Reference-tier load and evaluate (`test_reference_tier_gpu.py`) | 48 | 1.0 % |
| Research-engine processes | 23 | 0.5 % |
| All reference work | 479 | 9.6 % |
| **Cacheable** (MACE calculator + research engine) | **103** | **2.1 %** |

Applied to the median full-tier run (37.9 min), the cacheable part is about **0.8 min per run**. The
gate references (6.6 %) test the gate itself and stay. The largest items:

| Test | Reference s / test s | Kind |
| --- | ---: | --- |
| `test_lbfgs_with_fm_calc_reaches_maces_minimum[hybrid]` and `[pytest]` | 24.8 / 70.9, 24.2 / 69.6 | MACE's LBFGS minimum, the same 42 calls in both |
| `test_mh1_agrees_with_maceforces[cueq-float64-eager-plain]` | 22.9 / 48.2 | research engine |
| `test_loading_gpu.py::test_mh1_plain_cueq_float64` | 11.1 / 15.4 | MACE calculator |
| `test_polar_gpu.py::test_polar1_s_float64_forces_and_stress` | 8.7 / 12.4 | MACE calculator |
| Gate tests (`test_selfcheck_gpu.py`, `test_eager_gpu.py`, `test_tiny_released_gpu.py`) | 19–66 each | gate references (stay) |

## Acceptance tier

Shares from the instrumented acceptance subset (shared GPU), and from the ASE-equivalence JSONs
(`results/1b-T09-murphy-20261019/`), applied to 1b-T27's acceptance times on murphy:

| Test | 1b-T27 s | Reference share | Reference s | Kind |
| --- | ---: | ---: | ---: | --- |
| `test_ase_equivalence_gpu.py::test_langevin_reproduces_ase_in_every_50_K_bin` | 3749 | 53 % of the two runs (ASE 1990 s, FastMACE 1738 s) | 1990 | ASE Langevin + MACE |
| `test_ase_equivalence_gpu.py::test_baoab_variants_reproduce_their_references` | 936 | 47 % (ASE 220 s, corrected reference 220 s) | 440 | ASE BAOAB + MACE, reference integrator |
| `test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine` | 365 | 43 % | 158 | research engine |
| `test_engine_gpu.py::test_mh1_plain_cueq_float64_npt_against_the_research_engine` | 259 | 44 % | 114 | research engine |
| `test_engine_gpu.py::test_mh1_hybrid_float32_against_the_research_engine` | 202 | 62 % | 125 | research engine |
| `test_forces_gpu.py::test_mh1_agrees_with_maceforces[hybrid-float32-compiled-production]` | 96 | 60 % | 58 | research engine |
| `test_forces_gpu.py::test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]` | 80 | 58 % | 46 | research engine |
| `test_canonical_restart_gpu.py::test_mh1_hybrid_resumes_statistically` | 347 | 0 % | 0 | FastMACE both sides |
| `test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[omol0-xl]` | 73 | 27 % | (20) | gate references (stay) |
| `test_selfcheck_gpu.py::test_gate_time_on_nbcl5` | 47 | 19 % | (9) | gate references inside a timing record (stay) |
| **Total** | **6154** | **48 %** | **2931** | cacheable |

So a reference cache would take about **49 of the tier's 103 minutes**. Two thirds of that is one
item: ASE's 18,000-step Langevin run with MACE's calculator (33 min).

The research-engine shares come from processes that compiled into a cache that was partly cold, on
both sides, and from a shared GPU. They are shares, not timings; an idle rerun would move them by a
few percentage points, not change the conclusion.

## Proposed design: a test-only on-disk reference cache

Only for reference work that does not run FastMACE's code: MACE's calculator, ASE's integrators with
it, and the research engine. Not for gate references, the reference tier, or any FastMACE comparison
run.

- **Where:** `tests/_support/reference_cache.py`, with the values under `<FASTMACE_CACHE>/test-references/`
  (on the workstations under `/mnt/homes/zwingel`). No library code reads it.
- **What is stored:** the reference's outputs as the test consumes them, as `.npz` plus a JSON
  manifest: the ASE run's per-step temperature and potential energy (or the binned means), the research
  engine's positions and observables, MACE's single points and LBFGS minimum.
- **Key:** a SHA-256 of a canonical JSON of everything the value depends on:
  - the reference record id (test node id plus a version the test bumps when it changes what it
    computes);
  - the checkpoint's SHA-256 and head;
  - the versions of MACE, torch, cuEquivariance (and its ops package), e3nn, ASE, and for the research
    engine the baseline tag's commit (d2faa9fd) and the export's file hashes;
  - for BAOAB's corrected reference, which runs FastMACE's own `fastmace/reference/integrators.py`
    with MACE's calculator, the SHA-256 of that file, so that a change there recomputes it;
  - the structure (numbers, positions, cell, pbc) hash, dtype, seed and run spec (steps, timestep,
    temperatures, friction, pressure);
  - the device kind: GPU name, compute capability, driver and CUDA runtime versions (float64 cuEq is
    not bitwise between kernel choices, so a value from another GPU is not reused).
- **Invalidation:** any key change misses. A manifest also records the creation time and host;
  entries older than a set age (for example 30 days) are recomputed.
- **Opt-out:** `pytest --fresh-references` (and `FASTMACE_FRESH_REFERENCES=1` for child processes)
  computes everything and overwrites the entries. **Phase acceptance (2-T23 and its successors) runs
  with it**, so every phase is still checked against freshly computed references; merge-time
  acceptance runs may use the cache.
- **Concurrency:** write to a temporary name and rename, under FastMACE's soft lock, so two workers
  computing the same entry cannot corrupt it.

**What it loses.** Within one key, the cache cannot see a change in the reference's own numbers: MACE
float64 drift from something the key does not cover (a rebuilt environment with the same versions, a
different BLAS or cuBLAS, a nondeterministic kernel), or a research-engine export that was changed
without a new tag. The fresh run at phase acceptance and the age limit bound how long such a change
can go unnoticed. The cache also hides a reference that has become slower or broken, until the next
fresh run.

**Saving.** About 49 min of the acceptance tier's 103 min on each merge-time acceptance run with a
warm cache. The full tier would save about 0.8 min of 38, which does not justify the cache there.

## Cheaper wins found

1. **MACE's LBFGS minimum is computed twice per full-tier run.** Both parametrisations of
   `test_ase_calculator_gpu.py::test_lbfgs_with_fm_calc_reaches_maces_minimum` relax the same model
   and structure with `MACECalculator` (42 identical float64 calls, about 24 s each on the shared GPU).
   Computing it once per run (one child that returns the minimum for both, or a session-scoped result)
   saves about 24 s, half of the full tier's cacheable time, with no cache and no loss of protection.
2. **The PolarMACE float64 single point is computed three times** across
   `test_polar_gpu.py::test_polar1_s_float64_forces_and_stress` and
   `test_polar1_s_hybrid_float32_in_a_fresh_process` (same model, positions and dtype). The saving is
   small (about 4 s).
3. **The ASE Langevin reference dominates the acceptance tier** (33 min). Before a cache, check
   whether it can run with fewer steps or a smaller cell. Each 50 K bin needs enough samples for the
   0.1 K and 0.01 meV/atom limits, so this needs a statistics check first. It would make every
   acceptance run cheaper, the fresh phase-acceptance run included.

No reference was computed twice in the acceptance subset.

## Win 1, done

`test_ase_calculator_gpu.py`: MACE's float64 LBFGS minimum now runs once per pytest process (a cached
child of this environment's Python), and each parametrisation's child relaxes with `fm.calc` only.
murphy (shared GPU): 2 passed, maximum displacement from MACE's minimum 5.8e-7 and 6.9e-7 Å (limit
5e-3). The PolarMACE "triple" stays: two of its three calls are different computations (the released
and the enabled path), and the one repeat costs 0.4 s.

## Shortening the ASE Langevin reference (owner: "shorten first")

The test `test_langevin_reproduces_ase_in_every_50_K_bin` runs 18,000 steps (heat 0 → 400 K in 2,000,
cool 400 → 0 K in 16,000; SPEC §12.2 says "NaCl 9 ps") twice, with ASE and with FastMACE, and takes
62 min, half of it ASE.

**It is not a statistical test.** Both sides use one seed and FastMACE reproduces ASE's noise stream,
so the two trajectories are the same to rounding. In 1b-T09's run the largest bin differences were
3.5e-10 K and 3.8e-11 meV/atom, against limits of 0.1 K and 0.01 meV/atom, and the final positions
differed by 1.0e-11 Å after 18,000 steps. Bin means do not need many samples here: every bin compares
the same steps of two identical trajectories. A difference in the integrator shows from the first
steps on.

**Power check** (`results/2-T14e-murphy-20261020/langevin-power/`, murphy CPUs, no GPU). ASE's
Langevin plays both sides on a Lennard-Jones fcc solid (256 atoms, epsilon 0.1 eV, solid to 400 K),
with the test's timestep, friction, `fixcm` and ramp shape and the test's bin metrics. Each run pairs
a baseline with a variant that carries one injected difference, from the same seed:

| Schedule (heat + cool steps) | friction +1 % | target one step late | T +0.2 % | forces ×(1+1e-6) | forces ×(1+1e-9) | final T |
| --- | --- | --- | --- | --- | --- | ---: |
| 2,000 + 16,000 (now) | caught (1.08 K) | caught (0.163 K) | caught (0.72 K) | missed (1e-4 K) | missed | 24.8 K |
| 2,000 + 8,000 | caught (1.14 K) | caught (0.163 K) | caught (0.65 K) | missed | missed | 49.0 K |
| 2,000 + 4,000 | caught (1.16 K) | caught (0.163 K) | caught (0.58 K) | missed | missed | 95.4 K |
| 2,000 + 2,000 | caught (1.20 K) | caught (0.163 K) | caught (0.54 K) | missed | missed | 154 K |
| 1,000 + 1,000 | caught (1.19 K) | caught (0.235 K) | caught (0.41 K) | missed | missed | 147 K |

(The values are the largest bin |dT|; the dPE column behaves the same way and is in `results.jsonl`.
The unchanged pair gives exactly 0 at every length.)

- Every schedule catches the same differences, by the same margins. The weakest catch, the
  one-step-late target, is 1.6 times the limit at 18,000 steps and the same at 4,000.
- Length does not amplify small differences: a 1e-6 force error stays at about 1e-4 K in every
  schedule (the Langevin friction damps it), so no length catches it under the current limits.
- The only assertion that depends on length is "the cooling ramp really cooled it" (final T < 50 K).
  The stand-in's 24.8 K at 18,000 steps matches MH-1's 23.3 K in 1b-T09, as the thermostat's lag
  (friction 0.02 per ASE time unit, about 0.5 ps) predicts for any solid: final T ≈ cooling rate × 0.5
  ps.

**Proposed step count.**
- **Recommended: 4,000 steps** (heat 2,000 + cool 2,000, 2 ps). The test drops from about 62 to about
  14 min (both sides scale with steps: ASE about 110 ms/step, FastMACE about 97 ms/step with a sync
  every step), saving about 48 min of the acceptance tier. All 16 bins remain, with 250 steps each.
  The final-temperature check becomes a ramp check that does not depend on length: the final T is
  below half the peak (about 154 K expected, 200 K limit) and both sides end at the same T, which the
  bins check anyway. SPEC §12.2's "NaCl 9 ps" becomes "NaCl 2 ps" (a conductor amendment).
- **Conservative: 12,000 steps** (cool 10,000) keeps every assertion as it is (final T about 40 K
  expected; 49 K at cool 8,000 is too close to 50 K). It saves about 21 min.
- **Optional, independent of the length:** the limits are 10^8 times the agreement observed (0.1 K
  against 3.5e-10 K). Limits of 1e-6 K and 1e-7 meV/atom would also catch a 1e-6 force error, at any
  length, and still leave 3,000 times the observed agreement. That is a SPEC §12.2 change too, so it
  is listed only as an option.

Before the test changes, one MH-1 run of the chosen schedule on an idle murphy GPU would confirm the
final temperature and the bin agreement on the real model. The stand-in shows the integrator-level
behaviour, not MH-1's.

## Next steps (for the conductor's decision)

- Win 1 is done (above).
- The owner chose to shorten first; the reference cache waits for that result. Choose 4,000 (recommended)
  or 12,000 steps, and amend SPEC §12.2. Then one MH-1 run of that schedule on an idle murphy GPU, and
  the test change.
- With 4,000 steps the acceptance tier falls from about 103 to about 54 min, and its cacheable part
  from about 49 to about 23 min (43 %: ASE Langevin 7 min, ASE BAOAB 7 min, research engine 8 min).
  Decide on the cache with those numbers.
