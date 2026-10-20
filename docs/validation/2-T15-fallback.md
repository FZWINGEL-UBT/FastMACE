# 2-T15: the fallback algorithm (acceptance 2)

Plan task 2-T15 (SPEC §9.4, §9.3 rules 13–14, §9.5; decision 29). Date: 20 October 2026.

## What runs

On a failure before the first step (build, compile, capture, self-check, or a failed gate, classified
by the step that raised it), `Simulation` calls `planner.fallback.Fallback` under
`FallbackConfig.fallback="auto"`:

1. rule 13 for a self-check failure (2-T04's reaction);
2. otherwise diagnosis on the reduced gate (one rattled copy, the failing quantity; references computed
   once, under `check="skip"` one copy's references at the first run that needs them): eager on the full
   rewrite set; eager passes → the graph mode is dropped; eager fails and the empty set passes → ddmin
   over the registry's switches with the `requires` closure, the minimal failing set and its dependants
   off;
3. every candidate validated by `activate` (self-checks and the full gate, or the one-copy gate under
   `check="skip"`) before it becomes active;
4. budget spent or fault not located: the terminal sequence (same mode with every rewrite off, eager
   with every rewrite off, the reference tier if `reference_fits`);
5. nothing passes: `FallbackExhausted` with every tried candidate and the budget.

The reduced-gate budget (64) counts the eager diagnosis run and the empty-set test; full gate runs
are counted apart. A resume rebuilds a stored fallback revision and never falls back.

## GPU: acceptance 2's injected cases (murphy)

- Machine murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03, idle GPU (the job waited
  until no process of any user was on it); untimed.
- Environment `mace_cueq_oeq` (hybrid cuEq/OEq), torch 2.14.1+cu130; MH-1 (`mh1-r2scan`) float32 on
  NaCl 3×3×3, NVT Langevin 300 K, `check="gate"`, 20 steps after setup.
- FastMACE commit `cb60a7f` (branch `fastmace/2-T15`); no baseline involved.
- Test: `tests/gpu/test_fallback_gpu.py` (`slow`, `acceptance`): 3 passed in 406 s. Records in
  `results/2-T15-murphy-20261020/`.

| Case (injected) | Result | Reduced / full gate runs | Setup |
| --- | --- | --- | --- |
| broken rewrite: readout 0.1 % off when `block_linear` is on | revision 1, graph: `block_linear` and its dependants `grouped_linear`, `merged_residual` off; gate passed (force 3.0e-6 eV/Å) | 8 / 1 | 116 s |
| broken pair: the same only when `joint_radial_input` and `grouped_linear` are both on | revision 1, graph: exactly the pair off; gate passed (3.1e-6 eV/Å) | 19 / 1 | 195 s |
| broken capture: captured stage B scales the forces by 1.001 | revision 1, eager, rewrites unchanged; gate passed in eager (2.6e-6 eV/Å) | 1 / 1 | 67 s |

Each report shows `plan revision 1 (fallback): <change>, reason <why>`.

## Finding: ddmin's minimal set is not unique under `requires`

The first murphy run (commit `f064d19`) injected the pair (`density_first`, `grouped_linear`).
ddmin returned {`grouped_linear`, `merged_residual`}: `merged_residual` requires `density_first`, so
that set's closure holds the pair too, and it is 1-minimal. The fallback was valid (gate passed,
`density_first` kept on), but it is not the injected pair. ddmin finds *a* minimal failing set (SPEC
§9.4), and a member that another record requires admits a second one. The GPU pair is now one that no
record requires. The algorithm is unchanged.

## CPU

- `tests/cpu/test_fallback_search.py`: ddmin, diagnose and terminal on oracles; `Fallback.run` on a
  tiny model with stubbed gates (single culprit, pair, culprit others require, capture, spent budget,
  exhaustion, reference fits or not, rule 13, `check="skip"` references once, pristine builds, no
  failing candidate activated).
- `tests/cpu/test_fallback_simulation.py`: `Simulation` with no failure never enters the fallback;
  `refuse`; broken capture with the report line; resume of a fallback revision; broken rewrite and
  pair through the real gate (slow).
- The CPU fast tier (`-m "not slow and not acceptance"`, `-n 2`) on the branch before the main merge:
  2284 passed, 232 skipped.

## 2-T15c: a failed candidate's memory is freed before the next build

An outside review found that the fallback kept each failure's exception, whose traceback frames held the
failed engine. Reproduced on the CPU with weakrefs (`tests/cpu/test_fallback_memory.py`: the first failed
engine was alive at every later build, a failed validation's engine at the next one). Fixed by
`fallback.detach` (type, message, chain and the traceback text kept; frames dropped), for every failure
the search keeps.

GPU check (`tests/gpu/test_fallback_gpu.py::test_a_failed_candidates_memory_is_freed_before_the_next_build`):
local NVIDIA GeForce RTX 5060 Ti 16 GB, driver 610.88, desktop holding about 4.6 GB, through
`tools/ci/local_gpu.py run`; `mace_cueq_oeq` hybrid float32, MH-1 on NaCl 3×3×3, the broken-rewrite
case; FastMACE commit `48878d2`; untimed; 1 passed in 281 s. `torch.cuda.memory_allocated` at the start
of each of the 9 builds after the first (8 eager diagnosis runs, the graph candidate), above the first
build's start; one engine is 156 MiB:

| Run | Extra at each later build |
| --- | --- |
| with `detach` | 64.0 MiB at every build (library workspaces, no engine) |
| control, `detach` disabled (as before 2-T15c) | 220.3 MiB at every build (the same 64 MiB plus the first, failed engine) |
