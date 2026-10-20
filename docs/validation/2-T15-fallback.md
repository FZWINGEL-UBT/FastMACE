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
