# 1b-T11: MP-0 medium speed gap and setup time

Plan task 1b-T11 (phase 1b), branch `fastmace/1b-T11`. This covers the two findings of the 1a acceptance
report ([phase-1a.md](phase-1a.md)):
- MP-0 medium ran 1.4–1.6 % slower than the research engine;
- FastMACE's setup took 4–33 s longer.

## Summary

| Finding | Cause | Disposition |
| --- | --- | --- |
| MP-0 medium 1.4–1.6 % slower (2.3 % in the B graph) | **A deviation from the research engine.** The research engine's `SEGMENTED_FIRST_SUM` also acts on the generic (MP-0) reassociated first interaction: the run-segmented Triton sum, and each receiver's edges grouped by sender species. FastMACE only had `segmented_first_sum` as an MH-1 record, so on MP-0 it stayed off. Its B graph runs Inductor's `zeros` + `index_add` instead of the segmented kernel. | Fix proposed: a `generic_segmented_first_sum` record, and sender grouping in the force field. With the fix, the same-process A/B gives 0.9988 (NaCl) and 1.0002 (NbCl5), and the B graphs have identical kernel lists. Pending the conductor's agreement, because it touches two modules. |
| Setup 4–33 s longer | **A measurement artefact of the 1a runs.** The FastMACE-only control processes compiled into an Inductor cache directory that no earlier process had used. The research-engine controls reused the cache that the A/B processes had warmed. The self-checks themselves cost 0.1–1.1 s per structure with a warm cache, and about 3.5 s in a cold first process. | No change. With equal cache state, FastMACE's setup is equal to the research engine's or shorter (table below). |

## Setup

- murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03.
- NAS `mace_cueq_oeq`: torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0, OEq 0.7.0. Hybrid FP32, TF32 off,
  `torch.compile` default.
- FastMACE `2da309e` (main). "Patched" runs use the same tree with the fix below, uncommitted at the time
  (`.source_commit` reads `2da309e+segmented-generic-experiment`).
- Baseline: tag `research-engine-baseline-20261017` (commit `d2faa9fd`), export `/mnt/homes/zwingel/fastmace-baseline/1a-T18`.
- Model `mp0-medium` (`2023-12-03-mace-128-L1_epoch-199.model`), and `mh1-r2scan` for setup. Structures: NaCl and
  NbCl5, 3×3×3.
- Runs: 18 October 2026, 22:29–23:04 UTC. Before every job, murphy had no compute process from any user
  (`idle_check.txt` in every run directory). Leto and the local GPU were not used.
- Files, caches and TMPDIR were under `/mnt/homes/zwingel/fastmace-work/1b-T11` and
  `/mnt/homes/zwingel/fastmace-cache/1b-T11-*`.

Tools (all in `results/1b-T11-probes/`; none is library code):
- `graph_probe.py` builds both engines in one process, as `tools/ab_step_time.py` does: same structure and
  seed, main.py's options, the A/B tool's Inductor custom-pass slot handling, and each side warmed up (300
  steps) right after its build. It then measures, in order:
  1. whole MD steps, 6 alternating blocks of 300 (ABAB);
  2. each side's ACP and B graphs replayed alone, 10 alternating blocks of 200 replays, timed with CUDA events;
  3. one replay of each graph under `torch.profiler`: every kernel, memset and memcpy with its count and GPU
     time (the research engine's `step_breakdown_probe.py` method).

  It also compares the two sides' starting forces. Dynamo's counters are checked across the timed part, and
  nothing compiled in any probe.
- `setup_probe.py` runs one engine per process (`--side`). It puts timers around:
  - load;
  - each rewrite's apply and self-check (FastMACE) or `generic_rewrites` / `block_linears` (research engine);
  - the model copy and `simplify`;
  - the initial force evaluation, which is the first compiled call (dynamo + Inductor);
  - each graph capture.

  `--check off` builds FastMACE without self-checks.
- `run_murphy.sh` (idle check, environment, detached start) and `queue_murphy.sh` (sequential jobs) run the jobs.
  `summarize_setup.py` prints the setup table.

Data: `results/1b-T11-murphy-20261019/<run>/` (`probe.json` or `setup.json`, `idle_check.txt`, `command.txt`).

## 1. MP-0 medium: the B graph

### Plans and rewrites

The plans agree, but the force fields do not. FastMACE's plan for MP-0 resolves `segmented_first_sum = on`, as
the research engine does: its metadata records `segmented_first_sum: true` for MP-0. At build, though, the field
reaches only the MH-1 record `segmented_first_sum`. That record requires MH-1's `reassociate_first` and is not
enabled for MP-0, so the force field's decisions read "this record stays off". `FixedPlanner.rewrite_resolution`'s
docstring says the research engine's switch is also a no-op beside `generic_reassociate`. That is not so:

- `gpu_langevin.generic_rewrites` builds the generic reassociation with `segmented=SEGMENTED_FIRST_SUM`;
- `MACEForces` sets `segmented_first_sum = (reassociate_first or generic_reassociate) and SEGMENTED_FIRST_SUM` and
  passes `sender_groups` to the neighbour list, so each receiver's edges are ordered by sender species.

Every other rewrite matches, module for module:
- species skip, reassociation, fused radial backward and streaming;
- `block_linear` and `grouped_linear`;
- `merged_residual`: off on both sides, since it needs `density_first`;
- `generic_interaction_forward` and `SpeciesSkip`, which are line-for-line copies.

### Graph node counts and kernels

| Structure | Build | Nodes ACP b/f | Nodes B b/f | ACP replay ratio | B replay ratio | Step ratio (6 × 300, ABAB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| NaCl (216 atoms, B at 7680 edges) | main | 73/73 | 190/190 | 0.9999 | **1.0231** | **1.0159** |
| NaCl | patched | 73/73 | 190/190 | 0.9999 | 1.0019 | 0.9988 |
| NbCl5 (1944 atoms) | main | 73/73 | 190/190 | 1.0003 | **1.0261** | **1.0228** |
| NbCl5 | patched | 73/73 | 190/190 | 1.0005 | 1.0000 | 1.0002 |

Ratios are FastMACE over the research engine. B-graph medians, main: NaCl 1.1994 / 1.2272 ms, NbCl5 8.5027 /
8.7249 ms. Patched: 1.2002 / 1.2024 ms and 8.5214 / 8.5218 ms.

The node counts are equal, but on main one kernel is replaced. Per replay, from the profiles:

| Kernel | NaCl b → f (µs) | NbCl5 b → f (µs) |
| --- | --- | --- |
| `kernel` (custom Triton kernels; the third call is the segmented first sum) | 3 calls 149.7 → 2 calls 137.8 | 3 calls 1273.0 → 2 calls 1187.8 |
| `triton_poi_fused_index_add_mul_unsqueeze_view_zeros` (FastMACE only) | 0 → 39.6 | 0 → 293.6 |
| `triton_poi_fused_zeros` | 1 call 0.8 → 2 calls 2.9 | 0.9 → 18.4 |
| sum of B kernels | 1214.5 → 1243.9 (+29.4) | 8527.8 → 8752.2 (+224.4) |

The unsegmented `edge_outer_sum` (zero fill + `index_add` over the edges) costs about 3.5 times the segmented kernel. On
NaCl it makes up the whole 29.4 µs difference of the B kernels; on NbCl5, 224 of the 224 µs. With the patch, both
B graphs launch the same kernels in the same numbers. The largest per-kernel difference left is ±2.5 µs (NaCl) and
±15 µs (NbCl5) of 1.2 and 8.5 ms, the run-to-run scatter of single kernels. The ACP graphs, which hold the
integrator, the neighbour check and the edge compaction, were equal throughout. This also explains why MH-1 was at
parity: its `segmented_first_sum` record does apply.

Starting forces of the two sides in one process, max |ΔF|:
- NbCl5: 1.8e-6 eV/Å on main, 9.0e-7 patched;
- NaCl: 3.4e-7 patched.

They are not bitwise equal. The segmented kernel and OEq's backward accumulate with atomics, so both engines are
FP32-nondeterministic in summation order. The patch halves the difference on NbCl5, as expected when both sides
now sum in the same order and in the same kernels.

### The fix (proposed; two modules)

1. `fastmace/rewrites/generic/__init__.py`: a record `generic_segmented_first_sum`:
   - config field `segmented_first_sum`, the same pattern as `generic_streaming` sharing `streaming_conv`;
   - requires `generic_reassociate`;
   - constraints `single_graph`, `fixed_species_superset`, `sort_receivers`; exactness "exact" (summation order
     only), as MH-1's;
   - apply: MH-1's `segment_first_sum` on the reassociated generic first interaction;
   - self-check: the generic interaction check.
2. `fastmace/model/forces.py`: `SEGMENTED_RECORDS = ("segmented_first_sum", "generic_segmented_first_sum")`. The
   neighbour list's `sender_groups` are set when either is enabled.
3. Tests (`tests/cpu/test_rewrites_generic.py`):
   - the record list;
   - the generic segmented sum is bitwise unchanged physics on the CPU, where the segmented forward is the same
     `index_add`;
   - it needs `generic_reassociate`;
   - regression: a hybrid MP-0 plan's rewrite resolution enables `generic_reassociate` and
     `generic_segmented_first_sum`, which turns on the force field's sender grouping.

CPU float64 is unchanged: the reassociation is enabled only for the hybrid graph engine, and on the CPU the
segmented forward is the same `index_add`. Hybrid MP-0 forces change at the FP32 rounding level, towards the
research engine's.

Still open after the fix is approved:
- the `FixedPlanner.rewrite_resolution` docstring;
- `docs/design/switch-map.md`, the §6.2 row of `segmented_first_sum`, which should name the generic record. That
  file belongs to 1b-T01.

## 2. Setup time

### Why 1a measured 4–33 s more

In 1a-T19 every process with a research-engine side ran under `oeq_no_nvcc.sh`. The launcher sets
`MACE_COMPILE_CACHE` to `…/oeq-launcher/mace`, and `tools/ab_step_time.py` then points the Inductor cache there
(`compile_cache.configure`). The A/B processes, which ran first, warmed that cache for both engines. Then:
- The research-engine control processes reused it. Their metadata shows FX-graph cache hits, for example MP-0:
  4 hits, 227 async-compile hits.
- The FastMACE-only controls ran without the launcher and without `compile_cache.configure`. They used the job's
  `TORCHINDUCTOR_CACHE_DIR=…/acceptance-murphy/inductor`, which no earlier process had filled for these models.

So the FastMACE controls compiled cold, and the research-engine controls mostly warm. The gap grew with the
model's compile time: MP-0 8 s, MH-1 18 s, OMOL-0 XL 33 s.

### Stage budget with equal cache state

One process per side, NaCl then NbCl5. Values are seconds.

- *Setup* is engine construction and includes the first compiled call.
- *Warm-up* is 300 steps, which include the graph captures.
- *Warm* means the second process with the same cache directory. *Cold* means a fresh cache directory per side.

| Model, cache | Side | Structure | Load | Setup | of which: self-checks | rewrites (apply) | model copy | first compiled call | Warm-up (captures) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MP-0, warm | baseline | NaCl | 24.9 | 14.90 | – | 0.92 | 0.62 | 12.30 | 8.67 (8.23) |
| | FastMACE | NaCl | 24.2 | **14.43** | 0.83 | 0.22 | 0.60 | 12.03 | 8.86 (8.42) |
| | FastMACE, `check=False` | NaCl | 24.1 | 13.15 | – | 0.24 | 0.60 | 11.36 | 8.63 (8.19) |
| | baseline | NbCl5 | | 11.63 | – | 0.45 | 0.42 | 10.74 | 3.00 (0.13) |
| | FastMACE | NbCl5 | | **11.56** | 0.10 | 0.07 | 0.41 | 10.93 | 3.08 (0.13) |
| MP-0, cold | baseline | NaCl | 24.6 | 30.53 | – | 0.92 | 0.41 | 27.46 | 16.94 (16.50) |
| | FastMACE | NaCl | 25.7 | **33.75** | 3.76 | 0.22 | 0.60 | 28.03 | 17.25 (16.81) |
| | baseline | NbCl5 | | 27.37 | – | 0.45 | 0.99 | 25.92 | 3.02 (0.14) |
| | FastMACE | NbCl5 | | **24.41** | 0.10 | 0.07 | 0.41 | 23.79 | 3.09 (0.13) |
| MH-1, warm | baseline | NaCl | 25.1 | 17.67 | – | 0.10 | 0.71 | 14.78 | 11.83 (10.79) |
| | FastMACE | NaCl | 24.7 | **16.63** | 1.05 | 0.15 | 0.49 | 13.97 | 12.24 (11.20) |
| | FastMACE, `check=False` | NaCl | 24.7 | 15.42 | – | 0.21 | 0.49 | 13.35 | 11.86 (10.82) |
| | baseline | NbCl5 | | 16.51 | – | 0.03 | 1.02 | 14.83 | 9.40 (0.26) |
| | FastMACE | NbCl5 | | **15.69** | 0.35 | 0.10 | 0.50 | 14.70 | 9.39 (0.26) |

Reading:
- **The first compiled call dominates setup on both sides:** 75–90 % of it, and twice as long with a cold cache.
  It is the same work on both sides, within 0.3–0.8 s.
- **Self-checks:**
  - With a warm cache: 0.8–1.1 s on a process's first structure, 0.1–0.35 s on later ones.
  - In a cold first process: 3.8 s (MP-0 NaCl). The first eager calls build their kernels and library handles
    there.
  - `check=False` saves 1.2–1.3 s on the first structure, which matches.
- **FastMACE's rewrites apply faster** than the research engine's `generic_rewrites`: 0.22 against 0.92 s on
  MP-0. This was not investigated further.
- **Load** (checkpoint, MACE conversion, OEq checks) is equal: 24–26 s on both sides.
- **Captures** are equal. NaCl's first B capture includes the first compilation at the padded edge count.
- Step times in these single processes agree within cross-process scatter. The MP-0 gap of section 1 shows here too:
  9.88 against 9.63 ms on NbCl5.

Conclusion: FastMACE's own setup overhead is the self-checks, 1 s warm or about 4 s cold per process, plus 0.1–0.35 s
per further structure. It is not 4–33 s. Speeding it up stays after 1b, as planned:
- qualification cached by checkpoint hash and software versions;
- the persistent OEq extension cache.

1b-T15's run report should show these stages: load, plan, rewrites, self-checks, model copy, first compiled call and
captures.

## Recommendations

1. Adopt the `generic_segmented_first_sum` fix. It restores the research engine's numerics and kernel path, and
   closes the gap.
2. For future setup comparisons, give both sides the same Inductor cache state: the same directory, or both fresh.
   An A/B tool change could make `ab_step_time.py`'s FastMACE-only processes use `MACE_COMPILE_CACHE` as the others
   do. That tool is not this task's file.
3. Correct the `rewrite_resolution` docstring and the switch map's `segmented_first_sum` row (T01's file) when the
   fix merges.
4. No other generic-path deviation was found: plans, rewrites, node counts and per-kernel times agree.
