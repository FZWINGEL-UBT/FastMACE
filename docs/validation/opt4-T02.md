# OPT4-T02: POLAR-1's small kernels and the float64 scatter (24 October 2026)

Plan task OPT4-T02 (`plans/opt-4.md` §3 candidate 2). Branch `fastmace/opt4-T02`, built on main `13a1983`.
Records and run scripts: `results/opt4-T02/`.

## Verdict

**Two opt-in records. Both pass the POLAR-1 gate: S and M on NaCl and NbCl5 2×2×2, and L on NaCl. L on
NbCl5 timed out cold-compiling on the Colab A100. Locally the pair cuts POLAR-1-S's MD step by 24 % on NaCl and
18 % on NbCl5; on the Colab L4 by 19 % and 16 %.** One record does the work:

- `fx_e3nn_linear` makes e3nn's `o3.Linear` traceable. PolarMACE's forward then compiles as one frame instead of
  running eagerly.
- `polar_graph_sums` removes the float64 `index_put` backward. It gains 8–9 % alone, nothing on top of
  `fx_e3nn_linear`, and makes those sums deterministic.

Both records are `never_automatic`; the owner decides defaults.

| Same-process A/B, POLAR-1-S, default (a) against both records (b) | NaCl b/a | NbCl5 b/a |
| --- | ---: | ---: |
| Local RTX 5060 Ti (sm_120), order a-b, ABAB | **0.7634** | **0.8225** |
| Local RTX 5060 Ti, order b-a, ABBA | **0.7589** | **0.8231** |
| Colab L4 (sm_89), order a-b, ABAB | **0.8184** | **0.8395** |
| Colab L4, order b-a, ABBA | **0.8108** | **0.8366** |

## 1. Where the kernels come from (the falsifying measurement)

**Tool:** `tools/module_profile.py` (new). It builds the planner's engine as `tools/step_breakdown.py` does, with
every default of the plan, and equilibrates 50 MD steps. It then profiles 5 eager force calls, the calls whose
kernels the B graph replays, with Python stacks. Each kernel is attributed to the compiled graph, the eager
Python frame or the autograd node that launched it. Dynamo's graph-break log is written beside it
(`TORCH_LOGS=graph_breaks`).

**Run:** local RTX 5060 Ti, WSL2, `~/environments/mace_cueq_oeq_polar` (torch 2.14.1+cu130, mace-torch 0.3.16,
cuEquivariance 0.12.0, OEq 0.7.0, graph_longrange 0.4.0, e3nn 0.4.4). POLAR-1-S (`MACE-POLAR-1-S.model`, sha256
`e4495612…`), hybrid float32, 3×3×3, NVT 1000 K. Source: main `13a1983`. One untimed reservation, 09:52–09:56 UTC.

| Kernels of one force call, by source | NaCl (216 atoms) | NbCl5 (1,944) |
| --- | ---: | ---: |
| Total | 3.60 ms, 1,226 kernels | 25.04 ms, 1,134 kernels |
| Compiled graphs (**31** separate graphs) | 2.57 ms (71 %), 682 | 17.91 ms (72 %), 597 |
| Eager autograd nodes other than `IndexBackward0` | 0.43 ms (12 %), 349 | 0.74 ms (3 %), 333 |
| `IndexBackward0`: float64 `x[batch]` gathers, a serial sort-based `index_put` | 0.30 ms (8 %), 56 | **2.28 ms (9 %)**, 56 |
| e3nn `o3.Linear` run as TorchScript (eager) | 0.19 ms (5 %), 116 | 0.49 ms (2 %), 114 |
| Custom ops (OEq, Triton, cuEq) | 0.10 ms, 6 | 3.58 ms, 17 |

**Cause** (graph-break log, frame `[9/0]`). e3nn's `o3.Linear` keeps its generated code as a TorchScript module
(`_compiled_main`). A pickled checkpoint stores TorchScript IR and restores it as TorchScript
(`CodeGenMixin.__setstate__`), whatever `mace.tools.compile.disable_e3nn_codegen` says. Dynamo cannot trace
TorchScript. The break falls inside a loop of FastMACE's rewritten `PolarMACE.forward`, so dynamo skips the whole
frame ("graph break in a for/while loop … falling back to eager"):

- the forward body runs eagerly;
- its submodules compile as 31 graphs;
- the hundreds of small `mul/add/div/copy/fill` kernels and their eager autograd nodes come from that body.

`IndexBackward0` is the backward of `scatter_sum(x.double(), batch, dim_size=1)[batch]`: the Fukui
normalisations, the predicted total charges and the barycentre. Every atom is accumulated into one row,
serially.

**Falsifier passed.** The plan asked for at least 0.4 ms (10 %) of NaCl with a changeable source. NaCl has 0.91 ms
of eager work with a changeable source (25 %, 538 kernels), and NbCl5 2.3 ms of the `index_put` backward. The
other suspects from the plan:

- `model._untraced_blocks`: empty; every block traces;
- the long-range k-space path: compiled once the frame compiles.

## 2. The records

| | `fx_e3nn_linear` | `polar_graph_sums` |
| --- | --- | --- |
| Module | `fastmace/rewrites/codegen.py` | `fastmace/rewrites/polar/graph_sums.py`, `fastmace/longrange/polar.py` (`GRAPH_SUM_EDITS`) |
| What | Each TorchScript `o3.Linear` gets e3nn's own fx graph back (`_codegen_linear` with the module's irreps, instructions and weight sharing); the weights are untouched | One graph: `scatter_sum(x, batch)[batch]` becomes `x.sum(0, keepdim=True).expand_as(x)`; `scatter_mean` becomes sum / N; `values[batch]` becomes `expand`. Several graphs: MACE's forms. A source rewrite of `PolarMACE.forward` with eight sites (six with PR 1762's options, whose Fukui normalisation takes the helpers itself), composable with PR 1762 and the float64 energy reduction |
| Exactness | rounding: eager, bitwise equal to the TorchScript on the CPU (31 of 31 Linears of POLAR-1-S, and the tiny model's energy and forces); compiled, Inductor fuses what ran eagerly | rounding: a reduction instead of atomics and the serial `index_put` |
| Deterministic | yes | yes (no atomics, no `index_put`) |
| Applies to | any model with a TorchScript `o3.Linear`; refused without one, and for Linears with `f_in`/`f_out` | PolarMACE whose forward takes the edits; refused otherwise; workload constraint `single_graph` |
| Pins | e3nn 0.4.4: `_codegen_linear`, `Linear.__init__`, `Linear.forward`, `CodeGenMixin.__setstate__` | `fastmace.longrange.polar`'s own pins (its `SOURCE_HASHES`, MACE's unwrapped originals) and the edit's site counts |
| Config | `RewriteConfig.fx_e3nn_linear`, `never_automatic` | `RewriteConfig.polar_graph_sums`, `never_automatic` |

**Copies.** e3nn pickles its code as TorchScript, so a deep copy of a rewritten model runs TorchScript again,
with the same values. The force field applies records to its private copy right before `compile_model`
simplifies and compiles it, after the last copy.

**Tests:**

- `tests/cpu/test_fx_e3nn_linear.py` (6): fields; applicability and refusals; CPU bitwise equality; the
  self-check passes and catches a wrong graph; dynamo traces the rewritten Linear inside a loop with
  `fullgraph=True` and fails on the TorchScript one.
- `tests/cpu/test_polar_graph_sums.py` (13): the helpers' values and gradients against MACE's forms (bitwise for
  several graphs); fields; every source site with and without PR 1762 and the float64 reduction; refusal of
  another source and of non-polar models; the self-check and a wrong helper; qualification after polar's wrappers
  are installed; a tiny PolarMACE with and without PR 1762's regularisation giving the same energy and forces
  (1e-12).

**Bug found by the first gate, fixed in `9b9e3e0`.** `polar_graph_sums` first pinned `PolarMACE.forward`.
Once the force field installs polar's class-level wrappers, the registry hashes the wrapper, and the record was
refused. A regression test covers it.

## 3. Profiles with the records (untimed, `9b9e3e0`, 11:03–11:09 UTC)

Same probe, 5 eager force calls each:

| POLAR-1-S | default | `fx_e3nn_linear` | `polar_graph_sums` | both |
| --- | ---: | ---: | ---: | ---: |
| NaCl kernel time | 3.60 ms | 2.73 ms (−24 %) | 3.29 ms (−8 %) | 2.74 ms |
| NaCl kernels / compiled graphs | 1,226 / 31 | 642 / 5 | 1,166 / 31 | 652 / 5 |
| NbCl5 kernel time | 25.26 ms | 20.76 ms (−18 %) | 22.93 ms (−9 %) | 20.70 ms |
| NbCl5 kernels / compiled graphs | 1,134 / 31 | 560 / 5 | 1,074 / 31 | 569 / 5 |

- With `fx_e3nn_linear`, the eager autograd work and the eager e3nn Linears are gone. The `IndexBackward0` is
  compiled into Inductor's backward, so `polar_graph_sums` then adds no speed.
- Force difference on the starting structure, against the default: `fx_e3nn_linear` up to 1.6e-6 (NaCl) and
  6.3e-4 eV/Å (NbCl5); `polar_graph_sums` 1.5e-6 and 4.8e-7. NbCl5's 6e-4 lies within two float32 paths that are
  each about 3e-4 eV/Å from float64 there (gate below).

## 4. POLAR-1 gate (both records on)

`tools/force_stress_gate.py --engine fastmace --rewrites fx_e3nn_linear=on polar_graph_sums=on`:

- references: `mace_cueq_latest`, float64 and the float32 yardstick, each in a child process;
- production: `mace_cueq_oeq_polar`;
- 3 configurations per cell mode;
- NbCl5 at 2×2×2: the 3×3×3 float64 reference needs 18.6 GB (phase 1b);
- source `9b9e3e0`, local RTX 5060 Ti, untimed reservations, 10:40–11:30 UTC.

Ratio = production error / float32 yardstick error against float64; the limit is 2, or the floor.

| Label, structure | Energy | Forces | Stress | Pass | Phase 1b without the records |
| --- | ---: | ---: | ---: | --- | --- |
| POLAR-1-S, NaCl | 0.28 | 1.29 (6.0e-6 eV/Å) | 0.52 | yes | 0.28 / 1.08 / 0.53 (local subset) |
| POLAR-1-S, NbCl5 2×2×2 | 1.04 | 1.19 (3.6e-4 eV/Å) | 1.40 | yes | – |
| POLAR-1-M, NaCl | 1.13 | 1.11 | 0.86 | yes | 1.44 / 1.15 / 0.81 (murphy) |
| POLAR-1-M, NbCl5 2×2×2 | 1.07 | 1.01 | 0.68 | yes | – |
| POLAR-1-L, NaCl (local, 13:04–13:12 UTC) | 0.72 | 0.91 (4.8e-6 eV/Å) | 0.95 | yes | 0.93 / 0.91 / 0.94 (leto) |
| POLAR-1-L, NbCl5 2×2×2 (Colab A100) | – | – | – | **not run to the end** | 1.12 / 0.98 / 0.70 (leto) |

**POLAR-1-L on NbCl5 2×2×2 (Colab A100) timed out cold-compiling at the 30-minute cap** (`fm-opt4-t02-a100-gate-114430`,
NVIDIA A100-SXM4-40GB, sm_80):

- The four references (NaCl and NbCl5, float64 and float32) finished in 3.5 min.
- The production child was still compiling 22 min later: there was no sm80 compile cache for POLAR-1-L, and
  every POLAR engine build resets dynamo (below).
- The conductor ruled out a rerun (Colab budget), so POLAR-1-L is gated on NaCl only. A default proposal is
  scoped to POLAR-1-S and POLAR-1-M.

**MH-1** (`mh1-r2scan`, NaCl, `fx_e3nn_linear=on`, the conductor's check): refused at plan time with "no e3nn
o3.Linear runs TorchScript". The hybrid MH-1 model has no such Linear, so the record has nothing to change there
and cannot regress it. Under `auto` it stays off.

## 5. Timed A/B

`tools/ab_rewrites.py`, same process:

- POLAR-1-S, NVT 300 K;
- side a is the default; side b is `fx_e3nn_linear=on polar_graph_sums=on`;
- 12 blocks of 300 steps, both build orders (a-b ABAB, b-a ABBA);
- pinned source `9b9e3e0` on ext4;
- the conductor's quiet windows on the local RTX 5060 Ti.

| Run | Structure | a (ms/step) | b (ms/step) | b/a | Valid |
| --- | --- | ---: | ---: | ---: | --- |
| 11:30 UTC, a-b, warm-up 300 | NbCl5 | 25.914 | 21.313 | **0.8225** | yes (pairs 0.818–0.823) |
| 11:37 UTC, b-a, warm-up 300 | NbCl5 | 25.830 | 21.259 | **0.8231** | yes (pairs 0.818–0.826) |
| 12:00 UTC, a-b, warm-up 3000 | NaCl | 3.935 | 3.004 | **0.7634** | yes |
| 12:02 UTC, b-a, warm-up 3000 | NaCl | 3.947 | 2.996 | **0.7589** | yes |
| 11:30 and 11:37 UTC, warm-up 300 | NaCl | 3.947 / 3.934 | 3.003 / 2.985 | 0.761 / 0.759 | **no**: one compile during timing (block 6, a new edge size, once on each side); the other 11 pairs 0.746–0.767 |

- The step times agree with the plan's baseline: 3.914 ms on NaCl at 1000 K.
- The gain is far outside the local noise floor (1.1 % NaCl, 0.4 % NbCl5).

**Colab L4** (`tools/colab`, session `fm-opt4-t02-l4-ab-120416`, NVIDIA L4 sm_89, driver 580.82.07, the pinned
hybrid stack of `tools/modal/requirements-hybrid.txt`, source `9b9e3e0`):

- one VM, both orders, NaCl and NbCl5, warm-up 3000;
- every run valid.

| Order | NaCl a / b (ms/step) | NaCl b/a | NbCl5 a / b (ms/step) | NbCl5 b/a |
| --- | --- | ---: | --- | ---: |
| a-b, ABAB | 4.389 / 3.592 | **0.8184** (pairs 0.815–0.821) | 37.433 / 31.425 | **0.8395** (0.836–0.843) |
| b-a, ABBA | 4.442 / 3.602 | **0.8108** (0.808–0.817) | 37.394 / 31.284 | **0.8366** (0.835–0.837) |

L4 numbers compare only within this container (OPT-T10).

## 6. Compute and time

- **Local GPU:** module profiles, 4 S/M gates (2 refused by the pin bug before computing anything and rerun),
  the MH-1 check, and 3 quiet A/B reservations.
- **Colab:** 4.11 CU in all (balance 45.56 → 41.45; the two VMs overlapped 12:04–12:20 UTC, so the per-job deltas
  3.34 and 2.45 overlap). The first A100 call (10:14 UTC) got no VM and used nothing.
  - A100 gate, `fm-opt4-t02-a100-gate-114430`: 35.6 min wall, 4.6 min of it setup.
  - L4 A/B, `fm-opt4-t02-l4-ab-120416`: 46.0 min wall, 8.2 min of it setup.
  - Both VMs stopped by the driver.

## 7. Left for the owner and the conductor

- **Finding: cold compile cost of POLAR builds.** `fastmace.longrange.polar`'s process constraint resets dynamo
  before every PolarMACE force field is compiled (`PROCESS_CONSTRAINTS`). So each engine a tool builds (the gate
  builds one per cell mode and structure) compiles from nothing. On a VM without caches that is more than 20
  minutes for POLAR-1-L. This is a setup-time question for OPT4-T05, not for these records.

- **Defaults.** `fx_e3nn_linear` is a candidate for a default wherever it applies. It gains on sm_120 and on
  the L4, rounding class, gate passed on POLAR-1-S and -M (POLAR-1-L on NaCl only), and it is a no-op where the
  model has no TorchScript Linear (MH-1 hybrid).
  `polar_graph_sums` is worth having only for determinism once `fx_e3nn_linear` is on.
- **Other families.** Any model loaded with e3nn TorchScript Linears on the executed path takes
  `fx_e3nn_linear`. Plain-e3nn models are rebuilt with codegen off by `compile_model` already. Not screened
  beyond POLAR and MH-1.
- **Reproducible mode.** Not run for these records. `polar_graph_sums` removes atomics; `fx_e3nn_linear` changes
  only what Inductor compiles.
