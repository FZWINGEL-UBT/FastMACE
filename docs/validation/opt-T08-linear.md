# OPT-T08: species tables and folded linears on the GPU (local RTX 5060 Ti)

Optimisation round 1, task OPT-T08 (`docs/design/opt-T08-linear-audit.md`). Three opt-in records,
merged to main in `e19f4fc`:
- `species_tables` (C1);
- `fold_product_linear` (C3, needs C1);
- `fold_readout_tail` (C2).

This note has the GPU results on the local card. murphy was not measured: another user had it on
21 October 2026, and workstation time is short before Thursday's power outage.

## Setup

| Item | Value |
| --- | --- |
| Machine | local PC, WSL2, NVIDIA GeForce RTX 5060 Ti (sm_120), driver 610.88 |
| Environments | `~/environments/mace_cueq_latest` (plain cuEq float64 check, gate references), `~/environments/mace_cueq_oeq` through the hybrid launcher (everything else): torch 2.14.1+cu130, cuequivariance 0.12.0 |
| FastMACE | main `f993713` (includes `e19f4fc`), pinned copy `~/fastmace-work/opt-T08/src-f993713` |
| Model | MH-1 (`mh1-r2scan`, `mace-mh-1.model`) |
| Run | `results/opt-T08-linear-audit/run.sh … local`, under the local GPU reservation (`tools/ci/local_gpu.py`), 11:18–11:58 UTC 21 October 2026, run `local-20261021` |
| Conditions | No other WSL compute process; 1-minute load 1.0–1.6. At the start, Windows showed 31 % utilisation and 5.2 GB on the card from other programs, which WSL's idle check cannot see. In every A/B job the blocks stayed within the spreads below, with no recompilation while timed |
| Raw data | `results/opt-T08-linear-audit/local-20261021/` (the gate's value arrays stay outside git) |

## Correctness

| Check | Result |
| --- | --- |
| First force evaluation, plain cuEq float64, MH-1 package and block linears on both sides, all three records on against off; NaCl and NbCl5 3×3×3 rattled | Forces on vs off: 9.4e-16 eV/Å (NaCl), 6.1e-15 (NbCl5). Off vs off in the same process: 7.9e-16, 3.9e-15. Energy: bitwise (NaCl), 1.1e-11 eV (NbCl5, the same as off vs off). The records are within the engine's own scatter. |
| The same, hybrid float32 (planner defaults + the three records) | Forces on vs off: 4.6e-7 (NaCl), 2.2e-6 eV/Å (NbCl5); off vs off 3.6e-7, 1.6e-6. Energy 7.9e-6 and 1.4e-4 eV (4e-8 and 7e-8 eV/atom: float32 rounding of the folded readout), far below the gate floor of 0.01 meV/atom. |
| Force, stress and energy gate (`tools/force_stress_gate.py --engine fastmace --rewrites species_tables=on fold_readout_tail=on fold_product_linear=on`); the production plan has all three on | **passed**. Ratios to the float32 calculator: NaCl energy 0.05, forces 1.13, stress 0.33; NbCl5 energy 0.19, forces 1.02, stress 0.65 (limit 2). |
| CPU (before the merge) | 37 tests in `tests/cpu/test_rewrites_species.py`: C1 bitwise; C2 and C3 within 1e-12 (energy) and 1e-10 (forces) relative in float64; a species change between evaluations; a refusal for each excluded model and the 19-label mapping |

## Speed (same-process A/B, 12 blocks of 300 steps after 300 warm-up steps, hybrid float32 NVT 300 K)

Both sides are planner defaults (with `gated_rows` and `segment_gemm`); side "on" adds the records named.

| Records on | Structure | Order, pattern | off (ms/step) | on (ms/step) | on/off | Block-pair ratios |
| --- | --- | --- | ---: | ---: | ---: | --- |
| all three | NaCl 216 atoms | off first, ABAB | 3.403 | 3.088 | **0.908** | 0.894–0.909 |
| all three | NaCl | on first, ABBA | 3.411 | 3.084 | **0.904** | 0.895–0.913 |
| all three | NbCl5 1944 atoms | off first, ABAB | 23.20 | 21.42 | **0.924** | 0.921–0.925 |
| all three | NbCl5 | on first, ABBA | 23.12 | 21.47 | **0.928** | 0.927–0.930 |
| C1 | NaCl | off first | 3.413 | 3.263 | 0.956 | 0.931–0.963 |
| C1 | NbCl5 | off first | 23.14 | 22.64 | 0.979 | 0.977–0.983 |
| C1 + C3 | NaCl | off first | 3.451 | 3.178 | 0.921 | 0.906–0.924 |
| C1 + C3 | NbCl5 | off first | 23.21 | 21.71 | 0.935 | 0.933–0.944 |
| C2 | NaCl | off first | 3.451 | 3.373 | 0.977 | 0.964–0.991 |
| C2 | NbCl5 | off first | 23.35 | 23.14 | 0.991 | 0.988–0.996 |

The local noise floor is 1.1 % on NaCl and 0.4 % on NbCl5 (2-T12).

All three records together save **9.2–9.6 % on NaCl and 7.2–7.6 % on NbCl5**, in both build orders. That
is more than the audit's kernel-time estimate (9 % and 5.7 %):
- the estimate counted only GEMM time;
- the records also remove the elementwise kernels, copies and concatenations around the products
  they replace.

Attribution (off-first runs; the parts overlap, so they do not add exactly):

| Record | NaCl | NbCl5 |
| --- | ---: | ---: |
| C1 (species tables) | −4.4 % | −2.1 % |
| C3 on top of C1 (fold `products.0.linear`) | −3.4 % | −4.4 % |
| C2 (fold the readout tail) | −2.3 % | −0.9 % |

## Reading

- **Passes on sm_120.** All three records are correct within the engine's scatter and pass the force
  gate. They make both crystals faster, well above the noise floor, in both orders.
- **The adoption rule (both GPUs) is not met yet.** murphy (sm_89) is not measured; it needs a slot
  after its current user, before Thursday evening.
- **Scope.** The records apply to the three MH-1 labels (C2 also to OMOL-0); every other label is
  refused by the records' own applicability checks. The 19-label gate in a later acceptance would
  confirm that.
