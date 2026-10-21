# Native Ada A/B: fastmace-2 against current main (leeloo RTX 4000 Ada)

Conductor request, 22 October 2026. This is the native Ada (sm_89) data point the round-1 and round-2
defaults had not had on a workstation: murphy was in use by another user.

## Setup

| Item | Value |
| --- | --- |
| Machine | **leeloo RTX 4000 Ada** (sm_89), driver 595.84; the same card as murphy |
| Environments | NAS `mace_cueq_oeq` (A/B, production side of the gate), `mace_cueq_latest` (gate references) |
| Sides | `2`: the release `fastmace-2` (e52c482), exported as `fastmace_2` through its own minimal planner. `3`: main `5c48e26` plus the `--release 2` support in `tools/ab_fastmace.py` (`cf83d31`), defaults plus `EngineConfig(fused_integrator=True)` |
| Protocol | `tools/ab_fastmace.py --release 2`: one process, hybrid float32 NVT 300 K, 300 warm-up steps, 12 alternating blocks of 300 steps, MH-1, NaCl and NbCl5 3×3×3. Both build orders (ABAB, then ABBA) |
| Plan check | the plans differ only in `species_tables`, `fold_product_linear`, `fold_readout_tail` and `folded_backward_tiles` (allowed: defaults since `fastmace-2`). The engine options `overlap_rebuild` and `fused_integrator` are new since the release; nothing compiled and no patch changed while timed |
| Conditions | nvidia-smi showed no compute process before each job, load below 1.3; a watchdog would have stopped the run if another user had appeared (none did). 22:29–22:51 UTC, 21 October 2026 |
| Raw data | `results/ada-leeloo-20261022/` |

## Result

| Structure | Release first (ABAB) | Current first (ABBA) | Pair ratios (current / release) |
| --- | ---: | ---: | --- |
| NaCl 216 atoms | 3.360 → 2.994 ms, **0.891** | 3.405 → 3.020 ms, **0.887** | 0.884–0.900 |
| NbCl5 1944 atoms | 26.661 → 24.229 ms, **0.909** | 26.462 → 24.105 ms, **0.911** | 0.906–0.912 |

Current main with `fused_integrator=on` is **10.9–11.3 % faster on NaCl and 8.9–9.1 % faster on NbCl5**
than the phase 2 release on the RTX 4000 Ada. The difference is everything since `fastmace-2`: the
OPT-T08 records, `folded_backward_tiles`, `overlap_rebuild`, `fused_integrator` and the other round-2/3
changes on main. murphy's noise floor on the same card model is 0.16 % (1a-T18). No same-default floor
was measured on leeloo: `ab_fastmace.py` cannot time one version against itself.

## Force gate (current defaults)

`tools/force_stress_gate.py --engine fastmace` on main `5c48e26` with its defaults (no `--rewrites`):
**passed**.

| Structure | Energy ratio | Forces ratio | Stress ratio |
| --- | ---: | ---: | ---: |
| NaCl | 1.33 | 1.11 | 0.73 |
| NbCl5 | 0.30 | 1.00 | 0.86 |

The limit is 2.
