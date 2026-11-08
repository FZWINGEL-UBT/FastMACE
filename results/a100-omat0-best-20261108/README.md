# OMAT-0 medium on the A100: the H200 switches of 7–8 November (8 November 2026)

Branch `fastmace/a100-omat0-best` from main `1bd4f3be`. Owner, via the conductor: "Get a best configuration on a100
now for omat-0-medium based on the h200 results." Decision 66: a same-process MD-step A/B in both build orders on top
of the A100 defaults; rounding-class switches also pass `tools/force_stress_gate.py`; an A100 entry covers the family.

**Where it ran.** NHR@FAU Alex, NVIDIA A100-SXM4-80GB (driver 615.71.10), nodes a0533 and a0633, source `1bd4f3be`
(slim tar), FastMACE hybrid backend, float32, env `fm-offpolar/env` (Python 3.10.12, torch 2.14.1+cu130), run
unstaged (`STAGE=0`, see "Notes"). Benchmark_loup settings: OMAT-0 medium (`mace-omat-0-medium.model`), SrTiO3
10×10×10 (5,000 atoms), Langevin 300 K, 1 fs, skin 0.5; warm-up 200, 6 blocks of 256 steps per side.
b/a = candidate MD step over baseline, the two values = the two build orders (ab / ba); below 1 is faster.

| Job | What | Files |
| --- | --- | --- |
| 4963389 | pack 1: each switch group against the A100 defaults, both orders | `alex/pack1-4963389/` |
| 4963390 | pack 2: all ten on (side a), each turned off again (side b), both orders | `alex/pack2-4963390/` |
| 4963391 | force/stress gate, the A100 plan before this change | `alex/gate-default-4963391/` |
| 4963392 | spline/compose in-use check, then the gate with all ten on | `alex/gate-all-4963392/` |

Job files: [jobs/](jobs/) (`common.sh` holds the switch set, `pack1.env`, `pack2.env`, `gate.sh`).

## 1. Result

**All ten on: MD step b/a 0.7471 / 0.7469, 9.85 → 7.36 ms per step (1.97 → 1.47 µs per atom and step).** Adopted on
the A100 family: `NVIDIA A100-SXM4-80GB` added to the ten `*_GPUS` sets.

| Switch | Exactness | Pack 1, b = on, against the A100 defaults | Pack 2, b = off from all ten |
| --- | --- | --- | --- |
| `spline_forward.SPLINE_W` | rounding | 0.8654 / 0.8661 | – |
| `spline_forward.R128` | bitwise | with SPLINE_W 0.8491 / 0.8469 | with STAGE_EDGES 1.0325 / 1.0311 |
| `spline_forward.STAGE_EDGES` | bitwise | – | 1.0050 / 1.0044 |
| `wprime_backward.SPLINE_VECTOR` = 4 | bitwise | – | (2) 1.0122 / 1.0115 |
| `wprime_backward.STAGE_SH` | bitwise | – | 1.0079 / 1.0077 |
| `compose_skip.COMPOSE_SKIP` | rounding | 0.8027 / 0.9546 | – |
| `compose_skip.COMPOSE_FAST` | bitwise | – | 1.0240 / 1.0235 |
| `compose_skip.COUNT_PLAN` | bitwise | – | 1.0115 / 1.0130 |
| `cueq_projection.CUEQ_PROJECTED` | rounding | 0.9480 / 0.9945 | 1.0010 / 1.0020 |
| `species.GENERIC_FOLD_READOUT_TAIL` | rounding | 0.9861 / 0.9908 | 1.0097 / 1.0072 |
| all ten | | **0.7471 / 0.7469** | |
| `tabled_outer_sum.POWER_BACKWARD` (H200 only) | rounding | alone 0.9919 / 0.9969; all ten + it 0.7471 / 0.7492 | stays off on the A100 |

Every switch is faster in both orders when turned off again from the other nine, so each has its own A/B on top of
the rest. COMPOSE_SKIP's pack-1 orders differ (0.80 / 0.95) but it is part of every all-ten build, which are steady
(7.33–7.44 ms in eight builds across both jobs). `POWER_BACKWARD` gains nothing once the spline w′ is on.

Absolute steps (ms, side a / side b, ab order): defaults 9.83–9.87; all ten 7.36 (pack 1) and 7.33–7.35 (pack 2's a).

## 2. Accuracy

`tools/spline_w_check.py` with all ten (SrTiO3 10×10×10, rattled 0.05 Å): spline in use (731 intervals), COMPOSE_SKIP
in use; forces against the old plan max 1.0e-5 eV/Å (max force 7.07), the same build twice 2.4e-6.

`tools/force_stress_gate.py` (SrTiO3 3×3×3, NaCl, NbCl5; 3 configurations each; ratio to the ASE float32 error,
limit 3, energy floor 10 meV):

| Structure | Plan | Energy (eV) | Forces ratio | Stress ratio |
| --- | --- | --- | --- | --- |
| SrTiO3 | before | 4.7e-3 | 1.04 | 1.17 |
| SrTiO3 | all ten | 2.3e-3 | 0.97 | 0.69 |
| NaCl | before | 5.1e-4 | 0.83 | 1.19 |
| NaCl | all ten | 2.3e-4 | 0.89 | 1.04 |
| NbCl5 | before | 1.2e-3 | 1.02 | 2.74 |
| NbCl5 | all ten | 5.3e-4 | 0.97 | 0.97 |

Both pass; all ten is at or below the old plan everywhere except NaCl forces (0.89 against 0.83). The old plan's
NbCl5 stress (2.74, the accepted exception of 5 November) drops to 0.97.

## 3. Notes

- **Shared-cache bug, fixed in `alex_ab_pack.sh` v2.** The first submission (4963282–83, 4963330–31) failed in
  OpenEquivariance's host-extension build ("cuda_runtime_api.h: No such file"): every process points
  `<FASTMACE_CACHE>/oeq/cuda-runtime` at its own env, and with a staged (node-local) env two concurrent jobs sharing a
  cache left each other links into another node's `/tmp`. These jobs ran unstaged; v2 gives a staged job its own
  `oeq/` (docs/nhr-cluster.md §6.3).
- Setup per pack process: about 13–15 min for 1 + 8 builds (840–899 s wall per order).
