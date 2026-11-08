# OMAT-0 medium on the RTX 4000 Ada: the H200/A100 switches, and capacity (8 November 2026)

Owner request (8 November 2026, through the conductor): test the capacity of the newest, fastest configuration of
MACE-OMAT-0 medium. First, a same-process MD-step A/B on the RTX 4000 Ada of the ten H200/A100 switches of 7–8 November
(plus `JOINED_LAYOUT`, on for the H200 at main `73e421ea`). Then the capacity scan of
[omat0-capacity](../../docs/validation/omat0/omat0-capacity.md) with the new Ada defaults, on the Ada and the local
RTX 5060 Ti.

Branch `fastmace/omat0-ada-capacity`; source main `73e421ea` (A/B, gate) and `1cd369e4` (scan: the adoptions plus the
two capacity fixes below). GPU: leto's NVIDIA RTX 4000 Ada Generation (20 GB, driver 595.99.02); murphy's (same model)
for the first, inert pack. Jobs in [jobs/](jobs/), logs in [logs/](logs/).

## Summary

- **The ten switches were inert on the Ada as they stood.**
  - The spline family (`SPLINE_W`, `R128`, `STAGE_EDGES`, `SPLINE_VECTOR`, `STAGE_SH`) needs `HORNER`:
    `WPrime.attach_spline` requires table_w, horner and vector == 2. `HORNER_GPUS` had no Ada (alone a tie,
    1.004 / 1.001).
  - The compose family (`COMPOSE_SKIP`, `COMPOSE_FAST`, `COUNT_PLAN`, `JOINED_LAYOUT`) and `CUEQ_PROJECTED` are routed
    only inside the `tf32x3_gemms` rewrite (`rewrites/precision.py`). tf32x3 was off on the Ada (MH-1 measured slower).
    The Ada gets the Triton tf32x3 backend (`CUTLASS_CAPABILITIES` is sm_80 and sm_90).
  - The first pack measured them inert: `SPLINE_W` 1.0034, + `R128` 1.0001, `COMPOSE_SKIP` 0.9991.
- **With the prerequisites, the Ada set is 0.822 / 0.821 of the old Ada defaults: 26.2 → 21.5 ms per MD step**
  (Benchmark_loup, SrTiO3 10×10×10, 5,000 atoms).
  - Adopted for the Ada (decision 66; `NVIDIA RTX 4000 Ada Generation` added): `HORNER`, `SPLINE_W`, `R128`,
    `STAGE_EDGES`, `COMPOSE_SKIP`, `COMPOSE_FAST`, `GENERIC_FOLD_READOUT_TAIL`, and `tf32x3_gemms` for OMAT-0 medium
    only (`planner.fixed.DECISION_66_TF32X3_OMAT0_GPUS`; MH-1 keeps it off).
  - Measured and left off: `SPLINE_VECTOR` (4 slower), `STAGE_SH` (slower), `COUNT_PLAN` (tie), `JOINED_LAYOUT` (tie),
    `CUEQ_PROJECTED` (slower), CUTLASS tf32x3 on sm_89 (tie with Triton).
- **The force/stress gate passes** with the Ada set (`tools/force_stress_gate.py`, 3×, SrTiO3, NaCl, NbCl5; rule 13
  did not revise the plan). Its forces differ from the plain build's by at most 1.0e-5 eV/Å (same build 2.2e-6).
- **Two capacity bugs on main, found by the 5060 Ti scan, fixed on this branch.** Since `generic_wprime_backward`
  (3 November) the OMAT-0 limit had fallen from 53,240 to 34,295 atoms (SrTiO3) on the 5060 Ti. With the fixes it is
  163,840.
- **5060 Ti capacity (main `73e421ea` plus the fixes; the Ada adoptions do not touch the 5060 Ti's plan):** SrTiO3
  163,840 atoms (was 53,240 on 2 November), NaCl 195,112 (was 85,184). At 53,240 SrTiO3 the step is 291 ms
  (5.47 µs per atom) against 840 ms on 2 November.
- **Ada capacity with the new defaults: pending.** Both Adas (leto, murphy) were busy with other users' jobs; the scan
  and the old/new peak-memory comparison are queued on leto and follow in a second merge request.

## A/B (same process, both build orders)

`tools/ab_rewrites.py --pack`, OMAT-0 medium float32, Benchmark_loup SrTiO3 10×10×10. b/a below 1 means b is
faster. Packs of at most four entries (order ba holds every engine on the GPU at once).

### Stage 1: groups against the Ada defaults (b = group on)

| Entry | b | b/a ab | b/a ba |
|---|---|---:|---:|
| hsp | HORNER + SPLINE_W | 0.8396 | 0.8304 |
| hspr | + R128 | 0.8328 | 0.8312 |
| spline | HORNER + all five spline switches | 0.8306 | 0.8273 |
| t3 | tf32x3_gemms alone | 1.0438 | 1.0400 |
| t3c | tf32x3 + the compose group | 0.9902 | 0.9898 |
| t3ck | t3c with CUTLASS on sm_89 | 0.9900 | 0.9900 |
| tail | GENERIC_FOLD_READOUT_TAIL | 0.9974 | 0.9962 |
| all | spline + t3c + tail | 0.8219 | 0.8208 |
| allk | all, CUTLASS on sm_89 | 0.8198 | 0.8192 |
| nt3 | spline + tail, no tf32x3 | 0.8275 | 0.8248 |

"all": 26.2 → 21.5 ms per MD step.

### Stage 2: leave one out from "all" (b = that switch off; above 1 means the switch helps)

| Off again | b/a ab | b/a ba | Verdict |
|---|---:|---:|---|
| tf32x3 + compose group | 1.0117 | 1.0124 | on |
| COMPOSE_SKIP | 1.0572 | 1.0585 | on |
| COMPOSE_FAST | 1.0105 | 1.0104 | on |
| COUNT_PLAN | 1.0000 | 1.0004 | tie, off |
| JOINED_LAYOUT | 1.0006 | 0.9986 | off |
| CUEQ_PROJECTED | 0.9992 | 0.9987 | off |
| SPLINE_W + HORNER | 1.2003 | 1.2015 | on |
| GENERIC_FOLD_READOUT_TAIL | 1.0020 | 1.0026 | on |
| R128 (with STAGE_EDGES) | 1.0017 | 1.0004 | on |
| STAGE_EDGES | 1.0018 | 1.0028 | on |
| STAGE_SH | 0.9903 | 0.9905 | off |
| SPLINE_VECTOR (2) | 0.9918 | 0.9906 | off (2) |

## Force/stress gate

`tools/force_stress_gate.py` at 3× on SrTiO3, NaCl and NbCl5 against float64 references (ratio = the build's error
over the allowance; pass at or below the gate's limits). [jobs/gate2.sh](jobs/gate2.sh) `GATE=A`: the adopted set
(HORNER, the spline group with `SPLINE_VECTOR` and `STAGE_SH` measured in, `COMPOSE_SKIP`, `COMPOSE_FAST`, the
readout tail, `tf32x3_gemms=on`, `--require-compose`). Rule 13 revised neither plan.

| Plan | SrTiO3 forces / stress | NaCl forces / stress | NbCl5 forces / stress | Energy max abs (eV) | Verdict |
|---|---|---|---|---|---|
| Ada defaults (73e421ea) | 0.84 / 0.60 | 0.83 / 0.61 | 0.98 / 0.73 | 5.3e-4, 1.8e-4, 2.4e-4 | pass |
| Ada set (A) | 0.80 / 0.51 | 1.00 / 1.13 | 0.93 / 1.16 | 1.3e-3, 1.2e-4, 2.6e-4 | pass |

The in-use check before the gate (`tools/spline_w_check.py --rewrites tf32x3_gemms=on`, SrTiO3, rattle 0.05 Å):
spline and composed skip in use, joined layout not (as adopted); forces against the plain build 1.0e-5 eV/Å, same build
2.2e-6; energy −0.0033 eV of −39,724 eV.

## Capacity

`tools/bench/capacity_scan.py scan`, OMAT-0 medium float32 NVT, 20 warm-up and 200 timed steps per size, the largest
passing cubic repeat. Per the conductor: start at about the last maximum, grow ×1.25 until a size fails, bisect over
the repeat; nothing below a size known to fit. "est/peak": rule 12's estimate over the measured peak
(`max_reserved`).

### RTX 5060 Ti 16 GB (local PC, WSL2, `tools/ci/local_gpu.py run`)

Source `73e421ea` plus the two fixes ([logs/local-5060ti/scan-local-fix](logs/local-5060ti/scan-local-fix)); before
the fixes, main `eced7d75` ([logs/local-5060ti/scan-local-eced](logs/local-5060ti/scan-local-eced)).

| Crystal | Repeat | Atoms | Result | ms/step | µs/atom/step | Peak (GiB) | Estimate (GiB) | est/peak | Rescue rung |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| SrTiO3 | 22³ | 53,240 | ok | 291.2 | 5.47 | 5.17 | 8.08 | 1.56 | – |
| SrTiO3 | 24³ | 69,120 | ok | 383.1 | 5.54 | 6.56 | 10.42 | 1.59 | – |
| SrTiO3 | 26³ | 87,880 | ok | 507.3 | 5.77 | 8.51 | 10.03 | 1.18 | – |
| SrTiO3 | 29³ | 121,945 | ok | 729.7 | 5.98 | 11.02 | 13.06 | 1.19 | recompute_first_table |
| SrTiO3 | **32³** | **163,840** | **ok** | **1,286.3** | **7.85** | 14.59 | 17.47 | 1.20 | recompute_first_table |
| SrTiO3 | 33³ | 179,685 | out of memory | | | | | | |
| SrTiO3 | 35³ | 214,375 | out of memory | | | | | | |
| NaCl | 22³ | 85,184 | ok | 298.7 | 3.51 | 6.39 | 10.16 | 1.59 | – |
| NaCl | 24³ | 110,592 | ok | 419.5 | 3.79 | 8.18 | 9.74 | 1.19 | – |
| NaCl | 26³ | 140,608 | ok | 529.2 | 3.76 | 10.26 | 12.32 | 1.20 | – |
| NaCl | **29³** | **195,112** | **ok** | **808.9** | **4.15** | 14.11 | 17.01 | 1.21 | recompute_first_table |
| NaCl | 30³ | 216,000 | out of memory | | | | | | |
| NaCl | 32³ | 262,144 | out of memory | | | | | | |

- Before the fixes (main `eced7d75`): SrTiO3 34,295 (183.2 ms, 5.34 µs/atom, peak 3.54 GiB), 40,000 out of memory;
  NaCl 64,000 (218.8 ms, peak 4.90 GiB), 74,088 out of memory. The failures are the rewrites-off terminal plan.
- 2 November ([omat0-capacity](../../docs/validation/omat0/omat0-capacity.md)): SrTiO3 53,240 at 840 ms, NaCl 85,184
  at 623 ms.
- The estimate is 1.18–1.21 of the peak where the near-limit memory probe corrected it, 1.56–1.59 below that.
- On the 5060 Ti the spline and composed-skip paths are off (not in its GPU lists), so this scan does not answer
  whether they change peak memory; the Ada comparison does.

### RTX 4000 Ada 20 GB (leto): pending

Queued on leto ([jobs/chain-leto7.sh](jobs/chain-leto7.sh), [jobs/chain-leto8.sh](jobs/chain-leto8.sh)) with source
`1cd369e4`: the scan from the old maxima (SrTiO3 69,120 at 974 ms, NaCl 125,000 at 965 ms, leeloo, 2 November), then
the old and new plans' peaks at the same sizes (Benchmark_loup, SrTiO3 24³, NaCl 25³). Both Adas were taken by
other users' jobs on 8 November; the results follow in a second merge request.

## The capacity bugs

1. `attach_generic_wprime` raised ("no generic interaction with float32 radial tables and the chunked streaming
   convolution") when rule 12 took the rescue rung `generic_recompute_conv_weights` (B): every interaction then sits on
   the recompute path and none is a candidate. The planner fell back to the rewrites-off terminal plan, which runs out
   of memory. On the 5060 Ti the limit fell to SrTiO3 34,295 (peak only 3.54 GiB) and NaCl 64,000. Fix: it applies to
   none and returns no names when rule 12 put every candidate on the recompute path
   (`tests/cpu/test_generic_chunked.py::test_generic_wprime_stands_aside_on_the_rescue_rung`).
2. Rule 12 had no memory term for `generic_wprime_backward`, which, with the fused folded forward, stores no
   [E, 1280] weights. Without the term the planner predicted about 20.8 GiB at SrTiO3 40,000 (real about 4.1 GiB) and
   took rung (B) beyond about 35,000 atoms: twice the step cost, with the w′ backward off. Fix:
   `rewrites.memory.generic_wprime_backward` (B's term) and a `CONTEXT_TERMS` entry (B saves nothing beside it).

## CPU suite

trillian, merge `706b5ef0` (main `a557ea55` merged; the later commit `b10d9345` only rewraps comments),
[jobs/suite.sh](jobs/suite.sh): `tests/cpu -n 8 -m "not acceptance"`, `CUDA_VISIBLE_DEVICES=` empty: 5,470 passed,
203 skipped, 4 failed. The four failures (`test_output_chunks.py::test_crash_between_restart_and_flush`) were the
script's missing PYTHONPATH (the child process could not import `fastmace`); with it the file passes, 25 of 25. The
branch adds no ruff findings (each changed file has as many as on main).
[logs/suite-trillian/summary.txt](logs/suite-trillian/summary.txt).
