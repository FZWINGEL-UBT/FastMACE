# H200 ideas after SPLINE_W: staged-gradient quartets (idea 2) and the composed first projection (idea 1, lab)

7 November 2026. Modal H200 (`NVIDIA H200`, driver 610.57.04, sm_90), one container (unit `fm-compose-quads`,
`tools/modal/compose_quads_job.sh`), FastMACE commit `ec080422` (branch `fastmace/h200-compose-quads`, on top of the
H200 defaults of `fastmace/sx-layout-lab` 78a25fd6: SPLINE_W, R128, SPLINE_VECTOR 4). OMAT-0 medium, SrTiO3 10x10x10
(5,000 atoms), Benchmark_loup settings (Langevin 300 K, 0.002 / fs, 1 fs, skin 0.5, seed 1).

## Idea 2: `STAGE_QUADS` (w′ reads the STAGE_DOUT row as float4 quartets)

`wprime_backward.STAGE_QUADS` (off, `STAGE_QUADS_GPUS` empty) lays the staged output-gradient row out as
`(o // 4) * 4C + 4c + o % 4`, so a lane loads four components of its channel with one `LDS.128`. Same row size, same
values in the same order.

| Measurement | Result |
| --- | --- |
| In use in the production build ([check-quads.json](check-quads.json)) | yes; w′ 96 registers, 0 local bytes |
| SASS of the dumped w′ ([sources/wprime-0.cu](sources/wprime-0.cu), NVRTC sm_90) | 10 `LDS.128` in the channel loop body, no scalar `LDS` left |
| MD-step A/B, order ab (a = H200 defaults, b = STAGE_QUADS) | 4.1272 → 4.1106 ms, **0.9960** |
| MD-step A/B, order ba | 4.1165 → 4.1098 ms, **0.9984** |

About −0.3 % (−10 µs), the size of the run-to-run spread. The staged loads were not what bounds w′. **Measured, not
adopted.** The kernel is bitwise by construction. The check's force comparison (max |ΔF| 4.3e-6 eV/Å) is not
evidence either way: it compares two builds on the unrattled lattice (max |F| 2.0e-5 eV/Å), and w′ accumulates with
atomics.

## Idea 1: compose interaction 1's projection with the species skip (lab)

Production on the H200 per degree l: the reassociation product `left_l [n(2l+1), 192] @ right_l [192, 128]`
(`tf32x3_matmul`; the linear is already folded into `right_l`), join and divide by `density + 1`, then
`tf32x3_skip_select` (`[n(2l+1), 128] @ B_l [128, 3 x 128]`, own-species columns kept). Composed:
`C_l,z = right_l B_l,z` ([3, 192, 128], once per model) and `species_mm(left_l, C_l)` per l, then join and divide.

`tools/compose_lab.py`, real shapes (5,000 atoms in SrTiO3's Sr Ti O O O order, l = 0..3, K = 192, C = 128),
forward + backward (gradient of `left`) in one CUDA graph, median of 7 x 50 replays,
[compose-lab.json](compose-lab.json):

| Variant | ms (fwd + bwd) | saving vs old | max rel. error vs float64 (out / grad) |
| --- | ---: | ---: | --- |
| old (production chain) | 0.960 | | 3.9e-7 / 4.5e-7 |
| composed, species_mm CUTLASS | **0.448** | **−512 µs** | 3.2e-7 / 3.2e-7 |
| composed, species_mm Triton | 0.501 | −459 µs | 1.5e-7 / 1.9e-7 |
| composed, species_mm mma.sync | 0.633 | −327 µs | 1.9e-6 / 1.8e-6 |
| atoms pre-sorted by species (3 plain products per l) | 0.597 | −363 µs | 3.2e-7 / 3.2e-7 |

The composed chain halves the lab chain. The skip's three-species product (3x the columns, forward and backward) is
gone, and the gather of species_mm costs less than splitting into per-species products. The composed matrices are
built in float64 and stored as float32, so the result changes at the rounding level: integration needs the force/stress
gate as well as the MD A/B. The lab's old chain runs the join/divide glue eagerly; in production Inductor fuses part
of it, so the MD saving will be smaller than 512 µs. The profile puts interaction 1's skip_select forward alone at
205 µs per force call.

Integrated below (`COMPOSE_SKIP`).

## Idea 1 integrated: `COMPOSE_SKIP` on the H200 ([compose-skip/](compose-skip/))

`kernels.triton.compose_skip.COMPOSE_SKIP` (off; `COMPOSE_SKIP_GPUS` empty), commit `92f2fa7a`. On OMAT-0 medium the
first interaction's linear is a per-l `BlockLinear` (not the identity), so the composition is
`C_l,z = right_l L_l B_l,z` (float64, stored float32). The reassociation multiplies by it through `species_mm` (CUTLASS
launch), and the interaction then only divides by `density + 1`, skipping both its linear and `skip_tp`. Unit `fm-compose-skip2`, Modal
profile `sixth`.

| Measurement | Result |
| --- | --- |
| In use in the production build | yes ([check-compose.json](compose-skip/check-compose.json)) |
| Forces, SrTiO3 10x10x10 rattled 0.05 Å, vs defaults | max \|ΔF\| 4.3e-6 eV/Å (same build twice: 3.5e-6; max \|F\| 7.07) |
| Force/stress gate (limit 3) | **passed**; force ratios 0.968 / 0.847 / 0.970, stress 0.467 / 0.815 / 1.121 (SrTiO3 / NaCl / NbCl5); energies below their 0.01 eV floor |
| MD-step A/B, order ab | 4.1212 → 3.9967 ms, **0.9698** |
| MD-step A/B, order ba | 4.1172 → 4.0095 ms, **0.9738** |

**About −115 µs (−2.8 %) of the MD step.**

## Items 3-5 (Codex H200 follow-up), one job ([ideas345/](ideas345/))

Commit `d96b92a3`, unit `fm-ideas345`, Modal profile `sixth`. Each variant was checked in the production build: forces
on the rattled cell differ from the defaults by no more than the build's own run-to-run spread (3.2–3.6e-6 against
3.1–3.5e-6 eV/Å). The w′ kernel keeps 72 registers, no spill.

| Variant | Switch | What the compiler shows | MD b/a (ab / ba) |
| --- | --- | --- | --- |
| Item 3: first radial table two terms per float4 | `tabled_outer_sum.PACKED_COEF` | per walk (forward, power and recompute backward) LDG.64 −6, LDG.128 +3, registers unchanged (64 / 80 / 80) | **1.0072 / 1.0079** (slower) |
| Item 4a: w × common factor once per path | `spline_forward.FACTOR_W` | forward still 80 registers, 8 local bytes; gate passes (force ratios 0.969 / 0.895 / 0.980) | 0.9969 / 0.9995 |
| Item 4b: int loop state | `spline_forward.CURSOR32` | forward still 80 registers, 8 local bytes | 1.0016 / 1.0033 |
| Item 5: one spline locator per edge | `spline_forward.SHARED_LOCATOR` | producer 24 registers; forward still 80 / 8 | 0.9966 / 0.9974 |

None clears 1 %. The halved coefficient load count does not speed up the first radial table: it loses 0.7–0.8 %. Factoring
removes multiplies but no registers, so the forward's 8-byte spill stays. The shared locator's −0.3 % is about what
its record traffic costs. **All four are measured and left off.**
