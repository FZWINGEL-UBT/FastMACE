# SPLINE_W in production on the H200: check, gate, MD A/B, Benchmark_loup, in-depth profile

7 November 2026. Modal H200 (`NVIDIA H200`, driver 610.57.04, sm_90), torch 2.14.1+cu130, hybrid backend, float32,
OMAT-0 medium (SHA-256 `d4b14be9…`), FastMACE commit `c23ff357` (branch `fastmace/sx-layout-lab`), one container
(unit `fm-spline-w-20261107e`, `tools/modal/spline_w_job.sh`). Settings as Benchmark_loup
(`tools/bench/loup_md.py`): SrTiO3 10x10x10 (5,000 atoms, 373,760 padded edges), Langevin 300 K, friction
0.002 / fs, 1 fs, skin 0.5, seed 1, from rest.

## What was built

`SPLINE_W` (switch in `fastmace/kernels/oeq_streaming/spline_forward.py`; None: on for `SPLINE_W_GPUS`, empty
today): interaction 2's message in Symmetrix-XL's layout (one thread per receiver and channel, edges walked serially,
message in registers, block-layout output) with w from a per-model cubic Hermite spline in r, and the `TABLE_W` w′
kernel reading the same q-major spline table. The edge density alone comes from the certified radial table
(`radial_table_forward` with the density column only). The knots are fitted per model at build time
(`fastmace/rewrites/radial/spline.py`: bands dense at short r, refined until every band scores at most 0.5 of the
radial certificate; cached), and the spline is used only if its certificate passes over the whole domain.

For OMAT-0 medium the fit chose 731 intervals (0.3–0.5 Å: 157, 0.5–0.75: 112, 0.75–1.0: 87, 1.0–1.5: 124,
1.5–2.0: 54, …), certificate 0.483 (limit 1). Forward kernel: 80 registers, no local memory (spills).
[check-spline.json](check-spline.json).

## Results

| Measurement | Default | SPLINE_W | b/a |
| --- | ---: | ---: | ---: |
| MD-step A/B, order ab (8 blocks x 256 steps, one process) | 4.9015 ms | 4.3078 ms | **0.8789** |
| MD-step A/B, order ba | 4.9119 ms | 4.3100 ms | **0.8775** |
| Benchmark_loup, fresh processes (default, spline, spline, default) | 4.920 / 4.914 ms | 4.313 / 4.312 ms | 0.877 |
| Benchmark_loup, µs per atom-step | 0.984 / 0.983 | 0.863 / 0.862 | |

**About −12 % of the MD step (−0.60 ms)**, twice the −6.3 % expected from the isolated forward and w′ kernels of the
lab. The rest comes from the density-only radial table and from w′ itself being faster in production than in the lab.

Force/stress gate (`tools/force_stress_gate.py`, `--production-constant …SPLINE_W=True`, SrTiO3 3x3x3, NaCl,
NbCl5): **passed** on all three. Force ratios against ASE float32 0.920 / 0.883 / 0.983, stress ratios 0.660 /
1.059 / 1.417 (limit 3), energies below their 0.01 eV floor. [gate/](gate/).

## In-depth profile (tools/step_breakdown.py, 300 steps, eager kernel census per force call)

| | Default | SPLINE_W | Change |
| --- | ---: | ---: | ---: |
| Wall time per step | 4.927 ms | 4.307 ms | −0.620 ms |
| Force graph replay | 4.825 ms | 4.239 ms | −0.585 ms |
| Neighbour rebuild (23 in 300 steps), per step | 0.428 ms | 0.376 ms | −0.052 ms |
| Kernel time per force call | 4.744 ms | 4.158 ms | −0.587 ms |

Per kernel (µs per force call):

| Kernel | Default | SPLINE_W | Change |
| --- | ---: | ---: | ---: |
| interaction 2 forward (`fused_folded_forward` → `spline_forward`) | 1,269.4 | 959.0 | −310.4 |
| w′ backward (`oeq_projected_backward_wprime`) | 1,396.2 | 1,210.7 | −185.5 |
| radial table forward (all columns → density only) | 97.4 | 15.4 | −82.0 |
| tf32x3 GEMMs (four largest) | 805.7 | 806.9 | +1.2 |
| tabled outer sum backward / forward | 213.5 / 119.2 | 213.6 / 118.9 | 0 |
| symmetric contraction backward | 86.7 | 87.0 | 0 |

By category: edge elementwise 3.234 → 2.650 ms (now 64 % of kernel time); everything else unchanged. The two spline
kernels are now 52 % of the force call (2.17 of 4.16 ms): forward 959 µs at 80 registers, 38 % estimated
occupancy; w′ 1,211 µs at 72 registers, 44 %.
[profile-default/](profile-default/), [profile-spline/](profile-spline/) (stages, categories, ops, kernel sequence, GEMMs).

## Launch-shape variants of the spline forward (SX improvement screen items 1 and 2)

From the CPU-only SX opportunities report (`plans/notes/sx-layout-opportunities-20261107` on `h200-ideas-20261105`).
Both are switches in `spline_forward.py` and bitwise (the same terms per output slot, in the same edge order), so
no gate is needed. A/B: a = `SPLINE_W=True`, b = a plus the variant, 6 blocks x 256 steps, one container
(unit `fm-spline-var-20261107b`, profile `third`, `tools/modal/spline_variants_job.sh`). [variants/](variants/).

| Variant | Forward kernel(s) | b/a ab | b/a ba |
| --- | --- | ---: | ---: |
| `r128`: `THREADS=128 MIN_BLOCKS=6` (one receiver per block) | 80 registers, 8 bytes local memory | **0.9838** | **0.9848** |
| `parts`: `PARTS=True MIN_BLOCKS=4` (scalar-input paths 0, 2, 5, 8; vector-input 1, 3, 4, 6, 7, 9) | 64 + 64 registers, no local memory | 0.9938 | 0.9904 |

`r128` saves about 1.6 % of the step (~69 µs), as the report's static-degree bound predicted (68 µs). The 80-register cap
leaves an 8-byte spill. `parts` is faster in both orders but by under 1 %. The two were not combined, and each
was measured against SPLINE_W alone. Not tried: `r128` without the cap or with a softer cap (5 blocks), and
`r128` + `parts`.

## Cost

Main run: five containers on profile `fifth`. The full run took 24 min. Of the other four, two failed within seconds
in the new check (model trust hash, then a weakproxy in `gc`). One stopped after a passing gate, because the old
check searched the stored plan text, which holds no record names. One is the run before those.
Variants: one container on `third` (7 min); the first launch was refused by the budget guard (38-min timeout) and
relaunched with 20 min. Metered this month afterwards: fifth $24.87, third $27.68 of $30.

## Decision (conductor, 7 November 2026, decision 66)

SPLINE_W is on for the H200 family: `SPLINE_W_GPUS` holds `NVIDIA H200`, which also covers the H200 NVL. `R128`
(`R128_GPUS`: one receiver per 128-thread block, `__launch_bounds__(128, 6)`) is on there too. `PARTS` stays off. Both
stay off on the A100 until it is measured. Asserted in `tests/cpu/test_gpu_models.py::test_h200_defaults`.

## Follow-ups against the H200 defaults (SPLINE_W and R128 on)

One container (unit `fm-spline-fu-20261107a`, profile `fifth`, commit `efd4e41d`,
`tools/modal/spline_followups_job.sh`). MD-step A/B, a = H200 defaults (about 4.26 ms), b = the variant, 6 blocks
x 256 steps, both orders, Benchmark_loup settings. [followups/](followups/).

| Variant | What | b/a ab | b/a ba |
| --- | --- | ---: | ---: |
| `vec4` | `wprime_backward.SPLINE_VECTOR=4`: the w′ kernel reads the spline's four terms per weight as one float4 (two float2 today); bitwise | **0.9739** | **0.9793** |
| `cap5` | 128 threads, `MIN_BLOCKS=5`: 96 registers, no spill (R128's 6: 80 registers, 8-byte spill) | 0.9925 | 0.9937 |
| `nocap` | 128 threads, no cap | 1.0122 | 1.0134 |
| `r128parts` | `PARTS` on top of R128 | 1.0101 | 1.0075 |

`vec4` saves about 2.3 % of the step (~100 µs), from the w′ kernel's table loads alone. `cap5` beats R128's cap of 6
by 0.7 % (under the 1 % bound), and was not combined with `vec4`. `nocap` and `parts` on top of R128 are slower.

Context for the w′ backward (ledger check, 7 November 2026): every receiver- or sender-owned layout of the w′
lost before (conv-receiver-backward, fusion-receiver-local, omat0-wprime-split, piece order), and Symmetrix-XL's
own interaction-2 backward is 3.9 ms against our 1.83 ms. On the H200, removing all reductions saves 3.6 %, and
removing the atomics made the kernel slower (1.080). The kernel is bound by load instructions in L1, so a full layout
rework was not built; fewer load instructions (`vec4`) are what pay.
