# VIRIAL_ELEMENTWISE: the strain gradient without a split-K GEMM (10 November 2026)

Lead 1 from tiled-speed's kernel profile (`fm.ForceField(omat0-medium, float32).evaluate(stress=True)`, SrTiO3 18³,
RTX 5060 Ti, source f34f35e8). One `aten::mm` [3, E] × [E, 3] ran as cuBLAS `sgemm_largek_lds64` and cost 145 of
5,171 ns/atom of kernel time (2.8 %). It is the backward of the one-graph strain's einsums in
`rewrites.engine.single_graph_symmetric_displacement`: shifts = unit_shifts @ cell and positions @ strain.

**Switch.** `rewrites.engine.VIRIAL_ELEMENTWISE` (`single_graph_elementwise_displacement`) writes both as
elementwise products summed over three, so the strain gradient is one reduction over the rows. It is chosen when the
force field binds its strain path (`one_graph_choice`), per GPU. Rounding: another summation order. The strain
self-check covers both functions.

**Where it applies.** Only calls with a strain leaf: stress evaluations, NPT steps and cell relaxations. NVE MD has
none, so `tools/ab_rewrites.py` cannot see it; `tools/virial_ab.py` times the force call with stress instead (the
one-off evaluation plan's compiled force field, fixed positions; a = off, b = on).

## Results

| GPU, SrTiO3 | Split-K kernels a (ns/atom) | Kernel time a (ns/atom) | b/a (ab) | b/a (ba) |
| --- | ---: | ---: | ---: | ---: |
| RTX 5060 Ti, 10³ (5,000 atoms), local | 161 (3 kernels) | 4,592–4,657 | 0.9675 | 0.9648 |
| A100-SXM4-80GB, 10³ (Alex job 4976350) | 4.9–5.9 (2) | 1,161–1,396 | 0.9845 | 0.9705 |
| A100-SXM4-80GB, 18³ (29,160 atoms) | 0.8 (2) | 1,072–1,185 | 0.9957 | 0.9944 |

With the switch on no split-K kernel is left (the 5060 Ti keeps one of 0.5 ns/atom). Stress b − a: at most 3e-8
against 0.029; forces 2–3e-6.

**A100: rejected.** cuBLAS runs the GEMM well there: the bound is 0.4 % of kernel time at 5,000 atoms and 0.07 % at
29,160, under the 1 % bar. The 10³ wall-time ratios are host-dominated (17.5 ms per call against 5.8 ms of kernels)
and differ between orders; at 18³ the gain is under 1 %.

**RTX 5060 Ti: passes the A/B** (bound 3.5 %, 1 % in both orders). Its own gate has not run: the queued local gate
was lost when the PC crashed (17:38), and local runs are on hold. The A100 gate passed with the switch on
(forces 1.000 / 0.934 / 0.970 of ASE float32's error, energies under the 0.01 eV floor), and the job's strain and
layering tests passed (48). Stays off everywhere until the 5060 Ti gate passes.

Run folder: [ab-alex-4976350](ab-alex-4976350/) (`tools/modal/virial_ab_job.sh`, source 48cad68a).

## Lead 2: interaction 0's copies and the zero-fill (not built)

On the sm_80 / sm_90 MD plans (results/step-limiters-20261109 ncu records) interaction 0 has only two small clone
kernels (9 µs A100, 5 µs H200); the 5060 Ti profile's large cat / clone / transpose kernels belong to its one-off plan.
`tabled_outer_sum_forward`'s zero-fill is 31.9 µs of 6.21 ms on the A100 (0.51 %) and 14.9 µs of 3.26 ms on the H200
(0.46 %): under the 1 % bar. Not sized: an owner-writes layout would also drop the kernel's atomics (the kernel is
112 µs on the H200, 390 µs in the A100's older ncu record).
