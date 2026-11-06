# Two new w′ backward candidates on the H200: both rejected at the lab bound (6 November 2026)

Owner, 6 November: try the two new backward candidates of the Codex report
([plans/codex-ideas-20261106.md](../../plans/codex-ideas-20261106.md), "Second pass") on the H200 on Modal. Both
replace OMAT-0 medium's interaction-2 w′ backward (`fastmace/kernels/oeq_streaming/wprime_backward.py`); their gains
would not add. The plan was a cheap kernel-lab bound first for each, with a stop above 0.95 of today's default w′
kernel, then a production switch, gate and MD-step A/B for a survivor. Neither survived the bound, so no switch was
built and no gate or MD A/B was run. No default changes.

## Setup

- Modal NVIDIA H200 (sm_90, driver 610.57.04, CUDA 13.3 UMD), profile `fourth`, job
  [`tools/modal/cartesian_backward.sh`](../../tools/modal/cartesian_backward.sh) at FastMACE `47e8b56a` (main
  `6561cc13` plus the lab tool). Environment: the Modal image (torch 2.14.1+cu130, mace-torch 0.3.16,
  openequivariance 0.7.0, cuequivariance 0.12.0; `lab-h200/pip-freeze.txt`). No baseline engine involved (kernel
  lab only). Wall time 110 s, metered $0.12.
- MACE-OMAT-0 medium, float32, Benchmark_loup SrTiO3 10x10x10 (5,000 atoms, 373,760 edges), captured after 300 MD
  steps (300 K, 1 fs) from the plan's default engine on the H200: hybrid backend, w′ kernel with `STAGE_DOUT`,
  `FACTORED`, `HORNER`, `SCALAR_PATHS`, `BLOCK_MESSAGE` (four dout groups), `BUTTERFLY` off, 128 threads.
- Tool: [`tools/lab_cartesian.py`](../../tools/lab_cartesian.py). Round robin in one process, 21 rounds, 3 warm-up
  launches, 256 MB L2 flush before every timed launch, CUDA events, medians. Two fresh processes (`lab-1`, `lab-2`).
  The base is the plan's own `WPrimeBackward` launch (one block per 32 edges, as production).

## Results (`lab-h200/`)

| Variant | lab-1 µs | lab-2 µs | Ratio (1 / 2) | Registers |
| --- | ---: | ---: | --- | ---: |
| **base**: today's H200 default w′ (with `STAGE_DOUT`) | 1,401.2 | 1,400.1 | 1.000 | (plan) |
| base without `STAGE_DOUT` (reference) | 1,613.2 | 1,613.0 | 1.151 / 1.152 | 96 |
| N1 converter alone (adjoint rows to Cartesian coefficients) | 56.8 | 56.9 | 0.041 | 24 |
| N1 w′ kernel, staged dout | 1,432.3 | 1,431.5 | 1.022 / 1.022 | 56 |
| N1 w′ kernel, unstaged | 1,549.6 | 1,548.4 | 1.106 / 1.106 | 64 |
| **N1 staged + converter** | 1,485.0 | 1,487.1 | **1.060 / 1.062** | 56 |
| N1 unstaged + converter | 1,602.3 | 1,602.3 | 1.143 / 1.144 | 64 |
| N2 skeleton, one vector path, U stored | 741.3 | 740.9 | 0.529 | 56 |
| N2 skeleton, one vector path, U consumed on chip | 555.8 | 555.2 | 0.397 / 0.397 | 48 |
| **N2 floor: 6 vector paths x the on-chip skeleton** | 3,334.7 | 3,331.4 | **2.380 / 2.379** | |

No variant spilled (0 local bytes).

### N1: Cartesian harmonic adjoints — rejected (lab 1.060 / 1.062 with its converter, stop rule 0.95)

Built completely enough to be checked: a node converter maps each path's receiver adjoint (2L + 1 components) to
the coefficients of a Cartesian harmonic polynomial (the 10 conversion matrices fitted in float64 against e3nn's
harmonics and the kernel schedule's own coupling coefficients, residual ≤ 7.3e-16), and the w′ kernel gives `dx`
and the three edge-vector derivatives directly (value, gradient and Hessian of the polynomial; radial `w′ f n` and
tangential `(w / r)(∇f − l f n)` terms folded in). Against the default kernel's `dsh` and `dL/dr` converted to the
same edge-vector gradient in float64: max deviation 8.2e-7 / 8.8e-7 of the largest component, median per-edge
relative 7.1e-7, `dx` 1.9e-7 of its largest entry. So the rewrite is correct at float32 rounding; it is too slow.

- The register count falls from 96 to 56 as the report predicted, but the larger hot loop takes it back: staged,
  the kernel alone is 2.2 % slower than the default, and the converter adds another 4 % (57 µs, close to the
  report's 43 µs byte bound for a standalone conversion).
- Unstaged against unstaged, the Cartesian kernel is 0.961 of the old one (1,549 vs 1,613 µs); `STAGE_DOUT`, which
  the H200 adopted this morning, gains 13 % on the old kernel but only 7.6 % on the Cartesian one and so removes
  that margin. With the converter it is 0.993 even unstaged.
- What the lab does not count favours N1 only slightly: production would also skip the sh backward and the radial
  `dL/dr` chain for interaction 2 (small elementwise kernels) and could fold the converter into `linear_1`'s adjoint
  GEMMs (the report's 17 µs byte bound). Neither closes a 6 % gap to reach 0.95.

### N2: receiver-tiled FP64 matrix adjoints — rejected (optimistic floor 2.38x the default)

The Codex compile-only skeleton (16 edges per block, U = S @ D and M = (w x) @ Dᵀ on FP64 `mma.sync.m8n8k4`,
weighted-x staging in shared memory) launched for all 23,360 edge tiles of one vector path. With U consumed on chip
instead of stored it takes 555 µs per path; OMAT-0 has six paths with a vector x, so the matrix part alone needs at
least 3.33 ms against the whole default kernel's 1.40 ms. This floor leaves out the four scalar paths, the S
preparation, the receiver gather of D (the skeleton reads one fixed D), the radial tables and Horner, the geometry
contraction and the dx atomics. Even one vector path (0.40) is above what the report's own stop rule allows for the
whole non-radial part (1.0 ms). The FP64 tensor-core rate on these small staged shapes, the open question of the
report, is the answer: the staging and conversion dominate.

## Verdicts

| Candidate | Lab ratio vs default | Gate | MD-step A/B | Verdict |
| --- | --- | --- | --- | --- |
| N1 Cartesian harmonic backward | 1.060 / 1.062 (kernel alone 1.022) | not run | not run | REJECTED (bound) |
| N2 receiver-tiled FP64 matrix adjoints | ≥ 2.379 / 2.380 (optimistic floor) | not run | not run | REJECTED (bound) |

The H200 MD step stays at about 5.13 ms. Ledger entries: `cx-n1-cartesian-wprime`, `cx-n2-fp64-adjoint` in
[docs/optimisation-ledger.md](../../docs/optimisation-ledger.md).

## Files

- `lab-h200/`: the Modal job's output (`run.log`, per-process `lab-N.log` and `lab-N/lab.json` with timings, the N1
  fits and conversion matrices and the correctness check; the rendered sources `wprime.cu` (plan kernel),
  `N1-stage.cu`, `N1-nostage.cu`, `base-nostage.cu`, `convert.cu`; `modal.json`, `pip-freeze.txt`).
- `codex-artifacts/`: the report's CPU artifacts copied from its temporary directory (Cartesian sketch and
  generator, FP64 skeleton and its compiler script, algebra census and resource counts).

## A side note for the main kernel lab

`tools/omat0_kernel_lab.py`'s `wprime_launcher` sizes the grid as one warp per edge unless a variant carries a
`// lab launch` marker. Since `STAGE_DOUT` became the H200 and A100-SXM4-80GB default, the captured base kernel
expects one block per 32 edges, so the lab launches 8x the blocks it needs (the extra ones exit at once). Lab
ratios against that base taken after `6561cc13` carry this small overhead in the denominator. `lab_cartesian.py`
launches the base through `WPrimeBackward` itself and is not affected.

**Fixed (6 November 2026, ideas-queue branch):** `wprime_launcher` reads the chunk from the kernel source
(`stage_c0 = blockIdx.x * N`) and launches one block per N edges, as `WPrimeBackward.__call__`; a persistent or
tiled lab launch of a `STAGE_DOUT` kernel is refused.
