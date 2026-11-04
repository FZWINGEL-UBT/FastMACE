# A model-specialised tensor-product backward for OMAT-0 (4 November 2026)

Question (conductor, after the Symmetrix-XL comparison in `omat0-a100-optimisations.md` §6): can a tensor-product
backward specialised to MACE-OMAT-0 medium's paths and irreps make the wprime backward
(`fastmace::streaming_conv_wprime_backward`, kernel `oeq_projected_backward_wprime`, the largest kernel of the force
call) faster than the vendored OpenEquivariance PR 223 schedule? Owner, the same day: keep improving the kernel until
nothing more is possible, on leto and then on the A100 through Modal.

**Verdict: adopt `wprime_backward.FACTORED`. Default on the RTX 4000 Ada and every A100 (`FACTORED_GPUS`, owner,
4 November 2026: "Turn them on", decision 66), together with `radial.SYMMETRIC_PAIRS` (`SYMMETRIC_PAIRS_GPUS`); the
RTX 5060 Ti was not measured and keeps both off.** It regroups the backward's three trilinear sums per path. The result is rounding-only. The force gates pass on the
Ada; on the A100 FACTORED's errors equal the default's, which already fails one stress check there (§3.1). On Benchmark_loup (OMAT-0 medium, SrTiO3 5,000 atoms) the MD step drops by:

| GPU | wprime kernel | MD step b/a (abab / abba) | step a → b |
|---|---|---|---|
| RTX 4000 Ada (leto) | 0.90 | 0.9832 / 0.9801 | 30.45 → 29.94 ms |
| A100-SXM4-80GB (Modal) | 0.72–0.73 | **0.9381 / 0.9397** | 14.29 → 13.41 ms |

`radial.SYMMETRIC_PAIRS` (existing, bitwise) adds about 0.5 % on top of it (§4). Nothing else tried beats it; §5 lists
what was tried and why the kernel is now at its limit.

## 1. The kernel is already specialised

The wprime kernel is rendered with Jinja from the model's streaming schedule (`FactorizedComputationSchedule`) when the
model is built. For OMAT-0 medium it has 10 paths, 40 output components and 86 Clebsch–Gordan terms, all unrolled. The
coefficients are hex float literals, so there are no coefficient loads. "Constants baked in, paths unrolled per irrep"
is what Symmetrix-XL does, and it holds here already. What specialisation can still change is the arithmetic per term
and redundant loads.

## 2. Falsifiers (kernel lab, `tools/omat0_kernel_lab.py`, RTX 4000 Ada)

leto, NVIDIA RTX 4000 Ada Generation (driver 595.99.02), `mace_cueq_oeq` (torch 2.14.1+cu130), float32 default plan.
The kernel is captured from one eager force call: SrTiO3 10×10×10, 373,760 edges; the Ada's default `TABLE_W` +
`BLOCK_MESSAGE` kernel at degree 5, 96 registers. 21 rounds round robin, medians, ratios to the base kernel. Per term
the vendored body does three updates (weight, input and edge gradient), about 7 floating-point operations.

| Variant | What | Ratio | Registers |
|---|---|---|---|
| `F-tp-no-coef` | coefficient multiplies dropped | 0.947 | 96 |
| `F-tp-wg-only` | only the weight-gradient update per term (about 3 operations) | 0.737 | 64 |
| `F-no-tp` | no per-term arithmetic | 0.640 | 40 |
| `sh-hoist` (exact) | sh loaded once per edge into registers | 1.095 | 96 |

The bound (0.737) is far past the 10 % stop rule, so the candidate was built.

## 3. The factored backward (`wprime_backward.FACTORED`)

Per path and edge channel: `u_i = Σ c g_o y_j` and `v_j = Σ c g_o x_i` (per term `p = g_o·c`, then two FMAs), then
`wg = Σ_i x_i u_i`, `dx_i += w u_i` and `dsh_j += w v_j` once per component. These are the same sums regrouped. It is
the render option `factored=True` and the module constant `FACTORED = False` (the allow-listed A/B pattern of
`TABLE_W`). Every wprime kernel built while it is set uses it: stored w and `TABLE_W`, both layouts, the block message.

| Check | Result |
|---|---|
| lab, Ada (5 runs, `e9980251`…`c01b31c6`) | 0.900–0.904, 96 registers; dsh 1.8e-7, dr 2.9e-7, dx 2.7e-7 (max \|d\| / max \|base\|) |
| lab, A100-SXM4-80GB (`bcb1bf28`, `dcde706e`) | 0.727 / 0.722 (4.14 → 3.01 ms) |
| GPU test `tests/gpu/test_wprime_backward_cuda.py` | passes on the Ada (`78b27fd2`) and the A100 (`dcde706e`); factored cases (both layouts × stored w / `TABLE_W`): forward bitwise, dx ≤ 3e-7 and dsh ≤ 1.4e-7 relative to unfactored, dr 4.7–6.5e-7 against float64 (unfactored 4.7–7.0e-7), CUDA graph |
| force gate, Ada: OMAT-0 medium (SrTiO3 3×3×3, NaCl, NbCl5) and MH-1 r2SCAN (NaCl, NaNbO3, NbCl5) | pass, errors as the default's (table below) |
| force gate, A100 default plan (`8aec363b`) | FACTORED's errors are the default's (table below). OMAT-0 NbCl5 stress fails **for both**: §3.1 |
| MD step A/B (Benchmark_loup SrTiO3 10×10×10, 1 fs, skin 0.5 Å, 200 warm-up, 8 × 200 steps, same process) | Ada 0.9832 / 0.9801; A100 0.9381 / 0.9397 |

Force gate on the Ada, max |error| against float64 MACE and the ratio to the ASE float32 calculator's error
(`results/omat0-tp-backward-20261104/rtx4000ada-checks/gate-table.txt`):

| | default: energy, forces | FACTORED: energy, forces |
|---|---|---|
| OMAT-0 SrTiO3 | 5.41e-4 r1.49, 2.54e-5 r0.81 | 5.36e-4 r1.50, 2.50e-5 r0.76 |
| OMAT-0 NaCl | 1.63e-4 r1.16, 3.58e-6 r0.77 | 1.65e-4 r1.17, 3.63e-6 r0.77 |
| OMAT-0 NbCl5 | 2.03e-4 r0.91, 7.85e-5 r0.98 | 2.11e-4 r0.94, 7.83e-5 r0.96 |
| MH-1 NaCl | 7.41e-5 r0.69, 3.67e-6 r0.90 | 6.31e-5 r0.65, 3.64e-6 r0.91 |
| MH-1 NaNbO3 | 3.10e-4 r0.29, 1.37e-4 r1.02 | 3.05e-4 r0.28, 1.37e-4 r1.02 |
| MH-1 NbCl5 | 1.12e-4 r0.14, 7.89e-5 r0.98 | 1.12e-4 r0.14, 7.87e-5 r0.97 |

Force gate on the A100-SXM4-80GB, the A100's default plan (tf32x3 GEMMs, tensor-core forward, degree 5, `TABLE_W`,
`BLOCK_MESSAGE`; `a100-sxm4-gate-8aec363b/gate-table.txt`):

| | default: energy, forces, stress | FACTORED: energy, forces, stress |
|---|---|---|
| OMAT-0 SrTiO3 | 4.89e-3 r16.8, 3.54e-5 r1.25, 4.31e-7 r1.58 | 4.91e-3 r14.9, 3.54e-5 r1.18, 4.31e-7 r1.63 |
| OMAT-0 NaCl | 9.10e-4 r5.88, 3.79e-6 r0.85, 6.32e-8 r1.51 | 9.10e-4 r5.95, 3.91e-6 r0.85, 6.51e-8 r1.55 |
| OMAT-0 NbCl5 | 1.52e-3 r13.7, 8.35e-5 r1.03, **1.79e-7 r3.09 FAIL** | 1.52e-3 r13.4, 8.38e-5 r1.02, **1.81e-7 r2.69 FAIL** |
| MH-1 NaCl | 7.57e-4 r4.36, 3.67e-6 r0.89, 4.73e-8 r1.37 | 7.57e-4 r4.33, 3.67e-6 r0.90, 4.73e-8 r1.43 |
| MH-1 NaNbO3 | 1.77e-3 r1.59, 1.41e-4 r1.04, 2.87e-7 r0.76 | 1.77e-3 r1.61, 1.42e-4 r1.06, 3.06e-7 r0.90 |
| MH-1 NbCl5 | 8.89e-4 r1.09, 8.09e-5 r1.00, 4.40e-8 r0.43 | 8.90e-4 r1.09, 8.01e-5 r0.98, 4.63e-8 r0.47 |

### 3.1 The A100 default plan fails the OMAT-0 NbCl5 stress check (found here, not caused by FACTORED)

OMAT-0 NbCl5 stress: production 1.79e-7 (default) and 1.81e-7 (FACTORED) against the limit max(1e-7 floor, 2 × ASE
float32), here 1.16e-7 and 1.34e-7 (ASE's float32 error itself differs between the two runs, 5.8e-8 and 6.7e-8). The
energy ratios (13–17) are tf32x3's (§4 of `generic-wprime.md`: NbCl5 11.7) and stay far under the 10 meV floor; the
stress ratio was 1.69 with tf32x3 alone (3 November). The A100 default as it stands today (tf32x3, tensor-core forward,
degree 5, `TABLE_W`, `BLOCK_MESSAGE`) was gated on the RTX 5060 Ti (dztc5, stress 1.06), never on the A100 itself.
FACTORED changes the production error by 1 %.

**Bisection (A100-SXM4-80GB, Modal, source `3812b56e`, 5 November 2026; `a100-sxm4-stress-bisect-3812b56e/`):**
the OMAT-0 NbCl5 gate with one A100 default off at a time. Production stress error and its ratio to the ASE float32
error (limit 2):

| Plan | energy | stress | gate |
|---|---|---|---|
| default (FACTORED + SYMMETRIC_PAIRS on), two runs | 1.52e-3 r14.5 | 1.82e-7 r2.97, 1.81e-7 r2.55 | **fail** |
| main (FACTORED, SYMMETRIC_PAIRS off) | 1.52e-3 r15.0 | 1.79e-7 r2.53 | **fail** |
| `TABLE_W` off | 1.52e-3 r12.9 | 1.94e-7 r3.10 | **fail** |
| `BLOCK_MESSAGE` off | 1.52e-3 r15.2 | 1.79e-7 r2.40 | **fail** |
| `radial.DEGREE = 10` | 1.51e-3 r14.6 | 1.66e-7 r2.55 | **fail** |
| `radial_tables` off | 1.20e-3 r10.9 | 1.64e-7 r2.67 | **fail** |
| tensor-core forward off (`TENSOR_CORES=False`) | 1.24e-3 r11.2 | 1.21e-7 r1.63 | pass |
| `tf32x3_gemms` off | 2.18e-4 r2.07 | 8.02e-8 r1.30 | pass |
| both off | 1.04e-4 r0.92 | 2.89e-8 r0.42 | pass |

Forces stay at r0.97–1.14 in every plan. No single default causes the excess: it is the **sum of the two 3×TF32
precision changes**, the `tf32x3_gemms` GEMMs (about 1.2e-7 alone) and the tensor-core a·W product of the fused
forward (about 8e-8 alone). Each passes alone; together they exceed 2 × ASE float32 (1.2–1.5e-7) on NbCl5 only.
FACTORED and SYMMETRIC_PAIRS change nothing here (default vs main: 1.8e-7 both). The other structures pass with the
new defaults on the A100: OMAT-0 SrTiO3 stress r1.50, NaCl r1.47; MH-1 NaCl r1.48, NaNbO3 r0.93, NbCl5 r0.46 (forces
r0.85–1.07). Which of the two to give up on the A100 for OMAT-0, or whether to accept the excess (4e-8 eV/Å³ over the
limit, energy within the 10 meV floor), is the owner's decision.

The A100 gains three times as much as the Ada: on the A100 the kernel is bound by FP32 arithmetic (19.5 TFLOPS against
2 TB/s), on the Ada by the value-table loads (§5).

## 4. With mirror species pairs (`radial.SYMMETRIC_PAIRS`, bitwise)

Each edge reads the (min, max) species pair's table row, so SrTiO3 touches 6 of 9 slabs (164 → about 110 MB of value
table). This had no effect on the old kernel (0.984). With the factored kernel it helps a little:

| | lab, factored + unordered (vs factored) | MD step b/a, b = FACTORED + SYMMETRIC_PAIRS |
|---|---|---|
| RTX 4000 Ada | 0.877 (0.903) | 0.9792 / 0.9747 (FACTORED alone 0.9832 / 0.9801) |
| A100-SXM4-80GB | 0.721 (0.727) | 0.9365 / 0.9307 (FACTORED alone 0.9381 / 0.9397) |

About 0.5 % of the step on both GPUs. Bitwise, so a default needs no gate. Default on the same GPUs as FACTORED
(owner, 4 November 2026).

## 5. What else was tried (the kernel's limit)

| Idea | Result | Folder |
|---|---|---|
| register caps on factored: 80 / 72 / 64 | 1.243 / 1.467 / 2.700 (spills of 80–224 bytes) | `rtx4000ada-lab-maxreg` |
| two adjacent channels per lane: float2 x/dout, float4 table rows (`wprime_pair.py`, lab only) | Ada 0.898 (factored 0.900), A100 0.708 (0.722); 128 registers. About 0.4 % of the A100 step, not built | `rtx4000ada-lab-pair`, `a100-sxm4-*` |
| `FACTORED_SCALE`: each path's common \|c\| applied once, terms carry the sign (50 of 86 terms with no multiply) | A100 lab 0.830 (factored 0.722); MD step 1.0247 / 1.0238 against FACTORED. Rejected, kept off | `a100-sxm4-scale-dcde706e` |
| degree-7 tables, float4 rows (fewer, longer pieces) | base 7.97 ms against 6.76 ms at degree 5; factored 1.009 there | `rtx4000ada-lab-locality` |
| sh hoisted per edge (exact) | 1.095 | `rtx4000ada-lab-falsifiers` |

Where the factored kernel's time goes now (Ada, `rtx4000ada-lab-diagnostics`; ratios to the unfactored base, factored
0.904): every edge reading one slab 0.751, no table loads 0.599, no output-gradient loads 0.667 (80 registers), no x
gather 0.879, no atomics 0.901. On the A100: one slab 0.683, the 3-operation bound 0.588 (factored 0.722). The remaining
cost is the value-table slab: 30 KB per edge from a 164 MB table that does not fit the L2. Global edge orders by slab
(1.51× slower), cache hints, float4 slabs and other degrees were rejected in earlier rounds
(`results/omat0-kernel-lab-20261103` §1, §14–16). What specialisation could remove is gone: the arithmetic is factored,
and extra structure (common magnitudes, vector channels) costs more in registers than it saves.

## 6. Runs and provenance

All in `results/omat0-tp-backward-20261104/` (README there). leto: idle check in every run folder (only sovic's idle
leftover `python3`; one lab launch was refused while another session's A/B held the GPU, and rerun on the A100
instead). Modal: profile `second`, image builder pinned to 2025.06 for the Python 3.10 image
(`MODAL_IMAGE_BUILDER_VERSION`), driver 610.57.04, costs in each `modal.json` and the ledger (0.52 + 0.29 + 1.50 + 2.03 USD, the last the stress bisection). CPU suite on leto (`mace_cueq_latest`, 16 workers): 5,128 passed, 202 skipped at `78b27fd2` and at `8aec363b`. Ruff (`fastmace tests`): only the 3 long lines that already exist on main.

## 7. Not done

- The A100 default's NbCl5 stress gate failure (§3.1) is located (tf32x3 GEMMs + tensor-core forward together) but
  not resolved: the owner decides which to drop for OMAT-0 on the A100, or whether to accept it.
- The RTX 5060 Ti and the A100 80GB PCIe were not measured; the PCIe card gets FACTORED and SYMMETRIC_PAIRS through
  the "every A100" list without its own A/B or gate.
- Branch `fastmace/factored-default` (`3812b56e`): CPU suite on leto 5,122 passed, 200 skipped
  (`/mnt/homes/zwingel/fastmace-suite/fdef-20261104-2305`).
- The vendored non-wprime backward and `wrebuild_backward` keep the per-term body.
