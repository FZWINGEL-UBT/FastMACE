# OPT4-T04: `grouped_launch` on generic MACE (opt-4 candidate 4)

24 October 2026. Branch `fastmace/opt4-T04` on main `13a1983`. The record and its shape rule are unchanged since
OPT-T03b (`f2531cb`): `RewriteConfig.grouped_launch`, `never_automatic`, rounding, deterministic.

## Verdict

- **Gain on every generic label tried, both GPUs.** NaCl 3×3×3, b/a 0.946–0.955 on the local RTX 5060 Ti
  (sm_120) for all five labels, in both build orders; 0.957–0.959 on the Colab L4 (sm_89) for `omat0-medium`.
  Every one of the 12 block pairs of every NaCl run is below 1.
- **NbCl5 neutral** (1.0008 / 0.9984), as the shape rule predicts.
- **Force gate passed** on `omat0-medium` (NaCl and NbCl5).
- **No applicability change was needed.** The record already applies to generic MACE (`block_linear` applies;
  every block linear takes it). One CPU test now pins that.
- The plan's falsifier (b/a ≤ 0.98 in both orders on `omat0-medium` NaCl) passes. Against the opt-4 default
  rule (a gain on sm_120 and no loss on the L4), `grouped_launch` qualifies **for generic MACE**. On MH-1 the
  earlier result stands: sm_120 0.977 / 0.982, sm_89 1.005 / 1.003 (`docs/design/opt-T03b-diagnosis.md`).
  The owner decides the default.

## Why it gains here

The generic models' block linears are per-l products `[n · (2l+1), 128] @ [128, 128]` with the same weight
shape for every l. In NaCl (216 atoms) all of them are small by the shared shape rule (`tile_rule.uses_triton`:
fewer than 2 waves of 64-row tiles, under 4,544 rows on 36 SMs), so each block linear's equal-shape products
run as one Triton launch, forward and input gradient. The opt-4 baseline profile of `omat0-medium` NaCl
(`results/opt4-baseline-ccab528/local/models/omat0-medium-NaCl/`) had 75 atom-GEMM launches at 4.4 TFLOPS
(15 % of peak) and 0.39 ms; the A/B saves 0.08 ms per step, about a fifth of that time.

In NbCl5 (1,944 atoms) only the l = 0 products are small on 36 SMs, and a single product is not grouped, so
nothing changes; b/a is the noise floor.

The grouped products with K = 256 or 384 (from `grouped_linear`) have different weight shapes per l and stay
on cuBLAS, by the record's equal-shape rule.

## Measurements

**Environment:** local RTX 5060 Ti (sm_120, 36 SMs, driver 610.88), WSL2, `~/environments/mace_cueq_oeq`
(torch 2.14.1+cu130, Triton 3.8.0, mace-torch 0.3.16, cuEquivariance 0.12.0, OEq 0.7.0), hybrid backend,
float32, NVT 300 K. Pinned source `~/fastmace-opt4/src-T04-13a1983` (`git archive 13a1983`). Structures from
the baseline export `~/fastmace-baseline/1a-T18` (ResearchModule `research-engine-baseline-20261017`).

**Force gate** (`tools/force_stress_gate.py --engine fastmace --model omat0-medium --rewrites
grouped_launch=on`, 3 configurations per structure and cell mode, untimed, 09:55–10:07 UTC). Ratios are
production against the float32 ASE calculator, limit 2 (or the gate's floor). The record's self-check passed
in every force field (max relative error 8.7e-7).

| Structure | Energy | Forces | Stress | Passed |
| --- | ---: | ---: | ---: | --- |
| NaCl | 1.57 | 1.14 | 0.64 | yes |
| NbCl5 | 1.38 | 1.13 | 1.03 | yes |

`tests/gpu/test_grouped_gemm_cuda.py`: 2 passed.

**Same-process A/B** (`tools/ab_rewrites.py`, a = defaults with `grouped_launch=off`, b = the same with `on`,
12 blocks of 300 steps after 300 warm-up steps, a-b with ABAB and b-a with ABBA; each set in its own
`local_gpu.py --quiet` reservation in a window the conductor cleared; all runs valid, no compile while
timed). Per-pair ratios are the range over the 12 pairs.

| Label (checkpoint sha256) | Structure | a-b: a → b, b/a (pairs) | b-a: a → b, b/a (pairs) | UTC |
| --- | --- | --- | --- | --- |
| `omat0-medium` (`d4b14be9afa2`) | NaCl | 1.6535 → 1.5756 ms, **0.9529** (0.942–0.972) | 1.6504 → 1.5693 ms, **0.9509** (0.940–0.969) | 10:27, 11:01 |
| `mp0-medium` (`01bfe2210013`) | NaCl | 1.5100 → 1.4294 ms, **0.9467** (0.934–0.969) | 1.5087 → 1.4401 ms, **0.9546** (0.923–0.966) | 11:20–11:23 |
| `mpa0-medium` (`75428afe3a1d`) | NaCl | 1.6510 → 1.5698 ms, **0.9508** (0.928–0.970) | 1.6493 → 1.5612 ms, **0.9466** (0.932–0.970) | 11:23–11:25 |
| `mp0b3-medium` (`2f2be696351a`) | NaCl | 1.6485 → 1.5649 ms, **0.9493** (0.934–0.975) | 1.6470 → 1.5574 ms, **0.9456** (0.931–0.987) | 11:52–11:55 |
| `matpes-pbe` (`e618ad582b84`) | NaCl | 1.6640 → 1.5819 ms, **0.9507** (0.937–0.971) | 1.6451 → 1.5629 ms, **0.9500** (0.932–0.976) | 11:55–11:57 |
| `omat0-medium` | NbCl5 | 9.7032 → 9.7108 ms, 1.0008 (0.999–1.003) | 9.7631 → 9.7473 ms, 0.9984 (0.995–1.002) | 12:09–12:14 |

The local noise floor is 1.1 % (NaCl) and 0.4 % (NbCl5); the NaCl gains are 4–5 times the NaCl floor.

**Colab L4** (sm_89, driver 580.82.07, `tools/colab/fastmace_colab.py --gpu L4 --commit 13a1983`, session
`fm-opt4t04-l4-1`, one VM, both orders in one container, the pinned hybrid stack of
`tools/modal/requirements-hybrid.txt`):

| Label | Structure | a-b: a → b, b/a (pairs) | b-a: a → b, b/a (pairs) |
| --- | --- | --- | --- |
| `omat0-medium` | NaCl | 2.1058 → 2.0200 ms, **0.9593** (0.955–0.968) | 2.1361 → 2.0437 ms, **0.9568** (0.938–0.975) |

Wall 2,275 s, of which 1,660 s setup (one restore poll hung and was re-attached). Compute units 46.43 → 45.56:
**0.87 CU** (account-wide balance, so other VMs running at the same time count too). The VM was stopped; `colab sessions`
showed none live afterwards.

## What is not covered

- The force gate ran on `omat0-medium` only. The other four labels have the same block-linear shapes
  (128 channels per l), so they take the same kernels; they were not gated separately.
- MACE-OFF, POLAR-1 and OMOL-0 were not screened. OMOL-0's 1,024-channel products have different shapes, and
  POLAR-1's launches are candidate 2's subject (OPT4-T02).
- Cells between NaCl and NbCl5 sizes: the gain shrinks as products cross the shape rule's threshold.
  Above it the record is a no-op by construction.
- The A100 was not measured.

## Changes on the branch

- `tests/cpu/test_rewrites_fusion.py::test_grouped_launch_applies_to_generic_mace`: on a tiny generic model
  (residual interaction, cuEq naive kernel) the record applies to every block linear, the grouped launch runs
  (fake one-SM device), and energy and forces are bitwise unchanged.
- `results/opt4-T04-grouped-launch/run.sh` (the jobs above), `local-13a1983/` (A/B JSONs, gate metadata and
  values without the `.npz`, run log) and `colab-l4-13a1983/` (A/B JSONs, `colab.json`).
- `docs/rewrites.md`: the row of this screen.

Test command (WSL, no other pytest running):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 nice ~/environments/fastmace_cpu/bin/python -m pytest -n 2 -q -p no:cacheprovider tests/cpu/test_rewrites_fusion.py -k grouped
```

Output: `4 passed, 18 warnings in 8.34s`.
