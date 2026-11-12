# w′ backward switches of 12 November 2026 on MACE-OFF and MACE-POLAR (12 November 2026)

Brief (conductor, 12 November 2026): the w′ backward defaults adopted on 12 November were measured on OMAT-0 medium
only, but the GPU sets turn them on for every model. Verify them, or add per-model off-lists (decision 66), for
MACE-OFF and MACE-POLAR:

- H200: `SKIP_OUTSIDE` + `HOIST_DX` and `LATE_PULLBACK`
- A100 family: `SKIP_OUTSIDE` + `HOIST_DX` + `WEIGHTED_VECTOR`

Branch `fastmace/wprime-off-polar` from main `4f02f077`.

## Result

| Model | H200 | A100 family |
|---|---|---|
| OFF23 small / medium / large, OFF23b medium, OFF24 medium | no w′ kernel in the plan: nothing to verify | no w′ kernel in the plan: nothing to verify |
| POLAR-1-S | no w′ kernel (one interaction after the reassociated first) | the same |
| POLAR-1-L | no w′ kernel in the plan | no w′ kernel in the plan |
| POLAR-1-M | no w′ kernel in the plan | w′ on (MH-1 class). `SKIP_OUTSIDE`, `HOIST_DX`, `LATE_PULLBACK` do not apply (no TABLE_W). `WEIGHTED_VECTOR` applies, gains nothing (NaCl 0.9932 / 1.0041, NbCl5 0.9987 / 0.9986 with b = off) and passes the gate: **per-model off-list** `WEIGHTED_VECTOR_MODEL_OFF_GPUS["polar1-m"]` |

No H200 run was needed and no Modal money was spent. The only cell to measure was POLAR-1-M with
`WEIGHTED_VECTOR` on the A100 (one Alex job).

MH-1 (r2SCAN and the other MH-1 labels) shares the MH-1-class kernel and also takes `WEIGHTED_VECTOR` on the A100
family; it was outside this brief and is not measured.

## Step 1: offline, CPU only

### Which plans run the w′ kernel (`plan_check.py` → `plan_check.json`)

The fixed planner's "auto" plan for each label on `NVIDIA H200`, `NVIDIA A100-SXM4-80GB`, `NVIDIA A100 80GB PCIe` and
`NVIDIA A100-SXM4-40GB` (the CPU planner tests' stand-in traits, NaCl):

- MACE-OFF (all five labels): `wprime_backward` off on every GPU. The generic family's w′
  (`generic_wprime_backward`) is on only for the labels in `DECISION_66_GENERIC_WPRIME_LABELS` /
  `…_LABEL_GPUS` (OMAT-0 medium and small).
- POLAR-1: `POLAR_DECISION_66_WPRIME_GPUS` holds the RTX 4000 Ada only, and `POLAR_DECISION_66_WPRIME_LABEL_GPUS`
  adds POLAR-1-M on the A100 family (by family: SXM4-80GB, 80GB PCIe, SXM4-40GB) and the 5060 Ti. POLAR-1-S has no
  second interaction. The H200 is in neither set.
- Controls: OMAT-0 medium on on all four; MH-1 (r2SCAN) on on the A100 family, off on the H200.

### Which switches POLAR-1-M's kernel takes (`render_polar.py` → `render_polar.json`, `sources/`)

The OEq problem is read from `MACE-POLAR-1-M.model`'s second interaction: 128x0e+128x1o ⊗ 0e+1o+2e+3o, 10 uvu paths
(the same shapes as OMAT-0 medium's interaction 2, in MH-1's instruction order). `WPrimeBackward.__init__` then runs
unchanged with the device name patched and the compile replaced by a capture, so the production code decides every
switch for label `polar1-m`. `WPrime(generic=False)` (MH-1's class) has no TABLE_W, so no HORNER, FOLD_ALPHA,
BLOCK_MESSAGE or SPLINE_W:

| switch | A100-SXM4-80GB | H200 (no plan uses it) | why |
|---|---|---|---|
| FACTORED | on | on | `FACTORED_GPUS` |
| STAGE_DOUT | off | on | `STAGE_DOUT_MODEL_OFF_GPUS["polar1-m"]` holds the A100 |
| SKIP_OUTSIDE | off | off | needs TABLE_W (`self.skip_outside = self.table_w and …`) |
| HOIST_DX | off | off | by default only with SKIP_OUTSIDE and STAGE_DOUT |
| LATE_PULLBACK | off | off | needs TABLE_W, HORNER and BLOCK_MESSAGE (`wprime_pullback.supported` is False without blocks) |
| WEIGHTED_VECTOR | **on** | off | needs FACTORED only; paths 4, 5, 7, 8, 9 have more than 3 edge sums |

Forcing `SKIP_OUTSIDE = True` and `LATE_PULLBACK = True` on the A100 leaves the source byte-identical
(`a100_all_on_equals_default`): both fall back by themselves. The same holds for MH-1's own A100 kernel (same
class), which also takes WEIGHTED_VECTOR; MH-1 is outside this brief and not measured here.

### SASS census (`census_sass.py`, NVRTC + cuobjdump, WSL mace_cueq_oeq, `CUDA_VISIBLE_DEVICES=''`)

Per edge and lane (channel pass × 4 plus the rest of the edge loop where the census finds it):

| render | sm_80 regs | sm_80 channel pass | sm_80 per edge | sm_90 regs | sm_90 per edge |
|---|---:|---:|---:|---:|---:|
| A100 default (WEIGHTED_VECTOR on) | 96 | 513 | 2,052 | 96 | 2,356 |
| WEIGHTED_VECTOR off | 96 | 521 | 2,084 | 96 | 2,302 |
| ratio | | | **0.985** | | 1.023 |

On sm_80 (the A100) WEIGHTED_VECTOR saves 8 FP instructions per channel pass (32 per edge, −1.5 % of the kernel) at
the same 96 registers. On sm_80 the census does not find a separate edge loop for this render (no STAGE_DOUT), so
the per-edge figure is the channel pass × 4 only. sm_90 is shown for the record: there the weighted render costs
more, but no H200 plan runs this kernel. For OMAT-0 medium the sm_80 census gave 1,976 → 1,942 (−1.7 %) and the MD
step 0.995 / 0.991, 0.997 / 0.995.

### CPU checks and fallback tests

- The late-pullback float64 emulation (`wprime-late-pullback-20261112/emulate2.py`) covers LATE_PULLBACK sources
  only (STAGE_DOUT prologue, TABLE_W). It has nothing to run on here: no OFF or POLAR kernel takes LATE_PULLBACK.
- WEIGHTED_VECTOR is a regrouping that is exact in real arithmetic (`test_weighted_vector_source`); its rounding is
  checked by the 3× float32 gate on the GPU (Step 2).
- `tests/cpu/test_wprime_model_coverage.py` (new):
  - no OFF or POLAR plan runs w′ on the H200 (if one gains it, its H200 switches need measuring first);
  - on the A100 family, POLAR-1-M is the only OFF / POLAR plan with w′;
  - `wprime_pullback.supported` is False without BLOCK_MESSAGE;
  - an MH-1-class `WPrimeBackward` on the H200, A100-SXM4-80GB and A100 80GB PCIe decides SKIP_OUTSIDE, HOIST_DX
    and LATE_PULLBACK off, also with SKIP_OUTSIDE / LATE_PULLBACK forced on, with an unchanged source;
  - the same kernel takes WEIGHTED_VECTOR on the A100 and not on the H200, except POLAR-1-M (off-list, below);
  - `weighted_vector_default` honours `WEIGHTED_VECTOR_MODEL_OFF_GPUS`.
- Run on leeloo (GPU idle; tree `ee2c2207` plus the test fix): the new file and `test_wprime_backward.py`, 40 passed.

## Step 2: GPU

### H200

Nothing to run: no OFF or POLAR plan has the w′ kernel there. No Modal job, $0.

### A100 (NHR@FAU Alex, job 4990131)

`jobs/alex_wv_polar.sh` on source `ee2c2207`: `tools/jobs/alex_ab_pack.sh` (node-local env, source and cache on
$TMPDIR) with a = today's A100 defaults for POLAR-1-M (WEIGHTED_VECTOR on), b = `WEIGHTED_VECTOR=False`, NaCl and
NbCl5 3×3×3, warm-up 200, 6 × 256 steps, both build orders (ab = abab, ba = abba); then
`tools/force_stress_gate.py` (3× float32) with `WEIGHTED_VECTOR=True`.

Records: `alex-4990131/` (run.log, the A/B JSON and logs, the gate directory without its large arrays).
Node a0535, A100-SXM4-80GB, driver 615.71.10, torch 2.14.1+cu130, 16 cores.

MD-step A/B (b = `WEIGHTED_VECTOR=False`, a = defaults with it on; b/a < 1: off faster):

| structure | order | a (on) | b (off) | b/a |
|---|---|---:|---:|---:|
| NaCl 3×3×3 | ab | 4.0352 ms | 4.0075 ms | 0.9932 |
| NaCl 3×3×3 | ba | 4.0127 ms | 4.0291 ms | 1.0041 |
| NbCl5 3×3×3 | ab | 15.4334 ms | 15.4129 ms | 0.9987 |
| NbCl5 3×3×3 | ba | 15.4044 ms | 15.3824 ms | 0.9986 |

All four blocks valid. Builds 138–228 s per side.

3× float32 gate with `WEIGHTED_VECTOR=True` (plan: `wprime_backward` on), ratio to the ASE float32 spread:
NaCl forces 0.98, stress 1.18; NbCl5 forces 1.04, stress 0.71; energies within the 0.01 floor. **Passes.**

**Decision (decision 66, per-model off-list).** On POLAR-1-M, WEIGHTED_VECTOR is 0.13 % slower in both orders on
NbCl5 and mixed on NaCl. The −1.5 % kernel census is too small to show in a step that the w′ kernel shares with the
POLAR k-space and charge heads. Without a measured gain, the mixed-cell rule (as for POLAR-1-L's w′) keeps it off.
`WEIGHTED_VECTOR_MODEL_OFF_GPUS = {"polar1-m": A100-SXM4-80GB}` (family match: every A100). POLAR-1-M's A100 kernel is
then the pre-12-November source again, bitwise. An A/B side's `WEIGHTED_VECTOR = True` still wins.

Cost: one Alex A100 job, about 55 min (A/B 34 min, gate 19 min, staging 1 min).

## Files

- `plan_check.py` / `.json`: which plans run w′.
- `render_polar.py` / `.json`, `sources/{a100-default,a100-noweighted,h200-default}.cu`: POLAR-1-M's kernel.
- `census_sass.py` (copy of `wprime-h200-20261111`'s), `census_sm80.json`, `census_sm90.json`, `sass/*.sm80.sass`.
- `run_wsl.sh`: CPU runs in WSL.
- `jobs/alex_wv_polar.sh`: the Alex job; `alex-4990131/`: its records.
