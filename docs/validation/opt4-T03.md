# OPT4-T03 stage 1: 3×TF32 atom GEMMs, accuracy (optimisation round 4)

Plan: `plans/opt-4.md`, §3 candidate 3 and the OPT4-T03 row of §5. Stage 1 only: the accuracy of 3×TF32 atom GEMMs
through Triton's `tf32x3`. Stage 2 (a CUTLASS 3×TF32 kernel) is not built: the owner decided so after this
report (§8). Speed does not count here.

**Verdict: PASS.**
- MH-1 passes the force gate on NaCl and NbCl5 (local sm_120) and the NVE drift check (Colab A100).
- OMOL-0 XL passes the force gate (local).
- OMOL-0's NVE drift cannot be measured. On NaCl, the only structure available, the default plan's trajectory
  reaches NaN as well, as phase 1a recorded (model-limited).

## 1. What was built

- **`fastmace/kernels/triton/tf32x3_gemm.py`:** a custom op `fastmace::tf32x3_mm(a, b)`. It is OPT-T01's Triton GEMM
  with `tl.dot(input_precision="tf32x3")`. The tile is the one fixed tile of `tile_rule`: 64 × 64 × 32, 4 warps,
  3 stages. K is reduced in a fixed order and there are no atomics.
  - Its gradients run through the same op and are differentiable again.
  - `Tf32x3Matmul` uses the contiguous transpose for the input gradient where a module keeps one.
  - Off CUDA, and for other dtypes, it is PyTorch's product.
  - No global TF32 flag is set. cuBLAS stays IEEE everywhere else.
- **`segment_gemm`:** its kernel takes the `tl.dot` precision. With `"tf32x3"` every joined product runs in Triton,
  because the products the shape rule would leave to cuBLAS are IEEE.
- **The record `tf32x3_gemms`** (`fastmace/rewrites/precision.py`, `RewriteConfig.tf32x3_gemms`):
  - It is `rounding`, `deterministic`, default off and `never_automatic`.
  - It requires `block_linear` and conflicts with `grouped_launch`.
  - It runs after the species family, so it also covers the composed block linears.
  - It flags every module that multiplies atom rows: `BlockLinear`, `ResidualBlockLinear`, `GatedRows` (joined
    products), `SpeciesSkip` and `FirstInteractionReassociation`.
  - Its self-check compares each rerouted product with the IEEE product, values and input gradients.
  - The planner refuses it off float32, below compute capability 8.0 (from the detected device, not a table), and
    beside `grouped_launch`.
- **Tools:**
  - `tools/tf32x3_accuracy.py`: real operands of one force call (below).
  - `tools/bench/nve_drift.py --rewrites FIELD=VALUE`.
  - `tools/force_stress_gate.py --rewrites tf32x3_gemms=on`, which needed no change.
- **Tests:**
  - `tests/cpu/test_tf32x3_gemms.py`, 9 tests: the record's fields and order; every atom-GEMM module flagged on tiny
    MH-1 and residual cuEq models, with energy and forces bitwise equal to the stock path on the CPU; the planner's
    refusals; first and second derivatives through an fx trace.
  - `tests/gpu/test_tf32x3_gemm_cuda.py`, 7 tests: FP32-class error on Gaussian operands, bitwise repeat and CUDA-graph
    replay, gradients, joined products against the dense op.
  - `test_config`, `test_report` and `docs/design/switch-map.md` have the new field.

**Which GEMMs it covers.** A profiler pass of one force call with the record on lists what is still on cuBLAS:
- the edge-level radial networks: `[E, 10] @ [10, 128]`, `[E, 64] @ [64, 64]`, the folded last layer
  `[E, 68] @ [68, 1280]`, and their gradients;
- the cell's 3 × 3 products;
- species-only products, which have no position gradient: `node_attrs @ E0` (`[atoms, 89] @ [89, 6]`) and two
  tables of 13 and 29 rows.

Every position-dependent atom GEMM is rerouted: 52 launches per MH-1 force call, forward and input gradients, 35
shapes.

## 2. Runs

| Run | Machine, GPU | Environment | FastMACE |
| --- | --- | --- | --- |
| local | Fabi-PC, RTX 5060 Ti (sm_120, 36 SMs), driver 610.88, WSL2 | `mace_cueq_oeq` (torch 2.14.1+cu130, Triton 3.8.0); references in `mace_cueq_latest` | `5f48936` (gates, kernel test), `59672a2` (accuracy with the subnormal floor), `46b2cd2` (OMOL-0) |
| Colab A100 | A100-SXM4-40GB (sm_80), driver 580.82.07, 12 CPUs | Colab venv: torch 2.14.1+cu130, Triton 3.8.0, cuEq 0.12.0, OEq 0.7.0, MACE 0.3.16 | `5f48936` |

- **Model:** MH-1 multi-head (`mace-mh-1.model`, label `mh1-r2scan`), hybrid float32, every plan default. OMOL-0 XL:
  `omol0-xl`.
- **Structures:** 3×3×3 r²SCAN supercells from the baseline export `~/fastmace-baseline/1a-T18`.
- **Local GPU:** each job held its own `tools/ci/local_gpu.py` reservation, 24 October 2026, 10:14–12:00 UTC. The
  jobs are untimed.
- **Colab:** one VM, `fm-a100-opt4t03-100213`, 10:02–11:04 UTC, stopped by the driver.
  - Wall time 62 min. At 5.3 CU/h that is about 5.5 CU for this VM.
  - The account's balance fell 56.09 → 46.43 (9.66 CU), but other workers' A100 and L4 VMs were active at the same
    time (`active_after` 1 in `colab.json`).
- **Records:** `results/opt4-T03/` holds `local-5f48936/`, `local-59672a2/`, `local-46b2cd2/`, `colab-a100/` and the
  two run scripts. The value arrays (`*.npz`) stay in WSL under `~/fastmace-opt4/`.

## 3. Real operands (`tools/tf32x3_accuracy.py`)

**Method:**
- One force call of the compiled evaluation plan (no CUDA graphs, so every launch passes through Python) on a
  rattled supercell (±0.05 Å, seed 20261024), after a warm-up call.
- Every rerouted launch is checked: the tf32x3 output, and cuBLAS IEEE on the same float32 operands, each against a
  float64 product.
- Errors are max relative / relative Frobenius / cancellation-scaled max |c − c64| / Σ|a·b|
  (`opt-T01-tensor-cores.md` §6–§7).
- The cancellation-scaled error counts only outputs with Σ|a·b| ≥ 2^-103, float32's smallest normal over its unit
  roundoff. Below that, both modes lose the value to underflow, not to the split. The first run had no floor, and its
  worst entry was 0.5–0.63 in both modes alike, at Σ|a·b| ≈ 1e-47.

Worst over the 52 products of one call (local, `59672a2`):

| Structure | tf32x3 | cuBLAS IEEE | tf32x3 / IEEE per product: median, max | tf32x3 worse on |
| --- | --- | --- | --- | --- |
| NaCl (216 atoms) | 9.1e-7 / 2.2e-7 / 5.2e-7 | 8.5e-7 / 3.4e-7 / 6.4e-7 | 0.60, 6.9 (cancellation-scaled) | 9 of 52 |
| NbCl5 (1,944 atoms) | 7.9e-7 / 2.2e-7 / 4.7e-7 | 1.7e-6 / 4.8e-7 / 1.0e-6 | 0.31, 6.9 | 7 of 52 |
| OMOL-0 XL, NaCl (`46b2cd2`, 106 products) | 1.2e-6 / 3.0e-7 / 5.7e-7 | 1.2e-6 / 4.3e-7 / 1.0e-6 | 0.68, 2.7 | 29 of 106 |

- **The real operands cancel.** The median of Σ|a·b| / |c64| over a product's outputs reaches 30–33, and single
  outputs reach 5e7–6e8.
- **tf32x3 stays FP32-class on them.** Its worst cancellation-scaled error is below IEEE's: 8–9 float32 unit
  roundoffs (2^-24) of Σ|a·b|, against 11–17 for IEEE.
- **Where it is worse.** The products where tf32x3 is up to 6.9× worse than IEEE are those with K = 6 (species-sized
  inputs), on which cuBLAS's IEEE result is nearly exact (2.5e-8). In absolute terms, tf32x3 stays at 1–3e-7 there.
- **The A100** (`5f48936`, without the floor) gives the same worst errors: NaCl 9.6e-7 against 8.8e-7, NbCl5 8.8e-7
  against 1.6e-6.
  - cuBLAS IEEE runs other kernels on sm_80. On NaCl's small shapes it is more accurate than on sm_120: the median
    per-product ratio is 1.15, and tf32x3 is worse on 29 of 52 products.
- **§7's synthetic cancellation set does not occur here.** It made 3×TF32 7–14× worse than IEEE. In MACE's operands
  it is not seen.

## 4. Force gate (`tools/force_stress_gate.py`, local)

- **Setup:** per structure, 3 hydrostatic and 3 full-cell configurations (rattle 0.05 Å, strain ±2 %, gate seed
  20261015).
- **References:** the float64 reference and the float32 calculator (`ase_f32`), each in its own process.
- **Production:** the minimal planner with every default, plus `--rewrites tf32x3_gemms=on`. The same gate with the
  default plan ran beside it.

Production error against float64 / float32 calculator's error (ratio), largest over the 6 configurations:

| Structure | Plan | Energy, meV/atom | Forces, eV/Å | Stress, eV/Å³ | Gate |
| --- | --- | --- | --- | --- | --- |
| NaCl | **tf32x3** | 3.5e-4 / 7.7e-4 (0.45) | 4.20e-6 / 3.85e-6 (1.09) | 3.4e-8 / 9.0e-8 (0.38) | **passed** |
| NaCl | default | 3.2e-5 / 7.5e-4 (0.04) | 4.38e-6 / 3.82e-6 (1.15) | 3.0e-8 / 9.0e-8 (0.33) | passed |
| NbCl5 | **tf32x3** | 3.3e-4 / 7.3e-4 (0.45) | 8.05e-5 / 8.06e-5 (1.00) | 7.8e-8 / 9.9e-8 (0.79) | **passed** |
| NbCl5 | default | 1.4e-4 / 7.2e-4 (0.19) | 8.02e-5 / 8.04e-5 (1.00) | 7.4e-8 / 9.9e-8 (0.75) | passed |
| OMOL-0 XL, NaCl | **tf32x3** (`46b2cd2`) | 9.4e-3 / 5.8e-3 (1.62) | 8.8e-4 / 7.0e-4 (1.26) | 3.4e-6 / 4.7e-6 (0.71) | **passed** |
| OMOL-0 XL, NaCl | default (`59672a2`) | 7.5e-3 / 5.8e-3 (1.30) | 9.1e-4 / 7.5e-4 (1.22) | 3.1e-6 / 3.9e-6 (0.80) | passed |

- **The rule:** ≤ 2× the float32 calculator, or the quantity's floor. Every ratio is at most 1.09.
- **Forces, tf32x3 against the default plan directly:**
  - NaCl: the largest component difference over the 6 configurations is 3.9–6.0e-7 eV/Å.
  - NbCl5: 2.9–3.8e-6 eV/Å.
  - Both are 4–25× below the float32 error itself.
  - The float64 references of the two gates agree within 5e-15.
- **Energy shifts systematically.**
  - On all six NaCl configurations, tf32x3's energy is 6.2–7.5e-5 eV above float64 (about 3e-4 meV/atom). The
    default plan's errors are 0.3–6.9e-6 eV.
  - On NbCl5 the offset is about 1.9e-4 meV/atom.
  - An offset of one sign is what a biased split produces: Triton rounds the small parts by truncation.
  - It is still under half the float32 calculator's error. As a near-constant offset it does not affect forces or
    dynamics, but MC energy differences would see it only where configurations differ in it.

## 5. NVE drift (`tools/bench/nve_drift.py`, Colab A100)

Setup: the default plan with and without `tf32x3_gemms=on`, 3×3×3, 300 K, 1 fs, 20 ps, the conserved quantity every
10 fs (phase 4's protocol). The same VM ran both paths.

| Structure | Plan | Drift, meV/atom/ps | RMS, meV/atom | Range, meV/atom |
| --- | --- | ---: | ---: | ---: |
| NaCl | default | 1.1e-6 | 2.74e-4 | 6.8e-3 |
| NaCl | **tf32x3** | 1.4e-7 | 2.73e-4 | 6.7e-3 |
| NbCl5 | default | −3.0e-6 | 3.77e-4 | 1.25e-2 |
| NbCl5 | **tf32x3** | −5.2e-6 | 3.78e-4 | 1.26e-2 |

- Both plans are at the level phase 4 recorded on the G4 (2.1e-6 and 2.3e-6 meV/atom/ps).
- The RMS is equal to three digits.
- tf32x3 adds no measurable drift. **Pass.**

**Speed, untimed.** These are the drift loops' wall times, with Python sampling every 10 steps, not an A/B. On the
A100:
- NaCl: 3.1 ms/step default against 4.2 ms with tf32x3;
- NbCl5: 13.7 against 15.7 ms.

Triton's `tf32x3` kernel is slower than cuBLAS's SIMT FP32, as OPT-T01 §4 measured it (0 of 92 shapes won). Any
gain needs stage 2's CUTLASS kernel (OPT-T01 §5: 0.55–0.75× of cuBLAS GEMM time).

## 6. OMOL-0 XL

- **Force gate (local, sm_120): passed.** The rows are in §4.
  - OMOL-0's own float32 scatter is large: the float32 calculator is off by 7e-4 eV/Å, and two float32 processes
    differ by 2e-4 eV/Å (1b-T16).
  - tf32x3 against the default plan: forces 1.1–3.9e-4 eV/Å apart. The energy is 3–4e-4 eV higher on every
    configuration (about 1.7e-3 meV/atom), the same one-signed offset as on MH-1.
- **Real operands:** 106 products per call, forward and input gradients.
  - tf32x3 is FP32-class: its worst cancellation-scaled error is 5.7e-7, against IEEE's 1.0e-6.
  - It is worse than IEEE on 29 of 106 products, by at most 2.7×.
- **Coverage gap.** OMOL-0's nonlinear readouts are `torch.nn.Linear` (`addmm` with a bias, 1024 → 192), not
  cuEquivariance Linear layers, so the record leaves them, and their input gradients, on cuBLAS IEEE.
  - Together with a 2048 → 1024 product that runs once per call with no gradient, that is 1.8 of 120 GFLOP of
    atom-row GEMMs per call (1.5 %).
  - A stage 2 would cover them with a small `nn.Linear` wrapper. Stage 1's accuracy result is not affected at this
    share.
- **The fx-trace bug found here.** OMOL-0's first runs failed with "no autograd formula" for `tf32x3_mm`.
  - A block linear inside a block that MACE traces with fx (`mace/modules/blocks.py:145`) records the op itself,
    not `Tf32x3Matmul`.
  - The op now has its own differentiable gradient (`46b2cd2`, with a CPU test that traces and differentiates
    twice).
  - MH-1's runs do not reach that path, so their results at `5f48936` stand.
- **On the Colab A100, both plans failed for reasons that are not tf32x3's:**
  - The build of OMOL-0 on NaCl fails `reassociate_first`'s self-check, relative error 1.2–1.3e-4 against the
    float32 threshold of 1e-4, for the default plan as for tf32x3. That is 5-T03's known A100 margin case.
    `force_stress_gate.py` calls `planner.build` directly, so rule 13 does not re-plan without it.
  - The default plan's NVE run reached a non-finite energy at step 70, and the tf32x3 run gave NaN (it ran
    `5f48936`, before the fix).
  - A rerun on the A100 would need `--rewrites reassociate_first=off segmented_first_sum=off` on both sides, and a
    structure in OMOL-0's domain for the drift. None is in the baseline export. It is not run: Colab budget, and the
    gate's accuracy question is answered locally.

## 7. Verdict and what it leaves for the owner

- **Stage 1 passes** (MH-1 gate and drift, OMOL-0 gate; OMOL-0's drift is model-limited on the only available
  structure, §6). For MH-1:
  - The force gate passes on NaCl and NbCl5, with force ratios 1.09 and 1.00 against the default plan's 1.15 and 1.00.
  - Forces move by less than a tenth of the float32 error.
  - NVE drift on the A100 is unchanged.
  - On the real operands of a force call, 3×TF32 is FP32-class and on the worst products more accurate than cuBLAS
    IEEE.
  - The plan's fall-back, one corrected (Ootomo–Yokota) variant, is not needed.
- **One systematic effect:** a near-constant energy offset of 2–3e-4 meV/atom (§4), under half the float32
  calculator's error. A CUTLASS kernel uses the same three-product split. Its gate would show whether its rounding
  of the small parts has the same bias.
- **Not done, by the brief:** stage 2, the CUTLASS 3×TF32 custom op. The record and its applicability rule are
  ready for it: stage 2 would replace the kernel behind `fastmace::tf32x3_mm` and `segment_gemm`'s `tf32x3` path.
- **The owner decides:**
  1. whether to build stage 2;
  2. whether "compute capability ≥ 8.0 with TF32 tensor cores, from the detected device" is a general enough
     capability rule;
  3. whether the rule should restrict the record to devices where 3×TF32 can win. By OPT-T01 that is sm_80; on
     sm_89 and sm_120, TF32 is under 3× FP32.
- **Default:** none. The record stays opt-in (`never_automatic`). On sm_120 and the L4 it can only be slower.

## 8. Decision (owner, 24 October 2026)

**Stage 2 is not built now.** The owner: "Go ahead and do it as you recommended". The conductor's recommendation:

- **No production A100.** The A100 is not a production GPU for FastMACE: it is Colab-only, and the compute units
  are limited.
- **No gain elsewhere.** On the production GPUs (the local RTX 5060 Ti, the Ada workstations, Modal), 3×TF32 is
  slower than FP32.
- **Stage 1 is enough for now.** It answers the accuracy question. The record `tf32x3_gemms` stays opt-in
  (`never_automatic`). Stage 2 can start from it if the A100 becomes a production target.

No default changes, so SPEC §19 has no entry, as for OPT4-T07's close.

## 9. Follow-up: the gate tool applies rule 13 (24 October 2026, conductor, branch `fastmace/opt4-T03d`)

The OMOL-0 failure on the A100 (§6) was the gate tool's, not production's: `tools/force_stress_gate.py` called
`planner.build` once, so `reassociate_first`'s self-check (1.2–1.3e-4 against SPEC §6.4's float32 threshold of
1e-4) ended the run. Production's fallback revises the plan by rule 13 (`reactions.selfcheck_revision`: the
record and its dependants off) and builds again. The gate tool now does the same (`build_with_rule13`) and records
the revisions in the plan record (`rule13_revisions`). CPU test with an injected self-check failure in
`tests/cpu/test_gate_rewrites_option.py`. The threshold itself is SPEC §6.4 and unchanged. The A100 build of
OMOL-0 XL is exercised by the 5-T03 memory rerun.
