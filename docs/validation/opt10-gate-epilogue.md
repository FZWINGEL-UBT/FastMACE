# OPT10: the record `gate_epilogue`, opt-in, and its per-GPU default proposal (29 October 2026)

Plan: `plans/opt-10.md` (OPT10-T01 and its record). The screen it comes from is `docs/validation/opt10-T01.md`.
Decision rule: SPEC decision 66 (per-GPU defaults by exact device name, where same-process whole-step A/Bs in both
build orders measured a speed-up). Runs and scripts: `results/opt10-gate-epilogue-20261029/`.

**Verdict.**
- **Backward half:** built as the opt-in record `gate_epilogue`. Rounding-class; the force/stress gate passes on
  both GPUs and the NVE drift is unchanged. The whole-step gain is much smaller than the subgraph screen suggested:
  0.2-0.5 % on the RTX 4000 Ada, and 2 % (NaNbO3) and 0.1-0.7 % (NbCl5) on the RTX 5060 Ti.
- **Forward half:** dropped. Its Triton forward is slower than today's cuBLAS chain in the subgraph on every cell
  (1.05-1.29×), so no record was built.

**Final A/Bs** on the branch with main `030e2fe2` merged (FastMACE `61657b29` local, `8497f2fe` on murphy; the two
differ only in docs and golden texts). On the RTX 5060 Ti that plan includes decision 66's `grouped_launch` for
MH-1:

| Cell | RTX 5060 Ti (local), b/a, orders ab / ba | RTX 4000 Ada (murphy), b/a, orders ab / ba |
| --- | --- | --- |
| NaNbO3 (1,080 atoms) | **0.980 / 0.978** | **0.9966 / 0.9952** |
| NbCl5 (1,944 atoms) | **0.993 / 0.999** (pairs 0.980-1.017) | **0.9984 / 0.9986** |
| NaCl (216 atoms) | 1.007 / 0.980 (same code on both sides) | 0.9993 / 1.0008 (same code on both sides) |

**Before the merge** (FastMACE `cccb0521` local, `d903a4cc` on murphy):

| Cell | RTX 5060 Ti (local), b/a, orders ab / ba | RTX 4000 Ada (murphy), b/a, orders ab / ba |
| --- | --- | --- |
| NaNbO3 | 0.983 / 0.988 | 0.9965 / 0.9963 |
| NbCl5 | 1.000 / 1.002 | 0.9978 / 0.9983 |
| NaCl | 1.017 / 0.996 (same code) | 1.001 / 1.001 (same code) |

**Protocol.**
- b = `gate_epilogue=on`, a = off, everything else the default plan (MH-1 r2SCAN, hybrid float32, NVT 300 K).
- `tools/ab_rewrites.py`: 300 warm-up steps, then 12 alternating blocks of 300 steps; ABAB for order ab, ABBA for
  ba.
- Every run is valid: nothing compiled while timed, and no other GPU process.
- On NaCl the shape rule never picks Triton (1,512 rows for l = 3, below 2 waves of 64-row tiles on 36 or 48 SMs).
  Both sides run the stock path, so the spread there is noise.

## Proposed default (decision 66), for the owner

- **On for "NVIDIA RTX 4000 Ada Generation".** Both cells where the record acts measured faster in both orders,
  before and after the merge. The gain is small (0.2-0.5 %), but every per-pair ratio is at most 1.001.
- **On for "NVIDIA GeForce RTX 5060 Ti", by the letter of the rule; marginal.**
  - The final A/Bs are faster in both orders on both cells where the record acts.
  - NaNbO3 is a clear 2 %.
  - NbCl5 is within noise: 0.993 / 0.999, with pair ratios up to 1.017; it was 1.000 / 1.002 before the merge.
  - The fixed tile is not this GPU's best for NbCl5 (below). If the owner wants only clear gains, leave this GPU
    off.
- **Off for every other GPU.** None is measured. The A100 (Modal) waits for November's budget.

Not made default; `RewriteConfig.gate_epilogue` stays `never_automatic`.

## What was built

**`fastmace/kernels/triton/gated_products.py`.**
- `gated_joined_products(scalars, gated, gates, ...)`: `linear_2`'s joined products over the gated rows as one
  autograd function.
- Its forward is `segment_gemm`'s: the elementwise `gated * gate`, then `joined_products_op`. It is bitwise the
  same.
- Its backward is the custom op `fastmace::gated_products_input_grads`, with a fake kernel for torch.compile:
  - **Gated segments the shape rule gives to Triton** (`tile_rule.segment_gemm_choice`): OPT10-T01's atom-tiled
    kernel. Per (atom block, column block) it runs one K-loop per component m and loads the gate tile once. It
    stores `dgated = acc * gate` and sums `dgate = sum_m acc * gated` in registers: no `drows` tensor, no atomics.
  - **Elsewhere:** the stock operations. That covers segments the rule leaves to cuBLAS, the scalars' product
    (segment_gemm's Triton GEMM or cuBLAS), and the CPU.
- **Tile:** one fixed configuration, 32 atoms × 64 columns × 32 K, 4 warps, 3 stages. It was the fastest of four on
  OPT10-T01's Ada quick pass.

**Wiring.**
- `GatedRows.fold_gate_backward()` sets the flag. `GatedRows.forward` takes the new path where `segment_gemm`'s
  joined path runs (`_any_triton`), unless `tf32x3` is on.
- **Record** `gate_epilogue` (fusion family):
  - requires `segment_gemm` and conflicts with `tf32x3_gemms`;
  - `rounding`, deterministic, default `off`.
- **Self-check:**
  - the chains against the unrewritten interactions;
  - the folded products against `joined_products` over autograd-built rows, on enough atoms that every product
    takes Triton.
  - Its build memory is declared as `segment_gemm`'s (the same probe size).
- **`RewriteConfig.gate_epilogue`:**
  - `auto` rule `never_automatic`;
  - `on` requires `segment_gemm`;
  - refused in float64 (`gate-epilogue-needs-float32`) and beside `tf32x3_gemms`.
- **Docs and tests touched:** `docs/design/switch-map.md` (both tables), `CHANGELOG.md`, the golden plan texts, and
  the report golden (the field shows as off).

**Tests.**
- `tests/cpu/test_gate_epilogue.py`:
  - the CPU op equals `joined_products` over the same rows: forward bitwise, gradients within 1e-13 in float64;
  - an unread gate column gets a zero gradient;
  - the fake kernel's shapes;
  - the record's properties and order;
  - on the tiny MH-1 model the record applies, self-checks, and leaves energy and forces bitwise. With the shape
    rule forced to the joined path, energy is bitwise and forces are within 1e-12;
  - the planner's rules.
- `tests/gpu/test_gated_products_cuda.py`:
  - the fused kernel against the chain, gradients within 1e-5 relative, including a partial atom tile;
  - deterministic: two launches bitwise;
  - the cuBLAS path on 8 atoms;
  - compiled equals eager.
- Passed on the RTX 5060 Ti and on the RTX 4000 Ada (`check.log`), together with
  `tests/gpu/test_segment_gemm_cuda.py`.
- CPU targeted files: `test_gate_epilogue`, `test_rewrites_fusion`, `test_planner`, `test_config`, `test_report`,
  `test_plan_text`, `test_tf32x3_gemms`, `test_memory_estimate`, and the other files that read `REWRITE_FIELDS`
  or `AUTO_RULES`. Before the merge, in local WSL: 753 passed, 27 skipped.
- **Full CPU suite after the merge,** on murphy's CPUs (no CPU tests on the local PC since 29 October), GPU hidden,
  `-n 8`, basetemp on the NAS (`murphy-61657b29-cpu/`): 4,601 passed, 17 failed.
  - 16 of the failures are `test_colab_driver.py`: `git config` exits 128 in a test repo on the NAS, an
    environment problem unrelated to this branch.
  - 1 was `test_docs_options` (the new field had no row in `docs/reference/rewrites.md`), fixed in `8497f2fe`.
  - The affected files then passed on `8497f2fe` (`murphy-8497f2fe-cpu/`): 358 passed, 20 skipped.

## Force/stress gate (`tools/force_stress_gate.py --rewrites gate_epilogue=on`)

The production path's largest error against float64 (MH-1 r2SCAN), and in brackets its ratio to the float32
yardstick's (limit 2, or the floor).

| Cell | GPU | energy (meV/atom) | forces (eV/Å) | stress (eV/Å³) |
| --- | --- | --- | --- | --- |
| NaCl | RTX 5060 Ti | 6.7e-5 (0.09) | 3.8e-6 (1.00) | 3.2e-8 (0.36) |
| NbCl5 | RTX 5060 Ti | 7.7e-5 (0.11) | 8.3e-5 (1.04) | 4.0e-8 (0.40) |
| NaNbO3 | RTX 5060 Ti | 4.0e-4 (0.32) | 1.35e-4 (1.01) | 1.9e-7 (0.60) |
| NaCl | RTX 4000 Ada | 7.4e-5 (0.80) | 3.9e-6 (0.97) | 3.8e-8 (0.93) |
| NbCl5 | RTX 4000 Ada | 9.6e-5 (0.12) | 8.4e-5 (1.04) | 3.2e-8 (0.31) |
| NaNbO3 | RTX 4000 Ada | 3.2e-4 (0.29) | 1.34e-4 (0.99) | 1.7e-7 (0.51) |

All passed.

## NVE drift (`tools/bench/nve_drift.py`, 20 ps at 1 fs, 300 K, RTX 5060 Ti)

| Cell | gate_epilogue | drift (meV/atom/ps) | RMS (meV/atom) | range (meV/atom) |
| --- | --- | ---: | ---: | ---: |
| NaCl | off | −1.8e-7 | 2.74e-4 | 6.96e-3 |
| NaCl | on | −2.6e-7 | 2.74e-4 | 6.96e-3 |
| NbCl5 | off | 2.5e-7 | 3.80e-4 | 1.27e-2 |
| NbCl5 | on | −1.3e-6 | 3.80e-4 | 1.27e-2 |

Unchanged. On NaCl the record does not act on this GPU; the run confirms the plan with it on.

## Why the step gains less than the subgraph

**The screen's subgraph.** OPT10-T01 measured 0.27-0.31 ms saved per gated interaction on the Ada, with synthetic
operands against a compiled stand-in for the chain. The step saves far less: about 0.04 ms (NbCl5) and 0.06 ms
(NaNbO3) per step on the Ada.

**On the RTX 5060 Ti** the probe (`t01-probe-local-rtx5060ti.jsonl`) shows the tile matters:

| Cell | best tile | fused / chain, best tile | 32 × 64 tile, quick pass (ms) | best tile, quick pass (ms) |
| --- | --- | ---: | ---: | ---: |
| NbCl5 | 32 × 128 | 0.980 | 2.86 | 2.35 |
| NaNbO3 | 64 × 64 | 0.876 | 1.10 | 1.10 |

The record's fixed tile (32 × 64, Ada's best) is about 20 % slower on NbCl5 there, which fits NbCl5's neutral step
on that GPU.

**Possible follow-up.** A tile rule derived from the atom count and the SM count could recover the 5060 Ti's
NbCl5 gain. A per-GPU tile table is not allowed (SPEC §9.1); decision 66 covers on/off defaults only.

**Activation derivatives.** They were not moved into the epilogue. With them, the kernel would also need the gate
pre-activations and would write the 0e rows' gradient. Today Inductor computes them in the kernel that assembles
that gradient anyway, and the measured whole-step gain left no room to look for more. Not measured.

## Forward half: dropped (subgraph probe)

**Probe.** `tools/bench/opt10_gate_forward.py` uses synthetic operands with MH-1 interaction 1's shapes:
- linear_1 `128x0e+128x0e+3x128x1o+3x128x2e+2x128x3o`;
- linear_res `128x0e+128x1o`;
- outputs `2048x0e+512x(1o,2e,3o)`.

It compares two sides:
- **Chain:** today's compiled forward. Density division, row concatenation, cuBLAS products, then the gate.
- **Fused:** one Triton kernel per output segment. Both K ranges are read in place, and `1 / (density * beta +
  alpha)` is applied between them. The epilogue holds silu/sigmoid and `v * gate`, with an atom-tiled m-loop for
  the gated segments.

Local RTX 5060 Ti, forward only, best of four tiles:

| Cell | chain (ms) | fused v1 (ms) | v1 / chain | fused v2 (ms) | v2 / chain |
| --- | ---: | ---: | ---: | ---: | ---: |
| NbCl5 | 1.48 | 1.71 | 1.16 | 1.56 | 1.05 |
| NaNbO3 | 0.73 | 0.85 | 1.16 | 0.81 | 1.12 |
| NaCl | 0.17 | 0.22 | 1.33 | 0.21 | 1.29 |

v1 gathers each K element's piece offset; v2 loops over the 128-channel pieces. The cuBLAS products plus Inductor's
fused elementwise kernel beat a gathered Triton GEMM with the gate in its epilogue on every cell. That is the risk
the review named ("moving the merged product from cuBLAS to Triton"). It was not measured on the Ada.

## Setup

- **Local:** Fabi-PC (WSL2), RTX 5060 Ti (sm_120, driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130),
  `tools/ci/local_gpu.py run --quiet` per job group. The idle check runs before each job (`run.log`).
  - FastMACE `cccb0521` (`local-cccb0521/`): A/Bs, gate, drifts.
  - FastMACE `61657b29`, with main merged (`local-61657b29/`): the final A/Bs.
- **murphy:** RTX 4000 Ada (sm_89, driver 595.99.02), NAS `mace_cueq_oeq` (torch 2.14.1+cu130). The other-users and
  GPU check runs before every job (`run.log`).
  - FastMACE `d903a4cc` (`murphy-d903a4cc/`): check, gate and A/Bs. It differs from `cccb0521` only in the run
    script's test Python.
  - FastMACE `8497f2fe`, with main merged (`murphy-8497f2fe/`): check, gate (passed again) and the final A/Bs.
- **Unit tests** use `mace_cueq_latest`: the OEq environments have no pytest.
- **Gate references:** `mace_cueq_latest` float64 and float32. The baseline structures are
  `fastmace-baseline/1a-T18`.

**Not run:**
- A100 (Modal): November budget.
- NaCl drift with the record acting: it acts on NaCl on no measured GPU.
- The drifts were not repeated after the merge.
