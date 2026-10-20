# OPT-T02 stage A: `gated_rows` (residual products, gate and `linear_2` on rows)

Plan: `plans/opt-1.md` OPT-T02; design `docs/design/opt-T02-fusion.md`, stage A (approved 20 October 2026).
Branch `fastmace/opt-T02a`, from main `29203fe`.

**Result:** opt-in record `gated_rows` (`RewriteConfig.gated_rows`, default off).
- It is exact: bitwise equal on the CPU in float64, and on the GPU in the reproducible mode (plain cuEq
  float64, NVT and NPT). Outside that mode on against off scatters exactly as much as off against off.
- It passes the hybrid float32 force gate.
- On murphy (sm_89) it shortens the MH-1 step by **9.0 % on NbCl5** and **4.4 % on NaCl**, same process,
  both build orders.
- The local sm_120 A/B waits for the PC. Making it a default is for the conductor and owner after that
  A/B.

## What it does

In a `RealAgnosticResidualNonLinearInteractionBlock` that runs FastMACE's folded MH-1 forward, it keeps
the per-irrep products as rows `[atoms · (2l + 1), channels]`:
- through the gate, and
- into the block linear `linear_2`.

That replaces the stock chain, which joins the merged residual products into one `[atoms, 9728]` tensor
for MACE's `GatedEquivariantBlock`, then slices and copies the gate's output back into rows for
`linear_2`. The products are the block linears' own, and the gate's arithmetic is copied element by
element (its forward is pinned by sha256 in the record).

- **Record:** `fastmace/rewrites/fusion` (new family, after the linear family).
- **Requires:** `block_linear` and `merged_residual`. **Applies to:** interactions with MACE's
  `GatedEquivariantBlock` (the MH-1 class).
- **Classification:** `exact`, `deterministic=True`, no workload constraints.
- **Self-check (§6.4):** each chain against the unrewritten interaction's
  `linear_2(gate(linear_1(x) + linear_res(y)))`, values and both input gradients.
- **Planner:** `never_automatic`, so only an explicit `"on"` enables it. `BlockLinear` and
  `ResidualBlockLinear` expose their products and output rows through the same operations as before, so
  the default path is unchanged.

## Correctness

| Check | Where | Result |
| --- | --- | --- |
| Chain on rows against the stock chain, float64, values and input gradients | CPU, tiny MH-1-class cuEq model (`tests/cpu/test_rewrites_fusion.py`) | **bitwise equal** |
| Whole model, MH-1 package + linear family, `gated_rows` off against on, energy and forces, float64 | CPU, same | **bitwise equal** |
| Self-check against the unrewritten cuEq modules | CPU float64 | 5.2e-16 relative (block linears' own rounding), threshold 1e-10 |
| CPU suite | murphy CPUs, NAS `mace_cueq_latest` | 2652 passed after the registry-table updates (switch map, AUTO_RULES, report golden), then those re-run green |
| First force evaluation, plain cuEq float64, NaCl and NbCl5 3×3×3 rattled; MH-1 package and block linears requested on both sides | murphy GPU, NVT and NPT | on vs off: forces 3.6e-15 eV/Å, energy 2.5e-11 eV, stress 1.2e-17. Off vs off in the same process: forces 4.0e-15, energy 8.7e-11. The plain cuEq engine is not bitwise reproducible on the GPU (fused cuEq kernels), so on against off cannot be bitwise; its scatter is the engine's own |
| The same, reproducible mode (`Workload(reproducible=True)`, deterministic kernels, prepared before CUDA starts) | murphy GPU, NVT and NPT, commit `800e9b5` | **bitwise equal** on against off: energy, forces and stress, NaCl and NbCl5 (off against off bitwise too); `gated_rows = on` in every "on" plan |
| First force evaluation, hybrid float32 (planner defaults + `gated_rows`) | murphy GPU | on vs off: forces 1.9e-6 eV/Å (NbCl5), off vs off 1.5e-6 (the streaming convolution's atomic sums): the same scatter |
| Force, stress and energy gate (`tools/force_stress_gate.py --engine fastmace --rewrites gated_rows=on`), 3 configurations per cell mode | murphy | **passed**. NaCl ratios: energy 1.52, forces 1.16, stress 0.68. NbCl5: 0.23, 0.99, 0.78. `gated_rows = on` in all four production plans |

## Speed (murphy, RTX 4000 Ada, sm_89)

Setup:
- **Run:** `results/opt-T02a-gated-rows/ab_gated_rows.py`, same process, two engines from the minimal
  planner that differ only in `gated_rows`, MH-1 hybrid float32 NVT at 300 K.
- **Protocol:** 300 warm-up steps per side outside the compile-counter window, then 12 alternating
  blocks of 300 steps.
- **Environment:** NAS `mace_cueq_oeq` through the hybrid launcher, commit `c1bca8a`.
- **Conditions:** GPU empty, 1-minute load 1.0–1.2 before and after, 19:04–19:16 UTC 20 October 2026.
  No compilation while timed.

| Structure | Build order, pattern | off (ms/step) | on (ms/step) | on/off | Block-pair ratios |
| --- | --- | ---: | ---: | ---: | --- |
| NaCl 216 atoms | off first, ABAB | 3.627 | 3.472 | **0.957** | 0.956–0.964 |
| NaCl | on first, ABBA | 3.657 | 3.489 | **0.954** | 0.949–0.959 |
| NbCl5 1944 atoms | off first, ABAB | 31.40 | 28.55 | **0.909** | 0.909–0.912 |
| NbCl5 | on first, ABBA | 31.39 | 28.59 | **0.911** | 0.908–0.912 |

- Murphy's noise floor is 0.16 % (1a-T18). The first run (`stageA-20261020`, commit `fbb03ed`) gave the
  same ratios, 0.955 and 0.910. NaCl was flagged there only because the compile counters then included
  the warm-up.
- The gain exceeds the design note's estimate for NbCl5 (5–7 %, from the local card). murphy's
  atom-level elementwise time is larger (8.95 against 7.35 ms, 2-T20a §5), and the copies removed scale
  with it.
- **sm_120 (local RTX 5060 Ti):** pending, while the local PC is off limits. Its noise floor is 1.1 %
  (NaCl) and 0.4 % (NbCl5) (2-T12).

## Files

- New: `fastmace/rewrites/fusion/{__init__,modules}.py`, `tests/cpu/test_rewrites_fusion.py`,
  `tests/cpu/test_gate_rewrites_option.py`, `results/opt-T02a-gated-rows/`.
- Changed:
  - `fastmace/rewrites/linear/modules.py` (`products`, `output_rows`, `join_rows`; same operations);
  - `fastmace/rewrites/mh1/modules.py` (the folded forward reads `gated_rows`);
  - `fastmace/rewrites/registry.py` (`RECORD_SOURCES`);
  - `fastmace/config.py` (`RewriteConfig.gated_rows`);
  - `fastmace/planner/fixed.py` (rule, stage 1 and 2);
  - `docs/design/switch-map.md`;
  - `tests/cpu/test_config.py` and `tests/cpu/test_report.py` (registry tables);
  - `tools/force_stress_gate.py` (`--rewrites FIELD=VALUE` for FastMACE's minimal planner).
- Raw results: `results/opt-T02a-gated-rows/murphy/`.
