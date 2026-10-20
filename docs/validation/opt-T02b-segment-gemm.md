# OPT-T02 stage B: `segment_gemm` (`linear_2` writes its joined output in place)

Plan: `plans/opt-1.md` OPT-T02; design `docs/design/opt-T02-fusion.md` and the go/no-go note
`docs/design/opt-T02-BC-go-no-go.md` (stage B approved 20 October 2026). Branch `fastmace/opt-T02b`,
merged to main as `20118db` (opt-in). Default on by owner decision, 20 October 2026: branch
`fastmace/opt-T02a-default` (float32 plans that are not reproducible; float64 and `reproducible=True` keep
the stock path).

**Result:** record `segment_gemm` (`RewriteConfig.segment_gemm`), on top of `gated_rows`.
- `rounding` (another GEMM), deterministic (no atomics). It passes the hybrid float32 force gate.
- MH-1 NbCl5 3×3×3: **−4.8 to −5.2 %** per step on murphy (sm_89) and **−4.2 to −4.3 %** on the local
  RTX 5060 Ti (sm_120), same process, both build orders, on top of stage A.
- MH-1 NaCl 3×3×3: neutral on both GPUs. No product reaches the shape rule's threshold there, so the
  stock products and join run.
- The rule's tile configuration is within 2–5 % of the pre-check's best of five configurations, per
  product (below).

## What it does

In every `gated_rows` chain (stage A), `linear_2`'s per-l products write straight into the interaction's
joined `[atoms, 16 · 512]` output, and their input gradients read the gradient's rows in place. That
removes the join before the symmetric contraction and its backward (0.27–0.39 ms of copies per product
direction on NbCl5, table below).

- **Kernel:** `fastmace/kernels/triton/segment_gemm.py`, a gathered-row Triton GEMM with IEEE `tl.dot`
  (row `(atom, component)` addressing on the flat storage), as the custom ops `fastmace::joined_products`
  and `fastmace::joined_products_input_grads` with fake implementations and an autograd function. CPU and
  non-float32 inputs take a PyTorch path.
- **Shape rule:** `fastmace/kernels/triton/tile_rule.py`, shared with OPT-T03. Triton when the product's
  rows are at least 2 waves of 64-row tiles over the SMs, otherwise cuBLAS. Tiles are 64 × 64 × 32,
  4 warps, 3 stages. The rule uses only M, N, K and the SM count: no table of GPUs or shapes, and no
  autotuning (SPEC §9.1). When no product of a chain qualifies (NaCl-sized cells), the chain skips the
  joined op and runs the stock products and join, so it adds no launches.
- **View:** the interaction's `reshape_irreps` becomes `IrrepsView`, a view where its layout allows it
  (ir_mul, equal multiplicities). Without it the reshape copied the joined output again, and the first
  A/B (`6bd203d`) showed no gain. `reshape_irreps.forward` is pinned by sha256 in the record.
- **Record:** `fastmace/rewrites/fusion`, `requires=("gated_rows",)`. **Self-check (§6.4):** the chains
  as for `gated_rows`, the view against the stock reshape, and each `linear_2`'s joined products
  against its own products and join, values and input gradients, on enough atoms that the rule picks
  Triton for every product on the device.

## Correctness

| Check | Where | Result |
| --- | --- | --- |
| Joined products against products + join, values and input gradients | CPU (`tests/cpu/test_rewrites_fusion.py`) | equal within the GEMM's rounding; the skip path is the stock code |
| Triton kernel against cuBLAS, both directions, shapes on both sides of the rule | murphy GPU (`tests/gpu/test_segment_gemm_cuda.py`) | 2 passed (commits `6bd203d`, `3bb2845`, `3d301ec`) |
| Force, stress and energy gate (`tools/force_stress_gate.py --engine fastmace --rewrites gated_rows=on segment_gemm=on`) | murphy, commit `6bd203d` | **passed**. NaCl ratios: energy 1.29, forces 1.13, stress 0.72. NbCl5: 0.24, 1.01, 0.78. The later commits change only the reshape into a view (no arithmetic) and how the forward evaluates the rule |
| CPU suite | murphy CPUs, NAS `mace_cueq_latest` | 215 passed (`6bd203d`); ruff and 137 targeted tests (`3d301ec`); the conductor's trial of `c66a096`: 2678 passed, 288 skipped |

## The rule against the bench's upper bound

`tools/segment_gemm_bench.py` on murphy, NbCl5 3×3×3 shapes (1944 atoms; 9,720 and 13,608 rows).

| Product | cuBLAS + copy (µs) | cuBLAS only | best of 5 configurations (pre-check, `ff329d8`) | the rule's configuration (`--rule`, `6bd203d`) |
| --- | --- | --- | --- | --- |
| `linear_2` forward | 1750 | 1355 | 1377 | 1409 (+2.3 %) |
| `linear_2` input gradient | 1695 | 1414 | 1400 | 1436 (+2.5 %) |
| residual forward (not used by stage B) | 1098 | 916 | 897 | 945 (+5.4 %) |

The cuBLAS columns are the pre-check run's. The two runs are separate processes, so the comparison
carries cross-process noise (about 3 %). The rule's configuration takes 0.83–0.87 of the time of cuBLAS
plus the copy. It stays a few percent short of the best configuration per shape; choosing one per shape
would be tuning.

## Speed

Same-process A/B (`tools/ab_rewrites.py`): two engines from the minimal planner that differ only in
`segment_gemm` (a = `gated_rows=on`, b = `gated_rows=on segment_gemm=on`). MH-1 hybrid float32, NVT at
300 K, 3×3×3 cells, 300 warm-up steps per side outside the compile-counter window, then 12 alternating
blocks of 300 steps. Build order a-b with ABAB, then b-a with ABBA. All blocks valid, no compilation
while timed. Timed runs use pinned source copies.

| b/a | murphy, a-b / b-a | local, a-b / b-a |
| --- | --- | --- |
| NaCl (216 atoms) | 0.9972 / 0.9961 | 0.9974 / 0.9916 |
| NbCl5 (1944 atoms) | **0.9517 / 0.9479** | **0.9577 / 0.9567** |
| NbCl5 ms per step, a → b | 28.53 → 27.15 / 28.65 → 27.16 | 24.60 → 23.56 / 24.41 → 23.36 |

Noise floors of a 12-block same-process A/B: murphy 0.16 % (`docs/validation/1a-noise-floor.md`), local 1.1 % on NaCl and 0.4 % on NbCl5 (`docs/validation/2-T12-local-noise-floor.md`).

- **murphy:** RTX 4000 Ada (sm_89, 48 SMs), driver 595.58.03, NAS `mace_cueq_oeq` (torch 2.14.1+cu130,
  Triton 3.8.0, mace-torch 0.3.16, cuEquivariance 0.12.0, OpenEquivariance 0.7.0) through the hybrid
  launcher. Pinned copy `b-3d301ec` (`fastmace/opt-1-local` 3d301ec: `fastmace/opt-T02b` c66a096 merged
  with OPT-T03 and OPT-T01b, which add only tools and docs). GPU empty, 1-minute load 0.4–1.1,
  20:42–20:55 UTC 20 October 2026. Run `results/opt-T02b-segment-gemm/murphy/stageB5-3d301ec/`.
- **local:** RTX 5060 Ti (sm_120, 36 SMs), driver 610.88, WSL2, `~/environments/mace_cueq_oeq` (torch
  2.14.1+cu130, Triton 3.8.0). Pinned copy `~/fastmace-runs/src-3d301ec`, under the local GPU
  reservation. Before each job, utilisation was checked in both the Windows and WSL views, and a 5 s
  monitor logged utilisation, memory, clock and temperature (`monitor.log`). 20:42–20:55 UTC. Run
  `results/opt-T02b-segment-gemm/local/stageB3-3d301ec/`.
- **Structures and baseline:** ResearchModule tag `research-engine-baseline-20261017` (d2faa9fd),
  structures from the 1a-T18 export.

Step trace (`tools/step_breakdown.py`, murphy, `3bb2845`, `results/opt-T02b-segment-gemm/murphy/trace-3bb2845/`):
NbCl5 27.24 ms per step against 28.68 ms with stage A alone (`results/opt-T02-trace/`).

## Runs that do not count

- **`6bd203d` (murphy `stageB-20261020`):** before the view. NbCl5 1.000/1.000, NaCl 1.013/1.016. This
  run led to `IrrepsView`.
- **`3bb2845` (murphy `stageB2-ab`):** with the view, before the NaCl skip. NbCl5 0.949/0.949, NaCl
  1.003/1.002 (extra launches with no product on Triton). This run led to the skip.
- **`85d741d`, `4ddaf11` (murphy `stageB3-ab`, `stageB4-ab`; local `opt1-local-stageB-20261020`,
  `opt1-local-stageB2-20261020`):** the skip evaluated the shape rule inside the traced forward. NbCl5
  failed under torch.compile, with `Unknown format code 'f' for object of type 'str'` from the rule's
  reason string. NaCl ran: murphy 0.996/0.996 (both commits), local 0.993/0.999 (both commits). Fixed by
  `e78b3cb` (`tile_rule.uses_triton`, integer arithmetic only) and `c66a096` (the SM count read once at
  setup).

## Left over

- The 19-label force gate runs in the combined acceptance after the round, with the default on. A model
  that fails there gets `segment_gemm` switched off.
- Stage C (the gate in the GEMM's epilogue) builds on this kernel.
