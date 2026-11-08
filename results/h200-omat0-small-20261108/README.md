# OMAT-0 small on the H200 with OMAT-0 medium's settings (8 November 2026)

Owner's request: Benchmark_loup with OMAT-0 small (`mace-omat-0-small.model`, mace-foundations release
`mace_omat_0`, SHA-256 `0abfde07862cf1e9…`, 67.6 MB) on the H200 "with the same settings as medium". Modal H200
(profile `sixth`), branch `fastmace/h200-round6` (main `1bd4f3be` plus round-6 switches, all off). SrTiO3 10×10×10
(5,000 atoms), 300 K, 1 fs, skin 0.5.

OMAT-0 small has no FastMACE label. So medium's two label-keyed defaults, 3×TF32 GEMMs and the generic w′ backward,
were forced on by adding `None` to `DECISION_66_TF32X3_LABELS` and `DECISION_66_GENERIC_WPRIME_LABELS`. Every other
H200 default is per GPU and applies as is.

## Result (`fix/`, source `be0140c6`)

**About 2.29 ms per MD step** with medium's settings, in both build orders (0.46 µs per atom and step; medium about
3.65 ms). Without the label-keyed pair, small runs at 5.41 ms.

| A/B (`fix/run.log`) | order ab: a / b, b/a | order ba: b / a, b/a |
|---|---|---|
| plain: a = medium's settings, b = without the pair | 2.294 / 5.414 ms, 2.360 | 5.408 / 2.282 ms, 2.370 |
| joined: b = a + JOINED_LAYOUT | 0.9676 | 0.9708 |
| nospline: b = a − SPLINE_W | 1.1932 | 1.1948 |
| nocompose: b = a − COMPOSE_SKIP | 1.1372 | 1.1347 |
| nocueq: b = a − CUEQ_PROJECTED | 1.0011 | 0.9994 |
| notail: b = a − GENERIC_FOLD_READOUT_TAIL | 1.0131 | 1.0173 |

- Every entry is valid (no recompiles, no other GPU process). Both sides run 108 neighbour-list rebuilds in the
  1,536 timed steps and reach the same temperature (292.2 K).
- The step breakdown agrees. With medium's settings (`diag/breakdown`): 2.28 ms wall per step, of which 2.20 ms is
  the force graph. Without the pair (`fix/breakdown-nolabels`): 5.46 ms wall and 5.34 ms force graph.
- Without the pair, the plan turns off `wprime_backward`, `fused_folded_forward`, `chunked_conv_backward` and
  `tf32x3_gemms` (label `None`). The conv backward is then the streaming one (0.98 ms per call), and the radial
  networks run as GEMMs (0.73 ms plus 0.67 ms for the folded last layer's input gradient). That makes 5.27 ms of
  kernel time per force call, against 2.03 ms with the pair.
- JOINED_LAYOUT gains 3 % on small (medium: 2 %). The spline and composed skip are each worth 13–19 %. CUEQ_PROJECTED
  is a tie on small, and the tail fold costs 1.3–1.7 %. Small's defaults are not adjusted here.
- In-use record (`fix/check.json`, `tools/spline_w_check.py --model omat0-small`):
  - The spline is in use: 723 intervals, certificate 0.48, forward 70 registers, w′ 64 registers, no local memory.
  - The composed skip is in use. JOINED_LAYOUT is off by default.
  - Forces against the same build, rattled by 0.05 Å, differ by at most 3.1e-6 eV/Å at a maximum force of
    7.1 eV/Å. The run-to-run figure is 3.2e-6.

## Cause of the 5.4 / 2.2 ms bimodality: a `tools/ab_rewrites.py` bug (fixed)

The earlier runs (`pack/`, `diag/`) timed most builds at 5.40–5.46 ms. Four b builds of the packed order-ab job ran
at 2.2–3.1 ms, and `tools/step_breakdown.py` gave 2.3 ms. The settings were identical, but the plans were not:

- `ab_rewrites.py` set a side's `--a/--b-constant` values around `ForceField()` and `planner.build()`, but **not
  around `planner.plan()`**. The label-keyed pair is read in `FixedPlanner._stage2`, which runs inside `plan()`.
- `ConstantsSide` sets a side's constants before each of its steps and leaves them set. So a side's plan read
  whatever the **last timed step** had left behind, not its own value.
- That explains all 14 earlier results:
  - In the order-ab pack, a was built first, before any step, so its plan had the defaults (pair off): 5.4 ms.
  - joined was built before any step: 5.4 ms.
  - nolabel was planned after joined's last b step, which had the pair on: 2.2 ms.
  - nospline was planned after nolabel's step, pair off: 5.4 ms.
  - nocompose, nocueq and notail were each planned after a step with the pair on: 3.1, 2.6 and 2.3 ms.
  - Order ba and the one-pair diag jobs built every side before any step, so every side ran at 5.4 ms.
  - `step_breakdown.py` sets its constants process-wide before the plan, so it ran at 2.3 ms.
- Padded edge capacity, neighbour-list buffers, CUDA-graph capture and host time per step are not involved. Rebuild
  counts and trajectories were identical on both speeds, and the plan alone differs.
- Fix (`be0140c6`): each side is built under every constant either side sets, at its own value (the default where
  it leaves one alone). This covers the force field, the plan and the engine (`built_under`), and
  `tests/cpu/test_ab_rewrites_tool.py` pins it.
- Bound: on main, no `ab_rewrites.py` result in `results/` ever set a `fastmace.planner.*` constant. Every screened
  constant there is a kernel or rewrite constant read at build or launch time, which the old code did cover. Only
  this job is affected.
- The earlier readings were wrong for that reason. The "label-keyed pair makes no difference, 0.9999 / 0.9998"
  result (`diag/run.log`) compared two plans that both had the pair off. The order-ba pack ratios (0.9975–1.0010)
  compared each switch on the pair-off plan.

## Not done

- OMAT-0 small has no FastMACE label, so a plain run (`fastmace` without the forced label tuples) gets the 5.4 ms
  plan on the H200. Giving small the medium defaults by label or SHA-256 is a planner decision (decision 66) and is
  left open.
