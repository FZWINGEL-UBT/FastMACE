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
- Each row's b is a with one change, so b/a above 1 means the change is slower. JOINED_LAYOUT on gains 3 % on small
  (medium: 2 %). Turning SPLINE_W or COMPOSE_SKIP off costs 13–19 %, and turning the tail fold off costs 1.3–1.7 %, so
  both stay on for small. CUEQ_PROJECTED is a tie on small.
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

## The label `omat0-small` and its H200 defaults (`979785a7`, `a229d577`)

OMAT-0 small is now in the label table as `omat0-small` (checkpoint `small-omat-0`, head `omat_pbe`, ASL, PBE). It is
recognised by its SHA-256, as medium is, and listed in `labels.EXTRA_LABELS`: it is in the table but not one of the 19
acceptance labels of SPEC §12.5. Decision 66 gives it medium's pair **on the H200 only**:

- `DECISION_66_TF32X3_MODEL_GPUS["omat0-small"]` holds the 3×TF32 GEMMs.
- The new `DECISION_66_GENERIC_WPRIME_LABEL_GPUS["omat0-small"]` holds the generic w′ backward, with the fused forward
  and the chunked carrier.
- On every other GPU the pair stays off for small until that GPU has its own A/B (the A100: Alex, packed).
- The readout-tail fold stays on, as for every model on the H200. Turning it off makes small slower (1.0131 / 1.0173,
  the `notail` row above), so it is not excluded per model. `979785a7` briefly excluded it on a misread of that row,
  and `a229d577` restored it.

Gate (`gate/`, `tools/modal/omat0_small_gate.sh`, the plan's defaults, no constants): the 3× force/stress gate
passes on all three structures. Each figure is the ratio to the float32 calculator's error, against a limit of 3.

| | SrTiO3 3×3×3 | NaCl | NbCl5 |
| --- | ---: | ---: | ---: |
| Force | 0.97 | 0.99 | 1.01 |
| Stress | 0.50 | 1.20 | 1.22 |
| Energy | 5.30 (1.4 meV, under the 10 meV floor) | 0.37 | 1.22 |

With these defaults and no constants, the step breakdown gives Benchmark_loup **2.29 ms per step**, with the force
graph at 2.20 ms per replay. That matches the A/B's 2.29 ms.

The first gate run (`gate-tailoff/`, the tail fold off) also passed, with a worst force ratio of 1.00 and stress of
1.39. Its breakdown was 2.34 ms per step (kernel time 2.07 against 2.03 ms per force call with the fold on).

Modal `sixth`: $0.76 for the fixed A/B, then $0.62 and $0.38 for the two gate runs.

## Screen: medium's tuned H200 switches on small (`screen/`, `ec3a0c15`)

Owner: "do you think we could improve omat 0 small specifically?" Every H200 kernel switch was chosen on OMAT-0
medium. This screen runs 16 single-change A/Bs on small, Benchmark_loup, both build orders
(`tools/modal/omat0_small_screen.sh`).
- a is small's defaults, about 2.30 ms per step.
- b is a with one change, so b/a below 1 means the change is faster on small.
- All 32 entries are valid. Modal `sixth` cost $1.40.

**Faster on small in both orders** (candidates for small's own values):

| Change (b) | ab | ba | On medium |
| --- | ---: | ---: | --- |
| `compose_skip.JOINED_LAYOUT` on | 0.9668 | 0.9610 | 0.9805 / 0.9815, being adopted for the H200 (fastmace/joined-layout) |
| w′ `STAGE_EDGES` on | 0.9857 | 0.9825 | 0.9969 / 1.0010, a tie (off) |
| spline forward `R128` off, 128 threads, `MIN_BLOCKS` 8 | 0.9947 | 0.9893 | 1.0515 / 1.0507 with R128's shape (off) |
| spline forward `R128` off, 128 threads, `MIN_BLOCKS` 7 | 0.9951 | 0.9964 | 1.0097 / 1.0103 (off) |
| w′ `FOLD_ALPHA` off | 0.9940 | 0.9942 | on (0.9963 / 0.9954) |
| `tf32x3.TILE` 64x32 | 0.9913 | 0.9971 | 1.0014 / 1.0035 (off) |

**Ties:** `tf32x3.TILE` 32x64 (0.9982 / 0.9985) and `POWER_BACKWARD` off (0.9923 / 0.9997).

**Slower on small, so the current defaults stay:**

| Change (b) | ab | ba |
| --- | ---: | ---: |
| `R128` off | 1.0200 | 1.0214 |
| spline forward `STAGE_EDGES` off | 1.0181 | 1.0263 |
| `PREFETCH_AHEAD` on | 1.0137 | 1.0241 |
| `SPLINE_VECTOR` 2 | 1.0318 | 1.0376 |
| w′ `STAGE_SH` off | 1.0041 | 1.0156 |
| `STAGE_DOUT` off | 1.1743 | 1.1808 |
| `COUNT_PLAN` off | 1.0234 | 1.0302 |
| `COMPOSE_FAST` off | 1.0518 | 1.0524 |

So small's kernels are close to tuned already. Beyond JOINED_LAYOUT (3–4 %), four small-specific settings each gain
0.5–1.7 %, and they have not been combined yet. Each is a per-GPU module constant today. Giving small its own values
needs a per-model default in the kernels, and a combined A/B and gate first.

## Combined: the screen's four wins together (`combined/`, `3bc94302`)

Owner, 9 November 2026: "Yes, combine them" (`tools/modal/omat0_small_combined.sh`, Benchmark_loup, both orders).
- a is small's defaults with JOINED_LAYOUT on, about 2.23 ms per step.
- "all" is a with four changes: w′ `STAGE_EDGES` on, the spline forward at 128 threads with `MIN_BLOCKS` 8 instead
  of R128, w′ `FOLD_ALPHA` off, and tf32x3 `TILE` 64x32.
- Each "no…" row is all four minus one.
- Modal `sixth` cost $0.89.

| b | ab | ba |
| --- | ---: | ---: |
| all four | 0.9829 | 0.9850 |
| without w′ STAGE_EDGES | 0.9913 | 0.9900 |
| without MIN_BLOCKS 8 | 0.9893 | 0.9842 |
| without FOLD_ALPHA off | 0.9802 | 0.9841 |
| without TILE 64x32 | 0.9833 | 0.9829 |

- **Together the four gain about 1.6 %** (2.23 → 2.19 ms), less than their separate gains added up.
- **Most of it is w′ `STAGE_EDGES`:** dropping it costs about 0.7 % in both orders.
- **`MIN_BLOCKS` 8 adds about 0.5 % in order ab** and nothing in order ba.
- **`FOLD_ALPHA` off and `TILE` 64x32 add nothing on top.** The "without" rows are as fast as "all" or faster.
- So the setting worth having is w′ STAGE_EDGES on (with MIN_BLOCKS 8 as a marginal extra), worth about 1–1.5 % on
  top of JOINED_LAYOUT.
- Gate with all four and JOINED_LAYOUT on: passes. Ratios against the 3× limit:

  | | SrTiO3 | NaCl | NbCl5 |
  | --- | ---: | ---: | ---: |
  | Force | 0.87 | 0.95 | 1.00 |
  | Stress | 0.49 | 1.32 | 1.23 |

- Step breakdown with all of them: 2.18 ms per step, force graph 2.10 ms.

From small's defaults today (2.29 ms), JOINED_LAYOUT plus w′ STAGE_EDGES make about 4.5 % (to about 2.19 ms).

**Open:**
- JOINED_LAYOUT reaches small through the H200 adoption on fastmace/joined-layout.
- w′ STAGE_EDGES stays off for medium (a tie there). Small now has it by default on the H200 (next section).

## w′ STAGE_EDGES as small's own H200 default (`stage-edges/`, `387ec7bb`, job `c90d9292`)

Owner, 9 November 2026: "Yes, add it".
- `wprime_backward.STAGE_EDGES_MODEL_GPUS` is a new per-model GPU list, `{"omat0-small": NVIDIA H200}`.
  `stage_edges_default(device, model_label)` turns STAGE_EDGES on for `STAGE_EDGES_GPUS` (every model; the A100
  family since main's round-6 adoption) plus the label's own GPUs. The `STAGE_EDGES` constant still overrides both.
- The rewrites get the label from `Context.inputs["model_label"]` (`forces._table_label`). A path load finds the
  label by SHA-256 and head, as the planner's `label_of` does. The generic w′ (`attach_generic_wprime`) passes it to
  `WPrimeBackward`.
- On the job's H200, `stage_edges_default` returned True for small and False for medium.

Confirmation (`tools/modal/omat0_small_stage_edges.sh`, Benchmark_loup, JOINED_LAYOUT not on): a = small's new
defaults, b = `STAGE_EDGES=False`. Modal `sixth` cost $0.37.

| b | ab | ba |
| --- | ---: | ---: |
| w′ STAGE_EDGES off | 1.0072 (2.277 → 2.294 ms) | 1.0114 (2.272 → 2.298 ms) |

- The default is 0.7–1.1 % faster, matching the combined screen.
- Step breakdown with the defaults: 2.27 ms per step (2.29 before), force graph 2.19 ms.
- STAGE_EDGES only changes how the kernel stages edge data (results are bitwise unchanged), so no new force/stress
  gate was run. The combined gate above had it on and passed.

## Rescreen after main c8d08b5b (`rescreen/`, job `ff0bd317`)

Main c8d08b5b (merged as `a6eddf4a`) turned JOINED_LAYOUT, `BLOCK_ORDER` and `SORT_BY_LENGTH` on for the H200. With
w′ STAGE_EDGES, small's defaults run Benchmark_loup at **2.14 ms per step** (breakdown 2.13 ms). a = those defaults,
b = a plus one change. Modal `sixth` cost $0.83.

| b | ab | ba |
| --- | ---: | ---: |
| BLOCK_ORDER off | 1.0148 | 1.0157 |
| SORT_BY_LENGTH off | 1.0092 | 1.0102 |
| JOINED_LAYOUT off | 1.0307 | 1.0315 |
| PREFETCH_AHEAD on | 1.0218 | 1.0254 |
| R128 shape with MIN_BLOCKS 8 | **0.9873** | **0.9883** |
| w′ STAGE_SH off | 1.0193 | 1.0213 |
| w′ FOLD_ALPHA off | 0.9939 | 0.9948 |
| tf32x3 TILE 64x32 | 0.9948 | 0.9980 |

- Every H200 default that small now runs helps it.
- Three changes were faster in both orders: MIN_BLOCKS 8, FOLD_ALPHA off and TILE 64x32.

## Combo2 and defaults2 (`combo2/`, `b4c238ac`; `defaults2/`, `e2206bb6`)

Combo2: a = small's defaults (as in the rescreen). Modal cost $0.68.

| b | ab | ba |
| --- | ---: | ---: |
| MIN_BLOCKS 8 | 0.9923 | 0.9952 |
| MIN_BLOCKS 8, FOLD_ALPHA off | 0.9884 | 0.9908 |
| MIN_BLOCKS 8, TILE 64x32 | 0.9904 | 0.9935 |
| all three | 0.9895 | 0.9916 |
| FOLD_ALPHA off, TILE 64x32 | 0.9985 | 0.9959 |

The gate with all three passes, and the breakdown gives 2.11 ms.

Defaults2: a = small with MIN_BLOCKS 8 and FOLD_ALPHA off as its own defaults (`e2206bb6`); b sets one back to the
shared value. Modal cost $0.92.

| b | ab | ba |
| --- | ---: | ---: |
| MIN_BLOCKS 6 again | 1.0046 | 1.0057 |
| FOLD_ALPHA on again | 0.9936 | 0.9927 |
| both again | 0.9949 | 1.0047 |
| w′ ESTRIN on | 0.9940 | 0.9933 |

The gate passes; force ratios are 0.94 / 0.91 / 1.01 and stress ratios 0.58 / 1.10 / 1.32 (SrTiO3 / NaCl / NbCl5).
The breakdown gives 2.12 ms.

- **MIN_BLOCKS 8 is small's H200 default** (`4b80f1ff`: `spline_forward.R128_MIN_BLOCKS_MODEL_GPUS`, bitwise).
  It is 0.5–1.2 % faster in all three jobs. Medium keeps 6, since 8 is 5 % slower there.
- **FOLD_ALPHA stays on.** Off was 0.4–0.6 % faster in two jobs, but on again was 0.7 % faster in the third. That is
  within the run-to-run spread, so it is not adopted. `wprime_backward.FOLD_ALPHA_MODEL_OFF_GPUS` stays as an empty
  table.
- ESTRIN was measured only with FOLD_ALPHA off. On medium it adds nothing on top of FOLD_ALPHA, so it was not tried
  further.
- The spread between jobs is about ±0.5 %. Changes under that size need a repeat before adoption.

## The A100 family: small's decision 66 pair (`a100/`, Alex)

Small's A100 defaults left the pair off, because its per-model lists held the H200 only. NHR@FAU Alex,
A100-SXM4-80GB, venv of `fm-a100omat` (Python 3.10.12), float32, Benchmark_loup.

**Pack job 4967329** (source `7790b57c`, `tools/jobs/alex_ab_pack.sh`): a = small's A100 defaults; b = the per-model
lists with the A100 added. `tools/ab_rewrites.py` now takes a dict literal for a `MappingProxyType` constant.

| b | ab | ba |
| --- | ---: | ---: |
| the pair | **0.3911** (10.72 → 4.19 ms) | **0.3916** (10.72 → 4.20 ms) |
| 3×TF32 GEMMs alone | 0.8240 | 0.8243 |
| generic w′ alone | 0.5586 | 0.5585 |

**Adopted** in `5797d298`: both lists name the H200 and the A100 family.

**Gate job 4967421** (source `5797d298`, the plan's defaults, no constants): the 3× gate passes.

| | SrTiO3 | NaCl | NbCl5 |
| --- | ---: | ---: | ---: |
| Force | 1.00 | 0.92 | 1.03 |
| Stress | 0.79 | 1.31 | 1.33 |

- Energy errors are under the 0.01 eV floor.
- Step breakdown: **4.19 ms per step**, force graph 4.05 ms.
- Small's other H200-only defaults (w′ STAGE_EDGES, MIN_BLOCKS 8) need A100 A/Bs of their own. w′ STAGE_EDGES is
  already on for every model on the A100.

## Not done

- MIN_BLOCKS 8 on the A100, and small on the Ada and 5060 Ti, are not measured.
