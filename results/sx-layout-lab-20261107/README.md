# sx-layout lab: Symmetrix-XL's interaction 2 forward layout with spline w (H200, 7 November 2026)

Question (owner, via the conductor): if our interaction 2 forward copies Symmetrix-XL v0.1.2's whole layout and takes
w from a 4-value cubic spline table, does it drop toward their ~950 µs, and what does the same spline do for the w′
backward?

Verdict: **yes, it is faster in both directions.** At 256 knots per pair the forward plus w′ saves about **341 µs per MD
step**, about 6 % of the H200 Benchmark_loup step (5.72 ms). At 2,048 knots it saves about 129 µs (2.2 %). Both pass
the 3x force/stress gate. Only 2,048 knots passes the radial certificate at r ≥ 1 Å; 256 knots fails it there by 74x
on the slope.

The follow-up places the knots densely at short r (grid `a695`, below). It saves **310 µs (5.4 %)**, passes the
certificate over the whole domain and passes the gate at 1.37. A production switch is proposed below and **is not
built**; it waits for the conductor's go.

## Method

- **Layout** (`tools/sx_layout_lab.py`, `sx_layout_forward`): Symmetrix-XL's thread layout. One thread per
  (receiver, channel), 128 channels, 256-thread blocks, the receiver's CSR edges walked serially, all four message
  outputs accumulated in registers and stored once. Edges with r ≥ r_max or an invalid sender are skipped. The
  per-edge angular code is not hand-written. It is parsed out of production's H200 forward (`WS_FORWARD`, both path
  parts, with production's per-path W factors `cpf`), then re-indexed per thread. So the variants compute exactly
  production's message.
- **w per edge**, one mode per variant:
  - `sx-spline{256,2048,4096}`: a cubic Hermite spline of w = (h c) W over [R_LO, r_max), uniform knots per active
    species pair. It is fit in float64 from the radial chain's exact values and slopes, stored in float32 as
    [pair][interval][4][128·10] (channel-contiguous, as Symmetrix-XL), and evaluated as c0 + t(c1 + t(c2 + t c3)).
  - `sx-table6`: our certified degree-5 table (bins, pieces, 6 terms) in the same layout.
  - `sx-wbuf`: w read from a precomputed per-edge buffer (built in float64, so no TF32). This isolates the layout.
  - `sx-F-w1`: w = 1. This is the floor: layout and angular work only.
- **w′ backward** (`B-spline{n}`): production's `TABLE_W` w′ kernel with the piece lookup replaced by the spline
  interval and the 6-term table by the 4-term spline. Same α folding, same everything else.
- **Kernel lab** (`tools/omat0_kernel_lab.py`): MACE-OMAT-0 medium, Benchmark_loup SrTiO3 10×10×10 (300
  equilibration steps, then one captured step), 15 round-robin rounds, L2 flushed. Each variant's outputs are checked
  against the base. **Guard**: the job compares every forward variant's message with production in process and
  aborts before timing if one is off (`check_before_timing`). The first try aborted there after 63 s: the parser did
  not know `WS_FORWARD` (see `h200-try1/`). The parser was fixed and the job retried once, which passed.
- **Accuracy**, both on the CPU:
  - `tools/sx_spline_error.py` scores each spline the way `radial.certify` scores our table: the error over the
    column's largest exact value, divided by the float32 network's own error, for values and slopes, over the whole
    domain and over r ≥ 1 Å. It passes at ≤ 1.
  - `tools/sx_spline_gate.py` runs the force/stress gate (3 rattled and strained configurations per cell mode, seed
    20261015) with interaction 2's weights replaced by the float32 spline. It passes at ratio ≤ 3 (`selfcheck.RATIO`).

## Spline error against our certified table (`spline_error.json`)

Certificate score: the error over the float32 network's error. It passes at ≤ 1.

| w table | value, all r | value, r ≥ 1 Å | slope, all r | slope, r ≥ 1 Å | size (6 pairs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| our table (degree 5, certified) | 0.046 | 0.045 | 0.110 | 0.068 | 171 MB (9 pairs) |
| spline 256 | 31.3 | 1.03 | 2025 | **74.1** | 31 MB |
| spline 512 | 2.0 | 0.067 | 264 | 9.3 | 63 MB |
| spline 1024 | 0.13 | 0.027 | 34.9 | 1.18 | 126 MB |
| spline 2048 | 0.013 | 0.021 | 4.35 | **0.174** | 252 MB |
| spline 4096 | 0.015 | 0.021 | 0.58 | 0.091 | 503 MB |

**256 knots is too coarse for the certificate.** Its slope is 74x the network's error at r ≥ 1 Å and 2,025x over the
whole domain. The smallest spline that passes at r ≥ 1 Å is 2,048 knots (0.17). The whole domain needs 4,096 knots
(0.58), because the r < 1 Å region (never sampled in MD) has the steepest curvature.

## Force/stress gate (`spline_gate.json`)

Ratio to the float32 yardstick; it passes at ≤ 3.

| knots | NaCl E / F / σ | NbCl5 E / F / σ | SrTiO3 E / F / σ | worst | verdict |
| --- | --- | --- | --- | ---: | --- |
| 32 | 901 / 416 / 361 | 577 / 138 / 204 | 319 / 641 / 401 | 901 | FAIL |
| 256 | 1.46 / 1.52 / 1.25 | 1.47 / 1.03 / 1.20 | 0.74 / 1.81 / 0.67 | **1.81** | pass |
| 2048 | 0.87 / 0.97 / 0.86 | 1.03 / 0.94 / 1.00 | 0.86 / 1.05 / 1.14 | **1.14** | pass |
| 4096 | 1.46 / 0.96 / 1.10 | 1.43 / 0.94 / 1.34 | 0.76 / 1.05 / 0.69 | 1.46 | pass |

For comparison, Symmetrix-XL itself measured 2.19 / 2.86 on our NaCl gate. The gate's configurations are rattled
crystals (0.05 Å) with no pair closer than about 1.8 Å. So the 256-knot spline's failure lies where the gate does not
look; the certificate is the stricter test.

## H200 timings (`h200/`, Modal `fourth`, commit `02166191`, 7 November 2026 19:38 CET)

| kernel | variant | µs | ratio vs production | registers | message / gradient vs production |
| --- | --- | ---: | ---: | ---: | --- |
| forward | production (`WS_FORWARD`) | 1270.3 | 1.000 | 154 (+64 B local) | — |
| forward | sx-wbuf (layout, w from a buffer) | 1131.0 | 0.890 | 80 | 1.8e-6 |
| forward | sx-table6 (layout + our 6-term table) | 1345.1 | 1.059 | 99 | 1.8e-6 |
| forward | **sx-spline256** | **1105.8** | **0.871** | 80 | 2.3e-6 |
| forward | **sx-spline2048** | **1243.0** | **0.979** | 80 | 1.8e-6 |
| forward | sx-spline4096 | 1463.3 | 1.152 | 80 | 2.4e-6 |
| forward | sx-F-w1 (w = 1, floor) | 501.9 | 0.395 | 80 | — |
| w′ | production (`TABLE_W`) | 1403.1 | 1.000 | 72 | — |
| w′ | **B-spline256** | **1226.6** | **0.874** | 72 | dr 1.3e-5 |
| w′ | **B-spline2048** | **1301.7** | **0.928** | 72 | dr 1.6e-6 |
| w′ | B-spline4096 | 1497.3 | 1.067 | 72 | dr 1.7e-6 |

The spline's w agrees with a·W to 1.4e-7 (relative) at the captured edges, for every knot count.

Reading:

- **The layout helps by itself.** `sx-wbuf` beats production by 11 % although it reads w from a buffer (1.9 GB). The
  one-thread-per-channel layout needs 80 registers and no local memory, against production's 154 plus 64 B of spill.
- **The w source decides the rest.** `sx-F-w1` shows that layout plus angular work cost only 502 µs, so w costs
  600–960 µs, depending on how many bytes per edge it pulls through L2. The 256-knot table touches 7.7 MB at this
  structure and fits in L2; the 2,048-knot table touches 54 MB, the 4,096-knot table 100 MB.
- **Our 6-term table loses in this layout** (1.059): 6 coefficients instead of 4, plus the bin and piece lookups.
- Symmetrix-XL's ~950 µs is not reached. The closest is 1,106 µs at 256 knots. The 502 µs floor says that what
  separates the variants is w traffic, not the layout.

## Backward and net per MD step

The w′ backward reads the same w table as the forward, so its spline variant gains for the same reason (fewer
coefficients per edge, no piece lookup).

| knots | forward Δ µs | w′ Δ µs | net Δ µs | net % of the 5.72 ms step | gate worst | certificate (r ≥ 1 Å) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | −164.5 | −176.5 | **−341** | **−6.0 %** | 1.81 | FAIL (slope 74x) |
| 2048 | −27.3 | −101.4 | **−129** | **−2.2 %** | 1.14 | pass (0.17); whole domain 4.35 |
| 4096 | +193.0 | +94.2 | +287 | +5.0 % | 1.46 | pass (0.09) |

Not counted:

- A possible further gain: with the spline, interaction 2 no longer needs `radial_table_forward`'s [h c] columns,
  only the density column. That could save up to about 90 µs; it was not measured.
- The production path must also keep the spline's extra memory: 31 MB at 256 knots, 252 MB at 2,048.

## Follow-up: denser knots at short r (owner, 7 November 2026)

The uniform spline's error depends strongly on r. `tools/sx_spline_knots.py --diagnose` (`knots_diagnose.json`) maps
the certificate score by band at 256 uniform knots. Slope error:

- r < 0.5 Å: 2,025x;
- 1.0–1.25 Å: 74x;
- 1.5–1.75 Å: 5.9x;
- 2.0–2.5 Å: 1.07x;
- r ≥ 2.5 Å: 0.5x or less.

Hermite slope error scales as h³. So each band gets its own uniform spacing, chosen for a score of about 0.5. The
kernels find the segment with one compare per boundary. An edge belongs to one warp for all its channels, so the
branch is uniform within the warp. Few MD edges fall in the dense region. As a result, the rows the step actually
touches barely grow: 393 rows (8.0 MB), against 374 rows (7.7 MB) at 256 uniform knots.

| grid (`GRIDS` in `tools/sx_layout_lab.py`) | intervals | table, 6 pairs | certificate slope, all r / r ≥ 1 Å | gate worst |
| --- | ---: | ---: | --- | ---: |
| `a695`: 0.3–0.5 Å ×150, –0.75 ×110, –1.0 ×85, –1.5 ×110, –2.0 ×50, –2.5 ×30, –6.0 ×160 | 695 | 85 MB | **0.54 / 0.63, passes** | **1.37** |
| `p399`: 0.3–1.0 Å ×32, –1.5 ×110, –2.5 ×100, –6.0 ×157 | 399 | 49 MB | 1,957 / 0.63 (r ≥ 1 Å only) | 1.62 |

Gate per structure, energy / force / stress (`knots_gate.json`):

- `a695`: NaCl 1.37 / 1.32 / 1.14, NbCl5 1.18 / 0.96 / 0.89, SrTiO3 0.99 / 1.07 / 1.14.
- `p399`: NaCl 1.28 / 1.62 / 0.87, NbCl5 1.12 / 0.95 / 0.77, SrTiO3 0.94 / 1.05 / 1.41.

H200 run: `h200-knots/`, commit `1e71246a`, Modal `fourth`, 7 November 2026 19:50 CET, 51 s, $0.06 estimated. It is
the same capture and 15 rounds as above, with production re-timed in the same process.

| knots | forward µs (ratio) | w′ µs (ratio) | net Δ µs | net % of step | certificate | gate |
| --- | --- | --- | ---: | ---: | --- | ---: |
| production | 1279.6 (1.000) | 1410.0 (1.000) | — | — | certified table | — |
| uniform 256 | 1104.0 (0.863) | 1241.0 (0.880) | −345 | −6.0 % | fails (74x at r ≥ 1 Å) | 1.81 |
| uniform 2048 | 1249.7 (0.977) | 1312.8 (0.931) | −127 | −2.2 % | r ≥ 1 Å only | 1.14 |
| **a695** | **1120.0 (0.875)** | **1259.8 (0.894)** | **−310** | **−5.4 %** | **passes, whole domain** | **1.37** |
| p399 | 1100.5 (0.860) | 1257.7 (0.892) | −331 | −5.8 % | r ≥ 1 Å only | 1.62 |

The uniform rows reproduce the first run within 1 %. Every variant passed the in-process message check. The spline's
w matches a·W to 1.3–1.4e-7, and the guards' message and gradient errors are 1.9e-6 or less (w′ dr up to 3.2e-6).

The dense grid keeps 90 % of the 256-knot gain and passes both accuracy tests as they stand. The w′ backward gains
about 18 µs less than with uniform 256 knots, probably because the lookup branches per edge. The forward gains about
16 µs less, from the slightly larger table and the branch.

## Proposal (not built; waits for the conductor's go)

All candidates clear the brief's 2 % bar by the bound. **Recommended: `a695`.** Switch the H200 generic family's
interaction 2 to a spline w on piecewise-uniform knots, forward and w′ together. It is −5.4 % of the step, passes the
certificate unchanged over the whole domain, and passes the gate at 1.37.

Alternatives:

- `p399` gains 0.4 % more, but only by amending the certificate to r ≥ 1 Å.
- Uniform 2,048 knots gains less and also needs that amendment.
- Uniform 256 knots fails the certificate.

In production, the grid would be fitted per model by the certificate rule (per band, the coarsest spacing that passes
with margin), not taken from this table. The `a695` breakpoints are specific to OMAT-0 interaction 2, and AGENTS.md
forbids per-model tables.

Before it can become a default, the production build needs:

- the layout as a real kernel (code generation from the angular code, not the lab's regex parse);
- the fit and its certificate in `fastmace.rewrites.radial`;
- the w′ switch;
- the MD A/B on the H200 (12 blocks, both orders);
- the full gate.

The lab numbers are kernel bounds on one structure. They do not yet include the MD loop's other kernels, or other
structures, where the table's L2 footprint changes.

## Cost and provenance

- H200 lab: 3 Modal runs on `fourth`. Try 1 (guard abort): 63 s, about $0.11. Try 2: 73 s, $0.09 estimated. Knots
  follow-up: 51 s, $0.06 estimated. All CPU work was local (WSL).
- Knots follow-up files:
  - `tools/sx_spline_knots.py`: the map by band and the grid scores;
  - `tools/sx_spline_gate.py --grids`: the gate on the grids;
  - `tools/modal/sx_knots_lab.sh`: the job;
  - results: `knots_diagnose.json`, `knots_grids.json`, `knots_gate.json`, `h200-knots/`.
- Source: branch `fastmace/sx-layout-lab`. Lab at `02166191` (main `7bbc759e` merged in). Driver image and packages
  are in `h200/modal.json` and `h200/pip-freeze.txt`; the generated kernels are in `h200/lab/forward.cu` and
  `wprime.cu`.
- Files:
  - `tools/sx_layout_lab.py`: the layout, the spline, the variants and the guard.
  - `tools/sx_spline_error.py`: the certificate score.
  - `tools/sx_spline_gate.py`: the force/stress gate.
  - `tools/modal/sx_layout_lab.sh`: the job.
