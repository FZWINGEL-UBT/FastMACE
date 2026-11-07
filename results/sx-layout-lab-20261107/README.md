# sx-layout lab: Symmetrix-XL's interaction 2 forward layout with spline w (H200, 7 November 2026)

Question (owner, via the conductor): if our interaction 2 forward copies Symmetrix-XL v0.1.2's whole layout and takes
w from a 4-value cubic spline table, does it drop toward their ~950 µs, and what does the same spline do for the w′
backward?

Verdict: **yes, it is faster in both directions.** At 256 knots per pair the forward plus w′ saves about **341 µs per MD
step**, about 6 % of the H200 Benchmark_loup step (5.72 ms). At 2,048 knots it saves about 129 µs (2.2 %). Both pass
the 3x force/stress gate. Only 2,048 knots passes the radial certificate at r ≥ 1 Å; 256 knots fails it there by 74x
on the slope. A production switch is proposed below and **is not built**; it waits for the conductor's go.

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

## Proposal (not built; waits for the conductor's go)

Both clear the brief's 2 % bar by the bound. Switch the H200 generic family's interaction 2 to a spline w, forward and
w′ together:

1. **2,048 knots is the safe choice:** −2.2 % of the step, gate 1.14, certificate passed where MD samples. It needs a
   certificate amendment (r ≥ 1 Å instead of the whole domain) or a short-range fallback to the degree-5 table below
   1 Å.
2. **256 knots is the fast choice:** −6.0 % of the step and gate 1.81, but it fails the certificate by 74x on the
   slope. It would need the owner to accept the gate as the only accuracy test for this table. Alternatively,
   non-uniform knots (dense below about 1.5 Å, 256-knot spacing above) may keep most of the speed and pass the
   certificate; that was not tried.

Before it can become a default, the production build needs:

- the layout as a real kernel (code generation from the angular code, not the lab's regex parse);
- the fit and its certificate in `fastmace.rewrites.radial`;
- the w′ switch;
- the MD A/B on the H200 (12 blocks, both orders);
- the full gate.

The lab numbers are kernel bounds on one structure. They do not yet include the MD loop's other kernels, or other
structures, where the table's L2 footprint changes.

## Cost and provenance

- H200 lab: 2 Modal runs on `fourth`. Try 1 (guard abort): 63 s, about $0.11. Try 2: 73 s, $0.09 estimated, metered
  about $0.02 per run on the budget ledger. All CPU work was local (WSL).
- Source: branch `fastmace/sx-layout-lab`. Lab at `02166191` (main `7bbc759e` merged in). Driver image and packages
  are in `h200/modal.json` and `h200/pip-freeze.txt`; the generated kernels are in `h200/lab/forward.cu` and
  `wprime.cu`.
- Files:
  - `tools/sx_layout_lab.py`: the layout, the spline, the variants and the guard.
  - `tools/sx_spline_error.py`: the certificate score.
  - `tools/sx_spline_gate.py`: the force/stress gate.
  - `tools/modal/sx_layout_lab.sh`: the job.
