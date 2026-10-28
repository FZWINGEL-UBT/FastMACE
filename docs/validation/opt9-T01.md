# OPT9-T01: the radial backward as a dot with tabulated `w'(r)` (28–29 October 2026)

Plan: `plans/opt-9.md`. Probe: `tools/bench/opt9_wprime_dot.py` (commit `ee06eb44`). Records:
`results/opt9-T01-wprime-20261029/`.

**Verdict: the stop rule is not met on the RTX 4000 Ada**, so OPT9 stops here. On the A100 the added work is a
fifth of the removed work; on the Ada it is three quarters. Optimisations must be general, and a receiver-owned
kernel cannot have the sorted order that makes the dot cheap, so no fused kernel is built (OPT9-T02 not started).

## Setup

- **Operands:** synthetic, with MH-1's float32 shapes (3×3×3): NaNbO3 84,072 edges and 586 (pair, piece) groups,
  NbCl5 52,416 edges and 213 groups; degree-10 tables (10 derivative terms), `Q = 1280`, `[h c, c]` width 65,
  1,536-edge chunks. Edges in receiver order with uniformly random groups.
- **Timing:** CUDA graphs of 20 calls, L2 flushed before each call (flush timed alone and subtracted), 15
  alternating blocks, medians. `wdot`'s tile is the fastest of four on a quick pass.
- **GPUs:**
  - Modal A100-SXM4-80GB (sm_80, 40 MiB L2), driver 610.57, image torch 2.14.1+cu130; job `ap-DkKr07IwLuTAQJMaLHr1NG`,
    62 s, about $0.05. Not a research GPU.
  - murphy, RTX 4000 Ada (sm_89, 40 MiB L2), `mace_cueq_oeq` (torch 2.14.1+cu130), source in
    `/mnt/homes/zwingel/opt9-20261029`; GPU idle checked before the run.
  - The local RTX 5060 Ti run is queued behind other sessions' screens; not in this report.

## Results (ms per call, standalone)

| GPU | Cell | removed: input grad + table bwd | removed: `dw` stores | **added: `wdot`** | `wdot` / removed | `wdot_sorted` | `wdot_load` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | NaNbO3 | 4.19 | 0.25 | **0.91** | **0.21** | 0.51 | 1.09 |
| A100 | NbCl5 | 2.68 | 0.16 | **0.49** | **0.17** | 0.33 | 0.64 |
| RTX 4000 Ada | NaNbO3 | 2.82 | 0.40 | **2.46** | **0.76** | 0.46 | 2.97 |
| RTX 4000 Ada | NbCl5 | 1.80 | 0.26 | **1.45** | **0.71** | 0.31 | 1.61 |

Stop rule: `wdot` ≤ 0.5 × removed on NaNbO3 and NbCl5. **Met on the A100, not met on the Ada.**

Spreads: A100 `chain` 24 % between blocks (the other variants ≤ 23 %), Ada ≤ 6 %. Neither changes a verdict.

**Exactness:** `wdot` against a float64 reference on 256 sampled rows: 3.6e-8 (NaNbO3) and 2.5e-8 (NbCl5)
relative to the row's absolute dot. Rounding level, as expected for the reassociated series.

## Why the Ada loses

- In receiver order each edge reads its group's 50 KB of `G'` (10 × 1280 floats). That is 4.2 GB of L2 reads per
  call on NaNbO3, ten times the `dw` bytes it saves. The A100's L2 bandwidth absorbs it; the RTX 4000 Ada's does
  not (2.46 against 0.46 ms when the same edges are sorted by group).
- Sorting by group is not available to the fused kernel: the streaming convolution owns edges by receiver. An
  unfused sorted kernel would need `dw` materialised in full (430 MB on NaNbO3), which is what
  `chunked_conv_backward` avoids; `wdot_load` (receiver order, `dw` read) is slower still.
- The removed work is also mostly hidden today: the fp32 profile's F0 ceiling for removing the input gradient
  is 2.6 % of the NaNbO3 step and 0.5 % of NbCl5. Even on the A100, the step-level gain would be bounded by that
  ceiling plus the table backward and the stores.

## Not done

- OPT9-T02 (step-level ceiling, prototype in the vendored streaming backward).
- The local RTX 5060 Ti (32 MiB L2, smaller than NaNbO3's 29 MiB `G'` plus the working set); its record is added
  when the queued run finishes.
- Real operands (real r distribution: MD edges reach fewer pieces than uniform groups, which would shrink the
  working set). That would help the Ada case only if the reached pieces' `G'` fit well inside the L2 and the
  receiver-order gathers became L1-friendly, which the sorted/unsorted gap does not suggest.
