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

## Step-level ceiling (OPT9-T02 part 1, owner request, 29 October 2026)

The owner asked what the idea would save at step level. `tools/bench/opt9_ceiling.py` (source `9edef64b`) times
the default float32 MD step (side a) against the same plan with the input gradient and the radial tables'
backward stubbed out (side b: outputs zeroed, wrong forces, timing only), in one process with alternating blocks
(8 × 200 steps after 300 warm-up steps, ABAB). Both sides run at 0.001 fs so side b's missing radial forces cannot
heat it and both keep the starting neighbour list. The first try at the default 0.5 fs heated side b's NaNbO3 to
28,800 K and its step time grew block by block (b/a 0.91 to 1.42); that record is kept in
`results/opt9-T02-ceiling-20261029/first-try-0.5fs/` and not used. The `dw` chunk stores are not stubbed (they
are inside the streaming kernel); OPT9-T01 measured them standalone.

GPUs: leto (NaNbO3) and murphy (NaCl, NbCl5), both RTX 4000 Ada, `mace_cueq_oeq`; idle checks passed at
23:33 UTC. Every run is valid (no compile while timed, no other GPU process).

| Cell (RTX 4000 Ada) | default ms | stubbed ms | b/a (block range) | ceiling |
| --- | ---: | ---: | --- | ---: |
| NaCl | 2.659 | 2.461 | 0.926 (0.925–0.927) | 0.20 ms, 7.4 % |
| NbCl5 | 20.70 | 19.97 | 0.965 (0.963–0.967) | 0.72 ms, 3.5 % |
| NaNbO3 | 17.00 | 15.81 | 0.930 (0.929–0.933) | 1.19 ms, 7.0 % |

The ceiling clears the plan's T02 bar (≥ 3 % NaNbO3, ≥ 1.5 % NbCl5), but on this GPU the work a fused kernel
must add is larger than the ceiling: T01's `wdot` alone is 2.46 ms (NaNbO3) and 1.45 ms (NbCl5) standalone,
against at most 1.19 + 0.40 ms (ceiling plus the `dw` stores) and 0.72 + 0.26 ms. **On the RTX 4000 Ada the fused
backward would be slower than today's.** The verdict of T01 stands.

**A100 (owner request, Modal A100-SXM4-40GB, job `ap-37ElDNqiDn5khCxqRwXazl`, source `93bde431`, 707 s,
about $0.49):** the same tool and protocol.

| Cell (A100) | default ms | stubbed ms | b/a (block range) | ceiling |
| --- | ---: | ---: | --- | ---: |
| NaCl | 2.274 | 2.135 | 0.939 (0.936–0.939) | 0.14 ms, 6.1 % |
| NbCl5 | 12.56 | 11.39 | 0.907 (0.905–0.908) | 1.17 ms, 9.3 % |
| NaNbO3 | 11.49 | 9.51 | **0.828** (0.826–0.828) | 1.98 ms, 17.2 % |

On the A100 the removed work is much less hidden than on the Ada (17 % against 7 % of the NaNbO3 step), and
T01's added `wdot` is small there (0.91 / 0.49 ms, on the A100-SXM4-80GB). Estimate for a fused kernel (ceiling
+ `dw` stores − `wdot`; inference across two A100 variants, not a measurement): NaNbO3 1.98 + 0.25 − 0.91 ≈ 1.3 ms
(≈ 11 % of the step), NbCl5 1.17 + 0.16 − 0.49 ≈ 0.8 ms (≈ 7 %). That is large, but only on a GPU class whose L2
bandwidth absorbs the receiver-order `G'` gathers; on the RTX 4000 Ada it is a loss. A default would need a
general hardware rule (for example from a measured L2 gather rate), which is an owner decision.

## Not done

- The prototype in the vendored streaming backward (only worth building for an A100-class rule).
- The local RTX 5060 Ti (32 MiB L2, smaller than NaNbO3's 29 MiB `G'` plus the working set); its record is added
  when the queued run finishes.
- Real operands (real r distribution: MD edges reach fewer pieces than uniform groups, which would shrink the
  working set). That would help the Ada case only if the reached pieces' `G'` fit well inside the L2 and the
  receiver-order gathers became L1-friendly, which the sorted/unsorted gap does not suggest.
