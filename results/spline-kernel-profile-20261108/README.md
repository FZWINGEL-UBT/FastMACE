# SPLINE_W kernels under Nsight Compute: what limits w′ (8 November 2026)

This profile follows up the open measurement named in the [Codex screen assessment](../omat0-breakthroughs-assessment-20261108/README.md#second-report-codex-breakthrough-screen-after-the-364-ms-h200-step): what limits w′, given that its table share on the H200 is under 3 %.

Run details:

- **Machine:** local RTX 5060 Ti (sm_120, driver 610.88), booked through `tools/ci/local_gpu.py run`.
- **Software:** Nsight Compute 2026 from CUDA 13.3 (`--set full`), env `mace_cueq_oeq`.
- **Model and structure:** OMAT-0 medium (`d4b14be9…`); SrTiO3 10×10×10, 373,760 edges, after 200 MD steps.
- **Switches** forced to the H200's settings: `SPLINE_W`, `R128`, w′ `SPLINE_VECTOR=4`, `SORT_BY_LENGTH`, `BLOCK_ORDER`.
- **Source:** branch fastmace/breakthroughs-codex. Driver [tools/spline_kernel_profile.py](../../tools/spline_kernel_profile.py).
- **Method:** one launch each of w′ and the forward, on the real inputs and on the one-slab inputs (every edge reads one table slab; see `tools/spline_table_bound.py`).
- **Files:** [metrics.csv](5060ti/metrics.csv) and [details.csv](5060ti/details.csv) (ncu's raw and details pages), and [run.log](5060ti/run.log).

Times are ncu replay times with caches flushed, so they read slower than the timed lab.

| Metric | w′ real | w′ one-slab | forward real | forward one-slab |
| --- | ---: | ---: | ---: | ---: |
| Duration (ms) | 5.54 | 4.77 (0.861) | 3.56 | 2.13 (0.598) |
| Registers / warps active (of peak) | 80 / 49 % | 80 / 49 % | 80 / 49 % | 80 / 49 % |
| Issue active | 44.9 % | 52.0 % | 34.1 % | 57.5 % |
| LSU pipe | 46.9 % | 54.2 % | 30.3 % | 51.4 % |
| FMA pipe | 32.9 % | 38.2 % | 26.1 % | 45.0 % |
| L1 hit rate | 47.5 % | 81.1 % | 24.6 % | 90.3 % |
| L2 read hit rate | 97.3 % | 91.6 % | 97.4 % | 87.9 % |
| L2 read sectors (M) | 268 | 79 | 218 | 25 |
| Stall cycles per issue: long scoreboard | **9.31** | 7.11 | **13.99** | 5.79 |
| … not selected / wait / dispatch | 1.17 / 0.69 / 0.42 | 1.36 / 0.70 / 0.47 | 0.45 / 1.46 / 0.17 | 1.06 / 1.51 / 0.30 |
| … lg throttle / short scoreboard / math throttle | 0.29 / 0.11 / 0.07 | 0.46 / 0.12 / 0.09 | 0.02 / 0.16 / 0.03 | 0.16 / 0.16 / 0.06 |
| Local memory (sectors) | 0 | 0 | 80 k | 80 k |

## What limits w′

**w′ waits on load latency at an occupancy cap of 50 %.** Its 80 registers allow 6 blocks of 128 threads per SM, so about half the warp slots are filled. Between two issues an average warp spends 9.3 cycles on the long scoreboard, which means waiting for a global load. No other stall reason exceeds 1.2.

What w′ is *not* limited by:

- **Bytes.** The loads hit L2 97 % of the time, and the 268 M sectors read from L2 average about 1.5 TB/s over the call. That is latency, not bandwidth.
- **Arithmetic.** The FMA pipe is busy 33 % of the time, and math-throttle stalls are 0.07.
- **Atomics.** lg throttle is 0.29. This agrees with the ledger's no-atomics variant being slower.
- **Spills.** w′ has no local memory.

**The table is only part of the latency.** One-slab raises the L1 hit rate from 47 % to 81 %, yet long-scoreboard stalls only drop from 9.3 to 7.1. About three quarters of w′'s load waiting comes from its other loads (the edge's sender features and the staged adjoint rows). It does not come from the spline table. On the H200 the table share is smaller still (one-slab 0.973), so its w′ should be even more dominated by these other loads.

**The forward is the opposite case.** Its L1 hit rate is 24.6 %, and long-scoreboard stalls of 14.0 drop to 5.8 with one slab. Its latency is the table, which SORT_BY_LENGTH and BLOCK_ORDER already target.

## Correction: the same profile with the H200's switch set

The profile above ran w′ with the 5060 Ti's own defaults. Those leave `STAGE_DOUT` and `STAGE_SH` off and apply the
80-register cap, `BUTTERFLY` and `REDUX`; on the H200 the first two are on and the rest off. The run was repeated
with every differing switch set as on the H200 (`STAGE_DOUT`, `STAGE_SH`, `SCALAR_PATHS`, `FOLD_ALPHA` on;
`BUTTERFLY`, `REDUX` off; no register cap; the forward's `STAGE_EDGES` on).

Files: [5060ti-h200set/](5060ti-h200set/metrics.csv).

| Metric | w′ real | w′ one-slab | forward real | forward one-slab |
| --- | ---: | ---: | ---: | ---: |
| Duration (ms) | 4.73 | 3.87 (0.818) | 3.69 | 1.38 (0.374) |
| Registers / warps active | 96 / 33 % | 96 / 33 % | 80 / 50 % | 80 / 49 % |
| Issue active / LSU pipe / FMA pipe | 47 / 64 / 37 % | 57 / 77 / 45 % | 25 / 20 / 21 % | 73 / 58 / 61 % |
| L1 hit / L2 read hit | 10.7 / 96.7 % | 66.1 / 91.8 % | 11.2 / 98.7 % | 87.9 / 91.4 % |
| L1 global load sectors per edge | 733 | 733 | 700 | 700 |
| Long-scoreboard stall per issue | **4.07** | 2.16 | **19.2** | 1.29 |

With staging on, w′ is closer to LSU-issue bound: its LSU pipe is 64 % busy, and its load waits halve. The table
slab is 640 of its 733 sectors per edge, so the table accounts for about half of what is left. The other half is
mostly the sender rows, the one large global load left unstaged.

## Sender-row bound: stopped

`tools/spline_table_bound.py` has a new w′ variant, `local-sender`, which sets every edge's sender to its own
receiver. A receiver's edges then read one x row, which stays hot in L1. This is the most that staging or reordering
the sender rows could save. The output is wrong, and dx's atomics then land on one row per receiver, so the variant
flatters slightly.

Run: 21 alternating rounds, same switch set. Files: [stb-sender-1755/](stb-sender-1755/spline_table_bound.json) and
[stb-sender.log](stb-sender.log).

| w′ variant | Flushed / warm (µs) | × real |
| --- | ---: | ---: |
| real | 4,595 / 4,559 | 1.000 |
| local-sender | 4,409 / 4,377 | 0.960 |
| one-slab | 3,740 / 3,721 | 0.816 |
| one-slab + local-sender | 3,697 / 3,682 | 0.808 |

**The stop rule applies.** Perfect sender locality saves 4 % of w′. On the H200 that is at most about 44 µs of
1,099, or 1.3 % of a 3.42 ms step. The triage rule needs a lab gain of twice the 1 % bound, about 68 µs, and real
staging would also pay for a cooperative load pass. Stopped before any build or Modal run.

## What this leaves for w′

The two classic levers are already closed:

- **More warps (register caps):** `omat0-wprime-maxreg-a100`, ideas-r14 #36, and the 5060 Ti lab of `omat0-horner`. Caps of 64–72 spill and lose 1.11–1.48×.
- **Prefetching the next edge:** STAGE_EDGES on w′ was 0.9969 / 1.0010 on the H200, and PREFETCH_AHEAD on the forward was 1.0046 / 1.0067.

The H200 already stages the adjoint rows (`STAGE_DOUT`) and the edge features (`STAGE_SH`). The sender rows were
bounded above at 4 % of w′ and stopped. With the H200's staging, w′ sits at 64 % LSU issue and 37 % FMA, with the
table read half of its remaining load waiting. Its table share on the H200 itself is 3 % (one-slab 0.973).

No load-side lever for w′ is left above 1 % of the step. A further w′ gain needs fewer instructions per edge, not
fewer bytes.

Caveat: this is the 5060 Ti. The H200 has a larger L2 and higher L2 bandwidth, and its w′ runs 72 registers in the H200 build. The stall mix should be confirmed there before anyone builds a lever on it. An H200 ncu run is a few minutes on Modal `sixth`.
