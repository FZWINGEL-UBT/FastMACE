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

## What this leaves for w′

The two classic levers are already closed:

- **More warps (register caps):** `omat0-wprime-maxreg-a100`, ideas-r14 #36, and the 5060 Ti lab of `omat0-horner`. Caps of 64–72 spill and lose 1.11–1.48×.
- **Prefetching the next edge:** STAGE_EDGES on w′ was 0.9969 / 1.0010 on the H200, and PREFETCH_AHEAD on the forward was 1.0046 / 1.0067.

What remains is to issue fewer dependent loads per edge. Each edge requests 1,406 L1 sectors (525.5 M in total), the same with real and one-slab inputs, against a table slab of 640 sectors.

The candidate is a w′ that reads the sender rows and staged adjoints once per cohort into shared memory, as STAGE_SH already does for the edge features, so the per-edge loop reads only the table. That is a lab question for the 5060 Ti first. This profile alone does not bound it, because the other loads' L1 hits (81 % with one slab) are already high.

Caveat: this is the 5060 Ti. The H200 has a larger L2 and higher L2 bandwidth, and its w′ runs 72 registers in the H200 build. The stall mix should be confirmed there before anyone builds a lever on it. An H200 ncu run is a few minutes on Modal `sixth`.
