# SPLINE_W forward table reads and w′ locality: the 9 November lab

Lane: sx-layout-lab (branch `fastmace/bspline-lab`). Model OMAT-0 medium, float32, SrTiO3 10x10x10 (Benchmark_loup,
5,000 atoms, about 374,000 edges), H200 switch set (SPLINE_W, R128, STAGE_EDGES, BLOCK_ORDER, SORT_BY_LENGTH; w′
SPLINE_VECTOR 4, STAGE_DOUT, STAGE_SH, SCALAR_PATHS, FOLD_ALPHA). Tools: `tools/spline_table_bound.py` (alternating
kernel timings in one process, flush and warm L2), `tools/spline_read_lab.py` (the variants), and
`tools/spline_packed_certificate.py` (CPU).

## Results

| Variant | GPU | Forward / w′ against real | What it says |
| --- | --- | --- | --- |
| one-slab (every edge reads the same 20 KB slab) | H200 | forward 0.693 (623 → 432 µs) | the table's whole share of the forward is about 191 µs on this edge order |
| `bspline` (cubic B-spline on the 851-interval C2 grid, four scalar loads per weight) | 5060 Ti | forward 0.930 / 0.925 | 11 % of the real − one-slab gap; the L1 model predicted 61 %. **Rejected**: under half the predicted share, so no H200 run |
| `packed8` stand-in decode (8 bytes per weight) | 5060 Ti | 64 % of the gap | the read path pays on the 5060 Ti |
| `packed8` S12 (exact decode: midpoint basis economized to a quadratic, 6-bit shared exponent, 22/20/16-bit fixed point; certificate value part 0.161, target 0.5) | H200 | forward **1.060 / 1.060** (660 µs), message 2.7e-5 of max | **Rejected**: slower on the H200. The decode (SHF/LOP3/IMAD: 848 against 696 instructions in the kernel) costs more than halving the table bytes saves |
| w′ `local-sender` (every sender = its receiver) | H200 | 0.997 / 0.996 | sender-row locality is worth about 4 µs on the H200 (0.960 on the 5060 Ti) |
| w′ `local-dout` (every receiver = receiver 0) | H200 | 0.970 / 0.967 | the receivers' output-gradient rows: about 35 µs |
| w′ `one-slab+local-sender+local-dout` (all three ideal) | H200 | **0.962 / 0.962** | **every locality idea on w′ is bounded at about 42 µs (1.2 % of the 3.42 ms step)**, under the 68 µs lab bar |

H200 run: Modal profile `sixth`, NVIDIA H200 (driver 610.57.04), source 21adcc39, 96 s wall, $0.42
([h200/](h200/)).

## What this closes

- The table side of the forward: per-knot rows (B-spline) and compressed coefficients both lose on the target GPU.
  The H200's forward is not limited by the table bytes once the decode costs instructions; see the issue estimate
  below.
- The w′ gather side: the four 9 November reports' sender bins, L2 residency windows / persistence, TMA or
  cp.async operand pipelines, receiver clusters and dout staging all attack loads whose ideal removal is worth 42 µs
  at most (sender 4 µs, dout 35 µs, table 25 µs, not additive). w′'s 1,089 µs is not a locality problem; with
  every operand in L1 it still takes 1,047 µs.

## The forward is close to instruction-issue bound

SASS of the production forward (sm_90, NVRTC 13.0): the edge loop is 227 instructions per edge per thread (114 FFMA,
62 FMUL, 14 LDG, 8 LDS). 374,000 edges × 128 channels × 227 / 32 lanes ≈ 336 M warp instructions; at one instruction
per scheduler per cycle (132 SMs × 4 × about 1.98 GHz) that is about 320 µs, against the one-slab floor of 432 µs.
So the remaining forward cost is mostly issue, and fewer instructions per edge is the lever left. The `ghoist`
falsifier (G-hoisting from the third report: the channel-independent CG sums g_oi = Σ_j c_oij e_j per edge from
shared memory, acc_o += (w x_i) g_oi) cuts the loop to 192 instructions (FMUL 62 → 22, LDS 8 → 26); its timing is the
next lab.
