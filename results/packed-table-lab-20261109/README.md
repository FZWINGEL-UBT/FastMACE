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
shared memory, acc_o += (w x_i) g_oi) cuts the loop to 192 instructions (FMUL 62 → 22, LDS 8 → 26), but on the
5060 Ti it is 1.07 of real and, with every edge on one slab (the instruction part alone), **1.43× one-slab** (1,810
against 1,261 µs): the shared-memory g reads and their fill cost more than the FMULs they remove. Stopped before the
H200. The deferred-CG form (report 5 proposal 3, Symmetrix Φ) reaches the same FP count with 76 accumulators instead.

## Report 5 follow-ups: zero-byte operands, a disordered snapshot, spill and compiler A/Bs

`--wprime-zero`: w′ with one operand's loads replaced by one integer op each (the arithmetic kept), to see what the
operand's L1 / shared wavefronts cost rather than its misses (report 5's critique of the one-slab test). `--rattle
0.3`: SrTiO3 with 0.3 Å Gaussian position noise and 20 MD steps, so the neighbour shells are not sharp.

| w′ variant (against real) | 5060 Ti | H200 crystal | H200 disordered |
| --- | ---: | ---: | ---: |
| one-slab | 0.812 / 0.816 | 0.976 / 0.977 | 0.960 / 0.962 |
| zero-table | 0.716 / 0.719 | **1.043 / 1.047** | 1.032 / 1.030 |
| zero-dout | 0.876 / 0.877 | 1.013 / 1.016 | 1.012 / 1.012 |
| zero-x (sender row) | 0.929 / 0.931 | 1.055 / 1.054 | 1.072 / 1.070 |
| zero-all | 0.582 / 0.585 | 1.026 / 1.030 | 1.014 / 1.018 |

| Forward (H200) | crystal | disordered |
| --- | ---: | ---: |
| real | 622 µs | 743 µs (+19.5 %) |
| one-slab | 431 µs | 439 µs |
| table share (real − one-slab) | 191 µs | **304 µs** |
| w′ real | 1,085 µs | 1,121 µs (+3.3 %) |

- **On the H200, w′'s operand bytes cost nothing measurable:** replacing a load with one integer op makes w′ slower
  (zero-table 1.045), so w′ is instruction-issue bound there. This closes, for the H200, the bytes-side w′ ideas of
  report 5: mirror-pair w′ (proposal 2), the 12-byte table for w′ (proposal 5), dout in registers (proposal 4's
  premise). The 5060 Ti is the opposite (zero-table 0.716): it is L1-traffic bound, and its results do not carry
  over.
- **The forward's table matters more off-lattice:** on the disordered snapshot its share grows from 191 to 304 µs
  (the sharp shells of the crystal give L1 reuse). Smaller forward tables (packed8, 12-byte) are re-tested there.

MD-step A/Bs on the H200 (Benchmark_loup, abab / abba, 8 blocks of 256 steps): w′ capped at 64 registers with
shared-memory spilling 0.9999 / 1.0033, at 80 1.0005 / 1.0028, NVRTC 13.5.37 against 13.0.88 1.0018 / 1.0040: all
rejected. On the 5060 Ti: spill64 0.9876 / 0.9938, spill80 0.9989 / 0.9992, NVRTC 13.5 1.0225 / 1.0205
([5060ti/compiler-spill](5060ti/compiler-spill/)).

H200 container: Modal `sixth`, source 158399f2, 569 s, $0.96 ([h200-night2/](h200-night2/)).

12-byte table certificate (CPU, `tools/spline_packed_certificate.py` T1: a, b float32, c, d float16 scaled by 2^k per
(pair, weight)): value part 0.025 (passes, target 0.5), slope parts 2.87 (fails: w′ would keep the float32 table).
