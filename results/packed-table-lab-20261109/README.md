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
| packed8 S12 (against real) | 1.060 / 1.060 | **0.882 / 0.881** (672 µs, message 2.9e-6) |
| w′ real | 1,085 µs | 1,121 µs (+3.3 %) |

- **On the H200, w′'s operand bytes cost nothing measurable:** replacing a load with one integer op makes w′ slower
  (zero-table 1.045), so w′ is instruction-issue bound there. This closes, for the H200, the bytes-side w′ ideas of
  report 5: mirror-pair w′ (proposal 2), the 12-byte table for w′ (proposal 5), dout in registers (proposal 4's
  premise). The 5060 Ti is the opposite (zero-table 0.716): it is L1-traffic bound, and its results do not carry
  over.
- **The forward's table matters more off-lattice:** on the disordered snapshot its share grows from 191 to 304 µs
  (the sharp shells of the crystal give L1 reuse). packed8 re-tested there wins 91 µs (about 2.6 % of the step) where it lost 37 µs on the crystal
  ([h200-packed8-disordered/](h200-packed8-disordered/), source 3a6f3e6b, $0.06). Its edge loop is 378 SASS
  instructions against 227 (SHF 58, I2FP 30, IMAD 24, LOP3 20): a cheaper decode is the lever.

MD-step A/Bs on the H200 (Benchmark_loup, abab / abba, 8 blocks of 256 steps): w′ capped at 64 registers with
shared-memory spilling 0.9999 / 1.0033, at 80 1.0005 / 1.0028, NVRTC 13.5.37 against 13.0.88 1.0018 / 1.0040: all
rejected. On the 5060 Ti: spill64 0.9876 / 0.9938, spill80 0.9989 / 0.9992, NVRTC 13.5 1.0225 / 1.0205
([5060ti/compiler-spill](5060ti/compiler-spill/)).

H200 container: Modal `sixth`, source 158399f2, 569 s, $0.96 ([h200-night2/](h200-night2/)).

12-byte table certificate (CPU, `tools/spline_packed_certificate.py` T1: a, b float32, c, d float16 scaled by 2^k per
(pair, weight)): value part 0.025 (passes, target 0.5), slope parts 2.87 (fails: w′ would keep the float32 table).

Off-midpoint economized bases (B1: a0 float32, a1/a2 bfloat16; B2: a1 11-bit, a2 3-bit mantissa): value parts 1.745 and
0.650, both above the 0.5 target, so S12 stays the packed format.

## Round 2 follow-ups: packed8v2, deferred CG, split sums and the FP64 pipe

H200 lab (Modal `sixth`, source 7fcc99d6, 92 s, $0.12; [h200-round2/](h200-round2/)), forward and w′ against real
(flush / warm):

| Variant | crystal | 0.3 Å disordered | Registers |
| --- | ---: | ---: | ---: |
| packed8 (S12) | 1.061 / 1.057 | 0.905 / 0.904 | 80 |
| **packed8v2** (same values, cheaper decode) | **1.004 / 1.001** (625 µs) | **0.858 / 0.856** (636 µs, −107 µs) | 80 |
| deferred CG (76 phi) | 2.752 / 2.746 | 2.350 / 2.348 | 80 (spills) |
| deferred-a + deferred-b (41 / 35 phi, two kernels) | 0.412 + 0.610 = 1.02 | 0.401 + 0.702 = 1.10 | 77, 80 |
| w′ split (u / v sums of 4+ terms in two halves) | 1.032 / 1.033 | 1.031 / 1.029 | 80 |

- **packed8v2** stores the S12 fields biased and placed so that each decodes as
  `as_float(exponent | field) − as_float(exponent | 2^22)`, exact by Sterbenz: x = Q0 << 10 | Q2 >> 6,
  y = (Q2 & 63) << 26 | Q1 << 6 | E6, the third field taken with one funnel shift. About 14 instructions per weight
  against about 19, no integer-to-float conversions; the decoded values equal packed8's bit for bit (CPU check), so
  the certificate (value part 0.161) carries over. It removes packed8's crystal penalty and saves 107 µs where the
  neighbour shells are not sharp.
- **Deferred CG** needs 76 live accumulators on top of the kernel's state; under the forward's 80-register cap it
  spills. Round 1's two-thread split does not spill, but its halves together are slower than the real kernel.
- **w′ split sums**: the FACTORED chains are at most 8 dependent FFMAs (21 terms spread over 3 u and 5–7 v sums).
  Splitting them costs registers and instructions and gains nothing.

Round 2 idea 1 gates ([round2-gates/](round2-gates/), `tools/fma_dual_pipe_bench.py`, 10 s per mix, no throttling):

| FMA results per clock per SM | FP32 only | FP64 only | 1:1 | 2:1 | 3:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB (1,410 MHz) | 63.7 | 31.9 | 62.2 | **93.2** | 82.4 |
| H200 (1,980 MHz) | 126.5 | 63.9 | 124.3 | 122.8 | 122.0 |

Nsight Compute on today's w′ (one launch, running clocks):

| | A100 | H200 |
| --- | ---: | ---: |
| FMA pipe active | **85.8 %** | 52.3 % |
| issue active | 61.2 % | 77.0 % |
| warps active | 36.8 % | 42.6 % |
| stalls per issue: long scoreboard / math throttle / wait / not selected | 2.18 / **1.93** / 1.07 / 2.08 | 2.20 / 0.14 / 0.89 / 2.85 |

The A100's w′ is FP32-pipe bound and its FP64 pipe is idle; the H200's is close to issue-bound.

Third gate, A100 w′ with the two 21-term paths in double (`tools/modal/a100_fp64_wprime.sh`, Modal A100-SXM4-80GB,
sources 67d01628 and 312fdb4d, $0.15; [round2-gates/a100/](round2-gates/a100/)):

| Variant | Registers (spill) | Kernel against base |
| --- | --- | ---: |
| fp64-paths (real conversions; dx 2.5e-7, dr 1.4e-7 of max against base) | 128 | 1.186 |
| F-fp64-free (conversions free: the staged-doubles bound) | 128 | 1.178 |
| fp64-paths, capped at 96 | 96 (88 B) | 1.314 |
| F-fp64-free, capped at 96 | 96 (92 B) | 1.317 |
| F-fp64-free, capped at 80 | 80 (220 B) | 2.059 |

**Rejected.** The pipes would balance, but the doubles take register pairs: 128 registers cut w′'s occupancy, and any
cap spills. Even with free conversions it is 18 % slower, so staging doubles cannot rescue it.

Smallest offloads (source 4cad5f10, Modal A100 80GB PCIe, $0.10): one of the two paths, and/or only its u sums in
double (the v sums stay FP32), so fewer doubles are live at once:

| Variant | Registers (spill) | Kernel against base |
| --- | --- | ---: |
| fp64-1path (one path, u and v) | 128 | 1.078 |
| fp64-u (both paths, u only; dsh bitwise) | 100 | 1.163 |
| fp64-1path-u (one path, u only) | 96 | **1.039** |
| F-fp64-free-1path-u (its conversions free) | 100 | 1.094 |
| fp64-1path-u, capped at 80 | 80 (56 B) | 1.242 |

The smallest offload still needs 96 registers: 80 → 96 drops w′ from 6 to 5 blocks per SM (the kernel is
latency-bound, the occupancy loss outweighs the pipe), and at 80 it spills. Every form of the FP64 pipe on w′ is
slower; it would only pay in a kernel that already runs below its register cap and is FP32-pipe bound, and w′ is
the only FP32-pipe-bound kernel on the A100.

Could a register saving elsewhere pay for the FP64 pipe? The only earlier register-freeing switch that was not slower
on its own is SMEM_SPILL (w′ capped, spills to shared memory; H200 MD step 0.9999 / 1.0033 at cap 64). Lab on the
A100 (sources 68eac1ef and 183e7d3d, $0.20; [round2-gates/a100/](round2-gates/a100/)):

| Variant (cap; spill to) | Registers (local bytes) | Kernel against base |
| --- | --- | ---: |
| base, cap 72 | 72 (24) | 1.095 |
| base, cap 64 | 64 (72) | 1.300 |
| fp64-1path-u, uncapped | 96 (0) | 1.036 |
| fp64-1path-u, cap 80 | 80 (72) | 1.255 |
| fp64-1path-u, cap 64 | 64 (144) | 1.911 |
| fp64-u, cap 80 | 80 (88) | 1.357 |
| fp64-1path, cap 80 | 80 (120) | 1.551 |
| fp64-paths, cap 80 | 80 (160) | 1.925 |
| fp64 free conversions, cap 80 | 80 (160) | 1.986 |

**Closed.** On sm_80 the `enable_smem_spilling` pragma has no effect: the spills go to local memory, and any cap
below what the FP64 forms need costs more than the pipe gives. Every route to the second pipe on w′ is slower.

Found on the way: with `__launch_bounds__` present ptxas ignores `--maxrregcount`. The first run (68eac1ef) kept
96 / 128 registers under "caps" of 80 / 64, and `wprime_backward.SMEM_SPILL` had the same bug, so the H200 / 5060 Ti
SMEM_SPILL MD A/Bs above (source 158399f2) most likely measured uncapped kernels. Fixed in 183e7d3d (the cap is now
`__launch_bounds__`' minimum blocks per SM); SMEM_SPILL on the H200 (sm_90, where the pragma may work) is untested.

## PACKED8: packed8v2 in production, adopted on the H200

`spline_forward.PACKED8` (commits 42637640, ed312e14): with STAGE_EDGES the SPLINE_W forward reads an 8-byte copy of
the spline table (`encode_packed8`, built once per model in `WPrime.prepare_blocks`), decoded as in packed8v2; the w′
kernel keeps the float32 table. Modal H200 (`sixth`), OMAT-0 medium, `tools/modal/packed8_gate_ab.sh`
([h200-packed8-gate-ab/](h200-packed8-gate-ab/)):

| Check | Default | PACKED8 |
| --- | --- | --- |
| MD step SrTiO3 10×10×10, 300 K (abab / abba) | 3.518 / 3.512 ms | b/a **0.9930 / 0.9945** |
| MD step SrTiO3 10×10×10, 1500 K (abab / abba) | 3.736 / 3.749 ms | b/a **0.9818 / 0.9768** |
| Gate forces, max error / ase_f32's (SrTiO3, NaCl, NbCl5) | 0.966, 0.897, 0.971 | 0.903, 0.872, 0.960 |
| Gate stress, same | 0.535, 0.863, 0.954 | 0.415, 0.863, 0.986 |

Both gates pass (source ed312e14). The first gate (source 54b4da74) failed on the default side: the wiring passed the
float table twice to the autograd function, and the aliased inputs made torch.compile recompile with the spline id a
SymInt; ed312e14 passes None instead. The A/Bs ran on 54b4da74 (the packed side is unchanged by the fix). The gain
grows off-lattice, as the kernel lab predicted (crystal 1.004, 0.3 Å snapshot 0.858). On for the H200 family
(decision 66); the A100 and Ada are not measured.

## The FP64 tensor cores (DMMA) as the second pipe

The FP64 FMA units cannot help the H200 (an issue-bound kernel gains nothing from a pipe that does one multiply-add
per lane per issue) and cost the A100 registers. The FP64 tensor cores do 8–64 multiply-adds per lane per issue on
the tensor pipe, with a few doubles per lane. `tools/fma_dual_pipe_bench.py --dmma` (`tools/modal/dmma_pipe_bench.sh`,
sources cee11ca9 and b2b3a8da, Modal `sixth`, about $0.40; [dmma-pipe/](dmma-pipe/)): independent FFMA and
`mma.sync` f64 chains, FMA results per clock per SM, sustained at the full clock:

| Mix (FFMA : DMMA instructions per lane) | A100-SXM4 m8n8k4 | H200 m8n8k4 | H200 m16n8k4 | H200 m16n8k8 | H200 m16n8k16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFMA alone | 63.8 | 126.4 | 126.5 | 126.5 | 126.5 |
| DMMA alone (8 chains) | 56.9 | 63.8 | 126.8 | 127.8 | 127.9 |
| 8 : 1 (1 chain) | **101.0** | 99.8 | 138.5 | 133.2 | 134.9 |
| 16 : 1 | 88.4 | 111.2 | 135.3 | 137.7 | 130.8 |
| 64 : 1 | – | – | 129.0 | 133.8 | **160.8** |

- **A100: the gate passes.** FFMA + DMMA reach 101.0 against 63.8 (1.58x). The tensor cores slow FFMA down a
  little (63.8 → 50.5 in the 8 : 1 mix), but together they give 58 % more multiply-adds, in FP64 for the DMMA share.
- **H200: no second pipe.** The legacy m8n8k4 runs at half rate; the sm_90 shapes reach the full 128 alone, but FFMA
  and DMMA share the throughput: mixes stay at 129–139, and only 64 FFMA per m16n8k16 reaches 160.8 (1.27x), under
  the gate. What the H200 can still gain is issue slots: one m16n8k16 does the work of 64 FFMA issues at the same
  multiply-add rate, and its w′ and forward are near the issue limit.

Next: a kernel-lab form of the per-edge contraction as a DMMA product. Per edge and output degree,
t[c, m3] = sum over (path, m1) of (w[c, path] x[c, m1]) B_path[m1, m3], with B_path[m1, m3] = sum over m2 of
CG[m1, m2, m3] sh[m2], built once per edge and shared by all channels; rows are channels, the accumulators stay in
the tensor-core fragments across the receiver's edges. Its cost is the float-to-double conversions of x and w
(slow on both GPUs) unless x is kept in double per layer and w's spline is evaluated in double.

### Bound: w′ and the SPLINE_W forward as DMMA products (rejected before a lab)

Counted from the generated OMAT-0 medium w′ source (A100 build): x carries 4 components per channel (0e + 1o), the
10 paths write 40 output components, and the paths hold 86 CG terms against 88 entries of their dense (o x i)
blocks, so the factored FFMA form is already as small as a dense product. Every product form pays tile padding:

| Form (rows = channels) | K x N per channel | Dense useful | Executed (tile) | Inflation |
| --- | --- | ---: | ---: | ---: |
| forward / w′ X' = x · M_e (M_e[i, (p, o)] from sh and CG per edge) | 4 x 40 | 88 | 160 | 1.8x |
| w′ dx = (w ∘ dout) · M_e | 40 x 4 (N padded to 8) | 88 | 320 | 3.6x |
| forward per path, edges stacked in K, w ∘ x in A, sums kept in the fragments | (edges · i) x (2 l3 + 1) (N padded to 8) | 88 | 176 | 2.0x |

- DMMA alone runs at the FFMA rate (A100 57–64, H200 128 FMA/clk/SM), and together they reach 1.58x on the A100
  and nothing extra on the H200. So the product forms can win only with less than about 1.6x inflation on the
  A100 and none on the H200; every form inflates 1.8–3.6x.
- The forms without per-edge padding (X') leave the path weight w outside the product, so its 40 double results
  per channel and edge need converting back to float: conversions run at 16 per clock per SM on both GPUs, about
  320 clocks per edge against about 140 for the whole FFMA contraction on the H200.
- The form that keeps the sums in double across edges (no per-edge conversion) needs w and w ∘ x in double: about
  52 DFMA per channel and edge, on the A100 at half the FP32 rate, plus 2x padded DMMA work.

**Rejected (bound).** The contraction is too small per channel (4 input components, paths of 1–7 outputs) for
8-wide tensor-core tiles, and MACE's per-path, per-channel weight keeps the paths from being merged into one
dense product. A second pipe through the FP64 tensor cores would need a contraction with dense dimensions of at
least 8 on both sides of the product; the GEMM-shaped parts of the model already run on TF32 tensor cores
(3xTF32: A100 about 52, H200 about 165 TFLOPS against DMMA's 19.5 and 67).

