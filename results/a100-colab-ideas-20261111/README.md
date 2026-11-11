# Five OMAT-0 medium speed ideas on the A100, screened on a Colab A100 (11 November 2026)

Branch `fastmace/a100-colab-ideas` (from `d7dd7e23`). Model OMAT-0 medium (`omat0-medium`), float32, hybrid backend,
Benchmark_loup SrTiO3 10x10x10 (5,000 atoms, 373,760 padded edges) unless stated. GPU runs: Google Colab
**NVIDIA A100-SXM4-40GB** (sm_80, driver 580.82.07, 1410 MHz max SM clock, 400 W), torch 2.14.1+cu130, through
`tools/colab/fastmace_colab.py` (records under `screen1/`, `batch2/`). CPU runs: WSL `~/environments/fastmace_cpu`
(MACE float64/float32 on the CPU) and `mace_cueq_oeq` (NVRTC census, no GPU).

| # | Idea | First test | Result | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Far-shell edges (r > r_h) in packed `half2` | CPU oracle: far-edge inputs, messages, dout and per-receiver sums in float16 | force/stress error of the float32 run 0.9–1.8× the float32 rounding at r_h 5.0–5.8 Å (gate 3×); 5.0 Å with half accumulation 2.96× (SrTiO3) | **Bound, census and A100 lab pass** (lab 0.48 of fp32 per far edge) |
| 1 | | HFMA2 beside FFMA on sm_80 (`tools/fma_dual_pipe_bench.py --hfma2`) | FFMA alone 63.8, HFMA2 alone 127.3, 1:1 mix 167.7 FMA results/clk/SM (55.9 FFMA + 55.9 HFMA2 instructions) | **HFMA2 is a second pipe** on the A100 |
| 2 | Two half-systems out of phase | two independent half-size force graphs (SrTiO3 10x10x5) on two streams | concurrent 6,326 µs, back to back 6,894 µs, the full system's graph 6,273 µs: concurrent / full **1.008** | **Rejected** (even the uncoupled upper bound gains nothing) |
| 3 | Machine search over kernel formulations | exhaustive contraction-order search of w′'s trilinear sums with the real CG sparsity | from the unfactored form (602 FP instructions per edge and channel) FACTORED is 251, rank 2 of 27; the best tree 241.7 (−3.7 %), the best per-path mix 238.25 (−5.1 % of the TP arithmetic, about 1 % of the step) | **Closed** (below the bound; the same M-precontraction dropped on 5 Nov) |
| 4 | GEMM cross terms in FP16, big·big in TF32 | `tools/fp16x3_gemm_bench.py` hybrid rows | error 6.3e-7 to 1.4e-6 against the 2.7e-7 stop rule; time 0.74–1.22 of CUTLASS 3xTF32 (two cuBLAS GEMMs, split unfused) | **Stopped** (accuracy) |
| 5 | `forward-ghoist` re-screen | `tools/spline_table_bound.py --table-reads ghoist` on today's A100 default (PACKED8, PREFETCH_AHEAD) | forward 0.912 of real (1,256.4 → 1,145.9 µs); every edge on one slab 0.863 | Built as `GHOIST`, MD step 0.977–0.989, gate passes; **superseded by `GEOM`** (main, same regrouping) |

## Idea 1: far-shell edges in half precision

`half_far_oracle.py` (CPU, float64 reference) rounds, for interaction 2's edges longer than r_h, the inputs
x[sender], Y and w (cutoff folded, as the tables hold it), the per-edge message and, in the backward, dout and the
input gradients to float16 (`io`), and in `acc` also sums the far messages per receiver in a float16 accumulator
(edge by edge, as one thread's `__hfma2` accumulator would), merged once per receiver. `_s` variants scale each w
row by a power of two first (the cutoff makes 8–45 % of far w values float16-subnormal; the scaling changes
nothing measurable). Score: max error of the float32 run with float16 far edges against float64, over the float32
run's own error.

| Structure (atoms, edges) | r_h | half edges | io F / S | acc F / S |
| --- | --- | --- | --- | --- |
| SrTiO3 4x4x4, σ 0.05 Å (320, 23,628) | 5.0 Å | 32.3 % | 0.93 / 1.24 | 2.96 / 1.42 |
| | 5.5 Å | 26.4 % | 1.02 / 1.31 | 1.47 / 1.57 |
| | 5.8 Å | 12.8 % | 0.89 / 1.16 | 0.90 / 1.08 |
| NbCl5 2x (576, 15,572) | 5.0 Å | 39.5 % | 1.20 / 1.40 | 1.72 / 1.19 |
| | 5.5 Å | 23.8 % | 1.09 / 1.12 | 1.08 / 1.25 |
| | 5.8 Å | 12.7 % | 0.98 / 0.94 | 1.10 / 0.87 |

Not modelled: the float16 rounding inside one edge's tensor product (the paths and CG sums run on rounded inputs
in the run's dtype) and the Horner evaluation of the table in half precision. So these are lower bounds on a
kernel's error; r_h 5.5–5.8 Å leaves 2–3× of room.

HFMA2 (`hfma2/fma_hfma2.json`): on sm_80 packed `fma.rn.f16x2` issues at FFMA's instruction rate (63.7 per clk
per SM) and does not take FFMA's pipe: a 1:1 mix issues 111.8 instructions per clk per SM (167.7 multiply-adds),
close to the four-warp-instructions-per-clock issue limit. So far edges in half2 would both halve the far edges'
FMA instructions and move them off the FP32 pipe that bounds w′ on the A100 (85 % FMA pipe, step-limiters
20261109). Size: w′ is 2,265 µs of a 6.8 ms force call; its tensor-product arithmetic for 13–26 % of the edges is
the part that moves, so about 2–5 % of the step. The layout that fits w′'s one-thread-per-channel walk is two far
edges of one receiver per `half2` (g broadcast, x from the two senders, per-edge u, v and weight gradient in the
two halves).

### Stage 2: offline census (`half2_wprime_census.py`, NVRTC, no GPU)

Today's w′ channel loop (the captured A100 kernel, FACTORED, HORNER, SCALAR_PATHS) in a minimal harness, once
unchanged (`fp32`) and once with every channel value an `__half2` of two far edges of one receiver (`half2`: x of the
two senders packed from a float16 copy, dout broadcast in a staged `__half2` row, the table from a float16 copy, two
8-byte loads and four PRMTs per path, the slope masked per half, dx by two float atomics). Loop instructions per
edge (`half2_wprime_census.json`):

| | fp32 | half2 (per edge) | ratio | registers | spills |
| --- | --- | --- | --- | --- | --- |
| sm_80 | 460 (310 FFMA, 39 FMUL) | 291.5 (317 HFMA2, 32 HMUL2, 44 PRMT per pair) | 0.634 | 80 / 104 | none |
| sm_80, `__launch_bounds__(256, 3)` | 460 | 316.5 | 0.688 | 80 / 80 | none |
| sm_90 | 462 | 293.5 | 0.635 | 80 / 106 | none |

### Stage 3: A100 lab (`half2_wprime_lab.py`, Colab A100-SXM4-40GB, records `half2/`)

The three census kernels on far-edge-like data (5,000 senders, the real table geometry 731 × 1,280 × 4, every edge
interior and sloped), 100 interleaved launches each; numbers of half2 against fp32 on the same float16-rounded inputs:

| far edges | fp32 | half2 | half2, 80 registers | half2 / fp32 | numbers (max\|d\|/max\|a\|: edge grad, radial grad, dx) |
| --- | --- | --- | --- | --- | --- |
| 131,072 | 1,139.7 µs | 601.1 µs | 551.9 µs | 0.527 / **0.484** | 1.5e-3, 2.5e-3, 7.0e-4 |
| 65,536 | 608.3 µs | 341.5 µs | 301.6 µs | 0.561 / **0.496** | 1.4e-3, 1.2e-3, 8.0e-4 |

The falsifier does not fire: half2 does a far edge in about half the time (less than the 0.63–0.69 of the issue
slots: the float16 x and table halve the far edges' load bytes too), and its numbers differ from fp32 by float16
rounding only. Size, if w′'s time scales with edges: far edges at r_h 5.5–5.8 Å are 13–26 % of interaction 2's
edges, so 0.5 × (13–26 %) × 2,265 µs ≈ 150–300 µs, **2–4.5 % of the force call** (stage-3 rule: twice the 1 % bound,
passes). Not yet covered: the half-precision Horner in the force/stress gate (the oracle did not model it), the
pairing of far edges per receiver (an odd edge per receiver runs in fp32), and the float16 copies of x and the
table (32 MB). Next stage: a production w′ far-edge path behind a `*_GPUS` switch, then the MD-step A/B and the gate.
Colab cost 0.74 CU (699 s, 11 min set-up).

## Idea 2: two half-systems on two streams

`tools/bench/half_system_streams.py` builds three engines with the default plan (the full 10x10x10 and two 10x10x5
copies), captures their force graphs and replays them in 21 alternating rounds of 20 replays: full 6,272.8 µs,
A then B 6,894.4 µs, A and B on two streams 6,325.7 µs. Splitting costs 10 % back to back (half-size kernels fill
the A100 worse); running the halves concurrently wins that back but not more. The real split couples the halves
at every layer, so it can only overlap less. Rejected.

## Idea 3: formulation search (bounded)

`formulation_search.py` enumerates, per path of interaction 2 (the 10 paths, 86 CG terms, e3nn `wigner_3j`),
every contraction tree for (T, U, V) = (weight gradient, dx sums, dsh sums): U by terms, by an edge-only
M_io = Σ_j C y_j, or through N_ij = Σ_o C g_o; V by terms, through N, or through K_oj = Σ_i C x_i; T as Σ x U,
Σ y V or terms. Cost: channel-dependent FP instructions per edge and channel (edge-only products / 128), live
values as a register proxy. The search rediscovers FACTORED from the unfactored start (602 → 251, rank 2 of 27);
the best tree uses M for U (241.7, live 10), the best per-path mix 238.25: at most 5.1 % of the w′
tensor-product arithmetic, about 1 % of the step. The forward's best is the G-hoist form (idea 5). Closed.
The space is a hand-written enumeration, not an e-graph; a SASS census was not needed below the bound.

## Idea 4: TF32 big·big plus one FP16 cross GEMM

`tools/fp16x3_gemm_bench.py` (new `hyb_*` columns): A rows and B columns scaled into [0.5, 1), Ah, Bh in float16,
Al, Bl the remainders times 2^11; big·big = Ah Bh in TF32 (exact products), cross = [Ah, Al'] @ [Bl'; Bh] in one
FP16 GEMM over 2K, both cuBLAS. Error relative to |A||B|: 6.3e-7 to 1.4e-6 (3xTF32 1.1–2.7e-7); the split alone in
exact arithmetic is 2.7–5.5e-8 (CPU), so the error is the GPU GEMMs' accumulation. Time of the two GEMMs 0.74–1.22
of 3xTF32 (worse on the K = 128 shapes). Stopped at the rule.

## Idea 5: forward-ghoist on the A100, and GHOIST

Kernel lab (`screen1/ghoist`, `tools/spline_read_lab.py` now transforms the captured kernel's own source, so
PACKED8 and PREFETCH_AHEAD stay on): real 1,256.4 µs, ghoist 1,145.9 µs (0.912), one-slab 0.989, one-slab + ghoist
0.863. On the 5060 Ti it was 1.07 (9 Nov); the A100's forward is L2- and latency-bound with PACKED8, and the
regrouping removes 40 FMULs per edge and channel.

Built as `fastmace.kernels.oeq_streaming.spline_forward.GHOIST` (None / True / False; `GHOIST_GPUS` empty):
`ghoist_source` rewrites a STAGE_EDGES source so each output's sum is Σ_i (w x_i) g_oi, with the 76 g_oi = Σ_j c e_j
per edge formed by warp 1 while warp 0 locates the tile (no extra barrier; the harmonics' shared copy is gone).
Rounding (regrouped sums). Census (`tools/bench/cu_census.py`, NVRTC, offline): sm_80 edge loop 356 → 321 SASS
instructions (lab 322), 80 registers, no spills, shared 2.7 → 10.4 KB per block; sm_90 354 → 331. Tests:
`tests/cpu/test_spline_w.py` (regrouping counts, NVRTC compile for sm_80/sm_90). Local 5060 Ti smoke (SPLINE_W,
R128, STAGE_EDGES forced; SrTiO3 6x6x6): runs, b/a 1.003 (not a claim).

MD-step A/B and gate: see batch 2 below.

### Batch 2: GHOIST against today's A100 defaults (Colab A100-SXM4-40GB, source `cd7575bb`)

Records in `batch2/` (`ab-srtio3/`, `ab-nbcl5/`, `gate/`, `ghoist-prod/`). Side a: the GPU's default plan; side b:
`spline_forward:GHOIST=True`. `tools/jobs/ab_pack.sh` (warm-up 200, 6 blocks of 256 steps per side, 1 fs, skin 0.5).

| | ab | ba |
| --- | --- | --- |
| SrTiO3 10x10x10 (5,000 atoms) | a 6.6166, b 6.4669 ms: **0.9774** | b 6.4716, a 6.5996 ms: **0.9806** |
| NbCl5 3x3x3 (1,944 atoms) | a 1.7760, b 1.7567 ms: **0.9891** | b 1.7528, a 1.7730 ms: **0.9886** |

Kernel lab in the same job (`ghoist-prod`, the production transform on the captured kernel): forward 1,256.4 →
1,125.4 µs (**0.896**; the falsifier 0.913), message against today's max|d|/max|a| 5.09e-7.

Force/stress gate (`tools/jobs/gate_pack.sh`, SrTiO3 3x3x3, NaCl, NbCl5; ratios to ASE float32, limit 3):

| | SrTiO3 F / S | NaCl F / S | NbCl5 F / S |
| --- | --- | --- | --- |
| default plan | 0.89 / 0.67 | 0.88 / 0.87 | 0.96 / 1.23 |
| GHOIST | 0.89 / 0.66 | 0.86 / 0.86 | 0.98 / 1.45 |

Both pass. Colab cost: screen 3.34 CU (34 min, 27 of them set-up), batch 2 10.24 CU (98 min).

**Superseded by GEOM.** Merging main (`1299b32e`) showed that `spline_forward.GEOM` (other-ways-20261109 §4,
default on the A100 family since 10 November) is the same regrouping, with the g_oi form chosen per path where it
saves operations (FMULs 69 -> 48 per edge and channel; Alex A100-SXM4-80GB MD step 0.9756 / 0.9806). GHOIST
applies the g form to every path; it reproduces GEOM's gain on another A100 model (0.9774 / 0.9806) and does not
stack with it (both rewrite the same sums). GHOIST was therefore dropped at the merge (the source at `cd7575bb`
keeps it), and the Alex confirmation jobs 4981264 / 4981265 were cancelled before they started.
