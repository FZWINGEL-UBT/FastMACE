# OPT-T03 design: small-M GEMMs (fuller grids for small cells)

Status: **design for review** (20 October 2026), no code. Plan: `plans/opt-1.md`, OPT-T03. Data: 2-T20a
(`fastmace/2-T20a`, `results/2-T20a-step-breakdown-20261020/`; local RTX 5060 Ti, 36 SMs, with Nsight
Compute; murphy RTX 4000 Ada, 48 SMs, profiler only). The OPT-T01 and OPT-T02 B benches are on murphy.

## 1. The problem: NaCl's atom-level GEMMs leave most of the GPU idle

NaCl 3×3×3 has 216 atoms. Its atom-level GEMMs take **1.59 ms of the 3.20 ms force call** on the RTX
5060 Ti (50 %) and 1.56 of 3.37 ms on murphy. The products of one block linear are per irrep l, with
M = 216 · (2l + 1) rows and K = N = 512:

| Product (per force call) | Invocations | µs (local) | TFLOPS (local) | Blocks | Waves on 36 SMs | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| l = 0, `[216, 512] @ [512, 512]` | 11 | 214 | **5.8** | 84 | 0.78 | 17 % |
| l = 1, `[648, 512] @ [512, 512]` | 8 | 256 | 10.6 | – | – | – |
| l = 2, `[1080, 512] @ [512, 512]` | 4 | 205 | 11.0 | 204 | 1.9 | 23 % |
| l = 3, `[1512, 512] @ [512, 512]` | 4 | 274 | 11.6 | 288 | 2.7 | 23 % |
| about 36 smaller ones (N = 6–2048, many with cuBLAS split-K) | – | ≈ 640 | 0.05–10 | 8–96 | 0.04–0.9 | 8–21 % |

Achieved occupancy, blocks and waves are from Nsight Compute (local). The same shapes on NbCl5
(1,944 atoms) run at 11.6–15.1 TFLOPS. The GEMMs themselves are fine; the grids are too small.

**What the l-products have in common:** the products of one `BlockLinear`, `ResidualBlockLinear` or (after
OPT-T02 stage A) `GatedRows` are independent of each other. Each reads its own rows and writes its own
output. Today they run one after another, each underfilling the GPU.

The research engine's "grouped node GEMMs" ceiling (0.1 ms, 14 October 2026) counted only the l = 0
GEMMs reaching the rate of the larger ones. Running the l-products of one layer at the same time was not
screened.

## 2. Candidates

| # | Candidate | Mechanism | Expected (NaCl) | Exactness | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | **Concurrent l-products:** the independent products of one block linear on 2–4 streams, forked and joined inside the captured force graph | a custom op (or the block linear's own forward) that records an event, launches each product on its own side stream and joins them. CUDA-graph capture turns this into parallel graph branches. cuBLAS stays the GEMM | per block linear about max(products) instead of their sum. The atom GEMMs 1.59 → about 1.0–1.2 ms, **−0.4 to −0.6 ms (10–15 % of the step)**, upper estimate | **exact**: the same cuBLAS calls on the same operands, each writing its own output, no atomics | low to medium: Inductor sees an opaque op; side-stream workspaces; OPT-T01b showed that kernels which fill the GPU do not overlap, but these do not fill it |
| 2 | **Grouped GEMM, one launch per layer:** all l-products of a layer in one kernel with a tile scheduler over the groups | Triton grouped GEMM (IEEE `tl.dot`) with the gathered-row addressing of OPT-T02 B (`tools/segment_gemm_bench.py`), so one launch also removes B's copies; or cuBLAS `cublasSgemmGroupedBatched` (no torch binding: a C++ extension) | as 1, plus B's copy removal where B applies; a single launch instead of up to 4 | rounding (another GEMM) | medium: Triton IEEE on small contiguous M was 38 % slower than cuBLAS (OPT-T01 bench); grouping fills the grid, which may close that gap, unmeasured |
| 3 | **Split-K by a shape rule** for the smallest products (l = 0 and the N ≤ 128 ones) | B's Triton segment GEMM with `SPLIT_K`: partial sums in a buffer, then a reduction kernel (two passes, no atomics) | l = 0: 214 → about 120 µs (−0.1 ms); the small N ones, part of 0.64 ms | rounding | low: deterministic; but cuBLAS already split-Ks some of them (its `splitKreduce` kernels show in the trace) |

Ruled out:
- **Batching structures or larger cells:** owner decision, one crystal at a time.
- **Per-GPU tile tables or autotuning:** SPEC §9.1.
- **TF32:** accuracy (and OPT-T01 found no FP32-accurate tensor-core mode that wins).

## 3. Shared tile rule (with OPT-T02 B)

One module, `fastmace/kernels/triton/tile_rule.py`, decides from (M, N, K, number of SMs) and nothing else:
- the backend (cuBLAS or Triton);
- BLOCK_M, BLOCK_N, BLOCK_K, warps and stages;
- SPLIT_K;
- the number of side streams.

The SM count is read from the device. The thresholds are expressed in waves (tiles / SMs), so the same rule
holds on 36 and 48 SMs.

From B's pre-check:
- **Tiles:** 64×64 for N ≤ 512 with M < 8k, 64×128 for wider N or more rows, BLOCK_K 32, 4 warps,
  3 stages.
- **Triton only where M ≥ 2 · SMs · 64** (B).
- **Concurrency where a product's tiles < 1 wave** (candidate 1).

The rule and its thresholds are tested on CPU against tables of shapes, and recorded in the plan's reasons.

## 4. Measurement before building (murphy now; local when the PC is free)

A micro-bench of one MH-1 block linear's products, NaCl and NbCl5 shapes, forward and input gradient,
each captured in a CUDA graph:
- (a) cuBLAS one after another, today's path;
- (b) cuBLAS on 2–4 streams (candidate 1);
- (c) a Triton grouped segment GEMM in one launch (candidate 2);
- (d) split-K for l = 0 (candidate 3).

**Decision rule:** build the cheapest candidate that saves ≥ 10 % of NaCl's atom-level GEMM time on both
GPUs and loses nothing on NbCl5. It goes behind its own opt-in record, with the same checks as stage A:
self-check, reproducible-mode bitwise or determinism, the force gate and the A/B.

**My expectation:** (b) is the cheapest and exact. (c) is the most complete, since it merges with B, but
depends on the Triton IEEE kernel at small M.

## 5. Open points for review

1. (b) puts multi-stream launches into the captured force graph. Fine for the reproducible mode? The
   results are bitwise unchanged when every product writes its own buffer, but the cuBLAS workspace per
   stream must be fixed (`CUBLAS_WORKSPACE_CONFIG` is already set there).
2. Should B and T03 converge on one "grouped segment GEMM" kernel (candidate 2), or is (b) + B the end state?
   I would decide after the bench.

## 6. Result of the micro-bench (murphy, 20 October 2026)

`tools/block_linear_concurrency_bench.py` on murphy (RTX 4000 Ada, 48 SMs), 19:49 UTC, GPU empty, load
0.27. Commit `fe20449`. Raw data: `results/opt-T03-concurrency/murphy/`.

Setup:
- one MH-1 block linear's four products (l = 0–3, 512 → 512), each variant captured in a CUDA graph;
- the median of 100 replays, in µs;
- each speed-up is relative to the sequential variant.

| Case | Sequential (TFLOPS) | 2 streams | 4 streams | Grouped Triton launch |
| --- | ---: | ---: | ---: | ---: |
| NaCl forward | 162.6 (11.1) | 173.1 (0.94×) | 191.4 (0.85×) | 162.6 (1.00×) |
| NaCl input gradient | 188.4 (9.6) | 169.7 (1.11×) | 186.4 (1.01×) | 162.8 (1.16×) |
| NbCl5 forward | 1,325.9 (12.3) | 1,396.7 (0.95×) | 1,391.4 (0.95×) | 1,369.1 (0.97×) |
| NbCl5 input gradient | 1,413.6 (11.5) | 1,390.5 (1.02×) | 1,384.5 (1.02×) | 1,369.1 (1.03×) |

- The cuBLAS variants equal the sequential result bitwise.
- The grouped variant is within 1.2e-6.

**Reading:**
- Measured together, a block linear's four products already reach 11.1 TFLOPS forward on NaCl. The
  slow l = 0 product (6.6 TFLOPS on murphy, 5.8 locally in the model trace) carries only a small share
  of the work, so the four products together are already close to the rate of the large ones.
- **Streams:** the graph branches cost more than they overlap in the forward (0.85–0.95×) and give at
  most 1.11× in one direction.
- **Grouped launch:** saves 7 % of NaCl's forward + input gradient time (351 → 325 µs). It is neutral to
  −3 % on NbCl5.

**Decision (rule of §4): not met on sm_89.** The best candidate, the grouped launch, saves 7 % of these
products on NaCl. The rule needs ≥ 10 % of the atom-level GEMM time and no loss on NbCl5.
- Candidates 1 and 2 are **not built** on this evidence.
- The sm_120 half of the bench runs when the PC is free. Its l = 0 product was the slowest one there
  (5.8 TFLOPS).
- Split-K (candidate 3) was not measured. Its ceiling is l = 0's share, about 0.1 ms on NaCl, below the
  bar.
