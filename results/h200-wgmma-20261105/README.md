# H200: Hopper wgmma for the a·W products, and the tabled forward (5 November 2026)

Round 5 on the H200 (Modal, NVIDIA H200 141 GB, driver 610.57.04, sm_90). The question was whether Hopper's
asynchronous warpgroup MMA (`wgmma.mma_async`, sm_90a) speeds up FastMACE's 3xTF32 products. These are the fused
forward's a·W product, the node-level tf32x3 GEMMs and, by extension, the w′ backward. The tabled forward
(`TABLED_FORWARD`) was A/B-tested at the same time after its CUDA-graph recapture fix.

MACE-OMAT-0 medium, float32, SrTiO3 10×10×10 (5,000 atoms), 1 fs, skin 0.5 Å. The MD-step A/Bs are
`tools/ab_rewrites.py`, pattern abba, both build orders.

## Outcomes

| Idea | Switch | Result | Verdict |
|---|---|---|---|
| wgmma microbench: one m64nNk8 3xTF32 product over 373,760 edges, K 64, N 160 | `tools/bench/gen_wgmma_tf32x3.py` | wgmma 0.554 ms (2 warpgroups) against mma.sync 1.082 ms (2 blocks per SM): 1.95× faster for the product alone | basis for the kernels below |
| Fused forward's a·W product on wgmma | `fused_forward.WGMMA` | MD step b/a 1.0243 / 1.0245; Benchmark_loup 5.845 / 5.822 ms (default about 5.75) | rejected (off) |
| The same, software-pipelined (round i's product during round i−1's TP) | `+ WGMMA_PIPELINED` | MD step b/a 1.0354 / 1.0329; Benchmark_loup 5.850 ms | rejected (off) |
| Node-level 3xTF32 GEMMs on wgmma (hand-written, no TMA) | `tools/bench/wgmma_gemm_bench.py` | 24 shapes: 1,325 µs against CUTLASS's sm80 tf32x3 1,112 µs (best tile per shape); 1.0× at M 5,000, up to 2.0× slower at M 25,000–35,000 | rejected |
| Tabled forward on the H200 | `tabled_outer_sum.TABLED_FORWARD` | MD step b/a 0.9838 / 0.9871 | **default on the H200** (decision 66) |

### wgmma forward

`WGMMA=True` replaces the mma.sync product of the fused forward (`TC_WFRAG`) with one m64nNk8 wgmma per k step and
TF32 pass. A warpgroup's four warps walk their own receiver chunks 16 edges at a time in lockstep rounds. The W
slice sits in shared memory as K-major core matrices (both TF32 planes), and A comes from registers. The launch rule
is 8 warps (two warpgroups) per block and one block per SM, because the two W planes and the per-warp staging fill
the 227 KB of shared memory (rule `(2, 8, 17)` for OMAT-0's 10 paths).

- **Accuracy:** the GPU test passes. w errors are 0.85–1.15e-6 against the tc limit. The message is bitwise equal to
  the streaming convolution fed with the kernel's own w.
- **Gate:** NaCl and SrTiO3 pass. NbCl5 stress has ratio 2.12 against the limit of 2 (1.29e-7 against 1.21e-7). That
  is the same class as the A100's accepted NbCl5 stress exception.
- **Speed:** it is 2.4 % slower in MD. The product is about twice as fast, but the kernel's time is the TP: with
  one block of 8 warps per SM, fewer warps hide the TP's latency than the mma.sync kernel's 2 blocks.
- **Pipelined (`WGMMA_PIPELINED`):** each round's wgmma is issued before the previous round's TP and waited for after
  it. The TP keeps its own receiver cursor one round behind, and empty chunks write their zero rows directly. It
  compiles to 244 registers without spills, passes the GPU test and the gate as above (NbCl5 stress ratio 2.22), and
  is slower still: 3.4 %. Keeping the A fragments and the accumulators live across the TP costs the TP its
  registers, and the product it hides was not the bottleneck.
- **Backward:** the w′ backward has the same structure in reverse (TP, then dw·Wᵀ). Since neither forward variant
  wins, it was not ported. A wgmma version of either kernel would first need the shared memory for more warps per SM,
  for example 16-bit W planes or a smaller staging tile.

### wgmma GEMM

The hand-written kernel uses tiles BM×BN×32 with one or two stages. Each stage stages A and the split B planes
through registers into shared memory, then issues the three TF32 passes per k step. It is not competitive. CUTLASS's
sm80 tf32x3 kernels (`fm_tf32x3_rr/rc_64x64`) are within 1.0–1.2× at M 5,000 and pull away as M grows. The
hand-written kernel has no TMA, no warp specialisation and no producer/consumer overlap. The rc products with K 384
also lose accuracy: about 1.7e-6 against CUTLASS's 1.5e-7, which is 10×. A useful sm90 GEMM would be a CUTLASS 3
TMA warp-specialised kernel with a 3xTF32 mainloop. It would save at most part of the about 0.9 ms of tf32x3 GEMMs
per force call.

## Runs

| Directory | Commit | Profile | What |
|---|---|---|---|
| `runs/wgmma-microbench/` | `f38f5b0d` | third | product microbench (`wgmma.json`) |
| `runs/wgmma-forward/` | `f38f5b0d` | third | `tools/modal/h200_wgmma.sh`: GPU test, loup ×2, A/B, gate |
| `runs/wgmma-gemm/` | `8ff87b98` | third | `wgmma_gemm_bench.py` (`wgmma_gemm.json`; `bench-first-attempt.log` is `10656ce3`'s run, which stopped at the first inaccurate shape) |
| `runs/wgmma-pipelined/` | `5ddec837` | third | `h200_wgmma.sh` with `PIPE=1 LOUP=1` |
| `runs/tabled-forward/` | `0017749a` | second | `hopper_sweep.sh` A/B, b = `TABLED_FORWARD=True` |
