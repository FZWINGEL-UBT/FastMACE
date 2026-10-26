# OPT5-T01: Tacker-style scheduler-isolated and asymmetric co-execution probe

Plan: `plans/opt-5.md`, candidate 2 (`OPT5-T01`) and the probe phase of candidate 1 (`OPT5-T02`, `x3_pipe`).
Date: 26 October 2026.

**Verdict:**
- **`OPT5-T01` (`sched_iso` and asymmetric ratios on `sm_120`): FAIL (`gain` max 1.07 vs 1.10 threshold).**
  Isolating SIMT and MMA warps to disjoint SM warp schedulers (`warp % 4`) cuts both pipelines to the fraction of
  sub-partitions assigned to them (`gain = 0.72` on `4×4`), because each SM sub-partition on `sm_120` owns a
  private FFMA pipeline and a private Tensor Core unit. Asymmetric interleaved ratios (`6×2`, `7×1`) raise
  `keep` to `0.97–0.99` but leave 2–3 of the 4 sub-partition Tensor Cores idle (`gain = 1.01–1.03`). Dense
  single-SM GEMM co-execution on `sm_120` remains closed.
- **`OPT5-T02` probe (`x3_pipe`, DeepSeek-style SIMT-split producer + pure-MMA consumer): PASSES the +15 %
  MMA-throughput falsifier (+21 % to +41 % useful 3×TF32 MMA rate), but cannot beat pure SIMT on `sm_120`.**
  Offloading `cvt.rna.tf32` + `sub.f32` operand splitting from the MMA warps to the SIMT warps (synchronized with
  non-blocking PTX `bar.arrive 1`) lowers kernel register usage from 205 (`x3`) to 194 (`x3_pipe`, 0 spill) and
  raises concurrent 3×TF32 MMA throughput from **45.2 → 61.3 FLOP/cycle/SM (+35.6 %, `mma_share` 0.56 → 0.76)**
  at `4×4` and **60.1 → 72.9 FLOP/cycle/SM (+21.3 %, `mma_share` 0.73 → 0.89)** at `2×6`. On `sm_120` (where
  TF32 rate = FP32 rate), the SIMT producer cost (`56.9` FFMA FLOP/cycle/SM) exceeds the `+16.1` useful 3×TF32
  gain (`gain = 0.85`). This confirms that producer/consumer warp specialization is strictly an `sm_80` /
  high-TC-ratio mechanism.

## 1. What was added to `tools/megakernel_probe.py`

1. **Role bitmask (`simt_mask`) and `--sched-modes {interleaved,sched_iso}`:**
   - `interleaved`: warps `0 .. S-1` run SIMT, `S .. S+M-1` run MMA (`OPT4-T07`'s layout; in `4×4`, every
     sub-partition `0..3` holds one SIMT warp and one MMA warp).
   - `sched_iso`: warps are ordered by sub-partition (`w % 4`) before assigning the first `S` warps to SIMT, so
     SIMT and MMA warps sit on separate warp schedulers whenever `S` and `M` are multiples of 2 (in `4×4`,
     schedulers 0–1 hold two SIMT warps each and schedulers 2–3 hold two MMA warps each).
2. **Asymmetric warp configurations (`6×2`, `7×1`, `2×6`):** tested within the 8-warp (256-thread) ceiling of the
   162–205 register `gemm`/`x3`/`x3_pipe` kernels.
3. **`x3_pipe` (`VAR == 4`):**
   - Uses a 4-stage circular shared-memory buffer (`KSM = 4`, 35,848 bytes total, within the 48 KiB static
     shared-memory limit) for `mA`, `mB`, `mAs`, `mBs`.
   - SIMT warps split 256 FP32 elements per chunk into `(big, small)` TF32 pairs in shared memory and execute
     non-blocking `bar.arrive 1` before their SGEMM inner loop.
   - MMA warps load pre-split `fa, fb, as, bs` fragments via `ldmatrix.x4` and issue back-to-back
     `mma.sync.aligned.m16n8k8` instructions with `#pragma unroll 1` on the $k$-loop (194 registers, 0 bytes
     spill).

## 2. Measured results on `sm_120` (RTX 5060 Ti, 36 blocks = 1 block/SM)

Record: `results/opt5-T01-T02-probe-20261026/local/summary.json`.

| Variant | Schedule | `S×M` | Regs | `simt` / `s_all` | `mma` / `m_all` | `both`: `simt`, `mma` | `keep` | `mma%` | `gain` | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gemm` (tf32) | `interleaved` | `4×4` | 162 | 172.7 / 195.8 | 240.5 / 255.1 | 159.5, 137.9 | 0.92 | 0.57 | 1.05 | fail |
| `gemm` (tf32) | `sched_iso` | `4×4` | 162 | 172.8 / 195.7 | 240.5 / 255.2 | 98.4, 127.5 | 0.57 | 0.53 | 0.72 | fail |
| `gemm` (tf32) | `interleaved` | `6×2` | 162 | 183.9 / 195.7 | 119.7 / 255.2 | 178.3, 64.8 | 0.97 | 0.54 | 1.02 | fail |
| `gemm` (tf32) | `interleaved` | `7×1` | 162 | 189.9 / 195.8 | 59.9 / 255.2 | 187.2, 32.0 | 0.99 | 0.53 | 1.01 | fail |
| `x3` (tf32) | `interleaved` | `4×4` | 205 | 171.6 / 196.8 | 80.3 / 83.8 | 157.8, **45.2** | 0.92 | **0.56** | 1.03 | fail |
| `x3` (tf32) | `sched_iso` | `4×4` | 205 | 171.5 / 196.8 | 80.3 / 83.8 | 98.9, 41.9 | 0.58 | 0.52 | 0.72 | fail |
| `x3` (tf32) | `interleaved` | `6×2` | 205 | 183.8 / 196.8 | 40.2 / 83.8 | 177.6, **21.4** | 0.97 | **0.53** | 1.01 | fail |
| `x3` (tf32) | `interleaved` | `2×6` | 205 | 91.6 / 196.8 | 82.0 / 83.8 | 83.5, **60.1** | 0.91 | **0.73** | 0.73 | fail |
| **`x3_pipe`** | `interleaved` | `4×4` | **194** | 161.5 / 190.5 | 80.3 / 83.8 | 100.9, **61.3** | 0.62 | **0.76** | 0.85 | fail (`m_both` **+35.6 %**) |
| **`x3_pipe`** | `interleaved` | `6×2` | **194** | 175.5 / 190.4 | 40.2 / 83.8 | 122.6, **30.2** | 0.70 | **0.75** | 0.80 | fail (`m_both` **+41.1 %**) |
| **`x3_pipe`** | `interleaved` | `2×6` | **194** | 84.4 / 190.5 | 82.1 / 83.8 | 43.5, **72.9** | 0.51 | **0.89** | 0.61 | fail (`m_both` **+21.3 %**) |

## 3. Conclusions

1. **Sub-partition binding (`OPT5-T01`):** On NVIDIA SMs, each of the 4 warp schedulers is wired to its own
   sub-partition's FFMA and MMA execution units. Scheduler isolation (`sched_iso`) starves the unused execution
   unit in each sub-partition (`m_both = 127.5` is exactly $2/4 \times 255.2$). Simultaneous CUDA + Tensor Core
   execution requires both warp types on the *same* sub-partition (`interleaved`), where they share the
   1 instruction/cycle issue slot.
2. **Producer/consumer split offloading (`OPT5-T02`):** Moving the `cvt.rna.tf32` + `sub.f32` operand split off
   the MMA warps onto SIMT producer warps (`x3_pipe`) eliminates the SIMT instruction bottleneck on the MMA
   warps, raising their concurrent utilization from 56 % to 76 % (`4×4`) and from 73 % to 89 % (`2×6`). On
   `sm_120`, where 1 TF32 MMA FLOP equals 1 FP32 FFMA FLOP, spending SIMT cycles to feed a 3×TF32 decomposition
   cannot beat running FP32 FFMA directly on the SIMT warps.
