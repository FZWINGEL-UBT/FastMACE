# Exponent-Shifted `3×FP16` Tensor Core GEMMs: Standalone (`sm_80` / `sm_89`) and Fused SIMT+MMA (`sm_120`)

Date: 26 October 2026 (`OPT5-T06` / `OPT5-T08`).
Related documents:
- [Optimisation round 5 plan](../../plans/opt-5.md)
- [Optimisation round 5 validation report](../validation/opt-5.md)
- [Heterogeneous megakernel design note](2026-10-23-heterogeneous-megakernel.md)
- [OPT-T01 Tensor Core survey](../validation/opt-T01-tensor-cores.md)

---

## 1. The core numerical insight: why Exponent-Shifted `3×FP16` beats `3×TF32` and `BF16×6`

Every NVIDIA GPU from Ampere (`sm_80`) through Ada (`sm_89`) and Blackwell (`sm_120`) executes 16-bit Tensor Core
matrix-multiply-accumulate (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and `.bf16.bf16.f32`, which
accumulate in **full IEEE FP32**) at **2× the throughput of TF32** (`m16n8k8`):

| Architecture | Example GPUs | FP32 SIMT (FLOP/cyc/SM) | TF32 MMA (FLOP/cyc/SM) | FP16 / BF16 MMA (FLOP/cyc/SM) | `3×TF32` ceiling (`c = 3`) | `BF16×6` ceiling (`c = 6`) | **`3×FP16` ceiling (`c = 3`)** | **`3×FP16` / FP32 SIMT** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sm_80` (Ampere) | A100 | 128 | 1,024 | **2,048** | 341.3 | 341.3 | **682.7** | **5.33×** |
| `sm_89` (Ada) | RTX 4000 Ada (`leto`, `murphy`), L4 | 256 | 512 | **1,024** | 170.7 | 170.7 | **341.3** | **1.33×** |
| `sm_120` (Blackwell) | RTX 5060 Ti | 256 | 256 | **512** | 85.3 | 85.3 | **170.7** | **0.67× standalone / 1.22× co-exec** |

### Why previous screens missed `3×FP16`
1. **`BF16` has only 7 explicit mantissa bits** (8 with the hidden bit). Reaching 24-bit IEEE FP32 precision
   requires a 3-way operand split ($8 \times 3 = 24$ bits) and **6 MMA products (`BF16×6`, `c = 6`)**, which
   halves the 2× throughput advantage of 16-bit Tensor Cores (`512 / 6 = 85.3` FLOP/cycle/SM on `sm_120`,
   identical to `3×TF32`'s `256 / 3 = 85.3`).
2. **IEEE `FP16` has 10 explicit mantissa bits (11 with the hidden bit)—the exact same mantissa precision as
   `TF32`!** A 2-way split (`big` + `small`) therefore covers $11 + 11 = 22$ mantissa bits, requiring only **3
   MMA products (`3×FP16`, `c = 3`)**.
3. The only obstacle to `3×FP16` is FP16's narrow 5-bit exponent (normal range $[2^{-14}, 2^{15}] \approx
   [6.10 \times 10^{-5}, 65504]$):
   - Because the remainder $r = x - \text{fp32}(\text{fp16}(x))$ satisfies $|r| \le 2^{-11}|x|$, converting $r$
     directly to FP16 underflows whenever $|x| < 2^{-3} = 0.125$.
   - **Fix (exact power-of-two remainder shift):** multiply the remainder by $2^{11} = 2048.0$ on the CUDA cores
     before converting to FP16, and scale the cross-term accumulator by $2^{-11} = 1 / 2048.0$ in FP32 after the
     MMA loop:
     $$\text{big} = \text{rn}_{\text{fp16}}(x), \qquad \text{small} = \text{rn}_{\text{fp16}}\!\Big(\big(x - \text{fp32}(\text{big})\big) \times 2048.0\Big)$$
     $$C = \text{MMA}(A_{\text{big}}, B_{\text{big}}) + 2^{-11}\Big(\text{MMA}(A_{\text{big}}, B_{\text{small}}) + \text{MMA}(A_{\text{small}}, B_{\text{big}})\Big)$$
   - **No-overflow guarantee:** since $|x - \text{fp32}(\text{big})| \le 2^{-11}|x|$ under round-to-nearest-even,
     $|(x - \text{fp32}(\text{big})) \times 2048.0| \le |x|$. Thus `small` has the **exact same exponent ceiling
     as `big`** and can never overflow FP16 if `big` does not overflow FP16.

### Preventing subnormal underflow on small rows/columns (exponent normalization)
In MACE backward passes and sparse species tables, an operand row $A_{m,:}$ or column $B_{:,n}$ can have
magnitudes $\ll 2^{-14}$ (e.g. $10^{-6}$ in force gradients or $10^{-25}$ on inactive species rows). To ensure
`big` and `small` always stay in the normal FP16 range $[1.0, 2.0)$ with zero accuracy loss:
- Extract the power-of-two exponent of each row $m$ of $A$ (or tile of $A$) and each column $n$ of $B$ (or tile
  of $B$) using an IEEE FP32 exponent bit-mask (`bits & 0x7F800000`, clamped to $[2^{-120}, 2^{120}]$):
  $$S_m = 2^{\lfloor \log_2 \max_k |A_{mk}| \rfloor}, \qquad T_n = 2^{\lfloor \log_2 \max_k |B_{kn}| \rfloor}$$
- Normalize $\tilde{A}_{mk} = A_{mk} / S_m$ and $\tilde{B}_{kn} = B_{kn} / T_n$ (exact power-of-two exponent
  shifts via `0x7F000000 - exp_bits`, changing zero mantissa bits!), run `3×FP16` on $(\tilde{A}, \tilde{B})$,
  and multiply the FP32 output by $S_m T_n$ in the epilogue.

---

## 2. Two deployment paths: Standalone (`sm_80` / `sm_89`) vs. Fused Co-Execution (`sm_120`)

Even if the single-SM heterogeneous `SIMT + 3×FP16` megakernel on `sm_120` does not beat `cuBLAS IEEE FP32`
end-to-end, **standalone `3×FP16` is directly valuable on `sm_80` (A100) and `sm_89` (RTX 4000 Ada workstations
`leto`/`murphy` and Modal L4)**:

### Path A: Standalone `3×FP16` on A100 (`sm_80`) and Ada / L4 (`sm_89`)
- **On `sm_80` (A100):**
  - `3×FP16` has a hardware ceiling of **682.7 FLOP/cycle/SM** (`5.33×` FP32 SIMT, and **2× `3×TF32`'s 341.3**).
  - Already on `sm_120`, our standalone Triton `triton_fp16x3_kernel` (`tools/bench/opt5_t06_t07_bench.py`) is
    **1.78–1.88× faster than `triton_tf32x3`** across all production shapes (`77.5 µs` vs `142.0 µs` on NaCl
    `1512×512×512`, `276.5 µs` vs `491.6 µs` on NbCl5 `5832×512×512`, `354.3 µs` vs `664.6 µs` on OMOL-0 XL
    `1944×1024×1024`) and is **more accurate** than both `cuBLAS IEEE` and `triton_tf32x3` (`2.02e-7` vs
    `3.41e-7` and `2.29e-7` max relative error, with smaller energy offset on `MH-1 NaCl` because `cvt.rn.f16`
    rounds to nearest-even instead of truncating).
- **On `sm_89` (RTX 4000 Ada workstations `leto`/`murphy` and Modal L4):**
  - `3×TF32` could never beat `cuBLAS IEEE FP32` on `sm_89` (`OPT-T01 §2`: TF32 peak is `38.6` TFLOPS vs FP32's
    `16.3` TFLOPS, so `38.6 / 3 = 12.9 < 16.3`).
  - With `3×FP16`, 16-bit Tensor Cores on `sm_89` run at **2× TF32 speed** (`~77` TFLOPS peak on RTX 4000 Ada),
    giving a `3×FP16` ceiling of **`77 / 3 ≈ 25.7` TFLOPS vs `16.3` TFLOPS for cuBLAS FP32 (`1.58×` higher ceiling)**.
  - **Action when workstations (`leto`/`murphy`) return or on Colab/Modal L4 & A100:** benchmark standalone
    `triton_fp16x3_kernel` (and CUTLASS `3×FP16`) against `cuBLAS IEEE FP32` on the 92 production shapes of
    `gemms.csv`, and expose it as `RewriteConfig.fp16x3_gemms` (or upgrade `tf32x3_gemms` to `fp16x3` where
    faster).

### Path B: Single-Kernel Fused `SIMT + 3×FP16` on RTX 5060 Ti (`sm_120`)
- On `sm_120`, standalone `3×FP16` has a `170.7` FLOP/cycle/SM ceiling vs `256` for FP32 CUDA cores, so
  standalone `triton_fp16x3` ties `cuBLAS IEEE` on `NaCl` (`77.48` vs `77.46 µs`) and is `1.09–1.25×` of
  `cuBLAS IEEE` on large shapes (`276.5` vs `227.9 µs` on `NbCl5 5832×512×512`; `354.3` vs `324.3 µs` on
  `OMOL-0 1944×1024×1024`).
- In the single-SM co-execution probe (`OPT5-T06a`, `results/opt5-T06-fp16x3-20261026/probe/summary.json`),
  running **4 SIMT warps + 4 `3×FP16` MMA warps in the same thread block** achieved:
  - `157.7` SIMT FLOP/cycle/SM (`keep = 0.92`) + `82.8` useful `3×FP16` FLOP/cycle/SM = **`240.5` FLOP/cycle/SM**
    vs `196.8` for all 8 warps on CUDA cores (**`gain = 1.22×`, +22 %**).
- To test whether that `+22 %` inner-loop gain survives global memory traffic and shared-memory staging in a real
  GEMM on `sm_120`, we build a single NVRTC kernel (`64 + 32` column split per block: 4 SIMT warps compute a
  `64×64` FP32 tile while 4 MMA warps compute a `64×32` `3×FP16` tile sharing the same `64×K` strip of $A$) and
  benchmark it directly against `cuBLAS IEEE FP32`. If it beats `cuBLAS IEEE FP32`, we integrate it; if shared
  memory / memory-controller contention erodes the margin on `sm_120`, Path A (standalone `3×FP16` for `sm_80` /
  `sm_89`) remains the clean deliverable.


---

## 3. Measured results (`OPT5-T08`, `results/opt5-T08-hybrid-gemm-20261026/summary.json`)

### 3.1 Path A: Per-row ($A$) / per-column ($B$) normalized `triton_fp16x3_rowcol_kernel` on all 52 real `MH-1 NaCl` products

Replacing tile-wide `tl.max` with per-row `tl.max(..., axis=1)` for $A$ (`[BLOCK_M, 1]`) and per-column
`tl.max(..., axis=0)` for $B$ (`[1, BLOCK_N]`) inside `triton_fp16x3_rowcol_kernel` (`tools/bench/opt5_t08_hybrid_gemm.py`)
**eliminates the sparse species-row subnormal outlier completely** and beats both `cuBLAS IEEE FP32` and
`OPT5-T05`'s `tf32x3` on accuracy across the 52 real force-call products of `MH-1 NaCl`:

| Metric across 52 real `MH-1 NaCl` products | `cuBLAS IEEE FP32` | `tf32x3` (`OPT5-T05`) | **`triton_fp16x3_rowcol` (`OPT5-T08`)** |
| --- | ---: | ---: | ---: |
| Worst cancellation-scaled error vs FP64 | $6.44 \times 10^{-7}$ | $5.59 \times 10^{-7}$ | **$5.02 \times 10^{-7}$** |
| Median cancellation-scaled error ratio vs IEEE | $1.000\times$ | $0.590\times$ | **$0.568\times$** |
| **Max** cancellation-scaled error ratio vs IEEE | $1.000\times$ | $1.727\times$ | **$1.519\times$** |
| Median `max_rel` error ratio vs IEEE | $1.000\times$ | $0.665\times$ | **$0.578\times$** |
| **Max** `max_rel` error ratio vs IEEE | $1.000\times$ | $1.796\times$ | **$1.575\times$** |
| Products with higher error than `cuBLAS IEEE` | 0 of 52 | 6 of 52 | **3 of 52** |

### 3.2 Path B: Double-Buffered Pipelined `SIMT + 3×FP16` NVRTC Kernels (`OPT5-T08`) on `sm_120`

We engineered and benchmarked four progressively optimized versions of the single-launch fused `SIMT + 3×FP16`
NVRTC kernel (`tools/bench/opt5_t08_hybrid_gemm.py`):
1. **Coalesced `float4` global loads + register prefetching (2-stage software pipeline):** global loads for
   `k0 + BK` are issued into registers at the start of each step while warps compute stage `k0`, cutting
   `__syncthreads()` from 2 per step to **1 per step**.
2. **Post-barrier overlapped `split_f16x2`:** `A` and `B` are staged in shared memory as plain FP32; the MMA
   warps execute `split_f16x2` *after* `__syncthreads()` in parallel with the SIMT warps' `fma.rn.f32` loop.
3. **Single MMA accumulator array (`mma_acc` + 4-register `tmp[4]`):** eliminates the 32-register `mma_cross`
   array, keeping both `hybrid_gemm_64_32` (`4×4` warps, `64×96` tile) and `hybrid_gemm_6x2_64_128` (`6` SIMT
   warps on `64×96` + `2` MMA warps on `64×32` = **`64×128` power-of-two tile**) at **128 registers with 0 spill
   and 2 blocks (16 warps) per SM**.
4. **Vectorized `ld.shared.v4.f32` SIMT inner loop.**

| Shape (`[M, K, N]`) | `cuBLAS IEEE` (µs) | `triton_tf32x3` (µs) | `triton_fp16x3` tile-norm (µs) | **`nvrtc_hybrid_64_32` (µs)** | **`nvrtc_hybrid_6x2_64_128` (µs)** | Hybrid Accuracy (`max_rel` vs `cuBLAS`) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `NaCl [1512, 512, 512]` | 73.75–74.37 | 142.03 | 77.48 | **67.67–72.99 (−1.0 to −9.0 %)** | **72.10–73.86 (−0.7 to −2.2 %)** | $9.83\times 10^{-7}$ vs $3.41\times 10^{-7}$ |
| `NbCl5 [5832, 512, 512]` | **233.95–237.19** | 491.57 | 276.50 | 261.74–268.00 | **251.76–254.30 (+6.1 to +7.6 %)** | $9.21\times 10^{-7}$ vs $9.11\times 10^{-7}$ |
| `NbCl5 [13608, 512, 512]` | **475.61–484.20** | 1,128.96 | 612.30 | 615.81–616.33 | **579.89–580.66 (+19.8 %)** | $9.87\times 10^{-7}$ vs $9.37\times 10^{-7}$ |
| `OMOL [1944, 1024, 1024]` | **299.36–299.83** | 664.62 | 354.25 | 319.17–330.39 | **311.15–316.97 (+3.9 to +5.7 %)** | $1.62\times 10^{-6}$ vs $8.00\times 10^{-7}$ |

**Interpretation and limits:** the register/shared-memory tradeoff motivates larger tiles, but it is not
a measured attribution of the cuBLAS performance gap. No profiler evidence here establishes cuBLAS's
internal tile or which memory level limits it. The later screen below queries occupancy directly and
includes packing costs, larger matrices, boundary checks and real-operand accuracy.

### 3.3 Larger tiles and pre-split B: not qualified (26 October 2026)

Owner request: make the hybrid faster for more atoms. Tool:
`tools/bench/opt5_t08_large_gemm.py`. Records:
`results/opt5-T08-large-20261026/{screen1,screen2,screen3,screen4,screen5,screen6,validation}.json`.
These are dense GEMM screens, not MD trajectories; `M` is matrix rows, not necessarily atoms.

**Setup:** local `Fabi-PC`, RTX 5060 Ti (`sm_120`, 36 SMs), WSL
`~/environments/mace_cueq_oeq/bin/python` (Python 3.10.12, torch 2.14.1+cu130,
Triton 3.8.0). IEEE FP32 cuBLAS versus FP32 SIMT + scaled-residual `3×FP16` MMA;
TF32 disabled. FastMACE baseline `e63665b4`, with the new benchmark source;
each JSON records the measured script and imported baseline SHA-256 hashes.
The metadata field `driver = 13030` is CUDA's driver API version, not an NVIDIA
driver package number. All runs used a quiet `tools/ci/local_gpu.py` reservation.
No workstation job or NAS synchronization was performed.

**Variants tested:**

- Pack B once into the `hi/lo` FP16 pairs needed by MMA, eliminating repeated B splitting in each output tile.
  Report both already-packed GEMM and packing-plus-GEMM; do not assume reusable weights for every force-call product.
- Larger `128×128` row-partitioned tiles, shared accumulator storage for mutually exclusive SIMT/MMA warp roles,
  and `cp.async` B staging. The first row variants spilled; the asynchronous variant did not close the timing gap.
- `64×128` and `128×128` column partitions with aligned vector loads, packed B read directly or staged in shared memory.
  These aligned variants require `K % 16 == 0` and `N % 128 == 0`; they are not general drop-in GEMMs.

The CUDA occupancy query reports **two active blocks/SM for both old controls and all four final candidates**.
The old `6×2` control uses 26,624 shared bytes/block, 128 registers/thread and no local-memory spill;
the packed column variants use 21,504 / 29,696 shared bytes/block, 128 registers and no spill.
This disproves the earlier hypothesis that the old padded kernel was restricted to one block/SM by a
48 KiB aggregate shared-memory limit. Removing padding had already slowed the kernel; it was restored.

#### Final same-process timings

`validation.json`: 80 GEMMs per CUDA-graph replay, 12 blocks, reversed execution order on alternate blocks.
Medians in microseconds; both new columns **include B packing on every call**.
The fixed candidates are shown across all shapes, rather than selecting a different kernel for each shape.

| `[M, K, N]` | cuBLAS IEEE | Previous `6×2`, `64×128` | New packed `64×128` | New packed `128×128` |
| --- | ---: | ---: | ---: | ---: |
| `[1512, 512, 512]` | 67.90 | 67.52 | 69.31 | 92.55 |
| `[5832, 512, 512]` | 245.77 | 269.67 | 241.71 | 285.59 |
| `[13608, 512, 512]` | 527.83 | 613.62 | 616.55 | 596.65 |
| `[1944, 1024, 1024]` | 330.22 | 338.20 | 337.35 | 375.12 |
| `[32768, 512, 512]` | 1261.27 | 1498.13 | 1545.29 | 1545.68 |
| `[5832, 1024, 1024]` | 912.25 | 1028.52 | 974.71 | 1035.22 |

Packing alone costs about 4.4 µs for `512×512` B and 15.8 µs for `1024×1024`.
The small apparent win at 5,832 rows is not reliable evidence of superiority: cuBLAS's blocks range
220.44–258.65 µs and the new packed path's 230.56–287.18 µs. Independently timed packing-plus-GEMM can
even appear faster than GEMM alone in this noisy run; do not infer a negative packing cost from their medians.
At 13,608 rows the new larger tile remains 13% slower than cuBLAS; at 32,768 rows neither tile improves on
the old control. Prefetching packed B through shared memory also failed to produce a general win.

#### Numerical and indexing checks

- **18 boundary cases passed:** partial M tiles, the generic row candidate's odd K/N tails, output canaries,
  repeated-launch bitwise equality, and packing-plus-GEMM CUDA-graph replay equality.
  Worst max-relative error against a float64 product was `8.14e-7` on these Gaussian inputs.
- **Real operands:** capture one rattled MH-1 NaCl 3×3×3 force call (216 atoms, seed 20261024,
  ±0.05 Å displacement), then replay its 52 products individually. Model and structure SHA-256 hashes and
  the capture plan are in the JSON. Structures came from the frozen research-baseline export
  (`research-engine-baseline-20261017`, `d2faa9fd`); the control here is cuBLAS, not the research MD engine.
  The generic row candidate covers all 52 products; each aligned column candidate covers 38.
  All outputs are finite, but 36/52 row products and 28/38 products for each column variant have
  cancellation-scaled error greater than twice IEEE. For a full-tile `[648,128,512]` product,
  the column variants reach `0.244` versus IEEE's `3.08e-7` in the cancellation-scaled metric,
  despite a superficially good `2.98e-7` global max-relative error.
- **Dynamic range fails:** reciprocal operand scalings `A *= 2**-30, B /= 2**-30` and
  `A *= 2**20, B /= 2**20` leave the exact product scale unchanged but produce nonfinite hybrid output
  for every final candidate. The fixed FP16 split does not preserve FP32's exponent range.

**Decision:** do not integrate these candidates or claim a larger-system MD speedup. Production remains
cuBLAS IEEE on `sm_120`. No force/stress gate or MD timing was run for these rejected hybrid candidates:
they fail the earlier per-product accuracy and general-speed requirements.

### 3.4 Why the hybrid loses: Nsight Compute and sustained-power profile (26 October 2026)

Tool: `tools/bench/opt5_t08_profile.py`. Records:
`results/opt5-T08-profile-20261026/{sustain.json,sustain.smi.csv,tensor-energy.json,tensor-energy.smi.csv,ncu_raw.csv}`.
Nsight Compute used the full metric set, base clock control, three warm-up launches and then one measured
launch for each of four kernels at three shapes. The sustained run replayed a CUDA graph of 80 calls for
4 s per kernel while `nvidia-smi` sampled clock, power and throttle state every 50 ms.

At `[13608, 512, 512]`, cuBLAS selects CUTLASS `cutlass_80_simt_sgemm_128x128_8x4_nn_align1`
(128 threads/block, 200 registers/thread). The hybrid candidates use 256 threads/block and
128 registers/thread. Percentages below are Nsight Compute pipe-active or issue-active values, not
direct FLOP rates.

| Kernel | NCU duration (µs) | Issue active | FMA pipe | Tensor pipe | Shared wavefronts | Barrier stall* | DRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuBLAS FP32 | 523.6 | 76.6 % | 70.2 % | 0 % | 52.3 % | 0.13 | 25.5 % |
| Previous `6×2` hybrid | 615.0 | 61.5 % | 52.2 % | 24.8 % | 69.0 % | 0.97 | 44.2 % |
| Packed `64×128` | 597.7 | 62.4 % | 51.1 % | 24.7 % | 51.8 % | 1.34 | 44.0 % |
| Packed `128×128` | 601.6 | 58.2 % | 50.1 % | 22.7 % | 38.5 % | 1.92 | 43.2 % |

\*Average warps stalled at barriers per issued instruction.

The profile rules out the simplest explanations:

- **Not DRAM-bound:** DRAM throughput is at most 44 % and L2 at most 32 % in these NCU launches.
- **Shared-memory bank conflicts are not decisive:** the packed `64×128` kernel cuts shared wavefronts
  from 69.0 % to 51.8 % and bank conflicts from 7.37 million to 1.04 million versus the previous hybrid,
  yet improves only 615.0 → 597.7 µs and remains 14 % slower than cuBLAS.
- **Synchronization and lost SIMT efficiency dominate:** every hybrid couples SIMT and MMA warps at tile
  barriers. Barrier stalls are 0.97–1.92 per issued instruction versus cuBLAS's 0.13; short-scoreboard
  stalls are 0.35–0.68 versus 0.08. Issue activity falls from 76.6 % to 58–62 %, and FMA-pipe activity
  from 70.2 % to 50–52 %. The Tensor pipe is active only 23–25 % of cycles, so it compensates for part of
  the lost FP32 work rather than adding a second full-rate stream on top of an equally efficient SIMT path.
- **More resident warps do not fix it:** the hybrids have about 32 % active warps versus cuBLAS's 17 %,
  but lower issue activity. cuBLAS spends 200 registers/thread on a larger per-thread accumulator tile;
  the hybrids are capped at 128 registers/thread to keep two 256-thread blocks resident while also holding
  MMA accumulators in registers.

#### Sustained power ceiling

At the same `[13608, 512, 512]` shape, every sustained kernel below sits at the card's 180 W cap.
Energy is median sampled power × sustained time per GEMM.

| Kernel | Sustained time (µs) | Throughput | Median SM clock | Energy/GEMM |
| --- | ---: | ---: | ---: | ---: |
| cuBLAS FP32 | 522.3 | 13.66 TFLOP/s | 2520 MHz | 94.0 mJ |
| cuBLAS FP16, one pass (not FP32-accurate) | 174.5 | 40.88 TFLOP/s | 2685 MHz | 31.4 mJ |
| Previous `6×2` hybrid | 617.5 | 11.55 TFLOP/s | 2475 MHz | 111.1 mJ |
| Triton row/column-normalized `3×FP16` | 860.3 | 8.29 TFLOP/s | 2745 MHz | 154.8 mJ |

Three one-pass FP16 cuBLAS calls would take `3 × 174.5 = 523.6 µs`, essentially the FP32 time, before
any split, rescaling or synchronization work. A fused three-pass kernel shares operand loads, so this is
not a hard lower bound; it does show that cuBLAS-quality FP16 execution leaves only a few percent of
energy/time headroom for the extra work on this GPU and shape. The measured hybrids use 18 % more energy
per GEMM than FP32 cuBLAS, and the current normalized Triton kernel 65 % more.

**Profile verdict:** a faster kernel is not mathematically excluded, but it must preserve cuBLAS-level
SIMT issue/FMA efficiency while adding Tensor-core work, or reach near-cuBLAS FP16 efficiency in a
normalized tensor-only three-pass kernel. The present designs trade roughly 18 points of FMA-pipe activity
and 15 points of issue activity for 23–25 % Tensor-pipe activity, then lose more to barrier coupling.
That trade is negative on `sm_120`, and the 180 W cap removes the clock headroom that raw peak-FLOP
arithmetic suggests. No further hybrid variant is justified without a design that first removes the
per-tile SIMT/MMA barrier coupling.

### 3.5 Subgroup-barrier decoupling: implemented, no gain (26 October 2026)

ChatGPT Deep Research proposed, as its lowest-risk design, independent SIMT and MMA pipelines in one CTA.
We implemented that design as `col64_aligned_decoupled` in `tools/bench/opt5_t08_large_gemm.py`.
Records: `results/opt5-T08-large-20261026/{screen7-decoupled,validation-decoupled}.json` and
`results/opt5-T08-profile-20261026/ncu_decoupled_raw.csv`.

**What changed relative to `col64_aligned_packed`:**

- Six SIMT warps synchronize only with `barrier.cta.sync 1, 192`; two MMA warps only with
  `barrier.cta.sync 2, 64`. The unaligned named-barrier form and participant counts were checked against
  the PTX ISA documentation.
- SIMT and MMA each stage a private copy of the FP32 `A` tile (so `A` staging traffic doubles), SIMT stages
  its FP32 `B` tile, and MMA reads pre-packed `B` directly from global memory.
- Arithmetic, output partition and accumulator layout are unchanged. The kernel uses 126 registers/thread,
  30,208 shared bytes/block, no local-memory spill and still fits two blocks/SM, versus 128 registers and
  21,504 shared bytes for the coupled packed kernel.

**Graph-replay timings (already-packed B, medians, µs):**

| `[M, K, N]` | cuBLAS IEEE | Coupled packed `64×128` | Decoupled `64×128` | Decoupled vs coupled |
| --- | ---: | ---: | ---: | ---: |
| `[1512, 512, 512]` | 67.85 | 66.82 | 67.32 | +0.7 % |
| `[5832, 512, 512]` | 242.50 | 254.57 | 267.52 | +5.1 % |
| `[13608, 512, 512]` | 517.81 | 584.15 | 615.07 | +5.3 % |
| `[1944, 1024, 1024]` | 329.38 | 323.75 | 339.95 | +5.0 % |
| `[32768, 512, 512]` | 1260.33 | 1539.21 | 1567.07 | +1.8 % |
| `[5832, 1024, 1024]` | 908.91 | 971.73 | 1004.64 | +3.4 % |

Including B packing on every call also leaves the decoupled kernel slower on every shape.

**Nsight Compute at `[13608, 512, 512]`:**

| Kernel | NCU duration (µs) | Issue active | FMA pipe | Tensor pipe | LSU pipe | Barrier stall* | Instructions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuBLAS FP32 | 526.5 | 76.3 % | 70.3 % | 0 % | 39.2 % | 0.129 | 128.3 M |
| Coupled packed | 600.1 | 61.6 % | 51.2 % | 24.7 % | 47.1 % | 1.357 | 117.9 M |
| Decoupled | 619.6 | 62.9 % | 49.4 % | 23.9 % | 50.9 % | 0.880 | 123.3 M |

\*Average warps stalled at barriers per issued instruction.

The named barriers work: cross-group synchronization is removed and the barrier-stall metric falls by
35 %. It does not approach cuBLAS's 0.129 because each subgroup still synchronizes its own double-buffered
staging pipeline every K tile. The recovered synchronization time is more than spent on duplicate `A`
loads/stores and subgroup load work: instruction count rises 4.6 %, LSU-pipe activity rises, and both FMA
and Tensor activity fall slightly.

**Accuracy unchanged, still failing:** the decoupled kernel passes the same four aligned boundary cases
(bitwise repeat and CUDA-graph replay, output canaries intact, worst max-relative error `8.14e-7`).
On the 38 captured real MH-1 products it is finite everywhere and has the same error distribution as the
coupled kernel in the same run, with 27 products above twice IEEE cancellation-scaled error; both reciprocal
exponent-range scalings still produce nonfinite output. Synchronization was never the accuracy problem.

**Verdict:** Design 1 is falsified as implemented. Barrier coupling is real but not the dominant recoverable
cost; removing it does not justify the duplicate staging work. This does not falsify completely separate
kernels or a deeper `mbarrier` producer–consumer design, but it lowers their expected value: the remaining
gap is principally lost SIMT efficiency, instruction/LSU work and the power ceiling. ChatGPT's claims about
CUTLASS SM120 warp specialization, `setmaxnreg`, TMA pipelines and green contexts were not tested here and
remain hypotheses, not evidence for this kernel.

### 3.6 Five further improvement screens: all negative or neutral (26 October 2026)

Tools: `tools/bench/opt5_t09_screens.py`, plus `col64_aligned_bk32` and `col64_aligned_swizzled` in
`tools/bench/opt5_t08_large_gemm.py`. Records: `results/opt5-T09-screens-20261026/{screens,bk32-swizzle}.json`.
Primary shape `[13608, 512, 512]`; Gaussian operands; medians in µs.

**1. Two-kernel concurrent column split — rejected.** cuBLAS FP32 computes the first columns while the
Triton row/column-normalized `FP16×3` kernel computes the rest, launched on separate streams.
Full cuBLAS: 501.0. Best split (384 FP32 + 128 MMA columns): FP32 part alone 411.1, MMA part alone 216.4,
sequential 731.2, concurrent 734.1. Concurrency recovers nothing: the MMA slice is slower per column than
FP32 (216 µs for 128 columns), so co-scheduling two slower-per-FLOP parts cannot beat one cuBLAS call.

**2. BK=32 staging and block swizzling — no general gain.** `col64_aligned_bk32` (coupled kernel, one
`A` copy, 43,008 shared bytes, 128 registers, 8 local bytes, two blocks/SM) versus the BK=16 packed kernel
in the same run: 619.7 vs 610.4 µs at 13,608 rows; 269.9 vs 266.9 at 5,832; 71.6 vs 68.1 at 1,512.
Halving the staging/barrier frequency does not pay. The A-sharing swizzled rasterization is faster at
5,832 rows (252.8 vs 266.9) but slower at 1,512 (78.0 vs 68.1) and unchanged at 13,608 (608.0 vs 610.4):
within noise, no general rule. Both remain far behind cuBLAS (521.8 in that run).

**3. FP32 baseline options — cuBLAS stands.** `torch.mm` 501.0; cuBLASLt via
`torch.backends.cuda.preferred_blas_library("cublaslt")` 501.9 (493.9 in a first run: noise, no gain);
transposed-B view 697.7 (much slower). CUTLASS was checked, not benchmarked: no CUTLASS Python package is
installed in the GPU environment; `nvidia-cutlass-dsl` 4.8.1 installs into a temporary target but ships
only SM100 GEMM helpers, and adapting NVIDIA's repo SM120 examples to a normalized three-product kernel
is a separate project, not a screen. No FP32 alternative beat the default.

**4. Tensor-only `FP16×3` configuration sweep — still slow.** Six Triton row/column-normalized
configurations: best `(BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) = (64, 128, 32, 4, 3)` at 795.9 µs,
with good Gaussian accuracy (`cancel_scaled 8.5e-8`). Larger tiles are slower or exceed shared memory.
On-the-fly normalization costs more than the Tensor cores return in Triton on `sm_120`.

**5. Producer-side A splitting and epilogue fusion — conditional, not demonstrated.** A Triton kernel
consuming pre-split FP16 high/low operands with global row/column power-of-two scales runs in 505.9 µs,
essentially tying cuBLAS (501.0), with Gaussian `max_rel 1.09e-6`, `cancel_scaled 4.29e-7`. But producing
the A split with unfused PyTorch operations costs 1,231 µs per call, for 1,737 µs total. Producer-side
splitting therefore only helps if the producing layer emits high/low operands at near-zero marginal
cost, which no current MACE producer does; that is an end-to-end rewrite, not a GEMM screen result.
Fusing a representative bias + sigmoid-gate + residual epilogue into that kernel (866 µs) beats a naive
separate PyTorch chain (1,669 µs, inflated by intermediate allocations), but remains 360 µs above the
plain GEMM in this implementation: directional evidence only.

**Overall verdict:** none of the five directions produces a faster FP32-accurate GEMM on `sm_120`.
The pre-split tensor kernel tying cuBLAS *before* its recurring A-preparation cost is the closest result;
it confirms the recurring operand preparation, not arithmetic throughput, is the remaining barrier.
Production remains cuBLAS IEEE FP32 on this GPU.

### 3.7 Warpgroup registers, CUTLASS/TMA check, and fused producer split (26 October 2026)

Records: `results/opt5-T09-screens-20261026/{setmaxnreg,producer}.json`; CUTLASS run described below.

**1. 4+4 warpgroup layout with `setmaxnreg` (`sm_120a`) — improves its own control, still loses.**
`col128_4x4_setmaxnreg` in `tools/bench/opt5_t08_large_gemm.py`: four SIMT warps + four MMA warps
(warpgroup-aligned), SIMT warpgroup `setmaxnreg.inc 232`, MMA warpgroup `setmaxnreg.dec 64`, compiled
for `sm_120a` in a separate NVRTC module. It compiles to 165 registers/thread, no local spill, one
block/SM. Against its direct 128-register control (`col128_s4_packed`, 96 local bytes of spill) it is
much faster: 716.3 vs 1,002.7 µs at `[13608, 512, 512]`, 296.7 vs 440.7 at 5,832 rows. Register headroom
is therefore real for this layout. But the 4+4 / `BM=128` layout itself remains slower than the 6+2
`BM=64` packed kernel (606.6 µs) and cuBLAS (530.6 µs) in the same run. Giving the SIMT side registers
does not rescue a layout whose tile/work partition is inferior; no further register variant is justified.

**2. CUTLASS GeForce SM120 dense GEMM (TMA/`mbarrier` machinery) — runs, no rate advantage.**
NVIDIA's CuTe DSL `blackwell_geforce` dense GEMM example (CUTLASS repo, sparse checkout in `~/tmp`,
DSL 4.8.1 in a temporary target directory, no environment changed) passes on this GPU at
`[13608, 512, 512]` FP16→FP32 in **194.5 µs** per iteration (tile `128×256×64`). That is slower than
cuBLAS FP16's 174.5 µs from §3.4. The TMA/`mbarrier` pipeline therefore shows no per-pass rate advantage
to build on; adapting the example to a normalized three-product kernel would start from a slower FP16
base than cuBLAS and remains a project-scale effort with at most a few-percent theoretical ceiling
(§3.4). Not pursued.

**3. Fused producer-side A split — preparation is cheap when fused, total still loses.**
`tools/bench/opt5_t09_producer.py`: a single Triton kernel splitting FP32 `A` into FP16 high/low parts
with per-row power-of-two scales takes **174.6 µs**, versus 1,223.8 µs for the unfused PyTorch version
and 172.8 µs for a plain FP32 copy of `A`. A producer emitting high/low parts instead of FP32 `A`
would therefore pay almost nothing beyond the write it already performs (subject to obtaining each
row's scale, which a row-producing kernel has and an elementwise one may not). But split + pre-split
GEMM totals **682.1 µs** against cuBLAS's 503.7 µs, and even with free producer emission the GEMM alone
(507.6 µs) only ties cuBLAS. There is no end-to-end win to collect on this GPU.

**Verdict:** all three follow-ups are negative. The hybrid's problem is not one remaining trick; at
measured efficiencies the FP32 and FP16×3 paths cost the same energy per useful FLOP on `sm_120`, and
every structural variant pays at least as much in staging, instructions, or lost SIMT efficiency as it
recovers. Further GEMM work on this GPU is closed; the normalized tensor path's remaining case is
`sm_80`/`sm_89` hardware, pending the owner's permission for workstation/Modal runs.

The separate **row/column-normalized standalone `3×FP16` path** remains a candidate for `sm_80`/`sm_89`,
not an adopted backend. Its previously recorded local evidence does not qualify another GPU, a complete
force/stress gate or MD. Workstation validation and NAS synchronization wait for the owner's explicit permission.
