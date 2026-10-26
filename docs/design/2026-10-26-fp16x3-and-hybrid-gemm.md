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

**Why the pipelined NVRTC kernel beats `cuBLAS IEEE` on `NaCl` (−1 to −9 %) but stays 4–7 % behind on `5832` / `1944`:**
- Keeping 2 blocks (16 warps) per SM to hide `__syncthreads()` latency caps each thread at `128` registers (`32`
  accumulators per SIMT thread = `8×4` thread tile, `64×128` block tile).
- On `NaCl [1512, 512, 512]`, the added `3×FP16` Tensor Core throughput outweighs the smaller `64×128` block
  tile, beating `cuBLAS IEEE FP32` (`67.67–72.10 µs` vs `73.75–74.37 µs`).
- On very large $M$ (`5832` and `13608`), `cuBLAS`'s pure-SIMT `128×128` block tile (`8×8 = 64` accumulators per
  thread) achieves 2× higher global-memory $B$-tile reuse and 1.5× higher FMA-per-shared-load ratio than a
  `64×128` tile, giving `cuBLAS` a 4–7 % edge on `5832` (`234` vs `252 µs`) and `1944×1024×1024` (`299` vs
  `311 µs`).
- **Actionable conclusion:**
  1. On `sm_120` (RTX 5060 Ti), `cuBLAS IEEE FP32` stays the default because the small gain on `NaCl` (`~6 µs`
     per layer) is offset by the 4–7 % loss on `NbCl5`.
  2. On `sm_89` (RTX 4000 Ada workstations `leto`/`murphy` and Modal L4) and `sm_80` (A100), where 16-bit FP16
     Tensor Cores are **4× to 16× faster than FP32 SIMT** and **2× faster than TF32**, standalone `triton_fp16x3`
     (Path A, **1.78–1.88× faster than `triton_tf32x3`** with better-than-`tf32x3` accuracy across all 52 real
     `MH-1` products) is ready to benchmark and enable.