# Optimisation round 5: simultaneous CUDA-core and Tensor-core execution (26 October 2026)

Plan: `plans/opt-5.md`. Baseline: `v1.0.0` (`6ad8bfaf`). Machine: local reference PC (`Fabi-PC`), NVIDIA GeForce
RTX 5060 Ti (`sm_120`, 36 SMs, 16 GiB), WSL2 Ubuntu 24.04, driver 610.88, `~/environments/mace_cueq_oeq` and
`~/environments/mace_cueq_latest` (`torch 2.14.1+cu130`, `triton 3.8.0`).

## 1. Summary of tasks and verdicts

| Task | Candidate | Falsifier / Gate | Measured result | Verdict |
| --- | --- | --- | --- | --- |
| **OPT5-T01** | Tacker-style warp-scheduler isolation (`sched_iso`) and asymmetric warp ratios (`6×2`, `7×1`, `2×6`) in `tools/megakernel_probe.py` | `keep ≥ 0.90` and `gain ≥ 1.10` on `sm_120` | `sched_iso` caps both units at the sub-partition fraction (`gain = 0.72` on `4×4`, `0.86` on `6×2`). Asymmetric interleaved ratios reach `keep = 0.97–0.99` but `gain = 1.01–1.03` (vs `4×4`'s `1.03–1.05`). | **Closed on `sm_120`**: each SM sub-partition owns a private FFMA pipeline and a private Tensor Core unit; scheduler isolation starves the unused unit per sub-partition (`docs/validation/opt5-T01-scheduler-probe.md`). |
| **OPT5-T02** | DeepSeek-style SIMT-split producer + pure-MMA consumer (`x3_pipe` in `tools/megakernel_probe.py`, non-blocking `bar.arrive 1`) and `cvt.rna.tf32` PTX split | `≥ 15 %` higher concurrent 3×TF32 MMA rate in probe; beat cuBLAS IEEE (`229.6 µs`) on NbCl5 `[5832, 512] @ [512, 512]` | In the probe, offloading `split_tf32` to SIMT warps cuts registers `205 → 194` (0 spill) and raises concurrent 3×TF32 MMA throughput by **+35.6 %** (`45.2 → 61.3` FLOP/cycle/SM, `mma_share` `0.56 → 0.76` at `4×4`) and **+21.3 %** (`60.1 → 72.9`, `mma_share` `0.73 → 0.89` at `2×6`). On `sm_120` (where TF32 rate = FP32 rate), the SIMT producer cost reduces `s_both` (`gain = 0.85`), and on `[5832, 512, 512]` Triton 3×TF32 takes `497.5 µs` vs cuBLAS IEEE's `229.6 µs`. | **Mechanism confirmed for `sm_80` (`+21–41 %` MMA utilization); cut on `sm_120`**. Note: inline PTX `cvt.rna.tf32` before `tl.dot(..., input_precision="tf32")` is doubly rounded by Triton (`543.2 µs`, worse error); built-in `tl.dot(..., input_precision="tf32x3")` (`1.03e-7` cancellation error vs IEEE's `4.87e-7`) stays the kernel primitive. |
| **OPT5-T03** | Radial folded last layer (`[E, 65] @ [65, 1280]` and `K = 68`) `64 + tail` hybrid/streaming Triton kernel | `≤ 150 / 1,000 / 1,450 µs` on `NaCl` ($E=8,192$), `NbCl5` ($56,320$), `NaNbO3` ($82,688$) | Best Triton `64 + 1` with `.cs` streaming stores (`evict_first`): `160.7 / 1,440.3 / 2,435.8 µs` vs cuBLAS `K = 65` non-contiguous `199.8 / 1,339.4 / 1,984.1 µs` and cuBLAS `K = 64` floor `150.3 / 1,052.6 / 1,546.3 µs`. | **Cut at Wave 1 falsifier**: even cuBLAS at `K = 64` (`1,053 / 1,546 µs`) is above the `1,000 / 1,450 µs` targets, and Triton's IEEE `tl.dot` is compute-limited (`OPT4-T01` confirmed). |
| **OPT5-T04** | Complementary operator fusion: `gated_segment_gemm` (gate scaling in `segment_gemm` prologue) | `≥ 5 %` faster than sequential gate + `segment_gemm` on `NbCl5` (`atoms = 1,944`) | Forward-only CUDA-graph replay on `NbCl5`: `184.33 → 177.44 µs` (`0.9626×`, **−3.7 %**, bitwise equal `max_abs_diff = 0.0`). On `NaCl` (`atoms = 216`, where `segment_gemm` uses cuBLAS): `44.38 → 35.40 µs` in isolation, but full-step `OPT-T02` stage C regressed `NbCl5` by `+1.6–2.0 %` because backward still needs the gated rows. | **Cut at Wave 1 falsifier** (`−3.7 % < 5 %` on `NbCl5`, and backward materialisation remains required). |
| **OPT5-T05** | Libra/HC-SpMM hybrid shape dispatcher (`uses_tf32x3_tile(k, n)` in `tf32x3_gemm.py` and `segment_gemm.py`) | Eliminate the `2.0–6.9×` sub-tile accuracy spikes of `OPT4-T03`, reduce the systematic energy offset, and pass the force/stress gate on `MH-1` (`NaCl`, `NbCl5`) and `OMOL-0 XL` | Keeping sub-tile products ($\min(K, N) < 32$, i.e. $K \in \{6, 16\}$ species inputs and $N \in \{1, 6\}$ readout heads, $< 0.4\%$ of FLOPs) on SIMT IEEE FP32 while routing $\min(K, N) \ge 32$ to 3×TF32 cuts worst per-product cancellation-scaled error ratio `tf32x3 / ieee` from **6.90× → 1.73× (`NaCl`)**, **6.90× → 1.52× (`NbCl5`)**, and **2.74× → 1.30× (`OMOL-0 XL`)**, and lowers the `MH-1` gate energy error by **31–39 %** (`ratio 0.45 → 0.28 / 0.32`). | **Adopted inside opt-in `tf32x3_gemms`**; CPU (8/8) and GPU (8/8) unit tests and `NaCl`/`NbCl5` force gates pass. |
| **OPT5-T06** | Exponent-shifted `3×FP16` (`big = fp16(x)`, `small = fp16((x - fp32(big)) * 2048)`, `c = 3` at `512` FLOP/cycle/SM 16-bit Tensor Core rate) in `megakernel_probe.py` and standalone Triton kernel | Pass single-SM co-execution gate (`keep ≥ 0.90, gain ≥ 1.10`) on `sm_120`; compare standalone kernel speed vs `triton_tf32x3` and `cublas_ieee` | **Passes the single-SM co-execution gate on `sm_120`:** `x3 inter fp16 4×4` (with in-kernel `split_fp16x2`) achieves **`keep = 0.92`, `gain = 1.22×`** (`157.7` SIMT + `82.8` useful `3×FP16` = `240.5` vs `196.8` FLOP/cycle/SM), and `gemm` upper bound reaches **`gain = 1.28×`**. Standalone Triton `triton_fp16x3` is **1.78–1.88× faster than `triton_tf32x3`** on all production shapes (`77.5 µs` vs `142.0 µs` on NaCl `1512×512×512`, matching `cublas_ieee`'s `77.5 µs`!), and end-to-end `MH-1 NaCl` energy/force differences vs default FP32 are `3.77e-5 eV` and `4.14e-7 eV/Å`. | **First `sm_120` co-execution gate pass (`gain = 1.22×`); standalone Triton kernel is `1.00–1.25×` of cuBLAS IEEE on `sm_120` (`1.8×` faster than `tf32x3`)**. Two-stream column split (`241.3 µs` vs `227.9 µs` on `NbCl5 5832×512×512`) still pays stream-join overhead; realizing the `1.22×` single-SM gain in production requires a fused persistent SIMT+`3×FP16` C++/CUTLASS kernel. |
| **OPT5-T07** | Time-step delta GEMM ($Y_t = Y_{t-1} + \Delta X_t W$ across `1 fs` steps) and 50/50 channel-magnitude split (CUDA cores IEEE FP32 on top-norm channels + `1×TF32` or `2×FP16` Tensor Cores on tail/delta) | Cancellation-scaled error $\le 2\times$ IEEE FP32 on real `MH-1` force-call operands | Across a `1 fs` 300 K step (`Δr ≈ 0.006 Å`), intermediate equivariant features change by `‖ΔX‖ / ‖X‖ = 20.8 %` (median across 52 products), making `1×TF32` delta GEMM `1,838×` (median) to `6,130×` (max) worse than IEEE FP32, and `2×FP16` delta GEMM `482×` (median) worse. In 50/50 channel-norm splits, MACE's weight norms are nearly uniform across channels, giving `1,343×` (`1×TF32` tail) and `411×` (`2×FP16` tail) worse error than IEEE. | **Rejected (falsified by accuracy on real MACE operands)**: equivariant message features rotate/change by ~21 % per `1 fs` step and have flat channel norms, so neither `c = 1` delta nor `c = 2` channel-tail shortcuts preserve FP32 accuracy. |

## 2. Detailed evidence for `OPT5-T05` (`uses_tf32x3_tile`)

In `OPT4-T03` (`docs/validation/opt4-T03.md` §3–§4), routing all 52 `MH-1` (and 106 `OMOL-0 XL`) atom GEMMs
unconditionally through `tl.dot(..., input_precision="tf32x3")` produced two artefacts on sub-tile matrices
where $K < 32$ or $N < 32$ (the Tensor Core tile is $64 \times 64 \times 32$):
1. Products with $K = 6$ (species-sized inputs) or $K = 16$ were up to **6.90×** (`MH-1`) and **2.74×**
   (`OMOL-0 XL`) less accurate than cuBLAS IEEE FP32, and on the A100 (`OPT-T01 §5`) sub-tile $N = 1$ products
   were also slower on 3×TF32 than on cuBLAS SIMT FP32.
2. The readout products ($N = 6$ heads on `MH-1`, $N = 1$ on `OMOL-0 XL`) accumulated Triton's truncated
   small-part split directly into the total energy, creating a systematic $+2\text{–}3\times 10^{-4}$ meV/atom
   energy offset (`gate` energy ratio `0.45` vs default's `0.04–0.19`).

`uses_tf32x3_tile(k, n)` (`int(k) >= tile_rule.BLOCK_K and int(n) >= tile_rule.BLOCK_K`, i.e. $\min(K, N) \ge 32$)
applies the Libra / HC-SpMM hybrid dispatch principle with a closed-form tile-completeness rule (no lookup
tables): products that fill at least one $32 \times 32$ Tensor Core tile along $K$ and $N$ run on 3×TF32
Tensor Cores; sub-tile products ($\min(K, N) < 32$, $< 0.4\%$ of atom-GEMM FLOPs) run on SIMT IEEE FP32
(`torch.mm(a.contiguous(), b.contiguous(), out=out)`).

### Real-operand accuracy (`tools/tf32x3_accuracy.py`, `results/opt5-T05-20261026/`)

| Structure & Model | Metric | Pure `tf32x3` (`OPT4-T03`) | Hybrid `uses_tf32x3_tile` (`OPT5-T05`) | cuBLAS IEEE FP32 |
| --- | --- | ---: | ---: | ---: |
| `MH-1 NaCl` (52 products) | Worst cancellation-scaled error | 5.18×10⁻⁷ | 5.59×10⁻⁷ | 7.17×10⁻⁷ |
| `MH-1 NaCl` (52 products) | `tf32x3 / ieee` median, **max** (cancellation) | 0.60, **6.90×** | 0.59, **1.73×** | 1.00× |
| `MH-1 NaCl` (52 products) | `tf32x3 / ieee` **max** (`max_rel` / `frob_rel`) | **6.73× / 5.87×** | **1.80× / 1.49×** | 1.00× |
| `MH-1 NaCl` (52 products) | Products worse than IEEE | 9 of 52 | **6 of 52** | 0 |
| `MH-1 NbCl5` (52 products) | Worst cancellation-scaled error | 4.66×10⁻⁷ | 5.13×10⁻⁷ | 1.02×10⁻⁶ |
| `MH-1 NbCl5` (52 products) | `tf32x3 / ieee` median, **max** (cancellation) | 0.31, **6.90×** | 0.33, **1.52×** | 1.00× |
| `MH-1 NbCl5` (52 products) | `tf32x3 / ieee` **max** (`max_rel` / `frob_rel`) | **6.73× / 5.87×** | **1.31× / 1.47×** | 1.00× |
| `MH-1 NbCl5` (52 products) | Products worse than IEEE | 7 of 52 | **4 of 52** | 0 |
| `OMOL-0 XL NaCl` (106 products) | Worst cancellation-scaled error | 5.71×10⁻⁷ | 5.71×10⁻⁷ | 1.04×10⁻⁶ |
| `OMOL-0 XL NaCl` (106 products) | Worst `max_rel` error | 1.20×10⁻⁶ | **9.64×10⁻⁷** | 1.23×10⁻⁶ |
| `OMOL-0 XL NaCl` (106 products) | `tf32x3 / ieee` median, **max** (cancellation) | 0.68, **2.74×** | 0.68, **1.30×** | 1.00× |
| `OMOL-0 XL NaCl` (106 products) | Products worse than IEEE | 29 of 106 | **26 of 106** | 0 |

### Force and stress gate (`tools/force_stress_gate.py --rewrites tf32x3_gemms=on`, `results/opt5-T05-20261026/`)

| Structure | Scheme | Energy error (meV/atom, ratio vs `ase_f32`) | Force error (eV/Å, ratio vs `ase_f32`) | Stress error (eV/Å³, ratio vs `ase_f32`) | Verdict |
| --- | --- | --- | --- | --- | --- |
| `NaCl` (216 atoms) | Pure `tf32x3` (`OPT4-T03`) | 3.5×10⁻⁴ (0.45) | 4.20×10⁻⁶ (1.09) | 3.4×10⁻⁸ (0.38) | pass |
| `NaCl` (216 atoms) | **Hybrid `OPT5-T05`** | **2.15×10⁻⁴ (0.28, −39 %)** | 4.47×10⁻⁶ (1.18) | 3.32×10⁻⁸ (0.36) | **pass** |
| `NbCl5` (1,944 atoms) | Pure `tf32x3` (`OPT4-T03`) | 3.3×10⁻⁴ (0.45) | 8.05×10⁻⁵ (1.00) | 7.8×10⁻⁸ (0.79) | pass |
| `NbCl5` (1,944 atoms) | **Hybrid `OPT5-T05`** | **2.27×10⁻⁴ (0.32, −31 %)** | **8.00×10⁻⁵ (1.00)** | 7.42×10⁻⁸ (0.76) | **pass** |


## 3. Detailed evidence for `OPT5-T06` (Exponent-Shifted `3×FP16`) and `OPT5-T07` (Delta / Channel-Split GEMMs)

### 3.1 Why Exponent-Shifted `3×FP16` passes the `sm_120` co-execution gate (`OPT5-T06a`)

On `sm_120` (RTX 5060 Ti), TF32 Tensor Cores run at `256` FLOP/cycle/SM (`1×` FP32 CUDA cores), while 16-bit
(BF16/FP16) Tensor Cores run at `512` FLOP/cycle/SM (`2×` FP32 CUDA cores). Previously (`OPT4-T07`), only BF16
(7-bit mantissa, requiring `BF16×6`, `c = 6`) was tested at the `512` FLOP/cycle/SM rate, yielding the same
`512 / 6 = 85.3` FLOP/cycle/SM ceiling as `3×TF32` (`256 / 3 = 85.3`).

IEEE FP16 has **10 explicit mantissa bits** (11 with the hidden bit)—identical to TF32's mantissa precision—and
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` accumulates in full FP32. Shifting the remainder by
$2^{11} = 2048.0$ on the CUDA cores before converting to FP16:
$$\text{big} = \text{fp16}(x), \qquad \text{small} = \text{fp16}\big((x - \text{fp32}(\text{big})) \times 2048.0\big)$$
aligns `small`'s FP16 exponent with `big`'s so `small` never underflows FP16, requiring only **3 FP16 MMAs
(`c = 3`)** at the `512` FLOP/cycle/SM rate (`170.7` useful FLOP/cycle/SM ceiling, **2× `3×TF32`**).

Measured on `sm_120` (`tools/megakernel_probe.py --kinds fp16`, `results/opt5-T06-fp16x3-20261026/probe/summary.json`):

| Variant (`fp16`, `c = 3`) | `S×M` | Regs | `simt` / `s_all` | `mma` / `m_all` | `both`: `simt`, `mma` | `keep` | `mma%` | `gain` | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smem` | `8×8` | 127 | 199.4 / 207.2 | 507.8 / 508.5 | 183.1, 186.2 | **0.92** | 0.37 | **1.18** | **PASS** |
| `gemm` (upper bound) | `4×4` | 162 | 172.8 / 195.8 | 480.9 / 510.4 | 159.5, 275.8 | **0.92** | 0.57 | **1.28** | **PASS** |
| `gemm` (upper bound) | `6×2` | 162 | 183.9 / 195.8 | 239.9 / 510.3 | 178.3, 129.5 | **0.97** | 0.54 | **1.13** | **PASS** |
| **`x3` (split in kernel)** | `4×4` | 238 | 171.6 / 196.8 | 156.4 / 165.5 | **157.7, 82.8** | **0.92** | 0.53 | **1.22** | **PASS** |
| **`x3` (split in kernel)** | `6×2` | 238 | 183.7 / 196.8 | 78.2 / 165.4 | **177.6, 39.6** | **0.97** | 0.51 | **1.10** | **PASS** |
| `x3_pipe` (SIMT producer) | `4×4` | 194 | 161.1 / 190.2 | 156.4 / 165.5 | 102.4, **119.6** | 0.64 | **0.76** | **1.17** | fail (`keep`) |

### 3.2 Standalone Triton `3×FP16` kernel speed and accuracy (`OPT5-T06b`, `results/opt5-T06-fp16x3-20261026/summary.json`)

With tile-level IEEE exponent bit-masking (`tl.max(tl.abs(a).to(tl.int32, bitcast=True)) & 0x7F800000`),
`triton_fp16x3_kernel` (`tools/bench/opt5_t06_t07_bench.py`) is **1.78–1.88× faster than `triton_tf32x3`** on
`sm_120` and more accurate than both `cublas_ieee` and `triton_tf32x3`:

| Shape (`[M, K, N]`) | `cublas_ieee` (µs) | `triton_tf32x3` (µs) | **`triton_fp16x3` (µs)** | Speedup vs `tf32x3` | `cublas_ieee` `max_rel` / `cancel` | `triton_tf32x3` `max_rel` / `cancel` | **`triton_fp16x3` `max_rel` / `cancel`** |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `NaCl [1512, 512, 512]` | 77.46 | 142.03 | **77.48** | **1.83×** | 3.41e-7 / 1.26e-7 | 2.29e-7 / 8.19e-8 | **2.02e-7 / 7.50e-8** |
| `NbCl5 [5832, 512, 512]` | 227.90 | 491.57 | **276.50** | **1.78×** | 9.11e-7 / 3.37e-7 | 2.38e-7 / 8.86e-8 | **2.13e-7 / 8.40e-8** |
| `NbCl5 [13608, 512, 512]` | 490.95 | 1,128.96 | **612.30** | **1.84×** | 9.37e-7 / 3.80e-7 | 2.37e-7 / 8.71e-8 | **2.17e-7 / 8.54e-8** |
| `OMOL [1944, 1024, 1024]` | 324.33 | 664.62 | **354.25** | **1.88×** | 8.00e-7 / 2.02e-7 | 2.66e-7 / 6.88e-8 | **2.61e-7 / 6.81e-8** |

- On a full `MH-1 NaCl` 3×3×3 force call (`52` products), replacing all full-tile atom GEMMs with
  `triton_fp16x3_kernel` gives:
  - Energy difference vs default FP32: **`3.77×10⁻⁵ eV`** (smaller than `tf32x3`'s `4.36×10⁻⁵ eV`, because
    `cvt.rn.f16` rounds to nearest-even rather than truncating);
  - Max force component difference vs default FP32: **`4.14×10⁻⁷ eV/Å`** (vs `tf32x3`'s `3.73×10⁻⁷ eV/Å`, 9×
    below the float32 calculator's `3.8×10⁻⁶ eV/Å` error);
  - Median cancellation-scaled error ratio vs cuBLAS IEEE: **`0.594×`** (more accurate than IEEE on 45 of 52
    products).
  - With **per-row ($A$) and per-column ($B$) exponent normalization (`triton_fp16x3_rowcol_kernel`, `OPT5-T08`)**,
    the 1 sparse species-row subnormal outlier is eliminated completely: worst cancellation-scaled error across
    all 52 real `MH-1 NaCl` products is **`5.02×10⁻⁷`** (below `cuBLAS IEEE`'s `6.44×10⁻⁷` and `tf32x3`'s
    `5.59×10⁻⁷`), with median ratio **`0.568×`**, max ratio **`1.519×`** (vs `tf32x3`'s `1.727×`), and only **3
    of 52 products** above IEEE (`results/opt5-T08-hybrid-gemm-20261026/summary.json`).
  - **Fused `64 SIMT + 32 MMA` NVRTC kernel (`hybrid_gemm_64_32`, `OPT5-T08`):** 4 SIMT warps (`64×64` FP32) + 4
    MMA warps (`64×32` `3×FP16`) sharing one `64×16` $A$ tile (`127` regs, `0` spill, 2 blocks/SM) runs in
    `80.08 µs` on `NaCl [1512, 512, 512]` (vs `cuBLAS IEEE` `73.07 µs` and `triton_tf32x3` `142.03 µs`) and
    `295.32 µs` on `NbCl5 [5832, 512, 512]` (vs `cuBLAS IEEE` `241.95 µs`), with `max_rel = 9.11×10⁻⁷` matching
    `cuBLAS IEEE`. See [docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md](../design/2026-10-26-fp16x3-and-hybrid-gemm.md).

### 3.3 Why Time-Step Delta GEMM and Channel-Magnitude Split fail (`OPT5-T07`)

Tested on the 52 real `MH-1 NaCl` products across two consecutive `1 fs` 300 K steps (`Δr ≈ 0.006 Å`,
`results/opt5-T06-fp16x3-20261026/summary.json`):
- **Time-step delta GEMM ($Y_1 = Y_0 + (A_1 - A_0) B$):** even for a $0.006\text{ \AA}$ step, intermediate
  equivariant spherical-harmonic and message features change by **`‖A₁ − A₀‖ / ‖A₁‖ = 20.8 %` (median)**, not
  $1\%$. Multiplying $(A_1 - A_0) B$ in `1×TF32` (`c = 1`) produces a median error **1,838× worse than IEEE
  FP32** (max **6,130×**); `2×FP16` (`c = 2`) is **482× worse** (median).
- **Channel-magnitude split (top 50 % of $K$ by weight norm in IEEE FP32 on CUDA cores + bottom 50 % in
  `1×TF32` or `2×FP16` on Tensor Cores):** MACE's equivariant linear weights have nearly uniform row norms, so
  the bottom 50 % of channels still carry ~40 % of the product norm. `1×TF32` on the tail is **1,343× worse than
  IEEE** (median), and `2×FP16` is **411× worse**.

### 3.4 Larger hybrid GEMMs: screen completed, candidates rejected (`OPT5-T08`)

The owner's larger-system follow-up is recorded in
[the design report, §3.3](../design/2026-10-26-fp16x3-and-hybrid-gemm.md#33-larger-tiles-and-pre-split-b-not-qualified-26-october-2026).
It tested pre-split B, row/column partitions, larger tiles, aligned vector loads and asynchronous/staged B,
with the old kernels and cuBLAS in the same alternating-order process. The final screen includes
32,768-row and larger 1,024-channel products, packing costs, boundary/replay checks and real MH-1 operands.

Boundary indexing, repeatability and graph replay passed. The candidates did **not** establish a
general speed win and failed cancellation-scaled real-operand and wide-exponent-range accuracy checks.
No production code or default changed; no hybrid force/stress or MD qualification is claimed.
The normalized standalone `3×FP16` candidate remains separate and unqualified on the target datacenter GPUs.
All work stayed on the local PC; NAS sync and workstation jobs remain paused by the owner's instruction.

## 4. Records and files

- Probe and Wave-1 microbenchmarks:
  - `tools/megakernel_probe.py`, `docs/validation/opt5-T01-scheduler-probe.md`,
    `results/opt5-T01-T02-probe-20261026/local/summary.json`;
  - `tools/bench/opt5_microbench.py`, `results/opt5-wave1-20261026/summary.json`.
- `OPT5-T05` implementation and validation:
  - `fastmace/kernels/triton/tf32x3_gemm.py` (`uses_tf32x3_tile`),
    `fastmace/kernels/triton/segment_gemm.py`, `tests/gpu/test_tf32x3_gemm_cuda.py`;
  - `results/opt5-T05-20261026/{accuracy,omol-accuracy,gate-NaCl,gate-NbCl5}`.
- `OPT5-T06`, `OPT5-T07` & `OPT5-T08` (`3×FP16` and fused `SIMT + 3×FP16`):
  - `docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md`;
  - `tools/bench/opt5_t06_t07_bench.py`, `results/opt5-T06-fp16x3-20261026/`;
  - `tools/bench/opt5_t08_hybrid_gemm.py`, `results/opt5-T08-hybrid-gemm-20261026/summary.json`.
  - `tools/bench/opt5_t08_large_gemm.py`, `results/opt5-T08-large-20261026/` (six screens and final validation).
