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

## 3. Records and files

- Probe and Wave-1 microbenchmarks:
  - `tools/megakernel_probe.py`, `docs/validation/opt5-T01-scheduler-probe.md`,
    `results/opt5-T01-T02-probe-20261026/local/summary.json`;
  - `tools/bench/opt5_microbench.py`, `results/opt5-wave1-20261026/summary.json`.
- `OPT5-T05` implementation and validation:
  - `fastmace/kernels/triton/tf32x3_gemm.py` (`uses_tf32x3_tile`),
    `fastmace/kernels/triton/segment_gemm.py`, `tests/gpu/test_tf32x3_gemm_cuda.py`;
  - `results/opt5-T05-20261026/{accuracy,omol-accuracy,gate-NaCl,gate-NbCl5}`.
