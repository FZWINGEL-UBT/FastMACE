# OPT4-T07: single-SM co-execution probe (the megakernel gate)

Optimisation round 4 (`plans/opt-4.md`, candidate 7). This is the gate in
`docs/design/2026-10-23-heterogeneous-megakernel.md` ("The gate before any work"): can CUDA-core FP32 work and
tensor-core MMA work share one SM well enough to pay for an FP32-accurate decomposition? Date: 24 October 2026.
Speed only, untimed in the A/B sense.

**Result: fail on the reference GPU (RTX 5060 Ti, sm_120) and on the A100 (sm_80). The L4 (sm_89) passes
exactly at the threshold, and only as an upper bound with no memory traffic.** The heterogeneous part of the
megakernel idea cannot meet the design note's stage rule, which asks for a gain on the reference GPU.

## The probe

`tools/megakernel_probe.py` (NVRTC through `cuda-bindings`, no nvcc). One kernel, one block per SM:
- warps `0 .. S-1` run FP32 FFMA work (the SIMT side);
- warps `S .. S+M-1` run independent `mma.sync` work: TF32 m16n8k8 or BF16 m16n8k16, FP32 accumulate (the MMA
  side);
- warp `w` sits on scheduler `w % 4`, so every scheduler holds both kinds.

Every warp loops until one common `clock64` deadline (2 M cycles), so the two sides overlap for the whole
window. A side's rate is the sum of its warps' FLOPs per cycle. Each value is the median of 5 launches, and of
the blocks. Each configuration runs in five modes with the same kernel, so the register allocation is the same:
- each side alone with its own warps;
- both sides at once;
- all `S + M` warps on the SIMT side, and all of them on the MMA side.

The variants go from a pure issue test towards a real GEMM's registers and shared-memory traffic:

| Variant | SIMT side | MMA side | Registers (sm_120 / sm_89 / sm_80) |
| --- | --- | --- | ---: |
| `reg` | 16 independent FFMA chains per thread | 8 independent tiles, operands in registers | 48 / 52 / 52 |
| `smem` | SGEMM inner loop: 8×8 thread tile, 4 `ld.shared.v4` per 64 FFMA | 32×32 warp tile, `ldmatrix.x4` fragments, 8 MMAs per k-step | 127 / 123 / 116 |
| `gemm` | as `smem` | 64×64 warp tile, 32 MMAs per k-step, 128 accumulators | 162 / 166 / 166 |
| `x3` | as `smem` | `gemm` with the 3×TF32 split in the kernel (CUTLASS's scheme): each fragment becomes `big = cvt.rna.tf32(x)` and `small = cvt.rna.tf32(x − big)`, then `big·big + big·small + small·big` | 228 / 230 / 230 |

No variant spills. `gemm` and `x3` allow at most 256 threads, so only the 4×4 configuration fits.

**Pass rule** (the design note's, with its thresholds made explicit):
- `keep = simt_both / simt_alone ≥ 0.90`;
- `gain = (simt_both + mma_both / c) / max(simt_best, mma_best / c) ≥ 1.10`.

Here `c` is the number of products the decomposition needs: 3 for 3×TF32, 6 for BF16×6, and 1 for `x3`, whose
MMA rate already counts each product once and pays for its split. The `*_best` rates are the better of the
side's own warps alone and all `S + M` warps on that side. Giving those warps to the CUDA cores is the
alternative to giving them to the tensor cores. With one SIMT warp per scheduler the CUDA cores are
latency-bound (173 of 256 FLOP/cycle on sm_120), and without this baseline their idle issue slots would count as
co-execution gain. The 1.10 is the design note's stage rule (≥ 10 %). BF16×3 is in the JSON but is not
FP32-accurate, so it does not count.

## Results (one block per SM on every SM; the single-block runs agree within 2 %)

FLOP per cycle per SM. `simt` and `mma` are each side alone; `best` is the better of that and all warps on one
side. TC-only is the decomposition on tensor cores alone, `mma_best / c`, relative to `simt_best`.

| GPU | Variant, kind, S×M | simt / best | mma / best | both: simt, mma | keep | gain | TC-only | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 5060 Ti (sm_120) | reg tf32 4×4 | 220 / 241 | 255 / 256 | 191, 168 | 0.87 | 1.02 | 0.35 | fail |
| | reg tf32 8×8 | 241 / 244 | 256 / 256 | 234, 33 | 0.97 | 1.00 | 0.35 | fail |
| | smem tf32 4×4 | 173 / 196 | 250 / 256 | 156, 153 | 0.90 | 1.06 | 0.44 | fail |
| | smem bf16 8×8 | 196 / 204 | 512 / 512 | 182, 198 | 0.93 | 1.05 | 0.42 | fail |
| | gemm tf32 4×4 | 173 / 196 | 254 / 256 | 160, 149 | 0.92 | 1.07 | 0.44 | fail |
| | gemm bf16 4×4 | 173 / 196 | 508 / 512 | 160, 298 | 0.92 | 1.07 | 0.44 | fail |
| | **x3 tf32 4×4** | 173 / 199 | 83 / 84 | 158, 49 | 0.91 | **1.04** | 0.42 | **fail** |
| L4 (sm_89) | reg tf32 4×4 | 237 / 244 | 490 / 511 | 209, 196 | 0.88 | 1.12 | 0.70 | fail (keep) |
| | smem tf32 8×4 | 195 / 201 | 477 / 511 | 178, 131 | 0.91 | 1.10 | 0.85 | pass |
| | smem bf16 4×8 | 179 / 201 | 1022 / 1022 | 149, 458 | 0.83 | 1.12 | 0.85 | fail (keep) |
| | gemm tf32 4×4 | 180 / 200 | 446 / 510 | 168, 175 | 0.93 | 1.13 | 0.85 | pass |
| | gemm bf16 4×4 | 180 / 200 | 892 / 1019 | 168, 349 | 0.93 | 1.13 | 0.85 | pass |
| | **x3 tf32 4×4** | 180 / 202 | 160 / 165 | 165, 57 | 0.92 | **1.10** | 0.82 | **pass, at the threshold** |
| A100 (sm_80) | reg tf32 4×4 | 122 / 127 | 926 / 1004 | 93, 695 | 0.76 | 0.97 | 2.63 | fail |
| | smem tf32 4×8 | 117 / 126 | 902 / 941 | 69, 680 | 0.59 | 0.94 | 2.49 | fail |
| | gemm tf32 4×4 | 116 / 126 | 735 / 951 | 90, 557 | 0.78 | 0.87 | 2.52 | fail |
| | gemm bf16 4×4 | 116 / 126 | 1470 / 1902 | 90, 1113 | 0.78 | 0.87 | 2.52 | fail |
| | **x3 tf32 4×4** | 117 / 126 | 255 / 273 | 68, 182 | 0.58 | **0.92** | 2.17 | **fail** |

Every row is in the JSON (configurations 4×4, 8×4, 4×8 and 8×8; one block and one block per SM; BF16×3).

## What the numbers say

- **sm_120 (RTX 5060 Ti):**
  - TF32 runs at 256 FLOP/cycle/SM, the same as FP32 FFMA; BF16 runs at twice that. A decomposition on the
    tensor cores alone is therefore at most 0.44× of the CUDA cores.
  - Co-executed (`x3`), the SIMT side drops from 173 to 158 and the MMA warps add 49 useful FLOP/cycle: 207,
    against 199 with all eight warps on the CUDA cores.
  - The gain is 1.04–1.07 in every realistic variant: no pass.
- **sm_80 (A100):**
  - The CUDA cores give 128 FLOP/cycle/SM. 3×TF32 on the tensor cores alone (`x3`, split included) gives 255–273,
    2.2× the FP32 rate.
  - Running FFMA warps beside the MMA warps costs the SIMT side 22–42 % of its rate and the MMA side 25–33 %.
  - The best mixture is 8–13 % slower than tensor cores alone. On the A100 the path is all-tensor-core 3×TF32
    (OPT4-T03), not co-execution.
- **sm_89 (L4):**
  - TF32 runs at twice the FP32 rate, so co-execution helps in the probe: keep 0.92–0.93, gain 1.13 as an upper
    bound and 1.10 with the measured split.
  - This is an ideal case. Both sides read the same small, read-only shared-memory tiles. There are no global
    loads, no `cp.async` pipeline, no barriers, no epilogue, and the two sides are perfectly balanced. A real
    kernel adds every one of those to the same issue slots and the same shared-memory bandwidth, so 1.10 is a
    ceiling.
  - The gain also applies only to the GEMM share of a step.
  - OPT-T01b measured the real version on sm_89 (cuBLAS FP32 beside Triton tf32x3 on two streams): 0.77–0.85×.

## Decision

- **Fail on sm_120 and sm_80.** The heterogeneous part of the megakernel (tensor-core decomposition beside
  CUDA-core FP32 in one kernel) is closed for the RTX 5060 Ti and the A100.
- **sm_89** passes only at the threshold and only as a no-traffic ceiling. The design note asks for ≥ 10 %
  whole-step gain "on the reference GPU and on one other production GPU" before any later stage starts. The
  reference GPU is the RTX 5060 Ti (SPEC §18, the 23 October acceptance amendment), and it fails at the kernel level. So stage 2 cannot start on the
  L4 result alone. **Recommendation:** close the heterogeneous part for all three GPUs. A future GPU whose
  tensor-core TF32 rate is several times its FP32 rate needs this probe re-run (one command, about 10 s of GPU
  time).
- This probe does not decide homogeneous persistence (OPT-T05, regional persistence). Its own status in
  `plans/opt-4.md` §2 stands.

## Records

| Machine | GPU (capability, SMs) | Environment | Run |
| --- | --- | --- | --- |
| local PC, WSL2 | RTX 5060 Ti (12.0, 36), driver 610.88 | `~/environments/mace_cueq_latest`, torch 2.14.1+cu130, cuda-bindings | 10:26 UTC, one `local_gpu.py` reservation |
| Colab | L4 (8.9, 58), driver 580.82.07 | hybrid stack `f00d5798f8b5` (`tools/colab`), torch 2.14.1+cu130 | session `fm-opt4t07-l4-3`, 10:53 UTC |
| Colab | A100-SXM4-40GB (8.0, 108), driver 580.82.07 | as above | session `fm-opt4t07-a100-3`, 10:52 UTC |

- FastMACE commit `1f724c9` (branch `fastmace/opt4-T07`). No baseline engine is involved.
- JSON: `results/opt4-T07-megakernel-probe-20261024/{local,colab-l4,colab-a100}/summary.json`, with each Colab
  run's `colab.json`.
- Earlier runs on `27af911`: the first local run and one L4 and one A100 VM. They had the same kernel but no
  all-warps baselines, so their gains were measured against the latency-bound SIMT side. The same rows agree
  within 1 %. Their Colab JSON was lost: the WSL command line expanded `$OUT` before it reached the VM. A second
  L4/A100 pair (`1f724c9`) printed its tables but lost its JSON the same way. The `-3` sessions reran it with the
  command passed through a script file.
- Colab compute units, estimated from VM wall times (A100 5.3 CU/h, L4 about 1.7 CU/h): A100 2,482 s ≈ 3.7 CU, L4
  2,833 s ≈ 1.3 CU, together ≈ 5.0 CU. Two VMs of the second pair sat in hung `colab exec` polls for about
  20 minutes each, and that cost about 3 CU. The ledger's before/after balances are account-wide and overlap
  with other round 4 VMs.

Command (any CUDA GPU with compute capability ≥ 8.0, idle):

```bash
python tools/megakernel_probe.py --label "<machine>" --output <fresh directory>
```
