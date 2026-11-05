# Kernel review (5–6 November 2026)

Branch `fastmace/kernel-review-20261105`, from main `462751b6` (which contains `cdffe819`).

**Method.** The review covered the H200's real kernels, using the sources the earlier lab runs saved:
- the fused forward at `891bf370`, `results/h200-ideas-20261105/runs/gate-bounds-h200/out/lab-forward/forward.cu`;
- the w′ backward at `7a7b3a76`, `results/research-answers-20261105/runs/hyb-h200/lab-wprime/wprime.cu`;
- the generators, the 3xTF32 GEMMs, the Triton kernels and the rewrites.

Five reviewers (Claude Haiku subagents) each read one area and proposed findings. Every claim below was checked against
the code, the ledger or a measurement before it was kept. Claims that did not hold are listed as refuted, with the
reason.

**Workload.** OMAT-0 medium, float32 hybrid, Benchmark_loup (SrTiO3 10×10×10, 5,000 atoms, 373,760 edges), 1 fs,
skin 0.5 Å.

**Machines.**
- **H200 and A100:** Modal profile `third` (fzwingel54), the hybrid image (recipe C plus OEq), `tools/modal/hopper_sweep.sh`.
- **RTX 5060 Ti:** local, WSL `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130, OEq 0.7.0).

Kernel-lab ratios are a variant's median time over the base kernel's, round-robin in one process, 15 rounds
(`tools/omat0_kernel_lab.py`). MD-step b/a values are from `tools/ab_rewrites.py`, both build orders (ab = abab, ba =
abba), with b = the candidate.

## Outcome

| Result | Class | Verdict |
| --- | --- | --- |
| `fused_forward.SH_LDS4`: the TP's sh reads as float4 shared loads | bitwise | **default on the RTX 5060 Ti** (MD 0.9932 / 0.9902, decision 66); off on the H200 (1.0014 / 1.0027) and the A100-SXM4-80GB (1.0008 / 0.9949) |
| `wprime_backward`: zero `dsh` rows for edges with out-of-range indices | bug fix, bitwise for valid edges | merged into the branch (`e5280113`) |
| H200 section in `docs/current/defaults-by-gpu.md`; DEGREE_PARTS citation | docs | done (`5b4431fe`) |
| Everything else measured | | rejected or dropped (tables below) |

## 1. Fused folded forward (`fused_forward.py`)

The H200 kernel has two parts (DEGREE_PARTS), 12 warps, 168 registers, 175,104 B of shared memory and a 3xTF32 mma.sync
a·W product. Earlier bounds on the H200: no x gathers 0.834, no TP 0.619, no CG coefficients 0.971.

### Findings by expected gain

| # | Finding | Class | Falsifier | Result |
| --- | --- | --- | --- | --- |
| F1 | The TP reads each sh component with its own scalar LDS: 16 per edge, inside the 8-edge batch loop of every part. Reading them as 4 float4 loads (`shq[ED / 4]`) needs the 16-byte alignment that VEC_SH's float4 staging already guarantees. | bitwise | lab `sh-lds4` | RTX 5060 Ti forward 0.972; H200 0.989; outputs bitwise. Built as `SH_LDS4`. |
| F2 | The x gathers form a dependent chain (senders → x) that the product hides only partly. Proposal: prefetch the next batch's x in registers. | bitwise | lab `F-x-nodep`: x at hashed indices, no senders load | RTX 5060 Ti 0.989 against F-no-xload 0.930; H200 1.094 (hashing destroys the senders' locality, 8 B of spill) against F-no-xload 0.900. **Dropped:** the cost is the gathers' traffic, not the dependency, and cp.async prefetch was already rejected. |
| F3 | The C-fragment redistribution (48 STS + 48 LDS per batch) could use float2/float4. | bitwise | lab `F-tc-no-redistribute` | H200 0.998. **Dropped:** at most 0.2 % of the kernel. |
| F4 | Grouping equal CG coefficients (c·(x1e1 + x2e0 − …)). | rounding | existing bound F-tp-no-coef | H200 0.971 at `891bf370`. **Not built:** at most 2.9 % of the kernel, part of which is irreducible. |
| F5 | Coefficients such as `0x1.fffffffffffffp-1f` round to exactly 1.0f; part 0's 16 products multiply by 1. | bitwise | arithmetic | Verified (the float ulp below 1 is 2^-24). NVVM folds `x * 1.0f`, so the code is already free. No action. |
| F6 | Only one block fits per SM (168 × 384 = 64,512 registers; 175 KB of shared memory), yet the grid is sized for 2 per SM. Blocks dispatch x-fastest, so part 0 runs before part 1. | bitwise | launch screen | **Refuted as a speed lever:** each phase fills every SM, part-1 blocks start as part-0 blocks exit, and the BLOCKS_PER_SM screen is neutral (A100 1.000–1.003; H200 1 / 3 / 4 per SM 1.141 / 1.112 / 1.076). |
| F7 | A 2 × 6-warp layout, with the W slice compacted from 2,560 to 2,048 floats per path. | bitwise | warps screen | Not built: same warps per SM as now; tc-warps8 and tc-warps10 already lose. |

**Reviewer claims refuted:**
- "SKIP_ZERO_COLUMN is inert on the tensor-core path" is wrong. Without the skip, K = 65 pads KP to 72, which is 9
  k-steps instead of 8. The H200 measured the skip at 1.092 / 1.090 (b = off).
- "`const float e0` shadows the int64 batch index": true, but harmless. The inner blocks never read the batch `e0`.

**Hazards (not speed).** Receivers are trusted: an unsorted receiver list or one ≥ N would mis-flush, because only
senders are range-checked. The CSR `row_ptr` and the receiver list are assumed consistent. Both are invariants of the
neighbour list; no kernel check was added.

### SH_LDS4 (built)

- **Switch:** `fused_forward.SH_LDS4` (None: `SH_LDS4_GPUS`, empty) with `sh_lds4_default`; `render(..., sh_lds4=True)`.
  It needs `vec_sh` and is refused with tctp or wgmma.
- **GPU test:** `tests/gpu/test_fused_folded_forward_cuda.py` adds `sh lds4 bitwise simt` and `sh lds4 bitwise tc`.
  Both are bitwise against the VEC_SH kernel, in both layouts, at K 9 and 65 (RTX 5060 Ti).

MD step b/a, SrTiO3 10×10×10:

| GPU | ab | ba | Base step |
| --- | ---: | ---: | ---: |
| RTX 5060 Ti (local) | 0.9932 | 0.9902 | 25.2 ms |
| H200 (Modal, `5b4431fe`) | 1.0014 | 1.0027 | 5.99 ms |
| A100-SXM4-80GB (Modal, `5b4431fe`) | 1.0008 | 0.9949 | 10.92 ms |

Lab forward at `5b4431fe`: H200 0.989 (1,686 → 1,667 µs), A100-SXM4-80GB 0.993; bitwise. Default on the RTX 5060 Ti
only (decision 66). On the H200 the 19 µs saved in the kernel does not reach the step; on the A100 the orders
disagree.

**A100 finding:** the A100-SXM4-80GB's base forward compiles with 64 B of local memory (168 registers; sh-lds4 48 B),
while the H200's has none. Its spill was not examined further; a register-pressure check of the A100 tensor-core
forward is open.

## 2. w′ backward (`wprime_backward.py`)

The H200 kernel has HORNER, FACTORED, SCALAR_PATHS, TABLE_W, BLOCK_MESSAGE and BUTTERFLY, 96 registers and one edge per
32-lane warp.

| # | Finding | Class | Falsifier | Result |
| --- | --- | --- | --- | --- |
| W1 | `x`, `sh`, `dx`, `dsh` and others lack `__restrict__`, so loads cannot be reused across the dx atomics. | bitwise (dsh, dr); dx to the atomics' order | lab `restrict` | RTX 5060 Ti 1.016, H200 1.031. **Rejected.** |
| W2 | 16 scalar warp-wide atomics per edge; sm_90 has `atomicAdd(float4*)`. | atomics order | lab `F-no-atomics` (atomics replaced by a never-taken store) | RTX 5060 Ti 0.979, H200 1.080; the explore run measured H200 1.089. **Dropped:** atomics are not on the H200's critical path. |
| W3 | Table loads: one lane reads a 24-byte row as three LDG.64. A lane-quad, term-major float4 layout would halve the instructions. | rounding | `B-gslab` / TABLE_PAIRS evidence | Not built. TABLE_PAIRS (contiguous 256 B per warp load, bitwise) was neutral on the H200 (1.0008 / 1.0003), so the table reads are not wavefront-bound. A re-layout that also changes the channel-to-lane map is a larger change with no bound left to justify it. |
| W4 | Early exit for edges at or beyond r_max. | bitwise up to the sign of zero | arithmetic | Edges are compacted to the cutoff before the force call (`engine/graph.py`, stage C). Only bucket padding is beyond it: on average about half of `edge_bucket_fraction` = 0.05, so ≤ 2.5 % of edges, ≈ 0.5 % of the step at most. Not built. |
| W5 | Warp-uniform `if (inside)` and `if (sloped)` branches in the hot loop. | bitwise | none run | Not built: warp-uniform branches cost one predicate each; no measurable target. |
| W6 | Raise occupancy with a register cap (96 regs, 5 blocks of 128). | exact | caps on A100 | A100 caps 72 / 64 lose 1.20 / 1.48 (spills). Not repeated on the H200. |

**Reviewer claims refuted:**
- "The no-reduction falsifier is confounded by dead-code elimination" is moot here: the REDUX/BUTTERFLY verdicts came
  from MD-step A/Bs, not from that bound.
- "The H200 render multiplies by ±1": the render has no ±1 coefficients.

**Bug fixed (`e5280113`).** An edge whose sender or receiver is out of range returned early after writing `dr = 0`,
but left its `dsh` row unwritten. The wrappers allocate `dsh` with `torch.empty_like`, so the row held whatever was in
memory.
- **New case:** `tests/gpu/test_wprime_backward_cuda.py`, `invalid-sender-{ir_mul,mul_ir}`, 35 edges with sender −1.
- **Before the fix:** garbage up to 3.44 in those rows.
- **After:** zeros, and every valid row bitwise unchanged.
- **Reach:** latent. The engine's padding edges use valid indices beyond the cutoff, so production runs never hit this.
  The vendored OEq kernels (`factorized_projected.cuh`) skip such edges the same way; they are not changed here.
- **dr:** the density gradient is still added to `dr` for every edge, as before.

The test's existing `ir_mul-cublas` dsh check fails on the RTX 5060 Ti without this change (1 of 24,000 elements, the
REDUX default; recorded in `results/wprime-pairs-20261105`). That is why the new case was run on its own locally.

**Hazards.**
- `r = +inf` gives `t = inf`, so the zero table entries make Horner compute 0·inf = NaN in dx, dsh and dr. A NaN r
  gives zero gradients. The reference propagates NaN in both cases. Neither occurs in a valid MD state.
- `int32((clamped − r_lo) · bin_scale)` for huge finite r is an out-of-range conversion. It saturates on NVIDIA
  hardware.

## 3. 3xTF32 GEMMs (`cutlass/`, `triton/tf32x3_gemm.py`)

| # | Finding | Class | Falsifier | Result |
| --- | --- | --- | --- | --- |
| G1 | CUTLASS's FastF32 mainloop splits every B tile into big and small parts in every row block. The weights are constant, so they could be split once at plan time. | bitwise if the MMA order is kept | cubin rebuilt without the B split (timing only) | Not run: needs a new CUTLASS mainloop and a cubin build. The fused forward's analogue, TC_PRESPLIT, lost 1.36 on the A100 (registers). The sm_90 cubin's `fm_tf32x3_rr_64x64` has 48 HMMA, 112 FADD, 109 LOP3 and 142 IMAD (static count over the whole kernel, `cuobjdump -sass`, verified). The FADD and LOP3 are mostly the operand splits, so the split may matter more here than in the forward. **Open, top GEMM candidate.** |
| G2 | SEGMENT_ROWS (no copy glue) is A100-only. | — | — | **Refuted:** already measured on the H200, 1.0006 / 1.0000. |
| G3 | 64×64 tiles leave SMs idle at M = 5,000, N = 128 / 256 (60 % / 80 % wave efficiency on 132 SMs). | bitwise | — | **Refuted:** tile 32×64 measured on the H200, 1.009 / 1.006. |
| G4 | Species-skip GEMMs on the CUTLASS backend (`species_gemm.CUTLASS`). Never measured on the H200. | rounding | MD A/B | H200 1.0141 / 1.0175. **Rejected.** |
| G5 | Per-call Python host overhead in `mm_into`. | bitwise | — | No effect in graph replay, which is the MD path. Relevant only to `engine/compiled.py`. |

**Hazards.**
- `tf32x3.eligible()` does not check M/K/N multiples of 4 (only `takes()` does). All current callers go through
  `takes()`.
- `segment_rows` caches `torch.stack(weights)` keyed by `data_ptr()`. It is stale if weights change in place, and is
  dormant because SEGMENT_GROUPED is off.
- `torch.mm` fallbacks inherit `allow_tf32`.

## 4. Triton kernels

| # | Finding | Class | Falsifier | Result |
| --- | --- | --- | --- | --- |
| T1 | The tabled kernels run 1 warp on every GPU (`tabled_outer_sum.WARPS`, `FORWARD_WARPS`). | rounding (backward) | MD A/B, 2 warps | H200: forward 1.0147 / 1.0168, backward 1.0270 / 1.0295. **Rejected:** 1 warp is right. The reviewer's claim that the H200's 8-warp `radial_table.WARPS_GPUS` "never reaches the kernel that runs" is wrong: `radial_table.forward` still runs for interaction 2, and its A/B was 0.9946 / 0.9950. |
| T2 | K = 65 pads the tabled kernels' register tile to 128 lanes (63 masked), and the density column runs as a scalar Clenshaw chain. | bitwise | time at K = 64 against 65 | Not run. Open; the tabled backward is about 240 µs (4 % of the H200 step). |
| T3 | `edge_sh3` backward: 15 scalar strided loads per edge instead of one [BLOCK, 16] tile. | bitwise | kernel time | Not run; the kernel is about 0.1–0.3 % of the step. |
| T4 | Zero-filled outputs before atomics (`torch.zeros((slots, d·k))`). | — | — | Needed by the atomics. |
| T5 | No eviction hints in any Triton kernel. | bitwise | — | Not run; hints in the CUDA kernels were neutral or slower (omat0-ff-hints, omat0-stream-w). |

**Refuted:** "a NaN r makes the bin index negative" (`radial_table`, `tabled_outer_sum`). `tl.maximum(r, r_lo)`
compiles to `max.f32`, which returns the non-NaN operand, so the index is 0.

**Latent:** `edge_sh.py`'s int32 `offs * 16` overflows at E ≥ 2^27, about 134 M edges in one call. The tiled planner's
per-tile edge cap keeps it unreachable.

## 5. Rewrites, energy, neighbour list, integrator

The rewrites reviewer listed copy glue in `BlockLinear` (H200: SEGMENT_ROWS measured neutral), the pipelined step's
host wait on the flags, int64 compaction arrays and redundant zero tensors in the energy sum. Each is under 0.5 %, or
already measured neutral. None was built.

## 6. Switch audit (per-GPU defaults against the recorded measurements)

Every `*_GPUS` set was compared with `docs/current/defaults-by-gpu.md`, the ledger and the H200 result folders.

| Item | Finding | Verdict |
| --- | --- | --- |
| H200 coverage | `defaults-by-gpu.md` had no H200 entry, while 23 H200 sets were on in code. | Fixed: an H200 section with every b/a. |
| H200 precision switches | `fused_forward.TENSOR_CORES` (3xTF32 a·W) and the planner's `TF32X3_OMAT0_GPUS` include the NVIDIA H200 on measurements (1.107 / 1.105, 1.068 / 1.069) and a passing gate. On the A100 the owner approved both precision changes explicitly; no H200 approval is recorded. | **Owner decision needed.** Not changed. |
| DEGREE_PARTS | The comment cited `results/degree-parts-20261105`, which does not exist. | Fixed: cites `results/h200-ideas-20261105` (idea 2). |
| TABLED_FORWARD on the H200 | `results/h200-defaults-20261105` says it stays off; the code has it on. | Consistent: the later run after the recapture fix (`results/h200-wgmma-20261105`) measured 0.9838 / 0.9871. The ba JSON's inner ratio 1.0131 is the same measurement indexed by build slot (1 / 0.9871). |
| SKIP_ZERO_COLUMN on Ada and A100 | Single figures (0.981, 0.986), not both build orders. | Bitwise, and the H200's both-order A/B is large (1.092 / 1.090). No action. |
| `grouped_launch` (5060 Ti), `gate_epilogue` (5060 Ti, Ada) | On with mixed cells: NbCl5 1.002 / 0.998, NaCl 1.000 / 1.005. | Policy question for the owner: per cell or in aggregate. |

Every other set matches its recorded measurement. GPU names are consistent; `GPUModels` covers the A100 spellings.

## Runs

`results/kernel-review-20261105/` (README there).
