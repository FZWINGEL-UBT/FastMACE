# A megakernel for MACE-POLAR-1 on the RTX 5060 Ti: bound and scope (7 November 2026)

Status: design note, step 1 of the owner's request "build one megakernel for the local gpu for mace polar".
No kernel is adopted. Branch `fastmace/polar-megakernel` (from `main` 72852da3).

## 1. The profile

Source: the off-polar-alex session's local profiles, WSL `~/fm-runs/detailed-profiles-20261107/`
(RTX 5060 Ti, sm_120, 36 SMs, 32 MB L2, 99 KB shared memory per block, 64 K registers per SM;
`mace_cueq_oeq_polar`, hybrid backend, float32, CUDA graphs, `SPIN_BATCHED` on; code of
`fastmace/off-polar-alex` at the time). `step.log` gives the captured MD step; `module/sources.csv`
gives kernel time per launch, op and shape. Categories below are mine (script in §6).

| POLAR-1 | structure (atoms) | wall ms/step | kernels/step | idle gaps | wide GEMMs (K, N ≥ 256) | narrow GEMMs | splitK reduce + memset | pointwise, reductions | conv, radial, TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | NaCl (216) | 2.93 | 602 | 0.39 ms (12.4 %) | 40 % (36 launches) | 16 % (125) | 4.6 % (117) | 13 % (295) | 4 % |
| M | NaCl (216) | 5.73 | 858 | 0.55 ms (9.3 %) | 47 % (84) | 15 % (163) | 3.2 % (175) | 14 % (393) | 12 % |
| L | NaCl (216) | 11.96 | 1079 | 0.78 ms (6.4 %) | 45 % (144) | 12 % (178) | 1.9 % (236) | 14 % (466) | 20 % |
| M | NbCl5 (1944) | 41.5 | 813 | 0.61 ms (1.5 %) | 33 % + grouped GEMMs | 15 % | 0.2 % | 37 % (incl. grouped GEMMs, k-space cos/sin) | 17 % |

(Percentages are of the traced span. The module profile and the step profile are separate runs, so rows need
not add to 100 %.)

What dominates:

- **Wide GEMMs.** The e3nn linears at 512 channels, one per irrep (`[216·(2l+1), 512] @ [512, 512]`), forward and
  input gradient. `[648, 512] @ [512, 512]` runs in 17.5 µs = 19.4 TFLOPS, about 82 % of the card's FP32 peak
  (4608 lanes × 2 × ~2.6 GHz ≈ 23.7 TFLOPS). These are compute-bound and near peak.
- **Narrow GEMMs and their splitK companions.** About 290 launches on M NaCl. Many have tiny output widths: the
  long-range source heads `[·, 512] @ [512, 1 or 2]` and their gradients `[·, 2] @ [2, 512]`. Also the k-space sums
  `[243 k, 216 atoms] @ [216, 4…16]` and their transposes, and the radial last layer `[7680, 68] @ [68, 1280]`.
- **Gaps.** Graph replay leaves about 0.65 µs per kernel. That is 9–12 % of the step on 216-atom cells and under
  2 % on NbCl5.
- On NbCl5 the k-space part grows with the cell volume (2989 k-vectors × 1944 atoms). The `cos/sin` pointwise
  kernel alone takes 0.57 ms, and the `[2989, 1944] @ [1944, 8|16]` products take about 0.76 ms.

## 2. The bound

A complete megakernel can save only launch gaps, launch tails and the round trips of intermediates through L2 and
DRAM. It cannot do the arithmetic faster than the kernels it replaces. So:

1. **The wide GEMMs must stay as fast as cuBLAS.** No in-kernel FP32 GEMM primitive on sm_120 measured so far
   matches it:
   - cuBLASDx block GEMM: 1.062× slower ([opt-5](../../validation/speed/opt-5.md) §3.12).
   - Triton IEEE `tl.dot`: 7.2–7.5 TFLOPS against cuBLAS's ~19 (OPT4-T01).
   - The heterogeneous FFMA + MMA gate: 1.04 on sm_120 (OPT4-T07, rejected).

   The wide GEMMs are 40–47 % of the step on NaCl. A primitive 6 % slower costs 2.5–3 % of the step. Triton's
   costs over 60 %.
2. **The node features cannot stay on chip.** One layer's features on M NaCl are 216 × 512 × 16 × 4 B = 7.1 MB.
   The whole card holds 36 × (256 KB registers + 99 KB shared) ≈ 12.8 MB, and it also has to hold the weights
   (1 MB per 512 × 512 irrep linear, several per layer). A tile of 16 atoms × 512 channels × 16 components is
   512 KB, five times one SM's shared memory. Any 512-channel chain spills through L2, as the separate kernels do
   today. Only the narrow and k-space parts have intermediates small enough to keep on chip: `[k, ≤ 16]` structure
   factors, `[atoms, ≤ 16]` charges and potentials.
3. **Ceiling.** On M NaCl, removing every gap (9.3 %), every splitK reduce and memset (3.2 %) and half of the
   pointwise time (7 %) would give about 15–20 %, and only if the wide GEMMs kept their speed. This agrees with the
   ledger's 8–16 % NaCl ceiling ([launch-megakernel](../../optimisation-ledger.md#launch-megakernel)). On NbCl5 it
   is 2–5 %, almost all of it the k-space part.

The ledger's condition to reopen `launch-megakernel` ("graph gaps grow well above 5 % of a step") is met on
216-atom POLAR-1 cells (6–12 %). The GEMM-rate blocker in point 1 is not new evidence and still stands.

## 3. Fusion plan (what goes in, what stays out)

| Part | In the megakernel? | Why |
| --- | --- | --- |
| Wide irrep linears (interactions, products, readouts, fukui and update MLPs at 512 channels) | **out**: cuBLAS | 82 % of FP32 peak; no in-kernel primitive matches it on sm_120 (§2.1) |
| Edge convolution (`oeq` streaming conv, folded backward), radial tables, edge outer sums | **out**: existing fused kernels | already single fused kernels per direction |
| Symmetric contraction (`uniform_1d`, `segmented_polynomial`) | **out**: cuEquivariance | 4 launches, already fused |
| Long-range k-space chain per recursion step: structure factor → density basis → potential → projection, both spins, forward and backward | **in** (candidate) | `[k, ≤ 16]` and `[atoms, ≤ 16]` intermediates fit on chip; 2–6 % of the step; but see §4 |
| Narrow source heads `[·, 512] → [·, 1 or 2]` and their gradients, charge sums, fukui normalisation | **in** (candidate) | output columns ≤ 2; about 60 launches plus splitK companions; 3–4 % on NaCl; but see §4 |
| Inductor pointwise glue between wide GEMMs | **out** | it sits between cuBLAS calls; fusing it needs the GEMMs inside (§2.1) |

Budget on sm_120 for the in-kernel parts:

- **k-space (NaCl):** 243 k × 16 channels × 2 (cos, sin) × 4 B = 31 KB of structure factors, plus 216 × 16 × 4 B
  = 14 KB of charges. One CTA, or a few with a grid barrier, can hold all of it in shared memory.
- **k-space (NbCl5):** 2989 × 16 × 8 B = 383 KB of structure factors. That needs a two-phase grid (partial sums
  per atom tile, then projection). Phases are recomputed in registers (64 × 64 tiles: 64 regs for `cos`/`sin`
  plus accumulators, about 128 regs per thread at 4 warps, no spills).
- **Narrow heads:** a row tile of 64 atoms × 512 channels is 128 KB as FP32. That is over 99 KB, so K is streamed
  in 128-wide chunks (32 KB per stage, 2 stages). The weights `[512, ≤ 2]` fit in registers.

## 4. Overlap with the off-polar-alex branch

Both candidate parts in §3 are already being built there as A/B constants on `fastmace/off-polar-alex` (not on
`main`):

- `FUSED_FOURIER` and `fourier.JOINT_BACKWARD` (`fastmace/kernels/triton/fourier.py`) compute the structure
  factor and projection in 64 × 64 phase tiles without dense `cos`/`sin`, plus a joint backward. Rejected on the
  A100 (1.08–1.40× slower). Not yet measured on the RTX 5060 Ti.
- `SPIN_BATCHED` puts both spins in one k-space call.
- `SMALL_MATMULS` rewrites matmuls with inner dimension ≤ 8 or ≤ 4 output rows or columns as fused broadcast sums.
- `FOLD_UPDATE` folds the POLAR update MLP. A local A/B is queued on this GPU now.

A megakernel for the k-space chain or the narrow heads would be a further fusion of those switches. Its own gain
is the glue launches between them, about 1–2 % on NaCl. Building it on `main` now would duplicate their kernels.

## 5. Recommendation

- Do not build a complete POLAR megakernel on sm_120. The wide GEMMs block it (§2.1), and the 512-channel
  intermediates cannot stay on chip (§2.2).
- The remaining polar megakernel (k-space chain plus narrow heads in one or two launches per direction) is worth
  building only after `FUSED_FOURIER`, `JOINT_BACKWARD`, `SPIN_BATCHED` and `SMALL_MATMULS` have their RTX 5060 Ti
  A/Bs. Build it on top of whichever of them win, with a bound of 1–2 % on NaCl.

## 6. Results (8 November 2026)

What was built is the narrow one: the k-space field call (structure factor, density, Coulomb kernel, projection)
as one persistent kernel per direction, in Triton and in CUDA C++ (`fastmace/kernels/cuda/kspace_megakernel.cu`,
cubins for sm_80/89/90/120). The narrow heads were removed by algebra instead of fused (P1, R5). MD step b/a on
the RTX 5060 Ti, abab / abba; gates are 3× force/stress on S, M (NaCl, NbCl5) and L NaCl (RTX A4000), with
L NbCl5 on a Modal A100-80GB. Every gate ran with `RewriteConfig.polar_uvu = on` and no rule-13 revision.

| Part | Kernel level | NaCl S / M / L | NbCl5 S / M / L | Gate | Decision |
| --- | --- | --- | --- | --- | --- |
| P1 `COMPOSE_HEADS` | M NbCl5 42.05 → 37.35 ms of kernels | 0.906/0.909, 0.876/0.872, 0.883/0.879 | 1.001/1.005, 0.997/1.000, 1.002/0.998 | pass (L NbCl5 0.86 / 0.58 / 1.01) | default, every GPU |
| R5 `PRUNE_DEAD_DEGREES` | L only | L 0.979/0.975 (Ada 0.974/0.973) | L 0.998/1.000 | pass | default, every GPU |
| `PHASE_RECURRENCE` | – | 0.988–1.008 | 0.988–1.075 | – | rejected |
| CUDA k-space megakernel | A4000 field call 0.70× (NaCl), 0.30× (NbCl5) of dense | 1.012/1.009, 1.008/0.998, 1.003/1.000 | 0.893/0.896, 0.957/0.957, 0.972/0.973 | pass (L NbCl5 0.68 / 0.54 / 0.52) | default on the 5060 Ti from 1e6 atoms × k |
| R3 (rows across the head) | P1 removed its four target kernels | – | – | – | stopped at the bound |

The NaCl loss is outside the kernel, which is faster than dense there too. So `MEGAKERNEL_MIN_WORK` keeps NaCl
(216 × 243 = 52 k) on the dense path, and NbCl5 (1944 × 2989 = 5.8 M) takes the megakernel. Nothing between those
two sizes has been measured. Other GPUs are not listed.

## 7. Reproduce

`tools/bench/polar_profile_bound.py <profile-dir> <run> …` prints the table in §1 from the profiles' `step.log`
and `module/sources.csv`.
