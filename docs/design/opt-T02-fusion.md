# OPT-T02 design: fusion of the atom-level elementwise work and removal of layout copies

Status: **reviewed** (conductor, 20 October 2026). Stage A is approved as a new registry record that applies
wherever `block_linear` applies (MH-1, OMOL and the generic path where the structure matches), default
`off` until the A/B on both GPUs. B and OPT-T03 share one shape-based tile-rule module. D is out of scope (§7). Plan: `plans/opt-1.md`, OPT-T02. Evidence:
2-T20a's step breakdown on the local RTX 5060 Ti (`docs/validation/2-T20a-step-breakdown.md`, raw tables in
`results/2-T20a-step-breakdown-20261020/`), MH-1 hybrid float32, NaCl and NbCl5 3×3×3, FastMACE `67940f6`.
All times are kernel µs per force call (forward and backward), from the profiler's `ops.csv` and
`kernel_sequence.csv`.

## 1. What the atom-level elementwise time is

FastMACE's atom-level elementwise kernels take **7.35 ms per force call on NbCl5** (28 % of the kernel
time) and **0.36 ms on NaCl** (11 %). Nsight Compute puts the NbCl5 ones at 370 GB/s from DRAM, 83 % of
the card's 448 GB/s, so they are memory-bound. The table splits them by what the Inductor kernel does,
read from the fused op names:

| Bucket | NbCl5 µs | NaCl µs | Kernels | Example |
| --- | ---: | ---: | ---: | --- |
| Layout copies, forward: per-l slices copied into GEMM rows, GEMM outputs concatenated | 2,461 | 95 | 40 | `cat_slice_view_43`, `clone_slice_42`, `cat_view_38` |
| Layout copies, backward: slice gradients scattered and accumulated into the wide tensors | 1,924 | 92 | 17 | `add_slice_slice_backward_view_8`, `add_mul_slice_slice_backward_sum_view_9` |
| Gate, backward (sigmoid and SiLU derivatives) | 1,687 | 71 | 11 | `add_mul_sigmoid_sigmoid_backward_silu_backward_...` |
| Gate, forward (SiLU on scalars, sigmoid gates on l > 0) | 767 | 40 | 3 | `cat_mul_sigmoid_silu_slice_unsqueeze_view_39` |
| Density division before `linear_1` (fused with slicing) | 343 | 30 | 9 | `_unsafe_view_add_cat_clone_div_mul_slice_53` |
| Residual adds, sums, readout | 171 | 32 | 27 | `add_cat_view_45` |
| **Total** | **7,352** | **360** | 107 | |

**4.4 ms of NbCl5's 7.35 ms (16 % of the whole force call) is copying.** It is not arithmetic. The
copies come from `BlockLinear` and `ResidualBlockLinear` (`fastmace/rewrites/linear/modules.py`):

- **Before each per-l product** the segment `x[:, offset:offset + dim·mul]` of the `[N, W]` ir_mul tensor is
  reshaped to `[N·(2l+1), mul]`. That slice is not contiguous, so Inductor materialises it
  (`clone_slice_*`).
- **After the products** the per-l results are concatenated back into `[N, W]` (`cat_view_*`).
- **The backward** turns both into zero-filled wide gradients plus slice-adds (`slice_backward`).

The research engine found that no view can avoid this. The rows of one segment are not evenly spaced in
`[N, W]`, and strided-batched GEMMs were slower (`2026-10-15-fourth-round.md`, item 1).

## 2. Where they sit: one MH-1 interaction, forward (NbCl5, interaction 2; µs per kernel)

Order from `kernel_sequence.csv`, indices 117–142 (interaction 1 is the same with the reassociated first message):

```
OEq streaming convolution            message [N, 5120] (ir_mul)                         858
div + slice (density first)          message / (density*beta + alpha), per-l slices      59 + 73 + 59 + 10
ResidualBlockLinear GEMMs            l = 0 [N, 384+...] @ [.., 2048]; l = 1..3 [N(2l+1), 384/256] @ [.., 512]
cat (outputs -> [N, 9728])           cat_view_36 / 37 / 38                               24 + 83 + 143
Gate                                 [N, 9728] -> [N, 8192] (SiLU on 512 scalars,        383
                                     sigmoid(1536 gates) x l = 1..3 blocks)
clone (gate output -> per-l rows)    clone_slice_40 / 41 / 42                            35 + 83 + 146
linear_2 GEMMs                       [N(2l+1), 512] @ [512, 512]
cat (-> [N, 16, 512] for cuEq)       cat_slice_view_43                                   334
symmetric contraction (cuEq)         segmented_polynomial                                169
product linear GEMMs, add_cat_45     ...                                                 118
```

The gate needs the l = 0 GEMM's gate columns for each atom, and it works per atom and channel. Nothing in
the chain from the residual products through the gate to `linear_2` needs the concatenated `[N, W]`
layout. Only the OEq convolution (input), the cuEq symmetric contraction (input `[N, 16, 512]`) and the
skip/readout boundaries consume a concatenated tensor.

## 3. Candidates, ranked by measured time (NbCl5 per force call; NaCl in brackets)

| # | Candidate | Removes (measured) | Mechanism | Exactness | Risk |
| --- | --- | --- | --- | --- | --- |
| A | **Per-l row tensors between the residual products, the gate and `linear_2`**. The two block linears exchange lists of `[N·(2l+1), mul]` tensors instead of one `[N, W]` tensor, and the gate works on those rows (view `[N, 2l+1, mul]` × gate `[N, 1, mul]`). | forward `cat_view_36–38` + `clone_slice_40–42` (2 × 0.51 ms) and their backward slice-gradient scatters (estimated about 2 × 0.45 ms of the 1.92 ms; the backward kernels are not split by module): **≈ 1.9 ms** (≈ 0.1 ms) | PyTorch restructure of the hashed MH-1 forward (`cutoff_folded_forward`) and the linear modules; Inductor fuses the per-l gate itself; cuBLAS GEMMs unchanged | **exact**: the same products on the same operands, the same elementwise operations per element; expected bit-identical (to be shown on CPU float64 and GPU float32) | low; MH-1-class (gated) interactions only |
| B | **Segment-addressed GEMM** (Triton, IEEE FP32): a custom op whose A rows are read as `(atom n, component m) → n·S_n + off + m·mul` from the wide tensor, and whose output rows go straight into a strided destination (the `[N, 16, 512]` buffer of the symmetric contraction, or `[N, W]`). The density division goes into its prologue (one scale per atom). Its backward is the same op with the transposed weight, writing the input gradient in place. | the remaining forward copies (`cat_slice_view_43` 2 × 0.33 ms; div + slice 0.06 + 0.20 ms; product-linear clones and cats ≈ 0.3 ms) and the rest of the backward scatters (≈ 1.0 ms, estimated): **≈ 2.2–2.4 ms** (≈ 0.15 ms) | Triton GEMM custom op (`torch.library`, fake impl, `register_autograd`), one tile configuration chosen by a shape rule (no autotuning) | rounding (K summation order of a different GEMM), as `block_linear`; self-check per module (§6.4: 1e-4 float32, 1e-10 float64 through the PyTorch path) | **medium**: only pays if the Triton IEEE GEMM matches cuBLAS SIMT on these shapes; OPT-T01's bench measures exactly that (`triton_ieee` column); GEMMs are 9.9 ms, so a 5 % slower GEMM costs 0.5 ms |
| C | **Gate in the residual GEMM epilogue.** The l = 0 product (2048 columns) applies SiLU to the 512 scalars and keeps sigmoid(gates). The l > 0 products multiply by the atom's sigmoid gate in their epilogue. The pre-activations needed by the backward are written once. | the gate forward's extra pass over `[N, 9728]` (2 × 0.38 ms, minus the epilogue's own work): **≈ 0.5 ms** forward | epilogue of B's Triton GEMM | exact up to B's rounding (same elementwise formulas) | medium; needs B; a gated row needs its atom's gate, which a tile reads per row |
| D (rejected) | **Gate backward fused** with the input gradient of `linear_2`: `d pre = (dY · W₂ᵀ) · σ(g)` in the epilogue; `d g = Σ_m (dY · W₂ᵀ) · pre · σ'(g)` as a per-atom reduction | most of the gate backward (2 × 0.84 ms): **≈ 1.0–1.4 ms** | epilogue of B's backward GEMM; the Σ over the 2l + 1 rows of an atom needs tiles aligned to whole atoms or an atomic add per (atom, channel) | rounding (reduction order) | **high**: tile alignment to 2l + 1 rows, and atomics change determinism (the reproducible mode would need the unfused path) |
| E | Density division into the `ResidualBlockLinear` prologue without B (Inductor) | 0.34 ms (0.03) | already fused with the slice copies by Inductor; it disappears with A/B | exact | – (part of B) |

Mechanisms not chosen:

- **Inductor's own GEMM epilogue fusion** works only with Triton mm templates under `max_autotune`. That is
  per-shape autotuning (excluded by SPEC §9.1, and the research engine measured 140–756 s of setup). It
  cannot fuse a prologue gather into the extern cuBLAS call either.
- **cuBLASLt epilogues** offer only bias, ReLU and GELU (and their gradients). The MH-1 linears are bias-free.
  There is no sigmoid gate taken from another operand, and no gathered row addressing, so they remove none
  of the copies.
- **CUTLASS** (gather/scatter GEMM, example 36; epilogue visitor trees) can express B–D in C++. It is the
  fallback if Triton's IEEE GEMM cannot match cuBLAS: a compiled extension per architecture (sm_89,
  sm_120), which phase 6's wheels would have to carry.
- **cuBLASDx** (device GEMM inside an NVRTC kernel, as for the OEq streaming kernels) needs the MathDx
  headers. They are not in our environments, and adding them is an environment change for the owner. Not now.

## 4. Expected gain

| Stage | NbCl5 step (27.4 ms) | NaCl step (3.71 ms) |
| --- | --- | --- |
| A only | −1.5 to −1.9 ms (**−5 to −7 %**), exact | −0.05 to −0.1 ms (−1 to −3 %) |
| A + B | −3.3 to −4.3 ms (**−12 to −16 %**), if B's GEMMs are within ±2 % of cuBLAS | −0.15 to −0.25 ms (−4 to −7 %), depends on B at small M |
| A + B + C (+ D) | a further −0.5 ms (−1.4 ms with D) | small |

The ceiling for all copies plus the gate is 6.8 ms on NbCl5 (25 %). NaCl gains little: its elementwise
work is 0.36 ms, and its step is bound by the underfilled small-M GEMMs (2-T20a: l = 0 at 5.8 TFLOPS on
0.04–0.4 waves). That is OPT-T03's subject. B's GEMM could take a small-M rule as well.

## 5. Exactness and generality

- **Gate rule:** float32 forces, stress and energy against float64 at most 2× the float32 calculator's
  error (`plans/opt-1.md`; FastMACE's gate, 1b-T16), and all 19 labels through the gate (SPEC §12.5).
- **A:** declared `exact`, `deterministic=True`. CPU float64 tests show the forward and the forces are
  bit-identical to `block_linear` + `merged_residual` + the stock gate. That is possible because A changes no
  product and no elementwise formula. A GPU float32 test compares the captured graph's forces bitwise with
  the current default. If Inductor's fusion changes an FMA contraction, the class drops to `rounding` with
  the self-check, and the report says so.
- **B and C:** `rounding`, `deterministic=True` (no atomics, a fixed K order), with the §6.4 per-module
  self-check on random inputs in both layouts, against `BlockLinear`'s products, and the full gate.
- **D:** `rounding`, and `deterministic=False` if it uses atomics. It is then excluded from the reproducible
  mode (SPEC §9.3), as `streaming_conv` is.
- **Generality:**
  - Segment layouts come from the module's own irreps (`segments()`), as in `block_linear`. Any cuEq
    `Linear` qualifies, and the generic models (MP-0, MPA-0, OMAT-0) get B without the gate parts.
  - The gate parts apply where MACE's gated nonlinear interaction is used (MH-1 class: `irreps_gates`
    from the module).
  - B uses one tile configuration per GEMM, chosen by a rule from M, K and N (for example a 64-row tile
    when M < 4 × SMs × 64), the same on every GPU: no table, no autotuning (SPEC §9.1).
- **Registry:** new records with `requires=("block_linear",)` (A, B) and `("opt_t02_segment_gemm",)` (C, D),
  source hashes of the forwards they replace, and default `off` until the A/B and the owner's decision.
  This is the opt-in rule of `plans/opt-1.md`.

## 6. Proposed order and measurements

1. **A first:** exact, cuBLAS unchanged, about 1.9 ms on NbCl5.
   - CPU float64 bitwise tests, then the GPU fast-tier bitwise test.
   - Same-process A/B (`ab_fastmace`-style, option off against on) on NaCl and NbCl5, 12 blocks, against
     murphy's noise floor (0.16 %) and the local one (0.4 % on NbCl5, 1.1 % on NaCl).
   - The 19-label gate.
   - A repeated 2-T20a breakdown to confirm the removed kernels.
2. **B after OPT-T01's bench:**
   - B goes ahead only if the `triton_ieee` column is within about 2 % of cuBLAS on the production
     shapes (`gemms.csv`, sm_89 and sm_120).
   - Otherwise the CUTLASS path, or stop at A.
3. **C after A** (and after B, whose epilogue it uses). D is not built.

Done when each stage reports speed (per crystal and GPU) and accuracy (2× rule, 19 labels). Each stage is
recorded in `docs/rewrites.md`, adopted or not.

## 7. Open points for review

1. **Answered:** A is a new record (self-check, off switch, source-hash pin on what it replaces), applied
   wherever `block_linear` applies. The interface change is fine if every current user keeps working.
2. **Answered:** B and OPT-T03 share one shape-based tile-rule module (no per-GPU tables).
3. **D is rejected** (conductor, 20 October 2026): its per-atom reduction needs atomics or whole-atom
   tiles, and non-deterministic gradients would break the reproducible mode (SPEC §9.3) and the float64
   bitwise promise.
4. To verify at implementation: the gate module's class and layout. The compiled kernel's shapes,
   `[N, 9728] → [N, 8192]` with an unsqueeze on the gates, show the ir_mul form.
