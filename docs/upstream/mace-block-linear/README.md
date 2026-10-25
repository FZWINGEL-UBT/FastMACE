# MACE PR draft: block and grouped linear for cuEquivariance `Linear`

**Upstream:** ACEsuit/mace, base `main` at `e37c8583a8b1b586e5888fa6dca86e8f591e2606` (merge of
develop, 19 October 2026). **Patch:** [`0001-block-linear.patch`](0001-block-linear.patch) (new files
only: `mace/modules/block_linear.py`, `tests/unit/test_block_linear.py`). Not opened (owner opens it,
decision D5 (b)).

## Title

Add an inference-time block-linear conversion for cuEquivariance `Linear` layers (per-l GEMMs, 2–12 % faster MD steps)

## Description

### Motivation

With `enable_cueq=True`, every equivariant linear layer of MACE is a `cuet.Linear(method="naive")`
(`mace/modules/wrapper_ops.py`, `Linear.__new__`). For inference the weights are constant, and such a
layer is block-diagonal: each input segment `mul_in x ir` maps to each output segment of the same irrep
through one `[mul_in, mul_out]` matrix, applied alike to all `2l + 1` components. It can therefore run
as one plain GEMM per (input segment, output segment) on `[n * (2l + 1), mul]` rows of the `ir_mul`
layout.

This PR adds `mace.modules.block_linear.convert_block_linears(model, grouped=True)`, an opt-in,
inference-only conversion that:

1. extracts each layer's effective weights **from the module itself** (`linear.f` on a unit basis), so
   path normalisation and layout are cuEquivariance's own and not re-derived;
2. multiplies the input gradient (forces) by a precomputed **contiguous** transposed weight. The
   autograd backward otherwise multiplies by a transposed view, and cuBLAS picks slower kernels for it
   (15–25 % slower on MACE-MH-1's 512 x 512 blocks);
3. with `grouped=True` (default) concatenates the same-irrep input segments that feed one output segment
   (the `linear` after the convolution: several `128x1o` paths into one block) and multiplies them by
   their stacked weights in **one** GEMM, instead of one GEMM per path plus additions of wide outputs.

Nothing changes for training or for users who do not call the conversion.

### Measured gains (from our validation reports; same algorithm, FastMACE/research-engine implementation)

All MD-step timings are float32, cuEquivariance/OpenEquivariance hybrid, CUDA graphs, MACE-MH-1, 3x3x3
supercells, same-process A/B.

| Change | NaCl (216 atoms) | NaNbO3 | NbCl5 | GPU | Source |
| --- | ---: | ---: | ---: | --- | --- |
| block linear alone (per-l GEMMs + contiguous transposed backward) | −11.8 % | −1.5 % | −0.9 % | RTX 5060 Ti | ResearchModule `Experiments/Langevin_Thermostat/docs/roadmap.md` item 12, 14 October 2026 |
| grouped + merged residual (on top) | −3.5 to −4.6 % | −2.3 to −2.9 % | −4.2 % | RTX 4000 Ada (murphy) | same, item 12, 16 October 2026; `studies/other_models/README.md` |

Both passed the force-gate check (float32 force error against float64 at most 2x that of the stock
float32 calculator) and the CUDA-graph replay gate. FastMACE carries them as registry records
`block_linear`, `grouped_linear` (`fastmace/rewrites/linear/`), default on; see `docs/rewrites.md`.

The **merged residual** (one product per output segment over `linear_1(message)` and
`linear_res(linear_up(x))` of the MH-1-class `RealAgnosticResidualNonLinearInteractionBlock`) is **not**
in this patch: it needs the interaction forward to divide by the density before `linear_1`, which is a
separate change. It could follow as a second PR.

### Exactness

Same terms, different summation order: exact up to floating-point rounding. In float64 the test agrees
with `cuet.Linear` to 1e-12 (values and input gradients). Deterministic (no atomics).

### Limitations

- inference only: weights are frozen buffers, gradients with respect to them are not computed;
- not TorchScript/LAMMPS-exportable (custom autograd function); convert after export;
- only `cuet.Linear` with internal shared weights and one weight class; others are left unchanged.

## Test plan

- [x] `tests/unit/test_block_linear.py`: float64, `mul_ir` and `ir_mul`, grouped and ungrouped, three
  irreps patterns (MH-1-like 128→512, repeated input irrep, output irrep without input) against
  `cuet.Linear`, values and input gradients; non-cuEq modules untouched. **Run: 13 passed** on MACE
  0.3.16 / cuEquivariance 0.13.0 / torch 2.14.1+cpu (not on MACE main, which needs Python ≥ 3.11).
- [ ] maintainers' CI on main (Python 3.11–3.14).
- [ ] GPU: energy/forces of a converted `mace-mp-0` medium vs unconverted, float32 and float64, and an
  MD-step timing (`tests/benchmarks`), to reproduce the table above with this plain-PyTorch version.

## Risks

- **Gain depends on shapes and GPU.** The 11.8 % on NaCl is small-system; large cells gain ~1 %.
  Expect smaller gains without CUDA graphs (launch overhead dominates small systems).
- **Weight extraction relies on `linear.f([weight, x])`** being cuEquivariance's polynomial call. If
  cuEquivariance changes that internal API, the conversion fails loudly (no silent wrong values), and
  the unit test catches it.
- **Memory:** one extra transposed copy of every weight (a few MB for MH-1).
- Users may forget the model is frozen; the docstring and the function name say so.
