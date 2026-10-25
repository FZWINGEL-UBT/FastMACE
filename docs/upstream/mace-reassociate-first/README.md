# MACE PR draft: reassociated first interaction

**Upstream:** ACEsuit/mace, base `main` at `e37c8583a8b1b586e5888fa6dca86e8f591e2606` (19 October 2026).
**Patch:** [`0001-reassociate-first.patch`](0001-reassociate-first.patch) (new files only:
`mace/modules/reassociate.py`, `tests/unit/test_reassociate.py`). Not opened (owner opens it, decision
D5 (b)).

## Title

Add an inference-time reassociation of the first interaction: sum edges per (receiver, sender species) before the radial projection

## Description

### Motivation

In the first interaction block the sender features are species-only scalars
(`linear_up(node_embedding(one_hot))`), and the convolution is channelwise with one radial weight per
(path, channel). Today every edge forms its `[P*C]` tensor-product weights (`conv_tp_weights`, e.g.
4 x 128 = 512 values per edge) and the tensor product, then sums per receiver. Because the last radial
layer is affine and the sender features depend only on the sender's species, the sums can be
reassociated:

```
m[i, l, m, u] = gamma_l  sum_z x[z, u]  sum_k W[path(l)·C + u, k]  S[i, z, k, (l, m)]
S[i, z, k, (l, m)] = sum_{edges j→i, species(j)=z}  a[e, k] Y[e, (l, m)],     a[e] = [h_e c_e, c_e]
```

`S` is a small edge sum over K ≈ 65 hidden radial features (plus a bias column), and the projection
is one dense GEMM per path over `[atoms·(2l+1), species·K]`. The per-edge weights are never formed.

The conversion `reassociate_first_interaction(model, species_columns=None)` needs **no change to any
interaction forward**: it swaps `conv_tp_weights` for "hidden layers, then append a 1 column" (the
forward's existing `* cutoff` produces `a`), and `conv_tp` for a module with the fused calling
convention (`conv_fusion`), which gets `node_attrs` from a forward pre-hook. `gamma_l` and the output
layout are **measured** from the model's own `conv_tp` (e3nn, cuEquivariance or OpenEquivariance), so no
normalisation convention is assumed. The conversion returns `False` and leaves the model unchanged
where the form does not hold (non-scalar sender irreps, non-channelwise TP, non-affine last radial layer,
models with `joint_embedding`).

### Measured gains (FastMACE / research-engine implementation of the same algorithm)

| Model | Change | NaCl | NaNbO3 | NbCl5 | GPU | Source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| MACE-MH-1 | plain PyTorch prototype (as in this patch) | −0.6 to −2.6 % | | | RTX 5060 Ti | ResearchModule `docs/roadmap.md` item 11.2, 10 October 2026 |
| MACE-MH-1 | with a fused Triton backward of the edge sum | −4.9 % | −7.2 % | −4.9 % | RTX 5060 Ti | same |
| MP-0, MPA-0, OMAT-0, MatPES and other non-MH-1 benchmark models | with the receiver-streaming convolution, together | −16 to −28 % | | | RTX 5060 Ti | `docs/roadmap.md` item 13, `studies/other_models/README.md`, 15 October 2026 |
| MACE-OMOL-0 | reassociation added | | | −9 % | RTX 5060 Ti | AGENTS.md / `studies/other_models/README.md`, 16 October 2026 |

MD-step times, float32 hybrid, CUDA graphs, 3x3x3 cells. All passed the force and graph-replay gates.
FastMACE: records `reassociate_first` (MH-1 family, `fastmace/rewrites/mh1/`) and `generic_reassociate`
(`fastmace/rewrites/generic/`); float64 agreement 1.3e-16 to 3.2e-15 eV/Å on `mp0-medium`
(`docs/validation/1a-T08-generic-rewrites.md`). Its edge-sum cost after the rewrite is 0.06 ms (NaCl) to
0.5 ms (NbCl5) per force call (`docs/validation/2-T20a-step-breakdown.md`).

**Honest expectation for this patch:** it is the plain PyTorch form (an `index_add_` of outer products,
autograd backward). Our measurements say most of the gain needs a fused backward of the edge sum; this
PR is the correct, general base on which such a kernel can be added later.

### Exactness

The same products in a different summation order: exact up to rounding. The float64 test agrees with the
unconverted model to 1e-10 eV (energy) and 1e-9 eV/Å (forces) on Residual, DensityResidual and
ResidualNonLinear first blocks. `index_add_` on CUDA uses atomics: not bitwise reproducible on GPU (as
MACE's `scatter_sum` already is).

### Limitations

- inference only (weights frozen into a buffer);
- memory of `S` is `atoms x species x (lmax+1)^2 x (K+1)`; pass `species_columns` (the elements a
  simulation can contain) for 89-element foundation models;
- the converted model is not TorchScript-exportable (forward pre-hook with kwargs, Python state);
- the species table `x` is read back from `node_feats` with `index_reduce_(amax)`, which assumes that
  every atom of a species has identical first-layer features (true for species-only embeddings; models
  with `joint_embedding` are refused).

## Test plan

- [x] `tests/unit/test_reassociate.py`: float64 three-species MACE models with each residual first
  block, energy and forces against the unconverted model, with and without `species_columns`. **Run:
  6 passed** on MACE 0.3.16 / torch 2.14.1+cpu (MACE main itself not run: Python 3.10 environment).
- [ ] maintainers' CI on main.
- [ ] GPU: `mace-mp-0` medium and MH-1 with `enable_cueq` and `enable_oeq` (the `conv_fusion` probe
  path), float32 force check, MD-step timing.
- [ ] a model whose `conv_tp_weights` is e3nn's `FullyConnectedNet` with normalised weights (affinity
  probe covers it; add an explicit test).

## Risks

- **Small or no gain without a fused backward**, as measured (0.6–2.6 %). Reviewers may prefer to wait
  for a kernel.
- **Probing relies on internal structure** (`interaction.edge_irreps`, `edge_attrs_irreps`,
  `conv_tp.weight_numel`, radial net layer list). Any mismatch returns `False` rather than producing
  wrong numbers, and the affinity of the last radial layer is checked numerically.
- **Pre-hook state** (`conv_tp.node_attrs`) is per call; concurrent calls of one model object from two
  threads would race (MACE does not do this).
