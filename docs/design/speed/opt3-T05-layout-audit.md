# OPT3-T05: layout audit of the default step

Plan task OPT3-T05 (`plans/opt-3.md`). Branch `fastmace/opt3-T05`. Baseline: main `62aa6d4` (round-2
defaults).

**Question:** which transposes, contiguity copies, casts and layout conversions are left in the default
step? How much time does each take, and can it be removed exactly, with a view or with a producer that
writes the right layout?

## Method

- **Kernel times** come from OPT-T10's operator tables (`results/opt-T10-step-breakdown-072fa0a/local/`,
  RTX 5060 Ti). The times below are µs per force call.
- **Pure data movement** means:
  - kernels named `aten::copy_`, `aten::cat`, `aten::clone`, `aten::contiguous` or `aten::_to_copy`;
  - Inductor kernels whose fused name has only layout parts: clone, cat, view, slice, transpose,
    permute, `_to_copy`, `unsqueeze` and the like.

  Fused kernels that also compute (gates, divisions, layer norms) carry `clone` or `cat` in their names
  too. They are not counted: removing the layout part would not remove the kernel.
- **Attribution** comes from three sources:
  - the "Topologically Sorted Source Nodes" comments in Inductor's generated code from OPT-T10's
    cache;
  - the neighbours of each kernel in the step's kernel sequence;
  - `tools/layout_probe.py`, one profiled eager force call. It lists the CPU op and input shapes of
    every copy-only kernel.

## Totals

| Structure | All force-call kernels | Pure data movement | Share |
| --- | ---: | ---: | ---: |
| NaCl 3×3×3 | 2745 µs | 64.7 µs | 2.4 % |
| NbCl5 | 21013 µs | 158.0 µs | 0.75 % |
| NaNbO3 | 18242 µs | 153.7 µs | 0.84 % |

## Every group, and whether it can go

| Group | NaCl | NbCl5 | NaNbO3 | Source | Exact removal? |
| --- | ---: | ---: | ---: | --- | --- |
| Streaming convolution input copies `[n, 512]`, forward and backward | 5.3 | about 26 (part of the 52.4 of `[1944, 512]` copies) | about 47 (part of `[1080, 1536]` + `[1080, 512]`) | `streaming_conv_forward/backward` call `x.contiguous()`. `x` is `node_feats_attrs[:, :512]`, a slice of the `[n, 512 + species]` join that the folded MH-1 forward builds only for `handle_lammps` | **Yes: `direct_node_feats`.** Without LAMMPS `handle_lammps` is the identity, so the forward keeps the contiguous tensors |
| Folded radial weight `[W, b]` `[1280, 65]` cat, and its halves `[1280, 64]`, `[1280]` | 5.3 | 5.2 | 5.6 | `FoldedRadial.folded_weight()` concatenates per call; `folded_linear_input_grad` slices and copies per call | **Yes: `folded_constants`.** Constants built once as buffers; a new op takes the halves |
| Interaction 1's transposes of `edge_outer_sum`'s output (`clone_slice_transpose_view`) | 4.9 | 43.6 | 28.7 | `FirstInteractionReassociation`: `summed[:, :, l-block].transpose(1, 2).reshape(...)` | **Only by moving the transpose.** See below |
| Block-linear row gathers in the backward (`clone_slice_4/5/7`, `clone_slice_view_*`) | 19.1 + 3.3 | 33.8 | 9.6 | the gradient of a joined `[n, Σ(2l+1)·512]` output, cut into per-l rows `[n·(2l+1), 512]` for the input-gradient GEMM | **No.** Per atom an l-segment is contiguous, but rows of a GEMM need one stride. A gathered-row GEMM (`segment_gemm`) does it, and that is rounding; the large cells already use it, and NaCl skips it because it is slower there |
| Block-linear joins in the forward (`cat_view_*`) | 17.9 + 3.4 | 15.8 | 9.7 | per-l product rows joined back into the irrep layout | **No,** for the same reason. A GEMM cannot write rows with two strides |
| `joined_products`' cuBLAS fallback writing its product into the output slice (`aten::copy_` `[n, 512]`) | – | about 26 | about 47 | `out[:, offset:...] = (x @ w)` for the l = 0 product of `linear_2` in the `segment_gemm` chain | **Not as exact.** `torch.mm(out=strided slice)` lets cuBLAS write with `ldc` = width, which may choose another kernel (alignment), so it is rounding. Not taken |
| Positions `[n, 3]` | 2.7 | 2.7 | 2.7 | the graph's static input | no (it is the step's input) |
| `_to_copy_select` of the edge index, `index` gathers | 2.8 | 4.4 | 3.7 | int32 sender and receiver indices, species-table gathers | no (casts the kernels need; gathers that are the work) |

**The interaction-1 transposes (screened, not kept).** The GEMM `left @ right` has input
`[nodes·(2l+1), slots·k]` = `[·, 130]` and output `[nodes·(2l+1), 128]`. One of the two must be
transposed, because the next layer needs the irrep layout.

`lm_major_outer_sum` (commits `4dcaa30` and `37ef837`, reverted in `fce27e8`) stored the outer sums
(l, m)-major. Only the kernels' store and load strides changed, so it is exact. Each l's operand became
a view, and the transpose moved to the join and the gradient slices. The layout probe on NbCl5:

| | Off | On |
| --- | ---: | ---: |
| Input transposes | 43.6 µs | 0 |
| Joins | 15.8 µs | 37.8 µs |
| Gradient slices | 33.8 µs | 48.1 µs |
| Total | 93.2 µs | 85.9 µs |

The saving is 7 µs, 0.03 % of the step. A first node-major version left a gather, because between
nodes lie the other l-blocks.

## What is kept

Two records, both MH-1 family, `exact`, deterministic, default off (`never_automatic`), requiring
`fold_cutoff`:
- **`direct_node_feats`:** `cutoff_folded_forward` skips the join when `lammps_class is None` and the
  interaction has the flag.
- **`folded_constants`:** `FoldedRadial.hold_folded_constants()` registers the buffers `folded_constant`,
  `folded_main` and `folded_last`. `folded_linear(..., parts=)` passes them to
  `fastmace::folded_linear_input_grad_parts`: the same Triton kernel, or PyTorch's `grad @ weight` on
  the CPU.

The layout probe confirms the removals on NaCl: pure data movement goes from 59.4 to 47.8 µs per force
call. The `[216, 512]` copies, the `[1280, 65]` cat and the `[1280, 64]` and `[1280]` copies are gone.

Expected gain: about 11 µs on NaCl (0.4 % of 3.03 ms), about 31 µs on NbCl5 (0.15 %), about 50 µs on
NaNbO3 (0.3 %). Each change also saves a launch.
