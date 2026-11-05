# OPT3-T01: fused gate and residual VJP, what the trace leaves to win (analysis before building)

**Status: analysis, 21 October 2026.** Plan `plans/opt-3.md`, OPT3-T01 (estimate 3–6 % on large cells, rounding).
Nothing is built yet.

## Where the gate runs today

MH-1's nonlinear interactions gate the products of the residual block. Scalars go through SiLU, and each
l = 1–3 block of 512 channels is multiplied by a sigmoid gate column. In FastMACE this sits inside
`gated_rows` (OPT2, default on), whose PyTorch operations Inductor compiles and fuses.

## The kernels (OPT-T10 trace, NbCl5, local RTX 5060 Ti; `results/opt-T10-step-breakdown-072fa0a/local/NbCl5/ncu_kernels.csv`)

Atom-level elementwise kernels are 2.55 ms of a 22.39 ms force call (83 kernels). The gate's share, grouped by
Inductor kernel name (rows 5,832 / 9,720 / 13,608 = 1,944 atoms × 3 / 5 / 7, l = 1 / 2 / 3; both interactions):

| Kernel group | role | n | µs | MB moved |
| --- | --- | ---: | ---: | ---: |
| `joined_products_mul_sigmoid_silu_slice…` | forward: products and gate, already fused | 8 | 576 | 183 |
| `mul_sum_view` | backward: gate gradient, sum over m of grad × product (l = 2, 3) | 4 | 478 | 202 |
| `mul_sigmoid_slice_unsqueeze_view` | backward: product gradient, grad × sigmoid | 6 | 329 | 109 |
| `add_mul_slice_backward_squeeze_sum_view` | backward: l = 1, both parts already in one kernel | 2 | 161 | 68 |
| `add_mul_sigmoid_sigmoid_backward_silu_backward…` | backward: scalar SiLU and gate sigmoid derivatives | 2 | 165 | 69 |

Every one of them runs at 380–430 GB/s, at the card's DRAM bandwidth (448 GB/s). They are fused already;
only fewer bytes can make them faster.

## What a fused VJP can remove

The minimal backward per l-block reads grad, the products and the gate column once, and writes the product
gradient and the gate gradient. Today two kernels per l (l = 2, 3) read grad, and the second also reads the gate
columns again. For l = 1, Inductor already fuses both parts into one kernel. The bytes a fusion removes:
- the second read of grad for l = 2, 3: about half of `mul_sigmoid`'s 109 MB;
- the second read of the gate columns.

That is about 55–110 MB per force call, **0.13–0.25 ms, 0.6–1.1 % of NbCl5's force call**. The forward is
already one kernel per l with the products.

## Comparison

- **Plan estimate:** 3–6 %.
- **The research engine's ceiling** (October): a stub with no gate work at all, wrong forces, gave 4.7 % on
  NbCl5. Its estimate for a real kernel, which must still compute the activations, was 2–2.5 %. That was before
  Inductor's current fusions: `gated_rows`, and l = 1 already in one kernel.
- **This trace:** 0.6–1.1 %, before any loss of Inductor's neighbouring fusions. OPT2-T02 showed that loss
  costs about as much as such a kernel saves (the gate's grad comes from `linear_2`'s backward, and its product
  gradient feeds the residual products' backward, both fused with these kernels today).

## Recommendation

Do not build. The removable traffic is about 1 % of NbCl5 at best, inside the local noise floor (0.4 %) only
twice over, and the OPT2-T02 lesson says a custom op here may lose the neighbouring fusions and come out
neutral or slower. If the owner still wants it tried, the cheapest test is a backward-only Triton kernel per l
(grad, products, gates in; product and gate gradients out) behind an opt-in record, with the same stop rule
(NbCl5 below 1.0 in both orders). That is a few hours of work, plus GPU time.
