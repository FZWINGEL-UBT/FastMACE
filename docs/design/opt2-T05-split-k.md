# OPT2-T05: split-K for small-cell atom products

Plan task OPT2-T05 (`plans/opt-2.md`; candidate 5 of `docs/design/opt-T09-step-breakdown.md`).
Branch `fastmace/opt2-T05`.

## The problem

On NaCl 3×3×3 (216 atoms), the atom-level linears are 47 % of the force kernels. Each linear is
one product per irrep l, of the shape `[atoms × (2l+1), K] @ [K, N]`, in a block linear
(`fastmace.rewrites.linear.modules.block_products`, `ConstantMatmul`: cuBLAS, forward and input
gradient). The l = 0 products have M = 216 rows.

- With 64 × 64 output tiles, `[216, 512] @ [512, 512]` gives 4 × 8 = 32 tiles: 0.9 waves on the
  RTX 5060 Ti's 36 SMs and 0.7 on murphy's 48.
- ncu measured it at 0.8 waves, 5.9 TFLOPS and 18 % occupancy (OPT-T09). That is about half of
  what the l = 1–3 products reach.
- cuBLAS already splits K eight ways for the K = 2048 l = 0 inputs of `grouped_linear` (opt-T03b
  diagnosis, `[16, 1, 8]` grids with `splitKreduce`). It does not split the K = 512 ones.

OPT-T09 put the ceiling at about 2 % of the NaCl step. The large cells have enough row tiles, and
must stay unchanged.

## The rule

The rule uses only the product's M, N and K and the device's SM count. It keeps no table and
measures nothing at run time (SPEC §9.1). It lives beside the shared rule in
`fastmace.kernels.triton.tile_rule`:

```
split_k_parts(m, n, k, sms):
    tiles = ceil(m / 64) * ceil(n / 64)         # the same nominal 64 × 64 tiles as the shared rule
    if no SMs, or k > n, or tiles >= sms: 1     # deep K stays with cuBLAS; a full wave: leave it alone
    parts = 1
    while tiles * parts < 1 * sms               # aim for one wave (SPLIT_K_TARGET_WAVES)
          and k % (2 * parts) == 0
          and k / (2 * parts) >= 128:           # every part a real GEMM (SPLIT_K_MIN_PART)
        parts *= 2
    return parts
```

The first version aimed for two waves and also split deep K. The micro-benchmark
(`tools/split_k_bench.py`, local RTX 5060 Ti, products replayed in a CUDA graph) changed both:

| Product | cuBLAS | 2 parts | 4 parts | 8 parts |
| --- | ---: | ---: | ---: | ---: |
| `[216, 512] @ [512, 512]` (NaCl l = 0) | 20.8 µs | **18.5 µs** | 19.4 µs | 20.9 µs |
| `[216, 2048] @ [2048, 512]` (grouped l = 0, deep K) | 45.8 µs | 61.2 µs | 62.1 µs | 56.9 µs |
| `[216, 512] @ [512, 2048]` | 48.0 µs | 58.3 µs | 58.2 µs | 66.7 µs |
| `[648, 512] @ [512, 512]` (l = 1) | 38.1 µs | 43.0 µs | 45.2 µs | 50.5 µs |

- **Deep K stays with cuBLAS.** cuBLAS already splits deep K itself (the opt-T03b `[16, 1, 8]`
  grids), and splitting it again loses 24–36 %.
- **One wave is enough.** 2 parts are best on the one shape that gains.
- **The rule's other choices hold.** Splitting the products it leaves alone loses.

The rule uses integer arithmetic only, because the forward evaluates it while `torch.compile`
traces (the lesson of OPT-T02b, which hit a format error inside a traced rule).

It is applied to each GEMM separately:
- the forward `x @ W`: M rows, N = W's columns, K = W's rows;
- the input gradient `g @ Wᵀ`: M rows, N = W's rows, K = W's columns.

What it gives:

| Product (atoms × (2l+1), K, N) | sm_120 (36 SMs) | sm_89 (48 SMs) |
| --- | --- | --- |
| NaCl l = 0, `[216, 512] @ [512, 512]`, forward and gradient | 2 parts | 2 parts |
| NaCl l = 0, grouped `[216, 2048] @ [2048, 512]`, forward (deep K) | 1 | 1 |
| its input gradient `[216, 512] @ [512, 2048]` (128 tiles) | 1 | 1 |
| NaCl l ≥ 1 (M ≥ 648: 88 tiles and more) | 1 | 1 |
| every NbCl5 and NaNbO3 product (M ≥ 1944) | 1 | 1 |

**Expected size.** About 2 µs per split product, on a few products per step: about 0.5 % of NaCl's
3.2 ms step. That is below the local noise floor (1.1 % on NaCl) and above murphy's (0.16 %).

So NbCl5 is unchanged by construction. The rule's only cost there is evaluating it while tracing.

## The kernel

`split_k_mm(a, b, parts)` computes `bmm(a viewed as [parts, M, K/parts], b viewed as [parts,
K/parts, N]).sum(0)`:
- **One batched cuBLAS call** gives parts × as many tiles. The partial `[parts, M, N]` sums are
  reduced by one reduction kernel, which Inductor may fuse into the consumer.
- **No copy.** The row views of a block linear have a unit last stride, so they view as
  `[M, parts, K/parts]`, and the weights are contiguous.
- **Deterministic.** There are no atomics, and the partial sums are added in a fixed order, so the
  record declares `deterministic=True`. A reproducible plan that asks for it stays bitwise
  repeatable, with another fixed rounding.
- **Extra traffic.** The partials cost parts × M × N × 4 bytes: 1.8 MB per NaCl l = 0 product,
  about 4 µs at 448 GB/s. The product itself runs about 19 µs at 5.9 TFLOPS. The gain needs the
  batched call to reach more than about 8 TFLOPS. Only the A/B can tell.

A Triton split-K kernel that writes partials, like `segment_gemm`, is the fallback if `bmm` does
not pay. cuBLASLt's split-K heuristics are not reachable from PyTorch without a C++ extension.

## The record

`split_k` is a `Rewrite` in the fusion family:
- `rounding`, `deterministic`, `default="off"` (`never_automatic`);
- requires `block_linear`, and conflicts with `grouped_launch`, which takes the same small products
  into its own launch;
- applies to float32 on CUDA only. Elsewhere the module's SM count stays `None`, and every product
  keeps cuBLAS.

It sets `module.split_k` and `module.sms` on every block linear, as `grouped_launch` does.

The self-check runs on the probe rows of every block linear. It compares the split product, forced
to 2 parts where K allows, and the rule's own choice, against `ConstantMatmul`: values and input
gradients, in the model's dtype.

## Tests

CPU:
- **The rule:** the table above, the waves threshold, the K limits, no SMs, and integer-only
  evaluation.
- **`split_k_mm` against `a @ b` in float64,** including strided row views: agreement within
  1e-12 relative, and an input gradient that goes through the split.
- **The record:** opt-in; refused without `block_linear` and together with `grouped_launch`; applies
  to every block linear and passes its self-check. With a fake SM count that makes every product
  split, the energy and forces of the tiny MH-1-type model in float64 agree with the unsplit model
  within 1e-12 relative.
- **The default path** is untouched: with the record off, no `split_k` attribute is set and
  `block_products` is unchanged.

GPU, under the local GPU reservation, in short jobs:
- **The force gate:** `tools/force_stress_gate.py --engine fastmace --rewrites split_k=on` on NaCl
  and NbCl5.
- **The A/B:** `tools/ab_rewrites.py --b split_k=on`, both build orders, 12 blocks, NaCl and NbCl5,
  from a pinned source copy.
- **murphy:** the same A/B when the GPU is empty, before Thursday 16:00 UTC.
