"""Per-l products for cuEquivariance Linear layers: block, grouped and merged-residual (SPEC §6.2).

cuEquivariance's ``Linear`` (internal shared weights, one weight class) becomes per-l matrix
products with its effective weights extracted once from the module, and an input gradient that
multiplies by a precomputed contiguous transpose. Inductor's backward otherwise multiplies by a
transposed view of the weight, which cuBLAS runs with slower kernels (15-25% on the 512 x 512
blocks of MH-1). Exact up to rounding.

- ``BlockLinear(grouped=False)``: one product per (input segment, output segment) pair.
- ``grouped=True``: input segments of the same irrep that feed one output segment (MACE's
  linear_1 after the convolution: several 128x1o paths -> one 1024x1o block) concatenated along
  the channels and multiplied by their stacked weights in one product, instead of one product per
  segment plus an addition of the (wide) outputs. Same terms; exact up to rounding.
- ``ResidualBlockLinear``: linear_1(message / density) + linear_res(linear_up output) of the
  factored MH-1-class interaction as one product per output segment over both inputs (grouped as
  above), instead of two products and an addition of the wide [atoms, 4096 + ...] outputs. Exact
  up to rounding.

Nothing here reads a switch: grouping is a constructor argument.
"""

from __future__ import annotations

import torch


class ConstantMatmul(torch.autograd.Function):
    """x @ weight for a constant weight, with the input gradient grad @ weight_t (contiguous)."""

    @staticmethod
    def forward(ctx, x, weight, weight_t):
        ctx.save_for_backward(weight_t)
        return x @ weight

    @staticmethod
    def backward(ctx, grad):
        (weight_t,) = ctx.saved_tensors
        return grad @ weight_t, None, None


def block_products(module, lefts):
    """``lefts[k] @ weight_k`` for every product of a BlockLinear or ResidualBlockLinear.

    With ``module.grouped_launch`` (record ``grouped_launch``, OPT-T03 candidate 2) the products the shared
    shape rule calls small (``tile_rule.uses_triton`` false: too few row tiles for the GPU) run together in one
    Triton launch, if there are at least two of them; the others, and everything on the CPU or in other
    dtypes (``module.sms`` None), keep cuBLAS."""
    count = len(lefts)
    weights = [getattr(module, f"weight_{k}") for k in range(count)]
    weights_t = [getattr(module, f"weight_t_{k}") for k in range(count)]
    parts = [None] * count
    sms = getattr(module, "sms", None)
    if getattr(module, "grouped_launch", False) and sms:
        from fastmace.kernels.triton import tile_rule

        small = [k for k, x in enumerate(lefts) if not tile_rule.uses_triton(x.shape[0], sms)]
        if len(small) >= 2:
            from fastmace.kernels.triton.grouped_gemm import grouped_products

            done = grouped_products([lefts[k] for k in small], [weights[k] for k in small],
                                    [weights_t[k] for k in small])
            for k, part in zip(small, done):
                parts[k] = part
    return [part if part is not None else ConstantMatmul.apply(x, w, wt)
            for part, x, w, wt in zip(parts, lefts, weights, weights_t)]


def _group(blocks):
    """(input segments, output segment, stacked weight) per output segment, in first-seen order."""
    grouped = {}
    for i, j, weight in blocks:
        grouped.setdefault(int(j), []).append((int(i), weight))
    return [(tuple(i for i, _ in members), j, torch.cat([w for _, w in members], dim=0))
            for j, members in grouped.items()]


class BlockLinear(torch.nn.Module):
    """cuEquivariance Linear (ir_mul core) as per-l products with extracted weights.

    ``segments_in``/``segments_out``: (offset, mul, dim) of each irrep segment in the ir_mul
    core layout; ``blocks``: (input segment, output segment, weight [mul_in, mul_out]). The
    original module's layout transposes are kept.
    """

    def __init__(self, transpose_in, transpose_out, segments_in, segments_out, blocks, grouped=False):
        super().__init__()
        self.transpose_in = transpose_in
        self.transpose_out = transpose_out
        self.segments_in = tuple(segments_in)
        self.segments_out = tuple(segments_out)
        self.grouped = bool(grouped)
        self.grouped_launch = False  # grouped_launch: small products in one Triton launch (block_products)
        self.sms = None  # grouped_launch: the device's SM count for the shape rule (None: not float32 on CUDA)
        if self.grouped:  # one product per output segment
            blocks = _group(blocks)
            self.block_ends = tuple((i, j) for i, j, _ in blocks)
        else:
            self.block_ends = tuple((int(i), int(j)) for i, j, _ in blocks)
        for index, (_, _, weight) in enumerate(blocks):
            self.register_buffer(f"weight_{index}", weight.contiguous())
            self.register_buffer(f"weight_t_{index}", weight.t().contiguous())

    def forward(self, x):
        x = self.transpose_in(x)
        n = x.shape[0]
        rows = [x[:, offset:offset + dim * mul].reshape(n * dim, mul)
                for offset, mul, dim in self.segments_in]
        return self.transpose_out(join_rows(self.segments_out, self.products(rows), n, x))

    def products(self, rows):
        """Per output segment, its rows [n * dim, mul] (None where no product reaches it), from the
        rows of every input segment (in the ir_mul core layout, after ``transpose_in``)."""
        outputs = [None] * len(self.segments_out)
        lefts = []
        for i, _ in self.block_ends:
            if isinstance(i, tuple):  # grouped
                lefts.append(rows[i[0]] if len(i) == 1 else torch.cat([rows[k] for k in i], dim=1))
            else:
                lefts.append(rows[i])
        for (_, j), part in zip(self.block_ends, block_products(self, lefts)):
            outputs[j] = part if outputs[j] is None else outputs[j] + part
        return outputs


def raw_blocks(linear):
    """(input segment, output segment, weight) of a BlockLinear, grouped products split."""
    for index, (i, j) in enumerate(linear.block_ends):
        weight = getattr(linear, f"weight_{index}")
        if not isinstance(i, tuple):
            yield i, j, weight
            continue
        start = 0
        for k in i:
            mul = linear.segments_in[k][1]
            yield k, j, weight[start:start + mul]
            start += mul


def regroup(linear):
    """The grouped BlockLinear with the same products as ``linear`` (``linear`` itself if already grouped)."""
    if linear.grouped:
        return linear
    return BlockLinear(linear.transpose_in, linear.transpose_out, linear.segments_in, linear.segments_out,
                       list(raw_blocks(linear)), grouped=True)


class ResidualBlockLinear(torch.nn.Module):
    """first(x) + residual(y) for two BlockLinear layers with the same outputs.

    One product per output segment over the same-irrep segments of both inputs.
    """

    def __init__(self, first, residual):
        super().__init__()
        if first.segments_out != residual.segments_out:
            raise ValueError("merged_residual: the two layers have different outputs")
        self.first_transpose, self.residual_transpose = first.transpose_in, residual.transpose_in
        self.transpose_out = first.transpose_out
        self.first_count = len(first.segments_in)
        self.segments_in = tuple(first.segments_in) + tuple(residual.segments_in)
        self.segments_out = tuple(first.segments_out)
        grouped = {}
        for shift, linear in ((0, first), (self.first_count, residual)):
            for i, j, weight in raw_blocks(linear):
                grouped.setdefault(j, []).append((i + shift, weight))
        self.block_ends = tuple((tuple(i for i, _ in members), j) for j, members in grouped.items())
        self.grouped_launch = False  # grouped_launch (see block_products)
        self.sms = None
        for index, members in enumerate(grouped.values()):
            weight = torch.cat([w for _, w in members], dim=0).contiguous()
            self.register_buffer(f"weight_{index}", weight)
            self.register_buffer(f"weight_t_{index}", weight.t().contiguous())

    def forward(self, x, y):
        outputs, n, x = self.output_rows(x, y)
        return self.transpose_out(join_rows(self.segments_out, outputs, n, x))

    def output_rows(self, x, y):
        """(rows [n * dim, mul] of every output segment or None, n, x after its transpose), before the
        output segments are joined into one [n, width] tensor and ``transpose_out``."""
        x, y = self.first_transpose(x), self.residual_transpose(y)
        n = x.shape[0]
        rows = ([x[:, o:o + d * m].reshape(n * d, m) for o, m, d in self.segments_in[:self.first_count]]
                + [y[:, o:o + d * m].reshape(n * d, m) for o, m, d in self.segments_in[self.first_count:]])
        outputs = [None] * len(self.segments_out)
        lefts = [rows[i[0]] if len(i) == 1 else torch.cat([rows[k] for k in i], dim=1) for i, _ in self.block_ends]
        for (_, j), part in zip(self.block_ends, block_products(self, lefts)):
            outputs[j] = part
        return outputs, n, x


def join_rows(segments_out, outputs, n, like):
    """The [n, width] ir_mul tensor of per-segment rows (zeros for a segment without rows)."""
    pieces = [value.reshape(n, d * m) if value is not None else like.new_zeros((n, d * m))
              for (o, m, d), value in zip(segments_out, outputs)]
    return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]


def segments(irreps):
    """[(offset, mul, dim, irrep)] of each segment and the total dimension."""
    out, offset = [], 0
    for mul, ir in irreps:
        out.append((offset, int(mul), int(ir.dim), ir))
        offset += int(mul) * int(ir.dim)
    return out, offset


def is_cueq_linear(module):
    """A cuEquivariance Linear with internal shared weights of one weight class (what block_linear takes)."""
    return (type(module).__name__ == "Linear" and type(module).__module__.startswith("cuequivariance_torch")
            and bool(getattr(module, "internal_weights", False)) and getattr(module, "weight_classes", 1) == 1
            and getattr(module, "weight", None) is not None)


def block_linear(linear, grouped=False):
    """The BlockLinear of one cuEquivariance Linear, or None if it does not apply (unchecked; see the record)."""
    if not is_cueq_linear(linear):
        return None
    segments_in, dim_in = segments(linear.irreps_in)
    segments_out, _ = segments(linear.irreps_out)
    device, dtype = linear.weight.device, linear.weight.dtype
    blocks = []
    with torch.no_grad():
        for i, (offset, mul, dim, ir) in enumerate(segments_in):
            # Core layout ir_mul: component m of channel u at offset + m * mul + u; m = 0 suffices.
            basis = torch.zeros((mul, dim_in), device=device, dtype=dtype)
            basis[torch.arange(mul), offset + torch.arange(mul)] = 1
            response = linear.f([linear.weight, basis])[0]
            for j, (offset_out, mul_out, _, ir_out) in enumerate(segments_out):
                if ir_out == ir:
                    blocks.append((i, j, response[:, offset_out:offset_out + mul_out].clone()))
        return BlockLinear(linear.transpose_in, linear.transpose_out, [(o, m, d) for o, m, d, _ in segments_in],
                           [(o, m, d) for o, m, d, _ in segments_out], blocks, grouped=grouped)
