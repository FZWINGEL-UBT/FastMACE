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
- ``e3nn_block_linear``: an e3nn ``o3.Linear`` (MACE-POLAR's field blocks keep them after the cuEquivariance
  conversion) as the same ``BlockLinear``, with ``LayoutTranspose`` between e3nn's mul_ir layout and the ir_mul
  core, so that ``tf32x3_gemms`` reaches its products. Exact up to rounding.
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
    shape rule calls small (``tile_rule.uses_triton`` false: too few row tiles for the GPU) and that have the
    same weight shape run together in one Triton launch, if there are at least two of them. Equal K and N give
    every tile of the launch, forward and input gradient, the same work: a larger-K product's tiles would run
    several times longer and set the launch's length, where cuBLAS splits K (OPT-T03b diagnosis). The others,
    and everything on the CPU or in other dtypes (``module.sms`` None), keep cuBLAS.

    With ``module.tf32x3`` (record ``tf32x3_gemms``, OPT4-T03) every product and its input gradient run as
    3xTF32 (``fastmace.kernels.triton.tf32x3_gemm``)."""
    count = len(lefts)
    weights = [getattr(module, f"weight_{k}") for k in range(count)]
    weights_t = [getattr(module, f"weight_t_{k}") for k in range(count)]
    if getattr(module, "tf32x3", False):
        from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

        return [tf32x3_matmul(x, w, wt) for x, w, wt in zip(lefts, weights, weights_t)]
    parts = [None] * count
    sms = getattr(module, "sms", None)
    if getattr(module, "grouped_launch", False) and sms:
        from fastmace.kernels.triton import tile_rule

        groups = {}
        for k, x in enumerate(lefts):
            if not tile_rule.uses_triton(x.shape[0], sms):
                groups.setdefault(tuple(weights[k].shape), []).append(k)
        for members in groups.values():
            if len(members) < 2:
                continue
            from fastmace.kernels.triton.grouped_gemm import grouped_products

            done = grouped_products([lefts[k] for k in members], [weights[k] for k in members],
                                    [weights_t[k] for k in members])
            for k, part in zip(members, done):
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
        self.tf32x3 = False  # tf32x3_gemms: every product as 3xTF32 (block_products)
        self.segment_plan = None  # tf32x3 on CUTLASS with SEGMENT_ROWS: the blocks' plan (cutlass.segment_rows)
        self.linear_div = False  # tf32x3 on CUTLASS with LINEAR_DIV: ``divided`` (cutlass.linear_div; idea 31)
        if self.grouped:  # one product per output segment
            blocks = _group(blocks)
            self.block_ends = tuple((i, j) for i, j, _ in blocks)
        else:
            self.block_ends = tuple((int(i), int(j)) for i, j, _ in blocks)
        for index, (_, _, weight) in enumerate(blocks):
            self.register_buffer(f"weight_{index}", weight.contiguous())
            self.register_buffer(f"weight_t_{index}", weight.t().contiguous())

    def forward(self, x):
        if isinstance(x, tuple):  # wprime_backward.BLOCK_MESSAGE: the grouped products' left factors
            n = x[0].shape[0] // self.segments_in[self.block_ends[0][0][0]][2]
            outputs = [None] * len(self.segments_out)
            for (_, j), part in zip(self.block_ends, block_products(self, list(x))):
                outputs[j] = part if outputs[j] is None else outputs[j] + part
            return self.transpose_out(join_rows(self.segments_out, outputs, n, x[0]))
        x = self.transpose_in(x)
        if self.segment_plan is not None and self.tf32x3:  # products on x itself, results written in place
            from fastmace.kernels.cutlass.segment_rows import segment_rows

            count = len(self.block_ends)
            return self.transpose_out(segment_rows(
                x, [getattr(self, f"weight_{k}") for k in range(count)],
                [getattr(self, f"weight_t_{k}") for k in range(count)], self.segment_plan, self.width_out))
        n = x.shape[0]
        rows = [x[:, offset:offset + dim * mul].reshape(n * dim, mul)
                for offset, mul, dim in self.segments_in]
        return self.transpose_out(join_rows(self.segments_out, self.products(rows), n, x))

    @property
    def width_out(self):
        return sum(m * d for _, m, d in self.segments_out)

    def products(self, rows):
        """Per output segment, its rows [n * dim, mul] (None where no product reaches it), from the
        rows of every input segment (in the ir_mul core layout, after ``transpose_in``)."""
        outputs = [None] * len(self.segments_out)
        lefts = self.left_factors(rows)
        for (_, j), part in zip(self.block_ends, block_products(self, lefts)):
            outputs[j] = part if outputs[j] is None else outputs[j] + part
        return outputs

    def left_factors(self, rows):
        """Each product's left factor from the input segments' rows (grouped segments joined along the channels)."""
        lefts = []
        for i, _ in self.block_ends:
            if isinstance(i, tuple):  # grouped
                lefts.append(rows[i[0]] if len(i) == 1 else torch.cat([rows[k] for k in i], dim=1))
            else:
                lefts.append(rows[i])
        return lefts

    def divided(self, x, density):
        """``self(x) / (density + 1)`` with the division in the products' epilogue and the outputs written in place
        (idea 31, ``kernels.cutlass.linear_div``; bitwise the same), or None where it does not apply: ``linear_div``
        off (set at build by ``precision.route``: LINEAR_DIV on and 3xTF32 on CUTLASS), a layout transpose,
        SEGMENT_ROWS, an output segment without exactly one product, or operands the kernel does not take."""
        if not (self.tf32x3 and self.linear_div) or self.grouped_launch or not _identity(self.transpose_out):
            return None
        if isinstance(x, tuple):  # wprime_backward.BLOCK_MESSAGE: the grouped products' left factors
            lefts = list(x)
            n = x[0].shape[0] // self.segments_in[self.block_ends[0][0][0]][2]
        else:
            if self.segment_plan is not None or not _identity(self.transpose_in):
                return None
            n = x.shape[0]
            lefts = self.left_factors([x[:, offset:offset + dim * mul].reshape(n * dim, mul)
                                       for offset, mul, dim in self.segments_in])
        outputs = [j for _, j in self.block_ends]
        if sorted(outputs) != list(range(len(self.segments_out))) or density.shape[0] != n:
            return None
        count = len(self.block_ends)
        weights = [getattr(self, f"weight_{k}") for k in range(count)]
        for left, weight, j in zip(lefts, weights, outputs):
            _, mul, dim = self.segments_out[j]
            if left.shape[0] != n * dim or weight.shape[1] != mul:
                return None
        offsets = [self.segments_out[j][0] for j in outputs]
        from fastmace.kernels.cutlass import linear_div

        if not linear_div.applies(lefts, weights, density, offsets, self.width_out):
            return None
        return linear_div.tf32x3_linear_div(lefts, weights, [getattr(self, f"weight_t_{k}") for k in range(count)],
                                            offsets, self.width_out, density)


def _identity(transpose):
    """Whether a cuEquivariance layout transpose (``TransposeIrrepsLayout``) leaves its input as it is."""
    return isinstance(transpose, torch.nn.Identity) or isinstance(getattr(transpose, "f", None), torch.nn.Identity)


def reshape_view(reshape, x):
    """MACE's ``reshape_irreps(x)`` as a view of ``x`` ([n, width] -> [n, sum of dims, mul]) where that is its
    value: cuEquivariance's ir_mul layout with one multiplicity (its ``torch.cat`` of [n, dim, mul] pieces along
    the dims is then x's own memory order); None otherwise."""
    config = getattr(reshape, "cueq_config", None)
    muls = getattr(reshape, "muls", None)
    if (type(reshape).__name__ != "reshape_irreps" or config is None or getattr(config, "layout_str", None) != "ir_mul"
            or not muls or len(set(muls)) != 1 or not x.is_contiguous()):
        return None
    dims = sum(reshape.dims)
    if x.dim() != 2 or x.shape[1] != dims * muls[0]:
        return None
    return x.view(x.shape[0], dims, muls[0])


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
        self.tf32x3 = False  # tf32x3_gemms (see block_products)
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


class LayoutTranspose(torch.nn.Module):
    """e3nn's mul_ir layout to the ir_mul core layout (``to_ir_mul``) or back, per segment ``(offset, mul, dim)``
    of rows ``[n, width]``; scalar segments are copied as they are."""

    def __init__(self, segments, to_ir_mul):
        super().__init__()
        self.segments = tuple((int(o), int(m), int(d)) for o, m, d in segments)
        self.to_ir_mul = bool(to_ir_mul)

    def forward(self, x):
        n = x.shape[0]
        parts = []
        for offset, mul, dim in self.segments:
            part = x[:, offset:offset + mul * dim]
            if dim > 1:
                shape = (n, mul, dim) if self.to_ir_mul else (n, dim, mul)
                part = part.reshape(shape).transpose(1, 2).reshape(n, mul * dim)
            parts.append(part)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


# e3nn Linears below this many weights stay as they are (a readout to a few channels gains nothing from
# per-segment products and pays the layout transposes).
E3NN_MIN_WEIGHTS = 1 << 14


def is_e3nn_linear(module):
    """An e3nn ``o3.Linear`` with internal shared weights, no bias and no extra feature dimensions, of at least
    ``E3NN_MIN_WEIGHTS`` weights (what ``e3nn_block_linear`` takes)."""
    if type(module).__name__ != "Linear" or not type(module).__module__.startswith("e3nn"):
        return False
    weight = getattr(module, "weight", None)
    return (bool(getattr(module, "internal_weights", False)) and getattr(module, "bias_numel", 1) == 0
            and weight is not None and weight.dim() == 1 and weight.numel() >= E3NN_MIN_WEIGHTS)


def e3nn_block_linear(linear, grouped=False):
    """The BlockLinear of one e3nn ``o3.Linear``, or None if it does not apply: each instruction's weights
    ``[mul_in, mul_out]`` read off the module (its path normalisation included) by its response to channel basis
    rows at component 0, mul_ir layout transposes around the ir_mul core."""
    if not is_e3nn_linear(linear):
        return None
    segments_in, dim_in = segments(linear.irreps_in)
    segments_out, _ = segments(linear.irreps_out)
    device, dtype = linear.weight.device, linear.weight.dtype
    blocks = []
    with torch.no_grad():
        for instruction in linear.instructions:
            i, j = int(instruction.i_in), int(instruction.i_out)
            offset, mul, dim, _ = segments_in[i]
            offset_out, mul_out, dim_out, _ = segments_out[j]
            basis = torch.zeros((mul, dim_in), device=device, dtype=dtype)
            basis[torch.arange(mul), offset + torch.arange(mul) * dim] = 1  # mul_ir: channel u, component 0
            # The response of instruction (i, j) alone: the other instructions read other input segments or
            # write other output segments, and those of this pair add to the same block.
            response = linear(basis)
            blocks.append((i, j, response[:, offset_out:offset_out + mul_out * dim_out:dim_out].clone()))
    blocks = list({(i, j): (i, j, w) for i, j, w in blocks}.values())  # one block per pair
    core_in = [(o, m, d) for o, m, d, _ in segments_in]
    core_out = [(o, m, d) for o, m, d, _ in segments_out]
    return BlockLinear(LayoutTranspose(core_in, True), LayoutTranspose(core_out, False), core_in, core_out, blocks,
                       grouped=grouped)


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
