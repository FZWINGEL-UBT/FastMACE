"""Modules of the fusion family: the gated chain of an MH-1-class interaction on per-segment rows.

``GatedRows`` computes ``linear_2(gate(linear_1(x) + linear_res(y)))`` of a
``RealAgnosticResidualNonLinearInteractionBlock`` from the merged residual products
(``ResidualBlockLinear``), MACE's ``GatedEquivariantBlock`` and the block linear ``linear_2``. The
products of each irrep segment stay as rows ``[atoms * (2l + 1), channels]`` from the residual
products through the gate to ``linear_2``. The stock chain instead joins them into one
``[atoms, width]`` tensor for the gate, which ``linear_2`` slices and copies into rows again; the
backward does the same with zero-filled wide gradients (``docs/design/opt-T02-fusion.md``, stage A).

The arithmetic is the gate's, element by element (mace-torch 0.3.16
``GatedEquivariantBlock.forward``, ir_mul branch, pinned by sha256 in the record): the scalars and
the gates are column ranges of the 0e rows, and a gated segment's rows viewed as
``[atoms, 2l + 1, channels]`` are multiplied by the atom's gates. The products are the block
linears' own. Only the copies between them go away.
"""

from __future__ import annotations

import torch


def _same_layout(transpose, width, like):
    """True if a cuEquivariance layout transpose leaves a [2, width] input unchanged (ir_mul in and out)."""
    if isinstance(transpose, torch.nn.Identity):
        return True
    probe = torch.arange(2 * width, dtype=like.dtype, device=like.device).reshape(2, width)
    try:
        out = transpose(probe)
    except Exception:  # noqa: BLE001  (a transpose that needs another width is a real one)
        return False
    return out.shape == probe.shape and bool(torch.equal(out, probe))


def _segment_of(segments, start, length):
    """(index, column offset within its rows) of the segment holding columns [start, start + length)."""
    for index, (offset, mul, dim) in enumerate(segments):
        if dim == 1 and offset <= start and start + length <= offset + mul:
            return index, start - offset
    return None


class GatedRows(torch.nn.Module):
    """``second(gate(first(x, y)))`` on per-segment rows; see the module docstring."""

    def __init__(self, first, second, scalars, gates, gated, act_scalar, scalar_cst, act_gate, gate_cst):
        super().__init__()
        self.first, self.second = first, second
        self.scalars = scalars  # (segment of first's outputs, column offset, columns)
        self.gates = gates  # (segment, column offset, columns)
        self.gated = tuple(gated)  # per gated segment: (segment of first's outputs, gate offset, mul, dim)
        self.act_scalar, self.scalar_cst = act_scalar, scalar_cst
        self.act_gate, self.gate_cst = act_gate, gate_cst
        self.joined = None  # segment_gemm: (input index, output offset, dim) per product of ``second``
        self.sms = None  # segment_gemm: the device's SM count for the shape rule (None: not float32 on CUDA)
        self.gate_prologue = False  # gate_prologue: the Triton products apply the gate as they load their rows

    def join_in_place(self):
        """Let ``second``'s products write straight into its joined output (``segment_gemm``); returns
        (True, "ok") or (False, reason). Needs one single-input product per output segment."""
        second = self.second
        plan, seen = [], set()
        for index, (i, j) in enumerate(second.block_ends):
            if isinstance(i, tuple):
                if len(i) != 1:
                    return False, f"output segment {j} of linear_2 sums {len(i)} grouped inputs"
                i = i[0]
            if j in seen:
                return False, f"output segment {j} of linear_2 has several products"
            seen.add(j)
            offset, mul, dim = second.segments_out[j]
            if second.segments_in[i][2] != dim:
                return False, f"product {index} of linear_2 changes the irrep dimension"
            plan.append((i, offset, dim, index))
        from fastmace.kernels.triton import tile_rule

        weight = second.weight_0
        # Read once at setup: the forward evaluates the rule while torch.compile traces it.
        self.sms = tile_rule.device_sms(weight.device) if weight.dtype == torch.float32 else None
        self.joined = tuple(plan)
        return True, "ok"

    def _any_triton(self, rows, n):
        """Whether the shape rule picks the Triton GEMM for any of ``second``'s products at ``n`` atoms; if
        not, the joined op would only add launches, so the stock products and join run (NaCl-sized cells)."""
        from fastmace.kernels.triton import tile_rule

        return any(tile_rule.uses_triton(n * dim, self.sms) for _, _, dim, _ in self.joined)

    def forward(self, x, y):
        outputs, n, like = self.first.output_rows(x, y)
        index, start, length = self.scalars
        scalars = outputs[index].narrow(1, start, length)
        if self.act_scalar is not None:
            scalars = self.act_scalar(scalars) * self.scalar_cst
        joined = self.joined is not None and self._any_triton(None, n)  # segment_gemm: products write in place
        prologue = set()  # gate_prologue: inputs of second whose Triton GEMM applies the gate as it loads them
        if joined and self.gate_prologue:
            from fastmace.kernels.triton import tile_rule

            prologue = {i for i, _, dim, _ in self.joined if i > 0 and tile_rule.uses_triton(n * dim, self.sms)}
        rows, row_gates = [scalars], {}
        if self.gated:
            index, start, length = self.gates
            gates = outputs[index].narrow(1, start, length)
            if self.act_gate is not None:
                gates = self.act_gate(gates) * self.gate_cst
            for i, (segment, offset, mul, dim) in enumerate(self.gated, start=1):
                gate = gates.narrow(-1, offset, mul)
                if i in prologue:
                    rows.append(outputs[segment].reshape(n * dim, mul))
                    row_gates[i] = gate
                    continue
                gated = outputs[segment].view(n, dim, mul)
                rows.append((gated * gate.unsqueeze(-2)).reshape(n * dim, mul))
        from fastmace.rewrites.linear.modules import join_rows

        second = self.second
        if joined:
            from fastmace.kernels.triton.segment_gemm import joined_products

            width = sum(mul * dim for _, mul, dim in second.segments_out)
            out = joined_products([rows[i] for i, _, _, _ in self.joined],
                                  [getattr(second, f"weight_{k}") for _, _, _, k in self.joined],
                                  [getattr(second, f"weight_t_{k}") for _, _, _, k in self.joined],
                                  [offset for _, offset, _, _ in self.joined], [dim for _, _, dim, _ in self.joined],
                                  width, gates=[row_gates.get(i) for i, _, _, _ in self.joined] if row_gates else None)
            return second.transpose_out(out)
        return second.transpose_out(join_rows(second.segments_out, second.products(rows), n, like))


class IrrepsView(torch.nn.Module):
    """MACE's ``reshape_irreps`` for an ir_mul tensor whose segments all have the same multiplicity: the
    ``[atoms, sum (2l+1) mul]`` tensor is already laid out as ``[atoms, sum (2l+1), mul]``, so the slices,
    reshapes and concatenation along the component axis are a view (``segment_gemm``: the joined output
    of ``linear_2`` goes to the symmetric contraction without a copy)."""

    def __init__(self, components, mul):
        super().__init__()
        self.components, self.mul = int(components), int(mul)

    def forward(self, tensor):
        return tensor.reshape(tensor.shape[0], self.components, self.mul)


def irreps_view(reshape):
    """The ``IrrepsView`` equivalent to a ``reshape_irreps`` module, or (None, reason)."""
    if type(reshape).__name__ != "reshape_irreps":
        return None, f"the interaction's reshape is a {type(reshape).__name__}"
    config = getattr(reshape, "cueq_config", None)
    if config is None or getattr(config, "layout_str", "mul_ir") != "ir_mul":
        return None, "the reshape is not in the ir_mul layout"
    muls = set(int(m) for m in reshape.muls)
    if len(muls) != 1:
        return None, f"the reshaped irreps have different multiplicities {sorted(muls)}"
    return IrrepsView(sum(int(d) for d in reshape.dims), muls.pop()), "ok"


def gated_rows(first, gate, second):
    """The ``GatedRows`` of one interaction's merged residual products, gate and ``linear_2``, or
    (None, reason) when the chain does not have that structure."""
    from fastmace.rewrites.linear import modules as lm

    if not isinstance(first, lm.ResidualBlockLinear):
        return None, "no merged residual products (merged_residual)"
    if not isinstance(second, lm.BlockLinear):
        return None, "linear_2 is not a block linear (block_linear)"
    if type(gate).__name__ != "GatedEquivariantBlock":
        return None, f"the nonlinearity is a {type(gate).__name__}, not MACE's GatedEquivariantBlock"
    if not getattr(gate, "_layout_is_ir_mul", False):
        return None, "the gate works in the mul_ir layout"
    width_out = sum(mul * dim for _, mul, dim in first.segments_out)
    width_in = sum(mul * dim for _, mul, dim in second.segments_in)
    if not (_same_layout(first.transpose_out, width_out, first.weight_0)
            and _same_layout(second.transpose_in, width_in, second.weight_0)):
        return None, "the block linears transpose between layouts around the gate"
    segments = first.segments_out
    scalars = _segment_of(segments, gate._s_start, gate._s_len)
    if scalars is None or gate._s_len == 0:
        return None, "the gate's scalars are not columns of one 0e output segment"
    gates = (0, 0, 0)
    gated = []
    if gate._gated_info:
        found = _segment_of(segments, gate._g_start, gate._g_len)
        if found is None:
            return None, "the gate's gates are not columns of one 0e output segment"
        gates = (*found, gate._g_len)
        starts = {offset: index for index, (offset, _, _) in enumerate(segments)}
        for gd_start, gd_len, ir_dim, mul, g_off in gate._gated_info:
            index = starts.get(gd_start)
            if index is None or segments[index] != (gd_start, mul, ir_dim) or gd_len != mul * ir_dim:
                return None, f"a gated block at column {gd_start} is not one whole output segment"
            gated.append((index, g_off, mul, ir_dim))
    expected = [(gate._s_len, 1)] + [(mul, dim) for _, _, mul, dim in gated]
    if [(mul, dim) for _, mul, dim in second.segments_in] != expected:
        return None, "linear_2's input segments are not the gate's outputs in order"
    return GatedRows(first, second, (*scalars, gate._s_len), gates, gated, gate._act_scalar, gate._scalar_cst,
                     gate._act_gate, gate._gate_cst), "ok"


__all__ = ["GatedRows", "gated_rows"]
