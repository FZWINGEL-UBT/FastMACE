"""Modules of the species family (OPT-T08, docs/design/opt-T08-linear-audit.md).

- ``OneHotPassthrough``: the node embedding of a model whose embedding is read only through
  species tables. It hands the one-hot ``node_attrs`` on unchanged.
- ``SpeciesTable``: a module whose output depends only on each atom's species, as a table
  ``[n_elements, width]`` gathered by species. A one-hot row picks one table row, so the gather
  equals ``onehot @ table`` exactly. The table itself was computed by the replaced modules on
  the identity rows, which can round differently from the same modules on n atoms.
- ``compose``: two block linears applied one after the other, as one block linear with the
  products of their weights (computed in float64, stored in the first's dtype). The first's
  output transpose and the second's input transpose cancel.

Each module keeps what it replaced as ``reference`` (outside the module tree, so neither the
compiled graph nor the parameter list sees it) for the record's self-check.
"""

from __future__ import annotations

import torch

from fastmace.rewrites.linear.modules import BlockLinear, raw_blocks


class OneHotPassthrough(torch.nn.Module):
    """The node embedding under ``species_tables``: returns ``node_attrs`` itself."""

    def forward(self, node_attrs):
        return node_attrs


class SpeciesTable(torch.nn.Module):
    """``table[species]`` for one-hot ``node_attrs``; ``reference(node_attrs)`` is what it replaced."""

    def __init__(self, table, reference=None):
        super().__init__()
        self.register_buffer("table", table.detach().clone().contiguous())
        self._reference = (reference,)

    @property
    def reference(self):
        return self._reference[0]

    def forward(self, node_attrs):
        return self.table[node_attrs.argmax(dim=-1)]


class SpeciesResidualBlockLinear(torch.nn.Module):
    """A ``ResidualBlockLinear`` whose residual input is a species table's output (``species_residual``).

    ``first(x) + residual(table[species])``: the residual's products are computed once per species (in float64,
    stored in the model's dtype) and gathered by the one-hot ``node_attrs`` that the interaction passes in place of
    the residual input. ``first``'s products are the merged module's with the residual rows of each group removed,
    so the products lose those inputs and, as ``node_attrs`` carries no gradient, their backward.
    """

    def __init__(self, merged, input_table):
        from fastmace.rewrites.linear.modules import ResidualBlockLinear

        super().__init__()
        if not isinstance(merged, ResidualBlockLinear):
            raise TypeError(f"species_residual needs a ResidualBlockLinear, not {type(merged).__name__}")
        self.first_transpose, self.residual_transpose = merged.first_transpose, merged.residual_transpose
        self.transpose_out = merged.transpose_out
        self.first_count = merged.first_count
        self.segments_in, self.segments_out = merged.segments_in, merged.segments_out
        self.grouped_launch, self.sms, self.tf32x3 = merged.grouped_launch, merged.sms, merged.tf32x3
        first, residual = {}, {}
        for index, (members, j) in enumerate(merged.block_ends):
            weight = getattr(merged, f"weight_{index}")
            start = 0
            for k in members:
                mul = self.segments_in[k][1]
                (first if k < self.first_count else residual).setdefault(j, []).append((k, weight[start:start + mul]))
                start += mul
        self.block_ends = tuple((tuple(k for k, _ in members), j) for j, members in first.items())
        for index, members in enumerate(first.values()):
            weight = torch.cat([w for _, w in members], dim=0).contiguous()
            self.register_buffer(f"weight_{index}", weight)
            self.register_buffer(f"weight_t_{index}", weight.t().contiguous())
        with torch.no_grad():
            y = self.residual_transpose(input_table).double()
            species = y.shape[0]
            rows = {k: y[:, o:o + d * m].reshape(species * d, m) for k, (o, m, d) in enumerate(self.segments_in)
                    if k >= self.first_count}
            self.residual_segments = tuple(residual)
            for j, members in residual.items():
                _, mul_out, dim_out = self.segments_out[j]
                table = sum(rows[k] @ w.double() for k, w in members)
                self.register_buffer(f"residual_{j}", table.reshape(species, dim_out * mul_out).to(weight.dtype))
        self._reference = (merged, input_table)

    @property
    def reference(self):
        return self._reference

    def forward(self, x, node_attrs):
        from fastmace.rewrites.linear.modules import join_rows

        outputs, n, x = self.output_rows(x, node_attrs)
        return self.transpose_out(join_rows(self.segments_out, outputs, n, x))

    def output_rows(self, x, node_attrs):
        """As ``ResidualBlockLinear.output_rows``, with the one-hot ``node_attrs`` in place of the residual input."""
        from fastmace.rewrites.linear.modules import block_products

        x = self.first_transpose(x)
        n = x.shape[0]
        rows = [x[:, o:o + d * m].reshape(n * d, m) for o, m, d in self.segments_in[:self.first_count]]
        outputs = [None] * len(self.segments_out)
        lefts = [rows[i[0]] if len(i) == 1 else torch.cat([rows[k] for k in i], dim=1) for i, _ in self.block_ends]
        for (_, j), part in zip(self.block_ends, block_products(self, lefts)):
            outputs[j] = part
        species = node_attrs.argmax(dim=-1)
        for j in self.residual_segments:
            _, mul_out, dim_out = self.segments_out[j]
            add = getattr(self, f"residual_{j}")[species].reshape(n * dim_out, mul_out)
            outputs[j] = add if outputs[j] is None else outputs[j] + add
        return outputs, n, x


def compose(first, second):
    """The BlockLinear of ``second(first(x))`` for two BlockLinear layers (first's outputs = second's inputs)."""
    if tuple(first.segments_out) != tuple(second.segments_in):
        raise ValueError("compose: the first layer's outputs are not the second layer's inputs")
    dtype = first.weight_0.dtype
    products = {}
    for i, j, left in raw_blocks(first):
        for j2, k, right in raw_blocks(second):
            if j2 == j:
                term = left.double() @ right.double()
                products[(i, k)] = term if (i, k) not in products else products[(i, k)] + term
    blocks = [(i, k, weight.to(dtype)) for (i, k), weight in products.items()]
    composed = BlockLinear(first.transpose_in, second.transpose_out, first.segments_in, second.segments_out,
                           blocks, grouped=True)
    composed._reference = (first, second)
    return composed


class HeadActivation(torch.nn.Module):
    """The readout's scalar activation on the active head's hidden columns only (TAIL_HEAD prototype)."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


def slice_columns(layer, start, width):
    """A BlockLinear that keeps columns ``start:start + width`` of a one-segment scalar layer's output."""
    (offset, mul, dim), = layer.segments_out
    if offset != 0 or dim != 1:
        raise ValueError("slice_columns: the layer's output is not one scalar segment")
    weight = torch.zeros((mul, width), dtype=layer.weight_0.dtype, device=layer.weight_0.device)
    weight[start:start + width] = torch.eye(width, dtype=weight.dtype, device=weight.device)
    return BlockLinear(layer.transpose_out, layer.transpose_out, layer.segments_out, ((0, width, 1),),
                       [(0, 0, weight)])


class Chain:
    """``second(first(x))`` as a plain picklable callable (a self-check reference, not a module)."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def __call__(self, x):
        return self.second(self.first(x))


def reference_of(module):
    """The callable a rewritten module replaced: ``SpeciesTable.reference`` or the composition's pair."""
    pair = getattr(module, "_reference", (None,))
    return Chain(*pair) if len(pair) == 2 else pair[0]


class PadColumns(torch.nn.Module):
    """``[x | 0]`` with ``extra`` zero columns: ``products.0.linear`` under ``fold_product_linear``, so that
    adding the first interaction's ``[0 | one-hot]`` skip term carries the species next to the contraction."""

    def __init__(self, extra):
        super().__init__()
        self.extra = int(extra)

    def forward(self, x):
        return torch.nn.functional.pad(x, (0, self.extra))


class SpeciesCarrier:
    """``[0 | one-hot]``: the reference of the first interaction's skip table under ``fold_product_linear``."""

    def __init__(self, width):
        self.width = int(width)

    def __call__(self, node_attrs):
        return torch.nn.functional.pad(node_attrs, (self.width, 0))


class ConsumerReference:
    """``consumer(product_linear(a) + skip[species])`` for ``x = [a | one-hot]``: what a FoldedConsumer replaced."""

    def __init__(self, product_linear, consumer, skip_table, width):
        self.product_linear, self.consumer = product_linear, consumer
        self.skip_table, self.width = skip_table, int(width)

    def __call__(self, x):
        left, onehot = x[:, :self.width], x[:, self.width:]
        return self.consumer(self.product_linear(left) + self.skip_table[onehot.argmax(dim=-1)])


class FoldedConsumer(torch.nn.Module):
    """A linear consumer of layer 0's output with ``products.0.linear`` folded in (``fold_product_linear``).

    Its input is ``[a | one-hot]``: the symmetric contraction's output ``a`` and the species. It returns
    ``folded(a) + table[species]``, with ``folded`` the composition of ``products.0.linear`` and the
    consumer and ``table`` the consumer applied to the first interaction's per-species skip term.
    """

    def __init__(self, folded, width, table, reference=None):
        super().__init__()
        self.folded = folded
        self.width = int(width)
        self.register_buffer("table", table.detach().clone().contiguous())
        self._reference = (reference,)

    @property
    def reference(self):
        return self._reference[0]

    def forward(self, x):
        species = x.narrow(1, self.width, x.shape[1] - self.width).argmax(dim=-1)
        return self.folded(x.narrow(1, 0, self.width)) + self.table[species]
