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
