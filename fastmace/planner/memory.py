"""Rule 12's memory estimate for the MACE model (SPEC §9.3 rule 12, decision 51; plan task 5-T03).

The estimate counts what a force call keeps on the device, from the model's own widths
(``capabilities.ModelDims``) and the plan's sizes. It contains no GPU table:

- **weights**: the parameters at the force precision;
- **activations**: per interaction, the per-edge tensors of stock MACE (the gathered sender features, the
  radial network's layers, the tensor product's weights and messages) times the edge count, and the
  per-node tensors (node features in, the scattered messages, the product's terms, the features out)
  times the atom count; once for the edge geometry (vectors, lengths, spherical harmonics, radial
  basis). The force backward keeps every forward tensor; its transient is the messages' gradient of one
  interaction at a time, the widest one (tiny-model measurements, ``tests/cpu/test_memory_estimate.py``);
- **rewrites**: each enabled rewrite's declared term (``Rewrite.memory``, bytes per edge and per node; a
  negative term is a stock tensor the rewrite does not materialise);
- **neighbour list**: the MACE pair buffers at their capacity;
- **D3 list**: its buffers and build transient (``neighbors.adaptive.rule_12_d3``), where D3 is on;
- **workspace**: the library workspaces a capture allocates (cuBLAS, 32-64 MiB; ``engine/graph.py``).

The activations and rewrite terms are multiplied by ``SAFETY``, one factor for every model and GPU,
fitted on the 19 labels (SPEC §9.3 rule 12 as amended 23 October 2026; fit and residuals in
``docs/validation/5-T03.md``).

The graph tier's variants share one graph pool (``engine/graph.py``), so the largest variant counts once.

Decision 51 (D1 (a)) acts on ``MemoryEstimate``: above ``MARGIN`` of the free GPU memory, the planner turns
off the rewrites whose terms add edge-sized buffers, largest first, until it fits; never a rewrite the
caller set explicitly, never rule 11's sizes, the tier or the precision. Still above, it keeps the plan
and warns.

This module imports nothing from torch or MACE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Decision 51: the estimate must stay below this share of the free GPU memory.
MARGIN = 0.9
# One factor on the activations and rewrite terms, for every model and GPU (SPEC §9.3 rule 12), fitted on
# the 19 labels (docs/validation/5-T03.md). Provisional until that fit: the tiny models' CPU peaks are
# 0.84-0.91 of the unscaled activations (the remainder is small per-edge copies the structure leaves out).
SAFETY = 1.25
# cuBLAS and cuSOLVER workspaces a capture allocates outside the graph pool (engine/graph.py), bytes.
WORKSPACE_BYTES = 64 * 2**20
# The MACE pair list per capacity slot (neighbors/skin.py ``_allocate_now``): candidate and model edge
# indices (2 x 2 int64), four shift tensors at the list's float width and the float64 candidate shifts.
_INDEX_BYTES_PER_SLOT = 2 * 2 * 8
_SHIFT_FLOATS_PER_SLOT = 4 * 3
_SHIFT64_BYTES_PER_SLOT = 3 * 8
# Per-edge geometry once per force call: vectors (3), lengths (1), the cutoff envelope (1).
_GEOMETRY_FLOATS = 5
DTYPE_BYTES = {"float32": 4, "float64": 8}


@dataclass(frozen=True)
class MemoryTerm:
    """A rewrite's memory beyond stock MACE (``Rewrite.memory``), in bytes per edge and per node; negative
    for a stock tensor it does not materialise."""

    edge_bytes: float = 0.0
    node_bytes: float = 0.0

    def bytes(self, edges: int, nodes: int) -> float:
        return self.edge_bytes * edges + self.node_bytes * nodes

    @property
    def edge_sized(self) -> bool:
        """Does it add an edge-sized buffer (decision 51's candidates)?"""
        return self.edge_bytes > 0


@dataclass(frozen=True)
class MemoryEstimate:
    """Rule 12's estimate (bytes): the larger of the run's and the build's.

    ``terms`` are the run's parts (``run`` is their sum); ``build_terms`` the build's (``build``: the pristine
    and the execution copy of the weights, and the largest self-check transient). ``total`` is the larger
    of the two; ``resident`` the part already on the device when the plan is made (the weights of a loaded
    model), so ``needed`` is what the build or run still allocates."""

    terms: tuple[tuple[str, int], ...]
    edges: int
    nodes: int
    capacity: int
    dtype: str
    resident: int = 0
    rewrite_terms: tuple[tuple[str, int], ...] = field(default=())
    build_terms: tuple[tuple[str, int], ...] = field(default=())

    @property
    def run(self) -> int:
        return sum(value for _, value in self.terms)

    @property
    def build(self) -> int:
        return sum(value for _, value in self.build_terms)

    @property
    def total(self) -> int:
        return max(self.run, self.build)

    @property
    def needed(self) -> int:
        return self.total - self.resident

    def term(self, name: str) -> int:
        return dict(self.terms).get(name, 0)

    def text(self, free_bytes: int | None = None) -> str:
        """One line for the plan's ``memory`` decision and the report."""
        building = self.build > self.run
        terms = self.build_terms if building else self.terms
        largest = sorted((item for item in terms if item[1] > 0), key=lambda item: -item[1])[:3]
        parts = ", ".join(f"{name} {value / 2**20:.0f} MiB" for name, value in largest)
        out = (f"predicted {self.total / 2**30:.2f} GiB ({'build' if building else 'run'}; run "
               f"{self.run / 2**30:.2f}, build {self.build / 2**30:.2f}) at {self.edges} edges, {self.nodes} "
               f"atoms, {self.dtype} ({parts})")
        if free_bytes:
            out += f"; needs {self.needed / 2**30:.2f} of {free_bytes / 2**30:.2f} GiB free"
        return out


def model_edges(natoms: int, volume_A3: float, r_max_A: float, bucket_fraction: float) -> int:
    """The edges the model reads: both directions of every pair within ``r_max`` at uniform density, plus one
    edge bucket of padding (``neighbors.buckets``)."""
    from fastmace.neighbors.d3_list import estimated_pairs

    return int(math.ceil(2.0 * estimated_pairs(natoms, volume_A3, r_max_A) * (1.0 + bucket_fraction)))


def list_capacity(natoms: int, volume_A3: float, radius_A: float, headroom: float, multiple: int = 1) -> int:
    """The MACE pair list's capacity: both directions within cutoff + skin, at the pair-buffer headroom."""
    from fastmace.neighbors.d3_list import estimated_pairs

    needed = 2.0 * estimated_pairs(natoms, volume_A3, radius_A)
    return int(multiple * max(1, math.ceil(needed * headroom / multiple)))


def _network_floats(widths, saved: int) -> int:
    """Floats per edge a radial network keeps for the backward: its input, ``saved`` per hidden unit, its
    output."""
    if not widths:
        return 0
    return widths[0] + saved * sum(widths[1:-1]) + (widths[-1] if len(widths) > 1 else 0)


def interaction_floats(dims, fused_conv: bool = False) -> list[tuple[int, int]]:
    """(floats per edge, floats per node) that each interaction keeps for the force backward.

    ``fused_conv``: the convolution is one kernel (OEq's in the hybrid backend, cuEq's fused convolution in
    the plain one) that neither gathers the sender features nor stores the messages per edge.
    """
    out = []
    for block in dims.interactions:
        # sender features gathered; each radial network's input, saved hidden activations and output (the
        # convolution's output is the tensor product's weights); the messages per edge.
        edge = _network_floats(block.radial, block.radial_saved) + (0 if fused_conv else block.sender + block.messages)
        edge += sum(_network_floats(widths, saved) for widths, saved in block.other_radials)
        # node features in, scattered messages, the linear after them, the product's terms and its output.
        node = (block.node_in + block.messages + block.target + block.correlation * block.target
                + max(block.hidden, block.node_in))
        out.append((edge, node))
    return out


def estimate(dims, *, edges: int, nodes: int, dtype: str, capacity: int = 0, list_dtype: str | None = None,
             rewrite_terms: tuple[tuple[str, MemoryTerm], ...] = (), d3=None, resident_weights: bool = False,
             workspace: bool = True, fused_conv: bool = False,
             selfcheck_terms: tuple[tuple[str, int], ...] = ()) -> MemoryEstimate:
    """Rule 12's estimate for one force call of a model with ``dims`` on ``edges`` and ``nodes``.

    ``dtype`` is the force precision; ``capacity`` the MACE pair list's slots (0: no list), whose shifts are
    ``list_dtype`` wide (default ``dtype``); ``rewrite_terms`` the enabled rewrites' ``(name, MemoryTerm)``;
    ``d3`` a ``neighbors.adaptive.Rule12D3`` or ``None``; ``resident_weights`` whether the weights are on
    the device already; ``fused_conv`` as ``interaction_floats``; ``selfcheck_terms`` the enabled rewrites'
    ``(name, bytes)`` of their self-checks' transients (``Rewrite.selfcheck_memory``), of which the build
    holds the largest beside both copies of the weights.
    """
    width = DTYPE_BYTES[dtype]
    floats = interaction_floats(dims, fused_conv)
    first = dims.interactions[0] if dims.interactions else None
    shared = _GEOMETRY_FLOATS + (first.edge_attrs + first.edge_feats if first is not None else 0)
    per_layer = [edge * edges + node * nodes for edge, node in floats]
    # The backward's transient, one interaction at a time, the widest: the messages' gradient, or with a fused
    # convolution the tensor product weights' gradient.
    transient = max(((block.weight_numel if fused_conv else block.messages) for block in dims.interactions),
                    default=0) * edges
    activations = (sum(per_layer) + transient + shared * edges) * width
    rewrites = tuple((name, int(round(term.bytes(edges, nodes)))) for name, term in rewrite_terms)
    list_width = DTYPE_BYTES[list_dtype or dtype]
    pair_list = capacity * (_INDEX_BYTES_PER_SLOT + _SHIFT_FLOATS_PER_SLOT * list_width + _SHIFT64_BYTES_PER_SLOT)
    weights = dims.parameters * width
    terms = [("weights", weights),
             ("activations", int(math.ceil(SAFETY * activations))),
             ("rewrites", int(math.ceil(SAFETY * sum(value for _, value in rewrites)))),
             ("neighbour list", int(pair_list))]
    if d3 is not None:
        terms.append(("d3 list", int(d3.list_bytes + d3.build_bytes)))
    if workspace:
        terms.append(("workspace", WORKSPACE_BYTES))
    build = [("weights (pristine and execution copy)", 2 * weights)]
    largest = max(selfcheck_terms, key=lambda item: item[1], default=None)
    if largest is not None and largest[1] > 0:
        build.append((f"self-check {largest[0]}", int(largest[1])))
    return MemoryEstimate(terms=tuple(terms), edges=int(edges), nodes=int(nodes), capacity=int(capacity),
                          dtype=dtype, resident=weights if resident_weights else 0, rewrite_terms=rewrites,
                          build_terms=tuple(build))


def selfcheck_memory_terms(records: Any, dims, enabled: list[str], dtype: str, sms: int | None
                           ) -> tuple[tuple[str, int], ...]:
    """``(name, bytes)`` of every enabled record that declares its self-check's transient:
    ``Rewrite.selfcheck_memory(dims, width, sms)``."""
    width = DTYPE_BYTES[dtype]
    out = []
    for name in enabled:
        record = records[name] if name in records else None
        declare = getattr(record, "selfcheck_memory", None)
        if declare is not None:
            out.append((name, int(declare(dims, width, sms))))
    return tuple(out)


def fits(estimate_: MemoryEstimate, free_bytes: int | None) -> bool:
    """Decision 51: does the estimate stay within ``MARGIN`` of the free memory? ``True`` off CUDA."""
    return not free_bytes or estimate_.needed <= MARGIN * free_bytes


def drop_order(rewrite_terms: tuple[tuple[str, int], ...], edge_sized: set[str], explicit: set[str]) -> list[str]:
    """Decision 51's order: the rewrites with an edge-sized term, largest first, the caller's explicit ones
    left out. Ties keep application order."""
    candidates = [(name, value) for name, value in rewrite_terms
                  if name in edge_sized and name not in explicit and value > 0]
    return [name for name, _ in sorted(candidates, key=lambda item: -item[1])]


def rewrite_memory_terms(records: Any, dims, enabled: list[str], dtype: str) -> tuple[tuple[str, MemoryTerm], ...]:
    """``(name, MemoryTerm)`` of every enabled record that declares one: ``Rewrite.memory(dims, width)`` gives
    ``(bytes per edge, bytes per node)``."""
    width = DTYPE_BYTES[dtype]
    out = []
    for name in enabled:
        record = records[name] if name in records else None
        declare = getattr(record, "memory", None)
        if declare is not None:
            out.append((name, MemoryTerm(*declare(dims, width))))
    return tuple(out)


def annotate_out_of_memory(error, plan) -> Any:
    """Decision 51: an out-of-memory error carries rule 12's estimate. Appends the plan's ``memory`` decision
    to ``error``'s message (once) and returns ``error``; any other error is returned unchanged."""
    if not (isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError"):
        return error
    try:
        _, choice, reason = plan.decision("memory")
    except (KeyError, AttributeError, TypeError, ValueError):
        return error
    note = f"FastMACE plan (SPEC §9.3 rule 12): memory {choice}: {reason}"
    message = str(error.args[0]) if error.args else ""
    if note not in message:
        error.args = (f"{message}\n{note}" if message else note, *error.args[1:])
    return error


__all__ = ["DTYPE_BYTES", "MARGIN", "MemoryEstimate", "MemoryTerm", "SAFETY", "WORKSPACE_BYTES",
           "annotate_out_of_memory", "drop_order", "estimate", "fits", "interaction_floats", "list_capacity",
           "model_edges", "rewrite_memory_terms", "selfcheck_memory_terms"]
