"""Dead degrees across POLAR head boundaries, proved from e3nn instructions at setup (R5; polar-megakernel, 7 November
2026; an A/B constant, off). Pure work removal.

Two patterns, from the Codex profile audit ``plans/codex-off-polar-profiles-20261107-more.md`` (h200-ideas worktree):

- A. The field update's ``tp_out`` builds every node degree, but its result feeds only ``readout`` (a
  ``GeneralNonLinearBiasReadoutBlock``), whose ``linear_1`` reads only the input entries named in its instructions
  (POLAR-1-L: degrees 0 and 1 of ``512x0e+512x1o+512x2e``). A ``tp_out`` path writing an unread entry is dead: with
  FOLD_UPDATE its gate rows leave ``_fold_out`` (POLAR-1-L: 1536 x 64 -> 1024 x 64) and its output stays zero. The
  degree-two inputs of the invariants (``dot_products``) are live and stay.
- B. The electron readout (``OneBodyMLPFieldReadout``) expands charges and potentials (degrees 0 and 1) with
  ``linear_up_q`` / ``linear_up_v`` to the node irreps; an output entry no instruction writes (and with no bias:
  only scalars carry one) is exactly zero, so the ``dot_products_q`` / ``_v`` paths reading it add exact zeros.
  Removing them is bitwise exact. With COMPOSE_HEADS (P1) the composed dot already skips them.

The tensor products themselves are left whole (the registry's per-module self-check compares them with MACE on
random inputs); the blocks run only the live paths: the folded update through ``_live_out_meta`` (read by
``update_fold``), the readout through an instance forward. Liveness comes from the instructions, never from
values or model names.
"""

from __future__ import annotations

import hashlib
import inspect
import types

# PRUNE_DEAD_DEGREES: True prunes every supported POLAR head when polar_uvu is applied (an A/B side's choice; pure
# work removal, decision 66).
PRUNE_DEAD_DEGREES = False
UPDATE_CLASS = "AgnosticEmbeddedOneBodyVariableUpdate"
READOUT_CLASS = "OneBodyMLPFieldReadout"
UPDATE_READOUT_CLASS = "GeneralNonLinearBiasReadoutBlock"
MARKER = "_fastmace_head_liveness"
SOURCE_HASHES = {  # MACE 0.3.16
    "OneBodyMLPFieldReadout.forward": "a953dd17b3128b9828aaece883214e513872ea5977e7e3067eb4a0f729db2659",
    "GeneralNonLinearBiasReadoutBlock.forward": "f92245b2b66bbdd89c7cd6ab83260b2bbe3c539813a3944eda576a551f91fbb5",
}


def _digest(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _entry(irreps, start):
    starts = [s.start for s in irreps.slices()]
    return starts.index(start)


def dead_output_paths(module):
    """Indices of the update's ``tp_out`` paths whose output entry ``readout.linear_1`` never reads."""
    readout = module.readout
    if type(readout).__name__ != UPDATE_READOUT_CLASS:
        return []
    if _digest(type(readout).forward) != SOURCE_HASHES["GeneralNonLinearBiasReadoutBlock.forward"]:
        return []
    first = readout.linear_1
    instructions = getattr(first, "instructions", None)
    tp = module.tp_out
    if instructions is None or getattr(first, "irreps_in", None) != tp.irreps_out:
        return []  # another rewrite replaced linear_1: nothing proved
    live = {int(ins.i_in) for ins in instructions}
    return [k for k, meta in enumerate(tp._path_meta) if _entry(tp.irreps_out, meta[4]) not in live]


def dead_input_paths(up, tp):
    """Indices of ``tp``'s paths whose second input is an entry ``up`` (an o3.Linear) never writes."""
    instructions = getattr(up, "instructions", None)
    if instructions is None or getattr(up, "irreps_out", None) != tp.irreps_in2:
        return []
    live = {int(ins.i_out) for ins in instructions}
    live |= {k for k, (_, ir) in enumerate(up.irreps_out) if ir.l == 0}  # a bias (scalars only) also writes
    return [k for k, meta in enumerate(tp._path_meta) if _entry(tp.irreps_in2, meta[2]) not in live]


def _dot(tp, x1, x2, meta):
    from fastmace.rewrites.polar.uvu import uvu_forward

    if isinstance(x2, tuple):  # composed by head_compose (which skips the zero sources itself)
        return tp(x1, x2)
    return uvu_forward(tp, x1, x2, meta)


def pruned_readout_forward(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index, field_feats, charges_0,
                           charges_induced):
    """``OneBodyMLPFieldReadout.forward`` (MACE 0.3.16) with only the live dot-product paths."""
    import torch

    q_in = charges_induced + charges_0
    q_to_mul_ir = getattr(self, "_q_to_mul_ir", None)
    v_to_mul_ir = getattr(self, "_v_to_mul_ir", None)
    up_from_mul_ir = getattr(self, "_up_from_mul_ir", None)
    if q_to_mul_ir is not None:
        q_in = q_to_mul_ir(q_in)
    q_up = self.linear_up_q(q_in)
    if up_from_mul_ir is not None:
        q_up = up_from_mul_ir(q_up)
    v_in = field_feats
    if v_to_mul_ir is not None:
        v_in = v_to_mul_ir(v_in)
    v_up = self.linear_up_v(v_in)
    if up_from_mul_ir is not None:
        v_up = up_from_mul_ir(v_up)
    invar_feats = torch.cat([_dot(self.dot_products_q, node_feats, q_up, self._live_q_meta),
                             _dot(self.dot_products_v, node_feats, v_up, self._live_v_meta)], dim=-1)
    return self.mlp(invar_feats).squeeze(-1)


def apply(model, context=None, fold=None):
    """Prune every supported POLAR head of ``model`` in place (before FOLD_UPDATE and COMPOSE_HEADS); returns the
    pruned module names. The update's dead outputs go only where the fold will run (``fold``, default FOLD_UPDATE):
    MACE's own update runs ``tp_out`` whole."""
    from fastmace.rewrites.polar import update_fold

    fold = update_fold.FOLD_UPDATE if fold is None else fold

    names = []
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind not in (UPDATE_CLASS, READOUT_CLASS) or module.__dict__.get(MARKER):
            continue
        if kind == UPDATE_CLASS:
            if not fold or module.__dict__.get(update_fold.MARKER) or not update_fold.support(module)[0]:
                continue  # unfolded, or already folded (its gate columns follow every path)
            dead = set(dead_output_paths(module))
            if not dead:
                continue
            module._live_out_meta = [m for k, m in enumerate(module.tp_out._path_meta) if k not in dead]
        else:
            if _digest(type(module).forward) != SOURCE_HASHES["OneBodyMLPFieldReadout.forward"]:
                continue
            dead_q = set(dead_input_paths(module.linear_up_q, module.dot_products_q))
            dead_v = set(dead_input_paths(module.linear_up_v, module.dot_products_v))
            if not dead_q and not dead_v:
                continue
            module._live_q_meta = [m for k, m in enumerate(module.dot_products_q._path_meta) if k not in dead_q]
            module._live_v_meta = [m for k, m in enumerate(module.dot_products_v._path_meta) if k not in dead_v]
            module.forward = types.MethodType(pruned_readout_forward, module)
        setattr(module, MARKER, True)
        names.append(name)
    return names


__all__ = ["PRUNE_DEAD_DEGREES", "SOURCE_HASHES", "apply", "dead_input_paths", "dead_output_paths"]
