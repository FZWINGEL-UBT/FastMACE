"""PolarMACE's head projections composed into their dot products at setup (P1; polar-megakernel, 7 November 2026).

Both POLAR heads feed narrow sources through an equivariant linear map to the hidden width before an l x l -> 0
``SparseUvuTensorProduct`` (``dot_products``), whose path p computes, for every invariant channel u,

    z_p[u] = pw_p / sqrt(d) * sum_m h[u, m] * sum_v W_p[u, v] x[v, m].

- The field update (``AgnosticEmbeddedOneBodyVariableUpdate``) mixes ``x = A h + P phi + Q q``
  (``AgnosticChargeBiasedLinearPotentialEmbedding``: node features, potentials, charges; no nonlinearity between).
- The electron readout (``OneBodyMLPFieldReadout``) uses ``x = P q + b`` and ``x = P' phi + b'``
  (``linear_up_q`` / ``linear_up_v`` with biases).

Per path, ``W_p (A h + P phi + Q q) = (W_p A) h + (W_p [P Q]) [phi q]``: the square maps ``A`` (update) and
``P`` (readout, into the 512-wide intermediate) disappear. Each update keeps one square GEMM per degree instead of
two, each readout branch keeps only narrow ones (input width 2-8 instead of 512); the same holds for the input
gradients. The composed matrices (with ``pw_p / sqrt(d)`` and the biases) are formed in float64 from the modules
as they run (another rewrite may have replaced a linear) and rounded once. Rounding (reassociated products), not an
approximation. The MLPs, gates and every other consumer of the layer features are unchanged.

Mechanics: the embedding / up-projection instances return the narrow sources (a tuple) instead of the 512-wide
sum, and their ``dot_products`` instances take that tuple; any tensor argument still runs ``uvu_forward``. Works
with FOLD_UPDATE (which replaces only the update's MLP tail). Both layouts: under cuEquivariance (ir_mul) MACE wraps
the o3 linears in layout transposes, which are permutations; the composed matrices stay the same, and only the gather
index into the sources and the view of the node features follow the dot products' layout.
"""

from __future__ import annotations

import math
import types

# COMPOSE_HEADS: True composes every supported head of a PolarMACE model when polar_uvu is applied (an A/B side's
# choice; per GPU through decision 66 once measured).
COMPOSE_HEADS = False
UPDATE_CLASS = "AgnosticEmbeddedOneBodyVariableUpdate"
READOUT_CLASS = "OneBodyMLPFieldReadout"
EMBEDDING_CLASS = "AgnosticChargeBiasedLinearPotentialEmbedding"
MARKER = "_fastmace_head_compose"
_TAG = "fastmace_head_compose"
_READOUT_TRANSPOSES = ("_q_to_mul_ir", "_v_to_mul_ir", "_up_from_mul_ir")


def _irrep_entries(irreps):
    """[(offset, mul, l, p)] of an e3nn Irreps in mul_ir order."""
    out, offset = [], 0
    for mul, ir in irreps:
        out.append((offset, mul, ir.l, ir.p))
        offset += mul * ir.dim
    return out


def _linear_blocks(linear, irreps_in, irreps_out, dtype, device):
    """{(out entry index, in entry index): M [mul_out, mul_in]} and the bias [irreps_out.dim] of an equivariant
    linear map, evaluated in float64 where the module allows it (on one-hot inputs, m = 0 components)."""
    import copy

    import torch

    try:
        module, run_dtype = copy.deepcopy(linear).double(), torch.float64
    except (RuntimeError, TypeError, NotImplementedError):
        module, run_dtype = linear, dtype
    with torch.no_grad():
        bias = module(torch.zeros(1, irreps_in.dim, dtype=run_dtype, device=device))[0].double().cpu()
        blocks = {}
        outs = _irrep_entries(irreps_out)
        for i, (off_i, mul_i, l_i, p_i) in enumerate(_irrep_entries(irreps_in)):
            d = 2 * l_i + 1
            probe = torch.zeros(mul_i, irreps_in.dim, dtype=run_dtype, device=device)
            probe[torch.arange(mul_i), off_i + torch.arange(mul_i) * d] = 1.0
            response = (module(probe).double().cpu() - bias)  # [mul_i, irreps_out.dim]
            for o, (off_o, mul_o, l_o, p_o) in enumerate(outs):
                if (l_o, p_o) == (l_i, p_i):
                    cols = off_o + torch.arange(mul_o) * d
                    blocks[(o, i)] = response[:, cols].t().contiguous()  # [mul_o, mul_i]
    return blocks, bias


def _paths(tp):
    """The l x l -> 0 paths of a SparseUvuTensorProduct: (in1 slice, in2 slice, out slice, W*pw/sqrt(d), mul, d)."""
    import torch

    out = []
    weights = tp.weight.detach().double().cpu()
    for (in1_start, in1_stop, in2_start, in2_stop, out_start, out_stop, w_start, w_stop, path_weight, mode_code,
         mul1, mul2, d1) in tp._path_meta:
        if mode_code != 0:
            raise ValueError("not an l x l -> 0 path")
        w = weights[w_start:w_stop].view(mul1, mul2) * (float(path_weight) / math.sqrt(float(d1)))
        out.append(((in1_start, in1_stop), (in2_start, in2_stop), (out_start, out_stop), w, mul1, d1))
    return out


def _entry_of(irreps, start):
    for index, (offset, mul, l, p) in enumerate(_irrep_entries(irreps)):
        if offset == start:
            return index, mul, l, p
    raise ValueError(f"no irreps entry starts at {start}")


def _compose(tp, node_irreps, square, sources, dtype, device, layout="mul_ir"):
    """Per path of ``tp``: the square block (W A, or None), the narrow source block [mul, K] (or None), the gather
    index [d, K] into the concatenated sources, and the bias [mul] (or None)."""
    import torch

    plans = []
    for (in1, in2, out_slice, w, mul1, d) in _paths(tp):
        entry, mul2, l, p = _entry_of(node_irreps, in2[0])
        square_block = None
        if square is not None:
            if in1 != in2:
                raise ValueError("a dot-product path pairs two different irreps entries")
            blocks, _ = square
            parts = [(i, m) for (o, i), m in blocks.items() if o == entry]
            if len(parts) != 1 or parts[0][0] != entry:
                raise ValueError("the node map mixes irreps entries")
            square_block = w @ parts[0][1]
        columns, matrices, bias = [], [], None
        source_offset = 0
        for irreps_in, (blocks, b) in sources:
            for i, (off_i, mul_i, l_i, p_i) in enumerate(_irrep_entries(irreps_in)):
                if (o_block := blocks.get((entry, i))) is not None:
                    matrices.append(w @ o_block)
                    if layout == "mul_ir":  # component (u, m) of the entry at off_i + u d + m
                        column = torch.arange(mul_i)[None, :] * d + torch.arange(d)[:, None]
                    else:  # ir_mul: at off_i + m mul + u
                        column = torch.arange(mul_i)[None, :] + torch.arange(d)[:, None] * mul_i
                    columns.append(source_offset + off_i + column)  # [d, mul_i]
            if l == 0:
                lo = _irrep_entries(node_irreps)[entry][0]
                entry_bias = b[lo:lo + mul2]
                if entry_bias.abs().max() > 0:
                    bias = (w @ entry_bias) if bias is None else bias + w @ entry_bias
            source_offset += irreps_in.dim
        narrow = torch.cat(matrices, dim=1) if matrices else None
        index = torch.cat(columns, dim=1) if columns else None
        plans.append({"in1": in1, "out": out_slice, "mul": mul1, "d": d,
                      "square": None if square_block is None else square_block.to(dtype=dtype, device=device),
                      "narrow": None if narrow is None else narrow.to(dtype=dtype, device=device),
                      "index": None if index is None else index.to(device=device),
                      "bias": None if bias is None else bias.to(dtype=dtype, device=device)})
    return plans


def composed_dot(self, x1, x2):
    """``dot_products`` with the composed head: ``x2`` is (tag, *sources) from the replaced embedding; a tensor
    ``x2`` runs ``uvu_forward``."""
    import torch

    from fastmace.rewrites.polar.uvu import uvu_forward

    if not isinstance(x2, tuple):
        return uvu_forward(self, x1, x2)
    sources = x2[1:]
    src = sources[0] if len(sources) == 1 else torch.cat(sources, dim=-1)
    batch = x1.shape[0]
    out = x1.new_zeros((batch, self.irreps_out.dim))
    ir_mul = self.layout != "mul_ir"
    for plan in self._head_plans:
        plan = {key: (getattr(self, value) if key in _TENSORS and value is not None else value)
                for key, value in plan.items()}
        mul, d = plan["mul"], plan["d"]
        block = x1[:, plan["in1"][0]:plan["in1"][1]]
        # x1t [batch, d, mul] (a view in ir_mul, a transpose in mul_ir)
        x1t = block.view(batch, d, mul) if ir_mul else block.view(batch, mul, d).transpose(1, 2)
        y = None
        if plan["narrow"] is not None:
            narrow_in = src[:, plan["index"]].reshape(batch * d, -1)  # [batch * d, K]
            y = narrow_in @ plan["narrow"].t()
        if plan["bias"] is not None:  # l = 0 only: d = 1
            y = plan["bias"].expand(batch, mul) if y is None else y + plan["bias"]
        if plan["square"] is not None:
            h_t = x1t.reshape(batch * d, mul)
            y = h_t @ plan["square"].t() if y is None else torch.addmm(y, h_t, plan["square"].t())
        if y is None:
            continue
        mixed = (x1t * y.reshape(batch, d, mul)).sum(1)
        o0, o1 = plan["out"]
        out[:, o0:o1] = out[:, o0:o1] + mixed
    return out * self.output_mask


def _embedding_forward(self, potential_feats, node_feats, node_attrs, local_charges):
    return (_TAG, potential_feats, local_charges)


def _up_forward(self, x):
    return (_TAG, x)


def _wrappers_match(module, names, layout):
    """MACE's layout transposes around the o3 linears: none in mul_ir, all of them in ir_mul."""
    present = [getattr(module, name, None) is not None for name in names]
    return not any(present) if layout == "mul_ir" else all(present)


def support(module):
    """(supported, reason) for one update or readout block."""
    kind = type(module).__name__
    if kind == UPDATE_CLASS:
        emb = module.potential_embedding
        if type(emb).__name__ != EMBEDDING_CLASS:
            return False, f"potential embedding {type(emb).__name__}"
        tps, holder = (module.dot_products,), emb
        names = ("_potential_to_mul_ir", "_node_feats_to_mul_ir", "_charges_to_mul_ir", "_node_feats_from_mul_ir")
    elif kind == READOUT_CLASS:
        tps, holder = (module.dot_products_q, module.dot_products_v), module
        names = _READOUT_TRANSPOSES
    else:
        return False, f"{kind} is not a POLAR head"
    layouts = {getattr(tp, "layout", "mul_ir") for tp in tps}
    if len(layouts) != 1 or not layouts <= {"mul_ir", "ir_mul"}:
        return False, f"dot-product layouts {sorted(layouts)}"
    if not _wrappers_match(holder, names, layouts.pop()):
        return False, "the layout transposes do not match the dot products' layout"
    if any(meta[9] != 0 for tp in tps for meta in tp._path_meta):
        return False, "a dot-product path is not l x l -> 0"
    return True, "ok"


_TENSORS = ("square", "narrow", "index", "bias")


def _install(tp, plans):
    """The plans' tensors as non-persistent buffers of ``tp`` (they follow ``.to()`` and deep copies); the plans
    keep the buffer names."""
    for number, plan in enumerate(plans):
        for key in _TENSORS:
            if plan[key] is not None:
                name = f"_head_{number}_{key}"
                tp.register_buffer(name, plan[key], persistent=False)
                plan[key] = name
    tp._head_plans = plans
    tp.forward = types.MethodType(composed_dot, tp)


def apply(model, context=None):
    """Compose every supported POLAR head of ``model`` in place; returns the composed module names."""
    names = []
    for name, module in model.named_modules():
        if type(module).__name__ not in (UPDATE_CLASS, READOUT_CLASS) or module.__dict__.get(MARKER):
            continue
        if not support(module)[0]:
            continue
        if type(module).__name__ == UPDATE_CLASS:
            emb, tp = module.potential_embedding, module.dot_products
            ref = tp.weight
            node = module.node_feats_irreps
            square = _linear_blocks(emb.node_feats_linear, node, node, ref.dtype, ref.device)
            sources = [(emb.potential_irreps, _linear_blocks(emb.potential_linear, emb.potential_irreps, node,
                                                             ref.dtype, ref.device)),
                       (emb.charges_irreps, _linear_blocks(emb.charge_embedding, emb.charges_irreps, node,
                                                           ref.dtype, ref.device))]
            _install(tp, _compose(tp, node, square, sources, ref.dtype, ref.device, tp.layout))
            emb.forward = types.MethodType(_embedding_forward, emb)
        else:
            node = module.node_feats_irreps
            for up, tp, irreps in ((module.linear_up_q, module.dot_products_q, module.charges_irreps),
                                   (module.linear_up_v, module.dot_products_v, module.potential_irreps)):
                ref = tp.weight
                sources = [(irreps, _linear_blocks(up, irreps, node, ref.dtype, ref.device))]
                _install(tp, _compose(tp, node, None, sources, ref.dtype, ref.device, tp.layout))
                up.forward = types.MethodType(_up_forward, up)
            # ir_mul: the sources are gathered in their own layout and the result is never transposed back, so the
            # readout's layout transposes (permutations) go; its forward skips a None transpose.
            for wrapper in _READOUT_TRANSPOSES:
                if getattr(module, wrapper, None) is not None:
                    setattr(module, wrapper, None)
        setattr(module, MARKER, True)
        names.append(name)
    return names


__all__ = ["COMPOSE_HEADS", "apply", "composed_dot", "support"]
