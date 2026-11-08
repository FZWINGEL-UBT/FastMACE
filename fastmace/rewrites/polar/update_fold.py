"""PolarMACE's field-update MLP folded at setup (off-polar-alex lab, 7 November 2026; default on since 8 November).

``AgnosticEmbeddedOneBodyVariableUpdate`` (MACE 0.3.16) forms ``cat(invariants[C], source_embedding(Z)[C])``, runs
the ``RadialMLP`` 2C -> 64 -> 64 -> 64 -> C, and mixes those C scalars into ``tp_out``'s per-degree C x C weights
before scaling the node features. Two linear maps are folded into constants, computed in float64 and rounded once:

- N1, the last MLP linear through ``tp_out``: ``g_p = W_p (V h + b) = (W_p V) h + W_p b`` for every ``tp_out``
  path p, so the C-wide MLP output and the per-path C x C products become one [N, 64] @ [64, P C] product.
- N4, the species half of the first MLP input: ``W_b source_embedding(node_attrs)`` is linear in ``node_attrs``,
  so it becomes ``node_attrs @ S`` with S = source_embedding(I) W_b^T ([species, 64]); the first linear reads only
  the C invariants.

Rounding (the products are reassociated), not an approximation. The instance's forward is replaced; the class and
other models keep MACE's code. Ideas N1 and N4 of plans/codex-off-polar-current-20261107.md (h200-ideas worktree).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import types

# FOLD_UPDATE: True folds every update block of a PolarMACE model when polar_uvu is applied. Default on
# (off-polar-alex, 8 November 2026) under the pure-removal rule: MD step b/a, abab / abba, A100-SXM4-80GB S NaCl
# 0.965 / 0.971, M 0.969 / 0.970, L 0.972 / 0.971, NbCl5 1.000 / 1.000 on all three; RTX 4000 Ada S NaCl 0.966,
# M 0.971, L 0.976, NbCl5 1.000; RTX 5060 Ti NaCl 0.95-0.98, NbCl5 0.990-1.002; tools/force_stress_gate.py
# passes on S, M and L (A100). False: MACE's forward.
FOLD_UPDATE = True
CLASS_NAME = "AgnosticEmbeddedOneBodyVariableUpdate"
MARKER = "_fastmace_update_fold"
SOURCE_HASHES = {
    "forward": "f3383e061234f837210c2c837726f748d1418de34da4b421a0b6c0c275c0513b",
    "RadialMLP.forward": "a6f06e6c4a4fece818030d4b977f41d369273f2c33da75ccf1f06036ca5ff9eb",
}


def _digest(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _targets(model):
    return [(name, module) for name, module in model.named_modules() if type(module).__name__ == CLASS_NAME]


def support(module):
    """(supported, reason) for one update block."""
    import torch

    if _digest(type(module).forward) != SOURCE_HASHES["forward"]:
        return False, f"{CLASS_NAME}.forward differs from the pinned MACE 0.3.16 source"
    mlp = module.nonlinearity
    if _digest(type(mlp).forward) != SOURCE_HASHES["RadialMLP.forward"]:
        return False, "RadialMLP.forward differs from the pinned source"
    net = list(mlp.net)
    if not net or not all(isinstance(layer, torch.nn.Linear) for layer in net[::3]) or len(net) % 3 != 1:
        return False, "the MLP is not Linear (LayerNorm SiLU Linear)*"
    first, last = net[0], net[-1]
    tp = module.tp_out
    channels = last.out_features
    if first.in_features != 2 * channels or last.bias is None or first.bias is None:
        return False, "the MLP's input is not cat(invariants, species embedding) of its output width"
    for meta in tp._path_meta:
        in2_start, in2_stop, mode, mul2, d1 = meta[2], meta[3], meta[9], meta[11], meta[12]
        if (in2_start, in2_stop, mul2) != (0, channels, channels):
            return False, "a tp_out path does not read the whole MLP output"
        if mode == 0 and d1 != 1:
            return False, "a tp_out l x l -> 0 path with l > 0"
    if tp.irreps_in2.dim != channels:
        return False, "tp_out's second input is not the MLP output"
    return True, "ok"


def _constants(module):
    """(W_a [64, C], S [species, 64], b1 [64], T [sum mul1, 64], t [sum mul1]) in float64 on the CPU."""
    import copy

    import torch

    net = list(module.nonlinearity.net)
    first, last = net[0], net[-1]
    channels = last.out_features
    w1 = first.weight.detach().double().cpu()
    # source_embedding may already be another rewrite's module (BlockLinear): evaluate it as it runs, on the one-hot
    # identity, in float64 where it allows it.
    species = module.node_attrs_irreps.dim
    device = first.weight.device
    try:
        embedding = copy.deepcopy(module.source_embedding).double()
        with torch.no_grad():
            table = embedding(torch.eye(species, dtype=torch.float64, device=device))
    except (RuntimeError, TypeError, NotImplementedError):
        with torch.no_grad():
            table = module.source_embedding(torch.eye(species, dtype=first.weight.dtype, device=device))
    table = table.double().cpu()  # [species, C]
    s = table @ w1[:, channels:].t()  # [species, 64]
    v = last.weight.detach().double().cpu()  # [C, 64]
    b = last.bias.detach().double().cpu()
    tp = module.tp_out
    weights = tp.weight.detach().double().cpu()
    blocks, biases = [], []
    for meta in tp._path_meta:
        w_start, w_stop, path_weight, mul1, mul2, d1 = meta[6], meta[7], meta[8], meta[10], meta[11], meta[12]
        w = weights[w_start:w_stop].view(mul1, mul2) * (float(path_weight) / math.sqrt(float(d1)))
        blocks.append(w @ v)
        biases.append(w @ b)
    return w1[:, :channels].clone(), s, first.bias.detach().double().cpu(), torch.cat(blocks), torch.cat(biases)


def folded_forward(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index, potential_features,
                   local_charges):
    """``AgnosticEmbeddedOneBodyVariableUpdate.forward`` with the MLP's first and last linears folded."""
    import torch

    mixed_feats = self.potential_embedding(potential_features, node_feats, node_attrs, local_charges)
    invariants = self.dot_products(node_feats, mixed_feats)
    hidden = torch.nn.functional.linear(invariants, self._fold_first, self._fold_first_bias)
    hidden = hidden + node_attrs @ self._fold_species
    for layer in list(self.nonlinearity.net)[1:-1]:
        hidden = layer(hidden)
    gates = torch.nn.functional.linear(hidden, self._fold_out, self._fold_out_bias)  # [N, sum mul1]
    tp = self.tp_out
    batch = node_feats.shape[0]
    out = node_feats.new_zeros((batch, tp.irreps_out.dim))
    column = 0
    for meta in tp._path_meta:
        in1_start, in1_stop, out_start, out_stop, mul1, d1 = meta[0], meta[1], meta[4], meta[5], meta[10], meta[12]
        gate = gates[:, column:column + mul1]
        column += mul1
        block = node_feats[:, in1_start:in1_stop]
        if tp.layout == "mul_ir":
            contrib = (block.view(batch, mul1, d1) * gate.unsqueeze(-1)).reshape(batch, -1)
        else:
            contrib = (block.view(batch, d1, mul1) * gate.unsqueeze(1)).reshape(batch, -1)
        out[:, out_start:out_stop] = out[:, out_start:out_stop] + contrib
    new_feats = out * tp.output_mask
    readout_to_mul_ir = getattr(self, "_readout_to_mul_ir", None)
    readout_from_mul_ir = getattr(self, "_readout_from_mul_ir", None)
    if readout_to_mul_ir is not None:
        new_feats = readout_to_mul_ir(new_feats)
    multipoles = self.readout(new_feats)
    if readout_from_mul_ir is not None:
        multipoles = readout_from_mul_ir(multipoles)
    return multipoles


def apply(model, context=None):
    """Fold every supported update block of ``model`` in place; returns the folded module names."""
    names = []
    for name, module in _targets(model):
        if module.__dict__.get(MARKER) or not support(module)[0]:
            continue
        reference = list(module.nonlinearity.net)[-1].weight
        constants = _constants(module)
        for key, value in zip(("_fold_first", "_fold_species", "_fold_first_bias", "_fold_out", "_fold_out_bias"),
                              constants):
            module.register_buffer(key, value.to(dtype=reference.dtype, device=reference.device), persistent=False)
        module.forward = types.MethodType(folded_forward, module)
        setattr(module, MARKER, True)
        names.append(name)
    return names


_BUFFERS = ("_fold_first", "_fold_species", "_fold_first_bias", "_fold_out", "_fold_out_bias")


def revert(model):
    """Undo ``apply`` (MACE's forward and no fold buffers again); returns the module names."""
    names = []
    for name, module in _targets(model):
        if not module.__dict__.pop(MARKER, False):
            continue
        module.__dict__.pop("forward", None)
        for key in _BUFFERS:
            module._buffers.pop(key, None)
        names.append(name)
    return names


def selfcheck(original, rewritten, threshold, atoms=7, seed=0):
    """Each folded block of ``rewritten`` against the matching block of ``original`` (MACE's forward), in float64
    on random one-hot species, node, potential and charge features. Returns (worst relative error, blocks checked,
    details over ``threshold``); a folded block without a match is an infinite error."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    originals = dict(_targets(original))
    worst, checked, detail = 0.0, 0, []
    for name, module in _targets(rewritten):
        if not module.__dict__.get(MARKER):
            continue
        reference = originals.get(name)
        if reference is None or reference.__dict__.get(MARKER):
            return math.inf, checked, [f"{name}: no unfolded block in the original"]
        device = next(iter(module.buffers())).device
        test = copy.deepcopy(module).double()
        ref = copy.deepcopy(reference).double()
        species = module.node_attrs_irreps.dim
        node_attrs = torch.nn.functional.one_hot(
            torch.randint(species, (atoms,), generator=generator), species).double()
        tensors = [node_attrs] + [torch.randn(atoms, irreps.dim, generator=generator, dtype=torch.float64)
                                  for irreps in (module.node_feats_irreps, module.potential_irreps,
                                                 module.charges_irreps)]
        node_attrs, node_feats, potential, charges = (t.to(device) for t in tensors)
        with torch.no_grad():
            expected = type(ref).forward(ref, node_attrs, node_feats, None, None, None, potential, charges)
            got = test(node_attrs, node_feats, None, None, None, potential, charges)
        scale = float(expected.abs().max())
        error = float((got - expected).abs().max()) / (scale if scale > 0 else 1.0)
        worst = max(worst, error)
        checked += 1
        if error > threshold:
            detail.append(f"{name} (update MLP folded): relative error {error:.3g} > {threshold:g}")
    return worst, checked, detail


__all__ = ["FOLD_UPDATE", "SOURCE_HASHES", "apply", "folded_forward", "revert", "selfcheck", "support"]
