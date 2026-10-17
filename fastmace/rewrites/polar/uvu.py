"""``polar_uvu``: PolarMACE's SparseUvuTensorProduct with the l x l -> 0 path reassociated (MACE PR 1711).

SPEC §6.2. The l x l -> 0 paths contract the weight first, sum_d x1[u, d] (sum_v w[u, v] x2[v, d]),
as one [atoms * d, v] x [v, u] GEMM, instead of forming the [atoms, u, v] pair tensor. Exact up to
float rounding (summation order; "rounding" in SPEC §6.1). Measured 4-5x lower PolarMACE step time
together with the MH-1 rewrites (research engine, 16 October 2026).

The rewrite replaces the ``forward`` of each SparseUvuTensorProduct *instance* of the model it is
applied to; the class, and every other model in the process (reference calculators), keep
MACE's code. A deep copy of a rewritten model stays rewritten.
"""

from __future__ import annotations

import copy
import math
import types

NAME = "polar_uvu"
CLASS_NAME = "SparseUvuTensorProduct"
# The instance marker set by apply().
MARKER = "_fastmace_polar_uvu"
MACE_VERSIONS = ">=0.3.16,<0.3.17"
# sha256 of inspect.getsource in MACE 0.3.16: forward is what the rewrite replaces, __init__ builds
# the _path_meta, layout and output_mask it reads.
SOURCE_HASHES = {
    "mace.modules.field_blocks:SparseUvuTensorProduct.forward":
        "18fb5180cdc9c13e0772615c6fdc394cd6cceae5fa08a3c98524299928d6c15b",
    "mace.modules.field_blocks:SparseUvuTensorProduct.__init__":
        "c0804c5e7d0d2952261bc3478577587f081650ebbc4288472d14e4311a93da54",
}
# SPEC §6.4 per-module thresholds (relative error: max |rewritten - original| / max |original|).
THRESHOLDS = {"float64": 1e-10, "float32": 1e-4}


def uvu_forward(self, x1, x2):
    """mace.modules.field_blocks.SparseUvuTensorProduct.forward with the reassociated l x l -> 0 path."""
    import torch

    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("SparseUvuTensorProduct expects flattened 2D tensors with shape [batch, irreps.dim]")
    batch = x1.shape[0]
    out = x1.new_zeros((batch, self.irreps_out.dim))

    def to_mul_ir(block, mul, dim):
        if self.layout == "mul_ir":
            return block.view(batch, mul, dim)
        return block.view(batch, dim, mul).transpose(1, 2)

    def from_mul_ir(t):
        if self.layout == "mul_ir":
            return t.reshape(batch, -1)
        return t.transpose(1, 2).reshape(batch, -1)

    for (in1_start, in1_stop, in2_start, in2_stop, out_start, out_stop, w_start, w_stop, path_weight, mode_code,
         mul1, mul2, d1) in self._path_meta:
        in1_block = x1[:, in1_start:in1_stop]
        in2_block = x2[:, in2_start:in2_stop]
        out_block = out[:, out_start:out_stop]
        w = self.weight[w_start:w_stop].view(mul1, mul2)
        x1v = to_mul_ir(in1_block, mul1, d1)
        if mode_code == 0:
            # original: einsum("buv,uv->bu", einsum("bud,bvd->buv", x1v, x2v) / sqrt(d1), w)
            x2v = to_mul_ir(in2_block, mul2, d1)
            # one [batch * d, v] x [v, u] GEMM (a broadcast matmul(w, x2v) runs batch GEMVs)
            weighted = torch.matmul(x2v.transpose(1, 2).reshape(batch * d1, mul2), w.t())
            mixed = (x1v * weighted.view(batch, d1, mul1).transpose(1, 2)).sum(-1) / math.sqrt(float(d1))
            out[:, out_start:out_stop] = out_block + path_weight * mixed
        else:
            mixed = torch.einsum("bv,uv->bu", in2_block.view(batch, mul2), w)
            contrib = path_weight * (x1v * mixed.unsqueeze(-1) / math.sqrt(float(d1)))
            out[:, out_start:out_stop] = from_mul_ir(to_mul_ir(out_block, mul1, d1) + contrib)
    return out * self.output_mask


def _targets(model):
    return [(name, module) for name, module in model.named_modules() if type(module).__name__ == CLASS_NAME]


def applies_to(model):
    """(supported, reason): a PolarMACE model with SparseUvuTensorProduct modules."""
    if type(model).__name__ != "PolarMACE":
        return False, f"{type(model).__name__} is not PolarMACE"
    if not _targets(model):
        return False, f"the model has no {CLASS_NAME}"
    return True, "ok"


def apply(model, context=None):
    """Rewrite every SparseUvuTensorProduct of ``model`` in place; returns the rewritten module names."""
    names = []
    for name, module in _targets(model):
        module.forward = types.MethodType(uvu_forward, module)
        setattr(module, MARKER, True)
        names.append(name)
    return names


def revert(model):
    """Undo ``apply`` (the instances use the class forward again); returns the module names."""
    names = []
    for name, module in _targets(model):
        if module.__dict__.pop("forward", None) is not None:
            names.append(name)
        module.__dict__.pop(MARKER, None)
    return names


def is_applied(model):
    """True when every SparseUvuTensorProduct of ``model`` carries the rewrite."""
    targets = _targets(model)
    return bool(targets) and all(module.__dict__.get(MARKER, False) for _, module in targets)


def selfcheck(original, rewritten=None, context=None, rows=7, seed=0):
    """SPEC §6.4 per-module check: the rewritten forward against MACE's on random inputs, in float64.

    ``original`` and ``rewritten`` are models (or modules) with matching SparseUvuTensorProduct
    modules; ``rewritten`` defaults to a rewritten deep copy of ``original``. Each module is
    copied and evaluated in float64 on its own device. Returns {"passed", "max_relative_error",
    "threshold", "modules", "detail"}.
    """
    import torch

    if rewritten is None:
        rewritten = copy.deepcopy(original)
        apply(rewritten)
    generator = torch.Generator().manual_seed(seed)
    threshold = THRESHOLDS["float64"]
    worst, checked, detail = 0.0, 0, []
    originals = dict(_targets(original))
    for name, module in _targets(rewritten):
        reference = originals.get(name)
        if reference is None:
            return {"passed": False, "max_relative_error": math.inf, "threshold": threshold, "modules": checked,
                    "detail": f"{name}: no matching module in the original"}
        device = module.weight.device
        test = copy.deepcopy(module).double()
        ref = copy.deepcopy(reference).double()
        x1 = torch.randn(rows, module.irreps_in1.dim, generator=generator, dtype=torch.float64).to(device)
        x2 = torch.randn(rows, module.irreps_in2.dim, generator=generator, dtype=torch.float64).to(device)
        with torch.no_grad():
            expected = type(ref).forward(ref, x1, x2)
            got = test(x1, x2)
        scale = float(expected.abs().max())
        error = float((got - expected).abs().max()) / (scale if scale > 0 else 1.0)
        worst = max(worst, error)
        checked += 1
        if error > threshold:
            detail.append(f"{name}: relative error {error:.3g} > {threshold:g}")
    if not checked:
        return {"passed": False, "max_relative_error": math.inf, "threshold": threshold, "modules": 0,
                "detail": f"no {CLASS_NAME} to check"}
    return {"passed": worst <= threshold, "max_relative_error": worst, "threshold": threshold, "modules": checked,
            "detail": "; ".join(detail) or "ok"}


# The registry record (SPEC §6.1 fields) as a plain dict, for 1a-T06's Rewrite.
RECORD = {
    "name": NAME,
    "applies_to": applies_to,
    "requires": (),
    "conflicts": (),
    "mace_versions": MACE_VERSIONS,
    "source_hashes": SOURCE_HASHES,
    "exactness": "rounding",
    "default": "on",  # RewriteConfig.polar_uvu = True
    "workload_constraints": (),
    "apply": apply,
    "selfcheck": selfcheck,
    "config_field": "RewriteConfig.polar_uvu",
}
