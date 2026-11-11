"""Interaction 0's density division and symmetric contraction as one generated kernel pair (``SC_GENERATED``;
Codex report v3 proposal 5, results/sc0-bound-20261109).

With ``kernels.triton.compose_skip`` the first (density) interaction ends in ``message / (density + 1)`` and
product block 0 then runs cuEquivariance's symmetric contraction; on the H200 these are four kernels (division,
contraction, contraction backward, division backward) of 179.5 us for OMAT-0 medium on Benchmark_loup, FP32-issue
bound. ``kernels.cuda.sc_generated`` renders the contraction's polynomial as straight-line code with the division
folded in: 82.4 us for the pair in the lab.

``route`` marks the interaction's reassociation (``sc_generated``: return the undivided message and pass the
density in the product's ``sc`` slot as ``(density,)``, a slot the density kind leaves None) and swaps product 0's
class for a subclass that takes that tuple to the generated kernels. Any other ``sc`` runs the original forward.
Rounding: another summation order (rms about 3e-7 relative), checked with the force/stress gate (decision 66).
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

SC_GENERATED = None
# NVIDIA H200 (9 November 2026, results/sc0-op-20261109; OMAT-0 medium, Benchmark_loup SrTiO3 10x10x10, a = the H200
# defaults of a6430eb6): MD step b/a 0.9760 / 0.9731 (ab / ba); force/stress gate passes (forces 0.850 / 0.920 /
# 0.959, stress 0.415 / 0.778 / 1.209).
# A100 family (9 November 2026, same run id; Modal NVIDIA A100 80GB PCIe, a = its A100 defaults): MD step b/a
# 0.9685 / 0.9652; gate passes (forces 0.972 / 0.854 / 0.977, stress 0.701 / 0.898 / 1.237).
# RTX 4000 Ada (ada-5060-screen, 10-11 November 2026, results/ada-5060-screen-20261110; OMAT-0 medium, Benchmark_loup,
# SrTiO3 10x10x10 / NbCl5, ab / ba): alone 0.9785 / 0.9775, 0.9738 / 0.9749; left out of the
# adopted set 1.0177 / 1.0171, 1.0437 / 1.0434.
SC_GENERATED_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
                               "NVIDIA RTX 4000 Ada Generation"))
# SC_FACTORED (results/other-ways-20261109 section 3; rounding: another basis and summation order): with SC_GENERATED
# the kernel pair of kernels.cuda.sc_factored, the contraction through memoised CG pair products in a cheaper basis of
# the descriptor's span, the change of basis folded into the weights once at routing. OMAT-0 medium: 413 operations
# plus 61 output FMAs per channel against about 1,040 forward and 1,500 backward. Kernel lab, A100 defaults: Alex
# A100-SXM4-80GB forward 0.593, forward + backward 0.650 of the monomial pair (226.0 -> 147.0 us); local RTX 5060 Ti
# 1.003 (memory-bound there). NVIDIA A100-SXM4-80GB (10 November 2026, Alex job 4973813, source 219602cf;
# Benchmark_loup MD step b/a, a = the A100 defaults): 0.9889 / 0.9881 (ab / ba); force/stress gate passes with GEOM
# also on (forces 0.863 / 0.913 / 0.985, stress 0.727 / 0.896 / 1.343). The H200 stopped at its bound (ledger
# sc0-invariant-basis: under 1 % there). None: on for SC_FACTORED_GPUS; True / False: an A/B side's choice.
# RTX 4000 Ada (ada-5060-screen, 11 November 2026, results/ada-5060-screen-20261110; on top of SC_GENERATED, MD step
# b/a SrTiO3 10x10x10 0.9978 / 0.9978, NbCl5 0.9907 / 0.9906, ab / ba): on with the Ada set, gated with it.
SC_FACTORED = None
SC_FACTORED_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA RTX 4000 Ada Generation"))

_subclasses = {}


def sc_generated_default(name=None, config="auto") -> bool:
    """``SC_GENERATED`` (an A/B side's choice) when set; else ``RewriteConfig.sc_generated`` ``config`` when "on" or
    "off"; else ``name`` in ``SC_GENERATED_GPUS``."""
    on = SC_GENERATED
    if on is None:
        on = config == "on" if config in ("on", "off") else name in SC_GENERATED_GPUS
    return bool(on)


def sc_factored_default(name=None) -> bool:
    """``SC_FACTORED`` (an A/B side's choice) when set; else ``name`` in ``SC_FACTORED_GPUS``."""
    return bool(SC_FACTORED) if SC_FACTORED is not None else name in SC_FACTORED_GPUS


def _weight(contraction):
    weight = getattr(contraction, "projected_weight", None)
    if weight is None:
        weight = torch.einsum("zau,ab->zbu", contraction.weight, contraction.projection)
    return weight.flatten(1)


def _forward(self, node_feats, sc, node_attrs):
    base = type(self).sc_generated_base
    if not isinstance(sc, tuple):
        return base.forward(self, node_feats, sc, node_attrs)
    (density,) = sc
    contraction = self.symmetric_contractions
    if torch.is_grad_enabled() and contraction.weight.requires_grad:  # a trainable weight: the original steps
        return base.forward(self, node_feats / (density.reshape(-1, *([1] * (node_feats.dim() - 1))) + 1), None,
                            node_attrs)
    from fastmace.kernels.cuda.sc_generated import sc_generated_forward

    if self.cueq_config.layout_str == "mul_ir":
        node_feats = torch.transpose(node_feats, 1, 2)
    x = contraction.transpose_in(node_feats.flatten(1))
    weight = getattr(self, "sc_generated_weight", None)  # SC_FACTORED: folded once at routing
    if weight is None:
        weight = _weight(contraction)
    y = sc_generated_forward(x, density, weight, node_attrs.argmax(dim=-1).int(), self.sc_generated_id)
    return self.linear(contraction.transpose_out(y))


def _subclass(cls):
    if cls not in _subclasses:
        _subclasses[cls] = type(cls.__name__, (cls,), {"forward": _forward, "sc_generated_base": cls,
                                                        "__module__": __name__,
                                                        "__qualname__": f"sc_generated.{cls.__qualname__}"})
    return _subclasses[cls]


def applies(model) -> bool:
    """Interaction 0 ends in compose_skip's division and product 0 runs a generatable cuEq contraction."""
    from fastmace.kernels.cuda.sc_generated import shape_of

    interactions, products = getattr(model, "interactions", None), getattr(model, "products", None)
    if not interactions or not products:
        return False
    reassociation = getattr(interactions[0], "first_reassociation", None)
    product = products[0]
    if (reassociation is None or getattr(reassociation, "compose_skip", None) is None
            or type(product).__name__ != "EquivariantProductBasisBlock" or "forward" in product.__dict__
            or getattr(product, "region_widths", None) is not None or getattr(product, "use_agnostic_product", False)):
        return False
    config = getattr(product, "cueq_config", None)
    if config is None or not config.enabled or not (config.optimize_all or config.optimize_symmetric):
        return False
    contraction = getattr(product, "symmetric_contractions", None)
    if type(contraction).__name__ != "SymmetricContraction" or getattr(contraction, "projection", None) is None:
        return False
    return contraction.weight.device.type == "cuda" and shape_of(contraction) is not None


def generate(model):
    """Compile product 0's kernels, mark interaction 0 and swap product 0's class."""
    from fastmace.kernels.cuda.sc_generated import GeneratedContraction

    product = model.products[0]
    model.interactions[0].first_reassociation.sc_generated = True
    contraction = product.symmetric_contractions
    device = contraction.weight.device
    factored = sc_factored_default(torch.cuda.get_device_name(device))
    if hasattr(type(product), "sc_generated_base") and getattr(product, "sc_generated_factored", None) == factored:
        return model  # routed before with the same SC_FACTORED: the kernels are compiled
    with torch.cuda.device(device):
        kernels = GeneratedContraction(contraction, factored=factored)
        product.sc_generated_weight = (kernels.fold(_weight(contraction).detach()).contiguous() if kernels.factored
                                       else None)
    product.sc_generated_id = kernels.id
    product.sc_generated_factored = factored
    if not hasattr(type(product), "sc_generated_base"):
        product.__class__ = _subclass(type(product))
    return model


def route(model, config="auto") -> int:
    """``generate`` where ``applies`` and ``sc_generated_default`` say so (``config``: ``RewriteConfig.sc_generated``);
    the number of products changed (0 or 1)."""
    reassociation = getattr((getattr(model, "interactions", None) or [None])[0], "first_reassociation", None)
    if reassociation is not None:
        reassociation.sc_generated = False
    if not applies(model):
        return 0
    device = model.products[0].symmetric_contractions.weight.device
    if not sc_generated_default(torch.cuda.get_device_name(device), config):
        product = model.products[0]
        if hasattr(type(product), "sc_generated_base"):
            product.__class__ = type(product).sc_generated_base
        return 0
    generate(model)
    return 1


__all__ = ["SC_FACTORED", "SC_FACTORED_GPUS", "SC_GENERATED", "SC_GENERATED_GPUS", "applies", "generate", "route",
           "sc_factored_default", "sc_generated_default"]
