"""cuEquivariance's symmetric contraction with its projected weight kept (``CUEQ_PROJECTED``, h200-fastpath).

``cuequivariance_torch.SymmetricContraction.forward`` projects its weight on every call,
``einsum("zau,ab->zbu", weight, projection)``, before the contraction; both operands are constant for a frozen
model, so on the H200 that is six small launches (about 19 us per OMAT-0 medium force call, ``profile-spline``
Seq 16-21) for the same result each time. ``keep_projected`` computes it once into a buffer and swaps the module's
class for a subclass whose forward reads the buffer and otherwise runs the original's steps. The einsum is the same
call, so the values match an eager forward; a compiled forward may have lowered it differently, so the switch is
checked like a rounding change until measured otherwise.
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

CUEQ_PROJECTED = None
# NVIDIA H200 (8 November 2026, results/h200-fastpath-20261108): MD step b/a 0.9902 / 0.9954 (ab / ba); force/stress gate passes
# (force ratios 0.922 / 0.900 / 0.979, stress 0.425 / 0.852 / 1.206; decision 66).
CUEQ_PROJECTED_GPUS = GPUModels(("NVIDIA H200",))

_subclasses = {}


def cueq_projected_default(name=None) -> bool:
    """``CUEQ_PROJECTED`` (None: ``name`` in ``CUEQ_PROJECTED_GPUS``)."""
    on = CUEQ_PROJECTED
    if on is None:
        on = name in CUEQ_PROJECTED_GPUS
    return bool(on)


def _forward(self, x, indices):
    weight = self.projected_weight
    if torch.is_grad_enabled() and self.weight.requires_grad:  # a trainable weight: the original's projection
        weight = torch.einsum("zau,ab->zbu", self.weight, self.projection)
    output = self.f([weight.flatten(1), self.transpose_in(x)], input_indices={0: indices})
    return self.transpose_out(output[0])


def _subclass(cls):
    if cls not in _subclasses:
        # same name: loading and the planner recognise the module by its type name
        _subclasses[cls] = type(cls.__name__, (cls,), {"forward": _forward, "__module__": __name__,
                                                        "__qualname__": f"projected.{cls.__qualname__}"})
    return _subclasses[cls]


def applies(module) -> bool:
    """A cuEquivariance ``SymmetricContraction`` with a projection (by name: the package is optional)."""
    return (type(module).__name__ == "SymmetricContraction" and getattr(module, "projection", None) is not None
            and "projected_weight" not in module._buffers
            and isinstance(getattr(module, "weight", None), torch.Tensor) and hasattr(module, "f"))


def keep_projected(module):
    """Project ``module``'s weight once into the buffer ``projected_weight`` and use it from then on."""
    with torch.no_grad():
        projected = torch.einsum("zau,ab->zbu", module.weight, module.projection)
    module.register_buffer("projected_weight", projected, persistent=False)
    module.__class__ = _subclass(type(module))
    return module


def route(model) -> int:
    """``keep_projected`` on every applicable module on a CUDA device where ``cueq_projected_default`` says so;
    the number of modules changed."""
    changed = 0
    for module in list(model.modules()):
        if not applies(module) or module.weight.device.type != "cuda":
            continue
        if cueq_projected_default(torch.cuda.get_device_name(module.weight.device)):
            keep_projected(module)
            changed += 1
    return changed


__all__ = ["CUEQ_PROJECTED", "CUEQ_PROJECTED_GPUS", "applies", "cueq_projected_default", "keep_projected", "route"]
