"""The recomputed forward of ``recompute_node_segments`` (plans/capacity.md CAP-T08; ``fastmace.rewrites.segments``)."""

from __future__ import annotations

import types


def recomputed_forward(self, *args, **kwargs):
    """``self.recomputed_inner`` (the module's forward before the record) under non-reentrant
    ``torch.utils.checkpoint``: the backward keeps only the segment's inputs and rebuilds its intermediates when it
    reaches the segment. AOT autograd traces the recomputation into the compiled graph."""
    from torch.utils.checkpoint import checkpoint

    return checkpoint(self.recomputed_inner, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)


def recompute(module):
    """Give ``module`` the recomputed forward; its forward before is kept as ``recomputed_inner`` (a bound method,
    so a deep copy rebinds both to the copy)."""
    module.recomputed_inner = module.forward
    module.forward = types.MethodType(recomputed_forward, module)
    return module


def stored(module):
    """``module``'s forward before ``recompute`` (the self-check's reference)."""
    return getattr(module, "recomputed_inner", module.forward)


__all__ = ["recompute", "recomputed_forward", "stored"]
