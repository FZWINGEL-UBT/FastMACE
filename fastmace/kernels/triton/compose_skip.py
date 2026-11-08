"""Interaction 1's reassociated projection composed with its receiver-species skip (Codex H200 follow-up, item 1;
7 November 2026; ``results/h200-compose-quads-20261107``).

In a density interaction the first interaction's message per degree l is ``left_l @ right_l``
(``FirstInteractionReassociation``), through the linear's ``L_l`` (a per-l ``BlockLinear``, or the identity after
the generic fold), divided by ``density + 1`` and multiplied by the receiver's species matrix ``B_l,z``
(``SpeciesSkip``). The division is per row, so the products compose:
``(left_l right_l L_l / (d + 1)) B_l,z = left_l (right_l L_l B_l,z) / (d + 1)``. ``composed`` builds
``C_l [species, K, mul_out]`` (in float64, stored in the input's dtype) and the reassociation runs
``species_mm(left_l, C_l)``: one per-species product per l instead of the projection, the linear and the skip's
all-species product (3x the columns, forward and backward). The interaction then skips its linear and ``skip_tp``.

Not bitwise: the composed matrices round differently from the two products (a rounding-class change: force/stress
gate). Lab, H200, forward + backward at OMAT-0 medium's shapes: 0.960 -> 0.448 ms (species_mm's CUTLASS launch).

``COMPOSE_SKIP``: None: on for ``COMPOSE_SKIP_GPUS``; True / False: an A/B side's choice. Set at build
(``precision.route``) where the interaction qualifies (``generic.modules.compose_skip_products``).
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

COMPOSE_SKIP = None
# NVIDIA H200 (7 November 2026, results/h200-compose-quads-20261107): MD step b/a 0.9698 / 0.9738 against the H200
# defaults (OMAT-0 medium, SrTiO3 10x10x10, Benchmark_loup settings, ab / ba), force/stress gate passes (decision 66).
COMPOSE_SKIP_GPUS = GPUModels(("NVIDIA H200",))


def compose_skip_default(name=None) -> bool:
    """``COMPOSE_SKIP`` (None: ``name`` in ``COMPOSE_SKIP_GPUS``)."""
    on = COMPOSE_SKIP
    if on is None:
        on = name in COMPOSE_SKIP_GPUS
    return bool(on)


def composed(right, ws, linear=None):
    """``right [K, c] @ linear [c, mul_in] @ ws[z] [mul_in, mul_out]`` for every species z (``linear`` None: the
    identity), as ``(C [species, K, mul_out], C transposed [species, mul_out, K])``, both contiguous, computed in
    float64 and returned in ``right``'s dtype."""
    left = right.double() if linear is None else right.double() @ linear.double()
    c = torch.matmul(left[None], ws.double()).to(right.dtype)
    return c.contiguous(), c.transpose(1, 2).contiguous()


__all__ = ["COMPOSE_SKIP", "COMPOSE_SKIP_GPUS", "compose_skip_default", "composed"]
