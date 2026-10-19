"""``reproducible=True`` (SPEC §9.3 rule 5 and "reproducible=True", §8.7 "Reproducible processes").

Plan task 2-T01 fixes the records; plan task 2-T16 fills the allow-list and ``qualify``. A
reproducible plan is bitwise within the scope of §9.3 (same hardware model, driver, software,
model, plan, seeds and restart cadence). It is offered only for a qualified (backend, model
family): plain cuEq with the unfused convolution, hybrid-only rewrites off, the deterministic
torch flag with its cuBLAS workspace set before CUDA starts, and every kernel in use on the
allow-list. A kernel without a deterministic implementation (``deterministic`` not True) makes
the planner refuse (``PlanRefused``, rule 5); a family that fails its qualification raises
``ReproducibilityError`` naming the blocking kernels.

This module imports neither torch nor MACE at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TASK = "plan task 2-T16"
# What an allow-list entry covers: a cuEquivariance kernel, a torch operator under
# torch.use_deterministic_algorithms(True), or one of FastMACE's own kernels.
KINDS = ("cueq-kernel", "torch-op", "fastmace-kernel")


@dataclass(frozen=True)
class AllowListEntry:
    """One component of a reproducible plan and whether it is deterministic.

    - ``component``: readable name ("ChannelWiseTensorProduct (unfused)", "scatter_add_");
    - ``kind``: one of ``KINDS``;
    - ``deterministic``: True, False, or None = undeclared (refused like False);
    - ``reason``: the evidence or the source of the claim;
    - ``condition``: what it holds under ("cuequivariance_ops 0.x pinned"), empty if unconditional.
    """

    component: str
    kind: str
    deterministic: bool | None
    reason: str
    condition: str = ""

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component:
            raise ValueError(f"AllowListEntry.component must be a non-empty string, got {self.component!r}")
        if self.kind not in KINDS:
            raise ValueError(f"AllowListEntry.kind {self.kind!r} is not one of {KINDS}")
        if self.deterministic is not None and not isinstance(self.deterministic, bool):
            raise ValueError(f"AllowListEntry.deterministic must be True, False or None, got {self.deterministic!r}")


@dataclass(frozen=True)
class Qualification:
    """The reproducibility verdict for one (backend, model family): ``qualified``, the ``blocking``
    components when not, and the allow-list ``entries`` it was judged on."""

    backend: str
    family: str
    qualified: bool
    blocking: tuple[str, ...] = ()
    entries: tuple[AllowListEntry, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "blocking", tuple(str(name) for name in self.blocking))
        object.__setattr__(self, "entries", tuple(self.entries))
        if self.qualified and self.blocking:
            raise ValueError(f"a qualified family has no blocking components, got {self.blocking}")


# The allow-list of every qualified family's components; empty until 2-T16 qualifies one.
ALLOW_LIST: tuple[AllowListEntry, ...] = ()


def qualify(plan: Any, *args: Any, **kwargs: Any) -> Qualification:
    """The ``Qualification`` of ``plan``'s backend and model family against ``ALLOW_LIST``."""
    raise NotImplementedError(f"reproducible qualification is {TASK}")
