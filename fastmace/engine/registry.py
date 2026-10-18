"""The runtime-state registry's record: one entry per persistent device buffer (SPEC §8.1, §10.6).

Plan task 1b-T01 fixes the record and its policies; plan task 1b-T04 builds the registry here
(seeded by ``GraphEngine.runtime_state``) and the snapshot/restore that capture warm-ups use; 1b-T14
calls the ``reset`` functions at canonical restart points. ``docs/design/contracts-1b.md`` gives
the policy of every buffer phase 1a has.

Two independent questions per buffer:

- **restart**: ``"save"`` (in the restart file), ``"reset"`` (recomputed at a canonical restart
  point by the function ``recompute`` names), or ``"derived"`` (recomputed from saved state when
  the engine is built or resumed);
- **capture**: snapshotted and restored around capture warm-ups (every registered buffer is, unless
  it is written before it is read in every stage); **sticky**: a flag that only a host decision
  clears, such as an overflow flag, which a warm-up must neither set nor clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RestartPolicy = Literal["save", "reset", "derived"]
RESTART_POLICIES = ("save", "reset", "derived")


@dataclass(frozen=True)
class RegisteredBuffer:
    """One persistent device buffer: where it lives, who owns it, and its policies.

    - ``name``: the attribute path from the engine, e.g. ``"force_field.polar.overflow"``;
    - ``owner``: the object that allocates and writes it (``"state"``, ``"integrator"``,
      ``"force_field"``, ``"neighbours"``, ``"polar"``, ``"engine"``);
    - ``restart``: one of ``RESTART_POLICIES``;
    - ``recompute``: for ``"reset"`` (required) and ``"derived"`` (optional), the name of the
      function that recomputes it; ``None`` for ``"save"``;
    - ``capture``: restored after capture warm-ups;
    - ``sticky``: a flag cleared only by a host decision.
    """

    name: str
    owner: str
    restart: RestartPolicy
    recompute: str | None = None
    capture: bool = True
    sticky: bool = False

    def __post_init__(self):
        for field in ("name", "owner"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"RegisteredBuffer.{field} must be a non-empty string, got {value!r}")
        if self.restart not in RESTART_POLICIES:
            raise ValueError(f"RegisteredBuffer {self.name}: restart policy {self.restart!r} is not one of "
                             f"{RESTART_POLICIES}")
        if self.recompute is not None and (not isinstance(self.recompute, str) or not self.recompute):
            raise ValueError(f"RegisteredBuffer {self.name}: recompute must name a function, got {self.recompute!r}")
        if self.restart == "reset" and self.recompute is None:
            raise ValueError(f"RegisteredBuffer {self.name}: a 'reset' buffer names the function that recomputes "
                             "it at the canonical restart point (recompute=...)")
        if self.restart == "save" and self.recompute is not None:
            raise ValueError(f"RegisteredBuffer {self.name}: a 'save' buffer comes from the restart file and is "
                             f"not recomputed (recompute={self.recompute!r})")
        for field in ("capture", "sticky"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"RegisteredBuffer {self.name}: {field} must be a bool, got {getattr(self, field)!r}")
