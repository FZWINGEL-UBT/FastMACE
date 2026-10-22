"""The hybrid MD/MC driver (SPEC §10.4; contracts: plan task 3-T01, filled by 3-T09 and 3-T10).

A block is ``every`` MD steps, then ``attempts`` attempts of each attached move against a baseline
evaluated with the MC evaluator; forces are recomputed once at the end of a block with an
acceptance (the transaction's resume, ``fastmace.events``). The three device streams
(``config.MC_STREAMS``) are seeded by ``config.mc_stream_seeds(seed)`` and advance by fixed counts per
attempt. Counters (attempted, impossible, accepted per move type) live on the device and are read
with the per-block flags. ``state_dict`` is the restart's ``mc`` record (``io.restart.MC_RECORD_KEYS``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fastmace.config import MCConfig


class MCDriver:
    """``MCDriver(moves, *, seed, config=None)``: the attached move set of a ``Simulation``."""

    def __init__(self, moves: Sequence[Any], *, seed: int, config: MCConfig | None = None):
        raise NotImplementedError("the MC driver: plan task 3-T09")

    def attach(self, engine: Any) -> None:
        """Bind to a built engine: streams, counters and the captured attempt graphs."""
        raise NotImplementedError("MCDriver.attach: plan task 3-T09")

    def run_block(self, engine: Any, step: int) -> Mapping[str, Any]:
        """One MC block at ``step`` (after its MD steps): the attempts and, after an acceptance, the resume."""
        raise NotImplementedError("MCDriver.run_block: plan task 3-T09")

    def state_dict(self) -> dict[str, Any]:
        """The restart's ``mc`` record (§10.6), keys ``io.restart.MC_RECORD_KEYS``."""
        raise NotImplementedError("MCDriver.state_dict: plan task 3-T09")

    def load_state_dict(self, record: Mapping[str, Any]) -> None:
        """Restore the move set's position, counters, stream positions and δ from a restart record."""
        raise NotImplementedError("MCDriver.load_state_dict: plan task 3-T09")
