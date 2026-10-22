"""Displacement moves (SPEC §10.4 "Proposals", "Fixed positions versus displacements", "Graph capture"; §8.5
step 4; plan task 3-T10).

One atom, chosen uniformly, moves to a uniform point in a sphere of radius δ. The proposal is symmetric,
so the acceptance is Metropolis on the MC evaluator's ΔE, ``moves.metropolis``: accept iff
``ln u < −ΔE/kT`` (``u = 0`` accepts), as for swaps (``docs/design/contracts-3.md``, "MC streams").

**Numbers.** Attempt ``k`` of a block reads ``PROPOSAL_COUNT`` = 5 proposal numbers from
``base + 5k``: number 0 as a uniform picks the atom ``min(floor(u N), N − 1)``, numbers 1–3 as normals
give the direction, number 4 as a uniform the radius ``δ u^(1/3)``. It reads one acceptance number at
``base + k`` and no velocity number (momenta stay). The counts are fixed whatever the outcome, and the
layout is a restart compatibility promise like the swap's.

**Two paths per attempt** (``DisplaceAttempt``, run by ``MCDriver``):
- *The captured attempt* (key ``("mc.displace", driver id, index, precision, edges)``): propose, apply,
  the rebuild test against the list's reference positions (the MD drift included), compaction, the
  energy-only stage, the decision and the masked restore, then compaction at the committed positions. It
  is valid only when no rebuild is needed and the compacted count fits the active edges. An invalid
  attempt restores itself, does not advance the device attempt index and sets ``halted``: the attempts
  replayed after it do nothing. The driver reads the index and the flag once after the move's replays.
- *The host path* for a halted attempt, with the same numbers (the index did not move): the committed
  list is saved (``SkinNeighbourList.save_list``), the attempt is applied, the list is built eagerly
  around the trial geometry when the test asks for it (``GraphEngine.trial_rebuild``: the GPU build
  reads its pair count on the host and may reallocate), the edges are sized for the trial
  (``use_trial_edges``) and the captured energy graph runs. Accept keeps the trial list
  (``adopt_trial_list``); reject restores the position and the committed list. The trial is never
  re-tested at the restored positions, so nothing loops (§8.5 step 4). The driver then replays the
  move's remaining attempts.

**Overflows** are execution events, not outcomes: an edge count beyond the active edges halts the
captured attempt, which the host path redoes on enough edges; a trial build that grows the capacity
(moved buffers, dropped graphs) restores the position, rebuilds at the committed positions and redoes
the attempt with the same numbers. Both count in the engine's ``overflows_redone``.

**δ** is a device scalar, so changing it needs no re-capture. It is fixed while sampling; with
``adaptive=True`` the driver changes it at the end of each block, δ ← δ·clamp(acc/0.3, 0.5, 2), at most
``delta_A`` and ``Workload.max_displacement_A`` (3-T10 answer 2). Every value is recorded
(``delta_history``; the restart keeps the current one).

Displacements under a variable cell are refused (``displace-variable-cell``; 3-T10 answer 3), and so is a
``delta_A`` beyond ``Workload.max_displacement_A``, which sized the skin (rule 11).
"""

from __future__ import annotations

import math
from typing import Any

from fastmace.errors import PlanRefused, Reason
from fastmace.mc import streams
from fastmace.mc.moves import Displace

PROPOSAL_COUNT = 5  # atom uniform, three normals (direction), radius uniform
ACCEPTANCE_COUNT = streams.ACCEPTANCE_COUNT
# A displacement move's counts in a block record: io.restart.MC_COUNTER_KEYS ("impossible" stays 0: every
# atom can be displaced), then its trial builds (3-T10 answer 4; a run counter, not restart state).
COUNTER_KEYS = ("attempted", "impossible", "accepted", "trial_rebuilds")
ADAPT_TARGET = 0.3  # the acceptance adaptive δ steers to
ADAPT_LIMITS = (0.5, 2.0)  # the factor per block
VARIABLE_CELL_CODE = "displace-variable-cell"
BEYOND_MAX_CODE = "displace-beyond-max-displacement"
MOVE_KEY = "mc.displace"


def attempt_counts(natoms: int) -> dict[str, int]:
    """Numbers per attempt by stream (``config.MC_STREAMS`` order), the same for every outcome."""
    return {"proposal": PROPOSAL_COUNT, "acceptance": ACCEPTANCE_COUNT, "velocity": 0}


def decode_displace(u, g, natoms: int, delta):
    """A displacement from an attempt's proposal numbers, given as uniforms ``u`` and normals ``g`` (``[5]``, both
    from the same numbers): the atom ``min(floor(u[0] N), N − 1)`` and the vector ``δ u[4]^(1/3) g[1:4]/|g[1:4]|``,
    uniform in the ball of radius ``δ`` (a ``[1]`` float64 tensor). Returns ``(atom [1] int64, d [1, 3])``.
    Device operations only, so it runs inside a captured graph."""
    import torch

    atom = torch.clamp((u[:1] * natoms).to(torch.int64), max=natoms - 1)
    direction = g[1:4]
    norm = torch.clamp(torch.linalg.vector_norm(direction), min=torch.finfo(torch.float64).tiny)
    radius = delta * torch.pow(u[4:5], 1.0 / 3.0)
    return atom, (radius * direction / norm).reshape(1, 3)


def check_displace(move: Displace, *, max_displacement_A: float | None, variable_cell: bool) -> None:
    """Refuse what a displacement move cannot run with, every reason at once (``PlanRefused``): a variable cell,
    and a ``delta_A`` beyond ``Workload.max_displacement_A`` (or none declared), which rule 11 sized the skin for."""
    reasons = []
    if variable_cell:
        reasons.append(Reason("displacement moves under a variable cell (NPT) are not offered in phase 3: run them "
                              "at a fixed cell; swaps and transmutations work under NPT", code=VARIABLE_CELL_CODE))
    if max_displacement_A is None or move.delta_A > max_displacement_A:
        declared = "none is declared" if max_displacement_A is None else f"it is {max_displacement_A:g} A"
        reasons.append(Reason(f"Displace(delta_A={move.delta_A:g}) needs Workload.max_displacement_A >= delta_A, which "
                              f"sizes the skin (rule 11), but {declared}", code=BEYOND_MAX_CODE))
    if reasons:
        raise PlanRefused.from_reasons(reasons)


def adapted_delta(delta: float, accepted: int, attempted: int, *, delta_A: float, max_displacement_A: float) -> float:
    """The next block's δ: δ·clamp(acc / 0.3, 0.5, 2), at most ``delta_A`` and ``max_displacement_A``; unchanged
    when nothing was attempted."""
    if attempted <= 0:
        return float(delta)
    factor = min(max((accepted / attempted) / ADAPT_TARGET, ADAPT_LIMITS[0]), ADAPT_LIMITS[1])
    return float(min(delta * factor, delta_A, max_displacement_A))


class DisplaceAttempt:
    """The attempt of a ``Displace`` move on a built graph engine, in the shape of ``fastmace.mc.moves.Attempt``:
    ``bound(stage, trial)`` is what the driver captures, ``counters`` (int64 ``[3]``: attempted, impossible,
    accepted) count on the device, and the block's ``streams`` (``streams.BlockStreams``) give the numbers,
    the attempt index, kT and the baseline. ``moves.build_attempt`` builds it; ``MCDriver`` runs it.

    A captured attempt that cannot decide (a rebuild is needed, or the count passes the active edges)
    restores itself and sets ``halted``; the driver then calls ``host_attempt()`` for it and replays the
    rest (module docstring). ``delta`` is a device scalar (``set_delta``); ``max_displacement_A`` caps it
    (the driver sets the workload's at attach).
    """

    def __init__(self, move: Displace, engine: Any, evaluator: Any, block_streams: Any):
        import torch

        from fastmace.mc import moves

        if not isinstance(move, Displace):
            raise TypeError(f"DisplaceAttempt needs a Displace move, got {type(move).__name__}")
        if not callable(getattr(engine, "trial_rebuild", None)):
            raise TypeError(f"displacement attempts need a graph engine, got {type(engine).__name__}")
        self.move, self.engine, self.evaluator, self.streams = move, engine, evaluator, block_streams
        self.max_displacement_A = move.delta_A
        state = engine.state
        device = state.positions.device
        self.natoms = int(state.positions.shape[0])
        self.velocities = None  # momenta stay
        self.counts = attempt_counts(self.natoms)
        self.block_velocity_count = 0
        self.write_charges, self.charges = False, None
        self.accept = moves.metropolis  # looked up when the attempt is built, as for swaps (tests replace it)
        self.counters = torch.zeros(3, dtype=torch.int64, device=device)
        self.accepted = torch.zeros(1, dtype=torch.bool, device=device)
        self.halted = torch.zeros(1, dtype=torch.int64, device=device)
        self.trial_energy = torch.zeros(1, dtype=torch.float64, device=device)  # the last evaluated trial's
        self.delta = torch.full((1,), move.delta_A, dtype=torch.float64, device=device)
        self._old = torch.zeros(1, 3, dtype=torch.float64, device=device)
        self.stage, self.trial = None, None
        self._saved = None  # the committed list during a host-path attempt (SkinNeighbourList.save_list)
        self.delta_A = move.delta_A
        self.delta_history = [move.delta_A]
        self.trial_rebuilds = 0  # host count, cumulative (3-T10 answer 4)

    # ------------------------------------------------------------------ the captured attempt

    def bound(self, stage, trial):
        """``run`` with one edge size's energy stage and output (``MCEvaluator.stage``), for the driver to capture."""

        def run() -> None:
            self.stage, self.trial = stage, trial
            self.run()

        return run

    def _proposal(self):
        block = self.streams
        u = block.uniforms("proposal", PROPOSAL_COUNT)
        g = block.normals("proposal", PROPOSAL_COUNT)
        acceptance = block.uniforms("acceptance", ACCEPTANCE_COUNT)
        atom, d = decode_displace(u, g, self.natoms, self.delta)
        return atom, d, acceptance

    def _energy(self):
        if self.stage is not None:
            self.stage()
            return self.trial
        return self.evaluator.evaluate()

    def run(self) -> None:
        import torch

        state, field, block = self.engine.state, self.engine.force_field, self.streams
        edges = int(field.data["edge_index"].shape[-1])
        with torch.no_grad():
            positions = state.positions
            active = self.halted == 0
            atom, d, acceptance = self._proposal()
            old = positions.index_select(0, atom)
            self._old.copy_(old)
            new = old + d * active.to(torch.float64)[:, None]
            positions.index_copy_(0, atom, new)
            rebuild = field.needs_rebuild(positions).reshape(1)
            field.compact(positions)
            trial = self._energy()  # flags an edge count beyond the active edges
            valid = active & ~rebuild & (field.edge_count.reshape(1) <= edges)
            accepted = valid & self.accept(acceptance, trial - block.baseline, block.kT)
            self.accepted.copy_(accepted)
            positions.index_copy_(0, atom, torch.where(accepted[:, None], new, self._old))
            field.compact(positions)  # the list of the committed positions
            self.trial_energy.copy_(torch.where(valid, trial, self.trial_energy))
            block.baseline.copy_(torch.where(accepted, trial, block.baseline))
            self.counters.add_(torch.cat((valid, torch.zeros_like(valid), accepted)).to(torch.int64))
            block.index.add_(valid.to(torch.int64))
            self.halted.bitwise_or_((active & ~valid).to(torch.int64))

    def tensors(self):
        """The attempt's own mutable device tensors (snapshots around a capture)."""
        return (self.counters, self.accepted, self.halted, self.trial_energy, self._old)

    # ------------------------------------------------------------------ the host path

    def poll(self) -> tuple[int, bool]:
        """``(attempt index, halted)`` from the device (one host read)."""
        import torch

        index, halted = (int(value) for value in torch.cat((self.streams.index, self.halted)).tolist())
        return index, bool(halted)

    def host_attempt(self) -> bool:
        """The halted attempt again, with the same numbers, outside the graph (module docstring). Returns
        whether it took a trial build around its geometry."""
        import torch

        engine, state, field = self.engine, self.engine.state, self.engine.force_field
        # The committed lists of a trial build: the MACE list's, and with D3 the D3 term's too (3b-T04).
        save = getattr(field, "save_trial", field.neighbours.save_list)
        restore = getattr(field, "restore_trial", field.neighbours.restore_list)
        word = engine.overflowed()
        if word:  # the captured attempt's edges were too few: redone below on enough edges
            engine.record_mc_overflow(word)
        self.halted.zero_()
        with torch.no_grad():
            atom, d, acceptance = self._proposal()
            while True:
                self._saved = save(self._saved)
                positions = state.positions
                old = positions.index_select(0, atom)
                positions.index_copy_(0, atom, old + d)
                rebuild = bool(field.needs_rebuild(positions))  # host sync
                if rebuild and engine.trial_rebuild():
                    # The trial build grew the capacity: restore, build at the committed positions, redo.
                    positions.index_copy_(0, atom, old)
                    engine.trial_rebuild()
                    engine.use_trial_edges()
                    engine.record_mc_overflow(engine.overflowed() or _edge_overflow())
                    continue
                engine.use_trial_edges()
                energy = self.evaluator.evaluate()
                word = engine.overflowed()
                if word:  # pragma: no cover - the edges were sized for this count; grow and redo if not
                    positions.index_copy_(0, atom, old)
                    if rebuild:
                        restore(self._saved)
                    engine.use_trial_edges()
                    engine.record_mc_overflow(word)
                    continue
                break
            block = self.streams
            accepted = bool(self.accept(acceptance, energy - block.baseline, block.kT))  # host sync
            self.trial_energy.copy_(energy)
            self.accepted.fill_(accepted)
            if accepted:
                block.baseline.copy_(energy)
                if rebuild:
                    engine.adopt_trial_list()
            else:
                positions.index_copy_(0, atom, old)
                if rebuild:
                    restore(self._saved)
                engine.use_trial_edges()
            self.counters.add_(torch.tensor([1, 0, int(accepted)], dtype=torch.int64, device=self.counters.device))
            block.index.add_(1)
        if rebuild:
            self.trial_rebuilds += 1
        return rebuild

    # ------------------------------------------------------------------ δ

    def set_delta(self, delta_A: float, *, record: bool = True) -> None:
        """Set δ (Å; a device fill, no re-capture), at most the move's ``delta_A`` and the workload's maximum;
        ``record``: append it to ``delta_history``."""
        delta_A = float(delta_A)
        limit = min(self.move.delta_A, self.max_displacement_A)
        if not (math.isfinite(delta_A) and 0 < delta_A <= limit):
            raise ValueError(f"delta_A must be in (0, {limit:g}], got {delta_A!r}")
        self.delta.fill_(delta_A)
        self.delta_A = delta_A
        if record:
            self.delta_history.append(delta_A)

    def adapt(self, accepted: int, attempted: int) -> float:
        """With ``adaptive``: the next block's δ from this block's acceptance (answer 2); returns δ."""
        if self.move.adaptive:
            self.set_delta(adapted_delta(self.delta_A, accepted, attempted, delta_A=self.move.delta_A,
                                         max_displacement_A=self.max_displacement_A))
        return self.delta_A


def _edge_overflow() -> int:
    from fastmace.engine.graph import EDGE_OVERFLOW

    return EDGE_OVERFLOW


__all__ = ["ACCEPTANCE_COUNT", "ADAPT_LIMITS", "ADAPT_TARGET", "BEYOND_MAX_CODE", "COUNTER_KEYS", "MOVE_KEY",
           "PROPOSAL_COUNT", "VARIABLE_CELL_CODE", "DisplaceAttempt", "adapted_delta", "attempt_counts",
           "check_displace", "decode_displace"]
