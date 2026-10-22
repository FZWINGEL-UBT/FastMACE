"""Displacement moves (SPEC §10.4 "Proposals", "Fixed positions versus displacements", "Graph capture"; §8.5
step 4; plan task 3-T10).

One atom, chosen uniformly, moves to a uniform point in a sphere of radius δ. The proposal is symmetric,
so the acceptance is Metropolis on the MC evaluator's ΔE: accept iff ``ln u < −ΔE/kT`` (``u = 0`` accepts),
as for swaps (``docs/design/contracts-3.md``, "MC streams").

**Numbers.** Attempt ``k`` of a block reads ``PROPOSAL_COUNT`` = 5 proposal numbers from
``base + 5k``: number 0 as a uniform picks the atom ``min(floor(u N), N − 1)``, numbers 1–3 as normals
give the direction, number 4 as a uniform the radius ``δ u^(1/3)``. It reads one acceptance number at
``base + k`` and no velocity number (momenta stay). The counts are fixed whatever the outcome, and the
layout is a restart compatibility promise like the swap's.

**Two paths per attempt.**
- *The captured attempt* (key ``("mc.displace", index, precision, edges)``): propose, apply, the rebuild
  test against the list's reference positions (the MD drift included), compaction, the energy-only
  stage, the decision and the masked restore, then compaction at the committed positions. It is valid
  only when no rebuild is needed and the compacted count fits the active edges. An invalid attempt
  restores itself, does not advance the device attempt index and sets ``halted``: the attempts replayed
  after it do nothing, and the host reads the index and the flag once per block.
- *The host path* for a halted attempt, with the same numbers (the index did not move): the committed
  list is saved (``SkinNeighbourList.save_list``), the attempt is applied, the list is built eagerly
  around the trial geometry when the test asks for it (``GraphEngine.trial_rebuild``: the GPU build
  reads its pair count on the host and may reallocate), the edges are sized for the trial
  (``use_trial_edges``) and the captured energy graph runs. Accept keeps the trial list
  (``adopt_trial_list``); reject restores the position and the committed list. The trial is never
  re-tested at the restored positions, so nothing loops (§8.5 step 4).

**Overflows** are execution events, not outcomes: an edge count beyond the active edges halts the
captured attempt, which the host path redoes on enough edges; a trial build that grows the capacity
(moved buffers, dropped graphs) restores the position, rebuilds at the committed positions and redoes
the attempt with the same numbers. Both count in the engine's ``overflows_redone``.

**δ** is a device scalar, so changing it needs no re-capture. It is fixed while sampling; with
``adaptive=True`` it changes at the end of each block, δ ← δ·clamp(acc/0.3, 0.5, 2), at most ``delta_A``
and ``Workload.max_displacement_A`` (3-T10 answer 2). Every value is recorded (``delta_history``).

Displacements under a variable cell are refused (``displace-variable-cell``; 3-T10 answer 3), and so is a
``delta_A`` beyond ``Workload.max_displacement_A``, which sized the skin (rule 11).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from fastmace.errors import PlanRefused, Reason
from fastmace.mc import streams
from fastmace.mc.moves import K_B_EV_PER_K, Displace

PROPOSAL_COUNT = 5  # atom uniform, three normals (direction), radius uniform
ACCEPTANCE_COUNT = streams.ACCEPTANCE_COUNT
# io.restart.MC_COUNTER_KEYS, then the trial builds of this move (3-T10 answer 4). "impossible" stays 0:
# every atom can be displaced.
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


class DisplaceAttempts:
    """The attempts of one ``Displace`` move on a built graph engine, with the MC ``evaluator`` (3-T07).

    ``seeds``: the stream seeds (``config.mc_stream_seeds``). ``max_displacement_A``: the workload's
    ``max_displacement_A``. ``index``: the move's place in the attached set (part of its graph key).
    ``offsets``: the streams' positions (host ints), advanced by the fixed counts after each block; the
    MC driver sets them before a block when moves share the streams.

    ``run(attempts, temperature_K=, baseline=)`` runs one block's attempts against ``baseline`` (the
    block's MC-evaluator energy, a device tensor) and leaves the baseline after its acceptances in
    ``baseline``. A block with an acceptance must end at a canonical point (the driver's resume).
    """

    def __init__(self, move: Displace, engine: Any, evaluator: Any, *, seeds: Mapping[str, int],
                 max_displacement_A: float | None, index: int = 0, offsets: Mapping[str, int] | None = None):
        import torch

        if not isinstance(move, Displace):
            raise TypeError(f"DisplaceAttempts needs a Displace move, got {type(move).__name__}")
        check_displace(move, max_displacement_A=max_displacement_A, variable_cell=bool(engine.variable_cell))
        if not callable(getattr(engine, "trial_rebuild", None)):
            raise TypeError(f"displacement attempts need a graph engine, got {type(engine).__name__}")
        self.move, self.engine, self.evaluator, self.index_in_set = move, engine, evaluator, int(index)
        self.seeds = {name: int(seeds[name]) for name in ("proposal", "acceptance")}
        self.max_displacement_A = float(max_displacement_A)
        self.offsets = {"proposal": 0, "acceptance": 0} if offsets is None else {
            name: int(offsets[name]) for name in ("proposal", "acceptance")}
        self.natoms = int(engine.state.positions.shape[0])
        device = engine.state.positions.device

        def zeros(*shape, dtype=torch.float64):
            return torch.zeros(shape, dtype=dtype, device=device)

        # Device state of a block; the captured attempt reads and writes these at fixed addresses.
        self.delta = torch.full((1,), move.delta_A, dtype=torch.float64, device=device)
        self.kT = zeros(1)
        self.baseline = zeros(1)  # the block's baseline after the attempts so far
        self.trial_energy = zeros(1)  # the last evaluated trial's energy
        self.base = zeros(2, dtype=torch.int64)  # stream offsets at the block's start: proposal, acceptance
        self.index = zeros(1, dtype=torch.int64)  # the attempt index: advances only when an attempt completes
        self.halted = zeros(1, dtype=torch.int64)  # an attempt needed the host path; later replays do nothing
        self.counts = zeros(2, dtype=torch.int64)  # attempted, accepted in this block
        self._old = zeros(1, 3)
        self._keep = (self.baseline, self.trial_energy, self.index, self.halted, self.counts, self._old)
        self._replays = {}
        self._saved = None  # the committed list during a host-path attempt (SkinNeighbourList.save_list)
        self.delta_A = move.delta_A
        self.delta_history = [move.delta_A]
        self.counters = dict.fromkeys(COUNTER_KEYS, 0)

    # ------------------------------------------------------------------ numbers

    def _numbers(self, index):
        """Attempt ``index``'s proposal numbers as uniforms and normals, and its acceptance uniform."""
        offsets = streams.positions(self.base[0] + index * PROPOSAL_COUNT, PROPOSAL_COUNT)
        seed = self.seeds["proposal"]
        acceptance = streams.uniforms(self.seeds["acceptance"], streams.positions(self.base[1] + index, 1))
        return streams.uniforms(seed, offsets), streams.normals(seed, offsets), acceptance

    def _accepts(self, energy, acceptance):
        import torch

        return (acceptance == 0) | (torch.log(acceptance) < -(energy - self.baseline) / self.kT)

    # ------------------------------------------------------------------ the captured attempt

    def _attempt(self, stage, out, edges: int):
        import torch

        state, field = self.engine.state, self.engine.force_field

        def attempt():
            with torch.no_grad():
                positions = state.positions
                active = self.halted == 0
                u, g, acceptance = self._numbers(self.index)
                atom, d = decode_displace(u, g, self.natoms, self.delta)
                old = positions.index_select(0, atom)
                self._old.copy_(old)
                new = old + d * active.to(torch.float64)[:, None]
                positions.index_copy_(0, atom, new)
                rebuild = field.needs_rebuild(positions).reshape(1)
                field.compact(positions)
                stage()  # the trial energy into ``out``; flags an edge count beyond ``edges``
                valid = active & ~rebuild & (field.edge_count.reshape(1) <= edges)
                accept = valid & self._accepts(out, acceptance)
                positions.index_copy_(0, atom, torch.where(accept[:, None], new, self._old))
                field.compact(positions)  # the list of the committed positions
                self.trial_energy.copy_(torch.where(valid, out, self.trial_energy))
                self.baseline.copy_(torch.where(accept, out, self.baseline))
                self.counts.add_(torch.cat((valid, accept)).to(torch.int64))
                self.index.add_(valid.to(torch.int64))
                self.halted.bitwise_or_((active & ~valid).to(torch.int64))

        return attempt

    def _replay(self):
        """The captured attempt for the edges active now (one variant per edge size)."""
        edges = int(self.engine.force_field.data["edge_index"].shape[-1])
        key = (MOVE_KEY, self.index_in_set, self.evaluator.precision, edges)
        replay = self._replays.get(key)
        if replay is None:
            stage, out = self.evaluator.stage()
            replay = self.engine.captured(key, self._attempt(stage, out, edges), keep=self._keep)
            self._replays[key] = replay
        return replay

    # ------------------------------------------------------------------ the host path

    def _host_attempt(self) -> None:
        """The halted attempt again, with the same numbers, outside the graph (module docstring)."""
        import torch

        engine, state, field = self.engine, self.engine.state, self.engine.force_field
        neighbours = field.neighbours
        word = engine.overflowed()
        if word:  # the captured attempt's edges were too few: redone below on enough edges
            engine.record_mc_overflow(word)
        self.halted.zero_()
        with torch.no_grad():
            u, g, acceptance = self._numbers(self.index)
            atom, d = decode_displace(u, g, self.natoms, self.delta)
            while True:
                self._saved = neighbours.save_list(self._saved)
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
                        neighbours.restore_list(self._saved)
                    engine.use_trial_edges()
                    engine.record_mc_overflow(word)
                    continue
                break
            accept = self._accepts(energy, acceptance)
            accepted = bool(accept)  # host sync
            self.trial_energy.copy_(energy)
            if accepted:
                self.baseline.copy_(energy)
                if rebuild:
                    engine.adopt_trial_list()
            else:
                positions.index_copy_(0, atom, old)
                if rebuild:
                    neighbours.restore_list(self._saved)
                engine.use_trial_edges()
            self.counts.add_(torch.tensor([1, int(accepted)], dtype=torch.int64, device=self.counts.device))
            self.index.add_(1)
        if rebuild:
            self.counters["trial_rebuilds"] += 1
            self._block_trial_rebuilds += 1

    # ------------------------------------------------------------------ a block

    def run(self, attempts: int, *, temperature_K: float, baseline) -> dict[str, Any]:
        """``attempts`` attempts against ``baseline`` at ``temperature_K`` (the integrator's target of the block).

        One host read per block, plus the host path for each halted attempt. Returns the block's record:
        attempted, impossible, accepted, trial_rebuilds, overflows_redone, delta_A (the δ the block used) and,
        with ``adaptive``, adapted_delta_A (the next block's).
        """
        import torch

        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError(f"attempts must be an int of at least 1, got {attempts!r}")
        engine = self.engine
        engine.quiesce()
        redos = int(engine.overflows_redone)
        self._block_trial_rebuilds = 0
        with torch.no_grad():
            self.baseline.copy_(torch.as_tensor(baseline, dtype=torch.float64).reshape(1))
            self.kT.fill_(K_B_EV_PER_K * float(temperature_K))
            self.base.copy_(torch.tensor([self.offsets["proposal"], self.offsets["acceptance"]], dtype=torch.int64))
            for tensor in (self.index, self.halted, self.counts):
                tensor.zero_()
        done = 0
        while done < attempts:
            replay = self._replay()
            for _ in range(attempts - done):
                replay()
            index, halted = (int(value) for value in torch.cat((self.index, self.halted)).tolist())  # host sync
            if not halted:
                if index != attempts:  # pragma: no cover - a bug, never an outcome
                    raise RuntimeError(f"displacement block ended at attempt {index} of {attempts} without a halt")
                break
            self._host_attempt()
            done = index + 1
        attempted, accepted = (int(value) for value in self.counts.tolist())
        self.offsets = {"proposal": self.offsets["proposal"] + attempts * PROPOSAL_COUNT,
                        "acceptance": self.offsets["acceptance"] + attempts * ACCEPTANCE_COUNT}
        self.counters["attempted"] += attempted
        self.counters["accepted"] += accepted
        record = {"move": Displace.move, "attempts": attempts, "attempted": attempted, "impossible": 0,
                  "accepted": accepted, "trial_rebuilds": self._block_trial_rebuilds,
                  "overflows_redone": int(engine.overflows_redone) - redos, "delta_A": self.delta_A}
        if self.move.adaptive:
            self.set_delta(adapted_delta(self.delta_A, accepted, attempted, delta_A=self.move.delta_A,
                                         max_displacement_A=self.max_displacement_A))
            record["adapted_delta_A"] = self.delta_A
        return record

    def set_delta(self, delta_A: float) -> None:
        """Set δ (Å; a device fill, no re-capture) and record it; at most the move's ``delta_A`` and the workload's
        maximum."""
        delta_A = float(delta_A)
        if not (math.isfinite(delta_A) and 0 < delta_A <= min(self.move.delta_A, self.max_displacement_A)):
            raise ValueError(f"delta_A must be in (0, {min(self.move.delta_A, self.max_displacement_A):g}], "
                             f"got {delta_A!r}")
        self.delta.fill_(delta_A)
        self.delta_A = delta_A
        self.delta_history.append(delta_A)

    # ------------------------------------------------------------------ restart

    def state_dict(self) -> dict[str, Any]:
        """The move's part of the restart's ``mc`` record: δ and its history, the counters and the streams."""
        return {"move": Displace.move, "delta_A": self.delta_A, "delta_history": list(self.delta_history),
                "counters": dict(self.counters),
                "streams": {name: {"seed": self.seeds[name], "offset": self.offsets[name]} for name in self.seeds}}

    def load_state_dict(self, record: Mapping[str, Any]) -> None:
        if record.get("move") != Displace.move:
            raise ValueError(f"not a displacement record: {record.get('move')!r}")
        for name, entry in record["streams"].items():
            if int(entry["seed"]) != self.seeds[name]:
                raise ValueError(f"the {name} stream's seed differs from the record's")
        self.offsets = {name: int(entry["offset"]) for name, entry in record["streams"].items()}
        self.counters = {key: int(record["counters"][key]) for key in COUNTER_KEYS}
        self.delta_A = float(record["delta_A"])
        self.delta.fill_(self.delta_A)
        self.delta_history = [float(value) for value in record["delta_history"]]


def _edge_overflow() -> int:
    from fastmace.engine.graph import EDGE_OVERFLOW

    return EDGE_OVERFLOW


__all__ = ["ACCEPTANCE_COUNT", "ADAPT_LIMITS", "ADAPT_TARGET", "BEYOND_MAX_CODE", "COUNTER_KEYS", "MOVE_KEY",
           "PROPOSAL_COUNT", "VARIABLE_CELL_CODE", "DisplaceAttempts", "adapted_delta", "attempt_counts",
           "check_displace", "decode_displace"]
