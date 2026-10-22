"""The hybrid MD/MC driver (SPEC §10.4; contracts: plan task 3-T01, filled by 3-T09 and 3-T10).

A block is ``every`` MD steps, then ``attempts`` attempts of each due move against a baseline
evaluated with the MC evaluator; forces are recomputed once at the end of a block with an
acceptance (a canonical point, ``GraphEngine.canonicalise(reevaluate=True)``). The three device streams
(``config.MC_STREAMS``) are seeded by ``config.mc_stream_seeds(seed)`` and advance by fixed counts per
attempt (``fastmace.mc.streams``). Counters (attempted, impossible, accepted per move type) live on the
device and are read once per block, with the attempt indices and the overflow word. ``state_dict`` is the
restart's ``mc`` record (``io.restart.MC_RECORD_KEYS``).

The driver is attached to a built engine (``attach``); ``Simulation.run(mc=...)`` wires it (3-T15).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from fastmace.config import MC_STREAMS, MCConfig, mc_stream_seeds
from fastmace.mc.moves import Displace, Swap, Transmute

_IDS = itertools.count()
# k_B in eV/K (ase.units.kB, CODATA 2018), as fastmace.mc.moves.
K_B_EV_PER_K = 8.617333262e-5
COUNTER_KEYS = ("attempted", "impossible", "accepted")


def move_record(move: Swap | Transmute | Displace) -> dict[str, Any]:
    """A move as a JSON-safe dict with its ``move`` type (the restart's ``moves`` entries)."""
    record = {"move": move.move}
    for field in dataclasses.fields(move):
        value = getattr(move, field.name)
        if isinstance(value, tuple):
            value = [list(item) if isinstance(item, tuple) else item for item in value]
        record[field.name] = value
    return record


def move_from_record(record: Mapping[str, Any]) -> Swap | Transmute | Displace:
    """The move record of ``move_record``'s dict."""
    values = {key: value for key, value in record.items() if key != "move"}
    kind = record.get("move")
    if kind == "swap":
        return Swap(**{**values, "pairs": [tuple(pair) for pair in values["pairs"]]})
    if kind == "transmute":
        return Transmute(**{**values, "mu": [tuple(pair) for pair in values.get("mu") or ()],
                            "formal_charges": [tuple(pair) for pair in values.get("formal_charges") or ()]})
    if kind == "displace":
        return Displace(**values)
    raise ValueError(f"unknown MC move record {record!r}")


class MCDriver:
    """``MCDriver(moves, *, seed, config=None)``: the attached move set of a ``Simulation``.

    ``moves``: ``fm.mc`` records (one record or a sequence). ``seed``: the simulation's seed, from which
    the three streams' seeds derive. ``config``: ``MCConfig`` limits.
    """

    def __init__(self, moves: Sequence[Any], *, seed: int, config: MCConfig | None = None):
        moves = (moves,) if isinstance(moves, (Swap, Transmute, Displace)) else tuple(moves)
        if not moves:
            raise ValueError("MCDriver needs at least one move")
        for move in moves:
            if not isinstance(move, (Swap, Transmute, Displace)):
                raise TypeError(f"MCDriver takes fm.mc move records, got {type(move).__name__}")
        self.config = MCConfig() if config is None else config
        if len(moves) > self.config.max_move_sets:
            raise ValueError(f"{len(moves)} moves attached at once, above MCConfig.max_move_sets="
                             f"{self.config.max_move_sets}")
        for move in moves:
            if move.attempts > self.config.max_attempts_per_block:
                raise ValueError(f"{move.move} attempts={move.attempts} is above MCConfig.max_attempts_per_block="
                                 f"{self.config.max_attempts_per_block}")
        self.moves = moves
        self.seed = seed
        self.seeds = mc_stream_seeds(seed)
        self.offsets = {stream: 0 for stream in MC_STREAMS}
        self.counters = {move.move: dict.fromkeys(COUNTER_KEYS, 0) for move in moves}
        self.cycle_position = 0
        self.decisions: tuple = ()
        self.last_block: dict[str, Any] | None = None
        self.blocks = 0
        self.engine = None
        self.evaluator = None
        self.attempts: list = []
        self.streams = None
        self._replays: dict = {}
        self._delta_A: dict = {}  # restart record's δ by move key, until attach (3-T10)
        self.id = next(_IDS)

    # ------------------------------------------------------------------ setup

    def attach(self, engine: Any, *, evaluator: Any = None, max_displacement_A: float | None = None) -> None:
        """Bind to a built engine: the move checks, the device tables, streams and counters, and the attempts
        (captured at their first replay). ``evaluator``: the MC evaluator (``fastmace.mc.evaluator.MCEvaluator``);
        ``None`` evaluates in the force precision. ``max_displacement_A``: the workload's
        (``Workload.max_displacement_A``), required with ``Displace`` moves, which are refused above it and
        under a variable cell (``fastmace.mc.displace.check_displace``; 3-T10)."""
        from fastmace.mc import moves as moves_module
        from fastmace.mc.displace import check_displace
        from fastmace.mc.evaluator import MCEvaluator
        from fastmace.mc.moves import build_attempt
        from fastmace.mc.streams import BlockStreams
        from fastmace.planner.minimal import check_mc_moves

        for move in self.moves:
            if isinstance(move, Displace):
                check_displace(move, max_displacement_A=max_displacement_A,
                               variable_cell=bool(getattr(engine, "variable_cell", False)))
        field, state = engine.force_field, engine.state
        superset = getattr(field, "superset", None)
        if superset is None:
            superset = sorted(set(state.species.tolist()))
        charge_inputs = moves_module.charge_inputs(field)
        fixcm = bool(getattr(engine.integrator.config, "fixcm", False))
        self.decisions = check_mc_moves(self.moves, superset=superset, charge_inputs=charge_inputs, fixcm=fixcm)
        self.engine = engine
        self.evaluator = MCEvaluator(engine) if evaluator is None else evaluator
        self.streams = BlockStreams(self.seeds, state.positions.device)
        self.attempts = [build_attempt(move, engine, self.evaluator, self.streams, fixcm=fixcm) for move in self.moves]
        for attempt, move in zip(self.attempts, self.moves, strict=True):
            if isinstance(move, Displace):
                attempt.max_displacement_A = float(max_displacement_A)
                delta = self._delta_A.get(self._delta_key(self.moves.index(move)))
                if delta is not None:  # a restart record loaded before attach
                    attempt.set_delta(delta, record=False)
        self.fixcm = fixcm
        self._replays = {}
        # The per-atom charges the force field had at attach; a block that moved the total charge puts them
        # back and records the new total as a host total (a fresh construction from the committed state).
        charges = getattr(field, "charges", None)
        self._charges_at_attach = None if charges is None else charges.detach().clone()
        engine.state.mc = self.state_dict()

    def due(self, step: int) -> tuple:
        """The moves due at ``step``: those whose ``every`` divides it (step > 0), in attached order."""
        if step <= 0:
            return ()
        return tuple(move for move in self.moves if step % move.every == 0)

    # ------------------------------------------------------------------ the block

    def run_block(self, engine: Any, step: int) -> Mapping[str, Any] | None:
        """One MC block at ``step`` (after its MD steps): the attempts of every due move and, after an
        acceptance, the resume. Returns the block record (``last_block``), or ``None`` when no move is due."""
        import torch

        if engine is not self.engine:
            raise ValueError("run_block needs the engine the driver is attached to (attach)")
        due = [index for index, move in enumerate(self.moves) if step > 0 and step % move.every == 0]
        if not due:
            return None
        engine.quiesce()
        state = engine.state
        temperature = float(engine.integrator.temperature_K)
        if not temperature > 0:
            raise ValueError(f"an MC block needs a positive target temperature, got {temperature!r} K")
        before = engine.overflowed() if hasattr(engine, "overflowed") else 0
        streams = self.streams
        streams.kT.fill_(K_B_EV_PER_K * temperature)
        streams.baseline.copy_(self.evaluator.baseline().reshape(1))
        baseline = float(streams.baseline[0])  # one sync at the block start, after the quiesce
        cuda = state.positions.device.type == "cuda"
        if cuda:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        before_counts = torch.stack([attempt.counters for attempt in self.attempts]).clone()
        self._block_trial_rebuilds = {index: getattr(attempt, "trial_rebuilds", 0)
                                      for index, attempt in enumerate(self.attempts)}
        indices, redraw_all = [], {}
        offsets = dict(self.offsets)
        for index in due:
            attempt, move = self.attempts[index], self.moves[index]
            streams.start(offsets)
            replay = self._replay(index)
            for _ in range(move.attempts):
                replay()
            if isinstance(move, Displace):
                self._finish_displacements(index)
            indices.append(streams.index.clone())
            # The host knows the fixed counts; the device index says how many attempts completed (read below).
            for stream in MC_STREAMS:
                offsets[stream] += move.attempts * attempt.counts[stream]
            redraw_all[index] = offsets["velocity"]  # where its block draw starts (used after an acceptance)
            offsets["velocity"] += attempt.block_velocity_count
        if cuda:
            stop.record()
        # The block's one host read: counters, attempt indices, the overflow word.
        after_counts = torch.stack([attempt.counters for attempt in self.attempts])
        read = torch.cat((after_counts.reshape(-1), torch.cat(indices), streams.baseline.view(torch.int64))).cpu()
        overflow = engine.overflowed() if hasattr(engine, "overflowed") else 0
        device_ms = float(start.elapsed_time(stop)) if cuda else None
        count = len(self.attempts)
        delta = (after_counts.cpu() - before_counts.cpu()).tolist()
        done = read[3 * count:3 * count + len(due)].tolist()
        energy = float(read[-1:].view(torch.float64)[0])
        if overflow & ~before:
            raise RuntimeError(f"an MC attempt flagged a capacity overflow (word {overflow}); swaps and transmutations "
                               "move no atom, so this is a bug")
        for position, index in enumerate(due):
            move = self.moves[index]
            if done[position] != move.attempts:
                raise RuntimeError(f"{move.move}: {done[position]} of {move.attempts} attempts completed")
        # Advance the streams by the completed attempts (equal to the host's fixed counts here).
        self.offsets = offsets
        moves_out, accepted_moves = [], []
        for index in due:
            move = self.moves[index]
            attempted, impossible, accepted = delta[index]
            for key, value in zip(COUNTER_KEYS, (attempted, impossible, accepted), strict=True):
                self.counters[move.move][key] += int(value)
            entry = {"move": move.move, "attempts": move.attempts, "attempted": int(attempted),
                     "impossible": int(impossible), "accepted": int(accepted)}
            if isinstance(move, Displace):  # 3-T10: trial builds, the δ used and, with adaptive, the next one
                attempt = self.attempts[index]
                entry.update(trial_rebuilds=attempt.trial_rebuilds - self._block_trial_rebuilds[index],
                             delta_A=attempt.delta_A)
                if move.adaptive:
                    entry["adapted_delta_A"] = attempt.adapt(int(accepted), int(attempted))
            moves_out.append(entry)
            if accepted:
                accepted_moves.append(index)
        if accepted_moves:
            self._resume(engine, {index: redraw_all[index] for index in accepted_moves
                                  if self.attempts[index].velocities == "redraw_all"})
        self.cycle_position = int(step)
        self.blocks += 1
        self.last_block = {"step": int(step), "moves": moves_out, "baseline_eV": baseline, "energy_eV": energy,
                           "accepted": sum(entry["accepted"] for entry in moves_out), "device_ms": device_ms,
                           "temperature_K": temperature}
        state.mc = self.state_dict()
        return self.last_block

    def _finish_displacements(self, index: int) -> None:
        """A displacement move's replays may have halted at an attempt that needs a trial build or more edges
        (``DisplaceAttempt``): run that attempt on the host with the same numbers, then replay the rest. One
        host read after the move's replays, plus one per halted attempt (3-T10 answer 1)."""
        attempt, move = self.attempts[index], self.moves[index]
        while True:
            done, halted = attempt.poll()
            if not halted:
                return
            attempt.host_attempt()
            replay = self._replay(index)  # the edges may have changed
            for _ in range(move.attempts - done - 1):
                replay()

    def _replay(self, index: int):
        """The attempt's replay; its first capture runs without effect on the state and the driver's tensors."""
        import torch

        engine, attempt = self.engine, self.attempts[index]
        edges = engine._active_edges() if hasattr(engine, "_active_edges") else None
        # The driver's id: another driver on the same engine has its own graphs (and its own tensors in them).
        key = (f"mc.{attempt.move.move}", self.id, index, self.evaluator.precision, edges)
        replay = self._replays.get(key)
        if replay is None:
            stage, trial = self.evaluator.stage() if getattr(self.evaluator, "graph", False) else (None, None)
            run = attempt.bound(stage, trial)
            replay = engine.captured(key, run) if hasattr(engine, "captured") else run
            self._replays[key] = replay
        graphs = getattr(engine, "_graphs", None)
        if getattr(engine, "capture", False) and graphs is not None and key not in graphs:
            # The capture's warm-ups restore the engine's registered state but not the driver's tensors, and
            # the replay that follows the capture is a real attempt: run it, then put everything back.
            saved = engine.registry.snapshot()
            mine = [*self.streams.tensors(), *(tensor for each in self.attempts for tensor in each.tensors())]
            kept = [tensor.detach().clone() for tensor in mine]
            replay()
            engine.registry.restore(saved)
            with torch.no_grad():
                for tensor, value in zip(mine, kept, strict=True):
                    tensor.copy_(value)
        return replay

    def _resume(self, engine: Any, redraw_all: Mapping[int, int]) -> None:
        """After a block with an acceptance (SPEC §8.5 step 6, ``RESUME_ORDER``): ``redraw_all`` momenta for the
        moves with an acceptance that chose it (``redraw_all``: move index -> the velocity offset of its block
        draw), the mass and momentum refreshes, the total charge as a host value, then a canonical point
        (forces and stress once, ``integrator.start``) and the caches derived from forces and stress."""
        import torch

        from fastmace.mc import momenta
        from fastmace.mc import streams as mc_streams

        state = engine.state
        with torch.no_grad():
            for offset in redraw_all.values():
                base = torch.tensor(offset, dtype=torch.int64, device=state.positions.device)
                normals = mc_streams.normals(self.seeds["velocity"], mc_streams.positions(base, 3 * state.natoms))
                state.velocities.copy_(momenta.redraw_all(state.masses, normals.view(-1, 3), self.streams.kT))
        self._record_total_charge(engine)
        engine.derived.refresh(("masses", "momenta"))
        engine.transactions_committed += 1
        engine.canonicalise(reevaluate=True)
        engine.derived.refresh(("forces", "stress"))

    def _record_total_charge(self, engine: Any) -> None:
        """A transmutation with formal charges moved the force field's device total charge through its per-atom
        charges. Make that a host total, as a fresh construction from the committed state has it: the per-atom
        charges as at attach, the total charge set (``set_total_charge``) and recorded in ``state.total_charge``."""
        if not any(attempt.write_charges and attempt.charges is not None and engine.state.charges is None
                   for attempt in self.attempts):
            return
        field = engine.force_field
        total = field.derived_buffers().get("total_charge")
        if total is None:
            return
        value = float(total.reshape(-1)[0])
        field.charges.copy_(self._charges_at_attach)
        field.set_total_charge(value)
        engine.derived.refresh(("charges",))
        engine.state.total_charge = round(value) if abs(value - round(value)) < 1e-9 else value

    # ------------------------------------------------------------------ restart

    def state_dict(self) -> dict[str, Any]:
        """The restart's ``mc`` record (§10.6), keys ``io.restart.MC_RECORD_KEYS``."""
        return {"moves": [move_record(move) for move in self.moves], "cycle_position": int(self.cycle_position),
                "counters": {move: dict(values) for move, values in self.counters.items()},
                "streams": {stream: {"seed": int(self.seeds[stream]), "offset": int(self.offsets[stream])}
                            for stream in MC_STREAMS},
                "delta_A": self._deltas()}

    def load_state_dict(self, record: Mapping[str, Any]) -> None:
        """Restore the move set's position, counters, stream positions and δ from a restart record."""
        moves = tuple(move_from_record(entry) for entry in record["moves"])
        if moves != self.moves:
            raise ValueError(f"the restart's MC moves {moves} differ from the driver's {self.moves}")
        for stream in MC_STREAMS:
            entry = record["streams"][stream]
            if int(entry["seed"]) != self.seeds[stream]:
                raise ValueError(f"the restart's {stream} stream seed differs from the driver's (another seed)")
            self.offsets[stream] = int(entry["offset"])
        self.counters = {move: {key: int(values[key]) for key in COUNTER_KEYS}
                         for move, values in record["counters"].items()}
        for move in self.moves:
            self.counters.setdefault(move.move, dict.fromkeys(COUNTER_KEYS, 0))
        self.cycle_position = int(record["cycle_position"])
        self._delta_A = {str(key): float(value) for key, value in record.get("delta_A", {}).items()}
        for index, attempt in enumerate(self.attempts):  # 3-T10: each displacement move's δ
            delta = self._delta_A.get(self._delta_key(index))
            if isinstance(self.moves[index], Displace) and delta is not None:
                attempt.set_delta(delta, record=False)
        if self.engine is not None:
            self.engine.state.mc = self.state_dict()

    @staticmethod
    def _delta_key(index: int) -> str:
        return f"displace:{index}"

    def _deltas(self) -> dict[str, float]:
        """The restart's ``delta_A``: each displacement move's current δ by ``displace:<index>``."""
        if not self.attempts:
            return dict(self._delta_A)
        return {self._delta_key(index): float(attempt.delta_A) for index, attempt in enumerate(self.attempts)
                if isinstance(self.moves[index], Displace)}

    def info(self) -> dict[str, Any]:
        """For provenance and the report: the moves, the evaluator, the decisions and the counters."""
        return {"moves": [move_record(move) for move in self.moves], "seed": self.seed,
                "velocities": [attempt.velocities for attempt in self.attempts],
                "evaluator": None if self.evaluator is None else self.evaluator.info(),
                "decisions": [list(decision) for decision in self.decisions], "counters": self.counters,
                "offsets": dict(self.offsets), "blocks": self.blocks, "last_block": self.last_block,
                "displacements": {self._delta_key(index): {"delta_A": attempt.delta_A,
                                                           "delta_history": list(attempt.delta_history),
                                                           "trial_rebuilds": attempt.trial_rebuilds}
                                  for index, attempt in enumerate(self.attempts)
                                  if isinstance(self.moves[index], Displace)}}


__all__ = ["COUNTER_KEYS", "MCDriver", "move_from_record", "move_record"]
