"""T09's ``MCDriver`` for 3-T15's front-end tests: a host stand-in (``install``) and a spy on the real one (``spy``).

The front ends need the stand-in only where the real driver cannot run: a hand-made restart record
whose stream seeds are not the simulation's (``test_restart_v3.py``).

The front ends (``Simulation.run(mc=…)``, ``fm.md``, the CLI) only attach, call ``due`` and
``run_block``, read ``counters``, ``offsets`` and ``last_block``, and move ``state_dict`` through the
restart; they never look inside a block. This driver does exactly that much on the host: it changes
no state, counts every attempt as attempted and every ``accept_every``-th as accepted, and advances
its stream offsets by one per attempt. ``install(monkeypatch)`` puts it in place of the real one and
returns the list of drivers made, oldest first.
"""

from __future__ import annotations

import dataclasses

from fastmace.config import MC_STREAMS


class FakeDriver:
    accept_every = 3
    made: list = None  # set by install
    flag = None  # a staticmethod () -> bool set by a test: Simulation._mc_block_active inside run_block

    def __init__(self, moves, *, seed, config=None):
        self.moves = tuple(moves)
        self.seed = seed
        self.config = config
        self.engine = None
        self.blocks = []  # (step, sim flag) per block
        self.loaded = []  # load_state_dict records
        self.cycle_position = 0
        self.counters = {move.move: {"attempted": 0, "impossible": 0, "accepted": 0} for move in self.moves}
        self.offsets = {stream: 0 for stream in MC_STREAMS}
        self.last_block = None
        self.evaluator = None
        FakeDriver.made.append(self)

    def attach(self, engine, *, evaluator=None, max_displacement_A=None):
        self.engine = engine
        self.max_displacement_A = max_displacement_A
        self.evaluator = evaluator
        engine.state.mc = self.state_dict()

    def due(self, step):
        return [move for move in self.moves if step > 0 and step % move.every == 0]

    def run_block(self, engine, step):
        due = self.due(step)
        if not due:
            return None
        moves = []
        for move in due:
            counter = self.counters.setdefault(move.move, {"attempted": 0, "impossible": 0, "accepted": 0})
            accepted = sum(1 for k in range(move.attempts)
                           if (counter["attempted"] + k + 1) % self.accept_every == 0)
            counter["attempted"] += move.attempts
            counter["accepted"] += accepted
            for stream in self.offsets:
                self.offsets[stream] += move.attempts
            moves.append({"move": move.move, "attempts": move.attempts, "attempted": move.attempts,
                          "impossible": 0, "accepted": accepted})
        self.cycle_position = step
        self.blocks.append((step, None if self.flag is None else self.flag()))
        self.last_block = {"step": step, "moves": moves, "baseline_eV": 0.0, "energy_eV": 0.0,
                           "accepted": sum(m["accepted"] for m in moves), "device_ms": 2.0}
        engine.state.mc = self.state_dict()
        return self.last_block

    def state_dict(self):
        return {"moves": [{"move": move.move, **_plain(dataclasses.asdict(move))} for move in self.moves],
                "cycle_position": self.cycle_position,
                "counters": {move: dict(counter) for move, counter in self.counters.items()},
                "streams": {stream: {"seed": int(self.seed) % 2**62, "offset": offset}
                            for stream, offset in self.offsets.items()},
                "delta_A": {}}

    def load_state_dict(self, record, **options):
        self.loaded.append(record)
        self.cycle_position = int(record["cycle_position"])
        for move, counter in record["counters"].items():
            self.counters[move] = dict(counter)
        self.offsets = {stream: int(position["offset"]) for stream, position in record["streams"].items()}
        if self.engine is not None:
            self.engine.state.mc = self.state_dict()


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def install(monkeypatch):
    from fastmace.mc import driver

    made = []
    monkeypatch.setattr(FakeDriver, "made", made)
    monkeypatch.setattr(driver, "MCDriver", FakeDriver)
    return made


# --------------------------------------------------------------------------- the real driver, observed

flag = None  # () -> bool set by a test: Simulation._mc_block_active, read as each spied block starts


def spy(monkeypatch):
    """T09's real ``MCDriver``, recording each block's step in ``spied`` (with ``flag()``) and each ``load_state_dict``
    record; returns the list of drivers made, oldest first. For the CPU, where the real driver runs."""
    from fastmace.mc import driver

    real = driver.MCDriver
    made = []

    class Spy(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.spied = []
            self.loaded = []
            made.append(self)

        def run_block(self, engine, step):
            inside = None if flag is None else flag()
            block = super().run_block(engine, step)
            if block is not None:
                self.spied.append((step, inside))
            return block

        def load_state_dict(self, record, **options):
            self.loaded.append(record)
            return super().load_state_dict(record, **options)

    monkeypatch.setattr(driver, "MCDriver", Spy)
    return made
