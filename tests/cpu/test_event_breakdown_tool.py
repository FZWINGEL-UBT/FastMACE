"""tools/bench/event_breakdown.py without a GPU: its timers and the rest lines."""

import importlib.util
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "tools" / "bench"


def _load():
    spec = importlib.util.spec_from_file_location("_tool_event_breakdown", BENCH / "event_breakdown.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Engine:
    def __init__(self, clock):
        self.clock = clock

    def flush(self):
        self.clock.now += 0.002

    def canonicalise(self):
        self.flush()
        self.clock.now += 0.005


def test_timers_are_inclusive_sync_around_each_call_and_off_until_enabled():
    clock, syncs = Clock(), []
    timers = tool.Timers(sync=lambda: syncs.append(clock.now), clock=clock)
    engine = Engine(clock)
    undo = [timers.patch(engine, "flush"), timers.patch(engine, "canonicalise")]
    engine.canonicalise()
    assert timers.samples == {} and syncs == []
    timers.enabled = True
    engine.canonicalise()
    engine.canonicalise()
    assert timers.samples["canonicalise"] == pytest.approx([7.0, 7.0])
    assert timers.samples["flush"] == pytest.approx([2.0, 2.0])
    assert len(syncs) == 8  # before and after each of the four timed calls
    per_event, calls = timers.table(events=2)
    assert per_event["canonicalise"] == pytest.approx(7.0)
    assert per_event["canonicalise rest"] == pytest.approx(5.0)
    assert calls["flush"] == 1.0
    for restore in reversed(undo):
        restore()
    timers.samples.clear()
    engine.canonicalise()
    assert timers.samples == {}


def test_a_wrapped_method_on_a_class_still_binds_self():
    class Transaction:
        def quiesce(self):
            return self

    timers = tool.Timers()
    undo = timers.patch(Transaction, "quiesce", "Transaction.quiesce")
    timers.enabled = True
    item = Transaction()
    assert item.quiesce() is item and len(timers.samples["Transaction.quiesce"]) == 1
    undo()
    assert Transaction.quiesce.__name__ == "quiesce" and not hasattr(Transaction.quiesce, "__wrapped__")
