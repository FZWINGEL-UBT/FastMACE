"""The runtime monitor through the subprocess runner (plan task 5-T06; SPEC §8.7, §9.6).

Each child's monitor exports the segments of its call; the parent absorbs them, so ``counters.segments``
holds every call's segments, whether one worker serves every call or each call starts a fresh child
(whose monitor starts empty, as on a resume).
"""

from __future__ import annotations

import pytest
import test_subprocess_runner as runner_test

from fastmace.config import RunnerConfig
from fastmace.io.output import Output

loaded = runner_test.loaded
cpu_planner = runner_test.cpu_planner  # autouse there and here
runner_plans_later = runner_test.runner_plans_later


@pytest.mark.parametrize("persistent", [True, False], ids=["worker", "fresh-children"])
def test_the_parent_gathers_every_calls_segments(loaded, tmp_path, runner_plans_later, persistent):
    runner_plans_later()
    output = Output(tmp_path / "child", thermo_every=1, trajectory_every=None, restart_every=None)
    sim = runner_test._simulation(loaded, output=output, runner=RunnerConfig(persistent=persistent))
    assert sim.plan.runner == "subprocess"
    sim.run(steps=3)
    sim.run(steps=2)
    segments = sim._monitor.segments
    assert [(s.first_step, s.last_step, s.reason, s.canonical) for s in segments] == [
        (0, 3, "run-end", False), (3, 5, "run-end", False)]
    assert sim._monitor.totals().last_step == 5
    sim.close()
    metadata = runner_test._metadata(tmp_path / "child")
    assert [entry["last_step"] for entry in metadata["counters"]["segments"]] == [3, 5]
    assert metadata["counters"]["neighbour_lists"]["mace"]["rebuilds"] == sum(
        s.lists["mace"]["rebuilds"] for s in segments)
