"""The subprocess runner (SPEC §8.7, §10.6): a plan that a process constraint forbids here runs in a child.

Plan task 2-T01 fixes the contract; plan task 2-T07 implements it. Where §8.7 forbids a plan in this
process (a second backend family after a compiled model; ``reproducible=True`` after CUDA was
initialised), the planner marks it ``Plan.runner == "subprocess"``, and the runner:

1. writes a canonical restart of the last committed step into the hand-off directory
   (``<output>/.fastmace-handoff/``, or under ``FASTMACE_CACHE`` without output);
2. starts ``python -m fastmace.engine.subprocess <directory>`` with ``RunnerConfig``'s interpreter
   and environment;
3. the child resumes with ``allow_plan_change=True`` and runs the requested steps; its output
   chunks continue the parent's;
4. the parent reads the child's restart and metadata back, so ``sim.committed()``, ``sim.plan`` and
   the report reflect the child.

State crosses only through the restart write, never as pickled live objects. A child that fails,
is killed or stalls (``RunnerConfig.stall_timeout_s``) raises ``SubprocessRunnerError`` with its
message, never a hang. The hand-off directory is deleted on a clean ``close()`` and kept, with its
path recorded, after an error.

This module imports neither torch nor MACE: the parent may not be allowed to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmace.config import RunnerConfig

TASK = "plan task 2-T07"
HANDOFF_NAME = ".fastmace-handoff"  # the hand-off directory under the run's output directory


class SubprocessRunner:
    """Runs a ``Simulation``'s next ``run()`` call in a child process (SPEC §8.7).

    ``config``: a ``RunnerConfig``; ``None`` takes its defaults (``sys.executable``, the parent's
    environment, no total limit, a 30-minute stall limit).
    """

    def __init__(self, config: RunnerConfig | None = None):
        from fastmace.config import RunnerConfig

        self.config = RunnerConfig() if config is None else config

    def run(self, simulation: Any, *, steps: int | None = None, schedule: Any = None) -> None:
        """Hand ``simulation``'s last committed step to a child, run ``steps`` (or ``schedule``) there,
        and take the child's state, plan and records back. ``SubprocessRunnerError`` on any child
        failure."""
        raise NotImplementedError(f"the subprocess runner is {TASK}")


def main(argv: list[str] | None = None) -> int:
    """The child's entry point: ``python -m fastmace.engine.subprocess <directory>``."""
    raise NotImplementedError(f"the subprocess runner's child is {TASK}")


if __name__ == "__main__":
    raise SystemExit(main())
