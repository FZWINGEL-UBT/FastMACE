"""The GPU suite's tiers (plan tasks 1b-T24, 1b-T27): which marks put a test in which tier.

- fast: ``-m "not slow and not acceptance"`` (workers, every change);
- full: ``-m "not acceptance"`` (the conductor's merge tier);
- acceptance: ``-m acceptance`` (phase acceptance, and merges that touch ``model/``, ``rewrites/``
  or ``kernels/``).

An acceptance test also carries ``slow``, so that the older fast command (``-m "not slow"``) leaves it
out too. The acceptance tier needs the research engine, the hybrid interpreter and the released
models; without them it fails instead of skipping its tests.
"""

from __future__ import annotations

import os
from pathlib import Path

FAST = "not slow and not acceptance"
FULL = "not acceptance"
ACCEPTANCE = "acceptance"

# Variable: what the acceptance tier needs it for.
ACCEPTANCE_ENVIRONMENT = {
    "FASTMACE_BASELINE": "the research-engine export (1a-T18) for the A/B comparisons",
    "FASTMACE_HYBRID_PYTHON": "the Python of a hybrid (OpenEquivariance) environment",
    "FASTMACE_MODELS": "the directory of released checkpoints (MH-1, MP-0, POLAR-1-S, ...)",
}


def tier(marks) -> str:
    """'fast', 'full' or 'acceptance' for a test with these mark names."""
    marks = set(marks)
    if "acceptance" in marks:
        return "acceptance"
    return "full" if "slow" in marks else "fast"


def marker_problems(tests) -> list[str]:
    """Tests (``{node id: mark names}``) whose marks break the tier rules: acceptance without slow."""
    return [f"{node}: marked acceptance but not slow; add @pytest.mark.slow"
            for node, marks in tests.items() if "acceptance" in set(marks) and "slow" not in set(marks)]


def acceptance_environment_problems(environ=None) -> list[str]:
    """What is missing for the acceptance tier: each variable unset or naming nothing that exists."""
    environ = os.environ if environ is None else environ
    problems = []
    for name, purpose in ACCEPTANCE_ENVIRONMENT.items():
        value = environ.get(name)
        if not value:
            problems.append(f"{name} is not set ({purpose})")
        elif not Path(os.path.expanduser(value)).exists():
            problems.append(f"{name}={value} does not exist ({purpose})")
    return problems
