"""The GPU suite's tiers (plan tasks 1b-T24, 1b-T27): which marks put a test in which tier.

- fast: ``-m "not slow and not acceptance"`` (workers, every change);
- full: ``-m "not acceptance"`` (the conductor's merge tier);
- acceptance: ``-m acceptance`` (phase acceptance, and merges that touch ``model/``, ``rewrites/``
  or ``kernels/``);
- compat: ``-m compat``, the GPU compatibility check (SPEC decision 39, amended 23 October 2026; plan task
  4-T13), about 30-60 minutes on every production GPU type: the fast GPU tier plus ``COMPAT_TESTS``, the slow
  tests that check what differs between architectures, plus ``tests/gpu/test_compat_gpu.py``. The root
  conftest marks them; ``tools/acceptance/compat.py`` runs the tier and records the GPU and provider.

An acceptance test also carries ``slow``, so that the older fast command (``-m "not slow"``) leaves it
out too. A ``-m`` that selects acceptance tests needs the research engine, the hybrid interpreter
and the released models; without them the run stops instead of skipping those tests.
"""

from __future__ import annotations

import os
from pathlib import Path

FAST = "not slow and not acceptance"
FULL = "not acceptance"
ACCEPTANCE = "acceptance"
COMPAT = "compat"

# The slow tests of the compatibility tier (node ids under tests/gpu; an id without "[...]" takes every
# parametrisation). tests/cpu/test_tiers.py checks that each one is still collected.
COMPAT_TESTS = (
    # kernels compile and attach: every rewrite, OEq, Triton, the fused cuEq adapter (no naive fallback)
    "test_loading_gpu.py::test_mh1_plain_cueq_float64",
    "test_loading_gpu.py::test_mh1_hybrid_float32_in_a_fresh_process",
    "test_rewrites_mh1_gpu.py::test_mh1_hybrid_float32_with_streaming",
    "test_rewrites_mh1_gpu.py::test_mh1_float64_package_agrees_with_the_unrewritten_model",
    "test_rewrites_generic_gpu.py::test_stacked_generic_rewrites_hybrid_in_a_fresh_process[mp0-medium-float32]",
    "test_combo_kernels_cuda.py::test_combo_kernels_change_no_value",
    # the accuracy gate on MH-1 (NVT and full-cell NPT), MP-0 and POLAR-1-S
    "test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[mh1-r2scan-nvt]",
    "test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[mh1-r2scan-npt-full]",
    "test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[mp0-medium]",
    "test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[polar1-s]",
    # CUDA-graph replay against eager (the kernel counts: test_compat_gpu.py)
    "test_eager_gpu.py::test_mh1_hybrid_eager_equals_graph_within_float32_noise",
    # reproducible mode bitwise, and a restart round trip
    "test_reproducible_gpu.py::test_repeated_and_interrupted_runs_are_bitwise_equal[NaCl-float32]",
    "test_canonical_restart_gpu.py::test_mh1_plain_cueq_float64_resumes_within_1e_12",
)


def in_compat(nodeid: str, marks) -> bool:
    """Whether a test under tests/gpu (``nodeid`` relative to it) is in the compatibility tier: the fast tier,
    the listed slow tests, and tests marked ``compat`` themselves."""
    marks = set(marks)
    if "compat" in marks or tier(marks) == "fast":
        return True
    return any(nodeid == entry or ("[" not in entry and nodeid.startswith(entry + "["))
               for entry in COMPAT_TESTS)

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


def selects_acceptance(markexpr: str) -> bool:
    """True when ``-m markexpr`` (non-empty) selects an acceptance test (marks: acceptance, slow, gpu)."""
    if not markexpr:
        return False
    from _pytest.mark.expression import Expression

    marks = {"acceptance", "slow", "gpu"}
    return Expression.compile(markexpr).evaluate(lambda name, **kwargs: name in marks)


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
