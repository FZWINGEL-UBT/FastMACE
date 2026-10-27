"""Shared fixtures, GPU skipping and the run's compiler caches for the whole suite.

Tests import helpers as ``from _support import ...``: pytest puts this directory on
``sys.path`` because it holds the root conftest and is not a package.

Test tiers (plan tasks 1b-T24, 1b-T27, ``_support.tiers``, ``docs/development.md``): fast
``-m "not slow and not acceptance"`` (workers), full ``-m "not acceptance"`` (merges) and
``-m acceptance`` (phase acceptance), and ``-m compat`` (the GPU compatibility check on every production
GPU type). An acceptance test must also be ``slow``
(``tests/cpu/test_tiers.py`` checks it), and a ``-m`` that selects acceptance tests stops the run
without the research engine, the hybrid Python or the models.
"""

import fastmace  # noqa: F401  (before torch: it switches PyTorch's file locks, SPEC §14)

# isort: split
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

SEED = 0

_GPU_DIR = Path(__file__).resolve().parent / "gpu"

_NO_CUDA = "CUDA is not available (torch.cuda.is_available() is False); GPU test skipped"

_COMPILE_CACHE_KEY = pytest.StashKey[str]()


def pytest_addoption(parser):
    parser.addoption("--fresh-compile-cache", action="store_true", default=False,
                     help="compile from scratch: a new Inductor/Triton cache for this run "
                          "(default: the FastMACE cache root's, shared by all runs)")


def _check_acceptance_environment(config):
    """A -m expression that selects acceptance tests needs their environment, or the run stops (1b-T27).

    In the controller, before collection: an error raised in xdist workers' collection crashes them.
    A run without -m keeps the acceptance tests' own skips.
    """
    from _support import tiers

    if tiers.selects_acceptance(config.getoption("markexpr", "")):
        problems = tiers.acceptance_environment_problems()
        if problems:
            raise pytest.UsageError("\n".join(["-m selects the acceptance tier (docs/development.md, "
                                               "'Test tiers'), which needs:", *problems]))


# Per-worker thread limits under xdist (suite-speed, 27 October 2026). Without them every worker's
# OpenMP pool and Inductor compile pool take all cores: eight workers on 24 threads ran the CPU
# suite in about 100 min at load ~130, against about 14 min with OMP_NUM_THREADS=2 (6-T16).
_XDIST_THREAD_DEFAULTS = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "1"}


def _xdist_defaults(config):
    """With ``-n``: thread limits for the workers and their child processes, and
    ``--maxschedchunk=1`` for the load scheduler (docs/development.md, 'CPU test scheduling').

    The controller runs before it starts the workers, so they inherit these variables before they
    import torch. Values set by the caller and an explicit ``--maxschedchunk`` are kept.
    """
    if not getattr(config.option, "numprocesses", None):
        return
    for name, value in _XDIST_THREAD_DEFAULTS.items():
        os.environ.setdefault(name, value)
    if getattr(config.option, "dist", None) == "load" and getattr(config.option, "maxschedchunk", 0) is None:
        config.option.maxschedchunk = 1


def pytest_configure(config):
    """One Inductor/Triton cache root for the whole run, inherited by every child process (1b-T24).

    The default is the FastMACE cache root's (``fastmace.kernels.triton.cache``, under
    ``FASTMACE_CACHE``), which production runs use too; a ``TORCHINDUCTOR_CACHE_DIR`` set by the
    caller is kept. ``--fresh-compile-cache`` gives the run a new empty root. Tests that need a cold
    cache of their own (the edge-bucket regression tests) still set one explicitly. Without this,
    torch puts its caches in /tmp, which the workstations forbid.
    """
    if hasattr(config, "workerinput"):  # an xdist worker inherits the controller's variables
        return
    _check_acceptance_environment(config)
    _xdist_defaults(config)
    from fastmace import _cache
    from fastmace.kernels.triton import cache

    if config.getoption("fresh_compile_cache"):
        runs = _cache.cache_root() / "test-runs"
        runs.mkdir(parents=True, exist_ok=True)
        info = cache.configure(tempfile.mkdtemp(prefix="compile-", dir=runs))
        kind = "fresh"
    elif os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        info, kind = {"inductor": os.environ["TORCHINDUCTOR_CACHE_DIR"],
                      "triton": os.environ.get("TRITON_CACHE_DIR", "(torch default)")}, "caller's"
    else:
        info, kind = cache.configure(), "shared"
    config.stash[_COMPILE_CACHE_KEY] = f"compile caches ({kind}): inductor {info['inductor']}, triton {info['triton']}"


def pytest_report_header(config):
    line = config.stash.get(_COMPILE_CACHE_KEY, None)
    return [line] if line else []


def pytest_collection_modifyitems(config, items):
    """Mark everything under tests/gpu as `gpu` and the compatibility tier's tests as `compat`
    (``_support.tiers.in_compat``), and skip `gpu` tests when CUDA is absent."""
    from _support import tiers

    cuda = None
    for item in items:
        if _GPU_DIR in item.path.parents:
            item.add_marker(pytest.mark.gpu)
            nodeid = item.path.relative_to(_GPU_DIR).as_posix() + item.nodeid[len(item.nodeid.split("::")[0]):]
            if tiers.in_compat(nodeid, {mark.name for mark in item.iter_markers()}):
                item.add_marker(pytest.mark.compat)
        if item.get_closest_marker("gpu") is None:
            continue
        if cuda is None:
            cuda = torch.cuda.is_available()
        if not cuda:
            item.add_marker(pytest.mark.skip(reason=_NO_CUDA))
    if hasattr(config, "workerinput"):
        _longest_first(items)


_DURATIONS = Path(__file__).resolve().parent / "_support" / "data" / "cpu_durations.json"


def _longest_first(items):
    """In xdist workers: the test functions recorded as slow first, longest first (tools/test_order.py).

    With collection order the 50-75 s tests started in the last minutes of the run and left the other
    workers idle (suite-speed, 27 October 2026). Every worker sorts the same way, so their collections
    stay identical; unrecorded tests keep their order after the recorded ones.
    """
    import json

    try:
        seconds = json.loads(_DURATIONS.read_text())
    except (OSError, ValueError):
        return
    rank = {item: seconds.get(item.nodeid.split("[", 1)[0], 0.0) for item in items}
    items.sort(key=lambda item: -rank[item])


@pytest.fixture
def seed():
    """Seed torch and NumPy's global generators; returns the seed."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return SEED


@pytest.fixture
def rng(seed):
    """A fresh seeded NumPy generator."""
    return np.random.default_rng(seed)


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """MACE's calculator sets torch's global default dtype; keep it from leaking into later tests."""
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


# Process-wide patches that cannot be undone in a running process (fastmace._patches.restore refuses).
_PERMANENT_PATCHES = {"oeq_runtime", "polar_longrange"}


@pytest.fixture(autouse=True)
def _restore_process_patches():
    """Undo the process-wide MACE patches a test applied, so they do not leak into later tests.

    ``loading.load(backend="cueq")`` turns MACE's fused cuEq convolution on for the whole process,
    and later deep copies of fused tiny models then break (1a-T07/T08 reports). In the full GPU run
    of 18 October 2026, the MH-1 GPU tests failed only after other tests; alone they passed.
    Patches applied before the test (for example by a module-scoped fixture) are left alone.
    """
    from fastmace import _patches

    before = {record.name for record in _patches.applied()}
    yield
    for record in reversed(_patches.applied()):
        if record.name not in before and record.name not in _PERMANENT_PATCHES:
            _patches.restore(record.name)
