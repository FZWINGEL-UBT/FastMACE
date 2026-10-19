"""Shared fixtures, GPU skipping and the run's compiler caches for the whole suite.

Tests import helpers as ``from _support import ...``: pytest puts this directory on
``sys.path`` because it holds the root conftest and is not a package.

Test tiers (plan task 1b-T24, ``docs/development.md``): ``pytest tests/gpu -m "not slow"``
is the worker tier, the full ``pytest tests/gpu`` the merge and acceptance tier.
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
    """Mark everything under tests/gpu as `gpu`, and skip `gpu` tests when CUDA is absent."""
    cuda = None
    for item in items:
        if _GPU_DIR in item.path.parents:
            item.add_marker(pytest.mark.gpu)
        if item.get_closest_marker("gpu") is None:
            continue
        if cuda is None:
            cuda = torch.cuda.is_available()
        if not cuda:
            item.add_marker(pytest.mark.skip(reason=_NO_CUDA))


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
