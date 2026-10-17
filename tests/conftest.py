"""Shared fixtures and GPU skipping for the whole suite.

Tests import helpers as ``from _support import ...``: pytest puts this directory on
``sys.path`` because it holds the root conftest and is not a package.
"""

import fastmace  # noqa: F401  (before torch: it switches PyTorch's file locks, SPEC §14)

# isort: split
from pathlib import Path

import numpy as np
import pytest
import torch

SEED = 0

_GPU_DIR = Path(__file__).resolve().parent / "gpu"

_NO_CUDA = "CUDA is not available (torch.cuda.is_available() is False); GPU test skipped"


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
