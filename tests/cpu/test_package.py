"""Package skeleton: version, subpackages of SPEC §5 and the error base class."""

import importlib

import pytest

SUBPACKAGES = (
    "model",
    "rewrites",
    "kernels",
    "neighbors",
    "integrators",
    "mc",
    "optimize",
    "engine",
    "planner",
    "reference",
    "longrange",
    "io",
    "protocols",
    "ase",
    "cli",
    "bench",
)


def test_version():
    import fastmace

    assert fastmace.__version__ == "0.0.1.dev0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name):
    module = importlib.import_module(f"fastmace.{name}")
    assert module.__doc__ and module.__doc__.strip()
    assert hasattr(module, "__path__"), f"fastmace.{name} must be a package"


def test_error_base_class():
    from fastmace.errors import FastMACEError

    assert issubclass(FastMACEError, Exception)
