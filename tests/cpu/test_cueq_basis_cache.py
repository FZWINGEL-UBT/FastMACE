"""A float64 model loaded after a float32 cuEq conversion keeps every bit (plan task 2-T14).

MACE 0.3.16's ``O3_e3nn.clebsch_gordan`` computes Clebsch-Gordan coefficients in torch's default
dtype, and cuequivariance 0.12.0 keeps what it builds from them: the basis products in a module
dict (``reduced_tensor_product._cache_reduce_basis_product``) and the reduced bases and
descriptors in ``functools`` caches, all keyed without that dtype. A float32 conversion therefore
left float32-rounded symmetric-contraction coefficients for every later float64 conversion in the
process (up to 1.3e-8 in the basis). ``fastmace.model.loading`` clears these caches around each
conversion, so a conversion depends only on its own dtype, as the first one of a process does.
"""

import importlib.util
import sys
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="needs cuequivariance_torch")

CACHE_MODULE = "cuequivariance.group_theory.irreps_array.reduced_tensor_product"


def _basis_cache():
    module = pytest.importorskip(CACHE_MODULE)
    if not hasattr(module, "_cache_reduce_basis_product"):
        pytest.skip("this cuequivariance has no _cache_reduce_basis_product")
    return module._cache_reduce_basis_product


def _fresh():
    """cuequivariance's group-theory caches as in a fresh process (independent of the library's own clearing)."""
    _basis_cache().clear()
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith("cuequivariance.group_theory"):
            for value in list(vars(module).values()):
                if callable(getattr(value, "cache_clear", None)):
                    value.cache_clear()


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from _support.tiny_models import FAMILIES, tiny_mace_model

    dtype = torch.get_default_dtype()
    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    torch.set_default_dtype(dtype)
    return path


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _load(path, backend, dtype):
    from fastmace.model import loading

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", loading.ModelLicenceWarning)
        loaded = loading.load(str(path), backend=backend, dtype=dtype, device="cpu", trusted=True,
                              allow_naive_cueq=True)
    return _state(loaded.calculator.models[0])


def _pristine(path):
    from fastmace.model import loading

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", loading.ModelLicenceWarning)
        return _state(loading.load_pristine(str(path), trusted=True).model)


def _assert_bitwise(got, expected):
    assert got.keys() == expected.keys()
    differ = [name for name in expected if not torch.equal(got[name], expected[name])]
    assert not differ, f"{len(differ)} tensors differ, e.g. {differ[:3]}"


def test_float64_cueq_load_after_a_float32_one_is_bitwise_fresh(tiny_checkpoint):
    _fresh()
    fresh = _load(tiny_checkpoint, "cueq", "float64")
    _fresh()
    _load(tiny_checkpoint, "cueq", "float32")
    _assert_bitwise(_load(tiny_checkpoint, "cueq", "float64"), fresh)


def test_pristine_load_after_a_float32_conversion_is_bitwise_fresh(tiny_checkpoint):
    _fresh()
    fresh = _pristine(tiny_checkpoint)
    _fresh()
    _load(tiny_checkpoint, "cueq", "float32")
    _assert_bitwise(_pristine(tiny_checkpoint), fresh)


TRIGGER_PAIR = ("tests/cpu/test_acceptance_1b.py::test_two_simulation_sides_on_the_cpu",
                "tests/cpu/test_forces.py::test_polar_force_field_enables_its_own_copy")


def test_the_1b_t27_trigger_pair_passes_in_a_fresh_process(tmp_path):
    """A CPU Simulation (float32 cuEq plan) and then the polar test at 1e-12, in one fresh pytest process.

    The order of cache fills matters, so only a fresh process reproduces the state the full suite met.
    """
    import os
    import subprocess

    from _support.tiny_models import polar_available

    if not polar_available():
        pytest.skip("PolarMACE needs graph_longrange")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-q",
                             "-W", "ignore::DeprecationWarning", f"--basetemp={tmp_path / 'pair'}", *TRIGGER_PAIR],
                            cwd=root, capture_output=True, text=True, timeout=600, check=False)
    tail = (result.stdout + result.stderr)[-3000:]
    assert result.returncode == 0, tail
    assert "2 passed" in result.stdout, tail


def test_naive_cueq_converts_from_fresh_caches():
    """MACE's own converter inside naive_cueq (as the tests' polar calculators use it) starts fresh too."""
    from fastmace.model import compile as compiling

    cache = _basis_cache()
    cache["stale"] = None
    with compiling.naive_cueq():
        assert "stale" not in cache
        cache["inside"] = None
    assert len(cache) == 0


def test_a_missing_cache_is_skipped(monkeypatch):
    from fastmace.model import loading

    module = pytest.importorskip(CACHE_MODULE)
    monkeypatch.delattr(module, "_cache_reduce_basis_product")
    with loading._fresh_cueq_basis_cache():
        pass


def test_the_caches_hold_float32_rounded_bases_without_clearing():
    """The upstream behaviour the clearing works around; if this fails, cuequivariance or MACE changed."""
    import cuequivariance as cue
    from mace.tools.cg import O3_e3nn

    irreps_in, irreps_out = cue.Irreps(O3_e3nn, "0e+1o+2e"), cue.Irreps(O3_e3nn, "0e+1o")

    def basis(dtype):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            return np.asarray(cue.reduced_symmetric_tensor_product_basis(irreps_in, 2, keep_ir=irreps_out,
                                                                         layout=cue.ir_mul).array)
        finally:
            torch.set_default_dtype(previous)

    _fresh()
    exact = basis(torch.float64)
    _fresh()
    basis(torch.float32)
    stale = basis(torch.float64)
    _fresh()
    assert 1e-10 < np.abs(stale - exact).max() < 1e-6
