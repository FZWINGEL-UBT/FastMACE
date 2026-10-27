"""The test suite's cache of cuEq's naive CPU kernels (tests/_support/naive_cueq_cache.py, suite-speed)."""

import pytest
import torch

pytest.importorskip("cuequivariance_torch")

from cuequivariance_torch.primitives import segmented_polynomial_naive as naive  # noqa: E402

from _support.mc_fixtures import tiny_loaded  # noqa: E402
from _support.structures import nacl  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(naive._tensor_product_fx, "cache"),
                                reason="the cache is switched off (FASTMACE_TEST_NAIVE_CUEQ_CACHE=0)")


def _tensors(module):
    return [t for _, t in module.named_buffers()] + [p for _, p in module.named_parameters()]


def test_a_cached_kernel_is_a_private_copy_with_the_same_graph_and_constants():
    import cuequivariance as cue

    descriptor = cue.descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "2x0e + 1x1o"), cue.Irreps("O3", "1x0e + 1x1o"), cue.Irreps("O3", "2x0e + 1x1o")
    ).polynomial.operations[0][1]
    fresh = naive._tensor_product_fx.uncached(descriptor, None, torch.float64, True)
    first = naive._tensor_product_fx(descriptor, None, torch.float64, True)
    second = naive._tensor_product_fx(descriptor, None, torch.float64, True)
    assert first is not second
    assert first.code == second.code == fresh.code
    for a, b, c in zip(_tensors(first), _tensors(second), _tensors(fresh), strict=True):
        assert a.data_ptr() != b.data_ptr()
        assert torch.equal(a, c) and torch.equal(b, c) and a.dtype == c.dtype
    inputs = [torch.randn(5, operand.size, dtype=torch.float64) for operand in descriptor.operands[:-1]]
    assert torch.equal(first(*inputs), fresh(*inputs))


def test_another_math_dtype_is_another_entry():
    import cuequivariance as cue

    descriptor = cue.descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "1x0e + 1x1o"), cue.Irreps("O3", "1x0e"), cue.Irreps("O3", "1x0e + 1x1o")
    ).polynomial.operations[0][1]
    a = naive._tensor_product_fx(descriptor, None, torch.float64, True)
    b = naive._tensor_product_fx(descriptor, None, torch.float32, True)
    assert {t.dtype for t in _tensors(a)} == {torch.float64}
    assert {t.dtype for t in _tensors(b)} == {torch.float32}


def test_a_converted_model_with_the_cache_equals_one_without(tmp_path, monkeypatch):
    """A whole cuEq CPU conversion through the cache gives bitwise the same energy and forces."""
    from fastmace.model import loading

    pristine = tiny_loaded(tmp_path, [11, 17])
    options = dict(backend="cueq", dtype="float64", device="cpu", sha256=pristine.sha256, trusted=True)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    warm = loading.load(str(pristine.path), **options)  # the first evaluation of a process differs (0-T04)
    warm.calculator.get_potential_energy(atoms.copy())
    results = []
    for build in (naive._tensor_product_fx, naive._tensor_product_fx.uncached):
        monkeypatch.setattr(naive, "_tensor_product_fx", build)
        model = loading.load(str(pristine.path), **options)
        copy = atoms.copy()
        copy.calc = model.calculator
        results.append((copy.get_potential_energy(), copy.get_forces()))
    assert results[0][0] == results[1][0]
    assert (results[0][1] == results[1][1]).all()
