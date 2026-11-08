"""h200-fastpath's glue switches on the CPU: ``divided_blocks`` (``LINEAR_DIV_SPLIT``) and ``cueq_projection``
(``CUEQ_PROJECTED``)."""
import pytest
import torch

from fastmace.kernels.cutlass import linear_div
from fastmace.kernels.triton.divided_blocks import divided_blocks
from fastmace.rewrites import cueq_projection


def test_switch_defaults():
    assert linear_div.LINEAR_DIV_SPLIT is None and not linear_div.LINEAR_DIV_SPLIT_GPUS
    assert not linear_div.linear_div_split_default("NVIDIA H200")
    assert cueq_projection.CUEQ_PROJECTED is None and cueq_projection.cueq_projected_default("NVIDIA H200")
    assert not cueq_projection.cueq_projected_default("NVIDIA A100-SXM4-80GB")


def test_divided_blocks_equal_the_sliced_division():
    g = torch.Generator().manual_seed(0)
    k, atoms = 4, 5
    widths = [k * (2 * l + 1) for l in range(4)]  # noqa: E741
    grad = torch.randn(atoms, sum(widths), generator=g)
    other = torch.rand(atoms, 1, generator=g) + 1
    blocks = divided_blocks(grad, other, widths)
    offset = 0
    for block, width in zip(blocks, widths):
        assert block.is_contiguous() and block.shape == (atoms, width)
        assert torch.equal(block, (grad / other)[:, offset:offset + width])
        offset += width


def _contraction():
    cue = pytest.importorskip("cuequivariance")
    cuet = pytest.importorskip("cuequivariance_torch")
    irreps_in = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_out = cue.Irreps("O3", "4x0e")
    torch.manual_seed(0)
    try:
        module = cuet.SymmetricContraction(irreps_in, irreps_out, contraction_degree=2, num_elements=3,
                                           layout=cue.ir_mul, original_mace=True, dtype=torch.float64,
                                           method="naive")
    except TypeError:
        pytest.skip("this cuequivariance_torch has another SymmetricContraction signature")
    if getattr(module, "projection", None) is None:
        pytest.skip("no projection in this configuration")
    return module


def test_kept_projection_gives_the_same_output():
    module = _contraction()
    x = torch.randn(7, 16, dtype=torch.float64)
    indices = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    with torch.no_grad():
        before = module(x, indices)
        assert cueq_projection.applies(module)
        cueq_projection.keep_projected(module)
        assert type(module).__name__ == "SymmetricContraction" and not cueq_projection.applies(module)
        assert torch.equal(module(x, indices), before)
    assert "projected_weight" not in module.state_dict()
