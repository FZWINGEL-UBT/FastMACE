"""kernels.triton.compose_skip on the CPU: the reassociation's composed products against the reassociation followed
by the linear and the species skip, in float64."""
from types import SimpleNamespace

import pytest
import torch

from fastmace.kernels.triton import compose_skip
from fastmace.rewrites.generic.modules import SpeciesSkip, compose_skip_products
from fastmace.rewrites.linear.modules import BlockLinear
from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

LS = (0, 1, 2)


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    species, c, width, nodes = 3, 4, 2, 7
    dim = sum(2 * l + 1 for l in LS)  # noqa: E741
    columns = torch.tensor([0, 2, 3])
    attrs = torch.zeros(nodes, 5, dtype=torch.float64)
    attrs[torch.arange(nodes), columns[torch.tensor([0, 1, 2, 2, 1, 0, 2])]] = 1
    x = torch.randn(species, c, generator=g, dtype=torch.float64)
    reassociation = FirstInteractionReassociation(columns, c, LS, (1.0, 0.7, 0.4), ir_mul=True, species_feats=x)
    weight = torch.randn(len(LS) * c, width, generator=g, dtype=torch.float64)
    segments, linear_segments, offset = [], [], 0
    for l in LS:  # noqa: E741
        segments.append((offset, c, c * (2 * l + 1)))
        linear_segments.append((offset, c, 2 * l + 1))
        offset += c * (2 * l + 1)
    mats = [torch.randn(species, c, c, generator=g, dtype=torch.float64) for _ in LS]
    skip = SpeciesSkip(columns, segments, segments, [(k, k, m) for k, m in enumerate(mats)], "ir_mul")
    linear = BlockLinear(torch.nn.Identity(), torch.nn.Identity(), linear_segments, linear_segments,
                         [(k, k, torch.randn(c, c, generator=g, dtype=torch.float64)) for k in range(len(LS))])
    sums = torch.randn(nodes * species, dim * width, generator=g, dtype=torch.float64)
    edge_index = torch.zeros(2, 1, dtype=torch.int64)
    edge_attrs = torch.zeros(1, dim, dtype=torch.float64)

    def message():
        return reassociation(None, attrs, edge_attrs, None, edge_index, weight,
                             summed_fn=lambda key, slots: (sums, width))
    return reassociation, linear, skip, attrs, message


@pytest.mark.parametrize("with_linear", [False, True])
def test_composed_products_equal_the_projection_the_linear_then_the_skip(with_linear):
    reassociation, linear, skip, attrs, message = _case()
    if not with_linear:
        linear = torch.nn.Identity()
    expected = skip(linear(message()), attrs)
    interaction = SimpleNamespace(generic_kind="density", first_reassociation=reassociation, linear=linear,
                                  skip_tp=skip)
    products = compose_skip_products(interaction)
    assert products == (([0, 1, 2] if with_linear else []), [0, 1, 2])
    reassociation.compose_skip = (linear, products[0], skip, products[1])
    got = message()
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_compose_skip_products_refuses_other_interactions():
    reassociation, linear, skip, _, _ = _case()
    base = dict(generic_kind="density", first_reassociation=reassociation, linear=linear, skip_tp=skip)
    assert compose_skip_products(SimpleNamespace(**{**base, "generic_kind": "residual"})) is None
    assert compose_skip_products(SimpleNamespace(**{**base, "skip_tp": torch.nn.Identity()})) is None
    assert compose_skip_products(SimpleNamespace(**{**base, "linear": torch.nn.Linear(2, 2)})) is None


def test_composed_is_float64_then_cast():
    right = torch.randn(6, 4, dtype=torch.float32)
    mid = torch.randn(4, 4, dtype=torch.float32)
    ws = torch.randn(3, 4, 5, dtype=torch.float32)
    w, w_t = compose_skip.composed(right, ws, mid)
    assert w.shape == (3, 6, 5) and w_t.shape == (3, 5, 6) and w.dtype == torch.float32
    assert torch.equal(w, ((right.double() @ mid.double())[None] @ ws.double()).float())
    assert torch.equal(w_t, w.transpose(1, 2))


def test_on_for_the_listed_gpus_only():
    assert compose_skip.COMPOSE_SKIP is None and compose_skip.compose_skip_default("NVIDIA H200")
    assert not compose_skip.compose_skip_default("NVIDIA A100-SXM4-80GB")
    assert not compose_skip.compose_skip_default("NVIDIA GeForce RTX 5060 Ti")
