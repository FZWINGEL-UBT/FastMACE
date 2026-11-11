"""rewrites.radial.nested on the CPU: the rotated table maps back to the original, its far truncation is within the
tolerance, and the composed products of rotated sums equal the plain products of the original sums."""
import math
from types import SimpleNamespace

import pytest
import torch

from fastmace.kernels.triton import compose_skip
from fastmace.kernels.triton.radial_table import evaluate
from fastmace.rewrites import radial
from fastmace.rewrites.generic.modules import SpeciesSkip, compose_skip_products
from fastmace.rewrites.linear.modules import BlockLinear
from fastmace.rewrites.mh1.modules import FirstInteractionReassociation
from fastmace.rewrites.radial import nested

R_MAX = 6.0


class _Chain:
    """64 channels: 16 smooth functions over the whole range (mixed) plus 48 compact bumps that end before 3 Å,
    then the c column and a density column; one species pair."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.mix = torch.randn(64, 16, generator=g, dtype=torch.float64)
        self.centres = 0.8 + 1.9 * torch.rand(48, generator=g, dtype=torch.float64)

    def _columns(self, r):
        envelope = 0.5 * (torch.cos(math.pi * r / R_MAX) + 1.0)
        smooth = torch.stack([torch.cos(0.35 * (j + 1) * r) * envelope for j in range(16)], dim=1)
        u = (r[:, None] - self.centres[None, :]) / 0.3
        bumps = torch.where(u.abs() < 1, (1 - u * u) ** 4, torch.zeros_like(u))
        out = smooth @ self.mix.t()
        out[:, 16:] += bumps
        return torch.cat([out, envelope[:, None], (torch.tanh(r) * envelope)[:, None]], dim=1)

    def __call__(self, r, sender, receiver):
        return self._columns(r)

    def with_derivative(self, r, sender, receiver):
        return torch.func.jvp(self._columns, (r,), (torch.ones_like(r),))


@pytest.fixture(scope="module")
def table():
    tables = radial.fit(_Chain(), [(0, 0)], R_MAX)
    module = radial._table_module(tables, [0], torch.device("cpu"))
    assert nested.install(module)
    return module


def _values(module, coef, r, derivative=False):
    pair = torch.zeros(r.shape, dtype=torch.int32)
    return evaluate(r, pair, coef.double(), module.starts.double(), module.scales.double(), module.bins,
                    module.r_lo, module.bin_scale, module.r_max, derivative=derivative)


def test_rotated_table_maps_back_and_truncates_within_tolerance(table):
    r = torch.linspace(1.0, R_MAX - 1e-3, 3001, dtype=torch.float64)
    a = _values(table, table.coef, r)[:, :64]
    z = _values(table, table.nested_coef, r)[:, :64]
    m = table.nested_map[0]
    scale = a.abs().amax(0)
    assert ((z @ m - a).abs() / scale).max() < 2e-6  # the float32 rotated table, untruncated
    far = r >= nested.R_FAR
    truncated = z.clone()
    truncated[far, nested.FAR_COLUMNS:] = 0.0
    assert ((truncated @ m - a).abs() / scale).max() < 2e-6
    assert max(max(e) for e in table.nested_errors) <= nested.TOLERANCE
    # the density and c columns are the table's own
    assert torch.equal(table.nested_coef[..., 64:], table.coef[..., 64:])


def test_rotation_blocks_follow_the_tables_pair_rule(table):
    out = nested.rotation(table, 2, 2)
    assert out.shape == (2, 128, 128)
    m = table.nested_map
    fake = SimpleNamespace(nested_map=torch.stack([m[0], 2 * m[0], 3 * m[0], 4 * m[0]]), single=False,
                           symmetric=False, species=2)
    out = nested.rotation(fake, 2, 2)
    # receiver s, sender z reads row z * 2 + s
    assert torch.equal(out[1, :64, :64], fake.nested_map[1]) and torch.equal(out[0, 64:, 64:], fake.nested_map[2])
    assert torch.count_nonzero(out[0, :64, 64:]) == 0
    fake.symmetric = True
    out = nested.rotation(fake, 2, 2)
    assert torch.equal(out[0, 64:, 64:], fake.nested_map[1])


def test_off_by_default():
    assert nested.NESTED_RADIAL is None
    assert not nested.nested_default("NVIDIA A100-SXM4-80GB") and not nested.nested_default("NVIDIA H200")


LS = (0, 1)


def test_composed_products_of_rotated_sums_equal_the_plain_ones():
    g = torch.Generator().manual_seed(1)
    species, c, width, nodes = 2, 3, 4, 5
    dim = sum(2 * l + 1 for l in LS)  # noqa: E741
    columns = torch.tensor([0, 1])
    kinds = torch.tensor([0, 1, 1, 0, 1])
    attrs = torch.zeros(nodes, 2, dtype=torch.float64)
    attrs[torch.arange(nodes), kinds] = 1
    x = torch.randn(species, c, generator=g, dtype=torch.float64)
    reassociation = FirstInteractionReassociation(columns, c, LS, (1.0, 0.7), ir_mul=True, species_feats=x)
    weight = torch.randn(len(LS) * c, width, generator=g, dtype=torch.float64)
    segments, linear_segments, offset = [], [], 0
    for l in LS:  # noqa: E741
        segments.append((offset, c, c * (2 * l + 1)))
        linear_segments.append((offset, c, 2 * l + 1))
        offset += c * (2 * l + 1)
    mats = [torch.randn(species, c, c, generator=g, dtype=torch.float64) for _ in LS]
    skip = SpeciesSkip(columns, segments, segments, [(k, k, mat) for k, mat in enumerate(mats)], "ir_mul")
    linear = BlockLinear(torch.nn.Identity(), torch.nn.Identity(), linear_segments, linear_segments,
                         [(k, k, torch.randn(c, c, generator=g, dtype=torch.float64)) for k in range(len(LS))])
    interaction = SimpleNamespace(generic_kind="density", first_reassociation=reassociation, linear=linear,
                                  skip_tp=skip)
    products = compose_skip_products(interaction)
    reassociation.compose_skip = (linear, products[0], skip, products[1])
    # per (receiver species s, sender species z) an invertible map M: a = z M
    maps = torch.randn(species, species, width, width, generator=g, dtype=torch.float64) + 3 * torch.eye(width)
    rotation = torch.zeros(species, species * width, species * width, dtype=torch.float64)
    for s in range(species):
        for z in range(species):
            rotation[s, z * width:(z + 1) * width, z * width:(z + 1) * width] = maps[s, z]
    sums = torch.randn(nodes, species, dim, width, generator=g, dtype=torch.float64)
    rotated = torch.stack([torch.stack([sums[i, z] @ torch.linalg.inv(maps[kinds[i], z]) for z in range(species)])
                           for i in range(nodes)])
    edge_index = torch.zeros(2, 1, dtype=torch.int64)
    edge_attrs = torch.zeros(1, dim, dtype=torch.float64)

    def message(values, is_nested):
        def summed_fn(key, slots):
            return values.reshape(nodes * species, dim * width), width

        summed_fn.nested = is_nested
        return reassociation(None, attrs, edge_attrs, None, edge_index, weight, summed_fn=summed_fn)

    expected = message(sums, False)
    with pytest.raises(RuntimeError):
        message(rotated, True)  # no nested_rotation yet
    reassociation.nested_rotation = rotation
    got = message(rotated, True)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-10)
    # bound weights: the plain set and the rotated set after it
    reassociation.compose_radial = SimpleNamespace(weight=weight.t().clone(), h_in=1.0, var_in=1.0, var_out=1.0)
    reassociation.bind(x)
    assert reassociation._buffers.get(f"composed_{4 * len(LS) - 1}") is not None
    assert torch.allclose(message(rotated, True), expected, rtol=1e-10, atol=1e-10)
    assert torch.allclose(message(sums, False), expected, rtol=1e-10, atol=1e-10)


def test_composed_rotation_is_applied_in_float64():
    right = torch.randn(6, 4, dtype=torch.float32)
    ws = torch.randn(3, 4, 5, dtype=torch.float32)
    rotation = torch.randn(3, 6, 6, dtype=torch.float64)
    w, w_t = compose_skip.composed(right, ws, None, rotation)
    assert torch.equal(w, (rotation @ (right.double()[None] @ ws.double())).float())
    assert torch.equal(w_t, w.transpose(1, 2))
