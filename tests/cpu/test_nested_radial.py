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
    out = nested.rotation(table, 1, 1)
    assert out.shape == (1, 64, 64) and torch.equal(out[0], table.nested_map[0])
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


class _FarChain(_Chain):
    """48 compact bumps centred beyond R_FAR: the far band needs about 48 directions, more than 32."""

    def __init__(self):
        super().__init__()
        self.centres = torch.linspace(3.4, 5.6, 48, dtype=torch.float64)


def test_setup_certificate_fails_loudly_when_a_table_needs_more_far_columns():
    module = radial._table_module(radial.fit(_FarChain(), [(0, 0)], R_MAX), [0], torch.device("cpu"))
    assert not nested.install(module)
    assert max(module.nested_errors[0]) > nested.TOLERANCE
    assert not hasattr(module, "nested_coef")
    with pytest.raises(RuntimeError, match="need more than 32 far columns"):
        nested.require(module)
    with pytest.raises(RuntimeError):
        nested.activate(module)


def test_activate_holds_one_table_and_release_restores_the_plain_rows():
    module = radial._table_module(radial.fit(_Chain(), [(0, 0)], R_MAX), [0], torch.device("cpu"))
    plain = {name: getattr(module, name).clone() for name in ("coef", "dcoef", "pcoef")}
    assert nested.install(module)
    rotated = module.nested_coef.clone()
    nested.activate(module)
    assert torch.equal(module.coef, rotated) and not hasattr(module, "nested_coef")
    assert not hasattr(module, "nested_dcoef") and not hasattr(module, "nested_pcoef")
    r = torch.tensor([[1.5]])
    with pytest.raises(RuntimeError, match="split walks"):
        module(r, torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long))
    holder = SimpleNamespace(radial_table=module, first_reassociation=None)
    nested.release([holder])
    for name, value in plain.items():
        assert torch.equal(getattr(module, name), value)
    assert not module.nested_active and not hasattr(module, "nested_ok")


def test_whole_system_and_tiled_calls_agree_with_nested_on(monkeypatch):
    from fastmace.rewrites.generic import modules as gm

    module = radial._table_module(radial.fit(_Chain(1), [(0, 0)], R_MAX), [0], torch.device("cpu"))
    g = torch.Generator().manual_seed(4)
    c, nodes, edges = 3, 6, 60
    dim = sum(2 * l + 1 for l in LS)  # noqa: E741
    columns = torch.tensor([0])
    attrs = torch.ones(nodes, 1)
    x = torch.randn(1, c, generator=g)
    reassociation = FirstInteractionReassociation(columns, c, LS, (1.0, 0.7), ir_mul=True, species_feats=x,
                                                  segmented=True)
    segments, linear_segments, offset = [], [], 0
    for l in LS:  # noqa: E741
        segments.append((offset, c, c * (2 * l + 1)))
        linear_segments.append((offset, c, 2 * l + 1))
        offset += c * (2 * l + 1)
    skip = SpeciesSkip(columns, segments, segments, [(k, k, torch.randn(1, c, c, generator=g)) for k in range(2)],
                       "ir_mul")
    linear = BlockLinear(torch.nn.Identity(), torch.nn.Identity(), linear_segments, linear_segments,
                         [(k, k, torch.randn(c, c, generator=g)) for k in range(len(LS))])
    holder = SimpleNamespace(generic_kind="density", first_reassociation=reassociation, linear=linear, skip_tp=skip,
                             radial_table=module,
                             radial_last=SimpleNamespace(weight=torch.randn(64, len(LS) * c, generator=g), h_in=1.0,
                                                         var_in=1.0, var_out=1.0))
    products = compose_skip_products(holder)
    reassociation.compose_skip = (linear, products[0], skip, products[1])
    monkeypatch.setattr(gm, "fused_first_applies", lambda *args, **kwargs: True)
    receiver = torch.sort(torch.randint(0, nodes, (edges,), generator=g)).values
    sender = torch.randint(0, nodes, (edges,), generator=g)
    edge_index = torch.stack([sender, receiver])
    r = 0.8 + 5.0 * torch.rand((edges, 1), generator=g)
    edge_attrs = torch.randn(edges, dim, generator=g)

    def call(rows=None):
        return gm.fused_first_message(holder, r, None, attrs, edge_attrs, edge_index, rows=rows)

    whole, density = call()
    tiled, tiled_density = call(rows=nodes)
    assert nested.install(module)
    reassociation.nested_rotation = nested.rotation(module, 1, 1)
    nested.activate(module)
    nested_whole, nested_density = call()
    scale = whole.abs().max()
    assert (nested_whole - whole).abs().max() <= 1e-5 * scale  # the rotation round trip in float32 (no truncation)
    assert torch.equal(nested_density, density)
    with pytest.raises(RuntimeError, match="whole-system"):
        call(rows=nodes)  # the tiled path releases first
    nested.release([holder])
    again, again_density = call(rows=nodes)
    assert torch.equal(again, tiled) and torch.equal(again_density, tiled_density)
    assert torch.equal(call()[0], whole)
