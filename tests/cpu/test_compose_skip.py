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
    message.radial = SimpleNamespace(weight=weight.t().clone(), h_in=1.0, var_in=1.0, var_out=1.0)
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


def _composing(seed=0):
    reassociation, linear, skip, attrs, message = _case(seed)
    interaction = SimpleNamespace(generic_kind="density", first_reassociation=reassociation, linear=linear,
                                  skip_tp=skip)
    products = compose_skip_products(interaction)
    reassociation.compose_skip = (linear, products[0], skip, products[1])
    return reassociation, message


def test_fast_is_on_for_the_listed_gpus_only():
    assert compose_skip.COMPOSE_FAST is None and compose_skip.compose_fast_default("NVIDIA H200")
    assert not compose_skip.compose_fast_default("NVIDIA A100-SXM4-80GB")
    assert not compose_skip.compose_fast_default("NVIDIA GeForce RTX 5060 Ti")


def test_bound_weights_equal_the_forwards_own_composition():
    reassociation, message = _composing()
    inline = message()
    table = reassociation.species_feats
    reassociation.compose_radial = message.radial
    reassociation.bind(table)
    assert reassociation._buffers.get("composed_0") is not None
    assert torch.equal(message(), inline)  # bitwise: the same operations once instead of per call


def test_bound_weights_follow_the_table_in_place():
    reassociation, message = _composing()
    table = reassociation.species_feats
    reassociation.compose_radial = message.radial
    reassociation.bind(table)
    before = [reassociation._buffers[f"composed_{k}"] for k in range(6)]
    pointers = [t.data_ptr() for t in before]
    table.mul_(1.5)  # a refresh: same tensor, new values (forces._setup_tables' refresh)
    reassociation.refresh_composed(table)
    after = [reassociation._buffers[f"composed_{k}"] for k in range(6)]
    assert [t.data_ptr() for t in after] == pointers  # captured graphs keep their addresses
    expected = reassociation._compose_weights(table, reassociation._radial_weight())
    assert all(torch.equal(a, b) for a, b in zip(after, expected))
    reassociation.compose_radial = None  # the forward's own composition sees the new table too
    for k in range(6):
        del reassociation._buffers[f"composed_{k}"]
    unbound = message()
    reassociation.compose_radial = message.radial
    reassociation.bind(table)
    assert torch.equal(message(), unbound)


def test_a_second_table_gets_its_own_weights():
    reassociation, message = _composing()
    first = reassociation.species_feats
    second = first.clone() * 2
    reassociation.compose_radial = message.radial
    reassociation.bind(first)
    w_first = reassociation._buffers["composed_0"]
    reassociation.bind(second)
    w_second = reassociation._buffers["composed_0"]
    assert w_first is not w_second and not torch.equal(w_first, w_second)
    reassociation.bind(first)
    assert reassociation._buffers["composed_0"] is w_first


def test_joined_rows():
    from fastmace.kernels.triton.species_gemm import joined_rows

    order = torch.tensor([2, 0, 1], dtype=torch.int32)
    gathers, scatters = joined_rows(order, [1, 3])
    assert gathers[0].tolist() == [2, 0, 1] and scatters[0].tolist() == [8, 0, 4]
    assert gathers[1].tolist() == [6, 7, 8, 0, 1, 2, 3, 4, 5]
    assert scatters[1].tolist() == [9, 10, 11, 1, 2, 3, 5, 6, 7]
    assert gathers[0].dtype == scatters[1].dtype == torch.int32


def test_joined_product_and_gradient_equal_the_per_path_products():
    from fastmace.kernels.triton.species_gemm import joined_rows, species_mm, species_mm_joined, species_plan

    g = torch.Generator().manual_seed(3)
    species, atoms, k, n = 3, 7, 8, 4
    index = torch.tensor([0, 2, 1, 1, 0, 2, 2])
    order, counts = species_plan(index, species)
    sizes = [1, 3, 5]
    xs = [torch.randn(atoms, size, k, generator=g, dtype=torch.float64, requires_grad=True) for size in sizes]
    ws = [torch.randn(species, k, n, generator=g, dtype=torch.float64) for _ in sizes]
    w_ts = [w.transpose(1, 2).contiguous() for w in ws]
    gathers, scatters = joined_rows(order, sizes)
    joined = species_mm_joined(xs, ws, w_ts, index, order, counts, gathers, scatters, "tf32x3-cutlass")
    expected = torch.cat([species_mm(x, w, w_t, index, order, counts, "ieee") for x, w, w_t in zip(xs, ws, w_ts)],
                         dim=1)
    assert torch.equal(joined, expected)
    up = torch.randn(joined.shape, generator=g, dtype=torch.float64)
    got = torch.autograd.grad(joined, xs, up)
    want = torch.autograd.grad(expected, xs, up)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("species", [1, 3, 5])
def test_counting_plan_equals_the_stable_sort(species):
    from fastmace.kernels.triton.species_gemm import species_plan
    g = torch.Generator().manual_seed(species)
    for atoms in (1, 7, 1000):
        index = torch.randint(0, species, (atoms,), generator=g)
        order, counts = species_plan(index, species)
        counted, counted_counts = species_plan(index, species, counting=True)
        assert counted.dtype == order.dtype == torch.int32 and counted_counts.dtype == torch.int32
        assert torch.equal(counted, order) and torch.equal(counted_counts, counts)


def test_counting_plan_compiles_to_the_same_order():
    from fastmace.kernels.triton.species_gemm import species_plan
    index = torch.randint(0, 3, (500,), generator=torch.Generator().manual_seed(0))
    compiled = torch.compile(lambda i: species_plan(i, 3, counting=True), fullgraph=True)
    order, counts = compiled(index)
    assert torch.equal(order, species_plan(index, 3)[0]) and torch.equal(counts, species_plan(index, 3)[1])


def test_count_plan_is_on_for_the_h200_only():
    assert compose_skip.COUNT_PLAN is None and compose_skip.count_plan_default("NVIDIA H200")
    assert not compose_skip.count_plan_default("NVIDIA A100-SXM4-80GB")


def test_joined_layout_is_off_without_a_listed_gpu():
    assert compose_skip.JOINED_LAYOUT is None and list(compose_skip.JOINED_LAYOUT_GPUS) == []
    assert not compose_skip.joined_layout_default("NVIDIA H200")


def test_species_mm_rows_equals_the_joined_product_of_the_row_groups():
    """species_mm_rows reads each path as a row group of one [atoms * R, K] input (JOINED_LAYOUT) and returns one
    gradient tensor; values and gradients are those of species_mm_joined on the paths' contiguous copies."""
    from fastmace.kernels.triton.species_gemm import joined_rows, species_mm_joined, species_mm_rows, species_plan

    g = torch.Generator().manual_seed(5)
    atoms, k, n, species, sizes = 7, 12, 8, 3, [1, 3, 5, 7]
    index = torch.randint(0, species, (atoms,), generator=g)
    order, counts = species_plan(index, species)
    ws = [torch.randn(species, k, n, generator=g, dtype=torch.float64) for _ in sizes]
    w_ts = [w.transpose(1, 2).contiguous() for w in ws]
    x = torch.randn(atoms * sum(sizes), k, generator=g, dtype=torch.float64, requires_grad=True)
    gathers, scatters = joined_rows(order, sizes)
    out = species_mm_rows(x, sizes, ws, w_ts, index, order, counts, scatters, "ieee")
    views = x.detach().view(atoms, sum(sizes), k)
    starts = [sum(sizes[:p]) for p in range(len(sizes))]
    lefts = [views[:, s:s + size].clone().requires_grad_() for s, size in zip(starts, sizes)]
    ref = species_mm_joined(lefts, ws, w_ts, index, order, counts, gathers, scatters, "ieee")
    assert torch.equal(out, ref)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert torch.equal(x.grad.view(atoms, sum(sizes), k), torch.cat([left.grad for left in lefts], dim=1))
