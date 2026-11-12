"""step-layout-cleanup (12 November 2026), the layout-only copies around the products (CPU part; the CUDA part is
tests/gpu/test_step_layout_cleanup_cuda.py):

- the generic interactions return ``reshape_irreps(message)`` as a view of the message where that is its value
  (cuEquivariance's ir_mul layout, one multiplicity), not as an Inductor cat of the whole message;
- the composed first interaction takes the joined layout (``JOINED_LAYOUT``: the per-slot sums d-major, every path
  read in place by ``species_mm_rows``) on the Triton tf32x3 backend too, not only on CUTLASS."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fastmace.rewrites.generic.modules import _reshaped  # noqa: E402

from test_compose_skip import _composing  # noqa: E402


def _reshape(layout):
    pytest.importorskip("mace")
    from e3nn import o3
    from mace.modules.irreps_tools import reshape_irreps

    return reshape_irreps(o3.Irreps("4x0e+4x1o+4x2e+4x3o"), cueq_config=SimpleNamespace(layout_str=layout))


def test_the_message_is_reshaped_as_a_view_with_reshapes_values():
    reshape = _reshape("ir_mul")
    message = torch.randn(9, 4 * 16, dtype=torch.float64)
    got = _reshaped(reshape, message)
    want = reshape(message)
    assert got.data_ptr() == message.data_ptr() and got.is_contiguous()  # no copy
    assert got.shape == want.shape and torch.equal(got, want)


def test_other_layouts_keep_reshapes_copy():
    message = torch.randn(9, 4 * 16, dtype=torch.float64)
    reshape = _reshape("mul_ir")
    got = _reshaped(reshape, message)
    assert got.data_ptr() != message.data_ptr() and torch.equal(got, reshape(message))
    reshape = _reshape("ir_mul")
    strided = torch.randn(64, 9, dtype=torch.float64).t()  # not contiguous: no view of reshape's order
    assert torch.equal(_reshaped(reshape, strided), reshape(strided))


@pytest.mark.parametrize("backend", ["triton", "cutlass"])
def test_the_joined_layout_runs_on_the_triton_backend_too(backend):
    reassociation, message = _composing()
    reassociation.compose_joined = reassociation.joined_layout = True
    reassociation.tf32x3, reassociation.tf32x3_backend = True, backend
    g = torch.Generator().manual_seed(11)
    nodes, slots, dim, width = 7, 3, 9, 2
    sums = torch.randn(nodes * slots, dim * width, generator=g, dtype=torch.float64)
    asked = []

    def summed_fn(key, count, inner=1):
        asked.append(inner)
        if inner == 1:
            return sums, width
        joined = sums.view(nodes, slots, dim, width).transpose(1, 2).contiguous()  # d-major: [nodes, dim, slots, k]
        return joined.view(-1, width), width

    weight = torch.randn(3 * 4, width, generator=g, dtype=torch.float64)
    attrs = torch.zeros(nodes, 5, dtype=torch.float64)
    attrs[torch.arange(nodes), reassociation.columns[torch.tensor([0, 1, 2, 2, 1, 0, 2])]] = 1
    edge_index = torch.zeros(2, 1, dtype=torch.int64)
    edge_attrs = torch.zeros(1, dim, dtype=torch.float64)

    def run(joined):
        reassociation.joined_layout = joined
        return reassociation(None, attrs, edge_attrs, None, edge_index, weight, summed_fn=summed_fn)

    plain = run(False)
    assert asked == [1]
    got = run(True)
    assert asked == [1, slots]  # the d-major sums were asked for
    assert torch.equal(got, plain)  # the same products (the CPU reference: per path, joined)
