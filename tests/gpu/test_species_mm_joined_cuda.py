"""species_gemm.species_mm_joined on CUDA: bitwise the per-path ``species_mm`` products joined by ``torch.cat``, and
their input gradients, eager and compiled (the rows built in the compiled graph, the product opaque)."""
import pytest
import torch

from fastmace.kernels.triton import species_gemm as sg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _operands(seed=0, atoms=301, k=192, n=128, species=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    index = torch.randint(0, species, (atoms,), generator=g, device="cuda")
    sizes = [1, 3, 5, 7]
    xs = [torch.randn(atoms, size, k, generator=g, device="cuda") for size in sizes]
    ws = [torch.randn(species, k, n, generator=g, device="cuda") for _ in sizes]
    w_ts = [w.transpose(1, 2).contiguous() for w in ws]
    return index, xs, ws, w_ts


def _per_path(xs, ws, w_ts, index, precision):
    order, counts = sg.species_plan(index, ws[0].shape[0])
    return torch.cat([sg.species_mm(x, w, w_t, index, order, counts, precision) for x, w, w_t in zip(xs, ws, w_ts)],
                     dim=1)


def _joined(xs, ws, w_ts, index, precision):
    order, counts = sg.species_plan(index, ws[0].shape[0])
    gathers, scatters = sg.joined_rows(order, [x.shape[1] for x in xs])
    return sg.species_mm_joined(xs, ws, w_ts, index, order, counts, gathers, scatters, precision)


@pytest.mark.parametrize("precision", ["ieee", "tf32x3", "tf32x3-cutlass"])
@pytest.mark.parametrize("compiled", [False, True])
def test_joined_equals_the_per_path_products(precision, compiled):
    index, xs, ws, w_ts = _operands()
    a = [x.clone().requires_grad_() for x in xs]
    b = [x.clone().requires_grad_() for x in xs]
    joined = torch.compile(_joined, fullgraph=True) if compiled else _joined
    got = joined(a, ws, w_ts, index, precision)
    want = _per_path(b, ws, w_ts, index, precision)
    assert torch.equal(got, want)
    up = torch.randn_like(got)
    for ga, gb in zip(torch.autograd.grad(got, a, up), torch.autograd.grad(want, b, up)):
        assert torch.equal(ga, gb)
