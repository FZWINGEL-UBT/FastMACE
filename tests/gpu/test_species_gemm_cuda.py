"""fastmace.kernels.triton.species_gemm on CUDA: ``species_mm`` against float64 within float32 rounding of the row
scale |x| |w| (ieee) and the 3xTF32 bound (tf32x3: Triton, ``TENSOR_CORES`` and ``CUTLASS``), its input gradient,
strided inputs, a species without atoms, atoms changing species under one captured CUDA graph, and ``SpeciesSkip``
with ``SPECIES_GEMM`` bitwise or within rounding of the concatenated product."""

import pytest
import torch

from fastmace.kernels.triton import species_gemm as sg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _case(counts, size, k, n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    index = torch.cat([torch.full((c,), s, device="cuda") for s, c in enumerate(counts)])
    index = index[torch.randperm(index.numel(), generator=g, device="cuda")]
    x = torch.randn(index.numel(), k, size, device="cuda", generator=g).transpose(1, 2)  # strided, as SpeciesSkip's
    w = torch.randn(len(counts), k, n, device="cuda", generator=g) / k ** 0.5
    return index, x, w


def _cutlass(monkeypatch):
    """species_gemm.CUTLASS on any compute capability 8.x (the sm_80 cubin runs there too)."""
    from fastmace.kernels.cutlass import tf32x3
    if not tf32x3.available("cuda", family=True):
        pytest.skip("the CUTLASS cubin needs compute capability 8.x")
    monkeypatch.setattr(sg, "CUTLASS", True)
    monkeypatch.setattr(sg, "CUTLASS_FAMILY", True)


@pytest.mark.parametrize("precision, limit", [("ieee", 2e-6), ("tf32x3", 2e-5), ("tc", 2e-5), ("cutlass", 2e-5)])
@pytest.mark.parametrize("counts, size, k, n", [((1000, 1000, 3000), 7, 128, 128), ((37, 0, 101), 3, 128, 64),
                                                 ((5,), 1, 96, 130)])
def test_product_and_gradient(monkeypatch, precision, limit, counts, size, k, n):
    if precision != "ieee" and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("tf32x3 needs compute capability 8.0")
    if precision == "tc":  # the tensor-core kernel (n = 130 falls back to Triton)
        assert sg.tc_supported("cuda", k, n) == (n % 16 == 0)
        monkeypatch.setattr(sg, "TENSOR_CORES", True)
        precision = "tf32x3"
    if precision == "cutlass":  # gathered CUTLASS (strided x: lda = gcd of the row strides; n = 130 falls back)
        _cutlass(monkeypatch)
        precision = "tf32x3"
    index, x, w = _case(counts, size, k, n)
    if sg.CUTLASS:  # rows with unit column stride (the case's strided x would fall back to Triton)
        x = x.contiguous()
        assert bool(sg._cutlass_rows(x, w, x.new_empty(x.shape[0], size, n))) == (n % 4 == 0)
    x.requires_grad_()
    order, found = sg.species_plan(index, len(counts))
    y = sg.species_mm(x, w, w.transpose(1, 2).contiguous(), index, order, found, precision)
    wa = w[index]
    exact = torch.einsum("ajk,akn->ajn", x.detach().double(), wa.double())
    scale = torch.einsum("ajk,akn->ajn", x.detach().double().abs(), wa.double().abs())
    assert float(((y.detach().double() - exact).abs() / scale).max()) < limit
    grad = torch.randn_like(y)
    (gx,) = torch.autograd.grad(y, x, grad)
    exact = torch.einsum("ajn,akn->ajk", grad.double(), wa.double())
    scale = torch.einsum("ajn,akn->ajk", grad.double().abs(), wa.double().abs())
    assert float(((gx.double() - exact).abs() / scale).max()) < limit


@pytest.mark.parametrize("precision", ["ieee", "tc", "cutlass"])
def test_species_change_under_graph(monkeypatch, precision):
    if precision == "cutlass":
        _cutlass(monkeypatch)
        precision = "tf32x3"
    if precision == "tc":
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("tf32x3 needs compute capability 8.0")
        monkeypatch.setattr(sg, "TENSOR_CORES", True)
        precision = "tf32x3"
    index, x, w = _case((40, 30, 50), 3, 64, 64, seed=3)
    w_t = w.transpose(1, 2).contiguous()

    def call():
        order, counts = sg.species_plan(index, 3)
        return sg.species_mm(x, w, w_t, index, order, counts, precision)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    swap = torch.tensor([0, 1, 2], device="cuda")
    index[swap] = index[swap.flip(0)].clone()  # three atoms change species, same counts
    index[3] = (index[3] + 1) % 3  # and one changes the counts
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.einsum("ajk,akn->ajn", x, w[index])
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layout", ["ir_mul", "mul_ir"])
def test_species_skip_grouped_bitwise(monkeypatch, layout):
    from fastmace.rewrites.generic import modules as gm

    g = torch.Generator(device="cuda").manual_seed(4)
    mul = 128
    segments = [(0, mul, mul), (mul, mul, 3 * mul), (4 * mul, mul, 5 * mul)]
    blocks = [(s, s, torch.randn(3, mul, mul, device="cuda", generator=g) / mul ** 0.5) for s in range(3)]
    skip = gm.SpeciesSkip(torch.tensor([3, 7, 11], device="cuda"), segments, segments, blocks, layout).cuda()
    x1 = torch.randn(500, 9 * mul, device="cuda", generator=g)
    x2 = torch.zeros(500, 89, device="cuda")
    x2[torch.arange(500), torch.tensor([3, 7, 11], device="cuda")[torch.randint(3, (500,), device="cuda",
                                                                                  generator=g)]] = 1
    assert skip.species_gemm_gpu == (torch.cuda.get_device_name() in gm.SPECIES_GEMM_GPUS)  # the None default
    monkeypatch.setattr(gm, "SPECIES_GEMM", False)
    reference = skip(x1, x2)
    monkeypatch.setattr(gm, "SPECIES_GEMM", True)
    grouped = skip(x1, x2)
    scale = float(reference.abs().max())
    assert float((grouped - reference).abs().max()) <= 2e-6 * scale
