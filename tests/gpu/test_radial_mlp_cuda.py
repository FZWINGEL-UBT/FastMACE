"""The fused radial hidden stack's Triton kernels against PyTorch autograd (opt-2 OPT2-T02).

Forward output and input gradient of ``RadialStack`` in float32 on CUDA, against the original modules in
float64 (the exact values), held to the float32 rounding of the original modules in float32: the kernels
may differ from cuBLAS/Inductor's float32 rounding, never by more than a small multiple of it.
"""

import copy

import pytest
import torch

from fastmace.kernels.triton import radial_mlp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def stack(d, k, seed):
    torch.manual_seed(seed)
    modules = [torch.nn.LayerNorm(d), torch.nn.SiLU()]
    for _ in range(k):
        modules += [torch.nn.Linear(d, d), torch.nn.LayerNorm(d), torch.nn.SiLU()]
    with torch.no_grad():
        for module in modules:
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    return modules


def _errors(d, k, rows, seed=0, config=None):
    modules = stack(d, k, seed)
    exact = torch.nn.Sequential(*copy.deepcopy(modules)).double().cuda()
    plain = torch.nn.Sequential(*copy.deepcopy(modules)).cuda()
    fused = radial_mlp.RadialStack([m.cuda() for m in modules], config=config)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, d, device="cuda", generator=generator) * 2.0
    grad = torch.randn(rows, d, device="cuda", generator=generator)

    def run(model, inputs):
        inputs = inputs.detach().clone().requires_grad_(True)
        out = model(inputs)
        (dx,) = torch.autograd.grad(out, inputs, grad.to(out.dtype))
        return out.double(), dx.double()

    out64, dx64 = run(exact, x.double())
    out32, dx32 = run(plain, x)
    outf, dxf = run(fused, x)

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max())

    return dict(fused_out=rel(outf, out64), fused_dx=rel(dxf, dx64), plain_out=rel(out32, out64),
                plain_dx=rel(dx32, dx64))


# (rows per program, warps, store): the default and the screen's variants (opt-2 OPT2-T02 (a)+(b)).
CONFIGS = [None, (64, 8, False), (32, 4, True), (64, 8, True), (16, 4, True)]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize(("d", "k", "rows"), [(64, 2, 8192), (64, 2, 1001), (64, 1, 77), (32, 4, 513),
                                              (64, 3, 33), (32, 2, 129), (16, 1, 5)])
def test_kernels_match_autograd(d, k, rows, config):
    errors = _errors(d, k, rows, config=config)
    assert errors["fused_out"] <= max(4 * errors["plain_out"], 1e-6), errors
    assert errors["fused_dx"] <= max(4 * errors["plain_dx"], 1e-5), errors


@pytest.mark.parametrize("config", CONFIGS)
def test_strided_input_and_empty_rows(config):
    modules = stack(64, 2, 1)
    fused = radial_mlp.RadialStack([m.cuda() for m in modules], config=config)
    plain = torch.nn.Sequential(*copy.deepcopy(modules)).cuda()
    wide = torch.randn(300, 128, device="cuda")
    x = wide[:, 32:96]  # a column slice, as the joint radial input's split gives
    assert x.stride(1) == 1 and x.stride(0) == 128
    torch.testing.assert_close(fused(x), plain(x), rtol=1e-5, atol=1e-6)
    empty = torch.empty(0, 64, device="cuda", requires_grad=True)
    out = fused(empty)
    assert out.shape == (0, 64)
    out.sum().backward()
    assert empty.grad.shape == (0, 64)


@pytest.mark.parametrize("config", [None, (32, 4, True)])
def test_compiled_matches_eager(config):
    modules = stack(64, 2, 2)
    fused = radial_mlp.RadialStack([m.cuda() for m in modules], config=config)
    x = torch.randn(2048, 64, device="cuda", requires_grad=True)

    def energy(inputs):
        return (fused(inputs) ** 2).sum()

    (eager,) = torch.autograd.grad(energy(x), x)
    compiled = torch.compile(energy, fullgraph=True)
    (traced,) = torch.autograd.grad(compiled(x), x)
    torch.testing.assert_close(traced, eager, rtol=1e-5, atol=1e-6)


def _resources(d, k, config, rows=256):
    """Registers and spills of the forward and backward kernels for width d, K layers and config."""
    modules = stack(d, k, 0)
    fused = radial_mlp.RadialStack([m.cuda() for m in modules], config=config)
    gamma, beta, weight, bias, eps = fused.operands()
    block_rows, num_warps, store = config
    x = torch.randn(rows, d, device="cuda")
    out, grad, dx = torch.empty_like(x), torch.randn_like(x), torch.empty_like(x)
    saved = x.new_empty((k, rows, d) if store else (0,))
    kernels = radial_mlp._kernels()
    common = dict(BE=block_rows, D=d, K=k, STORE=store, num_warps=num_warps, grid=(1,))
    wt = weight.transpose(1, 2).contiguous()
    compiled = [kernels["forward"].warmup(x, x.stride(0), out, saved, gamma, beta, wt, bias, rows, eps, **common),
                kernels["backward"].warmup(x, x.stride(0), saved, grad, dx, gamma, beta, wt, weight.contiguous(), bias,
                                           rows, eps, **common)]
    for kernel in compiled:
        kernel._init_handles()  # Triton fills n_regs and n_spills when it loads the binary
    return [(int(kernel.n_regs), int(kernel.n_spills)) for kernel in compiled]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("d", radial_mlp.WIDTHS)
def test_the_tile_rule_never_spills(d, k):
    resources = _resources(d, k, radial_mlp.kernel_config(d))
    assert all(spills == 0 for _, spills in resources), resources


# --------------------------------------------------------------------------- the entry (opt-2 OPT2-T02c)


def _entry_case(d, k, rows, nodes, cutoff, seed=3):
    modules = stack(d, k, seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    first_wide = torch.randn(rows, 2 * d, device="cuda", generator=generator)
    first = first_wide[:, :d]  # the joint first layer's column view, as the interaction passes it
    tables = torch.randn(nodes, 2 * d, device="cuda", generator=generator)
    edge_index = torch.randint(0, nodes, (2, rows), device="cuda", generator=generator)
    cut = torch.rand(rows, 1, device="cuda", generator=generator) if cutoff else None
    upstream = torch.randn(rows, d + 1, device="cuda", generator=generator)
    return modules, first, tables[:, :d], tables[:, d:], edge_index, cut, upstream


def _entry_run(model_stack, first, source, target, edge_index, cut, upstream, fused, dtype):
    leaves = [first.detach().to(dtype).requires_grad_(True), source.detach().to(dtype).requires_grad_(True),
              target.detach().to(dtype).requires_grad_(True)]
    cutoff = cut.detach().to(dtype).requires_grad_(True) if cut is not None else None
    inputs = leaves + ([cutoff] if cutoff is not None else [])
    function = radial_mlp.radial_entry if fused else radial_mlp.entry_reference
    out = function(model_stack, leaves[0], leaves[1], leaves[2], edge_index, cutoff)
    grads = torch.autograd.grad(out, inputs, upstream.to(dtype))
    return [out.double()] + [g.double() for g in grads]


@pytest.mark.parametrize("cutoff", [True, False])
@pytest.mark.parametrize(("d", "k", "rows", "nodes"), [(64, 2, 8192, 216), (64, 2, 1001, 97), (32, 1, 77, 5),
                                                       (16, 3, 5, 3)])
def test_entry_matches_the_reference(d, k, rows, nodes, cutoff):
    modules, first, source, target, edge_index, cut, upstream = _entry_case(d, k, rows, nodes, cutoff)
    exact = radial_mlp.RadialStack([m.double().cuda() for m in copy.deepcopy(modules)])
    plain = radial_mlp.RadialStack([m.cuda() for m in copy.deepcopy(modules)])
    fused = radial_mlp.RadialStack([m.cuda() for m in modules])
    args = (first, source, target, edge_index, cut, upstream)
    reference = _entry_run(exact, *args, fused=False, dtype=torch.float64)
    rounding = _entry_run(plain, *args, fused=False, dtype=torch.float32)
    got = _entry_run(fused, *args, fused=True, dtype=torch.float32)
    names = ["output", "first", "source", "target", "cutoff"][:len(got)]
    for name, g, r, e in zip(names, got, rounding, reference):
        err = float((g - e).abs().max() / e.abs().max())
        plain_err = float((r - e).abs().max() / e.abs().max())
        assert err <= max(4 * plain_err, 1e-5), (name, err, plain_err)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("d", radial_mlp.WIDTHS)
def test_the_entry_kernels_never_spill(d, k):
    block_rows, num_warps, _ = radial_mlp.kernel_config(d)
    modules, first, source, target, edge_index, cut, upstream = _entry_case(d, k, 256, 16, True)
    fused = radial_mlp.RadialStack([m.cuda() for m in modules])
    gamma, beta, weight, bias, eps = fused.operands()
    rows = first.shape[0]
    out, saved = first.new_empty((rows, d + 1)), first.new_empty((k, rows, d))
    dx, dcut = first.new_empty((rows, d)), first.new_zeros((rows, 1))
    kernels = radial_mlp._kernels()
    common = dict(BE=block_rows, D=d, K=k, HAS_CUT=True, num_warps=num_warps, grid=(1,))
    strides = (first.stride(0), source, source.stride(0), target, target.stride(0), edge_index.contiguous(), cut)
    compiled = [
        kernels["entry_forward"].warmup(first, *strides, out, saved, gamma, beta,
                                        weight.transpose(1, 2).contiguous(), bias, rows, eps, **common),
        kernels["entry_backward"].warmup(first, *strides, saved, upstream, dx, dcut, gamma, beta, weight.contiguous(),
                                         rows, eps, **common)]
    for kernel in compiled:
        kernel._init_handles()
    resources = [(int(kernel.n_regs), int(kernel.n_spills)) for kernel in compiled]
    assert all(spills == 0 for _, spills in resources), resources
