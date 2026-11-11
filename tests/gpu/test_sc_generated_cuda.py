"""``rewrites.sc_generated``: product 0's generated contraction with the density division folded in, against MACE's
cuEquivariance product block on the divided features (values, the features' and the density's gradients).

OMAT-0 medium's first product block: 128x0e+128x1o+128x2e+128x3o -> 128x0e+128x1o, correlation 3, ir_mul. Another
summation order than cuEq's, so the comparison is to float32 rounding (the lab measured rms 3e-7 relative). Runs
where cuEquivariance and NVRTC are available.
"""

import pytest
import torch

ELEMENTS = 4
ATOMS = 300


def _available():
    if not torch.cuda.is_available():
        return False
    try:
        import cuequivariance_torch  # noqa: F401
        from mace.modules.blocks import EquivariantProductBasisBlock  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason="needs CUDA, cuEquivariance and mace-torch")


def _product(projected):
    from e3nn import o3
    from mace.modules.blocks import EquivariantProductBasisBlock
    from mace.modules.wrapper_ops import CuEquivarianceConfig

    torch.manual_seed(0)
    config = CuEquivarianceConfig(enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True)
    product = EquivariantProductBasisBlock(o3.Irreps("128x0e+128x1o+128x2e+128x3o"), o3.Irreps("128x0e+128x1o"),
                                           correlation=3, num_elements=ELEMENTS, use_sc=True,
                                           cueq_config=config).cuda()
    contraction = product.symmetric_contractions
    if contraction.projection.device.type != "cuda":  # a plain attribute where cuEq does not register it
        contraction.projection = contraction.projection.cuda()
    # both sides run the same linear after the contraction; cuEq's naive linear keeps CPU constants in its fx graph
    # when built on the CPU and moved, so the comparison leaves it out
    product.linear = torch.nn.Identity()
    product.requires_grad_(False)
    if projected:
        from fastmace.rewrites import cueq_projection

        cueq_projection.keep_projected(product.symmetric_contractions)
    return product


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(1)
    feats = torch.randn((ATOMS, 16, 128), generator=generator, device="cuda")
    density = 3.0 * torch.rand((ATOMS, 1), generator=generator, device="cuda")
    attrs = torch.nn.functional.one_hot(torch.randint(0, ELEMENTS, (ATOMS,), generator=generator, device="cuda"),
                                        ELEMENTS).float()
    grad = torch.randn((ATOMS, 512), generator=generator, device="cuda")
    return feats, density, attrs, grad


def _relative(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("projected", [False, True])
def test_generated_matches_cueq(projected, factored):
    """SC_GENERATED's kernel pair, and with ``factored`` SC_FACTORED's (folded weights), against cuEq on the divided
    features: output, feature and density gradients."""
    from fastmace.rewrites import sc_generated

    reference = _product(projected)
    feats, density, attrs, grad = _inputs()
    outputs = []
    for side in ("cueq", "generated"):
        product = reference
        if side == "generated":
            import copy

            from fastmace.kernels.cuda.sc_generated import GeneratedContraction

            product = copy.deepcopy(reference)
            kernels = GeneratedContraction(product.symmetric_contractions, factored=factored)
            assert kernels.factored == factored
            if factored:
                product.sc_generated_weight = kernels.fold(sc_generated._weight(product.symmetric_contractions))
            product.sc_generated_id = kernels.id
            product.__class__ = sc_generated._subclass(type(product))
        f = feats.clone().requires_grad_()
        d = density.clone().requires_grad_()
        with torch.enable_grad():
            if side == "cueq":
                out = product(f / (d[:, :, None] + 1), None, attrs)
            else:
                out = product(f, (d,), attrs)
            gf, gd = torch.autograd.grad(out, (f, d), grad)
        outputs.append((out.detach(), gf, gd))
    (a, ga, da), (b, gb, db) = outputs
    assert a.shape == b.shape and da.shape == db.shape == density.shape
    assert _relative(b, a) < 1e-5
    assert _relative(gb, ga) < 1e-5
    assert _relative(db, da) < 1e-5


def test_other_sc_runs_the_original_forward():
    from fastmace.kernels.cuda.sc_generated import GeneratedContraction
    from fastmace.rewrites import sc_generated

    product = _product(True)
    feats, density, attrs, _ = _inputs()
    expected = product(feats, None, attrs)
    product.sc_generated_id = GeneratedContraction(product.symmetric_contractions).id
    product.__class__ = sc_generated._subclass(type(product))
    assert torch.equal(product(feats, None, attrs), expected)


def test_compiled_graph_matches_eager():
    import copy

    from fastmace.kernels.cuda.sc_generated import GeneratedContraction
    from fastmace.rewrites import sc_generated

    product = copy.deepcopy(_product(True))
    product.sc_generated_id = GeneratedContraction(product.symmetric_contractions).id
    product.__class__ = sc_generated._subclass(type(product))
    feats, density, attrs, grad = _inputs()

    def forward(f, d):
        return product(f, (d,), attrs)

    def step(function):
        f = feats.clone().requires_grad_()
        d = density.clone().requires_grad_()
        with torch.enable_grad():
            out = function(f, d)
            return (out.detach(), *torch.autograd.grad(out, (f, d), grad))

    eager = step(forward)
    compiled = step(torch.compile(forward, fullgraph=True))
    for a, b in zip(eager, compiled):
        assert _relative(b, a) < 1e-6  # the linear may lower differently; the custom ops are opaque
