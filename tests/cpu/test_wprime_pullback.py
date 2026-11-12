"""LATE_PULLBACK (wprime_pullback, 12 November 2026): the receiver-adjoint conversion, the covered path forms, the
transform of an OMAT-0-medium-shaped interaction-2 render and its default. The float64 statement-by-statement
emulation of every mask against the direct contraction is in results/wprime-late-pullback-20261112 (emulate2.py)."""

import pytest
import torch

from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.oeq_streaming import wprime_pullback as wpb


@pytest.mark.parametrize("a, l, L", [(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 0, 1), (1, 1, 2), (1, 2, 3),
                                     (1, 1, 0), (1, 2, 1), (1, 3, 2)])
def test_conversion_reproduces_the_coupling(a, l, L):
    from e3nn import o3
    coef = o3.wigner_3j(a, l, L, dtype=torch.float64).numpy() * 1.7
    m, residual = wpb.conversion(a, l, L, coef)
    assert m.shape == (2 * L + 1, 2 * L + 1) and residual < 1e-12
    # q = M g: the Cartesian contraction equals the coupling for a random x, n, g
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2 * a + 1, generator=gen, dtype=torch.float64)
    n = torch.randn(3, generator=gen, dtype=torch.float64)
    g = torch.randn(2 * L + 1, generator=gen, dtype=torch.float64)
    y = o3.spherical_harmonics(l, n @ wpb.sh_frame().T, normalize=False, normalization="component")
    direct = torch.einsum("ijo,i,j,o->", torch.as_tensor(coef), x, y, g)
    assert torch.allclose(wpb.cartesian_f(a, l, L, x, n, m @ g), direct, rtol=1e-12, atol=1e-12)


def test_forms_it_covers():
    assert wpb.form(0, 2, 2) == "scalar" and wpb.form(1, 2, 3) == "raise" and wpb.form(1, 3, 2) == "lower"
    for bad in ((1, 1, 1), (2, 2, 2), (0, 4, 4), (1, 3, 4)):
        with pytest.raises(ValueError):
            wpb.form(*bad)


def test_default_on_the_h200_only(monkeypatch):
    """Decision 66 (12 November 2026): on for the H200 family (MD step SrTiO3 10x10x10 0.9597 / 0.9612, NbCl5
    0.9848 / 0.9825; gate passed); the A100 stays off until its own A/B."""
    cpu = torch.device("cpu")
    monkeypatch.setattr(wp, "LATE_PULLBACK", None)
    assert not wp.late_pullback_default(cpu) and set(wp.LATE_PULLBACK_GPUS) == {"NVIDIA H200"}
    for name, on in (("NVIDIA H200", True), ("NVIDIA H200 NVL", True), ("NVIDIA A100-SXM4-80GB", False),
                     ("NVIDIA GeForce RTX 5060 Ti", False), ("NVIDIA RTX 4000 Ada Generation", False)):
        monkeypatch.setattr(wp, "_device_name", lambda device, name=name: name)
        assert wp.late_pullback_default(None) is on, name
    monkeypatch.setattr(wp, "_device_name", lambda device: "NVIDIA H200")
    monkeypatch.setattr(wp, "LATE_PULLBACK", False)  # an A/B side's choice wins over the set
    assert not wp.late_pullback_default(None)
    monkeypatch.setattr(wp, "LATE_PULLBACK", True)
    assert wp.late_pullback_default(cpu)


def _omat_interaction2():
    """OMAT-0 medium's interaction 2 (128x0e+128x1o x 0e+1o+2e+3o, 10 uvu paths, MACE's output order) rendered as the
    H200 default without HOIST_DX (results/wprime-h200-20261111/render_sources.py)."""
    oeq = pytest.importorskip("openequivariance")
    import numpy as np

    from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, render_kernels
    ins, edge = [(0, 1), (1, -1)], [(l, (-1) ** l) for l in range(4)]
    outs, instructions = [], []
    for i1, (l1, p1) in enumerate(ins):
        for i2, (l2, p2) in enumerate(edge):
            for lo in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                if p1 * p2 == (-1) ** lo:
                    instructions.append((i1, i2, len(outs), "uvu", True))
                    outs.append(lo)
    order = sorted(range(len(outs)), key=lambda k: outs[k])
    position = {k: n for n, k in enumerate(order)}
    instructions = sorted(((a, b, position[o], m, t) for a, b, o, m, t in instructions), key=lambda i: i[2])
    irreps_out = "+".join(f"128x{outs[k]}{'e' if outs[k] % 2 == 0 else 'o'}" for k in order)
    problem = oeq.TPProblem(oeq.Irreps("128x0e+128x1o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o"), oeq.Irreps(irreps_out),
                            instructions, shared_weights=False, internal_weights=False)
    schedule, _ = render_kernels(problem, np.float32, NUM_THREADS, "ir_mul")
    by_dim = {}
    for slot in sorted(schedule.output_slots, key=lambda s: int(s.start)):
        by_dim.setdefault(int(slot.irrep_dim), [])
        if int(slot.start) not in by_dim[int(slot.irrep_dim)]:
            by_dim[int(slot.irrep_dim)].append(int(slot.start))
    blocks = tuple((d, tuple(by_dim[d])) for d in sorted(by_dim))
    source = wp.render(schedule, "float", 4, vector=4, table_w=True, blocks=blocks, factored=True, scale=True,
                       horner=True, scalar_paths=True, butterfly=True, stage_dout=True, fold_alpha=True,
                       spline=[(0.3, 0.5, 157), (0.5, 0.75, 112), (0.75, 1.0, 87), (1.0, 1.5, 124), (1.5, 2.0, 54),
                               (2.0, 2.5, 31), (2.5, 6.0, 166)],
                       stage_sh=4, skip_outside=True)
    return schedule, blocks, source


def test_transform_of_the_omat0_render():
    schedule, blocks, source = _omat_interaction2()
    assert wpb.supported(schedule, "float", blocks) and not wpb.supported(schedule, "double", blocks)
    assert wpb.apply(source, schedule, blocks, True, direct=()) == source
    out = wpb.apply(source, schedule, blocks, True)
    assert out.count("// LATE_PULLBACK path") == 10 and "edge_gradient[1] += pull_Gx" in out
    assert "if (sloped) radial_gradient += pull_R * sc;" in out
    assert "reinterpret_cast<float4*>(stage_row" not in out and out.count("stage_row[") > 40
    assert "scaled_weight * v_" not in out  # no spherical edge-gradient sums left
    hybrid = wpb.apply(source, schedule, blocks, True, direct=(2, 3))
    assert hybrid.count("// LATE_PULLBACK path") == 5 and "scaled_weight * v_" in hybrid
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt
    assert "dx_lane" in wt.hoist(out)  # HOIST_DX after it, or before it (the lab's captured H200 source):
    assert wpb.apply(wt.hoist(source), schedule, blocks, True).count("// LATE_PULLBACK path") == 10
    applied = wpb.APPLIED()
    wpb.apply(source, schedule, blocks, True)
    assert wpb.APPLIED() == applied + 1
    with pytest.raises(ValueError):  # not with CHANNEL_VECTOR (its loop runs over cbase)
        wpb.apply(source.replace("const int lane =", "const int cbase = 0; const int lane =", 1), schedule, blocks,
                  True)
    with pytest.raises(ValueError):
        wpb.apply(source, schedule, None, True)


def test_transformed_source_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin
    schedule, blocks, source = _omat_interaction2()
    for capability in ((9, 0), (8, 0)):
        assert compile_cubin(wt.hoist(wpb.apply(source, schedule, blocks, True)), capability)
