"""SPLINE_W without a GPU: the piecewise-uniform spline fit and its certificate (``rewrites.radial.spline``), the
q-major table both kernels read, the generated forward and w' sources and their NVRTC compilation, the switch."""

import math

import pytest
import torch

from fastmace.kernels.oeq_streaming import spline_forward as sf
from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.rewrites.radial import spline as sp

R_LO, R_MAX = 0.3, 6.0


class _Chain:
    """A smooth chain ``r -> [h c, c, density]`` with exact r-derivatives: steep at short r, as MACE's radial
    networks are (``_GenericChain``'s interface); ``noise`` mimics the float32 network's own error."""

    def __init__(self, dtype, noise=0.0, hidden=6):
        self.dtype, self.device, self.noise, self.hidden = dtype, torch.device("cpu"), noise, hidden

    def __call__(self, r, s, t):
        k = torch.arange(1, self.hidden + 1, dtype=self.dtype)
        cutoff = 0.5 * (1 + torch.cos(math.pi * r / R_MAX))
        h = torch.exp(-(1 + 0.3 * (s + t)) * r[:, None] * k / 3) * torch.sin(k * r[:, None]) / r[:, None]
        out = torch.cat([h * cutoff[:, None], cutoff[:, None], (cutoff ** 2)[:, None]], dim=1)
        if self.noise:
            out = out * (1 + self.noise * torch.sin(1e4 * r))[:, None]
        return out

    def with_derivative(self, r, s, t):
        return torch.func.jvp(lambda x: self(x, s, t), (r,), (torch.ones_like(r),))


def _fit(points=1500):
    weight = torch.randn(12, 7, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    weight[:, -1] = 0  # [W, 0]
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return sp.fit(_Chain(torch.float64), _Chain(torch.float32, noise=3e-6), weight, pairs, R_LO, R_MAX, points,
                  1.0), weight, pairs


def test_fit_refines_short_bands_and_passes_its_certificate():
    (spline, certificate), weight, pairs = _fit()
    grid = spline["grid"]
    assert grid[0][0] == R_LO and grid[-1][1] == R_MAX
    assert all(abs(b - a2) < 1e-12 for (_, b, _), (a2, _, _) in zip(grid, grid[1:]))
    assert [round(b, 6) for _, b, _ in grid[:-1]] == [e for e in sp.BAND_EDGES if R_LO < e < R_MAX]
    spacing = [(b - a) / n for a, b, n in grid]
    assert max(spacing) <= (R_MAX - R_LO) / sp.BASE_INTERVALS + 1e-12
    assert spacing[0] < spacing[-1]  # denser where the chain curves most
    assert certificate["worst"] <= 1.0
    assert all(band["score"] <= sp.TARGET for band in certificate["bands"])
    assert spline["coef"].shape == (len(pairs), sum(n for _, _, n in grid), 4, weight.shape[0])
    assert spline["coef"].dtype == torch.float32


def test_spline_reproduces_w_and_its_slope():
    (spline, _), weight, pairs = _fit()
    chain = _Chain(torch.float64)
    r = (R_LO + (R_MAX - R_LO) * torch.rand(400, generator=torch.Generator().manual_seed(5),
                                            dtype=torch.float64)).float()
    for k, (s, t) in enumerate(pairs):
        g, dg = chain.with_derivative(r.double(), s, t)
        w, dw = g[:, :-1] @ weight.T, dg[:, :-1] @ weight.T
        got, dgot = sp.evaluate(spline["coef"], spline["grid"], r, torch.full(r.shape, k, dtype=torch.long))
        assert torch.allclose(got.double(), w, atol=1e-4 * w.abs().max().item())
        assert torch.allclose(dgot.double(), dw, atol=2e-3 * dw.abs().max().item())


def test_locate_puts_every_length_in_its_band():
    grid = [(0.3, 1.0, 70), (1.0, 2.5, 50), (2.5, 6.0, 100)]
    r = torch.tensor([0.1, 0.3, 0.999, 1.0, 2.49, 2.5, 5.99, 6.0, 7.0], dtype=torch.float64)
    iv, t, inv = sp.locate(r, grid)
    assert iv.tolist()[:2] == [0, 0] and iv[3] == 70 and iv[5] == 120 and iv[-1] == 219 and iv[-2] == 219
    inside = r < 6.0  # beyond r_max the kernels mask the edge (forward: skipped; w': no weight, no slope)
    assert bool(((t[inside] >= 0) & (t[inside] <= 1 + 1e-9)).all())
    assert inv[0] == 100 and abs(inv[4] - 50 / 1.5) < 1e-9 and abs(inv[6] - 100 / 3.5) < 1e-9


def test_spline_table_is_q_major_with_the_row_scale():
    coef = torch.randn(3, 5, 4, 8, dtype=torch.float64)
    scale = torch.linspace(0.5, 2.0, 8, dtype=torch.float64)
    table = wp.spline_table(coef, scale)
    assert table.shape == (15, 8, 4) and table.dtype == torch.float32 and table.is_contiguous()
    assert torch.allclose(table[2 * 5 + 3, 6].double(), coef[2, 3, :, 6] * scale[6], rtol=1e-6)


def test_switch_is_off_without_a_listed_gpu(monkeypatch):
    assert sf.SPLINE_W is None
    assert not sf.spline_w_default("cpu")
    monkeypatch.setattr(sf, "SPLINE_W", True)
    assert sf.spline_w_default("cpu")


def _schedule():
    from test_wprime_backward import _schedule as schedule

    return schedule("ir_mul", channels=64)


GRID = [(0.3, 1.0, 70), (1.0, 2.5, 50), (2.5, 6.0, 100)]


def test_forward_source_divides_out_the_path_scale_and_stores_each_slot_once():
    schedule = _schedule()
    blocks = [(1, [0]), (3, [64])]
    source = sf.render(schedule, blocks, GRID, R_MAX, path_scale=[1.0, 0.25])
    assert "spline_forward(" in source and source.count("else if") == len(GRID) - 2
    assert (2.0).hex() + "f" in source and (0.5).hex() + "f" not in source  # path 1's CG 0.5 over its scale 0.25
    assert source.count("out0[(int64_t)rcv") == 1 and source.count("out1[(int64_t)rcv") == 3
    assert "a[" not in source and "W[" not in source  # no a.W product
    from types import SimpleNamespace
    with pytest.raises(ValueError):
        sf.render(SimpleNamespace(**{**vars(schedule), "layout": "mul_ir"}), blocks, GRID, R_MAX)


def _two_degree_schedule():
    """``_schedule`` with its second path's input a vector (1o x 0e-like: input_irrep_dim 3)."""
    from types import SimpleNamespace

    schedule = _schedule()
    first, second = schedule.paths
    second = SimpleNamespace(**{**vars(second), "input_start": schedule.channels, "input_irrep_dim": 3})
    return SimpleNamespace(**{**vars(schedule), "paths": (first, second), "input_dim": 4 * schedule.channels})


def test_parts_split_the_output_slots_and_launch_bounds_render():
    assert sf.spline_parts(_schedule()) is None  # one input degree
    schedule = _two_degree_schedule()
    blocks = [(1, [0]), (3, [64])]
    parts = sf.spline_parts(schedule)
    assert parts is not None and sorted(i for p in parts for i in p) == list(range(len(schedule.paths)))
    whole = sf.render(schedule, blocks, GRID, R_MAX)
    sources = [sf.render(schedule, blocks, GRID, R_MAX, part=p, name=f"spline_forward{k}") for k, p in enumerate(parts)]
    stores = [s.count("[(int64_t)rcv *") for s in sources]
    assert all(stores) and sum(stores) == whole.count("[(int64_t)rcv *")  # every slot stored by exactly one part
    assert "__launch_bounds__(256) spline_forward(" in whole
    bounded = sf.render(schedule, blocks, GRID, R_MAX, threads=128, min_blocks=6, name="spline_forward0")
    assert "__launch_bounds__(128, 6) spline_forward0(" in bounded


def test_wprime_source_reads_the_spline_interval():
    source = wp.render(_schedule(), "float", 4, vector=2, table_w=True, factored=True, scale=True, horner=True,
                       blocks=[(1, [0]), (3, [64])], spline=GRID)
    assert "sx_inv" in source and "bins[bin]" not in source
    assert "const float sc = sx_inv;" in source or "const scalar_t sc = sx_inv;" in source
    with pytest.raises(ValueError):
        wp.render(_schedule(), "float", 6, vector=2, table_w=True, horner=True, spline=GRID)
    wide = wp.render(_schedule(), "float", 4, vector=4, table_w=True, factored=True, scale=True, horner=True,
                     blocks=[(1, [0]), (3, [64])], spline=GRID)  # SPLINE_VECTOR 4: one float4 per weight
    assert "const float4 v = __ldg(gq + k);" in wide and "PADDED_TERMS = 4;" in wide and "float2" not in wide


def test_spline_sources_compile_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    blocks = [(1, [0]), (3, [64])]
    for capability in ((9, 0), (8, 0), (12, 0)):
        assert compile_cubin(sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7]), capability)
        two = _two_degree_schedule()
        for part in sf.spline_parts(two):
            assert compile_cubin(sf.render(two, blocks, GRID, R_MAX, [1.0, 0.7], part, 128, 6), capability)
        for vector in (2, 4):
            assert compile_cubin(wp.render(_schedule(), "float", 4, vector=vector, table_w=True, factored=True,
                                           scale=True, horner=True, blocks=blocks, spline=GRID), capability)


def test_stage_quads_read_four_staged_components_per_shared_load():
    """STAGE_QUADS: STAGE_DOUT's block-layout row as component quartets; one float4 shared load per quartet."""
    schedule = _schedule()
    blocks = [(1, [0]), (3, [64])]
    quads, quad_of = wp.stage_quad_layout(schedule, blocks)
    assert [len(q) for q in quads] == [4] and sorted(quad_of.values()) == [(0, k) for k in range(4)]
    plain = wp.render(schedule, "float", 4, vector=4, table_w=True, factored=True, scale=True, horner=True,
                      blocks=blocks, spline=GRID, stage_dout=True)
    quad = wp.render(schedule, "float", 4, vector=4, table_w=True, factored=True, scale=True, horner=True,
                     blocks=blocks, spline=GRID, stage_dout=True, stage_quads=True)
    assert "reinterpret_cast<const float4*>(stage_row)[0 + channel]" in quad and quad.count("sq0 = ") == 1
    assert all(f"sq0.{c}" in quad for c in "xyzw") and "stage_row[" not in quad.split("STAGE_QUADS: quartet")[1]
    total = lambda src: src.split("stage_row[")[1].split("]")[0]  # noqa: E731
    assert int(total(quad)) == 4 * schedule.channels <= int(total(plain)) + 3 * schedule.channels
    with pytest.raises(ValueError):
        wp.render(schedule, "float", 4, vector=4, table_w=True, horner=True, blocks=blocks, spline=GRID,
                  stage_quads=True)
    from fastmace.kernels import nvrtc_available
    if nvrtc_available():
        from fastmace.kernels.oeq_streaming.runtime import compile_cubin
        assert compile_cubin(quad, (9, 0))


def test_factor_w_and_cursor32_render_and_compile():
    """FACTOR_W hoists w times a path's common input / edge component; CURSOR32 keeps the loop state in int; off by
    default, and the default source is unchanged by either switch's code."""
    assert sf.FACTOR_W is False and sf.CURSOR32 is False
    blocks = [(1, [0]), (3, [64])]
    two = _two_degree_schedule()
    plain = sf.render(two, blocks, GRID, R_MAX, [1.0, 0.7])
    assert "wf" not in plain and "for (int64_t e = eb" in plain and "(int64_t)e * ED" not in plain
    factored = sf.render(two, blocks, GRID, R_MAX, [1.0, 0.7], factor_w=True)
    hoisted = [line for line in factored.splitlines() if "const float wf = w * " in line]
    assert hoisted and factored.count("fmaf(wf, ") >= len(hoisted)
    narrow = sf.render(two, blocks, GRID, R_MAX, [1.0, 0.7], cursor32=True)
    assert "for (int e = eb" in narrow and "(int64_t)e * ED" in narrow and "for (int owner" in narrow
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    both = sf.render(two, blocks, GRID, R_MAX, [1.0, 0.7], threads=128, min_blocks=6, factor_w=True, cursor32=True)
    assert compile_cubin(both, (9, 0))
