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


def test_shared_locator_is_off_without_a_listed_gpu(monkeypatch):
    assert sf.SHARED_LOCATOR is None and len(sf.SHARED_LOCATOR_GPUS) == 0
    assert not sf.shared_locator_default("cpu")
    monkeypatch.setattr(sf, "SHARED_LOCATOR", True)
    assert sf.shared_locator_default("cpu")


def _wprime(**options):
    return wp.render(_schedule(), "float", 4, table_w=True, factored=True, scale=True, horner=True,
                     blocks=[(1, [0]), (3, [64])], spline=GRID, **{"vector": 4, **options})


def test_shared_locator_off_leaves_both_sources_unchanged():
    blocks = [(1, [0]), (3, [64])]
    for threads, min_blocks in ((None, None), (128, 6)):
        plain = sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, threads, min_blocks)
        assert plain == sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, threads, min_blocks,
                                  shared_locator=False)
        assert "locator" not in plain and "SHARED_LOCATOR" not in plain
        assert sf.lookup_code(GRID, "re", "float", "            ") in plain
    for options in ({}, {"vector": 2}, {"stage_dout": True, "stage_quads": True}, {"hoist_scale": True}):
        plain = _wprime(**options)
        assert plain == _wprime(shared_locator=False, **options)
        assert "SHARED_LOCATOR" not in plain and "loc." not in plain
        assert "gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;" in plain


def test_shared_locator_sources_read_the_record_instead_of_locating():
    blocks = [(1, [0]), (3, [64])]
    lookup = sf.lookup_code(GRID, "re", "float", "        ")
    producer = sf.render_locator(GRID, R_MAX)
    assert lookup in producer  # the same lookup text as both kernels, so the same float32 operations
    assert "make_int4((int)((int64_t)pairs[e] * intervals + iv), __float_as_int(t), __float_as_int(sx_inv)," in producer
    assert f"constexpr float R_MAX = {(R_MAX).hex()}f;" in producer and "re < R_MAX ? 1 : 0" in producer
    for threads, min_blocks in ((None, None), (128, 6)):
        on = sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, threads, min_blocks, shared_locator=True)
        off = sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, threads, min_blocks)
        assert "const int4* __restrict__ locator," in on and "lengths" not in on
        assert "__ldg(locator + e)" in on and "if (!loc.w) continue;" in on
        assert "sx_inv" not in on and "pairs[e]" not in on and "else if" not in on
        assert "const float4* tq = table + (int64_t)loc.x * Q + ch;" in on
        # everything after the edge head (the message arithmetic and the stores) is the same text
        assert on.split("const float4* tq")[1].split("\n", 1)[1] == off.split("const float4* tq")[1].split("\n", 1)[1]
    two = _two_degree_schedule()
    for k, part in enumerate(sf.spline_parts(two)):
        assert "__ldg(locator + e)" in sf.render(two, blocks, GRID, R_MAX, None, part, 128, 6, f"spline_forward{k}",
                                                 shared_locator=True)
    with pytest.raises(ValueError):
        sf.located_source("// not a spline_forward source", GRID)
    for options in ({}, {"vector": 2}, {"stage_dout": True, "stage_quads": True}, {"hoist_scale": True}):
        on = _wprime(shared_locator=True, **options)
        assert "const int4 loc = __ldg(reinterpret_cast<const int4*>(bins) + e);" in on
        assert "gprime + int64_t(loc.x) * PADDED_TERMS * WEIGHT_DIM;" in on
        assert "else if" not in on and "pairs[e]" not in on and " sc = sx_inv;" in on
        assert "const bool sloped = r < r_max && r >= r_lo;" in on and "const bool inside = r < r_max;" in on
    with pytest.raises(ValueError):
        wp.render(_schedule(), "float", 4, vector=4, table_w=True, horner=True, shared_locator=True)


def _lengths():
    """float32 lengths: random in and around the grid, the band edges and their float32 neighbours, the cutoff."""
    r = torch.rand(4000, generator=torch.Generator().manual_seed(11), dtype=torch.float64) * 6.6
    edges = torch.tensor([a for a, _, _ in GRID] + [b for _, b, _ in GRID] + [0.0, 1e-30, 7.0, 1e9],
                         dtype=torch.float32)
    near = torch.cat([edges, torch.nextafter(edges, edges + 1), torch.nextafter(edges, edges - 1)])
    return torch.cat([r.float(), near])


def test_locator_reference_matches_the_python_locate():
    r = _lengths()
    pair = torch.randint(0, 4, r.shape, generator=torch.Generator().manual_seed(12))
    record = sf.locator_reference(r.tolist(), pair.tolist(), GRID, R_MAX)
    iv, t, inv = sp.locate(r, GRID)
    intervals = sum(n for _, _, n in GRID)
    assert record[:, 0].tolist() == (pair * intervals + iv).tolist()
    assert torch.equal(record[:, 1].view(torch.float32), t)  # bitwise
    assert torch.equal(record[:, 2].view(torch.float32), inv)
    assert record[:, 3].tolist() == (r < R_MAX).int().tolist()


def test_wprime_clamp_at_r_lo_does_not_change_the_record():
    """The w' kernel located max(r, r_lo); the record locates r. Equal wherever ``shared_locator_supported``."""
    assert sf.shared_locator_supported(GRID, GRID[0][0]) and sf.shared_locator_supported(GRID, 0.1)
    assert not sf.shared_locator_supported(GRID, 0.31)
    r = _lengths()
    r = torch.cat([r, torch.tensor([float("-inf"), -1.0, 0.29, float("inf")])])
    pair = [1] * len(r)
    plain = sf.locator_reference(r.tolist(), pair, GRID, R_MAX)
    for r_lo in (GRID[0][0], 0.1):
        clamped = torch.where(r > torch.tensor(r_lo, dtype=torch.float32), r,
                              torch.tensor(r_lo, dtype=torch.float32))
        assert torch.equal(sf.locator_reference(clamped.tolist(), pair, GRID, R_MAX)[:, :3], plain[:, :3])


def test_shared_locator_sources_compile_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    blocks = [(1, [0]), (3, [64])]
    assert compile_cubin(sf.render_locator(GRID, R_MAX), (9, 0))
    for threads, min_blocks in ((None, None), (128, 6)):
        assert compile_cubin(sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, threads, min_blocks,
                                       shared_locator=True), (9, 0))
    for options in ({}, {"vector": 2}, {"stage_dout": True, "stage_quads": True}):
        assert compile_cubin(_wprime(shared_locator=True, **options), (9, 0))


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


def test_stage_edges_is_off_without_a_listed_gpu():
    assert sf.STAGE_EDGES is None and list(sf.STAGE_EDGES_GPUS) == ["NVIDIA H200"]


def test_stage_edges_source_stages_the_tile_and_keeps_the_arithmetic():
    blocks = [(1, [0]), (3, [64])]
    plain = sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, 64, 6)
    staged = sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, 64, 6, stage_edges=True)
    assert "__ballot_sync" in staged and "s_sh[TILE * ED]" in staged and "she[" in staged
    assert "__ldg(sh + e * ED" in plain and "__ldg(sh + e * ED" not in staged
    fma = [line.strip() for line in plain.splitlines() if "acc[" in line and "fmaf" in line]
    assert fma == [line.strip() for line in staged.splitlines() if "acc[" in line and "fmaf" in line]
    with pytest.raises(ValueError, match="one thread per channel"):
        sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, 128, 6, stage_edges=True)
    with pytest.raises(ValueError, match="alternatives"):
        sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, 64, 6, shared_locator=True, stage_edges=True)


def test_stage_edges_source_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    blocks = [(1, [0]), (3, [64])]
    for capability in ((9, 0), (8, 0), (12, 0)):
        assert compile_cubin(sf.render(_schedule(), blocks, GRID, R_MAX, [1.0, 0.7], None, 64, 6,
                                       stage_edges=True), capability)


def test_stage_sh_reads_the_edge_features_from_shared_memory():
    assert wp.STAGE_SH is None and list(wp.STAGE_SH_GPUS) == ["NVIDIA H200"]
    blocks = [(1, [0]), (3, [64])]
    kwargs = dict(vector=4, table_w=True, factored=True, scale=True, horner=True, blocks=blocks, spline=GRID)
    plain = wp.render(_schedule(), "float", 4, **kwargs)
    staged = wp.render(_schedule(), "float", 4, stage_sh=4, **kwargs)
    assert "stage_sh[4][EDGE_DIM]" in staged and "sh_edge[" in staged and "__syncwarp();" in staged
    assert "sh[edge_base" in plain.replace("dsh[", "") and "sh[edge_base" not in staged.replace("dsh[", "")
    with pytest.raises(ValueError, match="stage_sh"):
        wp.render(_schedule(), "double", 4, stage_sh=4, **{**kwargs, "vector": 0, "table_w": False, "horner": False,
                                                            "spline": None})
    from fastmace.kernels import nvrtc_available
    if nvrtc_available():
        from fastmace.kernels.oeq_streaming.runtime import compile_cubin

        for capability in ((9, 0), (8, 0)):
            assert compile_cubin(staged, capability)
