"""HALF_FAR (wprime_backward / half_far): the source transforms, the float16 table and the default (off)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace.kernels.oeq_streaming import half_far
from fastmace.kernels.oeq_streaming import wprime_backward as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_wprime_backward import _schedule  # noqa: E402

GRID = [(0.3, 1.0, 70), (1.0, 2.5, 50), (2.5, 6.0, 100)]


def _source():
    return wp.render(_schedule("ir_mul", channels=64), "float", 4, table_w=True, factored=True, scale=True,
                     horner=True, blocks=[(1, [0]), (3, [64])], spline=GRID, vector=4, stage_dout=True, stage_sh=4,
                     stage_edges=True, fold_alpha=True)


def _lookup():
    from fastmace.kernels.oeq_streaming.spline_forward import lookup_code
    return lookup_code(GRID, "clamped", "float", "            ")


def test_off_by_default():
    assert wp.HALF_FAR is None and not wp.HALF_FAR_GPUS
    assert not wp.half_far_default(torch.device("cpu"))


def test_statements_follow_the_float32_loop():
    stmts = half_far.statements(half_far.channel_body(_source()))
    kinds = [s[0] for s in stmts]
    assert kinds.count("path") == kinds.count("horner") == kinds.count("slope") == kinds.count("radial") > 0
    assert kinds.count("atomic") > 0 and "let" in kinds and "acc" in kinds
    assert half_far.dout_blocks(_source()) == [(0, 0, 64), (1, 64, 192)]


def test_untranslated_statement_raises():
    unfolded = wp.render(_schedule("ir_mul", channels=64), "float", 4, table_w=True, factored=True, scale=True,
                         horner=True, blocks=[(1, [0]), (3, [64])], spline=GRID, vector=4, stage_dout=True,
                         stage_sh=4, stage_edges=True)
    with pytest.raises(ValueError, match="untranslated"):
        half_far.statements(half_far.channel_body(unfolded))


def test_kernel_source_scales_and_float32_radial_gradient():
    k = half_far.kernel_source(_source(), _lookup())
    for name in ("wprime_half_pairs", "wprime_half_scale", "wprime_half_far"):
        assert f"void {name}(" in k or f"void __launch_bounds__(256, 3) {name}(" in k
    assert "* dsc)" in k and "inv0 = ginv[slab0] / dsc" in k
    assert "rg0 = fmaf(lo2f(weight_gradient), slope0, rg0)" in k and "radial_gradient = hfma" not in k
    assert "__launch_bounds__(256, 2)" in half_far.kernel_source(_source(), _lookup(), min_blocks=2)


def test_with_skip():
    s = half_far.with_skip(_source())
    assert "const unsigned char* __restrict__ half_skip) {" in s
    assert s.index("if (half_skip[e]) continue;") < s.index("const int64_t input_base = int64_t(sender) * INPUT_DIM;")
    with pytest.raises(ValueError):
        half_far.with_skip(s.replace("    scalar_t* dr,\n", "    scalar_t* dr_,\n"))


def test_half_table_is_an_exact_power_of_two_scaling():
    torch.manual_seed(0)
    table = torch.randn(6, 8, 4) * torch.tensor([1e-6, 1e-3, 1.0, 30.0, 0.0, 1e-4])[:, None, None]
    bits, inverse = half_far.half_table(table)
    assert bits.dtype == torch.int16 and bits.shape == table.shape and inverse.shape == (6,)
    half = bits.view(torch.float16).double()
    top = half.abs().reshape(6, -1).amax(1)
    nonzero = table.abs().reshape(6, -1).amax(1) > 0
    assert torch.all((top[nonzero] >= 2.0 ** half_far.SLAB_TOP) & (top[nonzero] <= 2.0 ** (half_far.SLAB_TOP + 1)))
    assert inverse[~nonzero].item() == 1.0
    m, _ = np.frexp(inverse.numpy())
    assert np.all(m == 0.5)  # powers of two: undoing the scale is exact
    back = half * inverse.double()[:, None, None]
    assert torch.allclose(back, table.double(), rtol=2.0 ** -10, atol=0)


def test_kernels_compile_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    for capability in ((8, 0), (9, 0), (12, 0)):
        assert compile_cubin(half_far.kernel_source(_source(), _lookup()), capability)
    assert compile_cubin(half_far.with_skip(_source()), (8, 0))
