import sys

sys.path[:0] = ["/mnt/d/Margraf/fm-a100-colab", "/mnt/d/Margraf/fm-a100-colab/tests/cpu"]
from fastmace.kernels.oeq_streaming import half_far
from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.oeq_streaming.runtime import compile_cubin
from fastmace.kernels.oeq_streaming.spline_forward import lookup_code
from test_wprime_backward import _schedule

GRID = [(0.3, 1.0, 70), (1.0, 2.5, 50), (2.5, 6.0, 100)]
src = wp.render(_schedule("ir_mul", channels=64), "float", 4, table_w=True, factored=True, scale=True, horner=True, fold_alpha=True,
                blocks=[(1, [0]), (3, [64])], spline=GRID, vector=4, stage_dout=True, stage_sh=4, stage_edges=True)
st = half_far.statements(half_far.channel_body(src))
print([s[0] for s in st][:40])
print(half_far.dout_blocks(src))
k = half_far.kernel_source(src, lookup_code(GRID, "clamped", "float", "            "))
for cap in ((8, 0), (9, 0), (12, 0)):
    print(cap, len(compile_cubin(k, cap)))
print(len(compile_cubin(half_far.with_skip(src), (8, 0))))
