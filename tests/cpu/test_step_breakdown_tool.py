"""The step-breakdown tool of plan task 2-T20a without a GPU: its kernel attribution and categories.

The categories are the research engine's (step_breakdown_probe.py, 15 October 2026), so the two
tables compare row by row; FastMACE's own kernels are recognised by their custom-op names.
"""

import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = _load(TOOLS / "step_breakdown.py", "_tool_step_breakdown_t20a")
EDGES, ATOMS = 8192, 216


def test_custom_ops_get_their_research_categories():
    cases = {
        "fastmace::folded_linear_input_grad": "Radial networks: folded last layer, input gradient",
        "fastmace::edge_outer_sum_backward": "Interaction 1: reassociated edge sums",
        "fastmace::edge_outer_sum_forward": "Interaction 1: reassociated edge sums",
        "fastmace::streaming_conv_forward": "Interaction 2: convolution forward (streaming)",
        "fastmace::streaming_conv_backward": "Interaction 2: convolution backward (streaming)",
        "fastmace::fused_radial_hidden_forward": "Radial networks: fused hidden block",
    }
    for op, expected in cases.items():
        assert tool.category("kernel", op, "", EDGES) == expected


def test_gemms_split_by_rows_and_weight_width():
    folded = json.dumps([[EDGES, 65], [65, 1280]])
    other = json.dumps([[EDGES, 10], [10, 128]])
    atom = json.dumps([[ATOMS * 7, 512], [512, 512]])
    assert tool.category("sgemm", "aten::mm", folded, EDGES) == "Radial networks: folded last layer, forward GEMM"
    assert tool.category("sgemm", "aten::mm", other, EDGES) == "Radial networks: other GEMMs"
    assert tool.category("sgemm", "aten::mm", atom, EDGES) == "Atom-level linear layers (GEMM)"


def test_elementwise_and_copies():
    edge = json.dumps([[EDGES, 128]])
    atom = json.dumps([[ATOMS, 512]])
    assert (tool.category("triton_poi_fused_native_layer_norm_0", "triton", edge, EDGES)
            == "Radial networks: elementwise (norms, activations)")
    assert (tool.category("triton_poi_fused_index_1", "triton", edge, EDGES)
            == "Edge elementwise (geometry, radial basis, gathers, scatters)")
    assert (tool.category("triton_poi_fused_mul_sigmoid_2", "triton", atom, EDGES)
            == "Atom-level elementwise (gates, products, residuals, readouts)")
    assert tool.category("Memcpy DtoD", "aten::copy_", atom, EDGES) == "Copies and fills"
    assert tool.category("uniform_1d_kernel", "cuequivariance::x", atom, EDGES) == (
        "Atom-level cuEq tensor products (symmetric contraction)")


def test_kernels_go_to_the_innermost_enclosing_operator():
    events = [
        dict(cat="cpu_op", name="outer", tid=1, ts=0, dur=100, args={"Input Dims": [[ATOMS, 512]]}),
        dict(cat="cpu_op", name="aten::mm", tid=1, ts=10, dur=20, args={"Input Dims": [[ATOMS, 512], [512, 512]]}),
        dict(cat="cuda_runtime", name="cudaLaunchKernel", tid=1, ts=12, dur=2, args={"correlation": 7}),
        dict(cat="cuda_runtime", name="cudaLaunchKernel", tid=1, ts=50, dur=2, args={"correlation": 8}),
        dict(cat="kernel", name="sgemm", ts=200, dur=30, args={"correlation": 7}),
        dict(cat="kernel", name="sgemm", ts=300, dur=10, args={"correlation": 7}),
        dict(cat="kernel", name="triton_poi", ts=400, dur=5, args={"correlation": 8}),
        dict(cat="kernel", name="orphan", ts=500, dur=1, args={"correlation": 99}),
    ]
    table, count = tool.kernel_table(events)
    assert count == 4
    by_op = {(name, op): row for (name, op, _), row in table.items()}
    assert by_op[("sgemm", "aten::mm")] == [2, 40]
    assert by_op[("triton_poi", "outer")] == [1, 5]
    assert by_op[("orphan", "?")] == [1, 1]


def test_stage_names_match_the_research_rows():
    assert tool.stage_name("ACP").startswith("ACP: Langevin half kick")
    assert tool.stage_name(("B", 8192)) == "B: forces + second half kick"
    assert tool.stage_name("C") == "C: edge compaction after a rebuild"
