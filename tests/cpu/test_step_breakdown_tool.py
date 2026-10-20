"""The step-breakdown tool of plan task 2-T20a without a GPU: its kernel attribution and categories.

The categories are the research engine's (step_breakdown_probe.py, 15 October 2026), so the two
tables compare row by row; FastMACE's own kernels are recognised by their custom-op names.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
    events[4]["args"].update({"grid": [4, 2, 1], "block": [128, 1, 1], "est. achieved occupancy %": 50})
    events[5]["args"]["est. achieved occupancy %"] = 25
    table, count = tool.kernel_table(events)
    assert count == 4
    by_op = {(name, op): row for (name, op, _), row in table.items()}
    gemm = by_op[("sgemm", "aten::mm")]
    assert (gemm["launches"], gemm["us"], len(gemm["ops"])) == (2, 40, 1)
    assert gemm["grid"] == [4, 2, 1] and gemm["block"] == [128, 1, 1]
    assert tool.occupancy(gemm) == (50 * 30 + 25 * 10) / 40
    assert (by_op[("triton_poi", "outer")]["launches"], by_op[("triton_poi", "outer")]["us"]) == (1, 5)
    assert by_op[("orphan", "?")]["launches"] == 1 and tool.occupancy(by_op[("orphan", "?")]) is None

    rates = tool.op_gemm_rates(table, calls=1, peak_tflops=20.0)
    (rate,) = rates.values()
    flops = 2 * ATOMS * 512 * 512
    assert rate["gflop_per_call"] == flops / 1e9
    assert abs(rate["achieved_tflops"] - flops / 40e-6 / 1e12) < 1e-12
    assert abs(rate["share_of_peak"] - rate["achieved_tflops"] / 20.0) < 1e-12


def test_gemm_flops_from_shapes():
    assert tool.gemm_flops("aten::mm", json.dumps([[216, 512], [512, 128]])) == 2 * 216 * 512 * 128
    assert tool.gemm_flops("aten::addmm", json.dumps([[128], [216, 512], [512, 128]])) == 2 * 216 * 512 * 128
    assert tool.gemm_flops("aten::bmm", json.dumps([[3, 216, 512], [3, 512, 128]])) == 3 * 2 * 216 * 512 * 128
    out = 2 * 512 * 512 * 512
    assert tool.gemm_flops("aten::mm", json.dumps([[512, 512], [512, 512], [512, 512]])) == out  # out= variant
    assert tool.gemm_flops("aten::addmm", json.dumps([[512], [512, 512], [512, 512], [512, 512]])) == out
    assert tool.gemm_flops("aten::addmm", json.dumps([[216, 128], [216, 512], [512, 128]])) == 2 * 216 * 512 * 128
    assert tool.gemm_flops("aten::mm", json.dumps([[216, 512], [100, 128]])) is None
    assert tool.gemm_flops("aten::add", json.dumps([[216, 512], [216, 512]])) is None
    assert tool.fp32_peak_tflops(12, 0, 36, 3000) == 36 * 128 * 2 * 3000e6 / 1e12
    assert tool.fp32_peak_tflops(8, 0, 108, 1410) == 108 * 64 * 2 * 1410e6 / 1e12


def test_gaps_are_counted_inside_replays_only():
    replay = [("a", 0.0, 10.0), ("b", 11.0, 20.0), ("c", 50.0, 60.0)]
    kernels = replay + [(n, s + 1000, e + 1000) for n, s, e in replay]
    per_replay = tool.split_replays(kernels, 2)
    assert per_replay == [replay, [(n, s + 1000, e + 1000) for n, s, e in replay]]
    assert tool.split_replays(kernels[:-1], 2) is None
    gaps = tool.replay_gaps(per_replay)
    assert gaps["gaps_per_replay"] == 2
    assert gaps["total_gap_us_per_replay"] == 31.0
    assert gaps["histogram_per_replay"][">=20us"] == dict(count=1.0, us=30.0)
    assert gaps["histogram_per_replay"]["<2us"] == dict(count=1.0, us=1.0)
    assert gaps["largest"][0] == dict(gap_us=30.0, before="b", after="c")


def test_ncu_rows_join_the_sequence_in_launch_order(tmp_path):
    raw = tmp_path / "ncu.csv"
    metrics = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
               "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
               "sm__warps_active.avg.pct_of_peak_sustained_active"]
    raw.write_text("==PROF== Connected\n"
                   + ",".join(f'"{h}"' for h in ["ID", "Kernel Name"] + metrics) + "\n"
                   + ",".join(f'"{u}"' for u in ["", "", "nsecond", "byte", "byte", "inst", "%"]) + "\n"
                   + '"0","sgemm_kernel<float>(int)","2,000","1000","500","1000000","40"\n'
                   + '"1","triton_poi_fused_0","1000","300","300","0","80"\n')
    rows = tool.read_ncu_csv(raw)
    assert [r["gpu__time_duration.sum"] for r in rows] == [2000.0, 1000.0]
    sequence = [dict(index="0", kernel="sgemm_kernel<float>", op="aten::mm", category="Atom-level linear layers (GEMM)",
                     input_dims="d"),
                dict(index="1", kernel="Memcpy DtoD", op="aten::copy_", category="Copies and fills", input_dims=""),
                dict(index="2", kernel="triton_poi_fused_0", op="triton", category="Atom-level elementwise",
                     input_dims="")]
    joined, matched = tool.join_ncu(rows, sequence, peak_tflops=10.0)
    assert matched == 2 and [r["index"] for r in joined] == ["0", "2"]  # ncu profiles kernels only
    assert joined[0]["fp32_flops"] == 2e6 and abs(joined[0]["achieved_tflops"] - 1.0) < 1e-12
    groups = tool.group_ncu(joined, "category", peak_tflops=10.0)
    gemm = groups["Atom-level linear layers (GEMM)"]
    assert gemm["dram_mb"] == 1500 / 1e6 and gemm["achieved_occupancy_pct"] == 40
    assert gemm["share_of_fp32_peak"] == gemm["achieved_tflops"] / 10.0


def test_nsys_trace_keeps_kernels_in_time_order(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text("Generating SQLite file\n"
                     '"Start (ns)","Duration (ns)","CorrId","Name"\n'
                     '"3000","500","2","kernel_b"\n'
                     '"1000","1000","1","kernel_a"\n'
                     '"2500","10","3","[CUDA memcpy Device-to-Device]"\n')
    assert tool.read_nsys_gpu_trace(trace) == [("kernel_a", 1.0, 2.0), ("kernel_b", 3.0, 3.5)]


def test_stage_names_match_the_research_rows():
    assert tool.stage_name("ACP").startswith("ACP: Langevin half kick")
    assert tool.stage_name(("B", 8192)) == "B: forces + second half kick"
    assert tool.stage_name("C") == "C: edge compaction after a rebuild"


def test_rewrite_overrides():
    assert tool.rewrite_overrides(["block_linear=off", "polar_uvu=false"]) == {"block_linear": "off",
                                                                                "polar_uvu": False}
    for bad in (["nonsense=on"], ["block_linear"], ["polar_uvu=maybe"]):
        with pytest.raises(ValueError):
            tool.rewrite_overrides(bad)
