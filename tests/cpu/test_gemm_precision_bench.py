"""The OPT-T01 GEMM precision bench without a GPU: reading the production GEMMs and the error measures."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load(TOOLS / "gemm_precision_bench.py", "_tool_gemm_precision_opt_t01")


def test_operands_follow_the_profiler_shapes_and_strides():
    dims = json.dumps([[216, 512], [512, 512], [216, 512]])
    strides = json.dumps([[512, 1], [1, 512], [512, 1]])
    assert bench.operands("aten::mm", dims, strides) == (216, 512, 512, (512, 1), (1, 512))
    bias = json.dumps([[128], [216, 512], [512, 128], [216, 128]])
    assert bench.operands("aten::addmm", bias, "")[:3] == (216, 512, 128)
    bias2 = json.dumps([[216, 128], [216, 512], [512, 128]])
    assert bench.operands("aten::addmm", bias2, json.dumps([[128, 1], [512, 1], [128, 1]]))[3:] == ((512, 1),
                                                                                                    (128, 1))
    assert bench.operands("aten::mm", json.dumps([[216, 512], [10, 128]]), "") is None
    assert bench.operands("aten::add", dims, strides) is None


def test_read_gemms(tmp_path):
    path = tmp_path / "NaCl" / "gemms.csv"
    path.parent.mkdir()
    path.write_text(
        "category,op,gflop_per_call,us_per_call,invocations_per_call,achieved_tflops,share_of_fp32_peak,kernels,"
        "input_dims,input_strides\n"
        '"Atom-level linear layers (GEMM)",aten::mm,0.1,50.0,2.0,10,0.5,sgemm,'
        '"[[216, 512], [512, 512], [216, 512]]","[[512, 1], [512, 1], [512, 1]]"\n'
        '"Other",aten::mm,0,1,1,,,k,"[[2, 3]]",""\n')
    (gemm,) = bench.read_gemms(path)
    assert (gemm["m"], gemm["k"], gemm["n"]) == (216, 512, 512)
    assert gemm["invocations_per_call"] == 2.0 and gemm["production_us_per_call"] == 50.0
    assert gemm["a_stride"] == (512, 1) and gemm["b_stride"] == (512, 1)


def test_relative_errors():
    reference = np.array([[1.0, -2.0], [4.0, 0.0]])
    got = reference + np.array([[0.0, 0.0], [0.4, 0.0]])
    max_rel, frob_rel = bench.relative_errors(got, reference)
    assert abs(max_rel - 0.1) < 1e-15
    assert abs(frob_rel - 0.4 / 21 ** 0.5) < 1e-15
    assert bench.relative_errors(np.zeros(2), np.zeros(2)) == (0.0, 0.0)
