"""tf32x3.HYBRID (hybrid_f16.h): on for the A100 family only, its cubins shipped and recorded, the build flag wired."""
import hashlib
from pathlib import Path

from fastmace.kernels.cutlass import tf32x3

HERE = Path(tf32x3.__file__).parent


def test_on_for_the_a100_family_and_the_h200():
    assert tf32x3.HYBRID is None
    for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB PCIe", "NVIDIA H200"):
        assert name in tf32x3.HYBRID_GPUS
    for name in ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation"):
        assert name not in tf32x3.HYBRID_GPUS


def test_hybrid_cubins_shipped_and_recorded():
    record = (HERE / "tf32x3_sm80.build.txt").read_text()
    for capability, name in (((8, 0), "sm_80"), ((9, 0), "sm_90")):
        cubin = tf32x3.HYBRID_CUBINS[capability]
        assert cubin.is_file() and cubin.name.startswith("tf32x3_hyb_")
        assert f"sha256 hybrid {name}: {hashlib.sha256(cubin.read_bytes()).hexdigest()}" in record
    assert set(tf32x3.HYBRID_CUBINS) == set(tf32x3.CUBINS)


def test_build_flag_swaps_every_math_operator():
    source = (HERE / "tf32x3_sm80.cu").read_text()
    assert "#define FM_MATH_OP cutlass::arch::OpMultiplyAddHybridF16" in source
    assert source.count("FM_MATH_OP") >= 4 and "OpMultiplyAddFastF32>::GemmKernel" not in source
    header = (HERE / "hybrid_f16.h").read_text()
    assert "struct OpMultiplyAddHybridF16" in header and "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in header
    assert "-DFM_HYBRID_F16" in (HERE.parents[2] / "tools" / "build_cutlass_tf32x3.sh").read_text()


# Range check before the merge (results/hybrid-gemm-20261111/range_probe.py): HYBRID's cross parts are float16, so
# operands must stay below 32,752. Bar: every recorded operand and the OMAT-0 medium weights at most 3,000 (10x).
RANGE_RECORD = Path(__file__).resolve().parents[2] / "results" / "hybrid-gemm-20261111" / "range_probe.json"
HYBRID_LIMIT = 32752.0
MARGIN = 10.0


def test_recorded_operands_and_weights_keep_a_tenfold_margin():
    record = __import__("json").loads(RANGE_RECORD.read_text())
    assert record["limit"] == HYBRID_LIMIT
    assert record["max_abs_parameter"] * MARGIN <= HYBRID_LIMIT
    for name in ("SrTiO3", "NaCl", "NbCl5"):
        assert name in record["cases"]
    for name, case in record["cases"].items():
        assert max(case["max_a"], case["max_b"]) * MARGIN <= HYBRID_LIMIT, name


def test_omat0_medium_weights_keep_a_tenfold_margin():
    """With the checkpoint at hand ($FASTMACE_MODELS), its parameters still match the record's margin."""
    import os

    import pytest
    import torch
    models = os.environ.get("FASTMACE_MODELS")
    path = Path(models) / "mace-omat-0-medium.model" if models else None
    if path is None or not path.is_file():
        pytest.skip("no OMAT-0 medium checkpoint under FASTMACE_MODELS")
    model = torch.load(path, map_location="cpu", weights_only=False)
    largest = max(float(t.detach().abs().max()) for t in model.parameters() if t.numel())
    assert largest * MARGIN <= HYBRID_LIMIT
    record = __import__("json").loads(RANGE_RECORD.read_text())
    assert abs(largest - record["max_abs_parameter"]) <= 1e-3 * record["max_abs_parameter"]
