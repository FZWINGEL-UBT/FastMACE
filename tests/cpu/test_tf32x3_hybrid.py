"""tf32x3.HYBRID (hybrid_f16.h): off by default, its cubins shipped and recorded, the build flag wired."""
import hashlib
from pathlib import Path

from fastmace.kernels.cutlass import tf32x3

HERE = Path(tf32x3.__file__).parent


def test_off_by_default():
    assert tf32x3.HYBRID is None and not tf32x3.HYBRID_GPUS


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
