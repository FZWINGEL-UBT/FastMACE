"""The shared GEMM shape rule (fastmace/kernels/triton/tile_rule.py): a rule of M, N, K, the SM count and the L2
size only."""

from fastmace.kernels.triton import tile_rule as tr


def test_no_cuda_device_keeps_cublas():
    assert tr.segment_gemm_choice(13608, 512, 512, None).backend == "cublas"
    assert tr.segment_gemm_choice(13608, 512, 512, 0).backend == "cublas"


def test_threshold_is_in_waves_and_scales_with_the_sm_count():
    # murphy (48 SMs): 2 waves of 64-row tiles = 6144 rows; NbCl5's l = 2, 3 products qualify, l = 1 not.
    assert [tr.segment_gemm_choice(m, 512, 512, 48).backend for m in (1944, 5832, 9720, 13608)] == \
        ["cublas", "cublas", "triton", "triton"]
    # 36 SMs: 72 tiles (4545 rows and more); l = 1 (5832) qualifies too. NaCl (at most 1512 rows) never does.
    assert [tr.segment_gemm_choice(m, 512, 512, 36).backend for m in (1512, 4544, 4545, 5832)] == \
        ["cublas", "cublas", "triton", "triton"]


def test_one_fixed_configuration_and_a_reason():
    choice = tr.segment_gemm_choice(13608, 512, 512, 48)
    assert (choice.block_m, choice.block_n, choice.block_k, choice.num_warps, choice.num_stages) == (64, 64, 32, 4, 3)
    assert "waves" in choice.reason
    assert tr.segment_gemm_choice(13608, 512, 512, 48) == choice  # deterministic
    assert tr.row_waves(6144, 48) == 2.0


def test_large_tiles_only_on_the_measured_gpus(monkeypatch):
    # Decision 66: 128 rows and 8 warps where whole-step A/Bs gained in both build orders (the RTX 4000 Ada).
    assert tr.LARGE_TILE_GPUS == ("NVIDIA RTX 4000 Ada Generation",)
    choice = tr.segment_gemm_choice(13608, 512, 512, 48, "NVIDIA RTX 4000 Ada Generation")
    assert (choice.backend, choice.block_m, choice.num_warps) == ("triton", 128, 8)
    for other in ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", None):
        choice = tr.segment_gemm_choice(13608, 512, 512, 48, other)
        assert (choice.block_m, choice.num_warps) == (64, 4)
    # The waves rule stays in 64-row tiles: the backend does not depend on the tile shape.
    assert tr.segment_gemm_choice(5832, 512, 512, 48, "NVIDIA RTX 4000 Ada Generation").backend == "cublas"
    # An A/B side's override wins on every GPU.
    monkeypatch.setattr(tr, "TRITON_BLOCK_M", 64)
    monkeypatch.setattr(tr, "TRITON_NUM_WARPS", 4)
    choice = tr.segment_gemm_choice(13608, 512, 512, 48, "NVIDIA RTX 4000 Ada Generation")
    assert (choice.block_m, choice.num_warps) == (64, 4)


def test_column_tiles_run_together_only_when_a_exceeds_l2(monkeypatch):
    # MH-1 linear_2 at NbCl5 9,216 atoms on a 32 MiB L2: l = 0 (18.9 MB) fits, l = 1..3 do not.
    l2, gpu = 32 * 2**20, "NVIDIA GeForce RTX 5060 Ti"
    assert [tr.columns_together(9216 * d, 512, 4, l2, gpu) for d in (1, 3, 5, 7)] == [False, True, True, True]
    assert tr.columns_together(51840, 512, 4, l2, gpu)  # 106 MB: l = 0 at 51,840 atoms
    assert not tr.columns_together(51840, 512, 4, None, gpu)  # no L2 size known: the order before
    assert tr.columns_together(l2 // 2048 + 1, 512, 4, l2, gpu) and not tr.columns_together(l2 // 2048, 512, 4, l2, gpu)
    monkeypatch.setattr(tr, "TILE_ORDER", "rows")
    assert not tr.columns_together(51840, 512, 4, l2, gpu)
    monkeypatch.setattr(tr, "TILE_ORDER", "columns")
    assert tr.columns_together(16, 512, 4, l2, "NVIDIA A100-SXM4-80GB")


def test_column_tiles_together_only_on_the_measured_gpus():
    # Decision 66: the order is a default only where whole-step A/Bs measured a speed-up, by exact device name.
    l2 = 40 * 2**20
    assert set(tr.COLUMN_ORDER_GPUS) == {"NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation"}
    assert tr.columns_together(51840, 512, 4, l2, "NVIDIA RTX 4000 Ada Generation")
    for other in ("NVIDIA A100-SXM4-80GB", "NVIDIA L4", "NVIDIA GeForce RTX 4090", None):
        assert not tr.columns_together(51840, 512, 4, l2, other)


def test_no_cuda_device_has_no_l2_size():
    assert tr.device_l2_bytes("cpu") is None


def test_uses_triton_agrees_with_the_choice():
    for sms in (None, 0, 36, 48):
        for m in (1, 216, 1944, 4544, 4545, 5832, 6143, 6144, 13608):
            assert tr.uses_triton(m, sms) == (tr.segment_gemm_choice(m, 512, 512, sms).backend == "triton")
