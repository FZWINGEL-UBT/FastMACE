"""The shared GEMM shape rule (fastmace/kernels/triton/tile_rule.py): a rule of M, N, K and the SM count only."""

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


def test_uses_triton_agrees_with_the_choice():
    for sms in (None, 0, 36, 48):
        for m in (1, 216, 1944, 4544, 4545, 5832, 6143, 6144, 13608):
            assert tr.uses_triton(m, sms) == (tr.segment_gemm_choice(m, 512, 512, sms).backend == "triton")
