"""``skinny_linear.SPLIT_COLUMNS`` (A/B constant, Codex R1, off-polar-alex 7 November 2026): the folded input
gradient with a second grid axis over 16-column tiles of P, where the row grid alone would leave SMs idle.

- ``split_columns``: 16 when the split grid (row tiles x column tiles) still fits in one wave over the SMs and P
  has more than one tile, else 0 (the ordinary one-axis grid). A fixed rule from the kernel, no shape table;
- ``split_columns_default``: the constant when set, else ``SPLIT_COLUMNS_GPUS`` (empty until an MD A/B adopts a
  GPU); never off CUDA;
- exact: every output column is reduced in the same BLOCK_K order (tests/gpu/test_folded_tiles_cuda.py checks it
  bit for bit on the kernel).
"""

import pytest
import torch

from fastmace.kernels.triton import skinny_linear


@pytest.mark.parametrize(("rows", "p", "sms", "block_m", "expected"), [
    (512, 64, 36, 64, 16),     # MACE-OFF23-large's 512-row chunk on 36 SMs: 8 x 4 = 32 CTAs, one wave
    (576, 64, 36, 64, 16),     # 9 x 4 = 36: exactly one wave
    (577, 64, 36, 64, 0),      # 10 x 4 = 40: more than one wave
    (896, 64, 36, 64, 0),      # 14 x 4 = 56
    (1536, 64, 36, 64, 0),
    (35840, 64, 36, 64, 0),
    (1728, 64, 108, 64, 16),   # 27 x 4 = 108 on 108 SMs
    (512, 16, 36, 64, 0),      # one column tile: nothing to split
    (512, 64, None, 64, 0),    # not a CUDA device
    (0, 64, 36, 64, 0),        # no rows
    (100, 48, 36, 64, 16),     # 2 x 3 = 6
])
def test_split_columns_is_one_wave_of_the_split_grid(rows, p, sms, block_m, expected):
    assert skinny_linear.split_columns(rows, p, sms, block_m) == expected


def test_the_default_is_the_5060_ti_only_and_never_on_the_cpu(monkeypatch):
    assert skinny_linear.SPLIT_COLUMNS is None
    assert tuple(skinny_linear.SPLIT_COLUMNS_GPUS) == ("NVIDIA GeForce RTX 5060 Ti",)
    assert skinny_linear.split_columns_default("cpu") is False
    monkeypatch.setattr(skinny_linear, "SPLIT_COLUMNS", True)
    assert skinny_linear.split_columns_default("cpu") is False


@pytest.mark.parametrize(("constant", "listed", "expected"), [
    (None, False, False), (None, True, True), (True, False, True), (False, True, False)])
def test_the_constant_overrides_the_gpu_list(monkeypatch, constant, listed, expected):
    monkeypatch.setattr(skinny_linear, "SPLIT_COLUMNS", constant)
    monkeypatch.setattr(skinny_linear, "_split_listed", lambda index: listed)
    assert skinny_linear.split_columns_default(torch.device("cuda", 0)) is expected
