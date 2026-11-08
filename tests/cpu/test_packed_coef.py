"""``tabled_outer_sum.PACKED_COEF`` (idea 3): the channel-pair packed series carry the table's values, laid out so
that two successive terms of a channel pair are one aligned 16-byte vector, and (where Triton imports) the packed
walks compile for sm_90 to half as many coefficient loads, each 16 bytes wide. CPU only: no CUDA device is used."""

import importlib.util
from pathlib import Path

import pytest
import torch

from fastmace.kernels.triton import tabled_outer_sum as tos

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _coef(pairs=3, pieces=5, terms=6, columns=66, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((pairs, pieces, terms, columns), generator=g, dtype=torch.float32)


def test_the_switch_is_off_by_default():
    assert tos.PACKED_COEF is None and len(tos.PACKED_COEF_GPUS) == 0
    assert not tos.packed_coef_default("NVIDIA H200")
    assert not tos.packed_coef_default("cpu")
    assert not tos.packed_coef_default(torch.device("cpu"))


def test_the_switch_follows_the_constant_and_the_list(monkeypatch):
    monkeypatch.setattr(tos, "PACKED_COEF", True)
    assert tos.packed_coef_default("cpu")
    monkeypatch.setattr(tos, "PACKED_COEF", False)
    monkeypatch.setattr(tos, "PACKED_COEF_GPUS", tos.GPUModels(("NVIDIA H200",)))
    assert not tos.packed_coef_default("NVIDIA H200")
    monkeypatch.setattr(tos, "PACKED_COEF", None)
    assert tos.packed_coef_default("NVIDIA H200") and tos.packed_coef_default("NVIDIA H200 NVL")
    assert not tos.packed_coef_default("cpu")


@pytest.mark.parametrize("columns", [66, 65])
def test_the_packed_table_holds_the_channels_in_pairs(columns):
    coef = _coef(columns=columns)
    p, pieces, terms, _ = coef.shape
    channels = columns - 1
    packed = tos.packed_coef(coef)
    pairs = (channels + 1) // 2
    assert packed.shape == (p, pieces, pairs, terms, 2) and packed.is_contiguous() and packed.dtype == coef.dtype
    for i in range(pairs):
        for c in range(2):
            j = 2 * i + c
            expected = coef[:, :, :, j] if j < channels else torch.zeros_like(coef[:, :, :, 0])
            assert torch.equal(packed[:, :, i, :, c], expected)  # bit for bit; the density column is left out
    # The memory order: per (pair, piece) block of pairs * terms * 2 floats, channel pair i at i * terms * 2, and
    # terms 2 s and 2 s + 1 of it as the four floats at 4 s: (t = 2 s, c = 0), (2 s, 1), (2 s + 1, 0), (2 s + 1, 1).
    flat = packed.reshape(-1)
    for q, piece, i, s in [(0, 0, 0, 0), (2, 4, pairs - 2, 2), (1, 3, 7, 1)]:
        start = ((q * pieces + piece) * pairs + i) * terms * 2 + 4 * s
        assert start % 4 == 0  # 16-byte aligned from an aligned base
        quad = flat[start:start + 4]
        expected = torch.stack([coef[q, piece, 2 * s, 2 * i], coef[q, piece, 2 * s, 2 * i + 1],
                                coef[q, piece, 2 * s + 1, 2 * i], coef[q, piece, 2 * s + 1, 2 * i + 1]])
        assert torch.equal(quad, expected)


def test_odd_terms_are_not_packed():
    assert tos.packed_coef(_coef(terms=5)) is None
    assert tos.packed_coef(None) is None


def test_packs_checks_the_shapes():
    coef = _coef()
    packed = tos.packed_coef(coef)
    assert tos._packs(packed, coef, 64)
    assert tos._packs(packed, coef, 62)
    assert not tos._packs(packed, coef, 63)  # odd columns: the unpacked rows
    assert not tos._packs(None, coef, 64)
    assert not tos._packs(packed, coef[:, :, :4].contiguous(), 64)  # other terms
    assert not tos._packs(packed, coef, 68)  # more channels than packed
    assert not tos._packs(packed.double(), coef, 64)


def _quads(packed, q, piece, s, k):
    """The kernel's view of one 16-byte load per channel pair: [i, (t, c)] -> (lo, hi) as [i, c] each."""
    block = packed[q, piece, : k // 2].reshape(k // 2, -1)[:, 4 * s:4 * s + 4]
    lo_hi = block.reshape(k // 2, 2, 2).permute(0, 2, 1)  # [i, c, t]
    return lo_hi[..., 0], lo_hi[..., 1]


@pytest.mark.parametrize("terms", [6, 4, 2])
def test_the_packed_recurrences_are_the_unpacked_ones_bit_for_bit(terms):
    """The kernel's packed Clenshaw and dual Horner (``_packed_fns``) mirrored in float32 PyTorch: the same terms in
    the same order as the unpacked walks, so the same bits."""
    coef = _coef(terms=terms)
    packed = tos.packed_coef(coef)
    k, half = 64, 32
    for q, piece, x in [(0, 0, 0.3125), (2, 4, -0.9), (1, 2, 0.77)]:
        x = torch.tensor(x, dtype=torch.float32)
        rows = coef[q, piece, :, :k]
        # unpacked Clenshaw (the forward and RECOMPUTE walks)
        v1 = torch.zeros(k)
        v2 = torch.zeros(k)
        for t in range(terms - 1, 0, -1):
            v1, v2 = rows[t] + 2.0 * x * v1 - v2, v1
        reference = rows[0] + x * v1 - v2
        # packed
        w1 = torch.zeros(half, 2)
        w2 = torch.zeros(half, 2)
        for s in range(terms // 2 - 1, -1, -1):
            lo, hi = _quads(packed, q, piece, s, k)
            w1, w2 = hi + 2.0 * x * w1 - w2, w1
            if s > 0:
                w1, w2 = lo + 2.0 * x * w1 - w2, w1
            else:
                last = lo
        assert torch.equal((last + x * w1 - w2).reshape(k), reference)
        # unpacked dual Horner (POWER_BACKWARD)
        pa, pd = rows[terms - 1].clone(), torch.zeros(k)
        for t in range(terms - 2, -1, -1):
            pd = pd * x + pa
            pa = pa * x + rows[t]
        # packed
        qd = torch.zeros(half, 2)
        for s in range(terms // 2 - 1, -1, -1):
            lo, hi = _quads(packed, q, piece, s, k)
            if s == terms // 2 - 1:
                qa = hi
            else:
                qd = qd * x + qa
                qa = qa * x + hi
            qd = qd * x + qa
            qa = qa * x + lo
        assert torch.equal(qa.reshape(k), pa) and torch.equal(qd.reshape(k), pd)


def _lab():
    spec = importlib.util.spec_from_file_location("_tool_packed_coef_lab", REPOSITORY_ROOT / "tools" /
                                                  "packed_coef_lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("walk", ["forward", "power", "recompute"])
def test_the_packed_walks_compile_for_sm90_to_half_the_coefficient_loads(walk, tmp_path, monkeypatch):
    """The falsifier (``tools/packed_coef_lab.py``): sm_90 PTX without a GPU. The unpacked walk reads the six terms'
    channels as six 8-byte vectors per lane; the packed walk as three 16-byte vectors and no 8-byte one."""
    pytest.importorskip("triton")
    pytest.importorskip("triton.backends.nvidia.compiler")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    lab = _lab()
    unpacked = lab.ptx_loads(lab.compile_walk(walk, False).asm["ptx"])
    packed = lab.ptx_loads(lab.compile_walk(walk, True).asm["ptx"])
    terms = 6
    assert packed["v4"] - unpacked["v4"] >= terms // 2, (unpacked, packed)
    assert unpacked["v2"] - packed["v2"] >= terms, (unpacked, packed)
