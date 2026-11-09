"""``cutlass.segment_rows._indices``: one cached index tensor per layout, whatever the row count (a tiled call meets
a different row count in almost every tile; keyed by it, the cache grew without bound on the device)."""

import pytest

torch = pytest.importorskip("torch")


def test_row_indices_are_a_prefix_of_one_cached_tensor_per_layout(monkeypatch):
    from fastmace.kernels.cutlass import segment_rows

    monkeypatch.setattr(segment_rows, "_INDICES", {})
    base = torch.zeros(64, 3, 16)

    def expected(view, ld):
        rows, size, _ = view.shape
        return torch.tensor([(a * view.stride(0) + j * view.stride(1)) // ld for a in range(rows) for j in range(size)],
                            dtype=torch.int32)

    for rows in (5, 2, 9, 9, 7, 30):
        view = base[:rows]
        got = segment_rows._indices(view, 16)
        assert torch.equal(got, expected(view, 16)), rows
    assert len(segment_rows._INDICES) == 1
    assert next(iter(segment_rows._INDICES.values())).numel() == 30 * 3
    other = base[:4, :2]  # another layout (size 2): its own entry
    assert torch.equal(segment_rows._indices(other, 16), expected(other, 16))
    assert len(segment_rows._INDICES) == 2
