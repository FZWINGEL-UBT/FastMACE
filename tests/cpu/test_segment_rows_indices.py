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


def test_a_tensor_a_capture_read_survives_two_growth_steps(monkeypatch):
    """A captured graph reads the cached tensor without a reference to it: replaced by more rows, it is retired, not
    freed (else a later call with more rows frees memory the graph replays from). Growth at least doubles the rows."""
    import weakref

    from fastmace.kernels.cutlass import segment_rows

    monkeypatch.setattr(segment_rows, "_INDICES", {})
    monkeypatch.setattr(segment_rows, "_CAPTURED", set())
    monkeypatch.setattr(segment_rows, "_RETIRED", [])
    capturing = [False]
    monkeypatch.setattr(segment_rows, "_capturing", lambda view: capturing[0])
    base = torch.zeros(256, 3, 16)

    def cached():
        (tensor,) = segment_rows._INDICES.values()
        return tensor

    segment_rows._indices(base[:10], 16)
    first = cached()
    capturing[0] = True
    segment_rows._indices(base[:8], 16)  # the capture reads first's memory
    assert segment_rows._indices(base[:12], 16).numel() == 12 * 3  # a miss in a capture: built, not cached
    assert cached() is first
    capturing[0] = False
    first_ref = weakref.ref(first)
    del first
    segment_rows._indices(base[:12], 16)  # growth step 1: 12 rows -> 20
    second = cached()
    assert second.numel() == 20 * 3 and first_ref() is not None and segment_rows._RETIRED[0] is first_ref()
    capturing[0] = True
    segment_rows._indices(base[:20], 16)
    capturing[0] = False
    second_ref = weakref.ref(second)
    del second
    got = segment_rows._indices(base[:100], 16)  # growth step 2: 100 rows
    assert cached().numel() == 100 * 3 and got.numel() == 100 * 3
    assert [t is r() for t, r in zip(segment_rows._RETIRED, (first_ref, second_ref))] == [True, True]
    del got
    third_ref = weakref.ref(cached())
    segment_rows._indices(base[:250], 16)  # no capture read the 100-row tensor: freed, not retired
    assert len(segment_rows._RETIRED) == 2 and third_ref() is None
