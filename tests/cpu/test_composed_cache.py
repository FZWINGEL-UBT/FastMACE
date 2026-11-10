"""``FirstInteractionReassociation.refresh_composed``: entries of tables that are gone leave the cache when a new
table is bound (the tiled call binds one per tile; kept, their composed weights filled an A100 during a call)."""

import gc

import pytest

torch = pytest.importorskip("torch")


def test_composed_cache_keeps_live_tables_only(monkeypatch):
    from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

    module = FirstInteractionReassociation(torch.arange(3), 4, (0,), (1.0,), False)
    monkeypatch.setattr(module, "_radial_weight", lambda: None)
    monkeypatch.setattr(module, "_compose_weights", lambda table, weight: [table.clone() * 2])
    kept = torch.ones(3, 4)
    first = module.refresh_composed(kept)
    assert module.refresh_composed(kept) is first  # unchanged table: the same entry, no recompute
    for _ in range(5):
        table = torch.randn(3, 4)
        entry = module.refresh_composed(table)
        assert torch.equal(entry[2][0], table * 2)
        del table, entry
        gc.collect()
    cache = module.__dict__["_composed_cache"]
    assert len(cache) <= 2 and cache[id(kept)] is first  # the live table's entry stays (its addresses too)
    kept.add_(1)
    assert module.refresh_composed(kept) is first and torch.equal(first[2][0], kept * 2)
