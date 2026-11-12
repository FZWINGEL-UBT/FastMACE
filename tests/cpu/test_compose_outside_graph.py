"""compose-outside-graph (12 November 2026): COMPOSE_FAST's composed first-interaction weights are composed outside
the captured MD step. A bind inside a CUDA-graph capture keeps a table's cached weights when only the table's
version moved (the capture's snapshot restore copied the same values back); a new table or a changed parameter still
composes; the force field registers the weights as derived state beside the table, so a restore brings both back."""

import pytest

torch = pytest.importorskip("torch")

from fastmace.engine.registry import Registry  # noqa: E402
from fastmace.rewrites.mh1 import modules as mh1  # noqa: E402

from test_compose_skip import _composing  # noqa: E402


class _Counted:
    def __init__(self, reassociation, monkeypatch):
        self.calls = 0
        original = reassociation._compose_weights

        def counted(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(reassociation, "_compose_weights", counted)


def _capturing(monkeypatch, on):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: on)


def _bound(monkeypatch):
    reassociation, message = _composing()
    inline = message()  # the forward's own composition, before anything is bound
    table = reassociation.species_feats
    reassociation.compose_radial = message.radial
    counter = _Counted(reassociation, monkeypatch)
    _capturing(monkeypatch, False)
    reassociation.bind(table)
    assert counter.calls == 1
    return reassociation, message, inline, table, counter


def _weights(reassociation):
    return [reassociation._buffers[f"composed_{k}"] for k in range(6)]


def test_a_bind_inside_a_capture_keeps_the_cached_weights(monkeypatch):
    reassociation, message, inline, table, counter = _bound(monkeypatch)
    before = [w.clone() for w in _weights(reassociation)]
    held = _weights(reassociation)
    table.copy_(table.clone())  # a snapshot restore: the same values, a new version
    adopts = mh1.CAPTURE_ADOPTS()
    _capturing(monkeypatch, True)
    reassociation.bind(table)
    assert counter.calls == 1 and mh1.CAPTURE_ADOPTS() == adopts + 1  # nothing composed into the graph
    assert all(a is b for a, b in zip(_weights(reassociation), held))
    assert all(torch.equal(a, b) for a, b in zip(_weights(reassociation), before))
    assert torch.equal(message(), inline)  # bitwise the forward's own composition
    reassociation.bind(table)  # the version is adopted: a second bind is a plain hit
    assert counter.calls == 1 and mh1.CAPTURE_ADOPTS() == adopts + 1


def test_outside_a_capture_a_new_version_composes_again_in_place(monkeypatch):
    reassociation, message, inline, table, counter = _bound(monkeypatch)
    held = _weights(reassociation)
    table.copy_(table.clone())
    reassociation.bind(table)
    assert counter.calls == 2 and all(a is b for a, b in zip(_weights(reassociation), held))
    assert torch.equal(message(), inline)


@pytest.mark.parametrize("which", ["radial", "skip", "linear"])
def test_a_changed_parameter_composes_again_also_inside_a_capture(monkeypatch, which):
    reassociation, message, inline, table, counter = _bound(monkeypatch)
    held = _weights(reassociation)
    linear, _, skip, _ = reassociation.compose_skip
    tensor = {"radial": message.radial.weight, "skip": skip.ws1, "linear": linear.weight_1}[which]
    tensor.mul_(1.25)  # a reload or an optimiser step: in place, a new version
    adopts = mh1.CAPTURE_ADOPTS()
    _capturing(monkeypatch, True)
    reassociation.bind(table)
    assert counter.calls == 2 and mh1.CAPTURE_ADOPTS() == adopts
    assert all(a is b for a, b in zip(_weights(reassociation), held))  # same addresses, new values
    expected = mh1.FirstInteractionReassociation._compose_weights(reassociation, table, reassociation._radial_weight())
    assert all(torch.equal(a, b) for a, b in zip(_weights(reassociation), expected))


def test_a_new_table_composes_inside_a_capture(monkeypatch):
    reassociation, message, inline, table, counter = _bound(monkeypatch)
    _capturing(monkeypatch, True)
    other = table.clone() * 2
    reassociation.bind(other)
    assert counter.calls == 2 and reassociation._buffers["composed_0"] is reassociation.composed_buffers(other)[0]


def test_registry_restores_the_weights_with_the_table(monkeypatch):
    """A rejected transaction (or a capture's warm-up): snapshot, the table's refresh recomposes, restore. The
    weights come back with the table, bitwise, at the same addresses, and the next captured bind adopts them."""
    reassociation, message, inline, table, counter = _bound(monkeypatch)
    weights = reassociation.composed_buffers(table)
    assert len(weights) == 6 and all(a is b for a, b in zip(weights, _weights(reassociation)))
    sources = {"derived.table": table, **{f"derived.table.composed_{k}": w for k, w in enumerate(weights)}}
    registry = Registry(lambda: sources)
    saved = registry.snapshot()
    before = [w.clone() for w in weights]
    table.mul_(1.5)  # the trial's refresh (forces._setup_tables' refresh): the table, then its weights in place
    reassociation.refresh_composed(table)
    assert counter.calls == 2 and not torch.equal(weights[0], before[0])
    registry.restore(saved)
    assert all(torch.equal(a, b) for a, b in zip(weights, before))
    _capturing(monkeypatch, True)
    reassociation.bind(table)
    assert counter.calls == 2 and torch.equal(message(), inline)


def test_composed_buffers_of_an_unknown_table_are_empty():
    reassociation, _ = _composing()
    assert reassociation.composed_buffers(torch.zeros(3, 4)) == ()


def test_a_restored_species_plan_on_the_skip_composes_nothing(monkeypatch):
    """step-layout-cleanup: the force field binds the species plan onto the skip (``species_gemm.bind_plan``) and the
    capture's snapshot restore copies it back (new versions). The plan is no input of the composition, so the captured
    bind still adopts (the 2.15 attribution of c6511d8e showed the compose GEMMs back in the graph)."""
    from fastmace.kernels.triton.species_gemm import bind_plan

    reassociation, message, inline, table, counter = _bound(monkeypatch)
    skip = reassociation.compose_skip[2]
    kind = torch.tensor([0, 1, 2, 2, 1, 0, 2])
    order, counts = torch.argsort(kind, stable=True).to(torch.int32), torch.tensor([2, 2, 3], dtype=torch.int32)
    bind_plan(skip, (kind, order, counts))
    reassociation.bind(table)  # the plan's first binding is no parameter change either
    assert counter.calls == 1
    for tensor in (kind, order, counts, table):
        tensor.copy_(tensor.clone())  # the capture's snapshot restore: the same values, new versions
    _capturing(monkeypatch, True)
    reassociation.bind(table)
    assert counter.calls == 1
    assert torch.equal(message(), inline)
