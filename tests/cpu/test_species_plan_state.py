"""step-layout-cleanup (12 November 2026): the species plan of the grouped products (``species_gemm.species_plan``:
the stable argsort of the atoms' species and their counts) is derived state of the species, built by the force field
outside the captured step and bound onto the modules, instead of rebuilt by every forward (the Ada attribution found
the radix sort replayed every step). A forward reads the bound plan only where its rows are the plan's; the plan is
refreshed with the species and restored with them; the outputs are bitwise those of the forward's own plan."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastmace.engine.registry import Registry  # noqa: E402
from fastmace.events import DerivedStateRegistry  # noqa: E402
from fastmace.kernels.triton import species_gemm  # noqa: E402
from fastmace.kernels.triton.species_gemm import bind_plan, bound_plan, plan_of, species_plan  # noqa: E402

from test_compose_skip import _composing  # noqa: E402


class _Sorts:
    """Counts ``species_plan`` calls (the forwards import it from ``species_gemm`` at call time)."""

    def __init__(self, monkeypatch):
        self.calls = 0
        original = species_gemm.species_plan

        def counted(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(species_gemm, "species_plan", counted)


def _attrs(kinds, columns=(0, 2, 3), width=5):
    attrs = torch.zeros(len(kinds), width, dtype=torch.float64)
    attrs[torch.arange(len(kinds)), torch.tensor(columns)[torch.tensor(kinds)]] = 1
    return attrs


@pytest.mark.parametrize("counting", [False, True])
def test_plan_of_is_the_forwards_plan(counting):
    attrs = _attrs([2, 0, 1, 2, 2, 0, 1, 1])
    columns = torch.tensor([0, 2, 3])
    kind, order, counts = plan_of(attrs, columns, counting=counting)
    assert torch.equal(kind, attrs[:, columns].argmax(dim=1))
    expected = species_plan(kind, 3)  # the radix sort; the counting sort gives the same (stable) order
    assert torch.equal(order, expected[0]) and torch.equal(counts, expected[1])
    assert order.dtype == torch.int32 and counts.tolist() == [2, 3, 3]


def test_bound_plan_covers_its_rows_only():
    module = torch.nn.Module()
    assert bound_plan(module, 8) is None
    plan = plan_of(_attrs([2, 0, 1, 2, 2, 0, 1, 1]), torch.tensor([0, 2, 3]))
    bind_plan(module, plan)
    assert all(a is b for a, b in zip(bound_plan(module, 8), plan))
    assert bound_plan(module, 7) is None  # a tile, or the real atoms without their ghosts: the forward's own
    assert not any(name in module.state_dict() for name in species_gemm.PLAN_BUFFERS)  # never saved
    bind_plan(module, None)
    assert bound_plan(module, 8) is None


def test_composed_forward_reads_the_bound_plan_bitwise(monkeypatch):
    reassociation, message = _composing()
    inline = message()
    sorts = _Sorts(monkeypatch)
    message()
    assert sorts.calls == 1  # the forward's own plan, every call
    skip = reassociation.compose_skip[2]
    case_attrs = torch.zeros(7, 5, dtype=torch.float64)  # test_compose_skip._case's species
    case_attrs[torch.arange(7), skip.columns[torch.tensor([0, 1, 2, 2, 1, 0, 2])]] = 1
    assert torch.equal(reassociation.plan_columns(), skip.columns)
    bind_plan(reassociation, plan_of(case_attrs, reassociation.plan_columns()))
    sorts.calls = 0
    assert torch.equal(message(), inline)  # bitwise
    assert sorts.calls == 0  # no sort in the forward any more


def test_plan_columns_of_the_species_skip(monkeypatch):
    from fastmace.rewrites.generic import modules as generic

    reassociation, _ = _composing()
    skip = reassociation.compose_skip[2]
    monkeypatch.setattr(generic, "SPECIES_GEMM", False)
    assert skip.plan_columns() is None  # no grouped products, no plan
    monkeypatch.setattr(generic, "SPECIES_GEMM", True)
    assert torch.equal(skip.plan_columns(), skip.columns)
    reassociation.compose_skip = None
    assert reassociation.plan_columns() is None


# --------------------------------------------------------------------------- the force field's derived state


@pytest.fixture
def generic_field(monkeypatch):
    pytest.importorskip("mace")
    from fastmace.rewrites.generic import modules as generic

    monkeypatch.setattr(generic, "SPECIES_GEMM", True)  # the grouped skip's plan also on the CPU
    import test_superset as superset

    calc, atoms = superset.calculator("residual"), superset.structure()
    return superset, calc, atoms, superset.field(calc, atoms, superset.GENERIC)


def _skips(ff):
    return [i.skip_tp for i in ff.model.interactions if type(i.skip_tp).__name__ == "SpeciesSkip"]


def test_the_force_field_binds_one_plan_per_column_set(generic_field):
    _, _, atoms, ff = generic_field
    skips = _skips(ff)
    assert skips and len(ff._plans) == 1  # every skip reads the superset's columns
    _, plan, modules = ff._plans[0]
    assert all(any(skip is m for m in modules) for skip in skips)
    assert all(all(a is b for a, b in zip(bound_plan(skip, len(atoms)), plan)) for skip in skips)
    expected = plan_of(ff.data["node_attrs"], skips[0].columns)
    assert all(torch.equal(a, b) for a, b in zip(plan, expected))
    names = ff.derived_buffers()
    assert names["species_plan.0.kind"] is plan[0] and names["species_plan.0.order"] is plan[1]
    assert names["species_plan.0.counts"] is plan[2]


def test_a_species_refresh_rebuilds_the_plan_in_place(generic_field):
    superset, calc, atoms, ff = generic_field
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    _, plan, _ = ff._plans[0]
    pointers = [t.data_ptr() for t in plan]
    before = [t.clone() for t in plan]
    target = superset.mutated(atoms)
    ff.atomic_numbers.copy_(torch.as_tensor(target.numbers))
    refreshed = registry.refresh(["species"])
    assert refreshed.index("force_field.node_attrs") < refreshed.index("force_field.species_plan.0.kind")  # the one-hot first
    assert [t.data_ptr() for t in plan] == pointers  # captured graphs keep their addresses
    assert not torch.equal(plan[1], before[1])
    fresh = superset.field(calc, target, superset.GENERIC)
    assert all(torch.equal(a, b) for a, b in zip(plan, fresh._plans[0][1]))  # equals a fresh construction
    fresh_registry = DerivedStateRegistry()
    fresh.register_derived(fresh_registry)
    assert registry.check_against(fresh_registry) == []
    for a, b in zip(superset.outputs(ff, target), superset.outputs(fresh, target)):
        assert torch.equal(a, b)


def test_a_snapshot_restore_brings_the_plan_back(generic_field):
    """A rejected transaction (or a capture's warm-up): snapshot, refresh, restore. The plan comes back with the
    one-hot, bitwise, at the same addresses; nothing keys on a version counter, so the capture's own restore (same
    values, new versions) changes nothing."""
    superset, _, atoms, ff = generic_field
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    sources = {f"derived.{name}": entry.buffer for name, entry in
               ((name, registry[name]) for name in registry.names())}
    engine_registry = Registry(lambda: sources)
    saved = engine_registry.snapshot()
    _, plan, _ = ff._plans[0]
    before = [t.clone() for t in plan]
    ff.atomic_numbers.copy_(torch.as_tensor(superset.mutated(atoms).numbers))
    registry.refresh(["species"])
    assert not torch.equal(plan[1], before[1])
    engine_registry.restore(saved)
    assert all(torch.equal(a, b) for a, b in zip(plan, before))


def test_the_tiled_bind_unbinds_the_plan(generic_field):
    _, _, atoms, ff = generic_field
    ff._bind_streaming_count(block_order=False, plans=False)
    assert all(bound_plan(skip, len(atoms)) is None for skip in _skips(ff))
    ff._bind_streaming_count()
    assert all(bound_plan(skip, len(atoms)) is not None for skip in _skips(ff))


def test_a_released_one_hot_has_no_plan(generic_field):
    _, _, atoms, ff = generic_field
    ff.release_node_attrs()
    ff._bind_streaming_count()
    assert all(bound_plan(skip, len(atoms)) is None for skip in _skips(ff))
    ff._refresh_plan(ff._plans[0][0], ff._plans[0][1])  # a registered refresh after the release: nothing to read
    np.testing.assert_array_equal(ff._plans[0][1][2].sum().item(), len(atoms))
