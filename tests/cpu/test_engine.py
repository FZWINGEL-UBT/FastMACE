"""The CUDA-graph engine's bookkeeping on the CPU (plan task 1a-T13), against ASE.

Ported from the research engine's ``tests/core/test_gpu_langevin.py`` (``TrajectoryTests``): a torch
Lennard-Jones force field on FastMACE's skin neighbour list, stepped by ``GraphEngine`` and by ASE
from the same state and seed. ``ReplayEngine`` runs the ``capture=True`` code path on the CPU:
"capturing" a stage only keeps it for replay. So the stage keys, padded edge sizes, rebuilds,
pipelined redos and early rebuilds, and speculative rollbacks are all exercised without a GPU.
The research engine's module constants are config fields here (``NeighbourConfig.edge_multiple``,
``edge_bucket_fraction``, ``EngineConfig.pipeline_rebuild_fraction``), so no test patches a module.

The process constraints run on fakes: ``ProcessRecord`` instances, fake allocator segments and a
fake dynamo config.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.md.langevin import Langevin as AseLangevin

from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.base import Snapshot
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.model.forces import ForceField, ForceOutput
from fastmace.neighbors.buckets import edge_bucket
from fastmace.neighbors.skin import SkinNeighbourList, det_3x3
from fastmace.state import SimulationState

SIGMA, EPSILON, RC = 3.4, 0.0104, 7.0
STEPS = 300


class TorchLJ:
    """ASE's non-smooth LennardJones (shifted energy, unshifted forces) as a ForceField; stress optional."""

    def __init__(self, atoms, neighbours, compute_stress=False):
        self.nl = SkinNeighbourList(atoms.cell.array, RC, "cpu", torch.float64, neighbours, sort_receivers=False,
                                    gpu_build=False, variable_cell=compute_stress)
        self._compute_stress = compute_stress
        self.calls = 0
        self.num_edges = 0
        positions = torch.tensor(atoms.get_positions())
        self.rebuild(positions)
        self.compact(positions)
        self.use_edges(int(self.nl.count))

    device = torch.device("cpu")
    dtype = torch.float64
    padding_exact = True
    compile_mode = "none"

    @property
    def compute_stress(self):
        return self._compute_stress

    @property
    def edge_count(self):
        return self.nl.count

    @property
    def edge_capacity(self):
        return self.nl.capacity

    def evaluate(self, atoms, stress=False):
        raise NotImplementedError

    def rebuild(self, positions):
        return self.nl.build(positions)

    def use_edges(self, num_edges):
        self.num_edges = num_edges

    def set_cell(self, cell):
        self.nl.set_cell(cell)

    def needs_rebuild(self, positions, fraction=1.0):
        return self.nl.needs_rebuild(positions, fraction)

    def compact(self, positions):
        self.nl.compact(positions)

    def buffers(self):
        return {f"neighbours.{name}": value for name, value in self.nl.buffers().items()}

    def info(self):
        return {"streaming_interactions": 0, "sort_receivers": False, "neighbour_rebuilds": self.nl.rebuilds}

    def __call__(self, positions):
        self.calls += 1
        x = positions.detach().clone().requires_grad_(True)
        sender, receiver = self.nl.edge_index[:, :self.num_edges]
        shifts = self.nl.shifts[:self.num_edges]
        strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=self._compute_stress)
        eye = torch.eye(3, dtype=torch.float64)
        vectors = (x[receiver] - x[sender] + shifts) @ (eye + strain)
        r = torch.linalg.norm(vectors, dim=1)
        inside = r < RC
        r = torch.where(inside, r, torch.full_like(r, RC))  # padded/outer edges -> 0
        pair = 4 * EPSILON * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)
        shift = 4 * EPSILON * ((SIGMA / RC) ** 12 - (SIGMA / RC) ** 6)
        energy = 0.5 * torch.where(inside, pair - shift, torch.zeros_like(r)).sum()
        if not self._compute_stress:
            (gradient,) = torch.autograd.grad(energy, x)
            return ForceOutput(energy.detach().reshape(1), -gradient, None)
        gradient, virial = torch.autograd.grad(energy, (x, strain))
        stress = virial / det_3x3(self.nl.cell_t)
        return ForceOutput(energy.detach().reshape(1), -gradient, (stress + stress.T) / 2)


class ReplayEngine(GraphEngine):
    """The capture=True code path on the CPU: 'capturing' a stage only keeps it for replay."""

    def _capture(self, key):
        self._graphs[key] = SimpleNamespace(replay=self._stage(key))
        self.captures.append(key)


def argon(seed=3, displacement=0.05):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    atoms.rattle(displacement, seed=seed)
    return atoms


def build(*, engine=None, neighbours=None, replay=False, npt=False, noise_batch=7, seed=7, **integrator):
    """A test helper standing in for the planner (T15): force field, integrator, state and engine."""
    atoms = argon()
    neighbours = neighbours or NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off")
    engine = engine or EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu")
    field = TorchLJ(atoms, neighbours, compute_stress=npt)
    state = SimulationState.from_atoms(atoms, "cpu", seed, compute_stress=npt)
    if npt:
        config = IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=0.02, noise_batch=noise_batch,
                                  pressure_GPa=0.5, barostat_mass_factor=0.01, **integrator)
        dyn = LangevinBAOAB(state, config)
    else:
        config = IntegratorConfig(timestep_fs=2.0, friction=0.02, fixcm=True, noise_batch=noise_batch,
                                  **integrator)
        dyn = Langevin(state, config)
    cls = ReplayEngine if replay else GraphEngine
    record = constraints.ProcessRecord()
    out = cls(engine, field, dyn, state, neighbours=neighbours, capture=replay, constraints=record)
    if replay:
        out.captures = []
    return out


def run_pair(steps=STEPS, **options):
    """ASE's Langevin and the engine from the same state with the same seed, heating to 150 K."""
    reference = argon()
    reference.calc = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)
    ase_dyn = AseLangevin(reference, timestep=2 * units.fs, temperature_K=0, friction=0.02, fixcm=True,
                          rng=np.random.default_rng(7))
    engine = build(**options)
    for step in range(1, steps + 1):
        target = 150.0 * step / steps
        ase_dyn.set_temperature(temperature_K=target)
        engine.integrator.set_temperature(target)
        ase_dyn.run(1)
        engine.step()
    values = engine.sync()
    return reference, engine, values


def assert_matches(reference, values, engine):
    np.testing.assert_allclose(values["positions"], reference.positions, atol=1e-9)
    np.testing.assert_allclose(values["momenta"], reference.get_momenta(), atol=1e-9 * 40)
    assert engine.potential_energy() == pytest.approx(reference.get_potential_energy(), abs=1e-8)


# --------------------------------------------------------------------------- trajectories against ASE


def test_eager_steps_match_ase():
    reference, engine, values = run_pair()
    assert engine.force_field.nl.rebuilds > 1
    assert engine.force_field.calls == STEPS + 1  # one force call per step, plus the initial one
    assert engine.state.step == STEPS
    assert_matches(reference, values, engine)
    assert reference.get_temperature() > 50


@pytest.mark.parametrize("batch", [1, 7, 1000])
def test_replayed_stages_match_ase(batch):
    """The capture path (AC, then C again and B at the padded size on rebuild steps), any noise batch."""
    reference, engine, values = run_pair(replay=True, noise_batch=batch)
    assert engine.force_field.nl.rebuilds > 1
    kinds = {key if isinstance(key, str) else key[0] for key in engine.captures}
    assert kinds == {"AC", "B", "C"}
    assert_matches(reference, values, engine)


def test_speculative_steps_match_ase():
    engine_config = EngineConfig(pipelined="off", speculative_graph=True, device="cpu")
    reference, engine, values = run_pair(replay=True, engine=engine_config)
    info = engine.info()
    assert info["mode"] == "speculative"
    assert info["speculative_launches"] == STEPS
    assert engine.state.step == STEPS
    assert not engine._pending
    assert engine.force_field.nl.rebuilds > 1
    assert engine.force_field.calls >= STEPS + 1
    assert_matches(reference, values, engine)


def test_speculative_rollback_on_bucket_change_matches_ase():
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off", edge_multiple=1,
                                 edge_bucket_fraction=0.0)
    engine_config = EngineConfig(pipelined="off", speculative_graph=True, device="cpu")
    reference, engine, values = run_pair(replay=True, engine=engine_config, neighbours=neighbours)
    info = engine.info()
    assert info["speculative_rollbacks"] > 0
    assert info["speculative_bucket_rollbacks"] + info["speculative_rebuild_rollbacks"] > 0
    assert engine.state.step == STEPS
    assert_matches(reference, values, engine)


@pytest.mark.parametrize("replay", [True, False])
def test_pipelined_steps_match_ase(replay):
    engine_config = EngineConfig(pipelined="on", device="cpu")
    reference, engine, values = run_pair(replay=replay, engine=engine_config)
    info = engine.info()
    assert info["mode"] == "pipelined"
    assert engine._pipe_pending is None  # flushed by sync
    assert engine.state.step == STEPS
    assert info["pipelined_early_rebuilds"] > 0
    assert engine.force_field.nl.rebuilds > 1
    assert_matches(reference, values, engine)


def test_pipelined_redo_on_hard_rebuild_and_bucket_growth_matches_ase():
    """Without early rebuilds or an edge margin, redos of B (rebuild and count growth) stay exact."""
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off", edge_multiple=1,
                                 edge_bucket_fraction=0.0)
    engine_config = EngineConfig(pipelined="on", pipeline_rebuild_fraction=1.0, pipeline_edge_margin=0.0,
                                 device="cpu")
    reference, engine, values = run_pair(replay=True, engine=engine_config, neighbours=neighbours)
    info = engine.info()
    assert info["pipelined_early_rebuilds"] == 0
    assert info["pipelined_rebuild_redos"] > 0
    assert info["pipelined_redos"] > info["pipelined_rebuild_redos"]  # count growth too
    assert engine.state.step == STEPS
    assert_matches(reference, values, engine)


def test_padded_buckets_match_ase():
    """Graph-path padding (fine and coarse buckets) leaves the trajectory unchanged."""
    # Argon's edge count only fluctuates by a few edges, so use 1-edge granularity: fraction 0
    # pads nothing (one size per distinct count) and 1% (46 edges) needs fewer sizes.
    sizes = {}
    for fraction in (0.0, 0.01):
        neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off", edge_multiple=1,
                                     edge_bucket_fraction=fraction)
        reference, engine, values = run_pair(replay=True, neighbours=neighbours)
        assert_matches(reference, values, engine)
        info = engine.info()
        assert info["edge_bucket"] == edge_bucket(info["initial_edges"], fraction, 1)
        for size in info["edge_sizes_used"]:
            assert size % info["edge_bucket"] == 0 or size == engine.force_field.edge_capacity
        sizes[fraction] = set(info["edge_sizes_used"])
    assert len(sizes[0.0]) > 1  # the edge count really changed
    assert len(sizes[0.01]) < len(sizes[0.0])


# --------------------------------------------------------------------------- modes agree with each other


def trajectory(engine, steps=120):
    rows = []
    for step in range(1, steps + 1):
        engine.integrator.set_temperature(2.0 * step)
        engine.step()
        if step % 30 == 0:
            rows.append({key: value.copy() for key, value in engine.sync().items()})
    return rows


def assert_same(rows, expected):
    for got, want in zip(rows, expected, strict=True):
        assert got.keys() == want.keys()
        for key in got:
            np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_npt_modes_agree(cell_mode):
    """BAOAB (variable cell): eager, replayed and pipelined steps give the same numbers."""
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off", edge_multiple=1,
                                 edge_bucket_fraction=0.0)
    eager = trajectory(build(npt=True, cell_mode=cell_mode, neighbours=neighbours))
    replayed = build(npt=True, cell_mode=cell_mode, neighbours=neighbours, replay=True)
    assert_same(trajectory(replayed), eager)
    assert {"stress", "cell"} <= set(eager[-1])
    assert not np.array_equal(eager[-1]["cell"], argon().cell.array)
    pipelined = build(npt=True, cell_mode=cell_mode, neighbours=neighbours, replay=True,
                      engine=EngineConfig(pipelined="on", pipeline_rebuild_fraction=1.0, device="cpu"))
    assert_same(trajectory(pipelined), eager)
    assert pipelined.force_field.nl.rebuilds > 1


def test_npt_initial_state_has_stress_and_cell_force():
    engine = build(npt=True)
    assert engine.state.stress is not None and bool(engine.state.stress.abs().sum() > 0)
    assert bool(engine.integrator.force_eps.abs().sum() > 0)  # integrator.start ran


# --------------------------------------------------------------------------- output


@pytest.mark.parametrize("mode", ["checked", "pipelined", "speculative"])
def test_snapshot_values_equal_sync_values(mode):
    engine_config = EngineConfig(pipelined="on" if mode == "pipelined" else "off",
                                 speculative_graph=mode == "speculative", device="cpu")
    engine = build(engine=engine_config, replay=True)
    engine.integrator.set_temperature(100.0)
    for step in range(1, 41):
        engine.step()
        if step % 10 == 0:
            snapshot = engine.snapshot()
            assert isinstance(snapshot, Snapshot) and snapshot.event is None and snapshot.step == step
            values = snapshot.wait()
            synced = engine.sync()
            assert set(values) == set(synced) == {"positions", "momenta", "forces", "energy", "kinetic"}
            for key in values:
                np.testing.assert_array_equal(values[key].numpy(), synced[key])


def test_snapshot_copies_do_not_follow_the_state():
    engine = build(replay=True)
    engine.integrator.set_temperature(100.0)
    engine.step()
    snapshot = engine.snapshot()
    before = snapshot.wait()["positions"].clone()
    engine.run(5)
    assert torch.equal(snapshot.wait()["positions"], before)
    assert not torch.equal(engine.state.positions, before)


def test_run_flushes():
    engine = build(replay=True, engine=EngineConfig(pipelined="on", device="cpu"))
    engine.integrator.set_temperature(50.0)
    engine.run(10)
    assert engine._pipe_pending is None and engine.state.step == 10


# --------------------------------------------------------------------------- warm-up state and refusals


def test_runtime_state_covers_every_writer():
    """The tensors capture warm-ups save: integrator, force field, state results and engine buffers."""
    engine = build(replay=True, npt=True, engine=EngineConfig(pipelined="on", device="cpu"))
    names = set(engine.runtime_state())
    assert {"state.forces", "state.energy", "state.stress", "integrator.positions", "integrator.velocities",
            "integrator.p_eps", "integrator.force_eps", "integrator.cell", "force_field.neighbours.edge_index",
            "force_field.neighbours.count", "force_field.neighbours.cell", "engine.rebuild_needed",
            "engine.flags", "engine.pipe_flags", "engine.spec_flags", "engine.mid.0", "engine.mid.1"} <= names
    # one tensor appears once even when two owners name it (state.positions is the integrator's too)
    pointers = [tensor.data_ptr() for tensor in engine.runtime_state().values()]
    assert len(pointers) == len(set(pointers))


def test_warmups_restore_the_runtime_state():
    """GraphEngine._warm_up runs a stage three times and puts every runtime tensor back (here on the CPU)."""
    engine = build(npt=True, engine=EngineConfig(pipelined="on", device="cpu"))
    engine.integrator.set_temperature(100.0)
    engine.run(3)
    engine.integrator.prepare_step(engine.state)
    before = {name: tensor.clone() for name, tensor in engine.runtime_state().items()}
    for key in ("ACP", "AC", "C", ("B", engine.force_field.num_edges)):
        engine._warm_up(key)
        for name, tensor in engine.runtime_state().items():
            assert torch.equal(tensor, before[name]), (key, name)


def test_refusals():
    with pytest.raises(ValueError, match="pipelined"):
        build(engine=EngineConfig(pipelined="auto", device="cpu"))
    with pytest.raises(ValueError, match="NPT|variable"):
        build(npt=True, engine=EngineConfig(pipelined="off", speculative_graph=True, device="cpu"))
    with pytest.raises(ValueError, match="speculative"):
        build(engine=EngineConfig(pipelined="on", speculative_graph=True, device="cpu"))


def test_padding_must_be_exact_for_graphs():
    atoms = argon()
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off")
    field = TorchLJ(atoms, neighbours)
    field.padding_exact = False
    state = SimulationState.from_atoms(atoms, "cpu", 1)
    dyn = Langevin(state, IntegratorConfig())
    config = EngineConfig(pipelined="off", device="cpu")
    with pytest.raises(ValueError, match="padding"):
        ReplayEngine(config, field, dyn, state, capture=True, constraints=constraints.ProcessRecord())
    GraphEngine(config, field, dyn, state, capture=False, pad_edges=False, constraints=constraints.ProcessRecord())


def test_fake_force_field_satisfies_the_protocol():
    assert isinstance(build().force_field, ForceField)


# --------------------------------------------------------------------------- process constraints on fakes


def fake_field(backend="cueq", compile_mode="default", **extra):
    return SimpleNamespace(backend=backend, oeq=backend == "hybrid", compile_mode=compile_mode, **extra)


def test_compile_family():
    assert constraints.compile_family(fake_field("hybrid")) == constraints.OEQ_FAMILY
    assert constraints.compile_family(fake_field("cueq")) == constraints.AUTOGRAD_FAMILY
    assert constraints.compile_family(fake_field("e3nn")) == constraints.AUTOGRAD_FAMILY
    assert constraints.compile_family(fake_field("hybrid", compile_mode="none")) is None


def test_one_compile_family_per_process():
    record = constraints.ProcessRecord()
    record.claim_family(None, "eager cuEq")  # uncompiled: nothing fixed
    record.claim_family(constraints.OEQ_FAMILY, "hybrid engine")
    record.claim_family(constraints.OEQ_FAMILY, "second hybrid engine")
    record.claim_family(None, "eager")
    with pytest.raises(constraints.ProcessConstraintError, match="process"):
        record.claim_family(constraints.AUTOGRAD_FAMILY, "cuEq engine")
    assert record.family == constraints.OEQ_FAMILY and record.family_owner == "hybrid engine"


def test_check_engine_claims_the_family_last():
    record = constraints.ProcessRecord()
    dynamo = SimpleNamespace(cache_size_limit=8)
    config = EngineConfig(expandable_segments="on")
    bad = fake_field("hybrid", is_polar=True, dynamo_reset=False,
                     info=lambda: {"streaming_interactions": 0})
    with pytest.raises(constraints.ProcessConstraintError, match="PolarMACE"):
        constraints.check_engine(config, bad, record=record, segments=None, dynamo_config=dynamo)
    assert record.family is None and record.engines == 0
    good = fake_field("hybrid", is_polar=True, dynamo_reset=True, info=lambda: {"streaming_interactions": 0})
    facts = constraints.check_engine(config, good, record=record, segments=None, dynamo_config=dynamo)
    assert facts == {"allocator": "cpu", "dynamo_cache_size_limit": 64, "compile_family": constraints.OEQ_FAMILY}
    plain = fake_field("cueq", info=lambda: {"streaming_interactions": 0})
    with pytest.raises(constraints.ProcessConstraintError):
        constraints.check_engine(config, plain, record=record, segments=None, dynamo_config=dynamo)
    assert record.engines == 1


def test_dynamo_cache_limit_is_raised_never_lowered():
    dynamo = SimpleNamespace(cache_size_limit=8)
    assert constraints.ensure_dynamo_cache_limit(EngineConfig(), dynamo) == 64
    dynamo.cache_size_limit = 200
    assert constraints.ensure_dynamo_cache_limit(EngineConfig(dynamo_cache_size_limit=100), dynamo) == 200


def test_allocator_must_be_set_before_the_first_allocation():
    on = EngineConfig(expandable_segments="on")
    environ = {"PYTORCH_CUDA_ALLOC_CONF": constraints.EXPANDABLE}
    expandable, fixed = {"is_expandable": True}, {"is_expandable": False}
    assert constraints.check_allocator(on, [expandable, expandable], environ) == constraints.EXPANDABLE
    with pytest.raises(constraints.ProcessConstraintError, match="after the first CUDA allocation"):
        constraints.check_allocator(on, [expandable, fixed], environ)  # set too late
    with pytest.raises(constraints.ProcessConstraintError, match="not set"):
        constraints.check_allocator(on, [fixed], {})
    assert constraints.check_allocator(on, None, {}) == "cpu"
    assert constraints.check_allocator(EngineConfig(expandable_segments="off"), [fixed], {}) == "default"
    auto = EngineConfig(expandable_segments="auto")
    with pytest.raises(constraints.ProcessConstraintError):
        constraints.check_allocator(auto, [fixed], {}, torch_version="2.14.1+cu130")
    assert constraints.check_allocator(auto, [fixed], {}, torch_version="2.13.0") == "default"
    # a user's own allocator setting is left alone ("unless alloc conf set")
    assert constraints.check_allocator(auto, [fixed], {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
                                       torch_version="2.14.1") == "default"


class _SortedField:
    def __init__(self, receivers, streaming=2, sort=True):
        self.receivers = torch.tensor(receivers)
        self.streaming, self.sort = streaming, sort

    def info(self):
        return {"streaming_interactions": self.streaming, "sort_receivers": self.sort}

    @property
    def edge_count(self):
        return torch.tensor(len(self.receivers) - 1)  # the last edge is padding

    def buffers(self):
        return {"neighbours.edge_index": torch.stack([torch.zeros_like(self.receivers), self.receivers])}


def test_streaming_needs_receiver_sorted_edges():
    constraints.check_receiver_sorting(_SortedField([0, 0, 1, 2, 2, 0]))
    constraints.check_receiver_sorting(_SortedField([2, 0, 1, 0], streaming=0, sort=False))
    with pytest.raises(constraints.ProcessConstraintError, match="sort"):
        constraints.check_receiver_sorting(_SortedField([0, 1, 2, 0], sort=False))
    with pytest.raises(constraints.ProcessConstraintError, match="not sorted"):
        constraints.check_receiver_sorting(_SortedField([0, 2, 1, 0]))


def test_capture_safe_mask_head_is_put_back_after_another_install():
    """The research engine installs its own mask_head before every capture; FastMACE puts its own back.

    Compiled models guard on ``mace.modules.blocks.mask_head``. With another function in place, the
    warm-up of the next capture (a new edge bucket) compiled the model again (1a-T13b).
    """
    pytest.importorskip("mace")
    from mace.modules import blocks

    from fastmace import _patches
    from fastmace.rewrites import engine as engine_rewrites

    record = _patches.set_capture_safe_mask_head(True)
    assert blocks.mask_head is engine_rewrites.capture_safe_mask_head

    def other_copy(*args, **kwargs):  # what the research engine's use_capture_safe_mask_head installs
        return engine_rewrites.capture_safe_mask_head(*args, **kwargs)

    blocks.mask_head = other_copy
    assert _patches.set_capture_safe_mask_head(True) is record
    assert blocks.mask_head is engine_rewrites.capture_safe_mask_head
