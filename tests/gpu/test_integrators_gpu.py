"""The integrator half steps on CUDA: captured with ``torch.cuda.graph``, replays equal eager steps bitwise.

The engine's graphs (T13) record ``first_half`` in stage A and the force call plus ``second_half``
in stage B. Here a harmonic torch force field stands in for the model, so the check covers only
the integrators: no host synchronisation inside a half step, fixed addresses, device targets.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk

from fastmace.config import IntegratorConfig
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.neighbors.skin import det_3x3
from fastmace.state import SimulationState

STEPS = 20


def atoms():
    structure = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    structure.rattle(0.1, seed=6)
    structure.set_velocities(np.random.default_rng(1).standard_normal((len(structure), 3)) * 0.005)
    return structure


class HarmonicForces:
    """A tether and a volume spring, written into the state; graph-capturable."""

    def __init__(self, state, k=0.5, bulk_modulus=0.5):
        self.x0 = state.positions.clone()
        self.v0 = det_3x3(state.cell).clone()
        self.k, self.bulk_modulus = k, bulk_modulus
        self.eye = torch.eye(3, dtype=torch.float64, device=state.positions.device)

    def __call__(self, state):
        d = state.positions - self.x0
        state.forces.copy_(-self.k * d)
        state.energy.copy_((0.5 * self.k * (d * d).sum()).reshape(1))
        if state.stress is not None:
            volume = det_3x3(state.cell)
            state.stress.copy_(self.k * (d.T @ d) / volume + self.bulk_modulus * (volume / self.v0 - 1.0) * self.eye)


CASES = {
    "nvt": (Langevin, False, dict(ensemble="nvt")),
    "nvt-nofixcm": (Langevin, False, dict(ensemble="nvt", fixcm=False)),
    "npt-hydrostatic": (LangevinBAOAB, True, dict(ensemble="npt", cell_mode="hydrostatic")),
    "npt-full": (LangevinBAOAB, True, dict(ensemble="npt", cell_mode="full")),
    "npt-full-ase-trace": (LangevinBAOAB, True, dict(ensemble="npt", cell_mode="full",
                                                     ase_trace_drag_all_elements=True)),
}


def build(name):
    cls, stress, options = CASES[name]
    config = IntegratorConfig(timestep_fs=2.0, friction=0.02, noise_batch=7, pressure_GPa=1.5,
                              barostat_mass_factor=0.01, **options)
    state = SimulationState.from_atoms(atoms(), "cuda", 20261001, compute_stress=stress)
    dyn = cls(state, config)
    field = HarmonicForces(state)
    field(state)
    dyn.start(state)
    return state, dyn, field


def written(state, dyn):
    tensors = dict(dyn.buffers(state))
    tensors.update(forces=state.forces, energy=state.energy)
    if state.stress is not None:
        tensors["stress"] = state.stress
    return tensors


def capture(state, dyn, field):
    """Graphs A (first_half) and B (force call, second_half), warmed up without changing the state."""
    tensors = written(state, dyn)
    saved = {key: value.clone() for key, value in tensors.items()}
    stage_a = lambda: dyn.first_half(state)  # noqa: E731

    def stage_b():
        field(state)
        dyn.second_half(state)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            stage_a()
            stage_b()
    torch.cuda.current_stream().wait_stream(side)
    graphs = {}
    for key, stage in (("A", stage_a), ("B", stage_b)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage()
        graphs[key] = graph
    for key, value in tensors.items():
        value.copy_(saved[key])
    torch.cuda.synchronize()
    return graphs


@pytest.mark.parametrize("name", list(CASES))
def test_captured_half_steps_replay_the_eager_steps(name):
    eager_state, eager_dyn, eager_field = build(name)
    state, dyn, field = build(name)
    graphs = capture(state, dyn, field)
    pointers = {key: value.data_ptr() for key, value in written(state, dyn).items()}
    for step in range(1, STEPS + 1):
        for target in (eager_dyn, dyn):
            target.set_temperature(15.0 * step)
        eager_dyn.prepare_step(eager_state)
        eager_dyn.first_half(eager_state)
        eager_field(eager_state)
        eager_dyn.second_half(eager_state)
        dyn.prepare_step(state)
        graphs["A"].replay()
        graphs["B"].replay()
    torch.cuda.synchronize()
    assert {key: value.data_ptr() for key, value in written(state, dyn).items()} == pointers
    reference = written(eager_state, eager_dyn)
    for key, value in written(state, dyn).items():
        assert torch.equal(value, reference[key]), f"{key}: max |difference| {(value - reference[key]).abs().max()}"
    assert not torch.equal(state.positions, eager_field.x0)
    if dyn.variable_cell:
        assert not torch.equal(state.cell, torch.tensor(atoms().cell.array, device="cuda"))


# ------------------------------------------------------------- noise draw-ahead (round 6)

@pytest.mark.parametrize("draw_ahead", [True, False])
@pytest.mark.parametrize("which", ["langevin", "baoab"])
def test_drawn_ahead_noise_is_the_same_stream(monkeypatch, draw_ahead, which):
    """With NOISE_DRAW_AHEAD the batches after the first are drawn on a background thread into pinned buffers:
    the rows per step are still ASE's stream, bit for bit, also when the generator is read mid-run (a provenance
    record or a restart puts it back to before the draw-ahead, and the next refill draws the same rows again)."""
    from fastmace.integrators import langevin

    monkeypatch.setattr(langevin, "NOISE_DRAW_AHEAD", draw_ahead)
    seed, batch, steps = 20261105, 7, 60
    state = SimulationState.from_atoms(atoms(), "cuda", seed, compute_stress=which == "baoab")
    if which == "langevin":
        dyn = Langevin(state, IntegratorConfig(ensemble="nvt", timestep_fs=2.0, friction=0.02, noise_batch=batch,
                                               fixcm_projection="ase"))
    else:
        dyn = LangevinBAOAB(state, IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=0.02,
                                                    pressure_GPa=0.3, barostat_tau_fs=400.0, cell_mode="hydrostatic",
                                                    noise_batch=batch))
    width = state.rng.batch.shape[1:]
    reference = np.random.default_rng(seed)
    for step in range(steps):
        dyn.prepare_step(state)
        row = state.rng.batch[state.rng.cursor - 1].cpu().numpy()
        if which == "langevin":
            expected = reference.standard_normal(size=tuple(width))
        else:
            expected = reference.normal(size=tuple(width))
        np.testing.assert_array_equal(row, expected)
        if step in (10, 30):  # read the generator mid-batch (a record, a restart): the draw-ahead is undone
            state.rng.generator.bit_generator.state
            assert state.rng._ahead is None
        if step == 40 and draw_ahead:
            assert state.rng._ahead is not None  # the next batch is being drawn ahead
    torch.cuda.synchronize()
