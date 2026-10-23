"""FIRE (plan task 4-T06) on CUDA: one optimiser step captured with ``torch.cuda.graph`` replays the eager steps
bitwise.

The driver (4-T05) replays propose, the force call, ``dof.refresh`` and update. Here a harmonic torch force
field stands in for the model, so the check covers only the optimiser: no host synchronisation inside a step,
fixed addresses, and ASE's branches (reset, growth, clip, uphill) selected on the device.
"""

import pytest
import torch
from ase.build import bulk
from ase.constraints import FixAtoms

from fastmace import optimize
from fastmace.integrators.registry import build_dof, build_optimizer
from fastmace.state import SimulationState

STEPS = 40


def atoms(fixed=False):
    structure = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    structure.rattle(0.1, seed=6)
    if fixed:
        structure.set_constraint(FixAtoms(indices=[0, 7, 30]))
    return structure


class HarmonicForces:
    """An anisotropic tether, written into the state; graph-capturable. Forces are masked as ASE's
    ``atoms.get_forces()`` with ``FixAtoms``."""

    def __init__(self, state):
        self.x0 = state.positions.clone() + 0.05
        self.k = torch.linspace(0.2, 2.0, 3, dtype=torch.float64, device=state.positions.device)

    def __call__(self, state):
        d = state.positions - self.x0
        forces = -self.k * d
        state.forces.copy_(forces if state.free_mask is None else forces * state.free_mask)
        state.energy.copy_((0.5 * (self.k * d * d).sum()).reshape(1))


CASES = {
    "defaults": (dict(), False),
    "large-dt": (dict(dt=0.8, dtmax=4.0, Nmin=1, maxstep_A=0.04), False),
    "downhill": (dict(dt=1.0, dtmax=6.0, finc=1.5, Nmin=1, maxstep_A=0.3, downhill_check=True), False),
    "fix-atoms": (dict(dt=0.5, dtmax=2.0), True),
}


def build(name):
    options, fixed = CASES[name]
    state = SimulationState.from_atoms(atoms(fixed), "cuda", 0)
    config = optimize.FIRE(**options).to_config(fmax=1e-12)  # never converges within STEPS
    dof = build_dof(state, config)
    opt = build_optimizer(dof, config)
    field = HarmonicForces(state)
    field(state)
    opt.start()
    return state, dof, opt, field


def written(state, opt):
    return {**opt.buffers(), "positions": state.positions, "forces": state.forces, "energy": state.energy}


def capture(state, dof, opt, field):
    tensors = written(state, opt)
    saved = {key: value.clone() for key, value in tensors.items()}

    def step():
        opt.propose()
        field(state)
        dof.refresh()
        opt.update()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for key, value in tensors.items():
        value.copy_(saved[key])
    torch.cuda.synchronize()
    return graph


@pytest.mark.parametrize("name", list(CASES))
def test_captured_fire_step_replays_the_eager_steps(name):
    eager_state, eager_dof, eager_opt, eager_field = build(name)
    state, dof, opt, field = build(name)
    graph = capture(state, dof, opt, field)
    pointers = {key: value.data_ptr() for key, value in written(state, opt).items()}
    start = state.positions.clone()
    for _ in range(STEPS):
        eager_opt.propose()
        eager_field(eager_state)
        eager_dof.refresh()
        eager_opt.update()
        graph.replay()
    torch.cuda.synchronize()
    assert {key: value.data_ptr() for key, value in written(state, opt).items()} == pointers
    reference = written(eager_state, eager_opt)
    for key, value in written(state, opt).items():
        assert torch.equal(value, reference[key]), f"{key}: differs from the eager steps"
    assert not torch.equal(state.positions, start)
    assert bool(opt.started) and not bool(opt.converged)
    if CASES[name][1]:
        assert torch.equal(state.positions[[0, 7, 30]], start[[0, 7, 30]])
