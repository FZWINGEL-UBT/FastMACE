import faulthandler, sys
faulthandler.enable()
import fastmace  # noqa
import fastmace as fm
import torch
sys.path.insert(0, "tools/bench")
import tiled_capacity
from fastmace.config import NeighbourConfig
print("stage import", flush=True)
atoms = tiled_capacity.supercell("SrTiO3", (2, 2, 2), 0.05, 0.01)
loaded = fm.load("omat0-medium")
print("stage load", flush=True)
field = fm.ForceField(loaded, atoms, precision="float32")
print("stage forcefield", flush=True)
sim = fm.Simulation(atoms, field, integrator=fm.integrators.NVE(timestep_fs=1.0),
                    workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1, neighbours=NeighbourConfig(skin=0.5))
print("stage simulation", flush=True)
sim.run(steps=3)
torch.cuda.synchronize()
print("stage run", flush=True)
