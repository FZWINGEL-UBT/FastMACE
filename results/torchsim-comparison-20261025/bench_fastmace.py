"""FastMACE side: fm.Simulation, Langevin NVT, the planner's default plan (hybrid float32).

argv: label structure warmup steps blocks
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fastmace as fm  # noqa: E402  (before torch)
import torch  # noqa: E402

import common  # noqa: E402

label, name, warmup, steps, blocks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
atoms = common.structure(name)
t0 = time.perf_counter()
sim = fm.Simulation(atoms, fm.ForceField(fm.load(label), atoms),
                    integrator=fm.integrators.Langevin(timestep_fs=common.DT_FS, temperature_K=common.T_K,
                                                       friction_per_fs=common.FRICTION_PER_FS), seed=common.SEED)
setup = time.perf_counter() - t0
ms = common.timed(lambda n: sim.run(steps=n), warmup, steps, blocks, torch.cuda.synchronize)
common.report(dict(engine="fastmace", label=label, structure=name, natoms=len(atoms), setup_s=setup,
                   ms_per_step=ms, plan=sim.plan.describe().splitlines()[0], torch=torch.__version__,
                   peak_mib=torch.cuda.max_memory_allocated() / 2**20))
sim.close()
