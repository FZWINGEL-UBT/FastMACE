"""TorchSim side: MACE's first-party MaceTorchSimModel, nvt_langevin (BAOAB) step loop, float32.

argv: label structure warmup steps blocks accel(cueq|oeq|none) compile(none|default|reduce-overhead) [n_systems]
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import torch  # noqa: E402
import torch_sim as ts  # noqa: E402
from torch_sim.integrators.nvt import nvt_langevin_init, nvt_langevin_step  # noqa: E402
from torch_sim.models.mace import MaceModel  # noqa: E402
from torch_sim.units import MetalUnits  # noqa: E402

import common  # noqa: E402

label, name, warmup, steps, blocks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
accel, compile_mode = sys.argv[6], sys.argv[7]
n_systems = int(sys.argv[8]) if len(sys.argv) > 8 else 1
checkpoint, head = common.CASES[label]
device, dtype = torch.device("cuda"), torch.float32
atoms = common.structure(name)
t0 = time.perf_counter()
model = MaceModel(model=str(common.MODELS / checkpoint), device=device, dtype=dtype, compute_stress=False,
                  enable_cueq=accel == "cueq", enable_oeq=accel == "oeq",
                  compile_mode=None if compile_mode == "none" else compile_mode, head=head)
state = ts.initialize_state([atoms] * n_systems, device=device, dtype=dtype)
dt = torch.tensor(common.DT_FS * 1e-3 * MetalUnits.time, device=device, dtype=dtype)
kT = torch.tensor(common.T_K * MetalUnits.temperature, device=device, dtype=dtype)
gamma = torch.tensor(common.FRICTION_PER_FS * 1e3 / MetalUnits.time, device=device, dtype=dtype)
state = nvt_langevin_init(state, model, kT=kT)
setup = time.perf_counter() - t0
box = [state]


def run(n):
    s = box[0]
    for _ in range(n):
        s = nvt_langevin_step(s, model, dt=dt, kT=kT, gamma=gamma)
    box[0] = s


ms = common.timed(run, warmup, steps, blocks, torch.cuda.synchronize)
s = box[0]
kinetic = 0.5 * (s.momenta.double() ** 2 / s.masses.double()[:, None]).sum()
temp = float(2 * kinetic / (3 * s.positions.shape[0] * MetalUnits.temperature))
common.report(dict(engine="torchsim", label=label, structure=name, natoms=len(atoms), n_systems=n_systems,
                   accel=accel, compile=compile_mode, setup_s=setup, ms_per_step=ms, final_T_K=temp,
                   torch=torch.__version__, torch_sim=ts.__version__,
                   peak_mib=torch.cuda.max_memory_allocated() / 2**20))
