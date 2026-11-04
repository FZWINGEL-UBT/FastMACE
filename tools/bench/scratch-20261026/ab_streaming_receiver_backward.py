import torch
import fastmace as fm
import ase.build

pristine = fm.load("/home/fzwin/models/mace-mh-1.model", head="matpes_r2scan", trusted=True)
atoms = ase.build.bulk("NaCl", "rocksalt", 5.64).repeat((3, 3, 3))
sim = fm.Simulation(atoms, pristine, integrator=fm.integrators.Langevin(temperature_K=300, friction_per_fs=0.01), seed=42)

engine = sim._engine
field = engine.force_field
model = field.model

streaming_kernels = [getattr(inter, "streaming_kernels", None) for inter in model.interactions]
active_kernels = [k for k in streaming_kernels if k is not None]
assert len(active_kernels) > 0, "No streaming kernels found"
print(f"Found {len(active_kernels)} active streaming kernel instances in model.")

positions = engine.state.positions.clone()

def run_forces(reps=40):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        field(positions)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        field(positions)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps

# 1. Edge-owned backward (default)
for k in active_kernels:
    k.receiver_backward = False

t_edge = run_forces(reps=50)
e_edge, f_edge, s_edge = field(positions)

# 2. Receiver-owned backward (tested)
for k in active_kernels:
    k.receiver_backward = True

t_recv = run_forces(reps=50)
e_recv, f_recv, s_recv = field(positions)

# Numerical check
maxdiff_f = (f_edge - f_recv).abs().max().item()
maxdiff_e = (e_edge - e_recv).abs().max().item()

print(f"\n=== NaCl 3x3x3 Full Force Evaluation (Hybrid FP32) ===")
print(f"Edge-owned backward (production baseline): {t_edge:8.2f} µs")
print(f"Receiver-owned backward (tested):          {t_recv:8.2f} µs")
print(f"Speedup:                                   {(t_edge / t_recv - 1.0) * 100.0:+6.2f} %")
print(f"Time saved per force call:                 {t_edge - t_recv:+8.2f} µs")
print(f"Force max absolute difference:             {maxdiff_f:8.2e} eV/Å")
print(f"Energy max absolute difference:            {maxdiff_e:8.2e} eV")
