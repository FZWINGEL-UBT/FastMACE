import fastmace as fm
import ase.build

pristine = fm.load("/home/fzwin/models/mace-mh-1.model", head="matpes_r2scan", trusted=True)
atoms = ase.build.bulk("NaCl", "rocksalt", 5.64).repeat((3, 3, 3))
sim = fm.Simulation(atoms, pristine, integrator=fm.integrators.Langevin(temperature_K=300, friction_per_fs=0.01), seed=42)
print("Simulation engine:", type(sim._engine))
engine = sim._engine
field = engine.force_field
print("Force field:", type(field))
model = field.model
for i, inter in enumerate(model.interactions):
    sk = getattr(inter, "streaming_kernels", None)
    if sk is not None:
        print(f"Interaction {i}: streaming_kernels={sk}, receiver_backward={sk.receiver_backward}")
