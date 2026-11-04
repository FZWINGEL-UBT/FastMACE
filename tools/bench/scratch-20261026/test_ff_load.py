import fastmace as fm
import ase.io

pristine = fm.load("/home/fzwin/models/mace-mh-1.model", head="matpes_r2scan", trusted=True)
import ase.build
atoms = ase.build.bulk("NaCl", "rocksalt", 5.64).repeat((3, 3, 3))
ff = fm.ForceField(pristine, atoms=atoms)
em = ff.execution_model
print("Execution model:", type(em))
print("Interactions:", len(em.interactions))
int1 = em.interactions[1]
print("Interaction 1 streaming_kernels:", getattr(int1, "streaming_kernels", None))
