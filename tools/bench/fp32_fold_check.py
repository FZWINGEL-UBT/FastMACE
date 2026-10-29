"""Idea 7 check (CPU, float64, stock MACE MH-1): fold products[0].linear and products[1].linear into their consumers.

Facts checked first: the first skip connection sc0 is the same vector for every atom of a species and has no l=1
part; the Linear layers have no bias. Then the folded model replaces
  products[0].linear (512x0e+512x1o -> same)       by the identity (product 0 returns its contraction output y0),
  interactions[1].linear_up, interactions[1].skip_tp, readouts[0].linear
                                                     by the products with P0, plus sc0's image as a per-atom bias,
  products[1].linear (512 -> 512) and readouts[1].linear_1 (512 -> 96)
                                                     by P1 R1 (512 -> 96), the skip by SK1 R1 (96 wide),
and compares energy and forces with the stock model.
"""
import sys

import numpy as np
import torch
from ase.build import bulk
from mace.calculators import MACECalculator

torch.set_num_threads(4)
model_path = sys.argv[1]


def calculator():
    return MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64", head="matpes_r2scan")


atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
atoms.symbols[[1, 6, 9]] = "Nb"  # a third species
rng = np.random.default_rng(1)
atoms.positions += rng.normal(scale=0.08, size=atoms.positions.shape)

calc = calculator()
model = calc.models[0]
captured = {}
model.interactions[0].skip_tp.register_forward_hook(lambda m, i, o: captured.__setitem__("sc0", o.detach()))
atoms.calc = calc
e_ref, f_ref = atoms.get_potential_energy(), atoms.get_forces()
sc0 = captured["sc0"]
z = atoms.get_atomic_numbers()
spread = max(float((sc0[z == s] - sc0[z == s][0]).abs().max()) for s in set(z))
print(f"sc0 {tuple(sc0.shape)}: largest spread within a species {spread:.2e}; "
      f"l=1 part max {float(sc0[:, 512:].abs().max()):.2e}")


def matrix(module, width):
    eye = torch.eye(width, dtype=torch.float64)
    zero = module(torch.zeros(1, width, dtype=torch.float64))
    assert float(zero.abs().max()) == 0.0, f"{module} has a bias"
    return module(eye).detach()


class Fixed(torch.nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight, self.bias = weight, bias

    def forward(self, x):
        y = x @ self.weight
        return y if self.bias is None else y + self.bias


it1, p0, p1, ro0, ro1 = model.interactions[1], model.products[0], model.products[1], model.readouts[0], model.readouts[1]
P0, UP1, SK1 = matrix(p0.linear, 2048), matrix(it1.linear_up, 2048), matrix(it1.skip_tp, 2048)
RO0, P1, R1 = matrix(ro0.linear, 2048), matrix(p1.linear, 512), matrix(ro1.linear_1, 512)
print("shapes P0", tuple(P0.shape), "UP1", tuple(UP1.shape), "SK1", tuple(SK1.shape), "RO0", tuple(RO0.shape),
      "P1", tuple(P1.shape), "R1", tuple(R1.shape))
print("SK1 rows of the l=1 inputs, max", float(SK1[512:].abs().max()))

p0.linear = torch.nn.Identity()
p0.use_sc = False
it1.linear_up = Fixed(P0 @ UP1, sc0 @ UP1)
it1.skip_tp = Fixed(P0 @ SK1 @ R1, sc0 @ SK1 @ R1)
ro0.linear = Fixed(P0 @ RO0, sc0 @ RO0)
p1.linear = Fixed(P1 @ R1)
ro1.linear_1 = torch.nn.Identity()

atoms.calc = calc
calc.results = {}
e_fold, f_fold = atoms.get_potential_energy(), atoms.get_forces()
print(f"energy {e_ref:.10f} folded {e_fold:.10f} diff {e_fold - e_ref:.2e} eV")
print(f"forces max |diff| {np.abs(f_fold - f_ref).max():.2e} eV/A (max |F| {np.abs(f_ref).max():.3f})")
