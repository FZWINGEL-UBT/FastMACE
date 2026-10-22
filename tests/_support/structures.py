"""Small crystal cells built in code, so tests need no structure files.

Each builder returns a fresh ``ase.Atoms`` with periodic boundaries; ``rattle``
displaces the atoms (Gaussian, standard deviation in Å) with a fixed seed.
"""

import numpy as np
from ase import Atoms
from ase.build import bulk

# NbCl5, 36 atoms (six Nb2Cl10 dimers): the primitive cell of the PBE benchmark
# structure, fractional coordinates rounded to four digits.
_NBCL5_CELLPAR = (13.678, 13.678, 6.413, 89.411, 89.411, 85.947)
_NBCL5_NB = [
    (0.4415, 0.2251, 0.4752), (0.7749, 0.5585, 0.5248), (0.5585, 0.7749, 0.5248),
    (0.2251, 0.4415, 0.4752), (0.1082, 0.8918, 0.0), (0.8918, 0.1082, 0.0),
]
_NBCL5_CL = [
    (0.5235, 0.3307, 0.2576), (0.6693, 0.4765, 0.7424), (0.4765, 0.6693, 0.7424),
    (0.3307, 0.5235, 0.2576), (0.5703, 0.2017, 0.6993), (0.7983, 0.4297, 0.3007),
    (0.8685, 0.2375, 0.7725), (0.7625, 0.1315, 0.2275), (0.1315, 0.7625, 0.2275),
    (0.2375, 0.8685, 0.7725), (0.9973, 0.1898, 0.2102), (0.8102, 0.0027, 0.7898),
    (0.0027, 0.8102, 0.7898), (0.1898, 0.9973, 0.2102), (0.0479, 0.0479, 0.7947),
    (0.9521, 0.9521, 0.2053), (0.2858, 0.2858, 0.2736), (0.7142, 0.7142, 0.7264),
    (0.0962, 0.4650, 0.2513), (0.5350, 0.9038, 0.7487), (0.9038, 0.5350, 0.7487),
    (0.4650, 0.0962, 0.2513), (0.1432, 0.3357, 0.6923), (0.6643, 0.8568, 0.3077),
    (0.8568, 0.6643, 0.3077), (0.3357, 0.1432, 0.6923), (0.3810, 0.3810, 0.6765),
    (0.6190, 0.6190, 0.3235), (0.2017, 0.5703, 0.6993), (0.4297, 0.7983, 0.3007),
]


# NaNbCl6, 32 atoms (Na4Nb4Cl24): the PBE benchmark structure of ResearchModule
# (Structures/Benchmark/PBE/NaNbCl6.cif), fractional coordinates rounded to four digits.
_NANBCL6_CELLPAR = (7.0777, 6.6566, 19.9267, 89.4949, 90.0, 90.0)
_NANBCL6 = [
    ("Na", (0.8112, 0.2112, 0.7475)), ("Na", (0.3112, 0.7888, 0.7525)), ("Na", (0.1888, 0.7888, 0.2525)),
    ("Na", (0.6888, 0.2112, 0.2475)), ("Nb", (0.7171, 0.7429, 0.6166)), ("Nb", (0.2171, 0.2571, 0.8834)),
    ("Nb", (0.2829, 0.2571, 0.3834)), ("Nb", (0.7829, 0.7429, 0.1166)), ("Cl", (0.8204, 0.6139, 0.2345)),
    ("Cl", (0.3204, 0.3861, 0.2655)), ("Cl", (0.1796, 0.3861, 0.7655)), ("Cl", (0.6796, 0.6139, 0.7345)),
    ("Cl", (0.6028, 0.4597, 0.0942)), ("Cl", (0.1028, 0.5403, 0.4058)), ("Cl", (0.3972, 0.5403, 0.9058)),
    ("Cl", (0.8972, 0.4597, 0.5942)), ("Cl", (0.5023, 0.9176, 0.1578)), ("Cl", (0.0023, 0.0824, 0.3422)),
    ("Cl", (0.4977, 0.0824, 0.8422)), ("Cl", (0.9977, 0.9176, 0.6578)), ("Cl", (0.7467, 0.8864, 0.0120)),
    ("Cl", (0.2467, 0.1136, 0.4880)), ("Cl", (0.2533, 0.1136, 0.9880)), ("Cl", (0.7533, 0.8864, 0.5120)),
    ("Cl", (0.9590, 0.0346, 0.1582)), ("Cl", (0.4590, 0.9654, 0.3418)), ("Cl", (0.0410, 0.9654, 0.8418)),
    ("Cl", (0.5410, 0.0346, 0.6582)), ("Cl", (0.0704, 0.5926, 0.0911)), ("Cl", (0.5704, 0.4074, 0.4089)),
    ("Cl", (0.9296, 0.4074, 0.9089)), ("Cl", (0.4296, 0.5926, 0.5911)),
]


def _rattled(atoms, rattle, seed):
    if rattle:
        atoms.rattle(rattle, seed=seed)
    return atoms


def nacl(repeat=2, rattle=0.0, seed=0):
    """Rock-salt NaCl, conventional cubic cell (8 atoms, a = 5.64 Å) repeated ``repeat`` times."""
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((repeat,) * 3)
    return _rattled(atoms, rattle, seed)


def nanbo3(repeat=2, rattle=0.0, seed=0):
    """Ideal cubic perovskite NaNbO3 (5 atoms, a = 3.95 Å) repeated ``repeat`` times."""
    a = 3.95
    atoms = Atoms("NaNbO3", cell=np.eye(3) * a, pbc=True,
                  scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)])
    return _rattled(atoms.repeat((repeat,) * 3), rattle, seed)


def nbcl5(repeat=1, rattle=0.0, seed=0):
    """Monoclinic NbCl5 (36-atom triclinic primitive cell) repeated ``repeat`` times."""
    atoms = Atoms(symbols=["Nb"] * len(_NBCL5_NB) + ["Cl"] * len(_NBCL5_CL), cell=_NBCL5_CELLPAR,
                  pbc=True, scaled_positions=_NBCL5_NB + _NBCL5_CL)
    return _rattled(atoms.repeat((repeat,) * 3), rattle, seed)


def nanbcl6(repeat=1, rattle=0.0, seed=0):
    """NaNbCl6 (32-atom near-orthorhombic cell) repeated ``repeat`` times."""
    atoms = Atoms(symbols=[symbol for symbol, _ in _NANBCL6], cell=_NANBCL6_CELLPAR, pbc=True,
                  scaled_positions=[position for _, position in _NANBCL6])
    return _rattled(atoms.repeat((repeat,) * 3), rattle, seed)


STRUCTURES = {"NaCl": nacl, "NaNbO3": nanbo3, "NbCl5": nbcl5}  # nanbcl6 is for the D3 tests only
