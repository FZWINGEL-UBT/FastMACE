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


STRUCTURES = {"NaCl": nacl, "NaNbO3": nanbo3, "NbCl5": nbcl5}
