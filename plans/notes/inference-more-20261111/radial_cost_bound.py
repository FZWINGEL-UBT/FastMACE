"""Geometry counts and optimistic edge-work bounds only, CPU; no timings."""
from pathlib import Path
import json
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

root=Path('/mnt/d/Margraf/FastMACE')
records=[]
for sigma in (0.05,0.08,0.3):
    atoms=read(root/'results/loup-a100-20261103/SrTiO3.cif').repeat((10,10,10))
    atoms.positions+=np.random.default_rng(0).normal(0,sigma,atoms.positions.shape)
    r=neighbor_list('d',atoms,6.0)
    far=float(np.mean(r>=3.15))
    saved=far/2  # 64 coordinates below the cutoff; 32 at and above.
    row={'sigma_angstrom_per_axis':sigma,'atoms':len(atoms),'edges':len(r),
         'fraction_r_ge_3_15':far,'ideal_fraction_first_edge_work_removed':saved,
         'optimistic_h200_microseconds':saved*(211+111.8),
         'optimistic_a100_microseconds':saved*(444+399.7),
         'optimistic_h200_force_kernel_fraction':saved*(211+111.8)/3257.6,
         'optimistic_a100_force_kernel_fraction':saved*(444+399.7)/6511.7}
    records.append(row); print(json.dumps(row),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps({'scope':'Upper bound: assumes edge kernels scale with active width, excludes partitioning, transforms, fixed work; not an expected speedup',
      'records':records},indent=2)+'\n')
