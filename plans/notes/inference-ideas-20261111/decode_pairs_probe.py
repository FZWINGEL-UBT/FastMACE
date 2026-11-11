"""CPU-only combinatorial screen, production grid from saved evidence, no model execution."""
import json
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

root=Path('/mnt/d/Margraf/FastMACE')
grid=json.loads((root/'results/omat0-breakthroughs-assessment-20261108/table_ideas_bound.json').read_text())['certify']['grid']
rows=[]
for sigma in (0.05,0.08,0.3):
    atoms=read(root/'results/loup-a100-20261103/SrTiO3.cif').repeat((10,10,10))
    rng=np.random.default_rng(0)
    atoms.positions+=rng.normal(0,sigma,atoms.positions.shape)
    i,j,r=neighbor_list('ijd',atoms,6.0)
    order=np.lexsort((j,r,atoms.numbers[j],i))
    i,j,r=i[order],j[order],r[order].astype(np.float32)
    iv=np.zeros(len(r),dtype=np.int64); offset=0
    for k,(a,b,n) in enumerate(grid):
        mask=(r>=np.float32(a)) if k else np.ones(len(r),dtype=bool)
        u=np.maximum((r-np.float32(a))*np.float32(n/(b-a)),0)
        loc=np.clip(np.floor(u).astype(np.int64),0,n-1)
        iv[mask]=offset+loc[mask]; offset+=n
    starts=np.searchsorted(i,np.arange(len(atoms)+1))
    tile=(np.arange(len(i))-starts[i])//32
    def run_lengths(include_tile):
        change=(i[1:]!=i[:-1]) | (atoms.numbers[j[1:]]!=atoms.numbers[j[:-1]]) | (iv[1:]!=iv[:-1])
        if include_tile: change|=tile[1:]!=tile[:-1]
        return np.diff(np.r_[0,np.flatnonzero(change)+1,len(i)])
    groups=run_lengths(False); tiled=run_lengths(True)
    row={'sigma_angstrom_per_axis':sigma,'atoms':len(atoms),'edges':len(i),
         'mean_edges_per_piece_run':float(groups.mean()),
         'fraction_decodes_eliminated_pairing':float(np.sum(groups//2)/len(i)),
         'fraction_decodes_eliminated_pairing_inside_32edge_tiles':float(np.sum(tiled//2)/len(i)),
         'fraction_decodes_eliminated_unbounded_run_cache':float(1-len(groups)/len(i))}
    rows.append(row); print(json.dumps(row),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps({'grid_source':'FastMACE/results/omat0-breakthroughs-assessment-20261108/table_ideas_bound.json',
     'intervals':sum(g[2] for g in grid),'scope':'Synthetic displaced SrTiO3; reuse counts only, no timing or gate',
     'rows':rows},indent=2)+'\n')
