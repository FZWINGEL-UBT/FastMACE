"""CPU screen of broad-band radial subspaces using cached OMAT-0 interaction-0 polynomials.

This evaluates the existing float32 coefficient table in float64, not the original network.
Sampled errors here are screening quantities, not production certificates.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

torch.set_num_threads(1)
source=Path.home()/'.cache/fastmace/radial-tables/c697fb752157aea2cccd6085ebe9843e283d630c4388f8e419a4c3e6fcebff11.pt'
payload=torch.load(source,map_location='cpu',weights_only=True)
tables=payload['tables']
C=tables['coef'].numpy().astype(np.float64)[...,:64]
DC=tables['dcoef'].numpy().astype(np.float64)[...,:64]
starts=tables['starts'].numpy().astype(np.float64)
scales=tables['scales'].numpy().astype(np.float64)
bins=tables['bins'].numpy()
lo,hi=float(tables['r_lo']),float(tables['r_max'])

def evaluate(pair,r,derivative=False):
    idx=np.clip(((r-lo)*tables['bin_scale']).astype(np.int64),0,len(bins)-1)
    piece=bins[idx]
    x=(r-starts[piece])*scales[piece]-1
    cc=(DC if derivative else C)[pair,piece]
    b1=np.zeros((len(r),64)); b2=np.zeros_like(b1)
    for k in range(cc.shape[1]-1,0,-1):
        b1,b2=cc[:,k]+2*x[:,None]*b1-b2,b1
    return cc[:,0]+x[:,None]*b1-b2

fit_r=np.linspace(lo,hi,8193,endpoint=False)
test_r=lo+(np.arange(8209)+0.38196601125)/8209*(hi-lo)
fit_v=[evaluate(p,fit_r) for p in range(9)]
fit_d=[evaluate(p,fit_r,True) for p in range(9)]
test_v=[evaluate(p,test_r) for p in range(9)]
test_d=[evaluate(p,test_r,True) for p in range(9)]
vscale=[np.maximum(np.abs(v).max(axis=0),1e-20) for v in fit_v]
dscale=[np.maximum(np.abs(v).max(axis=0),1e-20) for v in fit_d]
tols=(1e-7,3e-7,1e-6)
result={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'table_shape':list(C.shape),'domain':[lo,hi],'fit_points_per_pair':len(fit_r),
        'test_points_per_pair':len(test_r),'errors':'max error per column / its global sampled maximum, separately for values and slopes',
        'scope':'Cache-polynomial subspace screen; not original-network certificate or force gate','partitions':{}}

for band_count in (1,2,4,8):
    edges=np.linspace(lo,hi,band_count+1)
    pair_rows=[]
    for p in range(9):
        bands=[]
        for a,b in zip(edges[:-1],edges[1:]):
            fmask=(fit_r>=a)&(fit_r<b); tmask=(test_r>=a)&(test_r<b)
            v=fit_v[p][fmask]/vscale[p]
            d=fit_d[p][fmask]/vscale[p]
            gamma=np.linalg.norm(v)/max(np.linalg.norm(d),1e-300)
            _,sv,basis=np.linalg.svd(np.vstack((v,gamma*d)),full_matrices=False)
            vv=test_v[p][tmask]/vscale[p]; dd=test_d[p][tmask]/vscale[p]
            zv=vv@basis.T; zd=dd@basis.T
            curve=[]
            for rank in range(4,65,4):
                ev=np.max(np.abs(zv[:,:rank]@basis[:rank]-vv))
                ed=np.max(np.abs((zd[:,:rank]@basis[:rank]-dd)*vscale[p]/dscale[p]))
                curve.append({'rank':rank,'value_error':float(ev),'slope_error':float(ed)})
            required={str(tol):next(row['rank'] for row in curve if max(row['value_error'],row['slope_error'])<=tol) for tol in tols}
            bands.append({'range':[float(a),float(b)],'ranks':required,'curve':curve})
        pair_rows.append(bands)
    summaries={}
    for tol in tols:
        ranks=np.array([[band['ranks'][str(tol)] for band in bands] for bands in pair_rows])
        padded8=((ranks.max(axis=0)+7)//8)*8
        padded_pow2=np.array([2**int(np.ceil(np.log2(x))) for x in ranks.max(axis=0)])
        summaries[str(tol)]={'ranks_min':ranks.min(axis=0).tolist(),'ranks_max':ranks.max(axis=0).tolist(),
                             'padded8_per_band':padded8.tolist(),'total_padded8_node_width':int(padded8.sum()),
                             'padded_power2_per_band':padded_pow2.tolist(),'total_power2_node_width':int(padded_pow2.sum()),
                             'uniform_r_mean_padded8_edge_width':float(padded8.mean())}
    result['partitions'][str(band_count)]={'edges':edges.tolist(),'summaries':summaries,'pairs':pair_rows}
    print(json.dumps({'bands':band_count,'summaries':summaries}),flush=True)

Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
