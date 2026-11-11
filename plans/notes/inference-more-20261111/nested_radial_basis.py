"""CPU-only nested radial basis screen; one 64-column node projection, band-dependent active prefixes."""
from pathlib import Path
import json
import numpy as np
import torch

torch.set_num_threads(1)
source=Path.home()/'.cache/fastmace/radial-tables/c697fb752157aea2cccd6085ebe9843e283d630c4388f8e419a4c3e6fcebff11.pt'
t=torch.load(source,map_location='cpu',weights_only=True)['tables']
c=t['coef'].numpy().astype(float)[...,:64]; dc=t['dcoef'].numpy().astype(float)[...,:64]
starts=t['starts'].numpy().astype(float); scales=t['scales'].numpy().astype(float); bins=t['bins'].numpy()
lo,hi=float(t['r_lo']),float(t['r_max'])

def ev(p,r,deriv=False):
    idx=np.clip(((r-lo)*t['bin_scale']).astype(np.int64),0,len(bins)-1)
    k=bins[idx]; x=(r-starts[k])*scales[k]-1
    coeff=(dc if deriv else c)[p,k]
    b1=np.zeros((len(r),64)); b2=np.zeros_like(b1)
    for j in range(coeff.shape[1]-1,0,-1): b1,b2=coeff[:,j]+2*x[:,None]*b1-b2,b1
    return coeff[:,0]+x[:,None]*b1-b2

rfit=np.linspace(lo,hi,8193,endpoint=False)
rtest=lo+(np.arange(8209)+0.38196601125)*(hi-lo)/8209
fv=[ev(p,rfit) for p in range(9)]; fd=[ev(p,rfit,True) for p in range(9)]
tv=[ev(p,rtest) for p in range(9)]; td=[ev(p,rtest,True) for p in range(9)]
physical=rfit>=1.0
vscale=[np.maximum(abs(v[physical]).max(0),1e-20) for v in fv]
dscale=[np.maximum(abs(v[physical]).max(0),1e-20) for v in fd]
vglobal=[np.maximum(abs(v).max(0),1e-20) for v in fv]
dglobal=[np.maximum(abs(v).max(0),1e-20) for v in fd]
configs=([lo,3.15,hi],[lo,2.0,4.0,hi],np.linspace(lo,hi,5).tolist())
result={'source':str(source),'scope':'Sampled cache-polynomial error, not force gate; one full 64-dimensional basis per pair',
        'fit_points':len(rfit),'test_points':len(rtest),'configs':[]}
for edges in configs:
    out={'edges':edges,'tolerances':{}}
    for tol in (1e-7,3e-7):
        pair_results=[]
        for p in range(9):
            # Exact endpoint value/slope directions in the smallest prefix make truncations meet at each boundary.
            boundary=np.array(edges[1:-1])
            seed=np.vstack((ev(p,boundary)/vscale[p],ev(p,boundary,True)/vscale[p]))
            seed/=np.maximum(np.linalg.norm(seed,axis=1,keepdims=True),1e-300)
            _,s,b=np.linalg.svd(seed,full_matrices=False)
            base=b[:int(np.sum(s>s[0]*1e-11))]
            ranks=[None]*(len(edges)-1); errors=[None]*len(ranks)
            seed_rank=len(base)
            for band in range(len(ranks)-1,-1,-1):
                a,z=edges[band:band+2]
                m=(rfit>=a)&(rfit<z); mm=(rtest>=a)&(rtest<z)
                v=fv[p][m]/vscale[p]; d=fd[p][m]/vscale[p]
                gamma=np.linalg.norm(v)/max(np.linalg.norm(d),1e-300)
                work=np.vstack((v,gamma*d))
                residual=work-(work@base.T)@base
                _,sv,new=np.linalg.svd(residual,full_matrices=False)
                # Orthogonalize the appended directions against the fixed prefix.
                candidate=np.vstack((base,new[:64-len(base)]))
                q,_=np.linalg.qr(candidate.T)
                candidate=q.T
                # QR preserves each prefix's span; signs of earlier vectors may change harmlessly.
                vv=tv[p][mm]/vscale[p]; dd=td[p][mm]/vscale[p]
                vnorm=np.where((rtest[mm]>=1.0)[:,None],vscale[p],vglobal[p])
                dnorm=np.where((rtest[mm]>=1.0)[:,None],dscale[p],dglobal[p])
                chosen=64; worst=(0.,0.)
                for rank in range(len(base),65):
                    bb=candidate[:rank]
                    e1=float(np.max(abs(((vv@bb.T)@bb-vv)*vscale[p]/vnorm)))
                    e2=float(np.max(abs(((dd@bb.T)@bb-dd)*vscale[p]/dnorm)))
                    if max(e1,e2)<=tol:
                        chosen=rank; worst=(e1,e2); break
                base=candidate[:chosen]
                ranks[band]=chosen; errors[band]=list(worst)
            pair_results.append({'pair':p,'seed_rank':seed_rank,'ranks':ranks,'errors':errors})
        rmax=np.max([p['ranks'] for p in pair_results],axis=0)
        pad8=np.minimum(64,8*((rmax+7)//8))
        pad2=[min(64,2**int(np.ceil(np.log2(x)))) for x in rmax]
        summary={'max_ranks':rmax.tolist(),'padded8':pad8.tolist(),'power2':pad2,
                 'node_width':64,'pair_results':pair_results}
        out['tolerances'][str(tol)]=summary
        print(json.dumps({'edges':edges,'tol':tol,'max_ranks':rmax.tolist(),'padded8':pad8.tolist(),'power2':pad2,'node_width':64}),flush=True)
    result['configs'].append(out)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
