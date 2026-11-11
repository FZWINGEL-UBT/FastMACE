"""CPU screen of a common small rational denominator for each radial table piece.

Bounds below are analytical Chebyshev coefficient bounds relative to the cached polynomial,
excluding new float32 evaluation error and the original-network discrepancy.
"""
from pathlib import Path
import json
import numpy as np
import torch
torch.set_num_threads(1)
source=Path.home()/'.cache/fastmace/radial-tables/c697fb752157aea2cccd6085ebe9843e283d630c4388f8e419a4c3e6fcebff11.pt'
t=torch.load(source,map_location='cpu',weights_only=True)['tables']
c=t['coef'].numpy().astype(float)[...,:64]; dc=t['dcoef'].numpy().astype(float)[...,:64]
start=t['starts'].numpy().astype(float); scale=t['scales'].numpy().astype(float)
end=start+2/scale
xx=np.cos(np.pi*(np.arange(33)+0.5)/33)
phi=np.polynomial.chebyshev.chebvander(xx,5)
dphi=np.polynomial.chebyshev.chebvander(xx,4)
value=np.einsum('nt,ptk->pnk',phi,c[0])
# Scale per pair from dense sampling of each existing piece.
result={'source':str(source),'scope':'Coefficient upper bounds relative to cached polynomials; not network certification or GPU rounding',
        'fits':[]}
degree=np.arange(8)
for numerator_degree in (2,3,4):
    scores=[]; pol_scores=[]; clipped=0; pairs=[]
    for p in range(9):
        v=np.einsum('nt,ptk->pnk',phi,c[p]); d=np.einsum('nt,ptk->pnk',dphi,dc[p])
        vglob=np.maximum(abs(v).max(axis=(0,1)),1e-20)
        dglob=np.maximum(abs(d).max(axis=(0,1)),1e-20)
        phys=end>=1.0
        vp=np.maximum(abs(v[phys]).max(axis=(0,1)),1e-20)
        dp=np.maximum(abs(d[phys]).max(axis=(0,1)),1e-20)
        vs=np.where(phys[:,None],vp[None,:],vglob[None,:])
        ds=np.where(phys[:,None],dp[None,:],dglob[None,:])
        prod=np.zeros((3,len(start),8,64))
        for qdegree in range(3):
            for k in range(6):
                prod[qdegree,:,k+qdegree,:]+=0.5*c[p,:,k,:]
                prod[qdegree,:,abs(k-qdegree),:]+=0.5*c[p,:,k,:]
        high=slice(numerator_degree+1,None)
        # Conservative derivative weighting on the discarded product coefficients.
        weight=np.sqrt(1/vs[:,None,:]**2+(scale[:,None,None]*degree[None,:,None]**2/ds[:,None,:])**2)
        z0=(prod[0]*weight)[:,high,:].reshape(len(start),-1)
        z1=(prod[1]*weight)[:,high,:].reshape(len(start),-1)
        z2=(prod[2]*weight)[:,high,:].reshape(len(start),-1)
        lhs=np.empty((len(start),2,2)); rhs=np.empty((len(start),2))
        lhs[:,0,0]=np.einsum('pk,pk->p',z1,z1); lhs[:,0,1]=np.einsum('pk,pk->p',z1,z2)
        lhs[:,1,0]=lhs[:,0,1]; lhs[:,1,1]=np.einsum('pk,pk->p',z2,z2)
        rhs[:,0]=-np.einsum('pk,pk->p',z1,z0); rhs[:,1]=-np.einsum('pk,pk->p',z2,z0)
        qq=np.array([np.linalg.lstsq(a,b,rcond=1e-12)[0] for a,b in zip(lhs,rhs)])
        norm=abs(qq).sum(axis=1)
        clipped+=int(np.count_nonzero(norm>0.8))
        qq*=np.minimum(1,0.8/np.maximum(norm,1e-300))[:,None]
        qmin=1-abs(qq).sum(axis=1)
        residual=prod[0]+qq[:,0,None,None]*prod[1]+qq[:,1,None,None]*prod[2]
        residual[:,:numerator_degree+1]=0
        vb=abs(residual).sum(axis=1)/qmin[:,None]
        qder=scale*(abs(qq[:,0])+4*abs(qq[:,1]))
        db=scale[:,None]*(abs(residual)*degree[None,:,None]**2).sum(axis=1)/qmin[:,None]+vb*qder[:,None]/qmin[:,None]
        score=np.maximum((vb/vs).max(1),(db/ds).max(1))
        pr=c[p].copy(); pr[:,:numerator_degree+1]=0
        pv=abs(pr).sum(1)
        pd=scale[:,None]*(abs(pr)*np.arange(6)[None,:,None]**2).sum(1)
        pol=np.maximum((pv/vs).max(1),(pd/ds).max(1))
        scores.append(score); pol_scores.append(pol)
        pairs.append({'pair':p,'rational_score_median':float(np.median(score)),
                      'polynomial_score_median':float(np.median(pol))})
    score=np.array(scores); pol=np.array(pol_scores)
    physical=end>=1.0
    entry={'numerator_degree':numerator_degree,'denominator_degree':2,
           'per_column_coefficients':numerator_degree+1,'denominator_coefficients_shared_over_64_columns':2,
           'clipped_pieces':clipped,'physical_piece_count':int(physical.sum()),
           'physical_median_rational_bound':float(np.median(score[:,physical])),
           'physical_median_polynomial_bound':float(np.median(pol[:,physical])),
           'pass_fraction_physical':{str(tol):{'rational':float(np.mean(score[:,physical]<=tol)),
                                           'plain_polynomial':float(np.mean(pol[:,physical]<=tol))}
                                     for tol in (1e-7,3e-7,1e-6)},'pairs':pairs}
    result['fits'].append(entry)
    print(json.dumps({k:v for k,v in entry.items() if k!='pairs'}),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
