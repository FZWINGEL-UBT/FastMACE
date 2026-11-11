"""CPU-only exact angular product decompositions; no checkpoint, no CUDA calls."""
from pathlib import Path
import json
import numpy as np
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3
torch.set_num_threads(1)

# Ten directions span homogeneous cubics; six selected directions span symmetric quadratics.
directions=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,-1,0],
                     [1,0,1],[1,0,-1],[0,1,1],[0,1,-1],[1,1,1]],dtype=float)
directions/=np.linalg.norm(directions,axis=1,keepdims=True)
def harmonics(l,points):
    return o3.spherical_harmonics(l,torch.tensor(points,dtype=torch.float64),normalize=True,normalization='component').numpy()
def cg(l1,l2,l3):
    return o3.wigner_3j(l1,l2,l3,dtype=torch.float64).numpy().transpose(2,0,1)

six=np.array([0,1,2,3,5,7])
cp={}; detail={}
for path,points in [((1,1,2),directions[six]),((1,2,3),directions)]:
    l1,l2,l3=path
    a=harmonics(l1,points); b=harmonics(l2,points)
    f=np.einsum('ri,rj->rij',a,b).reshape(len(points),-1)
    c=cg(*path)
    d=np.linalg.lstsq(f.T,c.reshape(c.shape[0],-1).T,rcond=1e-13)[0].T
    cp[path]=(a,b,d)
    cp[(l1,l3,l2)]=(a,d.T,b.T)

rng=np.random.default_rng(20261111)
for path,(a,b,d) in cp.items():
    c=cg(*path); reconstructed=np.einsum('or,ri,rj->oij',d,a,b)
    coefferr=float(abs(c-reconstructed).max())
    assert coefferr<1e-12,(path,coefferr)
    errors=[]; errors32=[]
    for _ in range(20):
        x=rng.standard_normal((128,a.shape[1])); y=rng.standard_normal(b.shape[1]); w=rng.standard_normal(128)
        ref=np.einsum('oij,ci,j,c->co',c,x,y,w)
        value=((x@a.T)*(b@y)[None,:]*w[:,None])@d.T
        err=float(abs(ref-value).max()/max(abs(ref).max(),1e-300))
        xx=x.astype(np.float32); yy=y.astype(np.float32); ww=w.astype(np.float32)
        value32=((xx@a.astype(np.float32).T)*(b.astype(np.float32)@yy)[None,:]*ww[:,None])@d.astype(np.float32).T
        errors.append(err); errors32.append(float(abs(ref-value32).max()/max(abs(ref).max(),1e-300)))
    nnz=int(np.count_nonzero(abs(c)>1e-12)); no=c.shape[0]; ni=c.shape[1]
    # Count a common hand-written forward: term products + accumulation of w*sum; constants not separately charged.
    # GEOM has per-input w*x then updates per nonzero(input,output), with geometry precomputed.
    nio=int(np.count_nonzero(np.any(abs(c)>1e-12,axis=2)))
    detail[str(path)]={'products':len(a),'physical_outputs':no,'cg_nonzeros':nnz,
                       'coefficient_max_abs_error':coefferr,'local_f64_relative_error':max(errors),
                       'local_f32_relative_error':max(errors32),
                       'hot_loop_symbolic_operations':{'term_products_plus_weighted_output':2*nnz+no,
                                                      'geometry_hoisted':ni+nio,'rank_one':2*len(a)},
                       'reconstruction_matrix_2norm':float(np.linalg.norm(d,2))}
result={'scope':'Four exact angular paths only; symbolic arithmetic is not compiled code or timing; float32 is a random local check, not a force gate',
        'physical_accumulators_for_all_paths':40,'overcomplete_accumulators_forward_only':52,
        'previous_uncompressed_deferred_CG_accumulators':76,'max_shared_sender_vector_forms':10,
        'extra_sender_bytes_for_5000_atoms_128channels':5000*128*(10-3)*4,
        'transformed_geometry_forms_for_four_paths':sum(len(x[0]) for x in cp.values()),'paths':detail}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez(Path(__file__).with_suffix('.npz'),directions=directions,**{f'{p}_{name}':v for p,parts in cp.items() for name,v in zip(('a','b','d'),parts)})
print(json.dumps(result,indent=2),flush=True)
