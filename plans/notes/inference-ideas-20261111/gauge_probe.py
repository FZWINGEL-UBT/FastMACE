"""CPU-only algebra screen; no model or CUDA initialization, no repository imports."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3

torch.set_num_threads(1)
paths = []
for l1, l2, l3 in itertools.product(range(2), range(4), range(4)):
    if (l1 + l2 - l3) % 2 or not abs(l1-l2) <= l3 <= l1+l2:
        continue
    w = o3.wigner_3j(l1, l2, l3, dtype=torch.float64).numpy()
    terms = [(int(o), int(i), int(j), float(w[i,j,o]))
             for i,j,o in zip(*np.nonzero(np.abs(w) > 1e-12))]
    paths.append(((l1,l2,l3), terms))

def count_scales(terms):
    return len({(o, round(abs(c), 10)) for o,i,j,c in terms if abs(abs(c)-1) > 1e-9})

def path_normalized(terms):
    vals, ns = np.unique(np.round([abs(t[3]) for t in terms], 10), return_counts=True)
    a = vals[np.flatnonzero(ns == ns.max())[-1]]  # production tie rule: largest magnitude
    return [(o,i,j,c/a) for o,i,j,c in terms]

def output_normalized(terms):
    a = {}
    for o in sorted({t[0] for t in terms}):
        vals, ns = np.unique(np.round([abs(t[3]) for t in terms if t[0]==o],10), return_counts=True)
        mode = vals[np.argmax(ns)]
        a[o] = next(abs(c) for oo,i,j,c in terms if oo==o and abs(abs(c)-mode)<1e-9)
    return [(o,i,j,c/a[o]) for o,i,j,c in terms]

baseline = sum(count_scales(path_normalized(t)) for p,t in paths)
per_output = sum(count_scales(output_normalized(t)) for p,t in paths)
# All nonzero CGs: log|C| + a_i + b_j - s_(path,o) = 0.
# Row-normalization gives a valid gauge even when not every equation can hold.
keys = sorted({('x',l1,i) for (l1,l2,l3),ts in paths for o,i,j,c in ts})
keys += sorted({('y',l2,j) for (l1,l2,l3),ts in paths for o,i,j,c in ts})
keys += [('m',p,o) for p,ts in paths for o in sorted({t[0] for t in ts})]
ix = {k:i for i,k in enumerate(keys)}
A=[]; b=[]
for p,ts in paths:
    l1,l2,l3=p
    for o,i,j,c in ts:
        row = np.zeros(len(keys))
        row[ix['x',l1,i]]=1
        row[ix['y',l2,j]]=1
        row[ix['m',p,o]]=-1
        A.append(row); b.append(-np.log(abs(c)))
A=np.array(A); b=np.array(b)
z=np.linalg.lstsq(A,b,rcond=1e-12)[0]
residual=np.max(np.abs(A@z-b))

# Greedy random maximal consistent subsets, then use each output's modal scale.
# This is a candidate screen, not a proof of the globally smallest circuit.
rng=np.random.default_rng(20261111)
best={'multiplies': per_output, 'trial':-1}
for trial in range(250):
    selected=[]
    for r in rng.permutation(len(b)):
        ss=selected+[r]
        zz=np.linalg.lstsq(A[ss],b[ss],rcond=1e-11)[0]
        if np.max(np.abs(A[ss]@zz-b[ss])) < 1e-8:
            selected=ss
    zz=np.linalg.lstsq(A[selected],b[selected],rcond=1e-11)[0]
    details=[]; total=0
    for p,ts in paths:
        l1,l2,l3=p
        scaled=[(o,i,j,c*np.exp(zz[ix['x',l1,i]]+zz[ix['y',l2,j]])) for o,i,j,c in ts]
        norm=output_normalized(scaled)
        n=count_scales(norm); total+=n
        details.append({'path':p,'remaining_coefficient_muls':n})
    if total < best['multiplies']:
        best={'multiplies':total,'trial':trial,'scale_range':[float(np.exp(zz).min()),float(np.exp(zz).max())],
              'scales':{str(k):float(np.exp(zz[v])) for k,v in ix.items()},'paths':details}
        best_z=zz.copy()

# Verify the local tensor product and every input derivative; no MACE force gate.
checks={name:0.0 for name in ('message','dx','dy','dw')}
for p,ts in paths:
    l1,l2,l3=p
    sx=np.exp([best_z[ix['x',l1,i]] for i in range(2*l1+1)])
    sy=np.exp([best_z[ix['y',l2,j]] for j in range(2*l2+1)])
    c=np.zeros((2*l3+1,2*l1+1,2*l2+1))
    for o,i,j,v in ts: c[o,i,j]=v
    prelim=c*sx[None,:,None]*sy[None,None,:]
    so=[]
    for o in range(c.shape[0]):
        vals,ns=np.unique(np.round(np.abs(prelim[o][np.abs(c[o])>1e-12]),10),return_counts=True)
        mode=vals[np.argmax(ns)]
        so.append(next(v for v in np.abs(prelim[o][np.abs(c[o])>1e-12]) if abs(v-mode)<1e-9))
    so=np.array(so)
    cc=prelim/so[:,None,None]
    unit=np.abs(np.abs(cc)-1)<1e-8
    cc[unit]=np.sign(cc[unit])
    for _ in range(20):
        x=rng.standard_normal((128,len(sx))); y=rng.standard_normal(len(sy))
        g=rng.standard_normal((128,len(so))); w=rng.standard_normal(128)
        xx=x/sx; yy=y/sy; gg=g*so
        ref={'message':np.einsum('oij,ci,j,c->co',c,x,y,w),
             'dx':np.einsum('oij,co,j,c->ci',c,g,y,w),
             'dy':np.einsum('oij,co,ci,c->j',c,g,x,w),
             'dw':np.einsum('oij,co,ci,j->c',c,g,x,y)}
        cand={'message':np.einsum('oij,ci,j,c->co',cc,xx,yy,w)*so,
              'dx':np.einsum('oij,co,j,c->ci',cc,gg,yy,w)/sx,
              'dy':np.einsum('oij,co,ci,c->j',cc,gg,xx,w)/sy,
              'dw':np.einsum('oij,co,ci,j->c',cc,gg,xx,yy)}
        for k in checks:
            checks[k]=max(checks[k],float(np.max(np.abs(cand[k]-ref[k]))/max(1e-300,np.max(np.abs(ref[k])))))

result={'scope':'coefficient pre-multiplications only, not complete kernel instructions',
        'terms':len(b),'path_common_scale_muls':baseline,'per_output_scale_muls':per_output,
        'all_unit_consistent':bool(residual<1e-8),'all_unit_log_residual':float(residual),'best':best,
        'float64_local_checks_relative_to_max_output':checks,
        'per_path':[{'path':p,'terms':len(t),'path_scale_muls':count_scales(path_normalized(t)),
                     'output_scale_muls':count_scales(output_normalized(t))} for p,t in paths]}
out=Path(__file__).with_suffix('.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('per_path','best')},indent=2))
print(json.dumps({k:v for k,v in best.items() if k!='scales'},indent=2))
