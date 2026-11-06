"""Small CPU-only tensor-algebra calculations, not a model/test-suite run."""
import json, math
from pathlib import Path
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(73)
paths=[(0,0,0),(1,1,0),(0,1,1),(1,0,1),(1,2,1),(0,2,2),(1,1,2),(1,3,2),(0,3,3),(1,2,3)]
rows=[]
for a,l,L in paths:
    C=o3.wigner_3j(a,l,L)
    n=torch.tensor([0.,1.,0.],requires_grad=True)
    def yy(n): return o3.spherical_harmonics(l,n,normalize=False,normalization='component')
    Y=yy(n)
    J=torch.autograd.functional.jacobian(yy,n)
    # Tangential derivatives at the axis; radial derivative belongs to w'(r).
    J=J @ (torch.eye(3)-n[:,None]*n[None,:])
    T=torch.einsum('ijo,j->io',C,Y)
    TJ=torch.einsum('ijo,jd->iod',C,J)
    active=((T.abs().sum(0)+TJ.abs().sum((0,2)))>1e-12)
    # A scalar identity for the same CG coupling. H is a homogeneous harmonic polynomial.
    err=0.; derr=0.; ratios=[]
    for trial in range(12):
        n=torch.randn(3,requires_grad=True)
        x=torch.randn(2*a+1,requires_grad=True)
        g=torch.randn(2*L+1)
        H=(o3.spherical_harmonics(L,n,normalize=False,normalization='component')*g).sum()
        dH=torch.autograd.grad(H,n,create_graph=True)[0] if L else 0*n
        if a==0: f=x[0]*H
        elif L==l+1: f=(x*dH).sum()
        else: f=(x*n).sum()*H-(n*n).sum()/(2*L+1)*(x*dH).sum()
        direct=torch.einsum('ijo,i,j,o',C,x,yy(n),g)
        if trial==0: alpha=(direct/f).detach()
        ratios.append(float((direct/f).detach()))
        err=max(err,float((direct-alpha*f).detach().abs()))
        dd=torch.autograd.grad(direct,(n,x),retain_graph=True,allow_unused=True)
        df=torch.autograd.grad(alpha*f,(n,x),retain_graph=True,allow_unused=True)
        derr=max(derr,*[float(((0*z if u is None else u)-(0*z if v is None else v)).abs().max().detach()) for u,v,z in zip(dd,df,(n,x))])
    rows.append(dict(path=[a,l,L],cg_nonzero=int((C.abs()>1e-12).sum()),axis_terms=int((T.abs()>1e-12).sum()),tangent_terms=int((TJ.abs()>1e-12).sum()),required_rotated_g=int(active.sum()),full_rotation_fmas=(2*L+1)**2,partial_rotation_fmas=int(active.sum())*(2*L+1),scalar_identity_scale=float(alpha),identity_max_abs=err,identity_gradient_max_abs=derr))

# Receiver-tiled VJP: (S @ D) for dx and ((w*x) @ D.T) for the scalar contractions.
matrix_errors=[]
for a,l,L in paths:
    ne,nc=5,8
    X=torch.randn(ne,2*a+1,nc,requires_grad=True)
    R=torch.randn(ne,3,requires_grad=True)
    W=torch.randn(ne,nc,requires_grad=True)
    D=torch.randn(2*L+1,nc)
    C=o3.wigner_3j(a,l,L)
    Y=o3.spherical_harmonics(l,R,normalize=True,normalization='component')
    S=torch.einsum('ijo,ej->eio',C,Y)
    F=torch.einsum('eio,eic,ec,oc->',S,X,W,D)
    ref=torch.autograd.grad(F,(X,R,W),retain_graph=True,allow_unused=True)
    U=(S.reshape(ne*(2*a+1),2*L+1) @ D).reshape_as(X)
    dx=U*W[:,None,:]
    M=((W[:,None,:]*X).reshape(ne*(2*a+1),nc) @ D.T).reshape(ne,2*a+1,2*L+1)
    dr=torch.autograd.grad((S*M.detach()).sum()+R.sum()*0,R,retain_graph=True)[0]
    dw=(U*X).sum(1)
    matrix_errors.append(max(float(((v*0 if u is None else u)-v).abs().max().detach()) for u,v in zip(ref,(dx,dr,dw))))

E=373760; N=5000; Cc=128
def up(x,k):return (x+k-1)//k*k
budget=[]
for B in [8,16,32]:
    # One path per receiver slab; up(K,4) for FP64 DMMA, m/n multiples of eight.
    nt=math.ceil(E/B)
    dxflop=sum(2*up(B*(2*a+1),8)*Cc*up(2*L+1,4) for a,l,L in paths)*nt
    # w' force uses M' as well as M. In practice radial can instead use dot(X,U), avoiding the second M.
    momentflop=sum(2*up(B*(2*a+1),8)*8*Cc for a,l,L in paths)*nt
    gout=sum(up(2*L+1,4)*Cc*8 for a,l,L in paths)*nt
    # S and X are doubles in the staged MMA operands, intermediates remain on chip.
    max_s=max(up(B*(2*a+1),8)*up(2*L+1,4)*8 for a,l,L in paths)
    max_x=max(up(B*(2*a+1),8)*Cc*8 for a,l,L in paths)
    budget.append(dict(edges_per_slab=B,ideal_slabs_without_receiver_tails=nt,dx_GFLOP=dxflop/1e9,moment_GFLOP=momentflop/1e9,staged_D_bytes=gout,max_S_bytes=max_s,max_X_bytes=max_x))

result=dict(paths=rows,total_partial_rotation_fmas=sum(r['partial_rotation_fmas'] for r in rows),total_full_rotation_fmas=sum(r['full_rotation_fmas'] for r in rows),matrix_identity_max_abs=max(matrix_errors),receiver_tiled_budget=budget)
out=Path(__file__).with_name('new_path_census.json')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
