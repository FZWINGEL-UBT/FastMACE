"""Compile-only sketch: node-converted Cartesian adjoints and scalarized CG VJP.
The wrapper and node basis-conversion kernel are NOT implemented.
"""
from pathlib import Path
import sympy as s
import re,json,ctypes as c,subprocess,collections,hashlib
scratch=Path(__file__).parent
out=scratch/'cartesian_vjp'
out.mkdir(exist_ok=True)
base=(scratch/'base.cu').read_text()
paths=[(0,0,0),(1,1,0),(0,1,1),(1,0,1),(1,2,1),(0,2,2),(1,1,2),(1,3,2),(0,3,3),(1,2,3)]
nn=s.symbols('nx ny nz'); xx=s.symbols('xx xy xz'); nx,ny,nz=nn
q=s.symbols('q0:7')
H=[q[0],q[0]*nx+q[1]*ny+q[2]*nz,
   q[0]*(nx*nx-nz*nz)+2*q[1]*nx*ny+2*q[2]*nx*nz+q[3]*(ny*ny-nz*nz)+2*q[4]*ny*nz,
   q[0]*(nx**3-3*nx*nz*nz)+3*q[1]*(nx*nx*ny-ny*nz*nz)+q[2]*(3*nx*nx*nz-nz**3)+3*q[3]*(nx*ny*ny-nx*nz*nz)+6*q[4]*nx*ny*nz+q[5]*(ny**3-3*ny*nz*nz)+q[6]*(3*ny*ny*nz-nz**3)]
def cc(e):
    v=s.ccode(e,standard='C99')
    v=re.sub(r'pow\((\w+), 2\)',r'(\1*\1)',v)
    v=re.sub(r'pow\((\w+), 3\)',r'(\1*\1*\1)',v)
    v=re.sub(r'(?<![\w.])(\d+\.\d+)(?![\w.])',r'\1f',v)
    return v

begin=base[:base.index('    acc_t edge_gradient[')]
# Reuse input indices, table lookup, cutoff handling, and the original Horner block.
original_horner=base[base.index('                scalar_t gv[PADDED_TERMS]'):base.index('                acc_t weight_gradient')]
begin+='''
    // sh l=1 is sqrt(3) times the normalized edge vector. The zero-length/eps case needs a wrapper fallback.
    const float nx=sh[edge_base+1]*0.57735026918962576451f;
    const float ny=sh[edge_base+2]*0.57735026918962576451f;
    const float nz=sh[edge_base+3]*0.57735026918962576451f;
    const float ri=1.f/r;
    float fx=0,fy=0,fz=0;
    for(int channel=lane;channel<128;channel+=32) {
        const float x0=x[input_base+channel];
        const float xx=x[input_base+128+channel],xy=x[input_base+256+channel],xz=x[input_base+384+channel];
        float dx0=0,dxx=0,dxy=0,dxz=0;
'''
blocks=[]; node_offset=0; metadata=[]
for p,(a,l,L) in enumerate(paths):
    poly=H[L]; gh=[s.diff(poly,n) for n in nn]
    hh=[[s.diff(gg,n) for n in nn] for gg in gh]
    hx=[sum(hh[j][i]*xx[i] for i in range(3)) for j in range(3)]
    # The polynomial and its derivative share Cartesian partial contractions.
    # CSE of scalar expressions alone; all calculations happen at compile time on the CPU.
    A=sum(xx[i]*nn[i] for i in range(3)); B=sum(xx[i]*gh[i] for i in range(3))
    if a==0:
        U=[poly]; F=s.Symbol('x0')*poly; G=[s.Symbol('x0')*gg for gg in gh]
    elif L==l+1:
        U=gh; F=B; G=hx
    else:
        k=2*L+1
        U=[nn[i]*poly-gh[i]/k for i in range(3)]
        F=A*poly-B/k
        G=[xx[i]*poly+A*gh[i]-2*nn[i]*B/k-hx[i]/k for i in range(3)]
    reps,exprs=s.cse([s.factor(z) for z in U+[F]+G],symbols=s.numbered_symbols('z'),order='canonical')
    st=['{',f'const int weight_index={p*128}+channel;',original_horner]
    for j in range(2*L+1):st.append(f'const float q{j}=dout0[int64_t(receiver)*5120+{node_offset+j*128}+channel];')
    if L==0:
        st+=['const float h=q0, hx=0,hy=0,hz=0, hhx=0,hhy=0,hhz=0;']
    elif L==1:
        st+=['const float h=q0*nx+q1*ny+q2*nz;','const float hx=q0,hy=q1,hz=q2,hhx=0,hhy=0,hhz=0;']
    elif L==2:
        st+=['const float qzz=-q0-q3;',
             'const float vx=q0*nx+q1*ny+q2*nz,vy=q1*nx+q3*ny+q4*nz,vz=q2*nx+q4*ny+qzz*nz;',
             'const float h=nx*vx+ny*vy+nz*vz,hx=2.f*vx,hy=2.f*vy,hz=2.f*vz;',
             'const float hhx=2.f*(q0*xx+q1*xy+q2*xz),hhy=2.f*(q1*xx+q3*xy+q4*xz),hhz=2.f*(q2*xx+q4*xy+qzz*xz);']
    else:
        st+=['const float q7=-q0-q3,q8=-q1-q5;',
             'const float Qxx=q0*nx+q1*ny+q2*nz,Qxy=q1*nx+q3*ny+q4*nz,Qxz=q2*nx+q4*ny+q7*nz;',
             'const float Qyy=q3*nx+q5*ny+q6*nz,Qyz=q4*nx+q6*ny+q8*nz,Qzz=-Qxx-Qyy;',
             'const float vx=Qxx*nx+Qxy*ny+Qxz*nz,vy=Qxy*nx+Qyy*ny+Qyz*nz,vz=Qxz*nx+Qyz*ny+Qzz*nz;',
             'const float h=nx*vx+ny*vy+nz*vz,hx=3.f*vx,hy=3.f*vy,hz=3.f*vz;',
             'const float hhx=6.f*(Qxx*xx+Qxy*xy+Qxz*xz),hhy=6.f*(Qxy*xx+Qyy*xy+Qyz*xz),hhz=6.f*(Qxz*xx+Qyz*xy+Qzz*xz);']
    if a==0:
        st+=['const float u0=h,f=x0*h,gx=x0*hx,gy=x0*hy,gz=x0*hz;']
    elif L==l+1:
        st+=['const float ux=hx,uy=hy,uz=hz,f=xx*ux+xy*uy+xz*uz,gx=hhx,gy=hhy,gz=hhz;']
    else:
        st+=['const float A=xx*nx+xy*ny+xz*nz,B=xx*hx+xy*hy+xz*hz;',f'const float ik={1/(2*L+1):.17g}f;',
             'const float ux=nx*h-ik*hx,uy=ny*h-ik*hy,uz=nz*h-ik*hz,f=A*h-ik*B;',
             'const float gx=xx*h+A*hx-ik*(2.f*nx*B+hhx),gy=xy*h+A*hy-ik*(2.f*ny*B+hhy),gz=xz*h+A*hz-ik*(2.f*nz*B+hhz);']
    for name,u in zip(['dx0'] if a==0 else ['dxx','dxy','dxz'],['u0'] if a==0 else ['ux','uy','uz']):
        st.append(f'{name}=fmaf(radial_weight,{u},{name});')
    st+=['const float wr=radial_weight*ri;',f'const float radial=(sloped ? radial_dt*sc : 0.f)*f-{l}.f*wr*f;',
         'fx=fmaf(wr,gx,fmaf(radial,nx,fx));','fy=fmaf(wr,gy,fmaf(radial,ny,fy));','fz=fmaf(wr,gz,fmaf(radial,nz,fz));','}']
    blocks.append('\n'.join(st))
    metadata.append(dict(path=[a,l,L],temporaries=len(reps),node_coefficients=2*L+1))
    node_offset+=(2*L+1)*128
end='''
        atomicAdd(dx+input_base+channel,dx0);
        atomicAdd(dx+input_base+128+channel,dxx);
        atomicAdd(dx+input_base+256+channel,dxy);
        atomicAdd(dx+input_base+384+channel,dxz);
    }
    for(int o=16;o;o>>=1) {fx+=__shfl_down_sync(FULL_MASK,fx,o);fy+=__shfl_down_sync(FULL_MASK,fy,o);fz+=__shfl_down_sync(FULL_MASK,fz,o);}
    if(lane==0){dsh[3*e]=fx;dsh[3*e+1]=fy;dsh[3*e+2]=fz;dr[e]=0;}
}
'''
src=begin+'\n'.join(blocks)+end
# Align skipped-edge writes to the proposed [E,3] vector-adjoint output.
src=src.replace('c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[e * EDGE_DIM + c]', 'c < 3; c += LOGICAL_GROUP_SIZE) dsh[e * 3 + c]')
src=src.replace('const float ik=1f;', 'const float ik=1.f;')
variants={'cartesian_factored':src}
harness='#'+(scratch/'compile130.py').read_text().split('# Compile through NVRTC only.',1)[1].split('for a,b in',1)[0]
harness=harness.replace('[A-Z][A-Z0-9.]+','[A-Z][A-Za-z0-9.]+')
exec(harness)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
