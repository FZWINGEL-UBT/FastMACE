"""CPU compile only. U/M contraction skeleton for one vector path, 16 edges.
No launch wrapper, Horner evaluation, geometry preparation, receiver segmentation,
force epilogue or node scatter. Those costs remain explicitly unverified.
"""
from pathlib import Path
import re,json,ctypes as c,subprocess,collections,hashlib
scratch=Path(__file__).parent
out=scratch/'adjoint_mma'; out.mkdir(exist_ok=True)
src=r'''
__device__ __forceinline__ void mm(double &d0,double &d1,double a,double b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d0),"+d"(d1) : "d"(a),"d"(b));
}
extern "C" __global__ __launch_bounds__(128)
void adjoint_mma(const float* S, const float* D, const float* X, const float* W, float* Uout, double* Mout) {
    __shared__ double ss[48*8];
    __shared__ double dd[8*32];
    __shared__ double wx[48*32];
    const int t=threadIdx.x,l=t&31,w=t/32;
    for(int i=t;i<48*8;i+=128) ss[i]=double(S[blockIdx.x*48*8+i]);
    double m0[2]={},m1[2]={};
    for(int cc=0;cc<128;cc+=32) {
        for(int i=t;i<8*32;i+=128) dd[i]=double(D[(i/32)*128+cc+i%32]);
        for(int i=t;i<48*32;i+=128) {
            int row=i/32,ch=cc+i%32;
            wx[i]=double(X[(blockIdx.x*48+row)*128+ch])*double(W[(blockIdx.x*16+row/3)*128+ch]);
        }
        __syncthreads();
        #pragma unroll
        for(int rt=0;rt<6;++rt) {
            double u0=0,u1=0;
            #pragma unroll
            for(int k=0;k<8;k+=4) {
                double a=ss[(rt*8+l/4)*8+k+l%4];
                double b=dd[(k+l%4)*32+w*8+l/4];
                mm(u0,u1,a,b);
            }
            int row=rt*8+l/4,ch=cc+w*8+2*(l%4);
            Uout[(blockIdx.x*48+row)*128+ch]=float(u0);
            Uout[(blockIdx.x*48+row)*128+ch+1]=float(u1);
        }
        #pragma unroll
        for(int it=0;it<2;++it) {
            int rt=w+4*it;
            if(rt<6) {
                #pragma unroll
                for(int k=0;k<32;k+=4) {
                    double a=wx[(rt*8+l/4)*32+k+l%4];
                    double b=dd[(l/4)*32+k+l%4];
                    mm(m0[it],m1[it],a,b);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int it=0;it<2;++it) {
        int rt=w+4*it;
        if(rt<6) {
            int row=rt*8+l/4,col=2*(l%4);
            Mout[(blockIdx.x*48+row)*8+col]=m0[it];
            Mout[(blockIdx.x*48+row)*8+col+1]=m1[it];
        }
    }
}
'''
variants={'adjoint_mma':src}
harness='#'+(scratch/'compile130.py').read_text().split('# Compile through NVRTC only.',1)[1].split('for a,b in',1)[0]
harness=harness.replace('[A-Z][A-Z0-9.]+','[A-Z][A-Za-z0-9.]+')
exec(harness)
