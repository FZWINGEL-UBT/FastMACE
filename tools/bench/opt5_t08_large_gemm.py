#!/usr/bin/env python3
"""OPT5-T08 follow-up: prepacked B and larger SIMT/MMA GEMM tiles.

Experimental microbenchmark only; no production dispatch or weight cache.
The FP16 branches are not exponent-normalized: wide-range and real-operand
accuracy failures are recorded by --validate, not treated as qualification.
The old kernels remain the measured control. Packing B is timed separately
and together with GEMM; no cached-weight saving is assumed in the latter.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import opt5_t08_hybrid_gemm as baseline
import torch
import triton
import triton.language as tl


@triton.jit
def pack_b_kernel(B, P, K: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr):
    # One uint4 per lane: hi[k:k+2], hi[k+8:k+10], lo[k:k+2], lo[k+8:k+10].
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = (x // 4) % NP
    k = (x // (NP * 4)) * 16 + (x % 4) * 2
    for delta in tl.static_range(2):
        v0 = tl.load(B + (k + delta * 8) * N + col, (col < N) & (k + delta * 8 < K), 0)
        v1 = tl.load(B + (k + delta * 8 + 1) * N + col, (col < N) & (k + delta * 8 + 1 < K), 0)
        h0 = v0.to(tl.float16)
        h1 = v1.to(tl.float16)
        l0 = ((v0 - h0.to(tl.float32)) * 2048.0).to(tl.float16)
        l1 = ((v1 - h1.to(tl.float32)) * 2048.0).to(tl.float16)
        hi = h0.to(tl.uint16, bitcast=True).to(tl.uint32) | (h1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
        lo = l0.to(tl.uint16, bitcast=True).to(tl.uint32) | (l1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
        valid = x < triton.cdiv(K, 16) * NP * 4
        tl.store(P + x * 4 + delta, hi, valid)
        tl.store(P + x * 4 + delta + 2, lo, valid)


def pack_b(b, packed):
    k, n = b.shape
    npad = triton.cdiv(n, 8) * 8
    pack_b_kernel[(triton.cdiv(triton.cdiv(k, 16) * npad * 4, 256),)](b, packed, k, n, npad, 256)


SOURCE = r"""
// Larger 8x8 SIMT thread tiles reuse each shared load twice as much as 8x4.
// Six warps produce 96 rows; two MMA warps produce the remaining 32 rows.
// B's FP16 pair representation is packed once, not once per output tile.
template<int BN, bool PACKED, bool CROSS_ACC, bool ASYNC_B=false>
__device__ void row_hybrid(const float* __restrict__ A, const float* __restrict__ B,
                          const uint4* __restrict__ P, float* __restrict__ C,
                          int M, int N, int K, int NP) {
  constexpr int BM=128, BK=16, RN=BN/16, NT=BN/8;
  __shared__ __align__(16) float sa[2][BK][BM+4];
  __shared__ __align__(16) float sb[2][BK][BN+4];
  const int t=threadIdx.x, w=t/32, lane=t%32;
  const int row=blockIdx.x*BM, col=blockIdx.y*BN;
  const int ar=t/4, ak=(t%4)*4;
  const int bc=(t%(BN/4))*4, bk=t/(BN/4);
  const int sr=(t/16)*8, sc=(t%16)*RN;
  // Disjoint warp roles must share accumulator storage, not reserve both sets.
  float acc[NT*4], cross[CROSS_ACC ? NT : 1][4];
  #pragma unroll
  for(int j=0;j<NT;++j) {
    #pragma unroll
    for(int q=0;q<4;++q) { acc[j*4+q]=0; if(CROSS_ACC) cross[j][q]=0; }
  }
  auto load=[&](int kk, int stage, float4 (&pa)[2], float4 (&pb)[BN/64]) {
    #pragma unroll
    for(int i=0;i<2;++i) {
      const int r=row+ar+i*64, k=kk+ak;
      float4 v=make_float4(0,0,0,0);
      if(r<M && k+3<K && K%4==0) v=*reinterpret_cast<const float4*>(A+r*K+k);
      else if(r<M) {
        if(k<K) v.x=A[r*K+k]; if(k+1<K) v.y=A[r*K+k+1];
        if(k+2<K) v.z=A[r*K+k+2]; if(k+3<K) v.w=A[r*K+k+3];
      }
      pa[i]=v;
    }
    #pragma unroll
    for(int i=0;i<BN/64;++i) {
      const int k=kk+bk+i*(1024/BN), n=col+bc;
      if(ASYNC_B && N%4==0) {
        const unsigned dst=(unsigned)__cvta_generic_to_shared(&sb[stage][bk+i*(1024/BN)][bc]);
        const int bytes=k<K && n+3<N ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                     :: "r"(dst),"l"(B+k*N+n),"r"(bytes) : "memory");
      } else {
        float4 v=make_float4(0,0,0,0);
        if(k<K && n+3<N && N%4==0) v=*reinterpret_cast<const float4*>(B+k*N+n);
        else if(k<K) {
          if(n<N) v.x=B[k*N+n]; if(n+1<N) v.y=B[k*N+n+1];
          if(n+2<N) v.z=B[k*N+n+2]; if(n+3<N) v.w=B[k*N+n+3];
        }
        if(ASYNC_B) *reinterpret_cast<float4*>(&sb[stage][bk+i*(1024/BN)][bc])=v;
        else pb[i]=v;
      }
    }
  };
  auto store=[&](int stage,const float4 (&pa)[2],const float4 (&pb)[BN/64]) {
    #pragma unroll
    for(int i=0;i<2;++i) {
      sa[stage][ak][ar+i*64]=pa[i].x; sa[stage][ak+1][ar+i*64]=pa[i].y;
      sa[stage][ak+2][ar+i*64]=pa[i].z; sa[stage][ak+3][ar+i*64]=pa[i].w;
    }
    if(!ASYNC_B) {
      #pragma unroll
      for(int i=0;i<BN/64;++i) *reinterpret_cast<float4*>(&sb[stage][bk+i*(1024/BN)][bc])=pb[i];
    }
  };
  float4 pa[2],pb[BN/64];
  load(0,0,pa,pb); store(0,pa,pb);
  if(ASYNC_B) asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  int stage=0;
  for(int kk=0;kk<K;kk+=BK) {
    if(kk+BK<K) load(kk+BK,stage^1,pa,pb);
    if(w<6) {
      #pragma unroll 4
      for(int k=0;k<BK;++k) {
        float bv[RN];
        #pragma unroll
        for(int j=0;j<RN;j+=4) {
          const float4 v=*reinterpret_cast<const float4*>(&sb[stage][k][sc+j]);
          bv[j]=v.x; bv[j+1]=v.y; bv[j+2]=v.z; bv[j+3]=v.w;
        }
        #pragma unroll
        for(int i=0;i<8;i+=4) {
          const float4 v=*reinterpret_cast<const float4*>(&sa[stage][k][sr+i]);
          const float av[4]={v.x,v.y,v.z,v.w};
          #pragma unroll
          for(int ii=0;ii<4;++ii) {
            #pragma unroll
            for(int j=0;j<RN;++j) acc[(i+ii)*RN+j]=fmaf(av[ii],bv[j],acc[(i+ii)*RN+j]);
          }
        }
      }
    } else {
      const int r=96+(w-6)*16+lane/4, k=2*(lane%4);
      unsigned ah[4],al[4];
      split_f16x2(sa[stage][k][r],sa[stage][k+1][r],ah[0],al[0]);
      split_f16x2(sa[stage][k][r+8],sa[stage][k+1][r+8],ah[1],al[1]);
      split_f16x2(sa[stage][k+8][r],sa[stage][k+9][r],ah[2],al[2]);
      split_f16x2(sa[stage][k+8][r+8],sa[stage][k+9][r+8],ah[3],al[3]);
      #pragma unroll
      for(int j=0;j<NT;++j) {
        unsigned bh[2],bl[2];
        if(PACKED) {
          const int n=col+j*8+lane/4;
          uint4 v=make_uint4(0,0,0,0);
          if(n<NP) v=P[(kk/16*NP+n)*4+lane%4];
          bh[0]=v.x; bh[1]=v.y; bl[0]=v.z; bl[1]=v.w;
        } else {
          const int n=j*8+lane/4;
          split_f16x2(sb[stage][k][n],sb[stage][k+1][n],bh[0],bl[0]);
          split_f16x2(sb[stage][k+8][n],sb[stage][k+9][n],bh[1],bl[1]);
        }
        mma_fp16(&acc[j*4],ah,bh);
        if(CROSS_ACC) {
          mma_fp16(cross[j],ah,bl); mma_fp16(cross[j],al,bh);
        } else {
          float tmp[4]={0,0,0,0};
          mma_fp16(tmp,ah,bl); mma_fp16(tmp,al,bh);
          #pragma unroll
          for(int q=0;q<4;++q) acc[j*4+q]=fmaf(tmp[q],1.0f/2048,acc[j*4+q]);
        }
      }
    }
    if(kk+BK<K) {
      store(stage^1,pa,pb); stage^=1;
      if(ASYNC_B) asm volatile("cp.async.wait_all;" ::: "memory");
      __syncthreads();
    }
  }
  if(w<6) {
    #pragma unroll
    for(int i=0;i<8;++i) {
      #pragma unroll
      for(int j=0;j<RN;++j) if(row+sr+i<M && col+sc+j<N) C[(row+sr+i)*N+col+sc+j]=acc[i*RN+j];
    }
  } else {
    const int r=row+96+(w-6)*16+lane/4, c=col+2*(lane%4);
    #pragma unroll
    for(int j=0;j<NT;++j) {
      #pragma unroll
      for(int q=0;q<4;++q) {
        const int rr=r+(q/2)*8, cc=c+j*8+q%2;
        if(rr<M && cc<N) C[rr*N+cc]=CROSS_ACC ? fmaf(cross[j][q],1.0f/2048,acc[j*4+q]) : acc[j*4+q];
      }
    }
  }
}
#define ROW(NAME,BN,PACK,CROSS,RES) \
extern "C" __global__ void __launch_bounds__(256,RES) NAME( \
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) { \
 row_hybrid<BN,PACK,CROSS>(A,B,P,C,M,N,K,NP); }
ROW(row128_inline,128,false,false,2)
ROW(row128_packed,128,true,false,2)
ROW(row128_cross,128,true,true,1)
ROW(row64_packed,64,true,false,2)
ROW(row64_cross,64,true,true,2)
extern "C" __global__ void __launch_bounds__(256,2) row128_async(
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) {
 row_hybrid<128,true,false,true>(A,B,P,C,M,N,K,NP);
}

template<int BM,int SW,bool PACKED,bool CROSS,bool ALIGNED=false,bool STAGED=false,bool SWIZZLE=false,int BKC=16>
__device__ void column_hybrid(const float* __restrict__ A,const float* __restrict__ B,
                             const uint4* __restrict__ P,float* __restrict__ C,
                             int M,int N,int K,int NP) {
  constexpr int SN=SW*16, BN=SN+32, BK=BKC, AM=BM/64, RM=BM/8;
  constexpr int A_PER=AM*(BK/16), B_PER=(BK*BN/4+255)/256;
  constexpr int MT=BM/((8-SW)*16), MC=MT*16, SA=RM*4;
  constexpr int AC=SA>(CROSS ? 2*MC : MC) ? SA : (CROSS ? 2*MC : MC);
  __shared__ __align__(16) float sa[2][BK][BM+4];
  __shared__ __align__(16) float sb[2][BK][SN+4];
  __shared__ __align__(16) float sm[2][PACKED ? 1 : 32][PACKED ? 1 : 20];
  __shared__ __align__(16) uint4 sp[2][STAGED ? 128 : 1];
  const int t=threadIdx.x,w=t/32,l=t%32;
  int tile_m=blockIdx.x, tile_n=blockIdx.y;
  if(SWIZZLE) {
    // Rasterize in groups of 8 M-tiles across all N-tiles so concurrent CTAs share A tiles.
    const int tiles_m=gridDim.x, tiles_n=gridDim.y, linear=blockIdx.y*gridDim.x+blockIdx.x;
    const int group_tiles=8*tiles_n, group_id=linear/group_tiles, first_m=group_id*8;
    const int group_m=(tiles_m-first_m<8 ? tiles_m-first_m : 8), in_group=linear%group_tiles;
    tile_m=first_m + in_group%group_m;
    tile_n=in_group/group_m;
  }
  const int row=tile_m*BM, col=tile_n*BN;
  const int ar=t/4,ak=(t%4)*4,sr=t/(SN/4),sc=(t%(SN/4))*4;
  float acc[AC];
  #pragma unroll
  for(int i=0;i<AC;++i) acc[i]=0;
  auto load=[&](int kk,float4 (&a)[A_PER],float4 (&b)[B_PER],uint4& p) {
    if(STAGED && t<128) {
      const int c=col+SN+t/4;
      p=make_uint4(0,0,0,0);
      if(c<NP) p=P[(kk/16*NP+c)*4+t%4];
    }
    #pragma unroll
    for(int i=0;i<AM;++i) {
      #pragma unroll
      for(int j=0;j<BK/16;++j) {
        const int r=row+ar+i*64,k=kk+ak+j*16;
        float4 v=make_float4(0,0,0,0);
        if(r<M && k+3<K && (ALIGNED || K%4==0)) v=*reinterpret_cast<const float4*>(A+r*K+k);
        else if(r<M) {
          if(k<K) v.x=A[r*K+k]; if(k+1<K) v.y=A[r*K+k+1];
          if(k+2<K) v.z=A[r*K+k+2]; if(k+3<K) v.w=A[r*K+k+3];
        }
        a[i*(BK/16)+j]=v;
      }
    }
    #pragma unroll
    for(int i=0;i<B_PER;++i) {
      const int idx=t+i*256,k=kk+idx/(BN/4),c=(idx%(BN/4))*4,n=col+c;
      float4 v=make_float4(0,0,0,0);
      if((ALIGNED || idx<BK*BN/4) && (!PACKED || c<SN)) {
        if(ALIGNED || (k<K && n+3<N && N%4==0)) v=*reinterpret_cast<const float4*>(B+k*N+n);
        else if(k<K) {
          if(n<N) v.x=B[k*N+n]; if(n+1<N) v.y=B[k*N+n+1];
          if(n+2<N) v.z=B[k*N+n+2]; if(n+3<N) v.w=B[k*N+n+3];
        }
      }
      b[i]=v;
    }
  };
  auto store=[&](int s,const float4 (&a)[A_PER],const float4 (&b)[B_PER],const uint4& p) {
    if(STAGED && t<128) sp[s][t]=p;
    #pragma unroll
    for(int i=0;i<AM;++i) {
      #pragma unroll
      for(int j=0;j<BK/16;++j) {
        const int kk16=ak+j*16;
        sa[s][kk16][ar+i*64]=a[i*(BK/16)+j].x; sa[s][kk16+1][ar+i*64]=a[i*(BK/16)+j].y;
        sa[s][kk16+2][ar+i*64]=a[i*(BK/16)+j].z; sa[s][kk16+3][ar+i*64]=a[i*(BK/16)+j].w;
      }
    }
    #pragma unroll
    for(int i=0;i<B_PER;++i) {
      const int idx=t+i*256,k=idx/(BN/4),c=(idx%(BN/4))*4;
      if(ALIGNED || idx<BK*BN/4) {
        if(c<SN) *reinterpret_cast<float4*>(&sb[s][k][c])=b[i];
        else if(!PACKED) {
          sm[s][c-SN][k]=b[i].x; sm[s][c-SN+1][k]=b[i].y;
          sm[s][c-SN+2][k]=b[i].z; sm[s][c-SN+3][k]=b[i].w;
        }
      }
    }
  };
  float4 pa[A_PER],pb[B_PER];
  uint4 pp=make_uint4(0,0,0,0);
  load(0,pa,pb,pp); store(0,pa,pb,pp); __syncthreads();
  int s=0;
  for(int kk=0;kk<K;kk+=BK) {
    if(kk+BK<K) load(kk+BK,pa,pb,pp);
    if(w<SW) {
      #pragma unroll (ALIGNED ? 16 : 4)
      for(int k=0;k<BK;++k) {
        const float4 v=*reinterpret_cast<const float4*>(&sb[s][k][sc]);
        const float bv[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int i=0;i<RM;i+=4) {
          const float4 u=*reinterpret_cast<const float4*>(&sa[s][k][sr*RM+i]);
          const float av[4]={u.x,u.y,u.z,u.w};
          #pragma unroll
          for(int ii=0;ii<4;++ii) {
            #pragma unroll
            for(int j=0;j<4;++j) acc[(i+ii)*4+j]=fmaf(av[ii],bv[j],acc[(i+ii)*4+j]);
          }
        }
      }
    } else {
      for(int sub=0;sub<BK;sub+=16) {
        const int k=(l%4)*2+sub, k16=(l%4)*2;
        unsigned ah[MT][4],al[MT][4];
        #pragma unroll
        for(int m=0;m<MT;++m) {
          const int r=(w-SW)*MT*16+m*16+l/4;
          split_f16x2(sa[s][k][r],sa[s][k+1][r],ah[m][0],al[m][0]);
          split_f16x2(sa[s][k][r+8],sa[s][k+1][r+8],ah[m][1],al[m][1]);
          split_f16x2(sa[s][k+8][r],sa[s][k+9][r],ah[m][2],al[m][2]);
          split_f16x2(sa[s][k+8][r+8],sa[s][k+9][r+8],ah[m][3],al[m][3]);
        }
        #pragma unroll
        for(int n=0;n<4;++n) {
          unsigned bh[2],bl[2];
          if(PACKED) {
            const int c=col+SN+n*8+l/4;
            uint4 v=make_uint4(0,0,0,0);
            if(STAGED) v=sp[s][n*32+l];
            else if(c<NP) v=P[((kk+sub)/16*NP+c)*4+l%4];
            bh[0]=v.x; bh[1]=v.y; bl[0]=v.z; bl[1]=v.w;
          } else {
            const float2 lo=*reinterpret_cast<const float2*>(&sm[s][n*8+l/4][k16]);
            const float2 hi=*reinterpret_cast<const float2*>(&sm[s][n*8+l/4][k16+8]);
            split_f16x2(lo.x,lo.y,bh[0],bl[0]); split_f16x2(hi.x,hi.y,bh[1],bl[1]);
          }
          #pragma unroll
          for(int m=0;m<MT;++m) {
            float* out=&acc[(m*4+n)*4];
            mma_fp16(out,ah[m],bh);
            if(CROSS) {
              mma_fp16(out+MC,ah[m],bl); mma_fp16(out+MC,al[m],bh);
            } else {
              float tmp[4]={0,0,0,0};
              mma_fp16(tmp,ah[m],bl); mma_fp16(tmp,al[m],bh);
              #pragma unroll
              for(int q=0;q<4;++q) out[q]=fmaf(tmp[q],1.0f/2048,out[q]);
            }
          }
        }
      }
    }
    if(kk+BK<K) {store(s^1,pa,pb,pp); s^=1; __syncthreads();}
  }
  if(w<SW) {
    #pragma unroll
    for(int i=0;i<RM;++i) {
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int r=row+sr*RM+i,c=col+sc+j;
        if(r<M && (ALIGNED || c<N)) C[r*N+c]=acc[i*4+j];
      }
    }
  } else {
    #pragma unroll
    for(int m=0;m<MT;++m) {
      #pragma unroll
      for(int n=0;n<4;++n) {
        #pragma unroll
        for(int q=0;q<4;++q) {
          const int r=row+(w-SW)*MT*16+m*16+l/4+(q/2)*8,c=col+SN+n*8+2*(l%4)+q%2;
          const int z=(m*4+n)*4+q;
          if(r<M && (ALIGNED || c<N)) C[r*N+c]=CROSS ? fmaf(acc[z+MC],1.0f/2048,acc[z]) : acc[z];
        }
      }
    }
  }
}

__device__ __forceinline__ void sync_simt_192() {
  asm volatile("barrier.cta.sync 1, 192;" ::: "memory");
}
__device__ __forceinline__ void sync_mma_64() {
  asm volatile("barrier.cta.sync 2, 64;" ::: "memory");
}

// Design-1 falsification variant: SIMT and MMA roles have disjoint A staging
// (A is loaded twice) and synchronize only within their own warp group.
// SIMT still uses FP32 B staging; MMA reads pre-packed B directly from global.
template<int BKC>
__device__ void column_hybrid_decoupled(const float* __restrict__ A,const float* __restrict__ B,
                                        const uint4* __restrict__ P,float* __restrict__ C,
                                        int M,int N,int K,int NP) {
  constexpr int BM=64, BK=BKC, SN=96, BN=128;
  __shared__ __align__(16) float sa_s[2][BK][BM+4];
  __shared__ __align__(16) float sb_s[2][BK][SN+4];
  __shared__ __align__(16) float sa_m[2][BK][BM+4];
  const int t=threadIdx.x,w=t/32,l=t%32,row=blockIdx.x*BM,col=blockIdx.y*BN;
  float acc[32];
  #pragma unroll
  for(int i=0;i<32;++i) acc[i]=0;
  if(w<6) {
    const int ts=t, sr=ts/(SN/4), sc=(ts%(SN/4))*4;
    constexpr int SA_ITERS=(BM*BK/4+191)/192, SB_ITERS=(BK*SN/4+191)/192;
    auto load_s=[&](int kk,float4 (&a)[SA_ITERS],float4 (&b)[SB_ITERS]) {
      #pragma unroll
      for(int i=0;i<SA_ITERS;++i) {
        const int q=ts+i*192, r=q/(BK/4), k=(q%(BK/4))*4;
        a[i]=make_float4(0,0,0,0);
        if(q<BM*BK/4 && row+r<M) a[i]=*reinterpret_cast<const float4*>(&A[(row+r)*K+kk+k]);
      }
      #pragma unroll
      for(int i=0;i<SB_ITERS;++i) {
        const int q=ts+i*192, k=q/(SN/4), c=(q%(SN/4))*4;
        b[i]=make_float4(0,0,0,0);
        if(q<BK*SN/4 && kk+k<K && col+c+3<N) b[i]=*reinterpret_cast<const float4*>(&B[(kk+k)*N+col+c]);
      }
    };
    auto store_s=[&](int s,const float4 (&a)[SA_ITERS],const float4 (&b)[SB_ITERS]) {
      #pragma unroll
      for(int i=0;i<SA_ITERS;++i) {
        const int q=ts+i*192, r=q/(BK/4), k=(q%(BK/4))*4;
        if(q<BM*BK/4) {
          sa_s[s][k][r]=a[i].x; sa_s[s][k+1][r]=a[i].y;
          sa_s[s][k+2][r]=a[i].z; sa_s[s][k+3][r]=a[i].w;
        }
      }
      #pragma unroll
      for(int i=0;i<SB_ITERS;++i) {
        const int q=ts+i*192, k=q/(SN/4), c=(q%(SN/4))*4;
        if(q<BK*SN/4) *reinterpret_cast<float4*>(&sb_s[s][k][c])=b[i];
      }
    };
    float4 pa[SA_ITERS],pb[SB_ITERS];
    load_s(0,pa,pb); store_s(0,pa,pb); sync_simt_192();
    int s=0;
    for(int kk=0;kk<K;kk+=BK) {
      if(kk+BK<K) load_s(kk+BK,pa,pb);
      #pragma unroll 4
      for(int k=0;k<BK;++k) {
        const float4 v=*reinterpret_cast<const float4*>(&sb_s[s][k][sc]);
        const float bv[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int i=0;i<8;i+=4) {
          const float4 u=*reinterpret_cast<const float4*>(&sa_s[s][k][sr*8+i]);
          const float av[4]={u.x,u.y,u.z,u.w};
          #pragma unroll
          for(int ii=0;ii<4;++ii) {
            #pragma unroll
            for(int j=0;j<4;++j) acc[(i+ii)*4+j]=fmaf(av[ii],bv[j],acc[(i+ii)*4+j]);
          }
        }
      }
      if(kk+BK<K) { store_s(s^1,pa,pb); s^=1; sync_simt_192(); }
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int r=row+sr*8+i, c=col+sc+j;
        if(r<M && c<N) C[r*N+c]=acc[i*4+j];
      }
    }
  } else {
    const int tm=t-192;
    constexpr int MA_ITERS=(BM*BK/4)/64;
    auto load_m=[&](int kk,float4 (&a)[MA_ITERS]) {
      #pragma unroll
      for(int i=0;i<MA_ITERS;++i) {
        const int q=tm+i*64, r=q/(BK/4), k=(q%(BK/4))*4;
        a[i]=make_float4(0,0,0,0);
        if(row+r<M) a[i]=*reinterpret_cast<const float4*>(&A[(row+r)*K+kk+k]);
      }
    };
    auto store_m=[&](int s,const float4 (&a)[MA_ITERS]) {
      #pragma unroll
      for(int i=0;i<MA_ITERS;++i) {
        const int q=tm+i*64, r=q/(BK/4), k=(q%(BK/4))*4;
        sa_m[s][k][r]=a[i].x; sa_m[s][k+1][r]=a[i].y;
        sa_m[s][k+2][r]=a[i].z; sa_m[s][k+3][r]=a[i].w;
      }
    };
    float4 pa[MA_ITERS];
    load_m(0,pa); store_m(0,pa); sync_mma_64();
    int s=0;
    for(int kk=0;kk<K;kk+=BK) {
      if(kk+BK<K) load_m(kk+BK,pa);
      for(int sub=0;sub<BK;sub+=16) {
        const int k=(l%4)*2+sub;
        unsigned ah[2][4],al[2][4];
        #pragma unroll
        for(int m=0;m<2;++m) {
          const int r=(w-6)*32+m*16+l/4;
          split_f16x2(sa_m[s][k][r],sa_m[s][k+1][r],ah[m][0],al[m][0]);
          split_f16x2(sa_m[s][k][r+8],sa_m[s][k+1][r+8],ah[m][1],al[m][1]);
          split_f16x2(sa_m[s][k+8][r],sa_m[s][k+9][r],ah[m][2],al[m][2]);
          split_f16x2(sa_m[s][k+8][r+8],sa_m[s][k+9][r+8],ah[m][3],al[m][3]);
        }
        #pragma unroll
        for(int n=0;n<4;++n) {
          unsigned bh[2],bl[2];
          const int c=col+SN+n*8+l/4;
          uint4 v=make_uint4(0,0,0,0);
          if(c<NP) v=P[((kk+sub)/16*NP+c)*4+l%4];
          bh[0]=v.x; bh[1]=v.y; bl[0]=v.z; bl[1]=v.w;
          #pragma unroll
          for(int m=0;m<2;++m) {
            float* out=&acc[(m*4+n)*4];
            float tmp[4]={0,0,0,0};
            mma_fp16(out,ah[m],bh);
            mma_fp16(tmp,ah[m],bl); mma_fp16(tmp,al[m],bh);
            #pragma unroll
            for(int q=0;q<4;++q) out[q]=fmaf(tmp[q],1.0f/2048,out[q]);
          }
        }
      }
      if(kk+BK<K) { store_m(s^1,pa); s^=1; sync_mma_64(); }
    }
    #pragma unroll
    for(int m=0;m<2;++m) {
      #pragma unroll
      for(int n=0;n<4;++n) {
        #pragma unroll
        for(int q=0;q<4;++q) {
          const int r=row+(w-6)*32+m*16+l/4+(q/2)*8, c=col+SN+n*8+2*(l%4)+q%2;
          if(r<M && c<N) C[r*N+c]=acc[(m*4+n)*4+q];
        }
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256,2) col64_aligned_decoupled(
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) {
 column_hybrid_decoupled<16>(A,B,P,C,M,N,K,NP); }
#define COL(NAME,BM,SW,PACK,CROSS,RES) \
extern "C" __global__ void __launch_bounds__(256,RES) NAME( \
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) { \
 column_hybrid<BM,SW,PACK,CROSS>(A,B,P,C,M,N,K,NP); }
COL(col64_s6_inline,64,6,false,false,3)
COL(col64_s6_packed,64,6,true,false,3)
COL(col64_s6_cross,64,6,true,true,2)
COL(col128_s6_inline,128,6,false,false,2)
COL(col128_s6_packed,128,6,true,false,2)
COL(col128_s4_inline,128,4,false,true,2)
COL(col128_s4_packed,128,4,true,true,2)
#define ALIGNED_COL(NAME,BM,PACK,RES) \
extern "C" __global__ void __launch_bounds__(256,RES) NAME( \
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) { \
 column_hybrid<BM,6,PACK,false,true>(A,B,P,C,M,N,K,NP); }
ALIGNED_COL(col128_aligned_inline,128,false,2)
ALIGNED_COL(col128_aligned_packed,128,true,2)
ALIGNED_COL(col64_aligned_inline,64,false,2)
ALIGNED_COL(col64_aligned_packed,64,true,2)
extern "C" __global__ void __launch_bounds__(256,2) col64_aligned_bk32(
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) {
 column_hybrid<64,6,true,false,true,false,false,32>(A,B,P,C,M,N,K,NP); }
extern "C" __global__ void __launch_bounds__(256,2) col64_aligned_swizzled(
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) {
 column_hybrid<64,6,true,false,true,false,true>(A,B,P,C,M,N,K,NP); }
#define STAGED_COL(NAME,BM) \
extern "C" __global__ void __launch_bounds__(256,2) NAME( \
 const float* A,const float* B,const uint4* P,float* C,int M,int N,int K,int NP) { \
 column_hybrid<BM,6,true,false,true,true>(A,B,P,C,M,N,K,NP); }
STAGED_COL(col64_aligned_staged,64)
STAGED_COL(col128_aligned_staged,128)
"""


def checked(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1:] if len(result) > 2 else result[1] if len(result) == 2 else None


def compile_kernels():
    from cuda.bindings import driver, nvrtc

    cap = torch.cuda.get_device_capability()
    src = baseline.HYBRID_NVRTC_SRC + SOURCE
    prog = checked(nvrtc.nvrtcCreateProgram(src.encode(), b"large_gemm.cu", 0, [], []))
    opts = [f"-arch=sm_{cap[0]}{cap[1]}".encode(), b"-std=c++17"]
    result = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log = b" " * checked(nvrtc.nvrtcGetProgramLogSize(prog))
    checked(nvrtc.nvrtcGetProgramLog(prog, log))
    if int(result[0]):
        raise RuntimeError(log.decode())
    cubin = b" " * checked(nvrtc.nvrtcGetCUBINSize(prog))
    checked(nvrtc.nvrtcGetCUBIN(prog, cubin))
    checked(nvrtc.nvrtcDestroyProgram(prog))
    module = checked(driver.cuModuleLoadData(cubin))
    kernels = {}
    attr = driver.CUfunction_attribute
    specs = [("hybrid_gemm_6x2_64_128", 64, 128), ("hybrid_gemm_64_32", 64, 96)]
    specs += [(name, 128, 64 if name.startswith("row64") else 128) for name in (
        "row128_inline", "row128_packed", "row128_cross", "row64_packed", "row64_cross")]
    specs += [(name, bm, sw * 16 + 32) for name, bm, sw in (
        ("col64_s6_inline", 64, 6), ("col64_s6_packed", 64, 6), ("col64_s6_cross", 64, 6),
        ("col128_s6_inline", 128, 6), ("col128_s6_packed", 128, 6),
        ("col128_s4_inline", 128, 4), ("col128_s4_packed", 128, 4))]
    specs.append(("row128_async", 128, 128))
    specs += [(f"col{bm}_aligned_{kind}", bm, 128) for bm in (64, 128) for kind in ("inline", "packed")]
    specs += [(f"col{bm}_aligned_staged", bm, 128) for bm in (64, 128)]
    specs.append(("col64_aligned_decoupled", 64, 128))
    specs.append(("col64_aligned_swizzled", 64, 128))
    specs.append(("col64_aligned_bk32", 64, 128))
    for name, bm, bn in specs:
        fn = checked(driver.cuModuleGetFunction(module, name.encode()))
        props = {key: int(checked(driver.cuFuncGetAttribute(getattr(attr, code), fn))) for key, code in (
            ("registers", "CU_FUNC_ATTRIBUTE_NUM_REGS"), ("shared_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"),
            ("local_bytes", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"))}
        props["active_blocks_per_sm"] = int(checked(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, 256, 0)))
        kernels[name] = dict(fn=fn, bm=bm, bn=bn, aligned="aligned" in name, **props)
    return module, kernels


def launch(info, a, b, p, c):
    from cuda.bindings import driver

    m, k = a.shape
    n = b.shape[1]
    if info.get("aligned") and (k % 16 or n % 128):
        raise ValueError("Aligned screen requires K divisible by 16 and N divisible by 128")
    args = [ctypes.c_void_p(t.data_ptr()) for t in (a, b, p, c)]
    args += [ctypes.c_int(v) for v in (m, n, k, triton.cdiv(n, 8) * 8)]
    argv = (ctypes.c_void_p * len(args))(*[ctypes.addressof(x) for x in args])
    checked(driver.cuLaunchKernel(info["fn"], triton.cdiv(m, info["bm"]), triton.cdiv(n, info["bn"]),
                                 1, 256, 1, 1, 0, driver.CUstream(torch.cuda.current_stream().cuda_stream),
                                 ctypes.addressof(argv), 0))


def paired_timings(calls, repeats=80, blocks=12):
    graphs = {}
    for name, fn in calls.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graphs[name] = graph
    times = {name: [] for name in graphs}
    # Reverse the full order every other block; compare the same operands and stream.
    for block in range(blocks):
        order = list(graphs) if block % 2 == 0 else list(reversed(graphs))
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / repeats)
    return {name: {"median_us": statistics.median(ts), "blocks_us": ts} for name, ts in times.items()}


def validation(kernels):
    """Check indexing/determinism, then measure (not assume) numerical range."""
    selected = {name: info for name, info in kernels.items() if not name.startswith("hybrid")}
    boundary_rows = []
    for m, k, n in ((1, 16, 128), (63, 48, 256), (129, 512, 128), (513, 80, 384),
                    (65, 17, 131), (129, 63, 257)):
        torch.manual_seed(m + k + n)
        a, b = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda")
        packed = torch.empty(triton.cdiv(k, 16) * triton.cdiv(n, 8) * 8 * 16,
                             dtype=torch.int32, device="cuda")
        pack_b(b, packed)
        ref, norm = a.double() @ b.double(), a.double().abs() @ b.double().abs()
        for name, info in selected.items():
            if info.get("aligned") and (k % 16 or n % 128):
                continue
            guard = torch.full((m * n + 16,), 12345.0, device="cuda")
            out = guard[8:-8].view(m, n)
            out.fill_(float("nan"))
            launch(info, a, b, packed, out)
            torch.cuda.synchronize()
            error = baseline.calc_errors(out, ref, norm)
            torch.testing.assert_close(out.double(), ref, rtol=0, atol=5e-6 * float(ref.abs().max()))
            expected = out.clone()
            launch(info, a, b, packed, out)
            assert torch.equal(expected, out), (name, "repeat differs")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pack_b(b, packed)
                launch(info, a, b, packed, out)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(expected, out), (name, "graph replay differs")
            assert bool((guard[:8] == 12345).all() and (guard[-8:] == 12345).all()), name
            boundary_rows.append({"kernel": name, "shape_mkn": [m, k, n], "errors": error,
                                  "repeat_bitwise": True, "graph_bitwise": True, "canaries_intact": True})

    stress = []
    for scale in (2.0**-30, 2.0**20):
        torch.manual_seed(7)
        a = torch.randn(129, 64, device="cuda") * scale
        b = torch.randn(64, 128, device="cuda") / scale
        packed = torch.empty(64 * 128, dtype=torch.int32, device="cuda")
        pack_b(b, packed)
        ref, norm = a.double() @ b.double(), a.double().abs() @ b.double().abs()
        for name, info in selected.items():
            out = torch.empty(129, 128, device="cuda")
            launch(info, a, b, packed, out)
            finite = bool(torch.isfinite(out).all())
            stress.append({"kernel": name, "a_scale": scale, "finite": finite,
                           "errors": baseline.calc_errors(out, ref, norm) if finite else None})

    # Record real operands without changing any production implementation.
    import numpy as np
    from ase.io import read

    from fastmace import ForceField, load
    from fastmace.config import RewriteConfig
    from fastmace.kernels.triton import segment_gemm, tf32x3_gemm

    structure_path = Path(os.environ["FASTMACE_STRUCTURES"]) / "Structures/Benchmark/R2SCAN/NaCl.cif"
    atoms = read(structure_path).repeat((3, 3, 3))
    atoms.set_pbc(True)
    atoms.positions += np.random.default_rng(20261024).uniform(-0.05, 0.05, atoms.positions.shape)
    field = ForceField(load("mh1-r2scan"), atoms, precision="float32", rewrites=RewriteConfig(tf32x3_gemms="on"))
    field.evaluate(atoms)
    ops = []
    original_dense, original_joined = tf32x3_gemm._launch, segment_gemm._gemm

    def dense(a, b, out):
        original_dense(a, b, out)
        ops.append((a.detach().clone(), b.detach().clone()))

    def joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
        original_joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision)
        if precision == "tf32x3":
            left = a.reshape(-1)[a_offset:].as_strided((rows // dim, dim, b.shape[0]),
                                                      (a_n, a_m, 1)).reshape(rows, b.shape[0])
            ops.append((left.detach().clone(), b.detach().clone()))

    tf32x3_gemm._launch, segment_gemm._gemm = dense, joined
    try:
        field.evaluate(atoms)
        torch.cuda.synchronize()
    finally:
        tf32x3_gemm._launch, segment_gemm._gemm = original_dense, original_joined
    real_rows = []
    for index, (a, b) in enumerate(ops):
        a, b = a.contiguous(), b.contiguous()
        m, k = a.shape
        n = b.shape[1]
        ref, norm = a.double() @ b.double(), a.double().abs() @ b.double().abs()
        ieee = baseline.calc_errors(a @ b, ref, norm)
        packed = torch.empty(triton.cdiv(k, 16) * triton.cdiv(n, 8) * 8 * 16,
                             dtype=torch.int32, device="cuda")
        pack_b(b, packed)
        for name, info in selected.items():
            if info.get("aligned") and (k % 16 or n % 128):
                continue
            out = torch.empty(m, n, device="cuda")
            launch(info, a, b, packed, out)
            finite = bool(torch.isfinite(out).all())
            errors = baseline.calc_errors(out, ref, norm) if finite else None
            ratio = errors["cancel_scaled"] / ieee["cancel_scaled"] if finite and ieee["cancel_scaled"] else None
            real_rows.append({"product": index, "kernel": name, "shape_mkn": [m, k, n],
                              "finite": finite, "errors": errors, "ieee": ieee,
                              "cancel_over_ieee": ratio})
    checkpoint = Path(os.environ["FASTMACE_MODELS"]) / "mace-mh-1.model"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"boundaries": boundary_rows, "dynamic_range": stress, "real_products": real_rows,
            "captured_products": len(ops), "model": "mh1-r2scan", "checkpoint_sha256": digest.hexdigest(),
            "structure_sha256": hashlib.sha256(structure_path.read_bytes()).hexdigest(),
            "structure": "NaCl 3x3x3, 216 atoms, rattle +/-0.05 A, seed 20261024",
            "capture_plan": field.evaluation_plan(atoms).describe(full=True)}


def run(args):
    from cuda.bindings import driver

    torch.zeros(1, device="cuda")
    module, kernels = compile_kernels()
    if args.kernels:
        selected = set(args.kernels.split(","))
        unknown = selected - kernels.keys()
        if unknown:
            raise ValueError(f"Unknown kernels: {unknown}")
        kernels = {name: info for name, info in kernels.items() if name in selected or name.startswith("hybrid")}
    root = Path(__file__).resolve().parents[2]
    result = {"metadata": {
        "utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "triton": triton.__version__, "python": platform.python_version(),
        "environment_python": sys.executable, "cuda": torch.version.cuda,
        "driver": int(checked(driver.cuDriverGetVersion())),
        "sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
        "precision": "FP32 SIMT + scaled-residual 3xFP16 MMA, TF32 disabled",
        "scope": "synthetic dense GEMM, not MD", "seed": 0,
        "protocol": "CUDA graphs, 80 GEMMs/replay, 12 alternating-order blocks, B packing reported explicitly",
    }, "kernels": {n: {k: v for k, v in p.items() if k != "fn"} for n, p in kernels.items()}, "shapes": []}
    print(json.dumps(result["kernels"]), flush=True)
    shapes = [(1512, 512, 512), (5832, 512, 512), (13608, 512, 512), (1944, 1024, 1024)]
    if args.quick:
        shapes = shapes[:2]
    if args.extended:
        shapes += [(32768, 512, 512), (5832, 1024, 1024)]
    for m, k, n in shapes:
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        out = torch.empty(m, n, device="cuda")
        packed = torch.empty(triton.cdiv(k, 16) * triton.cdiv(n, 8) * 8 * 16, dtype=torch.int32, device="cuda")
        pack_b(b, packed)
        ref = a.double() @ b.double()
        norm = a.double().abs() @ b.double().abs()
        calls = {"cublas": lambda: torch.mm(a, b, out=out)}
        for name, info in kernels.items():
            if name.startswith("hybrid"):
                calls[name] = lambda info=info: baseline.launch_hybrid_nvrtc(info, a, b, out)
            else:
                calls[name] = lambda info=info: launch(info, a, b, packed, out)
        errors = {}
        for name, fn in calls.items():
            out.fill_(float("nan"))
            fn()
            torch.cuda.synchronize()
            if not torch.isfinite(out).all():
                raise RuntimeError(f"Nonfinite output: {name}, {(m, k, n)}")
            errors[name] = baseline.calc_errors(out, ref, norm)
        for name, info in kernels.items():
            if not name.startswith("hybrid") and "inline" not in name:
                def with_pack(info=info):
                    pack_b(b, packed)
                    launch(info, a, b, packed, out)
                calls[name + "+pack"] = with_pack
        times = paired_timings(calls)
        pack_time = paired_timings({"pack_b": lambda: pack_b(b, packed)})
        row = {"shape_mkn": [m, k, n], "timings": times, "packing": pack_time, "errors": errors}
        result["shapes"].append(row)
        print(json.dumps({"shape": [m, k, n], "us": {x: round(y["median_us"], 3) for x, y in times.items()},
                          "errors": {x: y["max_rel"] for x, y in errors.items()}}), flush=True)
    if args.validate:
        result["validation"] = validation(kernels)
        print(json.dumps({"boundary_cases": len(result["validation"]["boundaries"]),
                          "captured_real_products": result["validation"]["captured_products"]}), flush=True)
    checked(driver.cuModuleUnload(module))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kernels", help="Comma-separated screen variants; old baselines are always included")
    parser.add_argument("--extended", action="store_true", help="Also screen 32768 rows and a larger 1024-channel case")
    parser.add_argument("--validate", action="store_true",
                        help="Run boundary, determinism, range and MH-1 operand checks")
    options = parser.parse_args()
    if options.output.exists():
        parser.error("output already exists; choose a new run path")
    run(options)
