// oeq_projected_backward_wprime (FastMACE opt-9 OPT9-T03), adapted from the vendored
// oeq_projected_backward (OpenEquivariance PR 223, BSD 3-Clause, see NOTICE.md).
using int64_t = signed long long;
using int32_t = int;
using scalar_t = float;
using acc_t = float;

constexpr int INPUT_DIM = 512;
constexpr int EDGE_DIM = 16;
constexpr int OUTPUT_DIM = 5120;
constexpr int WEIGHT_DIM = 1280;
constexpr int CHANNEL_DIM = 128;
constexpr int LOGICAL_GROUP_SIZE =
    32;
constexpr int TERMS = 6;
constexpr int PADDED_TERMS = 6;  // a q-major G' row: TERMS, then zero terms (SLOPE_VECTOR)
constexpr unsigned int FULL_MASK = 0xffffffffu;



// One logical thread group owns an edge; each lane owns channels separated by LOGICAL_GROUP_SIZE.
// dx and dsh as in oeq_projected_backward; in place of dweights, dr[e] = sum over the edge's weights of
// dweights * w'(r_e), w' from the derivative table gprime [pairs * pieces, TERMS, WEIGHT_DIM] (q-major with
// SLOPE_VECTOR: [pairs * pieces, WEIGHT_DIM, PADDED_TERMS]).
extern "C" __global__ void oeq_projected_backward_wprime(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* __restrict__ dout0,
    const scalar_t* __restrict__ dout1,
    const scalar_t* __restrict__ dout2,
    const scalar_t* __restrict__ dout3,
    scalar_t* dx,
    scalar_t* dsh,
    const scalar_t* lengths,
    const int32_t* pairs,
    const scalar_t* __restrict__ gprime,
    const scalar_t* starts,
    const scalar_t* scales,
    const int32_t* bins,
    scalar_t r_lo,
    scalar_t bin_scale,
    scalar_t r_max,
    int32_t n_bins,
    int32_t pieces,
    scalar_t* dr) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count) {
        // An edge with an out-of-range index contributes nothing: its dsh row and dr are written as zeros (the
        // wrappers allocate dsh with empty_like, so a skipped row would hold garbage; kernel review, 5 November 2026).
        for (int c = lane; c < 3; c += LOGICAL_GROUP_SIZE) dsh[e * 3 + c] = scalar_t(0);
        if (lane == 0) dr[e] = scalar_t(0);
        return;
    }

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    // The radial tables' piece of r and its Chebyshev argument; no slope at or beyond the cutoff, below r_lo.
    const scalar_t r = lengths[e];
    const bool sloped = r < r_max && r >= r_lo;
    const scalar_t clamped = r > r_lo ? r : r_lo;
    int32_t bin = int32_t((clamped - r_lo) * bin_scale);
    bin = bin < n_bins - 1 ? bin : n_bins - 1;
    const int32_t piece = bins[bin];
    const scalar_t t = (clamped - starts[piece]) * scales[piece] - scalar_t(1);
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;
    // HORNER: the value table holds power-basis coefficients in t; w and dw/dt by one dual Horner pass per weight.
    const bool inside = r < r_max;
    const scalar_t sc = scales[piece];


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
{
const int weight_index=0+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+0+channel];
const float h=q0, hx=0,hy=0,hz=0, hhx=0,hhy=0,hhz=0;
const float u0=h,f=x0*h,gx=x0*hx,gy=x0*hy,gz=x0*hz;
dx0=fmaf(radial_weight,u0,dx0);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-0.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=128+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+128+channel];
const float h=q0, hx=0,hy=0,hz=0, hhx=0,hhy=0,hhz=0;
const float A=xx*nx+xy*ny+xz*nz,B=xx*hx+xy*hy+xz*hz;
const float ik=1.f;
const float ux=nx*h-ik*hx,uy=ny*h-ik*hy,uz=nz*h-ik*hz,f=A*h-ik*B;
const float gx=xx*h+A*hx-ik*(2.f*nx*B+hhx),gy=xy*h+A*hy-ik*(2.f*ny*B+hhy),gz=xz*h+A*hz-ik*(2.f*nz*B+hhz);
dxx=fmaf(radial_weight,ux,dxx);
dxy=fmaf(radial_weight,uy,dxy);
dxz=fmaf(radial_weight,uz,dxz);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-1.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=256+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+256+channel];
const float q1=dout0[int64_t(receiver)*5120+384+channel];
const float q2=dout0[int64_t(receiver)*5120+512+channel];
const float h=q0*nx+q1*ny+q2*nz;
const float hx=q0,hy=q1,hz=q2,hhx=0,hhy=0,hhz=0;
const float u0=h,f=x0*h,gx=x0*hx,gy=x0*hy,gz=x0*hz;
dx0=fmaf(radial_weight,u0,dx0);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-1.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=384+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+640+channel];
const float q1=dout0[int64_t(receiver)*5120+768+channel];
const float q2=dout0[int64_t(receiver)*5120+896+channel];
const float h=q0*nx+q1*ny+q2*nz;
const float hx=q0,hy=q1,hz=q2,hhx=0,hhy=0,hhz=0;
const float ux=hx,uy=hy,uz=hz,f=xx*ux+xy*uy+xz*uz,gx=hhx,gy=hhy,gz=hhz;
dxx=fmaf(radial_weight,ux,dxx);
dxy=fmaf(radial_weight,uy,dxy);
dxz=fmaf(radial_weight,uz,dxz);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-0.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=512+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+1024+channel];
const float q1=dout0[int64_t(receiver)*5120+1152+channel];
const float q2=dout0[int64_t(receiver)*5120+1280+channel];
const float h=q0*nx+q1*ny+q2*nz;
const float hx=q0,hy=q1,hz=q2,hhx=0,hhy=0,hhz=0;
const float A=xx*nx+xy*ny+xz*nz,B=xx*hx+xy*hy+xz*hz;
const float ik=0.33333333333333331f;
const float ux=nx*h-ik*hx,uy=ny*h-ik*hy,uz=nz*h-ik*hz,f=A*h-ik*B;
const float gx=xx*h+A*hx-ik*(2.f*nx*B+hhx),gy=xy*h+A*hy-ik*(2.f*ny*B+hhy),gz=xz*h+A*hz-ik*(2.f*nz*B+hhz);
dxx=fmaf(radial_weight,ux,dxx);
dxy=fmaf(radial_weight,uy,dxy);
dxz=fmaf(radial_weight,uz,dxz);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-2.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=640+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+1408+channel];
const float q1=dout0[int64_t(receiver)*5120+1536+channel];
const float q2=dout0[int64_t(receiver)*5120+1664+channel];
const float q3=dout0[int64_t(receiver)*5120+1792+channel];
const float q4=dout0[int64_t(receiver)*5120+1920+channel];
const float qzz=-q0-q3;
const float vx=q0*nx+q1*ny+q2*nz,vy=q1*nx+q3*ny+q4*nz,vz=q2*nx+q4*ny+qzz*nz;
const float h=nx*vx+ny*vy+nz*vz,hx=2.f*vx,hy=2.f*vy,hz=2.f*vz;
const float hhx=2.f*(q0*xx+q1*xy+q2*xz),hhy=2.f*(q1*xx+q3*xy+q4*xz),hhz=2.f*(q2*xx+q4*xy+qzz*xz);
const float u0=h,f=x0*h,gx=x0*hx,gy=x0*hy,gz=x0*hz;
dx0=fmaf(radial_weight,u0,dx0);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-2.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=768+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+2048+channel];
const float q1=dout0[int64_t(receiver)*5120+2176+channel];
const float q2=dout0[int64_t(receiver)*5120+2304+channel];
const float q3=dout0[int64_t(receiver)*5120+2432+channel];
const float q4=dout0[int64_t(receiver)*5120+2560+channel];
const float qzz=-q0-q3;
const float vx=q0*nx+q1*ny+q2*nz,vy=q1*nx+q3*ny+q4*nz,vz=q2*nx+q4*ny+qzz*nz;
const float h=nx*vx+ny*vy+nz*vz,hx=2.f*vx,hy=2.f*vy,hz=2.f*vz;
const float hhx=2.f*(q0*xx+q1*xy+q2*xz),hhy=2.f*(q1*xx+q3*xy+q4*xz),hhz=2.f*(q2*xx+q4*xy+qzz*xz);
const float ux=hx,uy=hy,uz=hz,f=xx*ux+xy*uy+xz*uz,gx=hhx,gy=hhy,gz=hhz;
dxx=fmaf(radial_weight,ux,dxx);
dxy=fmaf(radial_weight,uy,dxy);
dxz=fmaf(radial_weight,uz,dxz);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-1.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=896+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+2688+channel];
const float q1=dout0[int64_t(receiver)*5120+2816+channel];
const float q2=dout0[int64_t(receiver)*5120+2944+channel];
const float q3=dout0[int64_t(receiver)*5120+3072+channel];
const float q4=dout0[int64_t(receiver)*5120+3200+channel];
const float qzz=-q0-q3;
const float vx=q0*nx+q1*ny+q2*nz,vy=q1*nx+q3*ny+q4*nz,vz=q2*nx+q4*ny+qzz*nz;
const float h=nx*vx+ny*vy+nz*vz,hx=2.f*vx,hy=2.f*vy,hz=2.f*vz;
const float hhx=2.f*(q0*xx+q1*xy+q2*xz),hhy=2.f*(q1*xx+q3*xy+q4*xz),hhz=2.f*(q2*xx+q4*xy+qzz*xz);
const float A=xx*nx+xy*ny+xz*nz,B=xx*hx+xy*hy+xz*hz;
const float ik=0.20000000000000001f;
const float ux=nx*h-ik*hx,uy=ny*h-ik*hy,uz=nz*h-ik*hz,f=A*h-ik*B;
const float gx=xx*h+A*hx-ik*(2.f*nx*B+hhx),gy=xy*h+A*hy-ik*(2.f*ny*B+hhy),gz=xz*h+A*hz-ik*(2.f*nz*B+hhz);
dxx=fmaf(radial_weight,ux,dxx);
dxy=fmaf(radial_weight,uy,dxy);
dxz=fmaf(radial_weight,uz,dxz);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-3.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=1024+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+3328+channel];
const float q1=dout0[int64_t(receiver)*5120+3456+channel];
const float q2=dout0[int64_t(receiver)*5120+3584+channel];
const float q3=dout0[int64_t(receiver)*5120+3712+channel];
const float q4=dout0[int64_t(receiver)*5120+3840+channel];
const float q5=dout0[int64_t(receiver)*5120+3968+channel];
const float q6=dout0[int64_t(receiver)*5120+4096+channel];
const float q7=-q0-q3,q8=-q1-q5;
const float Qxx=q0*nx+q1*ny+q2*nz,Qxy=q1*nx+q3*ny+q4*nz,Qxz=q2*nx+q4*ny+q7*nz;
const float Qyy=q3*nx+q5*ny+q6*nz,Qyz=q4*nx+q6*ny+q8*nz,Qzz=-Qxx-Qyy;
const float vx=Qxx*nx+Qxy*ny+Qxz*nz,vy=Qxy*nx+Qyy*ny+Qyz*nz,vz=Qxz*nx+Qyz*ny+Qzz*nz;
const float h=nx*vx+ny*vy+nz*vz,hx=3.f*vx,hy=3.f*vy,hz=3.f*vz;
const float hhx=6.f*(Qxx*xx+Qxy*xy+Qxz*xz),hhy=6.f*(Qxy*xx+Qyy*xy+Qyz*xz),hhz=6.f*(Qxz*xx+Qyz*xy+Qzz*xz);
const float u0=h,f=x0*h,gx=x0*hx,gy=x0*hy,gz=x0*hz;
dx0=fmaf(radial_weight,u0,dx0);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-3.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
{
const int weight_index=1152+channel;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }

const float q0=dout0[int64_t(receiver)*5120+4224+channel];
const float q1=dout0[int64_t(receiver)*5120+4352+channel];
const float q2=dout0[int64_t(receiver)*5120+4480+channel];
const float q3=dout0[int64_t(receiver)*5120+4608+channel];
const float q4=dout0[int64_t(receiver)*5120+4736+channel];
const float q5=dout0[int64_t(receiver)*5120+4864+channel];
const float q6=dout0[int64_t(receiver)*5120+4992+channel];
const float q7=-q0-q3,q8=-q1-q5;
const float Qxx=q0*nx+q1*ny+q2*nz,Qxy=q1*nx+q3*ny+q4*nz,Qxz=q2*nx+q4*ny+q7*nz;
const float Qyy=q3*nx+q5*ny+q6*nz,Qyz=q4*nx+q6*ny+q8*nz,Qzz=-Qxx-Qyy;
const float vx=Qxx*nx+Qxy*ny+Qxz*nz,vy=Qxy*nx+Qyy*ny+Qyz*nz,vz=Qxz*nx+Qyz*ny+Qzz*nz;
const float h=nx*vx+ny*vy+nz*vz,hx=3.f*vx,hy=3.f*vy,hz=3.f*vz;
const float hhx=6.f*(Qxx*xx+Qxy*xy+Qxz*xz),hhy=6.f*(Qxy*xx+Qyy*xy+Qyz*xz),hhz=6.f*(Qxz*xx+Qyz*xy+Qzz*xz);
const float ux=hx,uy=hy,uz=hz,f=xx*ux+xy*uy+xz*uz,gx=hhx,gy=hhy,gz=hhz;
dxx=fmaf(radial_weight,ux,dxx);
dxy=fmaf(radial_weight,uy,dxy);
dxz=fmaf(radial_weight,uz,dxz);
const float wr=radial_weight*ri;
const float radial=(sloped ? radial_dt*sc : 0.f)*f-2.f*wr*f;
fx=fmaf(wr,gx,fmaf(radial,nx,fx));
fy=fmaf(wr,gy,fmaf(radial,ny,fy));
fz=fmaf(wr,gz,fmaf(radial,nz,fz));
}
        atomicAdd(dx+input_base+channel,dx0);
        atomicAdd(dx+input_base+128+channel,dxx);
        atomicAdd(dx+input_base+256+channel,dxy);
        atomicAdd(dx+input_base+384+channel,dxz);
    }
    for(int o=16;o;o>>=1) {fx+=__shfl_down_sync(FULL_MASK,fx,o);fy+=__shfl_down_sync(FULL_MASK,fy,o);fz+=__shfl_down_sync(FULL_MASK,fz,o);}
    if(lane==0){dsh[3*e]=fx;dsh[3*e+1]=fy;dsh[3*e+2]=fz;dr[e]=0;}
}
