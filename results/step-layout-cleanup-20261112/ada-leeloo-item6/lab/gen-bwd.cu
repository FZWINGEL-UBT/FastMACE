// generated backward: 94 terms, 94 monomials, 13 groups
extern "C" __global__ void __launch_bounds__(128) sc0_bwd(
    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,
    const int* __restrict__ sp, const float* __restrict__ g, float* __restrict__ dm,
    float* __restrict__ dd) {
  const int a = blockIdx.x, ch = threadIdx.x;
  const float* xr = m + (long long)a * 2048 + ch;
  const float* wr = w + (long long)sp[a] * 1664 + ch;
  const float* gr = g + (long long)a * 128 + ch;
  const float s = 1.0f / (d[a] + 1.0f);
  const float x0 = __ldg(xr + 0) * s;
  const float x1 = __ldg(xr + 128) * s;
  const float x2 = __ldg(xr + 256) * s;
  const float x3 = __ldg(xr + 384) * s;
  const float x4 = __ldg(xr + 512) * s;
  const float x5 = __ldg(xr + 640) * s;
  const float x6 = __ldg(xr + 768) * s;
  const float x7 = __ldg(xr + 896) * s;
  const float x8 = __ldg(xr + 1024) * s;
  const float x9 = __ldg(xr + 1152) * s;
  const float x10 = __ldg(xr + 1280) * s;
  const float x11 = __ldg(xr + 1408) * s;
  const float x12 = __ldg(xr + 1536) * s;
  const float x13 = __ldg(xr + 1664) * s;
  const float x14 = __ldg(xr + 1792) * s;
  const float x15 = __ldg(xr + 1920) * s;
  const float g0 = __ldg(gr + 0);
  const float q0_0 = __ldg(wr + 0) * g0;
  const float q1_0 = __ldg(wr + 128) * g0;
  const float q2_0 = __ldg(wr + 256) * g0;
  const float q3_0 = __ldg(wr + 384) * g0;
  const float q4_0 = __ldg(wr + 512) * g0;
  const float q5_0 = __ldg(wr + 640) * g0;
  const float q6_0 = __ldg(wr + 768) * g0;
  const float q7_0 = __ldg(wr + 896) * g0;
  const float q8_0 = __ldg(wr + 1024) * g0;
  const float q9_0 = __ldg(wr + 1152) * g0;
  const float q10_0 = __ldg(wr + 1280) * g0;
  const float q11_0 = __ldg(wr + 1408) * g0;
  const float q12_0 = __ldg(wr + 1536) * g0;
  float dx0 = 0.0f;
  float dx1 = 0.0f;
  float dx2 = 0.0f;
  float dx3 = 0.0f;
  float dx4 = 0.0f;
  float dx5 = 0.0f;
  float dx6 = 0.0f;
  float dx7 = 0.0f;
  float dx8 = 0.0f;
  float dx9 = 0.0f;
  float dx10 = 0.0f;
  float dx11 = 0.0f;
  float dx12 = 0.0f;
  float dx13 = 0.0f;
  float dx14 = 0.0f;
  float dx15 = 0.0f;
    dx0 = fmaf(0x1.0000000000000p+0f, q0_0, dx0);
  { const float p = x0 * x0; float H = 0.0f;
    H = 0x1.0000000000000p-1f * q1_0;
  { float h;
    h = 0x1.6a09e60000000p-2f * q5_0;
    dx0 = fmaf(h, p, dx0);
    H = fmaf(h, x0, H); }
    dx0 = fmaf(H, x0, dx0); dx0 = fmaf(H, x0, dx0); }
  { const float p = x0 * x1; float H = 0.0f;
  { float h;
    h = 0x1.6a09e60000000p-2f * q6_0;
    dx1 = fmaf(h, p, dx1);
    H = h * x1; }
    dx0 = fmaf(H, x1, dx0); dx1 = fmaf(H, x0, dx1); }
  { const float p = x0 * x2; float H = 0.0f;
  { float h;
    h = 0x1.6a09e60000000p-2f * q6_0;
    dx2 = fmaf(h, p, dx2);
    H = h * x2; }
    dx0 = fmaf(H, x2, dx0); dx2 = fmaf(H, x0, dx2); }
  { const float p = x0 * x3; float H = 0.0f;
  { float h;
    h = 0x1.6a09e60000000p-2f * q6_0;
    dx3 = fmaf(h, p, dx3);
    H = h * x3; }
    dx0 = fmaf(H, x3, dx0); dx3 = fmaf(H, x0, dx3); }
  { const float p = x0 * x4; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q7_0;
    dx4 = fmaf(h, p, dx4);
    H = h * x4; }
    dx0 = fmaf(H, x4, dx0); dx4 = fmaf(H, x0, dx4); }
  { const float p = x0 * x5; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q7_0;
    dx5 = fmaf(h, p, dx5);
    H = h * x5; }
    dx0 = fmaf(H, x5, dx0); dx5 = fmaf(H, x0, dx5); }
  { const float p = x0 * x6; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q7_0;
    dx6 = fmaf(h, p, dx6);
    H = h * x6; }
    dx0 = fmaf(H, x6, dx0); dx6 = fmaf(H, x0, dx6); }
  { const float p = x0 * x7; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q7_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
    dx0 = fmaf(H, x7, dx0); dx7 = fmaf(H, x0, dx7); }
  { const float p = x0 * x8; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q7_0;
    dx8 = fmaf(h, p, dx8);
    H = h * x8; }
    dx0 = fmaf(H, x8, dx0); dx8 = fmaf(H, x0, dx8); }
  { const float p = x0 * x9; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
    dx0 = fmaf(H, x9, dx0); dx9 = fmaf(H, x0, dx9); }
  { const float p = x0 * x10; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx0 = fmaf(H, x10, dx0); dx10 = fmaf(H, x0, dx10); }
  { const float p = x0 * x11; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx0 = fmaf(H, x11, dx0); dx11 = fmaf(H, x0, dx11); }
  { const float p = x0 * x12; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
    dx0 = fmaf(H, x12, dx0); dx12 = fmaf(H, x0, dx12); }
  { const float p = x0 * x13; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx0 = fmaf(H, x13, dx0); dx13 = fmaf(H, x0, dx13); }
  { const float p = x0 * x14; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx0 = fmaf(H, x14, dx0); dx14 = fmaf(H, x0, dx14); }
  { const float p = x0 * x15; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q8_0;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx0 = fmaf(H, x15, dx0); dx15 = fmaf(H, x0, dx15); }
  { const float p = x1 * x1; float H = 0.0f;
    H = 0x1.279a740000000p-2f * q2_0;
  { float h;
    h = -0x1.c9f25c0000000p-4f * q9_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.8c97f00000000p-3f * q9_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
    dx1 = fmaf(H, x1, dx1); dx1 = fmaf(H, x1, dx1); }
  { const float p = x1 * x2; float H = 0.0f;
  { float h;
    h = 0x1.8c97f00000000p-2f * q9_0;
    dx5 = fmaf(h, p, dx5);
    H = h * x5; }
    dx1 = fmaf(H, x2, dx1); dx2 = fmaf(H, x1, dx2); }
  { const float p = x1 * x3; float H = 0.0f;
  { float h;
    h = 0x1.8c97f00000000p-2f * q9_0;
    dx4 = fmaf(h, p, dx4);
    H = h * x4; }
    dx1 = fmaf(H, x3, dx1); dx3 = fmaf(H, x1, dx3); }
  { const float p = x1 * x4; float H = 0.0f;
  { float h;
    h = -0x1.e990ce0000000p-5f * q10_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = -0x1.da05160000000p-3f * q10_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x4, dx1); dx4 = fmaf(H, x1, dx4); }
  { const float p = x1 * x5; float H = 0.0f;
  { float h;
    h = -0x1.2bcbe60000000p-3f * q10_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = -0x1.83091e0000000p-3f * q10_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx1 = fmaf(H, x5, dx1); dx5 = fmaf(H, x1, dx5); }
  { const float p = x1 * x6; float H = 0.0f;
  { float h;
    h = 0x1.a7f9ee0000000p-3f * q10_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx1 = fmaf(H, x6, dx1); dx6 = fmaf(H, x1, dx6); }
  { const float p = x1 * x7; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q10_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx1 = fmaf(H, x7, dx1); dx7 = fmaf(H, x1, dx7); }
  { const float p = x1 * x8; float H = 0.0f;
  { float h;
    h = 0x1.da05160000000p-3f * q10_0;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.e990ce0000000p-5f * q10_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx1 = fmaf(H, x8, dx1); dx8 = fmaf(H, x1, dx8); }
  { const float p = x2 * x2; float H = 0.0f;
    H = 0x1.279a740000000p-2f * q2_0;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q9_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
    dx2 = fmaf(H, x2, dx2); dx2 = fmaf(H, x2, dx2); }
  { const float p = x2 * x3; float H = 0.0f;
  { float h;
    h = 0x1.8c97f00000000p-2f * q9_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
    dx2 = fmaf(H, x3, dx2); dx3 = fmaf(H, x2, dx3); }
  { const float p = x2 * x4; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q10_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx2 = fmaf(H, x4, dx2); dx4 = fmaf(H, x2, dx4); }
  { const float p = x2 * x5; float H = 0.0f;
  { float h;
    h = 0x1.e990ce0000000p-3f * q10_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx2 = fmaf(H, x5, dx2); dx5 = fmaf(H, x2, dx5); }
  { const float p = x2 * x6; float H = 0.0f;
  { float h;
    h = 0x1.03a1a20000000p-2f * q10_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
    dx2 = fmaf(H, x6, dx2); dx6 = fmaf(H, x2, dx6); }
  { const float p = x2 * x7; float H = 0.0f;
  { float h;
    h = 0x1.e990ce0000000p-3f * q10_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx2 = fmaf(H, x7, dx2); dx7 = fmaf(H, x2, dx7); }
  { const float p = x2 * x8; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q10_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx2 = fmaf(H, x8, dx2); dx8 = fmaf(H, x2, dx8); }
  { const float p = x3 * x3; float H = 0.0f;
    H = 0x1.279a740000000p-2f * q2_0;
  { float h;
    h = -0x1.c9f25c0000000p-4f * q9_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.8c97f00000000p-3f * q9_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
    dx3 = fmaf(H, x3, dx3); dx3 = fmaf(H, x3, dx3); }
  { const float p = x3 * x4; float H = 0.0f;
  { float h;
    h = 0x1.da05160000000p-3f * q10_0;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = -0x1.e990ce0000000p-5f * q10_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx3 = fmaf(H, x4, dx3); dx4 = fmaf(H, x3, dx4); }
  { const float p = x3 * x5; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q10_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx3 = fmaf(H, x5, dx3); dx5 = fmaf(H, x3, dx5); }
  { const float p = x3 * x6; float H = 0.0f;
  { float h;
    h = 0x1.a7f9ee0000000p-3f * q10_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx3 = fmaf(H, x6, dx3); dx6 = fmaf(H, x3, dx6); }
  { const float p = x3 * x7; float H = 0.0f;
  { float h;
    h = -0x1.2bcbe60000000p-3f * q10_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = 0x1.83091e0000000p-3f * q10_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx3 = fmaf(H, x7, dx3); dx7 = fmaf(H, x3, dx7); }
  { const float p = x3 * x8; float H = 0.0f;
  { float h;
    h = -0x1.e990ce0000000p-5f * q10_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.da05160000000p-3f * q10_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx3 = fmaf(H, x8, dx3); dx8 = fmaf(H, x3, dx8); }
  { const float p = x4 * x4; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q3_0;
  { float h;
    h = -0x1.03a1a20000000p-2f * q11_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
    dx4 = fmaf(H, x4, dx4); dx4 = fmaf(H, x4, dx4); }
  { const float p = x4 * x5; float H = 0.0f;
  { float h;
    h = 0x1.c1b1dc0000000p-2f * q11_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
    dx4 = fmaf(H, x5, dx4); dx5 = fmaf(H, x4, dx5); }
  { const float p = x4 * x9; float H = 0.0f;
  { float h;
    h = -0x1.11acee0000000p-3f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx4 = fmaf(H, x9, dx4); dx9 = fmaf(H, x4, dx9); }
  { const float p = x4 * x10; float H = 0.0f;
  { float h;
    h = -0x1.11acee0000000p-2f * q12_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
    dx4 = fmaf(H, x10, dx4); dx10 = fmaf(H, x4, dx10); }
  { const float p = x4 * x11; float H = 0.0f;
  { float h;
    h = 0x1.a7f9ec0000000p-3f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.11acee0000000p-3f * q12_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx4 = fmaf(H, x11, dx4); dx11 = fmaf(H, x4, dx11); }
  { const float p = x5 * x5; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q3_0;
  { float h;
    h = 0x1.03a1a20000000p-3f * q11_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.c1b1dc0000000p-3f * q11_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
    dx5 = fmaf(H, x5, dx5); dx5 = fmaf(H, x5, dx5); }
  { const float p = x5 * x9; float H = 0.0f;
  { float h;
    h = 0x1.b0b8100000000p-3f * q12_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx5 = fmaf(H, x9, dx5); dx9 = fmaf(H, x5, dx9); }
  { const float p = x5 * x10; float H = 0.0f;
  { float h;
    h = 0x1.4f2ec40000000p-3f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = -0x1.b0b8100000000p-3f * q12_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx5 = fmaf(H, x10, dx5); dx10 = fmaf(H, x5, dx10); }
  { const float p = x5 * x11; float H = 0.0f;
  { float h;
    h = 0x1.5a2cd80000000p-4f * q12_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = -0x1.4f2ec40000000p-3f * q12_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx5 = fmaf(H, x11, dx5); dx11 = fmaf(H, x5, dx11); }
  { const float p = x6 * x6; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q3_0;
  { float h;
    h = 0x1.5a2cda0000000p-4f * q11_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
    dx6 = fmaf(H, x6, dx6); dx6 = fmaf(H, x6, dx6); }
  { const float p = x6 * x7; float H = 0.0f;
  { float h;
    h = 0x1.03a1a20000000p-3f * q11_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
    dx6 = fmaf(H, x7, dx6); dx7 = fmaf(H, x6, dx7); }
  { const float p = x6 * x8; float H = 0.0f;
  { float h;
    h = -0x1.03a1a20000000p-2f * q11_0;
    dx8 = fmaf(h, p, dx8);
    H = h * x8; }
    dx6 = fmaf(H, x8, dx6); dx8 = fmaf(H, x6, dx8); }
  { const float p = x6 * x9; float H = 0.0f;
  { float h;
    h = -0x1.31fa800000000p-3f * q12_0;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
    dx6 = fmaf(H, x9, dx6); dx9 = fmaf(H, x6, dx9); }
  { const float p = x6 * x11; float H = 0.0f;
  { float h;
    h = 0x1.6f2c9a0000000p-4f * q12_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx6 = fmaf(H, x11, dx6); dx11 = fmaf(H, x6, dx11); }
  { const float p = x6 * x12; float H = 0.0f;
  { float h;
    h = 0x1.e990ce0000000p-4f * q12_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
    dx6 = fmaf(H, x12, dx6); dx12 = fmaf(H, x6, dx12); }
  { const float p = x6 * x13; float H = 0.0f;
  { float h;
    h = 0x1.6f2c9a0000000p-4f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx6 = fmaf(H, x13, dx6); dx13 = fmaf(H, x6, dx13); }
  { const float p = x6 * x15; float H = 0.0f;
  { float h;
    h = -0x1.31fa800000000p-3f * q12_0;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx6 = fmaf(H, x15, dx6); dx15 = fmaf(H, x6, dx15); }
  { const float p = x7 * x7; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q3_0;
  { float h;
    h = 0x1.c1b1dc0000000p-3f * q11_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
    dx7 = fmaf(H, x7, dx7); dx7 = fmaf(H, x7, dx7); }
  { const float p = x7 * x9; float H = 0.0f;
  { float h;
    h = 0x1.b0b8100000000p-3f * q12_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx7 = fmaf(H, x9, dx7); dx9 = fmaf(H, x7, dx9); }
  { const float p = x7 * x10; float H = 0.0f;
  { float h;
    h = 0x1.4f2ec40000000p-3f * q12_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx7 = fmaf(H, x10, dx7); dx10 = fmaf(H, x7, dx10); }
  { const float p = x7 * x12; float H = 0.0f;
  { float h;
    h = 0x1.5a2cd80000000p-4f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx7 = fmaf(H, x12, dx7); dx12 = fmaf(H, x7, dx12); }
  { const float p = x7 * x13; float H = 0.0f;
  { float h;
    h = 0x1.4f2ec40000000p-3f * q12_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx7 = fmaf(H, x13, dx7); dx13 = fmaf(H, x7, dx13); }
  { const float p = x7 * x14; float H = 0.0f;
  { float h;
    h = 0x1.b0b8100000000p-3f * q12_0;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx7 = fmaf(H, x14, dx7); dx14 = fmaf(H, x7, dx14); }
  { const float p = x8 * x8; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q3_0;
    dx8 = fmaf(H, x8, dx8); dx8 = fmaf(H, x8, dx8); }
  { const float p = x8 * x9; float H = 0.0f;
  { float h;
    h = -0x1.11acee0000000p-3f * q12_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx8 = fmaf(H, x9, dx8); dx9 = fmaf(H, x8, dx9); }
  { const float p = x8 * x11; float H = 0.0f;
  { float h;
    h = -0x1.a7f9ec0000000p-4f * q12_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx8 = fmaf(H, x11, dx8); dx11 = fmaf(H, x8, dx11); }
  { const float p = x8 * x12; float H = 0.0f;
  { float h;
    h = -0x1.11acee0000000p-2f * q12_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx8 = fmaf(H, x12, dx8); dx12 = fmaf(H, x8, dx12); }
  { const float p = x8 * x13; float H = 0.0f;
  { float h;
    h = 0x1.a7f9ec0000000p-4f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = -0x1.11acee0000000p-3f * q12_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx8 = fmaf(H, x13, dx8); dx13 = fmaf(H, x8, dx13); }
  { const float p = x9 * x9; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx9 = fmaf(H, x9, dx9); dx9 = fmaf(H, x9, dx9); }
  { const float p = x10 * x10; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx10 = fmaf(H, x10, dx10); dx10 = fmaf(H, x10, dx10); }
  { const float p = x11 * x11; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx11 = fmaf(H, x11, dx11); dx11 = fmaf(H, x11, dx11); }
  { const float p = x12 * x12; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx12 = fmaf(H, x12, dx12); dx12 = fmaf(H, x12, dx12); }
  { const float p = x13 * x13; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx13 = fmaf(H, x13, dx13); dx13 = fmaf(H, x13, dx13); }
  { const float p = x14 * x14; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx14 = fmaf(H, x14, dx14); dx14 = fmaf(H, x14, dx14); }
  { const float p = x15 * x15; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q4_0;
    dx15 = fmaf(H, x15, dx15); dx15 = fmaf(H, x15, dx15); }
  float dot = 0.0f;
  dot = fmaf(dx0, x0, dot);
  dm[(long long)a * 2048 + 0 + ch] = dx0 * s;
  dot = fmaf(dx1, x1, dot);
  dm[(long long)a * 2048 + 128 + ch] = dx1 * s;
  dot = fmaf(dx2, x2, dot);
  dm[(long long)a * 2048 + 256 + ch] = dx2 * s;
  dot = fmaf(dx3, x3, dot);
  dm[(long long)a * 2048 + 384 + ch] = dx3 * s;
  dot = fmaf(dx4, x4, dot);
  dm[(long long)a * 2048 + 512 + ch] = dx4 * s;
  dot = fmaf(dx5, x5, dot);
  dm[(long long)a * 2048 + 640 + ch] = dx5 * s;
  dot = fmaf(dx6, x6, dot);
  dm[(long long)a * 2048 + 768 + ch] = dx6 * s;
  dot = fmaf(dx7, x7, dot);
  dm[(long long)a * 2048 + 896 + ch] = dx7 * s;
  dot = fmaf(dx8, x8, dot);
  dm[(long long)a * 2048 + 1024 + ch] = dx8 * s;
  dot = fmaf(dx9, x9, dot);
  dm[(long long)a * 2048 + 1152 + ch] = dx9 * s;
  dot = fmaf(dx10, x10, dot);
  dm[(long long)a * 2048 + 1280 + ch] = dx10 * s;
  dot = fmaf(dx11, x11, dot);
  dm[(long long)a * 2048 + 1408 + ch] = dx11 * s;
  dot = fmaf(dx12, x12, dot);
  dm[(long long)a * 2048 + 1536 + ch] = dx12 * s;
  dot = fmaf(dx13, x13, dot);
  dm[(long long)a * 2048 + 1664 + ch] = dx13 * s;
  dot = fmaf(dx14, x14, dot);
  dm[(long long)a * 2048 + 1792 + ch] = dx14 * s;
  dot = fmaf(dx15, x15, dot);
  dm[(long long)a * 2048 + 1920 + ch] = dx15 * s;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
  __shared__ float red[4];
  if ((ch & 31) == 0) red[ch >> 5] = dot;
  __syncthreads();
  if (ch == 0) dd[a] = -s * (red[0] + red[1] + red[2] + red[3]);
}
