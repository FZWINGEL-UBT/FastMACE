// generated backward: 480 terms, 410 monomials, 61 groups
extern "C" __global__ void __launch_bounds__(128) sc0_bwd(
    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,
    const int* __restrict__ sp, const float* __restrict__ g, float* __restrict__ dm,
    float* __restrict__ dd) {
  const int a = blockIdx.x, ch = threadIdx.x;
  const float* xr = m + (long long)a * 2048 + ch;
  const float* wr = w + (long long)sp[a] * 3712 + ch;
  const float* gr = g + (long long)a * 512 + ch;
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
  const float g1 = __ldg(gr + 128);
  const float g2 = __ldg(gr + 256);
  const float g3 = __ldg(gr + 384);
  const float q0_0 = __ldg(wr + 0) * g0;
  const float q1_1 = __ldg(wr + 128) * g1;
  const float q1_2 = __ldg(wr + 128) * g2;
  const float q1_3 = __ldg(wr + 128) * g3;
  const float q2_0 = __ldg(wr + 256) * g0;
  const float q3_0 = __ldg(wr + 384) * g0;
  const float q4_0 = __ldg(wr + 512) * g0;
  const float q5_0 = __ldg(wr + 640) * g0;
  const float q6_1 = __ldg(wr + 768) * g1;
  const float q6_2 = __ldg(wr + 768) * g2;
  const float q6_3 = __ldg(wr + 768) * g3;
  const float q7_1 = __ldg(wr + 896) * g1;
  const float q7_2 = __ldg(wr + 896) * g2;
  const float q7_3 = __ldg(wr + 896) * g3;
  const float q8_1 = __ldg(wr + 1024) * g1;
  const float q8_2 = __ldg(wr + 1024) * g2;
  const float q8_3 = __ldg(wr + 1024) * g3;
  const float q9_0 = __ldg(wr + 1152) * g0;
  const float q10_0 = __ldg(wr + 1280) * g0;
  const float q11_0 = __ldg(wr + 1408) * g0;
  const float q12_0 = __ldg(wr + 1536) * g0;
  const float q13_0 = __ldg(wr + 1664) * g0;
  const float q14_0 = __ldg(wr + 1792) * g0;
  const float q15_0 = __ldg(wr + 1920) * g0;
  const float q16_0 = __ldg(wr + 2048) * g0;
  const float q17_1 = __ldg(wr + 2176) * g1;
  const float q17_2 = __ldg(wr + 2176) * g2;
  const float q17_3 = __ldg(wr + 2176) * g3;
  const float q18_1 = __ldg(wr + 2304) * g1;
  const float q18_2 = __ldg(wr + 2304) * g2;
  const float q18_3 = __ldg(wr + 2304) * g3;
  const float q19_1 = __ldg(wr + 2432) * g1;
  const float q19_2 = __ldg(wr + 2432) * g2;
  const float q19_3 = __ldg(wr + 2432) * g3;
  const float q20_1 = __ldg(wr + 2560) * g1;
  const float q20_2 = __ldg(wr + 2560) * g2;
  const float q20_3 = __ldg(wr + 2560) * g3;
  const float q21_1 = __ldg(wr + 2688) * g1;
  const float q21_2 = __ldg(wr + 2688) * g2;
  const float q21_3 = __ldg(wr + 2688) * g3;
  const float q22_1 = __ldg(wr + 2816) * g1;
  const float q22_2 = __ldg(wr + 2816) * g2;
  const float q22_3 = __ldg(wr + 2816) * g3;
  const float q23_1 = __ldg(wr + 2944) * g1;
  const float q23_2 = __ldg(wr + 2944) * g2;
  const float q23_3 = __ldg(wr + 2944) * g3;
  const float q24_1 = __ldg(wr + 3072) * g1;
  const float q24_2 = __ldg(wr + 3072) * g2;
  const float q24_3 = __ldg(wr + 3072) * g3;
  const float q25_1 = __ldg(wr + 3200) * g1;
  const float q25_2 = __ldg(wr + 3200) * g2;
  const float q25_3 = __ldg(wr + 3200) * g3;
  const float q26_1 = __ldg(wr + 3328) * g1;
  const float q26_2 = __ldg(wr + 3328) * g2;
  const float q26_3 = __ldg(wr + 3328) * g3;
  const float q27_1 = __ldg(wr + 3456) * g1;
  const float q27_2 = __ldg(wr + 3456) * g2;
  const float q27_3 = __ldg(wr + 3456) * g3;
  const float q28_1 = __ldg(wr + 3584) * g1;
  const float q28_2 = __ldg(wr + 3584) * g2;
  const float q28_3 = __ldg(wr + 3584) * g3;
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
    dx1 = fmaf(0x1.0000000000000p+0f, q1_1, dx1);
    dx2 = fmaf(0x1.0000000000000p+0f, q1_2, dx2);
    dx3 = fmaf(0x1.0000000000000p+0f, q1_3, dx3);
  { const float p = x0 * x0; float H = 0.0f;
    H = 0x1.0000000000000p-1f * q2_0;
  { float h;
    h = 0x1.6a09e60000000p-2f * q9_0;
    dx0 = fmaf(h, p, dx0);
    H = fmaf(h, x0, H); }
  { float h;
    h = 0x1.0000000000000p-1f * q17_1;
    dx1 = fmaf(h, p, dx1);
    H = fmaf(h, x1, H); }
  { float h;
    h = 0x1.0000000000000p-1f * q17_2;
    dx2 = fmaf(h, p, dx2);
    H = fmaf(h, x2, H); }
  { float h;
    h = 0x1.0000000000000p-1f * q17_3;
    dx3 = fmaf(h, p, dx3);
    H = fmaf(h, x3, H); }
    dx0 = fmaf(H, x0, dx0); dx0 = fmaf(H, x0, dx0); }
  { const float p = x0 * x1; float H = 0.0f;
    H = 0x1.a20bd60000000p-1f * q6_1;
  { float h;
    h = 0x1.6a09e60000000p-2f * q10_0;
    dx1 = fmaf(h, p, dx1);
    H = fmaf(h, x1, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_3;
    dx4 = fmaf(h, p, dx4);
    H = fmaf(h, x4, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_2;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = -0x1.c9f25c0000000p-3f * q18_1;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.8c97ee0000000p-2f * q18_1;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
    dx0 = fmaf(H, x1, dx0); dx1 = fmaf(H, x0, dx1); }
  { const float p = x0 * x2; float H = 0.0f;
    H = 0x1.a20bd60000000p-1f * q6_2;
  { float h;
    h = 0x1.6a09e60000000p-2f * q10_0;
    dx2 = fmaf(h, p, dx2);
    H = fmaf(h, x2, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_1;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = 0x1.c9f25c0000000p-2f * q18_2;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_3;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
    dx0 = fmaf(H, x2, dx0); dx2 = fmaf(H, x0, dx2); }
  { const float p = x0 * x3; float H = 0.0f;
    H = 0x1.a20bd60000000p-1f * q6_3;
  { float h;
    h = 0x1.6a09e60000000p-2f * q10_0;
    dx3 = fmaf(h, p, dx3);
    H = fmaf(h, x3, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_1;
    dx4 = fmaf(h, p, dx4);
    H = fmaf(h, x4, H); }
  { float h;
    h = -0x1.c9f25c0000000p-3f * q18_3;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_2;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.8c97ee0000000p-2f * q18_3;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
    dx0 = fmaf(H, x3, dx0); dx3 = fmaf(H, x0, dx3); }
  { const float p = x0 * x4; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q11_0;
    dx4 = fmaf(h, p, dx4);
    H = h * x4; }
  { float h;
    h = 0x1.4f2ec40000000p-2f * q19_3;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q19_2;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.5a2cd80000000p-4f * q19_3;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.5a2cd80000000p-4f * q19_1;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.4f2ec40000000p-2f * q19_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx0 = fmaf(H, x4, dx0); dx4 = fmaf(H, x0, dx4); }
  { const float p = x0 * x5; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q11_0;
    dx5 = fmaf(h, p, dx5);
    H = h * x5; }
  { float h;
    h = 0x1.11acee0000000p-2f * q19_3;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = 0x1.5a2cd80000000p-2f * q19_2;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.a7f9ec0000000p-3f * q19_1;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.11acee0000000p-2f * q19_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx0 = fmaf(H, x5, dx0); dx5 = fmaf(H, x0, dx5); }
  { const float p = x0 * x6; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q11_0;
    dx6 = fmaf(h, p, dx6);
    H = h * x6; }
  { float h;
    h = 0x1.2bcbe80000000p-2f * q19_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.6f2c9a0000000p-2f * q19_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.2bcbe80000000p-2f * q19_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx0 = fmaf(H, x6, dx0); dx6 = fmaf(H, x0, dx6); }
  { const float p = x0 * x7; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q11_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
  { float h;
    h = 0x1.11acee0000000p-2f * q19_1;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.a7f9ec0000000p-3f * q19_3;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.5a2cd80000000p-2f * q19_2;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q19_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx0 = fmaf(H, x7, dx0); dx7 = fmaf(H, x0, dx7); }
  { const float p = x0 * x8; float H = 0.0f;
  { float h;
    h = 0x1.186f180000000p-2f * q11_0;
    dx8 = fmaf(h, p, dx8);
    H = h * x8; }
  { float h;
    h = 0x1.4f2ec40000000p-2f * q19_1;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = 0x1.5a2cd80000000p-4f * q19_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.5a2cd80000000p-4f * q19_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q19_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = 0x1.4f2ec40000000p-2f * q19_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx0 = fmaf(H, x8, dx0); dx8 = fmaf(H, x0, dx8); }
  { const float p = x0 * x9; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
    dx0 = fmaf(H, x9, dx0); dx9 = fmaf(H, x0, dx9); }
  { const float p = x0 * x10; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx0 = fmaf(H, x10, dx0); dx10 = fmaf(H, x0, dx10); }
  { const float p = x0 * x11; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
    dx0 = fmaf(H, x11, dx0); dx11 = fmaf(H, x0, dx11); }
  { const float p = x0 * x12; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
    dx0 = fmaf(H, x12, dx0); dx12 = fmaf(H, x0, dx12); }
  { const float p = x0 * x13; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx0 = fmaf(H, x13, dx0); dx13 = fmaf(H, x0, dx13); }
  { const float p = x0 * x14; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx0 = fmaf(H, x14, dx0); dx14 = fmaf(H, x0, dx14); }
  { const float p = x0 * x15; float H = 0.0f;
  { float h;
    h = 0x1.da05180000000p-3f * q12_0;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx0 = fmaf(H, x15, dx0); dx15 = fmaf(H, x0, dx15); }
  { const float p = x1 * x1; float H = 0.0f;
    H = 0x1.279a740000000p-2f * q3_0;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_1;
    dx1 = fmaf(h, p, dx1);
    H = fmaf(h, x1, H); }
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_2;
    dx2 = fmaf(h, p, dx2);
    H = fmaf(h, x2, H); }
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_3;
    dx3 = fmaf(h, p, dx3);
    H = fmaf(h, x3, H); }
  { float h;
    h = -0x1.c9f25c0000000p-4f * q13_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.8c97f00000000p-3f * q13_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = -0x1.4f2ec40000000p-3f * q21_1;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = -0x1.03a1a20000000p-3f * q21_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.a7f9ee0000000p-4f * q21_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.5a2cda0000000p-5f * q21_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.11acee0000000p-3f * q21_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = -0x1.4f2ec40000000p-3f * q21_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x1, dx1); dx1 = fmaf(H, x1, dx1); }
  { const float p = x1 * x2; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_1;
    dx2 = fmaf(h, p, dx2);
    H = h * x2; }
  { float h;
    h = 0x1.8c97f00000000p-2f * q13_0;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q21_3;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = 0x1.5a2cd80000000p-2f * q21_2;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.a7f9ec0000000p-3f * q21_1;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.11acee0000000p-2f * q21_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx1 = fmaf(H, x2, dx1); dx2 = fmaf(H, x1, dx2); }
  { const float p = x1 * x3; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_1;
    dx3 = fmaf(h, p, dx3);
    H = h * x3; }
  { float h;
    h = 0x1.8c97f00000000p-2f * q13_0;
    dx4 = fmaf(h, p, dx4);
    H = fmaf(h, x4, H); }
  { float h;
    h = 0x1.4f2ec40000000p-2f * q21_3;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q21_2;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.5a2cd80000000p-4f * q21_3;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.5a2cd80000000p-4f * q21_1;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.4f2ec40000000p-2f * q21_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x3, dx1); dx3 = fmaf(H, x1, dx3); }
  { const float p = x1 * x4; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_3;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_1;
    dx4 = fmaf(h, p, dx4);
    H = fmaf(h, x4, H); }
  { float h;
    h = 0x1.186f180000000p-2f * q23_3;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_2;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = -0x1.e990ce0000000p-5f * q14_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.da05160000000p-3f * q14_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x4, dx1); dx4 = fmaf(H, x1, dx4); }
  { const float p = x1 * x5; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_2;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_1;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = -0x1.186f180000000p-3f * q23_2;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_3;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.e5b9d20000000p-3f * q23_2;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = -0x1.2bcbe60000000p-3f * q14_0;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.83091e0000000p-3f * q14_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx1 = fmaf(H, x5, dx1); dx5 = fmaf(H, x1, dx5); }
  { const float p = x1 * x6; float H = 0.0f;
    H = -0x1.08654a0000000p-2f * q7_1;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_1;
    h = fmaf(0x1.43d1360000000p-3f, q23_1, h);
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.186f180000000p-2f * q23_1;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.a7f9ee0000000p-3f * q14_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx1 = fmaf(H, x6, dx1); dx6 = fmaf(H, x1, dx6); }
  { const float p = x1 * x7; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_1;
    h = fmaf(0x1.e5b9d20000000p-3f, q23_1, h);
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
  { float h;
    h = 0x1.83091e0000000p-3f * q14_0;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
    dx1 = fmaf(H, x7, dx1); dx7 = fmaf(H, x1, dx7); }
  { const float p = x1 * x8; float H = 0.0f;
    H = -0x1.c9f25c0000000p-2f * q7_1;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_1;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.da05160000000p-3f * q14_0;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = 0x1.e990ce0000000p-5f * q14_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx1 = fmaf(H, x8, dx1); dx8 = fmaf(H, x1, dx8); }
  { const float p = x1 * x9; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    h = fmaf(0x1.6150460000000p-4f, q25_1, h);
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.11acee0000000p-3f * q25_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.11acee0000000p-3f * q25_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.b0b8100000000p-3f * q25_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx1 = fmaf(H, x9, dx1); dx9 = fmaf(H, x1, dx9); }
  { const float p = x1 * x10; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
  { float h;
    h = -0x1.11acee0000000p-2f * q25_3;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q25_2;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.b0b8100000000p-3f * q25_2;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x10, dx1); dx10 = fmaf(H, x1, dx10); }
  { const float p = x1 * x11; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    h = fmaf(0x1.a7f9ec0000000p-5f, q25_1, h);
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
  { float h;
    h = 0x1.5a2cd80000000p-4f * q25_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.a7f9ec0000000p-3f * q25_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.4f2ec40000000p-3f * q25_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = 0x1.11acee0000000p-3f * q25_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x11, dx1); dx11 = fmaf(H, x1, dx11); }
  { const float p = x1 * x12; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    h = fmaf(-0x1.1aa69e0000000p-4f, q25_1, h);
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = 0x1.11acee0000000p-2f * q25_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx1 = fmaf(H, x12, dx1); dx12 = fmaf(H, x1, dx12); }
  { const float p = x1 * x13; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    h = fmaf(-0x1.3dfb720000000p-3f, q25_1, h);
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.11acee0000000p-3f * q25_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx1 = fmaf(H, x13, dx1); dx13 = fmaf(H, x1, dx13); }
  { const float p = x1 * x14; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx1 = fmaf(H, x14, dx1); dx14 = fmaf(H, x1, dx14); }
  { const float p = x1 * x15; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_1;
    h = fmaf(0x1.6150460000000p-4f, q25_1, h);
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx1 = fmaf(H, x15, dx1); dx15 = fmaf(H, x1, dx15); }
  { const float p = x2 * x2; float H = 0.0f;
    H = 0x1.279a740000000p-2f * q3_0;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_2;
    dx2 = fmaf(h, p, dx2);
    H = fmaf(h, x2, H); }
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_3;
    dx3 = fmaf(h, p, dx3);
    H = fmaf(h, x3, H); }
  { float h;
    h = 0x1.c9f25c0000000p-3f * q13_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.5a2cd80000000p-3f * q21_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.a7f9ee0000000p-3f * q21_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.5a2cd80000000p-3f * q21_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx2 = fmaf(H, x2, dx2); dx2 = fmaf(H, x2, dx2); }
  { const float p = x2 * x3; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_2;
    dx3 = fmaf(h, p, dx3);
    H = h * x3; }
  { float h;
    h = 0x1.8c97f00000000p-2f * q13_0;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q21_1;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.a7f9ec0000000p-3f * q21_3;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.5a2cd80000000p-2f * q21_2;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.11acee0000000p-2f * q21_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx2 = fmaf(H, x3, dx2); dx3 = fmaf(H, x2, dx3); }
  { const float p = x2 * x4; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_2;
    h = fmaf(0x1.e5b9d20000000p-3f, q23_2, h);
    dx4 = fmaf(h, p, dx4);
    H = h * x4; }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_3;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_1;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.83091e0000000p-3f * q14_0;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
    dx2 = fmaf(H, x4, dx2); dx4 = fmaf(H, x2, dx4); }
  { const float p = x2 * x5; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_1;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_2;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = -0x1.186f180000000p-3f * q23_1;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.e5b9d20000000p-3f * q23_1;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.e990ce0000000p-3f * q14_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx2 = fmaf(H, x5, dx2); dx5 = fmaf(H, x2, dx5); }
  { const float p = x2 * x6; float H = 0.0f;
    H = 0x1.08654a0000000p-1f * q7_2;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_2;
    h = fmaf(-0x1.43d1360000000p-4f, q23_2, h);
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.186f180000000p-3f * q23_3;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.03a1a20000000p-2f * q14_0;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
    dx2 = fmaf(H, x6, dx2); dx6 = fmaf(H, x2, dx6); }
  { const float p = x2 * x7; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_3;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_2;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_3;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.e990ce0000000p-3f * q14_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx2 = fmaf(H, x7, dx2); dx7 = fmaf(H, x2, dx7); }
  { const float p = x2 * x8; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_2;
    h = fmaf(0x1.e5b9d20000000p-3f, q23_2, h);
    dx8 = fmaf(h, p, dx8);
    H = h * x8; }
  { float h;
    h = 0x1.83091e0000000p-3f * q14_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx2 = fmaf(H, x8, dx2); dx8 = fmaf(H, x2, dx8); }
  { const float p = x2 * x9; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    h = fmaf(-0x1.6150460000000p-3f, q25_2, h);
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.b0b8100000000p-3f * q25_3;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = 0x1.b0b8100000000p-3f * q25_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx2 = fmaf(H, x9, dx2); dx9 = fmaf(H, x2, dx9); }
  { const float p = x2 * x10; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q25_3;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q25_1;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.b0b8100000000p-3f * q25_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx2 = fmaf(H, x10, dx2); dx10 = fmaf(H, x2, dx10); }
  { const float p = x2 * x11; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    h = fmaf(0x1.a7f9ee0000000p-4f, q25_2, h);
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
  { float h;
    h = 0x1.5a2cd80000000p-4f * q25_1;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.4f2ec40000000p-3f * q25_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx2 = fmaf(H, x11, dx2); dx11 = fmaf(H, x2, dx11); }
  { const float p = x2 * x12; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    h = fmaf(0x1.1aa69e0000000p-3f, q25_2, h);
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = 0x1.5a2cd80000000p-4f * q25_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx2 = fmaf(H, x12, dx2); dx12 = fmaf(H, x2, dx12); }
  { const float p = x2 * x13; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    h = fmaf(0x1.a7f9ee0000000p-4f, q25_2, h);
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q25_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx2 = fmaf(H, x13, dx2); dx13 = fmaf(H, x2, dx13); }
  { const float p = x2 * x14; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
  { float h;
    h = 0x1.b0b8100000000p-3f * q25_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx2 = fmaf(H, x14, dx2); dx14 = fmaf(H, x2, dx14); }
  { const float p = x2 * x15; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_2;
    h = fmaf(-0x1.6150460000000p-3f, q25_2, h);
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx2 = fmaf(H, x15, dx2); dx15 = fmaf(H, x2, dx15); }
  { const float p = x3 * x3; float H = 0.0f;
    H = 0x1.279a740000000p-2f * q3_0;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q20_3;
    dx3 = fmaf(h, p, dx3);
    H = fmaf(h, x3, H); }
  { float h;
    h = -0x1.c9f25c0000000p-4f * q13_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.8c97f00000000p-3f * q13_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q21_1;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = -0x1.5a2cda0000000p-5f * q21_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.a7f9ee0000000p-4f * q21_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.03a1a20000000p-3f * q21_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.11acee0000000p-3f * q21_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q21_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx3 = fmaf(H, x3, dx3); dx3 = fmaf(H, x3, dx3); }
  { const float p = x3 * x4; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_1;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_3;
    dx4 = fmaf(h, p, dx4);
    H = fmaf(h, x4, H); }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_2;
    dx5 = fmaf(h, p, dx5);
    H = fmaf(h, x5, H); }
  { float h;
    h = 0x1.186f180000000p-2f * q23_1;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.da05160000000p-3f * q14_0;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = -0x1.e990ce0000000p-5f * q14_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx3 = fmaf(H, x4, dx3); dx4 = fmaf(H, x3, dx4); }
  { const float p = x3 * x5; float H = 0.0f;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_3;
    h = fmaf(0x1.e5b9d20000000p-3f, q23_3, h);
    dx5 = fmaf(h, p, dx5);
    H = h * x5; }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_1;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.83091e0000000p-3f * q14_0;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
    dx3 = fmaf(H, x5, dx3); dx5 = fmaf(H, x3, dx5); }
  { const float p = x3 * x6; float H = 0.0f;
    H = -0x1.08654a0000000p-2f * q7_3;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_3;
    h = fmaf(0x1.43d1360000000p-3f, q23_3, h);
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.186f180000000p-3f * q23_2;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = 0x1.186f180000000p-2f * q23_3;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.a7f9ee0000000p-3f * q14_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx3 = fmaf(H, x6, dx3); dx6 = fmaf(H, x3, dx6); }
  { const float p = x3 * x7; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_2;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_3;
    dx7 = fmaf(h, p, dx7);
    H = fmaf(h, x7, H); }
  { float h;
    h = -0x1.e5b9d20000000p-3f * q23_2;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = -0x1.2bcbe60000000p-3f * q14_0;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.83091e0000000p-3f * q14_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx3 = fmaf(H, x7, dx3); dx7 = fmaf(H, x3, dx7); }
  { const float p = x3 * x8; float H = 0.0f;
    H = 0x1.c9f25c0000000p-2f * q7_3;
  { float h;
    h = 0x1.c9f25c0000000p-3f * q22_3;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = -0x1.e990ce0000000p-5f * q14_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.da05160000000p-3f * q14_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx3 = fmaf(H, x8, dx3); dx8 = fmaf(H, x3, dx8); }
  { const float p = x3 * x9; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    h = fmaf(0x1.6150460000000p-4f, q25_3, h);
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.b0b8100000000p-3f * q25_2;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.11acee0000000p-3f * q25_3;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.11acee0000000p-3f * q25_1;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx3 = fmaf(H, x9, dx3); dx9 = fmaf(H, x3, dx9); }
  { const float p = x3 * x10; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q25_2;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.11acee0000000p-2f * q25_1;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
    dx3 = fmaf(H, x10, dx3); dx10 = fmaf(H, x3, dx10); }
  { const float p = x3 * x11; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    h = fmaf(-0x1.3dfb720000000p-3f, q25_3, h);
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
  { float h;
    h = 0x1.a7f9ec0000000p-3f * q25_1;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.11acee0000000p-3f * q25_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx3 = fmaf(H, x11, dx3); dx11 = fmaf(H, x3, dx11); }
  { const float p = x3 * x12; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    h = fmaf(-0x1.1aa69e0000000p-4f, q25_3, h);
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = 0x1.5a2cd80000000p-4f * q25_2;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.11acee0000000p-2f * q25_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx3 = fmaf(H, x12, dx3); dx12 = fmaf(H, x3, dx12); }
  { const float p = x3 * x13; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    h = fmaf(0x1.a7f9ec0000000p-5f, q25_3, h);
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.4f2ec40000000p-3f * q25_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = -0x1.11acee0000000p-3f * q25_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx3 = fmaf(H, x13, dx3); dx13 = fmaf(H, x3, dx13); }
  { const float p = x3 * x14; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
  { float h;
    h = 0x1.b0b8100000000p-3f * q25_2;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx3 = fmaf(H, x14, dx3); dx14 = fmaf(H, x3, dx14); }
  { const float p = x3 * x15; float H = 0.0f;
  { float h;
    h = 0x1.83091e0000000p-3f * q24_3;
    h = fmaf(0x1.6150460000000p-4f, q25_3, h);
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx3 = fmaf(H, x15, dx3); dx15 = fmaf(H, x3, dx15); }
  { const float p = x4 * x4; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q4_0;
  { float h;
    h = -0x1.03a1a20000000p-2f * q15_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.bee9060000000p-4f * q27_1;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = -0x1.c53fee0000000p-4f * q26_1;
    h = fmaf(-0x1.5a2cda0000000p-4f, q27_1, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.158ec20000000p-3f * q26_2;
    h = fmaf(-0x1.1aa6a00000000p-4f, q27_2, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.c53fee0000000p-4f * q26_3;
    h = fmaf(-0x1.5a2cda0000000p-4f, q27_3, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.bee9060000000p-4f * q27_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx4 = fmaf(H, x4, dx4); dx4 = fmaf(H, x4, dx4); }
  { const float p = x4 * x5; float H = 0.0f;
  { float h;
    h = 0x1.c1b1dc0000000p-2f * q15_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
  { float h;
    h = 0x1.66534c0000000p-3f * q26_1;
    h = fmaf(0x1.11acf00000000p-2f, q27_1, h);
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.158ec00000000p-3f * q26_3;
    h = fmaf(-0x1.1aa69e0000000p-3f, q27_3, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.c53fee0000000p-3f * q26_2;
    h = fmaf(0x1.5a2cda0000000p-4f, q27_2, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.66534c0000000p-3f * q26_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = -0x1.bee9080000000p-4f * q27_2;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx4 = fmaf(H, x5, dx4); dx5 = fmaf(H, x4, dx5); }
  { const float p = x4 * x6; float H = 0.0f;
  { float h;
    h = -0x1.fabfa20000000p-3f * q26_3;
    h = fmaf(-0x1.8309200000000p-3f, q27_3, h);
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = -0x1.9dc22c0000000p-3f * q26_2;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = 0x1.05af100000000p-4f * q26_3;
    h = fmaf(0x1.2bcbe80000000p-3f, q27_3, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.05af100000000p-4f * q26_1;
    h = fmaf(0x1.2bcbe80000000p-3f, q27_1, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.fabfa20000000p-3f * q26_1;
    h = fmaf(0x1.8309200000000p-3f, q27_1, h);
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx4 = fmaf(H, x6, dx4); dx6 = fmaf(H, x4, dx6); }
  { const float p = x4 * x7; float H = 0.0f;
  { float h;
    h = 0x1.bee9080000000p-4f * q27_2;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.66534c0000000p-3f * q26_3;
    h = fmaf(0x1.11acf00000000p-2f, q27_3, h);
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = 0x1.c53fee0000000p-3f * q26_2;
    h = fmaf(0x1.5a2cda0000000p-4f, q27_2, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.158ec00000000p-3f * q26_1;
    h = fmaf(-0x1.1aa69e0000000p-3f, q27_1, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.66534c0000000p-3f * q26_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx4 = fmaf(H, x7, dx4); dx7 = fmaf(H, x4, dx7); }
  { const float p = x4 * x8; float H = 0.0f;
  { float h;
    h = 0x1.bee9060000000p-3f * q27_3;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.bee9060000000p-3f * q27_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx4 = fmaf(H, x8, dx4); dx8 = fmaf(H, x4, dx8); }
  { const float p = x4 * x9; float H = 0.0f;
    H = 0x1.83091e0000000p-2f * q8_3;
  { float h;
    h = -0x1.11acee0000000p-3f * q16_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx4 = fmaf(H, x9, dx4); dx9 = fmaf(H, x4, dx9); }
  { const float p = x4 * x10; float H = 0.0f;
    H = 0x1.3c03660000000p-2f * q8_2;
  { float h;
    h = -0x1.11acee0000000p-2f * q16_0;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
    dx4 = fmaf(H, x10, dx4); dx10 = fmaf(H, x4, dx10); }
  { const float p = x4 * x11; float H = 0.0f;
    H = -0x1.8fba8a0000000p-4f * q8_3;
  { float h;
    h = 0x1.a7f9ec0000000p-3f * q16_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.11acee0000000p-3f * q16_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx4 = fmaf(H, x11, dx4); dx11 = fmaf(H, x4, dx11); }
  { const float p = x4 * x13; float H = 0.0f;
    H = -0x1.8fba8a0000000p-4f * q8_1;
    dx4 = fmaf(H, x13, dx4); dx13 = fmaf(H, x4, dx13); }
  { const float p = x4 * x15; float H = 0.0f;
    H = -0x1.83091e0000000p-2f * q8_1;
    dx4 = fmaf(H, x15, dx4); dx15 = fmaf(H, x4, dx15); }
  { const float p = x5 * x5; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q4_0;
  { float h;
    h = 0x1.03a1a20000000p-3f * q15_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = -0x1.c1b1dc0000000p-3f * q15_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = -0x1.b6db6e0000000p-4f * q26_1;
    h = fmaf(-0x1.bee9060000000p-5f, q27_1, h);
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = 0x1.c53fee0000000p-6f * q26_1;
    h = fmaf(0x1.03a1a40000000p-3f, q27_1, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.158ec20000000p-4f * q26_2;
    h = fmaf(-0x1.1aa69e0000000p-5f, q27_2, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.53eff20000000p-4f * q26_3;
    h = fmaf(0x1.5a2cdc0000000p-5f, q27_3, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.66534c0000000p-4f * q26_2;
    h = fmaf(-0x1.11acf00000000p-3f, q27_2, h);
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = -0x1.b6db6e0000000p-4f * q26_3;
    h = fmaf(-0x1.bee9060000000p-5f, q27_3, h);
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx5 = fmaf(H, x5, dx5); dx5 = fmaf(H, x5, dx5); }
  { const float p = x5 * x6; float H = 0.0f;
  { float h;
    h = 0x1.9dc22c0000000p-4f * q26_3;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
  { float h;
    h = 0x1.05af100000000p-3f * q26_2;
    h = fmaf(0x1.2bcbe80000000p-2f, q27_2, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.407ef60000000p-4f * q26_1;
    h = fmaf(0x1.e990d00000000p-4f, q27_1, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.9dc22c0000000p-4f * q26_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx5 = fmaf(H, x6, dx5); dx6 = fmaf(H, x5, dx6); }
  { const float p = x5 * x7; float H = 0.0f;
  { float h;
    h = 0x1.b6db6e0000000p-3f * q26_3;
    h = fmaf(0x1.bee9060000000p-4f, q27_3, h);
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = 0x1.66534c0000000p-3f * q26_2;
    h = fmaf(0x1.11acf00000000p-2f, q27_2, h);
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.c53fee0000000p-5f * q26_3;
    h = fmaf(0x1.5a2cda0000000p-4f, q27_3, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.c53fee0000000p-5f * q26_1;
    h = fmaf(0x1.5a2cda0000000p-4f, q27_1, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.b6db6e0000000p-3f * q26_1;
    h = fmaf(-0x1.bee9060000000p-4f, q27_1, h);
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx5 = fmaf(H, x7, dx5); dx7 = fmaf(H, x5, dx7); }
  { const float p = x5 * x8; float H = 0.0f;
  { float h;
    h = 0x1.bee9080000000p-4f * q27_2;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
  { float h;
    h = -0x1.66534c0000000p-3f * q26_3;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.c53fee0000000p-3f * q26_2;
    h = fmaf(-0x1.5a2cda0000000p-4f, q27_2, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.158ec00000000p-3f * q26_1;
    h = fmaf(0x1.1aa69e0000000p-3f, q27_1, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.66534c0000000p-3f * q26_1;
    h = fmaf(0x1.11acf00000000p-2f, q27_1, h);
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx5 = fmaf(H, x8, dx5); dx8 = fmaf(H, x5, dx8); }
  { const float p = x5 * x9; float H = 0.0f;
  { float h;
    h = 0x1.b0b8100000000p-3f * q16_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx5 = fmaf(H, x9, dx5); dx9 = fmaf(H, x5, dx9); }
  { const float p = x5 * x10; float H = 0.0f;
    H = 0x1.3c03660000000p-2f * q8_3;
  { float h;
    h = 0x1.4f2ec40000000p-3f * q16_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.b0b8100000000p-3f * q16_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx5 = fmaf(H, x10, dx5); dx10 = fmaf(H, x5, dx10); }
  { const float p = x5 * x11; float H = 0.0f;
    H = 0x1.8fba8a0000000p-2f * q8_2;
  { float h;
    h = 0x1.5a2cd80000000p-4f * q16_0;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.4f2ec40000000p-3f * q16_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx5 = fmaf(H, x11, dx5); dx11 = fmaf(H, x5, dx11); }
  { const float p = x5 * x12; float H = 0.0f;
    H = -0x1.e990cc0000000p-3f * q8_1;
    dx5 = fmaf(H, x12, dx5); dx12 = fmaf(H, x5, dx12); }
  { const float p = x5 * x14; float H = 0.0f;
    H = -0x1.3c03660000000p-2f * q8_1;
    dx5 = fmaf(H, x14, dx5); dx14 = fmaf(H, x5, dx14); }
  { const float p = x6 * x6; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q4_0;
  { float h;
    h = 0x1.5a2cda0000000p-4f * q15_0;
    dx6 = fmaf(h, p, dx6);
    H = fmaf(h, x6, H); }
  { float h;
    h = 0x1.c53fee0000000p-4f * q26_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.158ec20000000p-3f * q26_2;
    h = fmaf(0x1.a7f9ee0000000p-3f, q27_2, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.c53fee0000000p-4f * q26_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx6 = fmaf(H, x6, dx6); dx6 = fmaf(H, x6, dx6); }
  { const float p = x6 * x7; float H = 0.0f;
  { float h;
    h = 0x1.03a1a20000000p-3f * q15_0;
    dx7 = fmaf(h, p, dx7);
    H = h * x7; }
  { float h;
    h = 0x1.9dc22c0000000p-4f * q26_1;
    dx10 = fmaf(h, p, dx10);
    H = fmaf(h, x10, H); }
  { float h;
    h = -0x1.407ef60000000p-4f * q26_3;
    h = fmaf(0x1.e990d00000000p-4f, q27_3, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.05af100000000p-3f * q26_2;
    h = fmaf(0x1.2bcbe80000000p-2f, q27_2, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.9dc22c0000000p-4f * q26_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx6 = fmaf(H, x7, dx6); dx7 = fmaf(H, x6, dx7); }
  { const float p = x6 * x8; float H = 0.0f;
  { float h;
    h = -0x1.03a1a20000000p-2f * q15_0;
    dx8 = fmaf(h, p, dx8);
    H = h * x8; }
  { float h;
    h = -0x1.fabfa20000000p-3f * q26_1;
    h = fmaf(-0x1.8309200000000p-3f, q27_1, h);
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = -0x1.05af100000000p-4f * q26_1;
    h = fmaf(-0x1.2bcbe80000000p-3f, q27_1, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.05af100000000p-4f * q26_3;
    h = fmaf(0x1.2bcbe80000000p-3f, q27_3, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.9dc22c0000000p-3f * q26_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = -0x1.fabfa20000000p-3f * q26_3;
    h = fmaf(-0x1.8309200000000p-3f, q27_3, h);
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx6 = fmaf(H, x8, dx6); dx8 = fmaf(H, x6, dx8); }
  { const float p = x6 * x9; float H = 0.0f;
  { float h;
    h = -0x1.31fa800000000p-3f * q16_0;
    dx9 = fmaf(h, p, dx9);
    H = h * x9; }
    dx6 = fmaf(H, x9, dx6); dx9 = fmaf(H, x6, dx9); }
  { const float p = x6 * x11; float H = 0.0f;
    H = 0x1.5a2cda0000000p-2f * q8_1;
  { float h;
    h = 0x1.6f2c9a0000000p-4f * q16_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx6 = fmaf(H, x11, dx6); dx11 = fmaf(H, x6, dx11); }
  { const float p = x6 * x12; float H = 0.0f;
    H = 0x1.a7f9ee0000000p-2f * q8_2;
  { float h;
    h = 0x1.e990ce0000000p-4f * q16_0;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
    dx6 = fmaf(H, x12, dx6); dx12 = fmaf(H, x6, dx12); }
  { const float p = x6 * x13; float H = 0.0f;
    H = 0x1.5a2cda0000000p-2f * q8_3;
  { float h;
    h = 0x1.6f2c9a0000000p-4f * q16_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx6 = fmaf(H, x13, dx6); dx13 = fmaf(H, x6, dx13); }
  { const float p = x6 * x15; float H = 0.0f;
  { float h;
    h = -0x1.31fa800000000p-3f * q16_0;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx6 = fmaf(H, x15, dx6); dx15 = fmaf(H, x6, dx15); }
  { const float p = x7 * x7; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q4_0;
  { float h;
    h = 0x1.c1b1dc0000000p-3f * q15_0;
    dx8 = fmaf(h, p, dx8);
    H = fmaf(h, x8, H); }
  { float h;
    h = 0x1.b6db6e0000000p-4f * q26_1;
    h = fmaf(0x1.bee9060000000p-5f, q27_1, h);
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = 0x1.53eff20000000p-4f * q26_1;
    h = fmaf(0x1.5a2cdc0000000p-5f, q27_1, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.158ec20000000p-4f * q26_2;
    h = fmaf(-0x1.1aa69e0000000p-5f, q27_2, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.c53fee0000000p-6f * q26_3;
    h = fmaf(0x1.03a1a40000000p-3f, q27_3, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.66534c0000000p-4f * q26_2;
    h = fmaf(0x1.11acf00000000p-3f, q27_2, h);
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = 0x1.b6db6e0000000p-4f * q26_3;
    h = fmaf(0x1.bee9060000000p-5f, q27_3, h);
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx7 = fmaf(H, x7, dx7); dx7 = fmaf(H, x7, dx7); }
  { const float p = x7 * x8; float H = 0.0f;
  { float h;
    h = 0x1.66534c0000000p-3f * q26_1;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
  { float h;
    h = -0x1.158ec00000000p-3f * q26_3;
    h = fmaf(-0x1.1aa69e0000000p-3f, q27_3, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.c53fee0000000p-3f * q26_2;
    h = fmaf(0x1.5a2cda0000000p-4f, q27_2, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.66534c0000000p-3f * q26_3;
    h = fmaf(0x1.11acf00000000p-2f, q27_3, h);
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = 0x1.bee9080000000p-4f * q27_2;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx7 = fmaf(H, x8, dx7); dx8 = fmaf(H, x7, dx8); }
  { const float p = x7 * x9; float H = 0.0f;
  { float h;
    h = 0x1.b0b8100000000p-3f * q16_0;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
    dx7 = fmaf(H, x9, dx7); dx9 = fmaf(H, x7, dx9); }
  { const float p = x7 * x10; float H = 0.0f;
    H = 0x1.3c03660000000p-2f * q8_1;
  { float h;
    h = 0x1.4f2ec40000000p-3f * q16_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx7 = fmaf(H, x10, dx7); dx10 = fmaf(H, x7, dx10); }
  { const float p = x7 * x12; float H = 0.0f;
    H = -0x1.e990cc0000000p-3f * q8_3;
  { float h;
    h = 0x1.5a2cd80000000p-4f * q16_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx7 = fmaf(H, x12, dx7); dx12 = fmaf(H, x7, dx12); }
  { const float p = x7 * x13; float H = 0.0f;
    H = 0x1.8fba8a0000000p-2f * q8_2;
  { float h;
    h = 0x1.4f2ec40000000p-3f * q16_0;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx7 = fmaf(H, x13, dx7); dx13 = fmaf(H, x7, dx13); }
  { const float p = x7 * x14; float H = 0.0f;
    H = 0x1.3c03660000000p-2f * q8_3;
  { float h;
    h = 0x1.b0b8100000000p-3f * q16_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx7 = fmaf(H, x14, dx7); dx14 = fmaf(H, x7, dx14); }
  { const float p = x8 * x8; float H = 0.0f;
    H = 0x1.c9f25c0000000p-3f * q4_0;
  { float h;
    h = -0x1.bee9060000000p-4f * q27_1;
    dx9 = fmaf(h, p, dx9);
    H = fmaf(h, x9, H); }
  { float h;
    h = -0x1.c53fee0000000p-4f * q26_1;
    h = fmaf(-0x1.5a2cda0000000p-4f, q27_1, h);
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.158ec20000000p-3f * q26_2;
    h = fmaf(-0x1.1aa6a00000000p-4f, q27_2, h);
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.c53fee0000000p-4f * q26_3;
    h = fmaf(-0x1.5a2cda0000000p-4f, q27_3, h);
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.bee9060000000p-4f * q27_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx8 = fmaf(H, x8, dx8); dx8 = fmaf(H, x8, dx8); }
  { const float p = x8 * x9; float H = 0.0f;
    H = 0x1.83091e0000000p-2f * q8_1;
  { float h;
    h = -0x1.11acee0000000p-3f * q16_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx8 = fmaf(H, x9, dx8); dx9 = fmaf(H, x8, dx9); }
  { const float p = x8 * x11; float H = 0.0f;
    H = 0x1.8fba8a0000000p-4f * q8_1;
  { float h;
    h = -0x1.a7f9ec0000000p-4f * q16_0;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
    dx8 = fmaf(H, x11, dx8); dx11 = fmaf(H, x8, dx11); }
  { const float p = x8 * x12; float H = 0.0f;
  { float h;
    h = -0x1.11acee0000000p-2f * q16_0;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx8 = fmaf(H, x12, dx8); dx12 = fmaf(H, x8, dx12); }
  { const float p = x8 * x13; float H = 0.0f;
    H = -0x1.8fba8a0000000p-4f * q8_3;
  { float h;
    h = 0x1.a7f9ec0000000p-4f * q16_0;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.11acee0000000p-3f * q16_0;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx8 = fmaf(H, x13, dx8); dx13 = fmaf(H, x8, dx13); }
  { const float p = x8 * x14; float H = 0.0f;
    H = 0x1.3c03660000000p-2f * q8_2;
    dx8 = fmaf(H, x14, dx8); dx14 = fmaf(H, x8, dx14); }
  { const float p = x8 * x15; float H = 0.0f;
    H = 0x1.83091e0000000p-2f * q8_3;
    dx8 = fmaf(H, x15, dx8); dx15 = fmaf(H, x8, dx15); }
  { const float p = x9 * x9; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
  { float h;
    h = -0x1.7106240000000p-3f * q28_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.c3f5d20000000p-4f * q28_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = -0x1.7106240000000p-3f * q28_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx9 = fmaf(H, x9, dx9); dx9 = fmaf(H, x9, dx9); }
  { const float p = x9 * x10; float H = 0.0f;
  { float h;
    h = 0x1.dc686e0000000p-4f * q28_1;
    dx10 = fmaf(h, p, dx10);
    H = h * x10; }
  { float h;
    h = -0x1.14c49c0000000p-2f * q28_3;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.2d4e8e0000000p-4f * q28_2;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.dc686e0000000p-3f * q28_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx9 = fmaf(H, x10, dx9); dx10 = fmaf(H, x9, dx10); }
  { const float p = x9 * x11; float H = 0.0f;
  { float h;
    h = -0x1.7d20580000000p-4f * q28_1;
    dx11 = fmaf(h, p, dx11);
    H = h * x11; }
  { float h;
    h = 0x1.7d20580000000p-3f * q28_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.2d4e8e0000000p-4f * q28_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx9 = fmaf(H, x11, dx9); dx11 = fmaf(H, x9, dx11); }
  { const float p = x9 * x12; float H = 0.0f;
  { float h;
    h = -0x1.14c49c0000000p-2f * q28_1;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx9 = fmaf(H, x12, dx9); dx12 = fmaf(H, x9, dx12); }
  { const float p = x9 * x13; float H = 0.0f;
  { float h;
    h = 0x1.7d20580000000p-4f * q28_1;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
    dx9 = fmaf(H, x13, dx9); dx13 = fmaf(H, x9, dx13); }
  { const float p = x9 * x14; float H = 0.0f;
  { float h;
    h = -0x1.dc686e0000000p-4f * q28_1;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx9 = fmaf(H, x14, dx9); dx14 = fmaf(H, x9, dx14); }
  { const float p = x10 * x10; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
  { float h;
    h = 0x1.7106240000000p-4f * q28_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = -0x1.2d4e8c0000000p-3f * q28_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.7106240000000p-4f * q28_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.dc686e0000000p-4f * q28_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx10 = fmaf(H, x10, dx10); dx10 = fmaf(H, x10, dx10); }
  { const float p = x10 * x11; float H = 0.0f;
  { float h;
    h = 0x1.7d205c0000000p-6f * q28_3;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = 0x1.5e163e0000000p-2f * q28_2;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.2d4e8e0000000p-4f * q28_2;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx10 = fmaf(H, x11, dx10); dx11 = fmaf(H, x10, dx11); }
  { const float p = x10 * x12; float H = 0.0f;
  { float h;
    h = 0x1.7d205c0000000p-6f * q28_1;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.14c49c0000000p-2f * q28_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx10 = fmaf(H, x12, dx10); dx12 = fmaf(H, x10, dx12); }
  { const float p = x10 * x14; float H = 0.0f;
  { float h;
    h = 0x1.dc686e0000000p-3f * q28_1;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx10 = fmaf(H, x14, dx10); dx14 = fmaf(H, x10, dx14); }
  { const float p = x11 * x11; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
  { float h;
    h = 0x1.27381e0000000p-5f * q28_1;
    dx11 = fmaf(h, p, dx11);
    H = fmaf(h, x11, H); }
  { float h;
    h = 0x1.001c5e0000000p-3f * q28_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.27381e0000000p-5f * q28_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = -0x1.5e163e0000000p-3f * q28_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = -0x1.7d20580000000p-4f * q28_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx11 = fmaf(H, x11, dx11); dx11 = fmaf(H, x11, dx11); }
  { const float p = x11 * x12; float H = 0.0f;
  { float h;
    h = 0x1.27381c0000000p-5f * q28_1;
    dx12 = fmaf(h, p, dx12);
    H = h * x12; }
  { float h;
    h = -0x1.7d205c0000000p-6f * q28_1;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx11 = fmaf(H, x12, dx11); dx12 = fmaf(H, x11, dx12); }
  { const float p = x11 * x13; float H = 0.0f;
  { float h;
    h = 0x1.27381e0000000p-5f * q28_1;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = -0x1.7d20580000000p-3f * q28_1;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx11 = fmaf(H, x13, dx11); dx13 = fmaf(H, x11, dx13); }
  { const float p = x11 * x14; float H = 0.0f;
  { float h;
    h = 0x1.7106240000000p-4f * q28_1;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
    dx11 = fmaf(H, x14, dx11); dx14 = fmaf(H, x11, dx14); }
  { const float p = x11 * x15; float H = 0.0f;
  { float h;
    h = -0x1.7106240000000p-3f * q28_1;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx11 = fmaf(H, x15, dx11); dx15 = fmaf(H, x11, dx15); }
  { const float p = x12 * x12; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
  { float h;
    h = 0x1.6991760000000p-4f * q28_2;
    dx12 = fmaf(h, p, dx12);
    H = fmaf(h, x12, H); }
  { float h;
    h = 0x1.27381c0000000p-5f * q28_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
    dx12 = fmaf(H, x12, dx12); dx12 = fmaf(H, x12, dx12); }
  { const float p = x12 * x13; float H = 0.0f;
  { float h;
    h = 0x1.001c5e0000000p-3f * q28_2;
    dx13 = fmaf(h, p, dx13);
    H = h * x13; }
  { float h;
    h = 0x1.7d205c0000000p-6f * q28_3;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
    dx12 = fmaf(H, x13, dx12); dx13 = fmaf(H, x12, dx13); }
  { const float p = x12 * x14; float H = 0.0f;
  { float h;
    h = -0x1.2d4e8c0000000p-3f * q28_2;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
  { float h;
    h = -0x1.14c49c0000000p-2f * q28_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx12 = fmaf(H, x14, dx12); dx14 = fmaf(H, x12, dx14); }
  { const float p = x12 * x15; float H = 0.0f;
  { float h;
    h = -0x1.c3f5d20000000p-4f * q28_2;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx12 = fmaf(H, x15, dx12); dx15 = fmaf(H, x12, dx15); }
  { const float p = x13 * x13; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
  { float h;
    h = 0x1.27381e0000000p-5f * q28_3;
    dx13 = fmaf(h, p, dx13);
    H = fmaf(h, x13, H); }
  { float h;
    h = 0x1.5e163e0000000p-3f * q28_2;
    dx14 = fmaf(h, p, dx14);
    H = fmaf(h, x14, H); }
  { float h;
    h = 0x1.7d20580000000p-4f * q28_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx13 = fmaf(H, x13, dx13); dx13 = fmaf(H, x13, dx13); }
  { const float p = x13 * x14; float H = 0.0f;
  { float h;
    h = 0x1.7106240000000p-4f * q28_3;
    dx14 = fmaf(h, p, dx14);
    H = h * x14; }
  { float h;
    h = 0x1.2d4e8e0000000p-4f * q28_2;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx13 = fmaf(H, x14, dx13); dx14 = fmaf(H, x13, dx14); }
  { const float p = x13 * x15; float H = 0.0f;
  { float h;
    h = -0x1.7106240000000p-3f * q28_3;
    dx15 = fmaf(h, p, dx15);
    H = h * x15; }
    dx13 = fmaf(H, x15, dx13); dx15 = fmaf(H, x13, dx15); }
  { const float p = x14 * x14; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
  { float h;
    h = 0x1.dc686e0000000p-4f * q28_3;
    dx15 = fmaf(h, p, dx15);
    H = fmaf(h, x15, H); }
    dx14 = fmaf(H, x14, dx14); dx14 = fmaf(H, x14, dx14); }
  { const float p = x15 * x15; float H = 0.0f;
    H = 0x1.83091e0000000p-3f * q5_0;
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
