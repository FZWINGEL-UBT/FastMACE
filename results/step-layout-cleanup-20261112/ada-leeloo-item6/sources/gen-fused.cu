// generated: 94 terms, 94 monomials, fused division: True
extern "C" __global__ void __launch_bounds__(128) sc0_fwd(
    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,
    const int* __restrict__ sp, float* __restrict__ y) {
  const int a = blockIdx.x, ch = threadIdx.x;
  const float* xr = m + (long long)a * 2048 + ch;
  const float* wr = w + (long long)sp[a] * 1664 + ch;
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
  float t0_0, t1_0, t2_0, t3_0, t4_0, t5_0, t6_0, t7_0, t8_0, t9_0, t10_0, t11_0, t12_0;
  t0_0 = 0x1.0000000000000p+0f * x0;
  { const float p = x0 * x0;
  t1_0 = 0x1.0000000000000p-1f * p;
  { const float q = p * x0;
  t5_0 = 0x1.6a09e60000000p-2f * q;
  }
  }
  { const float p = x0 * x1;
  { const float q = p * x1;
  t6_0 = 0x1.6a09e60000000p-2f * q;
  }
  }
  { const float p = x0 * x2;
  { const float q = p * x2;
  t6_0 = fmaf(0x1.6a09e60000000p-2f, q, t6_0);
  }
  }
  { const float p = x0 * x3;
  { const float q = p * x3;
  t6_0 = fmaf(0x1.6a09e60000000p-2f, q, t6_0);
  }
  }
  { const float p = x0 * x4;
  { const float q = p * x4;
  t7_0 = 0x1.186f180000000p-2f * q;
  }
  }
  { const float p = x0 * x5;
  { const float q = p * x5;
  t7_0 = fmaf(0x1.186f180000000p-2f, q, t7_0);
  }
  }
  { const float p = x0 * x6;
  { const float q = p * x6;
  t7_0 = fmaf(0x1.186f180000000p-2f, q, t7_0);
  }
  }
  { const float p = x0 * x7;
  { const float q = p * x7;
  t7_0 = fmaf(0x1.186f180000000p-2f, q, t7_0);
  }
  }
  { const float p = x0 * x8;
  { const float q = p * x8;
  t7_0 = fmaf(0x1.186f180000000p-2f, q, t7_0);
  }
  }
  { const float p = x0 * x9;
  { const float q = p * x9;
  t8_0 = 0x1.da05180000000p-3f * q;
  }
  }
  { const float p = x0 * x10;
  { const float q = p * x10;
  t8_0 = fmaf(0x1.da05180000000p-3f, q, t8_0);
  }
  }
  { const float p = x0 * x11;
  { const float q = p * x11;
  t8_0 = fmaf(0x1.da05180000000p-3f, q, t8_0);
  }
  }
  { const float p = x0 * x12;
  { const float q = p * x12;
  t8_0 = fmaf(0x1.da05180000000p-3f, q, t8_0);
  }
  }
  { const float p = x0 * x13;
  { const float q = p * x13;
  t8_0 = fmaf(0x1.da05180000000p-3f, q, t8_0);
  }
  }
  { const float p = x0 * x14;
  { const float q = p * x14;
  t8_0 = fmaf(0x1.da05180000000p-3f, q, t8_0);
  }
  }
  { const float p = x0 * x15;
  { const float q = p * x15;
  t8_0 = fmaf(0x1.da05180000000p-3f, q, t8_0);
  }
  }
  { const float p = x1 * x1;
  t2_0 = 0x1.279a740000000p-2f * p;
  { const float q = p * x6;
  t9_0 = -0x1.c9f25c0000000p-4f * q;
  }
  { const float q = p * x8;
  t9_0 = fmaf(-0x1.8c97f00000000p-3f, q, t9_0);
  }
  }
  { const float p = x1 * x2;
  { const float q = p * x5;
  t9_0 = fmaf(0x1.8c97f00000000p-2f, q, t9_0);
  }
  }
  { const float p = x1 * x3;
  { const float q = p * x4;
  t9_0 = fmaf(0x1.8c97f00000000p-2f, q, t9_0);
  }
  }
  { const float p = x1 * x4;
  { const float q = p * x13;
  t10_0 = -0x1.e990ce0000000p-5f * q;
  }
  { const float q = p * x15;
  t10_0 = fmaf(-0x1.da05160000000p-3f, q, t10_0);
  }
  }
  { const float p = x1 * x5;
  { const float q = p * x12;
  t10_0 = fmaf(-0x1.2bcbe60000000p-3f, q, t10_0);
  }
  { const float q = p * x14;
  t10_0 = fmaf(-0x1.83091e0000000p-3f, q, t10_0);
  }
  }
  { const float p = x1 * x6;
  { const float q = p * x11;
  t10_0 = fmaf(0x1.a7f9ee0000000p-3f, q, t10_0);
  }
  }
  { const float p = x1 * x7;
  { const float q = p * x10;
  t10_0 = fmaf(0x1.83091e0000000p-3f, q, t10_0);
  }
  }
  { const float p = x1 * x8;
  { const float q = p * x9;
  t10_0 = fmaf(0x1.da05160000000p-3f, q, t10_0);
  }
  { const float q = p * x11;
  t10_0 = fmaf(0x1.e990ce0000000p-5f, q, t10_0);
  }
  }
  { const float p = x2 * x2;
  t2_0 = fmaf(0x1.279a740000000p-2f, p, t2_0);
  { const float q = p * x6;
  t9_0 = fmaf(0x1.c9f25c0000000p-3f, q, t9_0);
  }
  }
  { const float p = x2 * x3;
  { const float q = p * x7;
  t9_0 = fmaf(0x1.8c97f00000000p-2f, q, t9_0);
  }
  }
  { const float p = x2 * x4;
  { const float q = p * x10;
  t10_0 = fmaf(0x1.83091e0000000p-3f, q, t10_0);
  }
  }
  { const float p = x2 * x5;
  { const float q = p * x11;
  t10_0 = fmaf(0x1.e990ce0000000p-3f, q, t10_0);
  }
  }
  { const float p = x2 * x6;
  { const float q = p * x12;
  t10_0 = fmaf(0x1.03a1a20000000p-2f, q, t10_0);
  }
  }
  { const float p = x2 * x7;
  { const float q = p * x13;
  t10_0 = fmaf(0x1.e990ce0000000p-3f, q, t10_0);
  }
  }
  { const float p = x2 * x8;
  { const float q = p * x14;
  t10_0 = fmaf(0x1.83091e0000000p-3f, q, t10_0);
  }
  }
  { const float p = x3 * x3;
  t2_0 = fmaf(0x1.279a740000000p-2f, p, t2_0);
  { const float q = p * x6;
  t9_0 = fmaf(-0x1.c9f25c0000000p-4f, q, t9_0);
  }
  { const float q = p * x8;
  t9_0 = fmaf(0x1.8c97f00000000p-3f, q, t9_0);
  }
  }
  { const float p = x3 * x4;
  { const float q = p * x9;
  t10_0 = fmaf(0x1.da05160000000p-3f, q, t10_0);
  }
  { const float q = p * x11;
  t10_0 = fmaf(-0x1.e990ce0000000p-5f, q, t10_0);
  }
  }
  { const float p = x3 * x5;
  { const float q = p * x10;
  t10_0 = fmaf(0x1.83091e0000000p-3f, q, t10_0);
  }
  }
  { const float p = x3 * x6;
  { const float q = p * x13;
  t10_0 = fmaf(0x1.a7f9ee0000000p-3f, q, t10_0);
  }
  }
  { const float p = x3 * x7;
  { const float q = p * x12;
  t10_0 = fmaf(-0x1.2bcbe60000000p-3f, q, t10_0);
  }
  { const float q = p * x14;
  t10_0 = fmaf(0x1.83091e0000000p-3f, q, t10_0);
  }
  }
  { const float p = x3 * x8;
  { const float q = p * x13;
  t10_0 = fmaf(-0x1.e990ce0000000p-5f, q, t10_0);
  }
  { const float q = p * x15;
  t10_0 = fmaf(0x1.da05160000000p-3f, q, t10_0);
  }
  }
  { const float p = x4 * x4;
  t3_0 = 0x1.c9f25c0000000p-3f * p;
  { const float q = p * x6;
  t11_0 = -0x1.03a1a20000000p-2f * q;
  }
  }
  { const float p = x4 * x5;
  { const float q = p * x7;
  t11_0 = fmaf(0x1.c1b1dc0000000p-2f, q, t11_0);
  }
  }
  { const float p = x4 * x9;
  { const float q = p * x13;
  t12_0 = -0x1.11acee0000000p-3f * q;
  }
  }
  { const float p = x4 * x10;
  { const float q = p * x12;
  t12_0 = fmaf(-0x1.11acee0000000p-2f, q, t12_0);
  }
  }
  { const float p = x4 * x11;
  { const float q = p * x13;
  t12_0 = fmaf(0x1.a7f9ec0000000p-3f, q, t12_0);
  }
  { const float q = p * x15;
  t12_0 = fmaf(0x1.11acee0000000p-3f, q, t12_0);
  }
  }
  { const float p = x5 * x5;
  t3_0 = fmaf(0x1.c9f25c0000000p-3f, p, t3_0);
  { const float q = p * x6;
  t11_0 = fmaf(0x1.03a1a20000000p-3f, q, t11_0);
  }
  { const float q = p * x8;
  t11_0 = fmaf(-0x1.c1b1dc0000000p-3f, q, t11_0);
  }
  }
  { const float p = x5 * x9;
  { const float q = p * x14;
  t12_0 = fmaf(0x1.b0b8100000000p-3f, q, t12_0);
  }
  }
  { const float p = x5 * x10;
  { const float q = p * x13;
  t12_0 = fmaf(0x1.4f2ec40000000p-3f, q, t12_0);
  }
  { const float q = p * x15;
  t12_0 = fmaf(-0x1.b0b8100000000p-3f, q, t12_0);
  }
  }
  { const float p = x5 * x11;
  { const float q = p * x12;
  t12_0 = fmaf(0x1.5a2cd80000000p-4f, q, t12_0);
  }
  { const float q = p * x14;
  t12_0 = fmaf(-0x1.4f2ec40000000p-3f, q, t12_0);
  }
  }
  { const float p = x6 * x6;
  t3_0 = fmaf(0x1.c9f25c0000000p-3f, p, t3_0);
  { const float q = p * x6;
  t11_0 = fmaf(0x1.5a2cda0000000p-4f, q, t11_0);
  }
  }
  { const float p = x6 * x7;
  { const float q = p * x7;
  t11_0 = fmaf(0x1.03a1a20000000p-3f, q, t11_0);
  }
  }
  { const float p = x6 * x8;
  { const float q = p * x8;
  t11_0 = fmaf(-0x1.03a1a20000000p-2f, q, t11_0);
  }
  }
  { const float p = x6 * x9;
  { const float q = p * x9;
  t12_0 = fmaf(-0x1.31fa800000000p-3f, q, t12_0);
  }
  }
  { const float p = x6 * x11;
  { const float q = p * x11;
  t12_0 = fmaf(0x1.6f2c9a0000000p-4f, q, t12_0);
  }
  }
  { const float p = x6 * x12;
  { const float q = p * x12;
  t12_0 = fmaf(0x1.e990ce0000000p-4f, q, t12_0);
  }
  }
  { const float p = x6 * x13;
  { const float q = p * x13;
  t12_0 = fmaf(0x1.6f2c9a0000000p-4f, q, t12_0);
  }
  }
  { const float p = x6 * x15;
  { const float q = p * x15;
  t12_0 = fmaf(-0x1.31fa800000000p-3f, q, t12_0);
  }
  }
  { const float p = x7 * x7;
  t3_0 = fmaf(0x1.c9f25c0000000p-3f, p, t3_0);
  { const float q = p * x8;
  t11_0 = fmaf(0x1.c1b1dc0000000p-3f, q, t11_0);
  }
  }
  { const float p = x7 * x9;
  { const float q = p * x10;
  t12_0 = fmaf(0x1.b0b8100000000p-3f, q, t12_0);
  }
  }
  { const float p = x7 * x10;
  { const float q = p * x11;
  t12_0 = fmaf(0x1.4f2ec40000000p-3f, q, t12_0);
  }
  }
  { const float p = x7 * x12;
  { const float q = p * x13;
  t12_0 = fmaf(0x1.5a2cd80000000p-4f, q, t12_0);
  }
  }
  { const float p = x7 * x13;
  { const float q = p * x14;
  t12_0 = fmaf(0x1.4f2ec40000000p-3f, q, t12_0);
  }
  }
  { const float p = x7 * x14;
  { const float q = p * x15;
  t12_0 = fmaf(0x1.b0b8100000000p-3f, q, t12_0);
  }
  }
  { const float p = x8 * x8;
  t3_0 = fmaf(0x1.c9f25c0000000p-3f, p, t3_0);
  }
  { const float p = x8 * x9;
  { const float q = p * x11;
  t12_0 = fmaf(-0x1.11acee0000000p-3f, q, t12_0);
  }
  }
  { const float p = x8 * x11;
  { const float q = p * x11;
  t12_0 = fmaf(-0x1.a7f9ec0000000p-4f, q, t12_0);
  }
  }
  { const float p = x8 * x12;
  { const float q = p * x14;
  t12_0 = fmaf(-0x1.11acee0000000p-2f, q, t12_0);
  }
  }
  { const float p = x8 * x13;
  { const float q = p * x13;
  t12_0 = fmaf(0x1.a7f9ec0000000p-4f, q, t12_0);
  }
  { const float q = p * x15;
  t12_0 = fmaf(-0x1.11acee0000000p-3f, q, t12_0);
  }
  }
  { const float p = x9 * x9;
  t4_0 = 0x1.83091e0000000p-3f * p;
  }
  { const float p = x10 * x10;
  t4_0 = fmaf(0x1.83091e0000000p-3f, p, t4_0);
  }
  { const float p = x11 * x11;
  t4_0 = fmaf(0x1.83091e0000000p-3f, p, t4_0);
  }
  { const float p = x12 * x12;
  t4_0 = fmaf(0x1.83091e0000000p-3f, p, t4_0);
  }
  { const float p = x13 * x13;
  t4_0 = fmaf(0x1.83091e0000000p-3f, p, t4_0);
  }
  { const float p = x14 * x14;
  t4_0 = fmaf(0x1.83091e0000000p-3f, p, t4_0);
  }
  { const float p = x15 * x15;
  t4_0 = fmaf(0x1.83091e0000000p-3f, p, t4_0);
  }
  float y0 = 0.0f;
  y0 = fmaf(__ldg(wr + 0), t0_0, y0);
  y0 = fmaf(__ldg(wr + 128), t1_0, y0);
  y0 = fmaf(__ldg(wr + 256), t2_0, y0);
  y0 = fmaf(__ldg(wr + 384), t3_0, y0);
  y0 = fmaf(__ldg(wr + 512), t4_0, y0);
  y0 = fmaf(__ldg(wr + 640), t5_0, y0);
  y0 = fmaf(__ldg(wr + 768), t6_0, y0);
  y0 = fmaf(__ldg(wr + 896), t7_0, y0);
  y0 = fmaf(__ldg(wr + 1024), t8_0, y0);
  y0 = fmaf(__ldg(wr + 1152), t9_0, y0);
  y0 = fmaf(__ldg(wr + 1280), t10_0, y0);
  y0 = fmaf(__ldg(wr + 1408), t11_0, y0);
  y0 = fmaf(__ldg(wr + 1536), t12_0, y0);
  y[(long long)a * 128 + 0 + ch] = y0;
}
