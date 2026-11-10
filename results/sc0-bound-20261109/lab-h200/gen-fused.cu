// generated: 480 terms, 410 monomials, fused division: True
extern "C" __global__ void __launch_bounds__(128) sc0_fwd(
    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,
    const int* __restrict__ sp, float* __restrict__ y) {
  const int a = blockIdx.x, ch = threadIdx.x;
  const float* xr = m + (long long)a * 2048 + ch;
  const float* wr = w + (long long)sp[a] * 3712 + ch;
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
  float t0_0, t1_1, t1_2, t1_3, t2_0, t3_0, t4_0, t5_0, t6_1, t6_2, t6_3, t7_1, t7_2, t7_3, t8_1, t8_2, t8_3, t9_0, t10_0, t11_0, t12_0, t13_0, t14_0, t15_0, t16_0, t17_1, t17_2, t17_3, t18_1, t18_2, t18_3, t19_1, t19_2, t19_3, t20_1, t20_2, t20_3, t21_1, t21_2, t21_3, t22_1, t22_2, t22_3, t23_1, t23_2, t23_3, t24_1, t24_2, t24_3, t25_1, t25_2, t25_3, t26_1, t26_2, t26_3, t27_1, t27_2, t27_3, t28_1, t28_2, t28_3;
  t0_0 = 0x1.0000000000000p+0f * x0;
  t1_1 = 0x1.0000000000000p+0f * x1;
  t1_2 = 0x1.0000000000000p+0f * x2;
  t1_3 = 0x1.0000000000000p+0f * x3;
  { const float p = x0 * x0;
  t2_0 = 0x1.0000000000000p-1f * p;
  { const float q = p * x0;
  t9_0 = 0x1.6a09e60000000p-2f * q;
  }
  { const float q = p * x1;
  t17_1 = 0x1.0000000000000p-1f * q;
  }
  { const float q = p * x2;
  t17_2 = 0x1.0000000000000p-1f * q;
  }
  { const float q = p * x3;
  t17_3 = 0x1.0000000000000p-1f * q;
  }
  }
  { const float p = x0 * x1;
  t6_1 = 0x1.a20bd60000000p-1f * p;
  { const float q = p * x1;
  t10_0 = 0x1.6a09e60000000p-2f * q;
  }
  { const float q = p * x4;
  t18_3 = 0x1.8c97ee0000000p-2f * q;
  }
  { const float q = p * x5;
  t18_2 = 0x1.8c97ee0000000p-2f * q;
  }
  { const float q = p * x6;
  t18_1 = -0x1.c9f25c0000000p-3f * q;
  }
  { const float q = p * x8;
  t18_1 = fmaf(-0x1.8c97ee0000000p-2f, q, t18_1);
  }
  }
  { const float p = x0 * x2;
  t6_2 = 0x1.a20bd60000000p-1f * p;
  { const float q = p * x2;
  t10_0 = fmaf(0x1.6a09e60000000p-2f, q, t10_0);
  }
  { const float q = p * x5;
  t18_1 = fmaf(0x1.8c97ee0000000p-2f, q, t18_1);
  }
  { const float q = p * x6;
  t18_2 = fmaf(0x1.c9f25c0000000p-2f, q, t18_2);
  }
  { const float q = p * x7;
  t18_3 = fmaf(0x1.8c97ee0000000p-2f, q, t18_3);
  }
  }
  { const float p = x0 * x3;
  t6_3 = 0x1.a20bd60000000p-1f * p;
  { const float q = p * x3;
  t10_0 = fmaf(0x1.6a09e60000000p-2f, q, t10_0);
  }
  { const float q = p * x4;
  t18_1 = fmaf(0x1.8c97ee0000000p-2f, q, t18_1);
  }
  { const float q = p * x6;
  t18_3 = fmaf(-0x1.c9f25c0000000p-3f, q, t18_3);
  }
  { const float q = p * x7;
  t18_2 = fmaf(0x1.8c97ee0000000p-2f, q, t18_2);
  }
  { const float q = p * x8;
  t18_3 = fmaf(0x1.8c97ee0000000p-2f, q, t18_3);
  }
  }
  { const float p = x0 * x4;
  { const float q = p * x4;
  t11_0 = 0x1.186f180000000p-2f * q;
  }
  { const float q = p * x9;
  t19_3 = 0x1.4f2ec40000000p-2f * q;
  }
  { const float q = p * x10;
  t19_2 = 0x1.11acee0000000p-2f * q;
  }
  { const float q = p * x11;
  t19_3 = fmaf(-0x1.5a2cd80000000p-4f, q, t19_3);
  }
  { const float q = p * x13;
  t19_1 = -0x1.5a2cd80000000p-4f * q;
  }
  { const float q = p * x15;
  t19_1 = fmaf(-0x1.4f2ec40000000p-2f, q, t19_1);
  }
  }
  { const float p = x0 * x5;
  { const float q = p * x5;
  t11_0 = fmaf(0x1.186f180000000p-2f, q, t11_0);
  }
  { const float q = p * x10;
  t19_3 = fmaf(0x1.11acee0000000p-2f, q, t19_3);
  }
  { const float q = p * x11;
  t19_2 = fmaf(0x1.5a2cd80000000p-2f, q, t19_2);
  }
  { const float q = p * x12;
  t19_1 = fmaf(-0x1.a7f9ec0000000p-3f, q, t19_1);
  }
  { const float q = p * x14;
  t19_1 = fmaf(-0x1.11acee0000000p-2f, q, t19_1);
  }
  }
  { const float p = x0 * x6;
  { const float q = p * x6;
  t11_0 = fmaf(0x1.186f180000000p-2f, q, t11_0);
  }
  { const float q = p * x11;
  t19_1 = fmaf(0x1.2bcbe80000000p-2f, q, t19_1);
  }
  { const float q = p * x12;
  t19_2 = fmaf(0x1.6f2c9a0000000p-2f, q, t19_2);
  }
  { const float q = p * x13;
  t19_3 = fmaf(0x1.2bcbe80000000p-2f, q, t19_3);
  }
  }
  { const float p = x0 * x7;
  { const float q = p * x7;
  t11_0 = fmaf(0x1.186f180000000p-2f, q, t11_0);
  }
  { const float q = p * x10;
  t19_1 = fmaf(0x1.11acee0000000p-2f, q, t19_1);
  }
  { const float q = p * x12;
  t19_3 = fmaf(-0x1.a7f9ec0000000p-3f, q, t19_3);
  }
  { const float q = p * x13;
  t19_2 = fmaf(0x1.5a2cd80000000p-2f, q, t19_2);
  }
  { const float q = p * x14;
  t19_3 = fmaf(0x1.11acee0000000p-2f, q, t19_3);
  }
  }
  { const float p = x0 * x8;
  { const float q = p * x8;
  t11_0 = fmaf(0x1.186f180000000p-2f, q, t11_0);
  }
  { const float q = p * x9;
  t19_1 = fmaf(0x1.4f2ec40000000p-2f, q, t19_1);
  }
  { const float q = p * x11;
  t19_1 = fmaf(0x1.5a2cd80000000p-4f, q, t19_1);
  }
  { const float q = p * x13;
  t19_3 = fmaf(-0x1.5a2cd80000000p-4f, q, t19_3);
  }
  { const float q = p * x14;
  t19_2 = fmaf(0x1.11acee0000000p-2f, q, t19_2);
  }
  { const float q = p * x15;
  t19_3 = fmaf(0x1.4f2ec40000000p-2f, q, t19_3);
  }
  }
  { const float p = x0 * x9;
  { const float q = p * x9;
  t12_0 = 0x1.da05180000000p-3f * q;
  }
  }
  { const float p = x0 * x10;
  { const float q = p * x10;
  t12_0 = fmaf(0x1.da05180000000p-3f, q, t12_0);
  }
  }
  { const float p = x0 * x11;
  { const float q = p * x11;
  t12_0 = fmaf(0x1.da05180000000p-3f, q, t12_0);
  }
  }
  { const float p = x0 * x12;
  { const float q = p * x12;
  t12_0 = fmaf(0x1.da05180000000p-3f, q, t12_0);
  }
  }
  { const float p = x0 * x13;
  { const float q = p * x13;
  t12_0 = fmaf(0x1.da05180000000p-3f, q, t12_0);
  }
  }
  { const float p = x0 * x14;
  { const float q = p * x14;
  t12_0 = fmaf(0x1.da05180000000p-3f, q, t12_0);
  }
  }
  { const float p = x0 * x15;
  { const float q = p * x15;
  t12_0 = fmaf(0x1.da05180000000p-3f, q, t12_0);
  }
  }
  { const float p = x1 * x1;
  t3_0 = 0x1.279a740000000p-2f * p;
  { const float q = p * x1;
  t20_1 = 0x1.c9f25c0000000p-3f * q;
  }
  { const float q = p * x2;
  t20_2 = 0x1.c9f25c0000000p-3f * q;
  }
  { const float q = p * x3;
  t20_3 = 0x1.c9f25c0000000p-3f * q;
  }
  { const float q = p * x6;
  t13_0 = -0x1.c9f25c0000000p-4f * q;
  }
  { const float q = p * x8;
  t13_0 = fmaf(-0x1.8c97f00000000p-3f, q, t13_0);
  }
  { const float q = p * x9;
  t21_1 = -0x1.4f2ec40000000p-3f * q;
  }
  { const float q = p * x11;
  t21_1 = fmaf(-0x1.03a1a20000000p-3f, q, t21_1);
  }
  { const float q = p * x12;
  t21_2 = -0x1.a7f9ee0000000p-4f * q;
  }
  { const float q = p * x13;
  t21_3 = -0x1.5a2cda0000000p-5f * q;
  }
  { const float q = p * x14;
  t21_2 = fmaf(-0x1.11acee0000000p-3f, q, t21_2);
  }
  { const float q = p * x15;
  t21_3 = fmaf(-0x1.4f2ec40000000p-3f, q, t21_3);
  }
  }
  { const float p = x1 * x2;
  { const float q = p * x2;
  t20_1 = fmaf(0x1.c9f25c0000000p-3f, q, t20_1);
  }
  { const float q = p * x5;
  t13_0 = fmaf(0x1.8c97f00000000p-2f, q, t13_0);
  }
  { const float q = p * x10;
  t21_3 = fmaf(0x1.11acee0000000p-2f, q, t21_3);
  }
  { const float q = p * x11;
  t21_2 = fmaf(0x1.5a2cd80000000p-2f, q, t21_2);
  }
  { const float q = p * x12;
  t21_1 = fmaf(-0x1.a7f9ec0000000p-3f, q, t21_1);
  }
  { const float q = p * x14;
  t21_1 = fmaf(-0x1.11acee0000000p-2f, q, t21_1);
  }
  }
  { const float p = x1 * x3;
  { const float q = p * x3;
  t20_1 = fmaf(0x1.c9f25c0000000p-3f, q, t20_1);
  }
  { const float q = p * x4;
  t13_0 = fmaf(0x1.8c97f00000000p-2f, q, t13_0);
  }
  { const float q = p * x9;
  t21_3 = fmaf(0x1.4f2ec40000000p-2f, q, t21_3);
  }
  { const float q = p * x10;
  t21_2 = fmaf(0x1.11acee0000000p-2f, q, t21_2);
  }
  { const float q = p * x11;
  t21_3 = fmaf(-0x1.5a2cd80000000p-4f, q, t21_3);
  }
  { const float q = p * x13;
  t21_1 = fmaf(-0x1.5a2cd80000000p-4f, q, t21_1);
  }
  { const float q = p * x15;
  t21_1 = fmaf(-0x1.4f2ec40000000p-2f, q, t21_1);
  }
  }
  { const float p = x1 * x4;
  t7_3 = 0x1.c9f25c0000000p-2f * p;
  { const float q = p * x4;
  t22_1 = 0x1.c9f25c0000000p-3f * q;
  }
  { const float q = p * x6;
  t23_3 = 0x1.186f180000000p-2f * q;
  }
  { const float q = p * x7;
  t23_2 = -0x1.e5b9d20000000p-3f * q;
  }
  { const float q = p * x13;
  t14_0 = -0x1.e990ce0000000p-5f * q;
  }
  { const float q = p * x15;
  t14_0 = fmaf(-0x1.da05160000000p-3f, q, t14_0);
  }
  }
  { const float p = x1 * x5;
  t7_2 = 0x1.c9f25c0000000p-2f * p;
  { const float q = p * x5;
  t22_1 = fmaf(0x1.c9f25c0000000p-3f, q, t22_1);
  }
  { const float q = p * x6;
  t23_2 = fmaf(-0x1.186f180000000p-3f, q, t23_2);
  }
  { const float q = p * x7;
  t23_3 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_3);
  }
  { const float q = p * x8;
  t23_2 = fmaf(0x1.e5b9d20000000p-3f, q, t23_2);
  }
  { const float q = p * x12;
  t14_0 = fmaf(-0x1.2bcbe60000000p-3f, q, t14_0);
  }
  { const float q = p * x14;
  t14_0 = fmaf(-0x1.83091e0000000p-3f, q, t14_0);
  }
  }
  { const float p = x1 * x6;
  t7_1 = -0x1.08654a0000000p-2f * p;
  { const float q = p * x6;
  t22_1 = fmaf(0x1.c9f25c0000000p-3f, q, t22_1);
  t23_1 = 0x1.43d1360000000p-3f * q;
  }
  { const float q = p * x8;
  t23_1 = fmaf(-0x1.186f180000000p-2f, q, t23_1);
  }
  { const float q = p * x11;
  t14_0 = fmaf(0x1.a7f9ee0000000p-3f, q, t14_0);
  }
  }
  { const float p = x1 * x7;
  { const float q = p * x7;
  t22_1 = fmaf(0x1.c9f25c0000000p-3f, q, t22_1);
  t23_1 = fmaf(0x1.e5b9d20000000p-3f, q, t23_1);
  }
  { const float q = p * x10;
  t14_0 = fmaf(0x1.83091e0000000p-3f, q, t14_0);
  }
  }
  { const float p = x1 * x8;
  t7_1 = fmaf(-0x1.c9f25c0000000p-2f, p, t7_1);
  { const float q = p * x8;
  t22_1 = fmaf(0x1.c9f25c0000000p-3f, q, t22_1);
  }
  { const float q = p * x9;
  t14_0 = fmaf(0x1.da05160000000p-3f, q, t14_0);
  }
  { const float q = p * x11;
  t14_0 = fmaf(0x1.e990ce0000000p-5f, q, t14_0);
  }
  }
  { const float p = x1 * x9;
  { const float q = p * x9;
  t24_1 = 0x1.83091e0000000p-3f * q;
  t25_1 = 0x1.6150460000000p-4f * q;
  }
  { const float q = p * x11;
  t25_1 = fmaf(0x1.11acee0000000p-3f, q, t25_1);
  }
  { const float q = p * x13;
  t25_3 = -0x1.11acee0000000p-3f * q;
  }
  { const float q = p * x14;
  t25_2 = 0x1.b0b8100000000p-3f * q;
  }
  }
  { const float p = x1 * x10;
  { const float q = p * x10;
  t24_1 = fmaf(0x1.83091e0000000p-3f, q, t24_1);
  }
  { const float q = p * x12;
  t25_3 = fmaf(-0x1.11acee0000000p-2f, q, t25_3);
  }
  { const float q = p * x13;
  t25_2 = fmaf(0x1.4f2ec40000000p-3f, q, t25_2);
  }
  { const float q = p * x15;
  t25_2 = fmaf(-0x1.b0b8100000000p-3f, q, t25_2);
  }
  }
  { const float p = x1 * x11;
  { const float q = p * x11;
  t24_1 = fmaf(0x1.83091e0000000p-3f, q, t24_1);
  t25_1 = fmaf(0x1.a7f9ec0000000p-5f, q, t25_1);
  }
  { const float q = p * x12;
  t25_2 = fmaf(0x1.5a2cd80000000p-4f, q, t25_2);
  }
  { const float q = p * x13;
  t25_3 = fmaf(0x1.a7f9ec0000000p-3f, q, t25_3);
  }
  { const float q = p * x14;
  t25_2 = fmaf(-0x1.4f2ec40000000p-3f, q, t25_2);
  }
  { const float q = p * x15;
  t25_3 = fmaf(0x1.11acee0000000p-3f, q, t25_3);
  }
  }
  { const float p = x1 * x12;
  { const float q = p * x12;
  t24_1 = fmaf(0x1.83091e0000000p-3f, q, t24_1);
  t25_1 = fmaf(-0x1.1aa69e0000000p-4f, q, t25_1);
  }
  { const float q = p * x14;
  t25_1 = fmaf(0x1.11acee0000000p-2f, q, t25_1);
  }
  }
  { const float p = x1 * x13;
  { const float q = p * x13;
  t24_1 = fmaf(0x1.83091e0000000p-3f, q, t24_1);
  t25_1 = fmaf(-0x1.3dfb720000000p-3f, q, t25_1);
  }
  { const float q = p * x15;
  t25_1 = fmaf(0x1.11acee0000000p-3f, q, t25_1);
  }
  }
  { const float p = x1 * x14;
  { const float q = p * x14;
  t24_1 = fmaf(0x1.83091e0000000p-3f, q, t24_1);
  }
  }
  { const float p = x1 * x15;
  { const float q = p * x15;
  t24_1 = fmaf(0x1.83091e0000000p-3f, q, t24_1);
  t25_1 = fmaf(0x1.6150460000000p-4f, q, t25_1);
  }
  }
  { const float p = x2 * x2;
  t3_0 = fmaf(0x1.279a740000000p-2f, p, t3_0);
  { const float q = p * x2;
  t20_2 = fmaf(0x1.c9f25c0000000p-3f, q, t20_2);
  }
  { const float q = p * x3;
  t20_3 = fmaf(0x1.c9f25c0000000p-3f, q, t20_3);
  }
  { const float q = p * x6;
  t13_0 = fmaf(0x1.c9f25c0000000p-3f, q, t13_0);
  }
  { const float q = p * x11;
  t21_1 = fmaf(0x1.5a2cd80000000p-3f, q, t21_1);
  }
  { const float q = p * x12;
  t21_2 = fmaf(0x1.a7f9ee0000000p-3f, q, t21_2);
  }
  { const float q = p * x13;
  t21_3 = fmaf(0x1.5a2cd80000000p-3f, q, t21_3);
  }
  }
  { const float p = x2 * x3;
  { const float q = p * x3;
  t20_2 = fmaf(0x1.c9f25c0000000p-3f, q, t20_2);
  }
  { const float q = p * x7;
  t13_0 = fmaf(0x1.8c97f00000000p-2f, q, t13_0);
  }
  { const float q = p * x10;
  t21_1 = fmaf(0x1.11acee0000000p-2f, q, t21_1);
  }
  { const float q = p * x12;
  t21_3 = fmaf(-0x1.a7f9ec0000000p-3f, q, t21_3);
  }
  { const float q = p * x13;
  t21_2 = fmaf(0x1.5a2cd80000000p-2f, q, t21_2);
  }
  { const float q = p * x14;
  t21_3 = fmaf(0x1.11acee0000000p-2f, q, t21_3);
  }
  }
  { const float p = x2 * x4;
  { const float q = p * x4;
  t22_2 = 0x1.c9f25c0000000p-3f * q;
  t23_2 = fmaf(0x1.e5b9d20000000p-3f, q, t23_2);
  }
  { const float q = p * x5;
  t23_3 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_3);
  }
  { const float q = p * x7;
  t23_1 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_1);
  }
  { const float q = p * x10;
  t14_0 = fmaf(0x1.83091e0000000p-3f, q, t14_0);
  }
  }
  { const float p = x2 * x5;
  t7_1 = fmaf(0x1.c9f25c0000000p-2f, p, t7_1);
  { const float q = p * x5;
  t22_2 = fmaf(0x1.c9f25c0000000p-3f, q, t22_2);
  }
  { const float q = p * x6;
  t23_1 = fmaf(-0x1.186f180000000p-3f, q, t23_1);
  }
  { const float q = p * x8;
  t23_1 = fmaf(0x1.e5b9d20000000p-3f, q, t23_1);
  }
  { const float q = p * x11;
  t14_0 = fmaf(0x1.e990ce0000000p-3f, q, t14_0);
  }
  }
  { const float p = x2 * x6;
  t7_2 = fmaf(0x1.08654a0000000p-1f, p, t7_2);
  { const float q = p * x6;
  t22_2 = fmaf(0x1.c9f25c0000000p-3f, q, t22_2);
  t23_2 = fmaf(-0x1.43d1360000000p-4f, q, t23_2);
  }
  { const float q = p * x7;
  t23_3 = fmaf(-0x1.186f180000000p-3f, q, t23_3);
  }
  { const float q = p * x12;
  t14_0 = fmaf(0x1.03a1a20000000p-2f, q, t14_0);
  }
  }
  { const float p = x2 * x7;
  t7_3 = fmaf(0x1.c9f25c0000000p-2f, p, t7_3);
  { const float q = p * x7;
  t22_2 = fmaf(0x1.c9f25c0000000p-3f, q, t22_2);
  }
  { const float q = p * x8;
  t23_3 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_3);
  }
  { const float q = p * x13;
  t14_0 = fmaf(0x1.e990ce0000000p-3f, q, t14_0);
  }
  }
  { const float p = x2 * x8;
  { const float q = p * x8;
  t22_2 = fmaf(0x1.c9f25c0000000p-3f, q, t22_2);
  t23_2 = fmaf(0x1.e5b9d20000000p-3f, q, t23_2);
  }
  { const float q = p * x14;
  t14_0 = fmaf(0x1.83091e0000000p-3f, q, t14_0);
  }
  }
  { const float p = x2 * x9;
  { const float q = p * x9;
  t24_2 = 0x1.83091e0000000p-3f * q;
  t25_2 = fmaf(-0x1.6150460000000p-3f, q, t25_2);
  }
  { const float q = p * x10;
  t25_3 = fmaf(0x1.b0b8100000000p-3f, q, t25_3);
  }
  { const float q = p * x14;
  t25_1 = fmaf(0x1.b0b8100000000p-3f, q, t25_1);
  }
  }
  { const float p = x2 * x10;
  { const float q = p * x10;
  t24_2 = fmaf(0x1.83091e0000000p-3f, q, t24_2);
  }
  { const float q = p * x11;
  t25_3 = fmaf(0x1.4f2ec40000000p-3f, q, t25_3);
  }
  { const float q = p * x13;
  t25_1 = fmaf(0x1.4f2ec40000000p-3f, q, t25_1);
  }
  { const float q = p * x15;
  t25_1 = fmaf(-0x1.b0b8100000000p-3f, q, t25_1);
  }
  }
  { const float p = x2 * x11;
  { const float q = p * x11;
  t24_2 = fmaf(0x1.83091e0000000p-3f, q, t24_2);
  t25_2 = fmaf(0x1.a7f9ee0000000p-4f, q, t25_2);
  }
  { const float q = p * x12;
  t25_1 = fmaf(0x1.5a2cd80000000p-4f, q, t25_1);
  }
  { const float q = p * x14;
  t25_1 = fmaf(-0x1.4f2ec40000000p-3f, q, t25_1);
  }
  }
  { const float p = x2 * x12;
  { const float q = p * x12;
  t24_2 = fmaf(0x1.83091e0000000p-3f, q, t24_2);
  t25_2 = fmaf(0x1.1aa69e0000000p-3f, q, t25_2);
  }
  { const float q = p * x13;
  t25_3 = fmaf(0x1.5a2cd80000000p-4f, q, t25_3);
  }
  }
  { const float p = x2 * x13;
  { const float q = p * x13;
  t24_2 = fmaf(0x1.83091e0000000p-3f, q, t24_2);
  t25_2 = fmaf(0x1.a7f9ee0000000p-4f, q, t25_2);
  }
  { const float q = p * x14;
  t25_3 = fmaf(0x1.4f2ec40000000p-3f, q, t25_3);
  }
  }
  { const float p = x2 * x14;
  { const float q = p * x14;
  t24_2 = fmaf(0x1.83091e0000000p-3f, q, t24_2);
  }
  { const float q = p * x15;
  t25_3 = fmaf(0x1.b0b8100000000p-3f, q, t25_3);
  }
  }
  { const float p = x2 * x15;
  { const float q = p * x15;
  t24_2 = fmaf(0x1.83091e0000000p-3f, q, t24_2);
  t25_2 = fmaf(-0x1.6150460000000p-3f, q, t25_2);
  }
  }
  { const float p = x3 * x3;
  t3_0 = fmaf(0x1.279a740000000p-2f, p, t3_0);
  { const float q = p * x3;
  t20_3 = fmaf(0x1.c9f25c0000000p-3f, q, t20_3);
  }
  { const float q = p * x6;
  t13_0 = fmaf(-0x1.c9f25c0000000p-4f, q, t13_0);
  }
  { const float q = p * x8;
  t13_0 = fmaf(0x1.8c97f00000000p-3f, q, t13_0);
  }
  { const float q = p * x9;
  t21_1 = fmaf(0x1.4f2ec40000000p-3f, q, t21_1);
  }
  { const float q = p * x11;
  t21_1 = fmaf(-0x1.5a2cda0000000p-5f, q, t21_1);
  }
  { const float q = p * x12;
  t21_2 = fmaf(-0x1.a7f9ee0000000p-4f, q, t21_2);
  }
  { const float q = p * x13;
  t21_3 = fmaf(-0x1.03a1a20000000p-3f, q, t21_3);
  }
  { const float q = p * x14;
  t21_2 = fmaf(0x1.11acee0000000p-3f, q, t21_2);
  }
  { const float q = p * x15;
  t21_3 = fmaf(0x1.4f2ec40000000p-3f, q, t21_3);
  }
  }
  { const float p = x3 * x4;
  t7_1 = fmaf(0x1.c9f25c0000000p-2f, p, t7_1);
  { const float q = p * x4;
  t22_3 = 0x1.c9f25c0000000p-3f * q;
  }
  { const float q = p * x5;
  t23_2 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_2);
  }
  { const float q = p * x6;
  t23_1 = fmaf(0x1.186f180000000p-2f, q, t23_1);
  }
  { const float q = p * x9;
  t14_0 = fmaf(0x1.da05160000000p-3f, q, t14_0);
  }
  { const float q = p * x11;
  t14_0 = fmaf(-0x1.e990ce0000000p-5f, q, t14_0);
  }
  }
  { const float p = x3 * x5;
  { const float q = p * x5;
  t22_3 = fmaf(0x1.c9f25c0000000p-3f, q, t22_3);
  t23_3 = fmaf(0x1.e5b9d20000000p-3f, q, t23_3);
  }
  { const float q = p * x7;
  t23_1 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_1);
  }
  { const float q = p * x10;
  t14_0 = fmaf(0x1.83091e0000000p-3f, q, t14_0);
  }
  }
  { const float p = x3 * x6;
  t7_3 = fmaf(-0x1.08654a0000000p-2f, p, t7_3);
  { const float q = p * x6;
  t22_3 = fmaf(0x1.c9f25c0000000p-3f, q, t22_3);
  t23_3 = fmaf(0x1.43d1360000000p-3f, q, t23_3);
  }
  { const float q = p * x7;
  t23_2 = fmaf(-0x1.186f180000000p-3f, q, t23_2);
  }
  { const float q = p * x8;
  t23_3 = fmaf(0x1.186f180000000p-2f, q, t23_3);
  }
  { const float q = p * x13;
  t14_0 = fmaf(0x1.a7f9ee0000000p-3f, q, t14_0);
  }
  }
  { const float p = x3 * x7;
  t7_2 = fmaf(0x1.c9f25c0000000p-2f, p, t7_2);
  { const float q = p * x7;
  t22_3 = fmaf(0x1.c9f25c0000000p-3f, q, t22_3);
  }
  { const float q = p * x8;
  t23_2 = fmaf(-0x1.e5b9d20000000p-3f, q, t23_2);
  }
  { const float q = p * x12;
  t14_0 = fmaf(-0x1.2bcbe60000000p-3f, q, t14_0);
  }
  { const float q = p * x14;
  t14_0 = fmaf(0x1.83091e0000000p-3f, q, t14_0);
  }
  }
  { const float p = x3 * x8;
  t7_3 = fmaf(0x1.c9f25c0000000p-2f, p, t7_3);
  { const float q = p * x8;
  t22_3 = fmaf(0x1.c9f25c0000000p-3f, q, t22_3);
  }
  { const float q = p * x13;
  t14_0 = fmaf(-0x1.e990ce0000000p-5f, q, t14_0);
  }
  { const float q = p * x15;
  t14_0 = fmaf(0x1.da05160000000p-3f, q, t14_0);
  }
  }
  { const float p = x3 * x9;
  { const float q = p * x9;
  t24_3 = 0x1.83091e0000000p-3f * q;
  t25_3 = fmaf(0x1.6150460000000p-4f, q, t25_3);
  }
  { const float q = p * x10;
  t25_2 = fmaf(0x1.b0b8100000000p-3f, q, t25_2);
  }
  { const float q = p * x11;
  t25_3 = fmaf(-0x1.11acee0000000p-3f, q, t25_3);
  }
  { const float q = p * x13;
  t25_1 = fmaf(-0x1.11acee0000000p-3f, q, t25_1);
  }
  }
  { const float p = x3 * x10;
  { const float q = p * x10;
  t24_3 = fmaf(0x1.83091e0000000p-3f, q, t24_3);
  }
  { const float q = p * x11;
  t25_2 = fmaf(0x1.4f2ec40000000p-3f, q, t25_2);
  }
  { const float q = p * x12;
  t25_1 = fmaf(-0x1.11acee0000000p-2f, q, t25_1);
  }
  }
  { const float p = x3 * x11;
  { const float q = p * x11;
  t24_3 = fmaf(0x1.83091e0000000p-3f, q, t24_3);
  t25_3 = fmaf(-0x1.3dfb720000000p-3f, q, t25_3);
  }
  { const float q = p * x13;
  t25_1 = fmaf(0x1.a7f9ec0000000p-3f, q, t25_1);
  }
  { const float q = p * x15;
  t25_1 = fmaf(0x1.11acee0000000p-3f, q, t25_1);
  }
  }
  { const float p = x3 * x12;
  { const float q = p * x12;
  t24_3 = fmaf(0x1.83091e0000000p-3f, q, t24_3);
  t25_3 = fmaf(-0x1.1aa69e0000000p-4f, q, t25_3);
  }
  { const float q = p * x13;
  t25_2 = fmaf(0x1.5a2cd80000000p-4f, q, t25_2);
  }
  { const float q = p * x14;
  t25_3 = fmaf(-0x1.11acee0000000p-2f, q, t25_3);
  }
  }
  { const float p = x3 * x13;
  { const float q = p * x13;
  t24_3 = fmaf(0x1.83091e0000000p-3f, q, t24_3);
  t25_3 = fmaf(0x1.a7f9ec0000000p-5f, q, t25_3);
  }
  { const float q = p * x14;
  t25_2 = fmaf(0x1.4f2ec40000000p-3f, q, t25_2);
  }
  { const float q = p * x15;
  t25_3 = fmaf(-0x1.11acee0000000p-3f, q, t25_3);
  }
  }
  { const float p = x3 * x14;
  { const float q = p * x14;
  t24_3 = fmaf(0x1.83091e0000000p-3f, q, t24_3);
  }
  { const float q = p * x15;
  t25_2 = fmaf(0x1.b0b8100000000p-3f, q, t25_2);
  }
  }
  { const float p = x3 * x15;
  { const float q = p * x15;
  t24_3 = fmaf(0x1.83091e0000000p-3f, q, t24_3);
  t25_3 = fmaf(0x1.6150460000000p-4f, q, t25_3);
  }
  }
  { const float p = x4 * x4;
  t4_0 = 0x1.c9f25c0000000p-3f * p;
  { const float q = p * x6;
  t15_0 = -0x1.03a1a20000000p-2f * q;
  }
  { const float q = p * x9;
  t27_1 = 0x1.bee9060000000p-4f * q;
  }
  { const float q = p * x11;
  t26_1 = -0x1.c53fee0000000p-4f * q;
  t27_1 = fmaf(-0x1.5a2cda0000000p-4f, q, t27_1);
  }
  { const float q = p * x12;
  t26_2 = -0x1.158ec20000000p-3f * q;
  t27_2 = -0x1.1aa6a00000000p-4f * q;
  }
  { const float q = p * x13;
  t26_3 = -0x1.c53fee0000000p-4f * q;
  t27_3 = -0x1.5a2cda0000000p-4f * q;
  }
  { const float q = p * x15;
  t27_3 = fmaf(-0x1.bee9060000000p-4f, q, t27_3);
  }
  }
  { const float p = x4 * x5;
  { const float q = p * x7;
  t15_0 = fmaf(0x1.c1b1dc0000000p-2f, q, t15_0);
  }
  { const float q = p * x10;
  t26_1 = fmaf(0x1.66534c0000000p-3f, q, t26_1);
  t27_1 = fmaf(0x1.11acf00000000p-2f, q, t27_1);
  }
  { const float q = p * x12;
  t26_3 = fmaf(-0x1.158ec00000000p-3f, q, t26_3);
  t27_3 = fmaf(-0x1.1aa69e0000000p-3f, q, t27_3);
  }
  { const float q = p * x13;
  t26_2 = fmaf(0x1.c53fee0000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.5a2cda0000000p-4f, q, t27_2);
  }
  { const float q = p * x14;
  t26_3 = fmaf(0x1.66534c0000000p-3f, q, t26_3);
  }
  { const float q = p * x15;
  t27_2 = fmaf(-0x1.bee9080000000p-4f, q, t27_2);
  }
  }
  { const float p = x4 * x6;
  { const float q = p * x9;
  t26_3 = fmaf(-0x1.fabfa20000000p-3f, q, t26_3);
  t27_3 = fmaf(-0x1.8309200000000p-3f, q, t27_3);
  }
  { const float q = p * x10;
  t26_2 = fmaf(-0x1.9dc22c0000000p-3f, q, t26_2);
  }
  { const float q = p * x11;
  t26_3 = fmaf(0x1.05af100000000p-4f, q, t26_3);
  t27_3 = fmaf(0x1.2bcbe80000000p-3f, q, t27_3);
  }
  { const float q = p * x13;
  t26_1 = fmaf(0x1.05af100000000p-4f, q, t26_1);
  t27_1 = fmaf(0x1.2bcbe80000000p-3f, q, t27_1);
  }
  { const float q = p * x15;
  t26_1 = fmaf(0x1.fabfa20000000p-3f, q, t26_1);
  t27_1 = fmaf(0x1.8309200000000p-3f, q, t27_1);
  }
  }
  { const float p = x4 * x7;
  { const float q = p * x9;
  t27_2 = fmaf(0x1.bee9080000000p-4f, q, t27_2);
  }
  { const float q = p * x10;
  t26_3 = fmaf(0x1.66534c0000000p-3f, q, t26_3);
  t27_3 = fmaf(0x1.11acf00000000p-2f, q, t27_3);
  }
  { const float q = p * x11;
  t26_2 = fmaf(0x1.c53fee0000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.5a2cda0000000p-4f, q, t27_2);
  }
  { const float q = p * x12;
  t26_1 = fmaf(-0x1.158ec00000000p-3f, q, t26_1);
  t27_1 = fmaf(-0x1.1aa69e0000000p-3f, q, t27_1);
  }
  { const float q = p * x14;
  t26_1 = fmaf(-0x1.66534c0000000p-3f, q, t26_1);
  }
  }
  { const float p = x4 * x8;
  { const float q = p * x9;
  t27_3 = fmaf(0x1.bee9060000000p-3f, q, t27_3);
  }
  { const float q = p * x15;
  t27_1 = fmaf(0x1.bee9060000000p-3f, q, t27_1);
  }
  }
  { const float p = x4 * x9;
  t8_3 = 0x1.83091e0000000p-2f * p;
  { const float q = p * x13;
  t16_0 = -0x1.11acee0000000p-3f * q;
  }
  }
  { const float p = x4 * x10;
  t8_2 = 0x1.3c03660000000p-2f * p;
  { const float q = p * x12;
  t16_0 = fmaf(-0x1.11acee0000000p-2f, q, t16_0);
  }
  }
  { const float p = x4 * x11;
  t8_3 = fmaf(-0x1.8fba8a0000000p-4f, p, t8_3);
  { const float q = p * x13;
  t16_0 = fmaf(0x1.a7f9ec0000000p-3f, q, t16_0);
  }
  { const float q = p * x15;
  t16_0 = fmaf(0x1.11acee0000000p-3f, q, t16_0);
  }
  }
  { const float p = x4 * x13;
  t8_1 = -0x1.8fba8a0000000p-4f * p;
  }
  { const float p = x4 * x15;
  t8_1 = fmaf(-0x1.83091e0000000p-2f, p, t8_1);
  }
  { const float p = x5 * x5;
  t4_0 = fmaf(0x1.c9f25c0000000p-3f, p, t4_0);
  { const float q = p * x6;
  t15_0 = fmaf(0x1.03a1a20000000p-3f, q, t15_0);
  }
  { const float q = p * x8;
  t15_0 = fmaf(-0x1.c1b1dc0000000p-3f, q, t15_0);
  }
  { const float q = p * x9;
  t26_1 = fmaf(-0x1.b6db6e0000000p-4f, q, t26_1);
  t27_1 = fmaf(-0x1.bee9060000000p-5f, q, t27_1);
  }
  { const float q = p * x11;
  t26_1 = fmaf(0x1.c53fee0000000p-6f, q, t26_1);
  t27_1 = fmaf(0x1.03a1a40000000p-3f, q, t27_1);
  }
  { const float q = p * x12;
  t26_2 = fmaf(0x1.158ec20000000p-4f, q, t26_2);
  t27_2 = fmaf(-0x1.1aa69e0000000p-5f, q, t27_2);
  }
  { const float q = p * x13;
  t26_3 = fmaf(0x1.53eff20000000p-4f, q, t26_3);
  t27_3 = fmaf(0x1.5a2cdc0000000p-5f, q, t27_3);
  }
  { const float q = p * x14;
  t26_2 = fmaf(-0x1.66534c0000000p-4f, q, t26_2);
  t27_2 = fmaf(-0x1.11acf00000000p-3f, q, t27_2);
  }
  { const float q = p * x15;
  t26_3 = fmaf(-0x1.b6db6e0000000p-4f, q, t26_3);
  t27_3 = fmaf(-0x1.bee9060000000p-5f, q, t27_3);
  }
  }
  { const float p = x5 * x6;
  { const float q = p * x10;
  t26_3 = fmaf(0x1.9dc22c0000000p-4f, q, t26_3);
  }
  { const float q = p * x11;
  t26_2 = fmaf(0x1.05af100000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.2bcbe80000000p-2f, q, t27_2);
  }
  { const float q = p * x12;
  t26_1 = fmaf(-0x1.407ef60000000p-4f, q, t26_1);
  t27_1 = fmaf(0x1.e990d00000000p-4f, q, t27_1);
  }
  { const float q = p * x14;
  t26_1 = fmaf(-0x1.9dc22c0000000p-4f, q, t26_1);
  }
  }
  { const float p = x5 * x7;
  { const float q = p * x9;
  t26_3 = fmaf(0x1.b6db6e0000000p-3f, q, t26_3);
  t27_3 = fmaf(0x1.bee9060000000p-4f, q, t27_3);
  }
  { const float q = p * x10;
  t26_2 = fmaf(0x1.66534c0000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.11acf00000000p-2f, q, t27_2);
  }
  { const float q = p * x11;
  t26_3 = fmaf(-0x1.c53fee0000000p-5f, q, t26_3);
  t27_3 = fmaf(0x1.5a2cda0000000p-4f, q, t27_3);
  }
  { const float q = p * x13;
  t26_1 = fmaf(-0x1.c53fee0000000p-5f, q, t26_1);
  t27_1 = fmaf(0x1.5a2cda0000000p-4f, q, t27_1);
  }
  { const float q = p * x15;
  t26_1 = fmaf(-0x1.b6db6e0000000p-3f, q, t26_1);
  t27_1 = fmaf(-0x1.bee9060000000p-4f, q, t27_1);
  }
  }
  { const float p = x5 * x8;
  { const float q = p * x9;
  t27_2 = fmaf(0x1.bee9080000000p-4f, q, t27_2);
  }
  { const float q = p * x10;
  t26_3 = fmaf(-0x1.66534c0000000p-3f, q, t26_3);
  }
  { const float q = p * x11;
  t26_2 = fmaf(-0x1.c53fee0000000p-3f, q, t26_2);
  t27_2 = fmaf(-0x1.5a2cda0000000p-4f, q, t27_2);
  }
  { const float q = p * x12;
  t26_1 = fmaf(0x1.158ec00000000p-3f, q, t26_1);
  t27_1 = fmaf(0x1.1aa69e0000000p-3f, q, t27_1);
  }
  { const float q = p * x14;
  t26_1 = fmaf(0x1.66534c0000000p-3f, q, t26_1);
  t27_1 = fmaf(0x1.11acf00000000p-2f, q, t27_1);
  }
  }
  { const float p = x5 * x9;
  { const float q = p * x14;
  t16_0 = fmaf(0x1.b0b8100000000p-3f, q, t16_0);
  }
  }
  { const float p = x5 * x10;
  t8_3 = fmaf(0x1.3c03660000000p-2f, p, t8_3);
  { const float q = p * x13;
  t16_0 = fmaf(0x1.4f2ec40000000p-3f, q, t16_0);
  }
  { const float q = p * x15;
  t16_0 = fmaf(-0x1.b0b8100000000p-3f, q, t16_0);
  }
  }
  { const float p = x5 * x11;
  t8_2 = fmaf(0x1.8fba8a0000000p-2f, p, t8_2);
  { const float q = p * x12;
  t16_0 = fmaf(0x1.5a2cd80000000p-4f, q, t16_0);
  }
  { const float q = p * x14;
  t16_0 = fmaf(-0x1.4f2ec40000000p-3f, q, t16_0);
  }
  }
  { const float p = x5 * x12;
  t8_1 = fmaf(-0x1.e990cc0000000p-3f, p, t8_1);
  }
  { const float p = x5 * x14;
  t8_1 = fmaf(-0x1.3c03660000000p-2f, p, t8_1);
  }
  { const float p = x6 * x6;
  t4_0 = fmaf(0x1.c9f25c0000000p-3f, p, t4_0);
  { const float q = p * x6;
  t15_0 = fmaf(0x1.5a2cda0000000p-4f, q, t15_0);
  }
  { const float q = p * x11;
  t26_1 = fmaf(0x1.c53fee0000000p-4f, q, t26_1);
  }
  { const float q = p * x12;
  t26_2 = fmaf(0x1.158ec20000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.a7f9ee0000000p-3f, q, t27_2);
  }
  { const float q = p * x13;
  t26_3 = fmaf(0x1.c53fee0000000p-4f, q, t26_3);
  }
  }
  { const float p = x6 * x7;
  { const float q = p * x7;
  t15_0 = fmaf(0x1.03a1a20000000p-3f, q, t15_0);
  }
  { const float q = p * x10;
  t26_1 = fmaf(0x1.9dc22c0000000p-4f, q, t26_1);
  }
  { const float q = p * x12;
  t26_3 = fmaf(-0x1.407ef60000000p-4f, q, t26_3);
  t27_3 = fmaf(0x1.e990d00000000p-4f, q, t27_3);
  }
  { const float q = p * x13;
  t26_2 = fmaf(0x1.05af100000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.2bcbe80000000p-2f, q, t27_2);
  }
  { const float q = p * x14;
  t26_3 = fmaf(0x1.9dc22c0000000p-4f, q, t26_3);
  }
  }
  { const float p = x6 * x8;
  { const float q = p * x8;
  t15_0 = fmaf(-0x1.03a1a20000000p-2f, q, t15_0);
  }
  { const float q = p * x9;
  t26_1 = fmaf(-0x1.fabfa20000000p-3f, q, t26_1);
  t27_1 = fmaf(-0x1.8309200000000p-3f, q, t27_1);
  }
  { const float q = p * x11;
  t26_1 = fmaf(-0x1.05af100000000p-4f, q, t26_1);
  t27_1 = fmaf(-0x1.2bcbe80000000p-3f, q, t27_1);
  }
  { const float q = p * x13;
  t26_3 = fmaf(0x1.05af100000000p-4f, q, t26_3);
  t27_3 = fmaf(0x1.2bcbe80000000p-3f, q, t27_3);
  }
  { const float q = p * x14;
  t26_2 = fmaf(-0x1.9dc22c0000000p-3f, q, t26_2);
  }
  { const float q = p * x15;
  t26_3 = fmaf(-0x1.fabfa20000000p-3f, q, t26_3);
  t27_3 = fmaf(-0x1.8309200000000p-3f, q, t27_3);
  }
  }
  { const float p = x6 * x9;
  { const float q = p * x9;
  t16_0 = fmaf(-0x1.31fa800000000p-3f, q, t16_0);
  }
  }
  { const float p = x6 * x11;
  t8_1 = fmaf(0x1.5a2cda0000000p-2f, p, t8_1);
  { const float q = p * x11;
  t16_0 = fmaf(0x1.6f2c9a0000000p-4f, q, t16_0);
  }
  }
  { const float p = x6 * x12;
  t8_2 = fmaf(0x1.a7f9ee0000000p-2f, p, t8_2);
  { const float q = p * x12;
  t16_0 = fmaf(0x1.e990ce0000000p-4f, q, t16_0);
  }
  }
  { const float p = x6 * x13;
  t8_3 = fmaf(0x1.5a2cda0000000p-2f, p, t8_3);
  { const float q = p * x13;
  t16_0 = fmaf(0x1.6f2c9a0000000p-4f, q, t16_0);
  }
  }
  { const float p = x6 * x15;
  { const float q = p * x15;
  t16_0 = fmaf(-0x1.31fa800000000p-3f, q, t16_0);
  }
  }
  { const float p = x7 * x7;
  t4_0 = fmaf(0x1.c9f25c0000000p-3f, p, t4_0);
  { const float q = p * x8;
  t15_0 = fmaf(0x1.c1b1dc0000000p-3f, q, t15_0);
  }
  { const float q = p * x9;
  t26_1 = fmaf(0x1.b6db6e0000000p-4f, q, t26_1);
  t27_1 = fmaf(0x1.bee9060000000p-5f, q, t27_1);
  }
  { const float q = p * x11;
  t26_1 = fmaf(0x1.53eff20000000p-4f, q, t26_1);
  t27_1 = fmaf(0x1.5a2cdc0000000p-5f, q, t27_1);
  }
  { const float q = p * x12;
  t26_2 = fmaf(0x1.158ec20000000p-4f, q, t26_2);
  t27_2 = fmaf(-0x1.1aa69e0000000p-5f, q, t27_2);
  }
  { const float q = p * x13;
  t26_3 = fmaf(0x1.c53fee0000000p-6f, q, t26_3);
  t27_3 = fmaf(0x1.03a1a40000000p-3f, q, t27_3);
  }
  { const float q = p * x14;
  t26_2 = fmaf(0x1.66534c0000000p-4f, q, t26_2);
  t27_2 = fmaf(0x1.11acf00000000p-3f, q, t27_2);
  }
  { const float q = p * x15;
  t26_3 = fmaf(0x1.b6db6e0000000p-4f, q, t26_3);
  t27_3 = fmaf(0x1.bee9060000000p-5f, q, t27_3);
  }
  }
  { const float p = x7 * x8;
  { const float q = p * x10;
  t26_1 = fmaf(0x1.66534c0000000p-3f, q, t26_1);
  }
  { const float q = p * x12;
  t26_3 = fmaf(-0x1.158ec00000000p-3f, q, t26_3);
  t27_3 = fmaf(-0x1.1aa69e0000000p-3f, q, t27_3);
  }
  { const float q = p * x13;
  t26_2 = fmaf(0x1.c53fee0000000p-3f, q, t26_2);
  t27_2 = fmaf(0x1.5a2cda0000000p-4f, q, t27_2);
  }
  { const float q = p * x14;
  t26_3 = fmaf(0x1.66534c0000000p-3f, q, t26_3);
  t27_3 = fmaf(0x1.11acf00000000p-2f, q, t27_3);
  }
  { const float q = p * x15;
  t27_2 = fmaf(0x1.bee9080000000p-4f, q, t27_2);
  }
  }
  { const float p = x7 * x9;
  { const float q = p * x10;
  t16_0 = fmaf(0x1.b0b8100000000p-3f, q, t16_0);
  }
  }
  { const float p = x7 * x10;
  t8_1 = fmaf(0x1.3c03660000000p-2f, p, t8_1);
  { const float q = p * x11;
  t16_0 = fmaf(0x1.4f2ec40000000p-3f, q, t16_0);
  }
  }
  { const float p = x7 * x12;
  t8_3 = fmaf(-0x1.e990cc0000000p-3f, p, t8_3);
  { const float q = p * x13;
  t16_0 = fmaf(0x1.5a2cd80000000p-4f, q, t16_0);
  }
  }
  { const float p = x7 * x13;
  t8_2 = fmaf(0x1.8fba8a0000000p-2f, p, t8_2);
  { const float q = p * x14;
  t16_0 = fmaf(0x1.4f2ec40000000p-3f, q, t16_0);
  }
  }
  { const float p = x7 * x14;
  t8_3 = fmaf(0x1.3c03660000000p-2f, p, t8_3);
  { const float q = p * x15;
  t16_0 = fmaf(0x1.b0b8100000000p-3f, q, t16_0);
  }
  }
  { const float p = x8 * x8;
  t4_0 = fmaf(0x1.c9f25c0000000p-3f, p, t4_0);
  { const float q = p * x9;
  t27_1 = fmaf(-0x1.bee9060000000p-4f, q, t27_1);
  }
  { const float q = p * x11;
  t26_1 = fmaf(-0x1.c53fee0000000p-4f, q, t26_1);
  t27_1 = fmaf(-0x1.5a2cda0000000p-4f, q, t27_1);
  }
  { const float q = p * x12;
  t26_2 = fmaf(-0x1.158ec20000000p-3f, q, t26_2);
  t27_2 = fmaf(-0x1.1aa6a00000000p-4f, q, t27_2);
  }
  { const float q = p * x13;
  t26_3 = fmaf(-0x1.c53fee0000000p-4f, q, t26_3);
  t27_3 = fmaf(-0x1.5a2cda0000000p-4f, q, t27_3);
  }
  { const float q = p * x15;
  t27_3 = fmaf(0x1.bee9060000000p-4f, q, t27_3);
  }
  }
  { const float p = x8 * x9;
  t8_1 = fmaf(0x1.83091e0000000p-2f, p, t8_1);
  { const float q = p * x11;
  t16_0 = fmaf(-0x1.11acee0000000p-3f, q, t16_0);
  }
  }
  { const float p = x8 * x11;
  t8_1 = fmaf(0x1.8fba8a0000000p-4f, p, t8_1);
  { const float q = p * x11;
  t16_0 = fmaf(-0x1.a7f9ec0000000p-4f, q, t16_0);
  }
  }
  { const float p = x8 * x12;
  { const float q = p * x14;
  t16_0 = fmaf(-0x1.11acee0000000p-2f, q, t16_0);
  }
  }
  { const float p = x8 * x13;
  t8_3 = fmaf(-0x1.8fba8a0000000p-4f, p, t8_3);
  { const float q = p * x13;
  t16_0 = fmaf(0x1.a7f9ec0000000p-4f, q, t16_0);
  }
  { const float q = p * x15;
  t16_0 = fmaf(-0x1.11acee0000000p-3f, q, t16_0);
  }
  }
  { const float p = x8 * x14;
  t8_2 = fmaf(0x1.3c03660000000p-2f, p, t8_2);
  }
  { const float p = x8 * x15;
  t8_3 = fmaf(0x1.83091e0000000p-2f, p, t8_3);
  }
  { const float p = x9 * x9;
  t5_0 = 0x1.83091e0000000p-3f * p;
  { const float q = p * x11;
  t28_1 = -0x1.7106240000000p-3f * q;
  }
  { const float q = p * x12;
  t28_2 = -0x1.c3f5d20000000p-4f * q;
  }
  { const float q = p * x13;
  t28_3 = -0x1.7106240000000p-3f * q;
  }
  }
  { const float p = x9 * x10;
  { const float q = p * x10;
  t28_1 = fmaf(0x1.dc686e0000000p-4f, q, t28_1);
  }
  { const float q = p * x12;
  t28_3 = fmaf(-0x1.14c49c0000000p-2f, q, t28_3);
  }
  { const float q = p * x13;
  t28_2 = fmaf(0x1.2d4e8e0000000p-4f, q, t28_2);
  }
  { const float q = p * x14;
  t28_3 = fmaf(0x1.dc686e0000000p-3f, q, t28_3);
  }
  }
  { const float p = x9 * x11;
  { const float q = p * x11;
  t28_1 = fmaf(-0x1.7d20580000000p-4f, q, t28_1);
  }
  { const float q = p * x13;
  t28_3 = fmaf(0x1.7d20580000000p-3f, q, t28_3);
  }
  { const float q = p * x14;
  t28_2 = fmaf(0x1.2d4e8e0000000p-4f, q, t28_2);
  }
  }
  { const float p = x9 * x12;
  { const float q = p * x14;
  t28_1 = fmaf(-0x1.14c49c0000000p-2f, q, t28_1);
  }
  }
  { const float p = x9 * x13;
  { const float q = p * x13;
  t28_1 = fmaf(0x1.7d20580000000p-4f, q, t28_1);
  }
  }
  { const float p = x9 * x14;
  { const float q = p * x14;
  t28_1 = fmaf(-0x1.dc686e0000000p-4f, q, t28_1);
  }
  }
  { const float p = x10 * x10;
  t5_0 = fmaf(0x1.83091e0000000p-3f, p, t5_0);
  { const float q = p * x11;
  t28_1 = fmaf(0x1.7106240000000p-4f, q, t28_1);
  }
  { const float q = p * x12;
  t28_2 = fmaf(-0x1.2d4e8c0000000p-3f, q, t28_2);
  }
  { const float q = p * x13;
  t28_3 = fmaf(0x1.7106240000000p-4f, q, t28_3);
  }
  { const float q = p * x15;
  t28_3 = fmaf(-0x1.dc686e0000000p-4f, q, t28_3);
  }
  }
  { const float p = x10 * x11;
  { const float q = p * x12;
  t28_3 = fmaf(0x1.7d205c0000000p-6f, q, t28_3);
  }
  { const float q = p * x13;
  t28_2 = fmaf(0x1.5e163e0000000p-2f, q, t28_2);
  }
  { const float q = p * x15;
  t28_2 = fmaf(-0x1.2d4e8e0000000p-4f, q, t28_2);
  }
  }
  { const float p = x10 * x12;
  { const float q = p * x13;
  t28_1 = fmaf(0x1.7d205c0000000p-6f, q, t28_1);
  }
  { const float q = p * x15;
  t28_1 = fmaf(0x1.14c49c0000000p-2f, q, t28_1);
  }
  }
  { const float p = x10 * x14;
  { const float q = p * x15;
  t28_1 = fmaf(0x1.dc686e0000000p-3f, q, t28_1);
  }
  }
  { const float p = x11 * x11;
  t5_0 = fmaf(0x1.83091e0000000p-3f, p, t5_0);
  { const float q = p * x11;
  t28_1 = fmaf(0x1.27381e0000000p-5f, q, t28_1);
  }
  { const float q = p * x12;
  t28_2 = fmaf(0x1.001c5e0000000p-3f, q, t28_2);
  }
  { const float q = p * x13;
  t28_3 = fmaf(0x1.27381e0000000p-5f, q, t28_3);
  }
  { const float q = p * x14;
  t28_2 = fmaf(-0x1.5e163e0000000p-3f, q, t28_2);
  }
  { const float q = p * x15;
  t28_3 = fmaf(-0x1.7d20580000000p-4f, q, t28_3);
  }
  }
  { const float p = x11 * x12;
  { const float q = p * x12;
  t28_1 = fmaf(0x1.27381c0000000p-5f, q, t28_1);
  }
  { const float q = p * x14;
  t28_1 = fmaf(-0x1.7d205c0000000p-6f, q, t28_1);
  }
  }
  { const float p = x11 * x13;
  { const float q = p * x13;
  t28_1 = fmaf(0x1.27381e0000000p-5f, q, t28_1);
  }
  { const float q = p * x15;
  t28_1 = fmaf(-0x1.7d20580000000p-3f, q, t28_1);
  }
  }
  { const float p = x11 * x14;
  { const float q = p * x14;
  t28_1 = fmaf(0x1.7106240000000p-4f, q, t28_1);
  }
  }
  { const float p = x11 * x15;
  { const float q = p * x15;
  t28_1 = fmaf(-0x1.7106240000000p-3f, q, t28_1);
  }
  }
  { const float p = x12 * x12;
  t5_0 = fmaf(0x1.83091e0000000p-3f, p, t5_0);
  { const float q = p * x12;
  t28_2 = fmaf(0x1.6991760000000p-4f, q, t28_2);
  }
  { const float q = p * x13;
  t28_3 = fmaf(0x1.27381c0000000p-5f, q, t28_3);
  }
  }
  { const float p = x12 * x13;
  { const float q = p * x13;
  t28_2 = fmaf(0x1.001c5e0000000p-3f, q, t28_2);
  }
  { const float q = p * x14;
  t28_3 = fmaf(0x1.7d205c0000000p-6f, q, t28_3);
  }
  }
  { const float p = x12 * x14;
  { const float q = p * x14;
  t28_2 = fmaf(-0x1.2d4e8c0000000p-3f, q, t28_2);
  }
  { const float q = p * x15;
  t28_3 = fmaf(-0x1.14c49c0000000p-2f, q, t28_3);
  }
  }
  { const float p = x12 * x15;
  { const float q = p * x15;
  t28_2 = fmaf(-0x1.c3f5d20000000p-4f, q, t28_2);
  }
  }
  { const float p = x13 * x13;
  t5_0 = fmaf(0x1.83091e0000000p-3f, p, t5_0);
  { const float q = p * x13;
  t28_3 = fmaf(0x1.27381e0000000p-5f, q, t28_3);
  }
  { const float q = p * x14;
  t28_2 = fmaf(0x1.5e163e0000000p-3f, q, t28_2);
  }
  { const float q = p * x15;
  t28_3 = fmaf(0x1.7d20580000000p-4f, q, t28_3);
  }
  }
  { const float p = x13 * x14;
  { const float q = p * x14;
  t28_3 = fmaf(0x1.7106240000000p-4f, q, t28_3);
  }
  { const float q = p * x15;
  t28_2 = fmaf(0x1.2d4e8e0000000p-4f, q, t28_2);
  }
  }
  { const float p = x13 * x15;
  { const float q = p * x15;
  t28_3 = fmaf(-0x1.7106240000000p-3f, q, t28_3);
  }
  }
  { const float p = x14 * x14;
  t5_0 = fmaf(0x1.83091e0000000p-3f, p, t5_0);
  { const float q = p * x15;
  t28_3 = fmaf(0x1.dc686e0000000p-4f, q, t28_3);
  }
  }
  { const float p = x15 * x15;
  t5_0 = fmaf(0x1.83091e0000000p-3f, p, t5_0);
  }
  float y0 = 0.0f;
  float y1 = 0.0f;
  float y2 = 0.0f;
  float y3 = 0.0f;
  y0 = fmaf(__ldg(wr + 0), t0_0, y0);
  y1 = fmaf(__ldg(wr + 128), t1_1, y1);
  y2 = fmaf(__ldg(wr + 128), t1_2, y2);
  y3 = fmaf(__ldg(wr + 128), t1_3, y3);
  y0 = fmaf(__ldg(wr + 256), t2_0, y0);
  y0 = fmaf(__ldg(wr + 384), t3_0, y0);
  y0 = fmaf(__ldg(wr + 512), t4_0, y0);
  y0 = fmaf(__ldg(wr + 640), t5_0, y0);
  y1 = fmaf(__ldg(wr + 768), t6_1, y1);
  y2 = fmaf(__ldg(wr + 768), t6_2, y2);
  y3 = fmaf(__ldg(wr + 768), t6_3, y3);
  y1 = fmaf(__ldg(wr + 896), t7_1, y1);
  y2 = fmaf(__ldg(wr + 896), t7_2, y2);
  y3 = fmaf(__ldg(wr + 896), t7_3, y3);
  y1 = fmaf(__ldg(wr + 1024), t8_1, y1);
  y2 = fmaf(__ldg(wr + 1024), t8_2, y2);
  y3 = fmaf(__ldg(wr + 1024), t8_3, y3);
  y0 = fmaf(__ldg(wr + 1152), t9_0, y0);
  y0 = fmaf(__ldg(wr + 1280), t10_0, y0);
  y0 = fmaf(__ldg(wr + 1408), t11_0, y0);
  y0 = fmaf(__ldg(wr + 1536), t12_0, y0);
  y0 = fmaf(__ldg(wr + 1664), t13_0, y0);
  y0 = fmaf(__ldg(wr + 1792), t14_0, y0);
  y0 = fmaf(__ldg(wr + 1920), t15_0, y0);
  y0 = fmaf(__ldg(wr + 2048), t16_0, y0);
  y1 = fmaf(__ldg(wr + 2176), t17_1, y1);
  y2 = fmaf(__ldg(wr + 2176), t17_2, y2);
  y3 = fmaf(__ldg(wr + 2176), t17_3, y3);
  y1 = fmaf(__ldg(wr + 2304), t18_1, y1);
  y2 = fmaf(__ldg(wr + 2304), t18_2, y2);
  y3 = fmaf(__ldg(wr + 2304), t18_3, y3);
  y1 = fmaf(__ldg(wr + 2432), t19_1, y1);
  y2 = fmaf(__ldg(wr + 2432), t19_2, y2);
  y3 = fmaf(__ldg(wr + 2432), t19_3, y3);
  y1 = fmaf(__ldg(wr + 2560), t20_1, y1);
  y2 = fmaf(__ldg(wr + 2560), t20_2, y2);
  y3 = fmaf(__ldg(wr + 2560), t20_3, y3);
  y1 = fmaf(__ldg(wr + 2688), t21_1, y1);
  y2 = fmaf(__ldg(wr + 2688), t21_2, y2);
  y3 = fmaf(__ldg(wr + 2688), t21_3, y3);
  y1 = fmaf(__ldg(wr + 2816), t22_1, y1);
  y2 = fmaf(__ldg(wr + 2816), t22_2, y2);
  y3 = fmaf(__ldg(wr + 2816), t22_3, y3);
  y1 = fmaf(__ldg(wr + 2944), t23_1, y1);
  y2 = fmaf(__ldg(wr + 2944), t23_2, y2);
  y3 = fmaf(__ldg(wr + 2944), t23_3, y3);
  y1 = fmaf(__ldg(wr + 3072), t24_1, y1);
  y2 = fmaf(__ldg(wr + 3072), t24_2, y2);
  y3 = fmaf(__ldg(wr + 3072), t24_3, y3);
  y1 = fmaf(__ldg(wr + 3200), t25_1, y1);
  y2 = fmaf(__ldg(wr + 3200), t25_2, y2);
  y3 = fmaf(__ldg(wr + 3200), t25_3, y3);
  y1 = fmaf(__ldg(wr + 3328), t26_1, y1);
  y2 = fmaf(__ldg(wr + 3328), t26_2, y2);
  y3 = fmaf(__ldg(wr + 3328), t26_3, y3);
  y1 = fmaf(__ldg(wr + 3456), t27_1, y1);
  y2 = fmaf(__ldg(wr + 3456), t27_2, y2);
  y3 = fmaf(__ldg(wr + 3456), t27_3, y3);
  y1 = fmaf(__ldg(wr + 3584), t28_1, y1);
  y2 = fmaf(__ldg(wr + 3584), t28_2, y2);
  y3 = fmaf(__ldg(wr + 3584), t28_3, y3);
  y[(long long)a * 512 + 0 + ch] = y0;
  y[(long long)a * 512 + 128 + ch] = y1;
  y[(long long)a * 512 + 256 + ch] = y2;
  y[(long long)a * 512 + 384 + ch] = y3;
}
