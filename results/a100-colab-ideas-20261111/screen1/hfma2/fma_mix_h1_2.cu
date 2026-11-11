extern "C" __global__ void __launch_bounds__(256) fma_mix(float* out, double* dout, int iters, float a, float c, double ad, double cd) {
    const float base = threadIdx.x * 1e-6f;
    float f0 = base + 0.f;
    float f1 = base + 1.f;
    float f2 = base + 2.f;
    float f3 = base + 3.f;
    const unsigned ah2 = 0x3BFE3BFEu, ch2 = 0x04000400u;
    unsigned h0 = 0x3C003C00u + threadIdx.x + 0u;
    unsigned h1 = 0x3C003C00u + threadIdx.x + 1u;
    unsigned h2 = 0x3C003C00u + threadIdx.x + 2u;
    unsigned h3 = 0x3C003C00u + threadIdx.x + 3u;
    unsigned h4 = 0x3C003C00u + threadIdx.x + 4u;
    unsigned h5 = 0x3C003C00u + threadIdx.x + 5u;
    unsigned h6 = 0x3C003C00u + threadIdx.x + 6u;
    unsigned h7 = 0x3C003C00u + threadIdx.x + 7u;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            f0 = fmaf(f0, a, c);
            f1 = fmaf(f1, a, c);
            f2 = fmaf(f2, a, c);
            f3 = fmaf(f3, a, c);
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h0) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h1) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h2) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h3) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h4) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h5) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h6) : "r"(ah2), "r"(ch2));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h7) : "r"(ah2), "r"(ch2));
        }
    }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = f0 + f1 + f2 + f3 + (float)(h0 & 0xffffu) + (float)(h1 & 0xffffu) + (float)(h2 & 0xffffu) + (float)(h3 & 0xffffu) + (float)(h4 & 0xffffu) + (float)(h5 & 0xffffu) + (float)(h6 & 0xffffu) + (float)(h7 & 0xffffu);
    dout[t] = 0.0;
}