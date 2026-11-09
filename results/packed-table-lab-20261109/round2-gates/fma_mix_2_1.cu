extern "C" __global__ void __launch_bounds__(256) fma_mix(float* out, double* dout, int iters, float a, float c, double ad, double cd) {
    const float base = threadIdx.x * 1e-6f;
    float f0 = base + 0.f;
    float f1 = base + 1.f;
    float f2 = base + 2.f;
    float f3 = base + 3.f;
    float f4 = base + 4.f;
    float f5 = base + 5.f;
    float f6 = base + 6.f;
    float f7 = base + 7.f;
    double d0 = (double)base + 0.0;
    double d1 = (double)base + 1.0;
    double d2 = (double)base + 2.0;
    double d3 = (double)base + 3.0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            f0 = fmaf(f0, a, c);
            f1 = fmaf(f1, a, c);
            f2 = fmaf(f2, a, c);
            f3 = fmaf(f3, a, c);
            f4 = fmaf(f4, a, c);
            f5 = fmaf(f5, a, c);
            f6 = fmaf(f6, a, c);
            f7 = fmaf(f7, a, c);
            d0 = fma(d0, ad, cd);
            d1 = fma(d1, ad, cd);
            d2 = fma(d2, ad, cd);
            d3 = fma(d3, ad, cd);
        }
    }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
    dout[t] = d0 + d1 + d2 + d3;
}