// cuBLASDx FP32 block-GEMM falsifier: one 64x64x64 block problem per CTA,
// many CTAs, compared with cuBLAS strided-batched SGEMM on the same tiles.
#include <cublas_v2.h>
#include <cublasdx.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECKC(x) do { auto s = (x); if (s != cudaSuccess) { \
    printf("CUDA error %s line %d\n", cudaGetErrorString(s), __LINE__); return 1; } } while (0)
#define CHECKB(x) do { auto s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d line %d\n", (int)s, __LINE__); return 1; } } while (0)

using GEMM = decltype(cublasdx::Size<64, 64, 64>()
                      + cublasdx::Precision<float>()
                      + cublasdx::Type<cublasdx::type::real>()
                      + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>()
                      + cublasdx::Function<cublasdx::function::MM>()
                      + cublasdx::SM<1200>()
                      + cublasdx::Block());

__global__ void dx_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    float* a = reinterpret_cast<float*>(smem_a);
    float* b = reinterpret_cast<float*>(smem_b);
    float* c = reinterpret_cast<float*>(smem_c);
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        a[i] = A[i];
        b[i] = B[i];
    }
    __syncthreads();
    GEMM().execute(1.0f, a, b, 0.0f, c);
    __syncthreads();
    float* out = C + (size_t)blockIdx.x * 64 * 64;
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) out[i] = c[i];
}

static float time_kernel(void (*fn)(void*), void* arg, int reps) {
    return 0.0f;
}

int main(int argc, char** argv) {
    const int batches = argc > 1 ? atoi(argv[1]) : 2048;
    const int reps = argc > 2 ? atoi(argv[2]) : 50;
    std::vector<float> hA(64 * 64), hB(64 * 64);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = sinf((float)i * 0.01f);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = cosf((float)i * 0.013f);
    float *A, *B, *Cdx, *Cref;
    CHECKC(cudaMalloc(&A, hA.size() * 4));
    CHECKC(cudaMalloc(&B, hB.size() * 4));
    CHECKC(cudaMalloc(&Cdx, (size_t)batches * 64 * 64 * 4));
    CHECKC(cudaMalloc(&Cref, (size_t)batches * 64 * 64 * 4));
    CHECKC(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CHECKC(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    const int smem = (int)cublasdx::get_shared_storage_size<GEMM>();
    printf("cublasdx shared_memory_bytes=%d block_dim=%d batches=%d\n", smem, (int)GEMM::block_dim.x, batches);
    CHECKC(cudaFuncSetAttribute(dx_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    dx_kernel<<<batches, GEMM::block_dim, smem>>>(A, B, Cdx);
    CHECKC(cudaDeviceSynchronize());

    // cuBLAS strided batched reference: C_i = A @ B for every batch (row-major via col-major transpose identity).
    cublasHandle_t handle; CHECKB(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&]() {
        return cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &alpha,
                                         B, 64, 0, A, 64, 0, &beta, Cref, 64, 64 * 64, batches);
    };
    CHECKB(run_cublas());
    CHECKC(cudaDeviceSynchronize());
    std::vector<float> hdx(64 * 64), href(64 * 64);
    CHECKC(cudaMemcpy(hdx.data(), Cdx, hdx.size() * 4, cudaMemcpyDeviceToHost));
    CHECKC(cudaMemcpy(href.data(), Cref, href.size() * 4, cudaMemcpyDeviceToHost));
    double maxdiff = 0;
    for (size_t i = 0; i < hdx.size(); ++i) maxdiff = fmax(maxdiff, (double)fabsf(hdx[i] - href[i]));
    printf("maxdiff_vs_cublas=%.3g\n", maxdiff);

    for (int i = 0; i < 5; ++i) dx_kernel<<<batches, GEMM::block_dim, smem>>>(A, B, Cdx);
    CHECKC(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < reps; ++i) dx_kernel<<<batches, GEMM::block_dim, smem>>>(A, B, Cdx);
    cudaEventRecord(e); CHECKC(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    double dx_us = ms * 1000.0 / reps;
    double flops = 2.0 * 64 * 64 * 64 * batches;
    printf("cublasdx_us=%.2f tflops=%.3f\n", dx_us, flops / (dx_us * 1e-6) / 1e12);

    for (int i = 0; i < 5; ++i) run_cublas();
    CHECKC(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < reps; ++i) run_cublas();
    cudaEventRecord(e); CHECKC(cudaDeviceSynchronize());
    cudaEventElapsedTime(&ms, s, e);
    double cb_us = ms * 1000.0 / reps;
    printf("cublas_batched_us=%.2f tflops=%.3f ratio_dx_over_cublas=%.3f\n", cb_us,
           flops / (cb_us * 1e-6) / 1e12, dx_us / cb_us);
    return 0;
}
