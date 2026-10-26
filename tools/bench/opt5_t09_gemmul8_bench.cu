#include "gemmul8.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECKC(x) do { auto s = (x); if (s != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(s), __LINE__); return 1; } } while (0)
#define CHECKB(x) do { auto s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d at line %d\n", (int)s, __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 13608;
    const int N = argc > 2 ? atoi(argv[2]) : 512;
    const int K = argc > 3 ? atoi(argv[3]) : 512;
    const int reps = argc > 4 ? atoi(argv[4]) : 15;
    printf("GEMMul8 Ozaki II test: shape MxNxK = %dx%dx%d\n", M, N, K);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = sinf((float)i * 0.001f) * 0.5f;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = cosf((float)i * 0.0013f) * 0.5f;
    float *A, *B, *C, *Cref;
    CHECKC(cudaMalloc(&A, hA.size() * 4)); CHECKC(cudaMalloc(&B, hB.size() * 4));
    CHECKC(cudaMalloc(&C, (size_t)M * N * 4)); CHECKC(cudaMalloc(&Cref, (size_t)M * N * 4));
    CHECKC(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CHECKC(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    cublasHandle_t handle; CHECKB(cublasCreate(&handle));
    // Set 32 MB workspace for cuBLAS as recommended in GEMMul8 README.
    void* cublas_ws = nullptr; CHECKC(cudaMalloc(&cublas_ws, 32 << 20));
    CHECKB(cublasSetWorkspace(handle, cublas_ws, 32 << 20));

    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C[M, N] = A[M, K] @ B[K, N] computed column-major: D = B^T @ A^T
    // In cuBLAS col-major: op(B_buf)=N, op(A_buf)=N -> m_blas=N, n_blas=M, k_blas=K.
    const int m_b = N, n_b = M, k_b = K;
    const int lda_b = N, ldb_b = K, ldc_b = N;

    // Baseline cuBLAS SGEMM
    CHECKB(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b, &alpha, B, lda_b, A, ldb_b, &beta, Cref, ldc_b));
    cudaDeviceSynchronize();
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < 3; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b, &alpha, B, lda_b, A, ldb_b, &beta, Cref, ldc_b);
    cudaDeviceSynchronize(); cudaEventRecord(ev_s);
    for (int i = 0; i < reps; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b, &alpha, B, lda_b, A, ldb_b, &beta, Cref, ldc_b);
    cudaEventRecord(ev_e); cudaDeviceSynchronize();
    float sgemm_ms = 0; cudaEventElapsedTime(&sgemm_ms, ev_s, ev_e);
    printf("cublas_sgemm_us,%.2f\n", sgemm_ms * 1000.0f / reps);

    // Query GEMMul8 workspace
    const int moduli_list[] = {6, 7, 8};
    for (int moduli : moduli_list) {
        for (int fast : {0, 1}) {
            bool fastmode = (fast == 1);
            // Query workspace with work = nullptr
            auto ws_sizes = gemmul8::gemm<float, gemmul8::Backend::INT8>(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b,
                &alpha, B, lda_b, A, ldb_b, &beta, C, ldc_b,
                moduli, fastmode, nullptr, nullptr, nullptr, false, true, false, false);
            size_t ws_total = (size_t)ws_sizes[0];
            size_t ws_A = (size_t)ws_sizes[1];
            size_t ws_B = (size_t)ws_sizes[2];

            void *work = nullptr, *workB = nullptr;
            CHECKC(cudaMalloc(&work, ws_total));
            if (ws_B > 0) CHECKC(cudaMalloc(&workB, ws_B));

            // Warm-up & preprocess B
            gemmul8::gemm<float, gemmul8::Backend::INT8>(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b,
                &alpha, B, lda_b, A, ldb_b, &beta, C, ldc_b,
                moduli, fastmode, work, nullptr, workB, false, true, false, false);
            cudaDeviceSynchronize();

            // Correctness check vs Cref
            std::vector<float> hc((size_t)M * N), hr((size_t)M * N);
            CHECKC(cudaMemcpy(hc.data(), C, hc.size() * 4, cudaMemcpyDeviceToHost));
            CHECKC(cudaMemcpy(hr.data(), Cref, hr.size() * 4, cudaMemcpyDeviceToHost));
            double maxdiff = 0, maxref = 0;
            for (size_t j = 0; j < hc.size(); ++j) {
                double diff = fabsf(hc[j] - hr[j]);
                if (diff > maxdiff) maxdiff = diff;
                if (fabsf(hr[j]) > maxref) maxref = fabsf(hr[j]);
            }

            // Timed run: without skip_scalB (full pipeline)
            cudaEventRecord(ev_s);
            std::vector<double> last_t;
            for (int i = 0; i < reps; ++i) {
                last_t = gemmul8::gemm<float, gemmul8::Backend::INT8>(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b,
                    &alpha, B, lda_b, A, ldb_b, &beta, C, ldc_b,
                    moduli, fastmode, work, nullptr, workB, false, true, false, false);
            }
            cudaEventRecord(ev_e); cudaDeviceSynchronize();
            float full_ms = 0; cudaEventElapsedTime(&full_ms, ev_s, ev_e);
            float full_us = full_ms * 1000.0f / reps;

            // Timed run: with skip_scalB (preprocessed B reuse!)
            cudaEventRecord(ev_s);
            std::vector<double> reuse_t;
            for (int i = 0; i < reps; ++i) {
                reuse_t = gemmul8::gemm<float, gemmul8::Backend::INT8>(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, m_b, n_b, k_b,
                    &alpha, B, lda_b, A, ldb_b, &beta, C, ldc_b,
                    moduli, fastmode, work, nullptr, workB, false, true, false, true);
            }
            cudaEventRecord(ev_e); cudaDeviceSynchronize();
            float reuse_ms = 0; cudaEventElapsedTime(&reuse_ms, ev_s, ev_e);
            float reuse_us = reuse_ms * 1000.0f / reps;

            printf("gemmul8,moduli=%d,fast=%d,ws_mb=%.1f,maxdiff=%.3e,maxref=%.3e,full_us=%.2f,reuse_us=%.2f,t_scale_us=%.1f,t_mma_us=%.1f,t_req_us=%.1f,t_crt_us=%.1f\n",
                   moduli, fast, (double)ws_total / (1024*1024), maxdiff, maxref, full_us, reuse_us,
                   reuse_t[0] * 1e6, reuse_t[1] * 1e6, reuse_t[2] * 1e6, reuse_t[3] * 1e6);

            CHECKC(cudaFree(work));
            if (workB) CHECKC(cudaFree(workB));
        }
    }
    return 0;
}
