// OPT5-T09: native cuBLASLt FP32 algorithm search (heuristic candidates,
// swept workspace budgets), timed against plain cublasSgemm on the same
// buffers. Row-major C[MxN] = A[MxK] @ B[KxN] is computed through the
// column-major identity D = B^T @ A^T on the same memory.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { auto s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d at line %d\n", (int)s, __LINE__); return 1; } } while (0)
#define CHECKC(x) do { auto s = (x); if (s != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(s), __LINE__); return 1; } } while (0)

static float time_lt(cublasLtHandle_t lt, cublasLtMatmulDesc_t desc,
                     const void* alpha, const void* A, cublasLtMatrixLayout_t Ad,
                     const void* B, cublasLtMatrixLayout_t Bd, const void* beta,
                     const void* C, cublasLtMatrixLayout_t Cd, void* D, cublasLtMatrixLayout_t Dd,
                     const cublasLtMatmulAlgo_t* algo, void* ws, size_t ws_bytes, int reps) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 3; ++i) {
        auto st = cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd, algo, ws, ws_bytes, 0);
        if (st != CUBLAS_STATUS_SUCCESS) { cudaEventDestroy(s); cudaEventDestroy(e); return -1.0f; }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < reps; ++i)
        cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd, algo, ws, ws_bytes, 0);
    cudaEventRecord(e);
    cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms * 1000.0f / reps;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 13608;
    const int N = argc > 2 ? atoi(argv[2]) : 512;
    const int K = argc > 3 ? atoi(argv[3]) : 512;
    const int reps = argc > 4 ? atoi(argv[4]) : 30;
    printf("shape MxNxK = %dx%dx%d\n", M, N, K);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = sinf((float)i * 0.001f) * 0.5f;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = cosf((float)i * 0.0013f) * 0.5f;
    float *A, *B, *C, *Cref;
    CHECKC(cudaMalloc(&A, hA.size() * 4)); CHECKC(cudaMalloc(&B, hB.size() * 4));
    CHECKC(cudaMalloc(&C, (size_t)M * N * 4)); CHECKC(cudaMalloc(&Cref, (size_t)M * N * 4));
    CHECKC(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CHECKC(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    // Baseline: plain SGEMM through the same transpose identity.
    cublasHandle_t handle; CHECK(cublasCreate(&handle));
    CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, Cref, N));
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 3; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, Cref, N);
    cudaDeviceSynchronize(); cudaEventRecord(s);
    for (int i = 0; i < reps; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, Cref, N);
    cudaEventRecord(e); cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    printf("cublas_sgemm_us,%.2f\n", ms * 1000.0f / reps);

    cublasLtHandle_t lt; CHECK(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t desc;
    CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    // A_lt = B buffer viewed column-major (N x K); B_lt = A buffer (K x M); D (N x M).
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_32F, N, K, N));
    CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_32F, K, M, K));
    CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, N, M, N));

    const size_t ws_sizes[] = {0, 1 << 22, 1 << 25};
    void* ws = nullptr; CHECKC(cudaMalloc(&ws, 1 << 25));
    int printed = 0;
    for (size_t wsi = 0; wsi < 3; ++wsi) {
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t wsb = ws_sizes[wsi];
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wsb, sizeof(wsb));
        const int want = 128;
        std::vector<cublasLtMatmulHeuristicResult_t> results(want);
        int count = 0;
        auto st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref,
                                                 want, results.data(), &count);
        if (st != CUBLAS_STATUS_SUCCESS) { printf("heuristic failed ws=%zu status=%d\n", wsb, (int)st); continue; }
        for (int i = 0; i < count; ++i) {
            if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
            float us = time_lt(lt, desc, &alpha, B, Ad, A, Bd, &beta, C, Cd, C, Cd,
                               &results[i].algo, ws, results[i].workspaceSize, reps);
            if (us < 0) continue;
            // Correctness spot check against the SGEMM reference.
            CHECKC(cudaMemcpy(C, Cref, 0, cudaMemcpyDeviceToDevice)); // no-op keep
            cublasLtMatmul(lt, desc, &alpha, B, Ad, A, Bd, &beta, C, Cd, C, Cd,
                           &results[i].algo, ws, results[i].workspaceSize, 0);
            cudaDeviceSynchronize();
            std::vector<float> hc((size_t)M * N), hr((size_t)M * N);
            CHECKC(cudaMemcpy(hc.data(), C, hc.size() * 4, cudaMemcpyDeviceToHost));
            CHECKC(cudaMemcpy(hr.data(), Cref, hr.size() * 4, cudaMemcpyDeviceToHost));
            double maxdiff = 0, maxref = 0;
            for (size_t j = 0; j < hc.size(); j += 997) {
                maxdiff = std::max(maxdiff, (double)fabsf(hc[j] - hr[j]));
                maxref = std::max(maxref, (double)fabsf(hr[j]));
            }
            int tile = -1, splitk = -1, stages = -1;
            size_t sz = sizeof(int);
            cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sz, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sz, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sz, &sz);
            printf("lt_algo,ws=%zu,idx=%d,tile=%d,splitk=%d,stages=%d,algo_ws=%zu,us=%.2f,maxdiff=%.3g,maxref=%.3g\n",
                   wsb, i, tile, splitk, stages, results[i].workspaceSize, us, maxdiff, maxref);
            ++printed;
        }
        cublasLtMatmulPreferenceDestroy(pref);
    }
    printf("printed %d algo configs\n", printed);
    return 0;
}
