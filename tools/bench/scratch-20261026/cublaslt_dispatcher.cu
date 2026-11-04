#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

static cublasLtHandle_t g_lt = nullptr;
static void* g_ws = nullptr;
static size_t g_ws_size = 4 << 20; // 4 MB workspace

struct CacheKey {
    int m, n, k;
    bool operator==(const CacheKey& o) const {
        return m == o.m && n == o.n && k == o.k;
    }
};

struct KeyHash {
    size_t operator()(const CacheKey& k) const {
        return ((size_t)k.m << 32) ^ ((size_t)k.n << 16) ^ (size_t)k.k;
    }
};

struct CachedMatmul {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t Ad;
    cublasLtMatrixLayout_t Bd;
    cublasLtMatrixLayout_t Cd;
    cublasLtMatmulAlgo_t algo;
    size_t ws_bytes;
};

static std::unordered_map<CacheKey, CachedMatmul, KeyHash> g_cache;

extern "C" {

int init_cublaslt() {
    if (!g_lt) {
        if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
    }
    if (!g_ws) {
        if (cudaMalloc(&g_ws, g_ws_size) != cudaSuccess) return -2;
    }
    return 0;
}

int cublaslt_gemm_fp32(const float* A, const float* B, float* C,
                       int M, int N, int K, cudaStream_t stream) {
    if (!g_lt) {
        if (init_cublaslt() != 0) return -1;
    }

    CacheKey key{M, N, K};
    auto it = g_cache.find(key);
    const float alpha = 1.0f, beta = 0.0f;

    if (it != g_cache.end()) {
        const auto& entry = it->second;
        auto status = cublasLtMatmul(g_lt, entry.desc, &alpha, B, entry.Ad, A, entry.Bd, &beta, C, entry.Cd, C, entry.Cd,
                                     &entry.algo, g_ws, entry.ws_bytes, stream);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : (int)status;
    }

    // First call: initialize layout and find best algorithm
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_32F, N, K, N);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_32F, K, M, K);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &g_ws_size, sizeof(g_ws_size));
    const int want = 32;
    cublasLtMatmulHeuristicResult_t results[want];
    int count = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, desc, Ad, Bd, Cd, Cd, pref, want, results, &count);
    cublasLtMatmulPreferenceDestroy(pref);

    int best_idx = -1;
    for (int i = 0; i < count; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
        int tile = -1;
        size_t sz = sizeof(int);
        cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sz, &sz);
        if (M >= 5000 && tile == 24) {
            best_idx = i;
            break;
        }
        if (M < 5000 && (tile == 18 || tile == 16)) {
            best_idx = i;
            break;
        }
    }
    if (best_idx < 0 && count > 0) best_idx = 0;
    if (best_idx < 0) return -2;

    CachedMatmul entry{desc, Ad, Bd, Cd, results[best_idx].algo, results[best_idx].workspaceSize};
    g_cache[key] = entry;

    auto status = cublasLtMatmul(g_lt, entry.desc, &alpha, B, entry.Ad, A, entry.Bd, &beta, C, entry.Cd, C, entry.Cd,
                                 &entry.algo, g_ws, entry.ws_bytes, stream);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : (int)status;
}

}
