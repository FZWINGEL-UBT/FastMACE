// cuBLASDx INT8-slice float32 emulation (MathDx 26.10, cuBLASDx RequiredMantissaBits<X>) on the production atom-GEMM
// shapes, against cuBLAS SGEMM (default math, IEEE float32). Follows MathDx example 21_sgemm_emulation_pipeline.
//
// For each X in {15, 23, 31} (2, 3, 4 INT8 slices) and each shape (M x K) @ (K x N), A row-major [M, K]
// (activations), B column-major [K, N] (a torch weight [N, K] row-major), C row-major:
//   full    suggest_pipeline (allocation and slicing of A and B) + the GEMM kernel, every call: the cost of a call
//           whose operands are new;
//   kernel  the GEMM kernel alone on one pipeline: the lower bound (operands sliced once, e.g. a weight-only reuse
//           would sit between the two);
//   cublas  cublasSgemm.
// Errors against a float64 product (cublasDgemm of the same float32 inputs): max |c - c64| / max |c64| and the
// cancellation-scaled max |c - c64| / (|A| @ |B|), on Gaussian operands and on the alternating-sign set of the GEMM
// probe (A[:, 2k+1] = -A[:, 2k] (1 + 1e-3 u), B[2k+1, :] = B[2k, :]). Medians of 7 timed repeats of 20 calls.
//
// Build (WSL, CUDA 13.3, MathDx 26.10.0 headers):
//   D=<mathdx>/nvidia/mathdx/26.10
//   nvcc -std=c++17 -O3 -w --expt-relaxed-constexpr -gencode arch=compute_89,code=sm_89 \
//     -gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=sm_80 -I$D/include \
//     -I$D/external/cutlass/include cublasdx_int8_emulation.cu -o cublasdx_int8_emulation -lcublas
// For Ubuntu 22.04 (glibc 2.35, leto) also link results/cublasdx-int8-20261103/glibc235_shim.c (gcc -c -O2) and add
// -Xcompiler -static-libstdc++,-static-libgcc.
// Run: ./cublasdx_int8_emulation > out.json
#include <cublas_v2.h>
#include <cublasdx.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECKC(x) do { auto s_ = (x); if (s_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s line %d\n", cudaGetErrorString(s_), __LINE__); exit(1); } } while (0)
#define CHECKB(x) do { auto s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d line %d\n", (int)s_, __LINE__); exit(1); } } while (0)

template<class BLAS, class CTensor, class Pipeline>
__launch_bounds__(Pipeline::max_threads_per_block, 1) __global__
void pipeline_kernel(CTensor global_c, __grid_constant__ Pipeline const device_pipeline) {
    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(BLAS) {  // only the device pass of the matching architecture
        extern __shared__ __align__(128) cublasdx::byte smem[];
        const unsigned tile_m = blockIdx.x, tile_n = blockIdx.y;
        auto tile_c = cublasdx::get_tile(global_c, BLAS::c_shape, tile_m, tile_n);
        auto tile_pipeline = device_pipeline.get_tile(smem, tile_m, tile_n);
        auto accumulator = tile_pipeline.get_accumulator();
        tile_pipeline.execute(accumulator);
        if (accumulator.is_thread_active()) {
            auto c_fragment = accumulator.get_results();
            auto d_fragment = accumulator.make_partition_and_copy(tile_c);
            cublasdx::axpby(1.0f, c_fragment, 0.0f, d_fragment);
            accumulator.partition_and_copy(d_fragment, tile_c);
        }
    }
}

struct Result { double full_us = 0, kernel_us = 0, max_rel = 0, cancel = 0; bool ok = false; };

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template<class F>
static double time_us(F&& f, int calls = 20, int repeats = 7) {
    cudaEvent_t a, b; CHECKC(cudaEventCreate(&a)); CHECKC(cudaEventCreate(&b));
    f(); f(); CHECKC(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < repeats; ++r) {
        CHECKC(cudaEventRecord(a));
        for (int i = 0; i < calls; ++i) f();
        CHECKC(cudaEventRecord(b)); CHECKC(cudaEventSynchronize(b));
        float ms; CHECKC(cudaEventElapsedTime(&ms, a, b)); t.push_back(1000.0 * ms / calls);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return median(t);
}

// errors of c (row-major [M, N]) against c64 (column-major [M, N] from cublasDgemm) and |A|@|B| (column-major)
static void errors(const std::vector<float>& c, const std::vector<double>& c64, const std::vector<double>& mag,
                   int m, int n, double& max_rel, double& cancel) {
    double top = 0, worst = 0; cancel = 0;
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        double ref = c64[size_t(j) * m + i], err = std::fabs(double(c[size_t(i) * n + j]) - ref);
        top = std::max(top, std::fabs(ref)); worst = std::max(worst, err);
        cancel = std::max(cancel, err / (mag[size_t(j) * m + i] + 1e-30));
    }
    max_rel = worst / top;
}

template<unsigned Bits, unsigned Arch>
Result emulated(const float* da, const float* db, float* dc, int m, int n, int k, const std::vector<double>& c64,
                const std::vector<double>& mag) {
    using common = decltype(cublasdx::Precision<float, float, float>() + cublasdx::Block() + cublasdx::WithPipeline()
                            + cublasdx::RequiredMantissaBits<Bits>() + cublasdx::SM<Arch>());
    constexpr auto tile = [](unsigned ratio) {
        return ratio > 1 ? std::array<unsigned, 3> {128, 128, 128} : std::array<unsigned, 3> {64, 64, 32};
    }(cublasdx::emulated_stage_size_ratio_v<common>);
    using BLAS = decltype(cublasdx::Size<tile[0], tile[1], tile[2]>() + common {} + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::row_major>()
                          + cublasdx::BlockDim<128>() + cublasdx::StaticBlockDim() + cublasdx::Alignment<16, 16, 16>()
                          + cublasdx::EnableInputStreaming());
    Result r;
    if (m % tile[0] || n % tile[1] || k % tile[2]) return r;
    auto ga = cublasdx::make_gmem_tensor<cublasdx::row_major>(da, m, k, k);
    auto gb = cublasdx::make_gmem_tensor<cublasdx::col_major>(db, k, n, k);
    auto gc = cublasdx::make_gmem_tensor<cublasdx::row_major>(dc, m, n, n);
    // The pipeline (an expected<>) can be neither copied nor moved: built in place where it is used.
#define MAKE_PIPELINE(p) \
    auto p = cublasdx::suggest_pipeline<BLAS, cublasdx::reusable_accumulator>(ga, gb, cudaStream_t(0)); \
    if (!p) { fprintf(stderr, "pipeline: %s\n", cublasdx::pipeline_error_string(p.error().code)); exit(1); }
    auto launch = [&](auto& p) {
        auto kernel = pipeline_kernel<BLAS, decltype(gc), decltype(p->get_device_handle())>;
        CHECKC(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(p->buffer_size())));
        kernel<<<dim3(m / tile[0], n / tile[1]), p->get_block_dim(), p->buffer_size()>>>(gc, p->get_device_handle());
    };
    r.full_us = time_us([&]() { MAKE_PIPELINE(p) launch(p); });
    {
        MAKE_PIPELINE(p)
        r.kernel_us = time_us([&]() { launch(p); });
    }
#undef MAKE_PIPELINE
    CHECKC(cudaDeviceSynchronize());
    std::vector<float> c(size_t(m) * n);
    CHECKC(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
    errors(c, c64, mag, m, n, r.max_rel, r.cancel);
    r.ok = true;
    return r;
}

template<unsigned Arch>
void run_bits(const char* label, const float* da, const float* db, float* dc, int m, int n, int k,
              const std::vector<double>& c64, const std::vector<double>& mag, bool& first) {
    Result rs[3] = {emulated<15, Arch>(da, db, dc, m, n, k, c64, mag), emulated<23, Arch>(da, db, dc, m, n, k, c64, mag),
                    emulated<31, Arch>(da, db, dc, m, n, k, c64, mag)};
    const unsigned bits[3] = {15, 23, 31};  // ceil((X + 1) / 8) = 2, 3, 4 INT8 slices
    for (int i = 0; i < 3; ++i) {
        printf("%s    {\"set\": \"%s\", \"bits\": %u, \"full_us\": %.2f, \"kernel_us\": %.2f, \"max_rel\": %.3e, "
               "\"cancel\": %.3e}", first ? "" : ",\n", label, bits[i], rs[i].full_us, rs[i].kernel_us, rs[i].max_rel,
               rs[i].cancel);
        first = false;
    }
}

int main() {
    int dev; CHECKC(cudaGetDevice(&dev));
    cudaDeviceProp prop; CHECKC(cudaGetDeviceProperties(&prop, dev));
    const int arch = prop.major * 100 + prop.minor * 10;
    cublasHandle_t h; CHECKB(cublasCreate(&h));
    const std::array<std::array<int, 3>, 4> shapes = {{{4096, 4096, 4096}, {1536, 512, 512}, {5888, 512, 512},
                                                      {13696, 512, 512}}};
    printf("{\"gpu\": \"%s\", \"arch\": %d, \"mathdx\": \"26.10.0\", \"shapes\": [\n", prop.name, arch);
    for (size_t s = 0; s < shapes.size(); ++s) {
        const int m = shapes[s][0], n = shapes[s][1], k = shapes[s][2];
        float *da, *db, *dc; double *da64, *db64, *dc64;
        CHECKC(cudaMalloc(&da, size_t(m) * k * 4)); CHECKC(cudaMalloc(&db, size_t(k) * n * 4));
        CHECKC(cudaMalloc(&dc, size_t(m) * n * 4)); CHECKC(cudaMalloc(&da64, size_t(m) * k * 8));
        CHECKC(cudaMalloc(&db64, size_t(k) * n * 8)); CHECKC(cudaMalloc(&dc64, size_t(m) * n * 8));
        printf("%s  {\"m\": %d, \"n\": %d, \"k\": %d, \"results\": [\n", s ? ",\n" : "", m, n, k);
        bool first = true;
        double cublas_us = 0;
        for (int set = 0; set < 2; ++set) {
            std::mt19937 gen(20261103 + set);
            std::normal_distribution<float> nd; std::uniform_real_distribution<float> ud;
            // a: row-major [m, k]; b: column-major [k, n] (column j = k contiguous values)
            std::vector<float> a(size_t(m) * k), b(size_t(k) * n);
            for (auto& x : a) x = nd(gen);
            for (auto& x : b) x = nd(gen);
            if (set == 1) {
                for (int i = 0; i < m; ++i) for (int p = 0; p + 1 < k; p += 2)
                    a[size_t(i) * k + p + 1] = -a[size_t(i) * k + p] * (1.0f + 1e-3f * ud(gen));
                for (int j = 0; j < n; ++j) for (int p = 0; p + 1 < k; p += 2)
                    b[size_t(j) * k + p + 1] = b[size_t(j) * k + p];
            }
            CHECKC(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
            CHECKC(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
            // float64 reference and |A|@|B| in column-major C: C^T = B^T A^T, i.e. column-major C[m, n] = A B with
            // A (row-major [m, k]) = column-major [k, m] transposed.
            std::vector<double> c64(size_t(m) * n), mag(size_t(m) * n);
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<double> a64(a.size()), b64(b.size());
                for (size_t i = 0; i < a.size(); ++i) a64[i] = pass ? std::fabs(double(a[i])) : double(a[i]);
                for (size_t i = 0; i < b.size(); ++i) b64[i] = pass ? std::fabs(double(b[i])) : double(b[i]);
                CHECKC(cudaMemcpy(da64, a64.data(), a64.size() * 8, cudaMemcpyHostToDevice));
                CHECKC(cudaMemcpy(db64, b64.data(), b64.size() * 8, cudaMemcpyHostToDevice));
                const double one = 1, zero = 0;
                CHECKB(cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, da64, k, db64, k, &zero, dc64, m));
                CHECKC(cudaMemcpy(pass ? mag.data() : c64.data(), dc64, size_t(m) * n * 8, cudaMemcpyDeviceToHost));
            }
            // cuBLAS SGEMM: row-major C [m, n] = A B  <=>  column-major C^T [n, m] = B^T A^T
            const float one = 1, zero = 0;
            auto sgemm = [&]() { CHECKB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, db, k, da, k, &zero, dc, n)); };
            const double us = time_us(sgemm);
            if (set == 0) cublas_us = us;
            sgemm(); CHECKC(cudaDeviceSynchronize());
            std::vector<float> c(size_t(m) * n);
            CHECKC(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
            double max_rel, cancel; errors(c, c64, mag, m, n, max_rel, cancel);
            const char* label = set ? "cancel" : "random";
            printf("%s    {\"set\": \"%s\", \"bits\": \"cublas\", \"full_us\": %.2f, \"kernel_us\": %.2f, "
                   "\"max_rel\": %.3e, \"cancel\": %.3e}", first ? "" : ",\n", label, us, us, max_rel, cancel);
            first = false;
            if (arch == 890) run_bits<890>(label, da, db, dc, m, n, k, c64, mag, first);
            else if (arch == 1200) run_bits<1200>(label, da, db, dc, m, n, k, c64, mag, first);
            else if (arch == 800) run_bits<800>(label, da, db, dc, m, n, k, c64, mag, first);
            fflush(stdout);
        }
        printf("\n  ], \"cublas_random_us\": %.2f}", cublas_us);
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(da64); cudaFree(db64); cudaFree(dc64);
    }
    printf("\n]}\n");
    return 0;
}
