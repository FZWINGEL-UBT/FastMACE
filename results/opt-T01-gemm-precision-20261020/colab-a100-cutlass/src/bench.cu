// OPT-T01 screen: CUTLASS 3xTF32 (OpMultiplyAddFastF32) vs cuBLAS IEEE SGEMM on FastMACE's production GEMMs.
// Modes per shape: cutlass_3xtf32 and cutlass_1xtf32 (each tile config of kernels.cuh), cublas_ieee (cublasSgemm,
// CUBLAS_DEFAULT_MATH) and cublas_dgemm (reference, float inputs converted to double). Timing as in FastMACE's
// tools/gemm_precision_bench.py: 5 warm-up calls, then `repeats` calls each between CUDA events, median.
// Errors against the float64 product: max|C - R| / max|R| and ||C - R||_F / ||R||_F.
// Usage: bench SHAPES_IN.tsv OUT_LONG.csv CAPTURE.json [repeats=50] [peak=4096]
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "kernels.cuh"

#define X(name) Runner* make_##name##_t0(bool); Runner* make_##name##_t1(bool); Runner* make_##name##_t2(bool);
#include "combos.inc"
#undef X

static Runner* make(const std::string& combo, int tile, bool fast) {
#define X(name) if (combo == #name) return tile == 0 ? make_##name##_t0(fast) : tile == 1 ? make_##name##_t1(fast) : make_##name##_t2(fast);
#include "combos.inc"
#undef X
  return nullptr;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1);} } while (0)

struct Shape {
  int idx; std::string structure; int m, k, n; char la; int lda, aa; char lb; int ldb, ab, ac; std::string combo; double inv;
};

static size_t storage(char lay, int rows, int cols, int ld) {
  return lay == 'R' ? (size_t)(rows - 1) * ld + cols : (size_t)(cols - 1) * ld + rows;
}

__global__ void to_double(const float* x, double* y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}

static double time_us(const std::function<void()>& fn, int repeats, cudaStream_t s) {
  for (int i = 0; i < 5; ++i) fn();
  CK(cudaStreamSynchronize(s));
  std::vector<double> t;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < repeats; ++i) {
    CK(cudaEventRecord(a, s));
    fn();
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms * 1e3);
  }
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  std::sort(t.begin(), t.end());
  size_t h = t.size() / 2;
  return t.size() % 2 ? t[h] : 0.5 * (t[h - 1] + t[h]);  // statistics.median
}

static void errors(const std::vector<float>& c, const std::vector<double>& r, double& mx, double& fr) {
  double dmax = 0, rmax = 0, d2 = 0, r2 = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    double d = (double)c[i] - r[i];
    dmax = std::max(dmax, std::fabs(d)); rmax = std::max(rmax, std::fabs(r[i]));
    d2 += d * d; r2 += r[i] * r[i];
  }
  mx = rmax ? dmax / rmax : 0.0;
  fr = r2 ? std::sqrt(d2) / std::sqrt(r2) : 0.0;
}

struct Bufs {
  float *A = nullptr, *B = nullptr, *C = nullptr; double *Ad = nullptr, *Bd = nullptr, *Cd = nullptr;
  size_t na, nb, nc;
  ~Bufs() { cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Ad); cudaFree(Bd); cudaFree(Cd); }
};

static FILE* out_csv;
static cublasHandle_t handle;
static cudaStream_t stream;

static void bench_shape(const Shape& sh, int repeats, std::mt19937_64& rng) {
  Bufs b;
  b.na = storage(sh.la, sh.m, sh.k, sh.lda); b.nb = storage(sh.lb, sh.k, sh.n, sh.ldb); b.nc = (size_t)sh.m * sh.n;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> ha(b.na), hb(b.nb), hc(b.nc);
  for (auto& v : ha) v = nd(rng);
  for (auto& v : hb) v = nd(rng);
  CK(cudaMalloc(&b.A, b.na * 4)); CK(cudaMalloc(&b.B, b.nb * 4)); CK(cudaMalloc(&b.C, b.nc * 4));
  CK(cudaMalloc(&b.Ad, b.na * 8)); CK(cudaMalloc(&b.Bd, b.nb * 8)); CK(cudaMalloc(&b.Cd, b.nc * 8));
  CK(cudaMemcpy(b.A, ha.data(), b.na * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.B, hb.data(), b.nb * 4, cudaMemcpyHostToDevice));
  to_double<<<(unsigned)((b.na + 255) / 256), 256>>>(b.A, b.Ad, b.na);
  to_double<<<(unsigned)((b.nb + 255) / 256), 256>>>(b.B, b.Bd, b.nb);
  CK(cudaDeviceSynchronize());
  // Row-major C = A B as column-major C^T = B^T A^T.
  cublasOperation_t opA = sh.la == 'R' ? CUBLAS_OP_N : CUBLAS_OP_T, opB = sh.lb == 'R' ? CUBLAS_OP_N : CUBLAS_OP_T;
  const double one_d = 1, zero_d = 0; const float one = 1, zero = 0;
  auto dgemm = [&] { CB(cublasDgemm(handle, opB, opA, sh.n, sh.m, sh.k, &one_d, b.Bd, sh.ldb, b.Ad, sh.lda, &zero_d, b.Cd, sh.n)); };
  auto sgemm = [&] { CB(cublasSgemm(handle, opB, opA, sh.n, sh.m, sh.k, &one, b.B, sh.ldb, b.A, sh.lda, &zero, b.C, sh.n)); };
  dgemm(); CK(cudaStreamSynchronize(stream));
  std::vector<double> ref(b.nc);
  CK(cudaMemcpy(ref.data(), b.Cd, b.nc * 8, cudaMemcpyDeviceToHost));
  double flops = 2.0 * sh.m * sh.k * sh.n;
  auto emit = [&](const char* mode, const char* tile, const char* status, double us, double mx, double fr) {
    fprintf(out_csv, "%d,%s,%s,%s,%.4f,%.6f,%.6e,%.6e\n", sh.idx, mode, tile, status, us,
            us > 0 ? flops / (us * 1e-6) / 1e12 : 0.0, mx, fr);
    fflush(out_csv);
  };
  auto check = [&](const char* mode, const char* tile, double us) {
    CK(cudaMemcpy(hc.data(), b.C, b.nc * 4, cudaMemcpyDeviceToHost));
    double mx, fr; errors(hc, ref, mx, fr); emit(mode, tile, "ok", us, mx, fr);
    printf("  %-15s %-24s %9.2f us  max_rel %.2e  frob %.2e\n", mode, tile, us, mx, fr);
  };
  double us = time_us(dgemm, repeats, stream);
  emit("cublas_dgemm", "-", "ok", us, 0, 0);
  CK(cudaMemset(b.C, 0, b.nc * 4));
  us = time_us(sgemm, repeats, stream); check("cublas_ieee", "-", us);
  Prob p{sh.m, sh.n, sh.k, b.A, sh.lda, b.B, sh.ldb, b.C, sh.n};
  for (int fast = 1; fast >= 0; --fast) {
    const char* mode = fast ? "cutlass_3xtf32" : "cutlass_1xtf32";
    for (int t = 0; t < 3; ++t) {
      Runner* r = make(sh.combo, t, fast);
      if (!r) { emit(mode, kTileName[t], "no_instantiation", 0, 0, 0); continue; }
      CK(cudaMemset(b.C, 0, b.nc * 4));
      if (!r->init(p)) { emit(mode, kTileName[t], "refused", 0, 0, 0); delete r; continue; }
      bool ok = true;
      us = time_us([&] { ok = ok && r->run(stream); }, repeats, stream);
      CK(cudaGetLastError());
      if (!ok) emit(mode, kTileName[t], "run_failed", 0, 0, 0); else check(mode, kTileName[t], us);
      delete r;
    }
  }
}

static void capture_check(const char* path, std::mt19937_64& rng) {
  // NbCl5's largest atom-level linear layer: 13608 x 512 x 512, row-major, example-27 tiles, 3xTF32.
  const int m = 13608, k = 512, n = 512;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> ha((size_t)m * k), hb((size_t)k * n);
  for (auto& v : ha) v = nd(rng);
  for (auto& v : hb) v = nd(rng);
  float *A, *B, *C;
  CK(cudaMalloc(&A, ha.size() * 4)); CK(cudaMalloc(&B, hb.size() * 4)); CK(cudaMalloc(&C, (size_t)m * n * 4));
  CK(cudaMemcpy(A, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(B, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  Runner* r = make("RR_4_4_4", 0, true);
  Prob p{m, n, k, A, k, B, n, C, n};
  bool ok = r->init(p) && r->run(stream);
  CK(cudaStreamSynchronize(stream));
  std::vector<float> eager((size_t)m * n), g1(eager.size()), g2(eager.size());
  CK(cudaMemcpy(eager.data(), C, eager.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemset(C, 0, eager.size() * 4));
  CK(cudaStreamSynchronize(stream));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  ok = ok && r->run(stream);
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, stream)); CK(cudaStreamSynchronize(stream));
  CK(cudaMemcpy(g1.data(), C, g1.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemset(C, 0, eager.size() * 4));
  CK(cudaGraphLaunch(exec, stream)); CK(cudaStreamSynchronize(stream));
  CK(cudaMemcpy(g2.data(), C, g2.size() * 4, cudaMemcpyDeviceToHost));
  bool eq1 = !memcmp(eager.data(), g1.data(), eager.size() * 4), eq2 = !memcmp(eager.data(), g2.data(), eager.size() * 4);
  FILE* f = fopen(path, "w");
  fprintf(f, "{\"shape\": [%d, %d, %d], \"mode\": \"cutlass_3xtf32\", \"tile\": \"%s\", \"ran\": %s, "
             "\"replay1_bitwise_equal_eager\": %s, \"replay2_bitwise_equal_eager\": %s}\n",
          m, k, n, kTileName[0], ok ? "true" : "false", eq1 ? "true" : "false", eq2 ? "true" : "false");
  fclose(f);
  printf("capture: ran %d replay1 %d replay2 %d\n", ok, eq1, eq2);
  CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
  delete r; cudaFree(A); cudaFree(B); cudaFree(C);
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: bench SHAPES_IN.tsv OUT_LONG.csv CAPTURE.json [repeats] [peak]\n"); return 2; }
  int repeats = argc > 4 ? atoi(argv[4]) : 50, peak = argc > 5 ? atoi(argv[5]) : 4096;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CB(cublasCreate(&handle));
  CB(cublasSetStream(handle, stream));
  cublasMath_t math; CB(cublasGetMathMode(handle, &math));
  const char* ovr = getenv("NVIDIA_TF32_OVERRIDE");
  printf("cublas math mode %d (CUBLAS_DEFAULT_MATH=%d, TF32_TENSOR_OP=%d), NVIDIA_TF32_OVERRIDE=%s\n", (int)math,
         (int)CUBLAS_DEFAULT_MATH, (int)CUBLAS_TF32_TENSOR_OP_MATH, ovr ? ovr : "(unset)");
  if (math != CUBLAS_DEFAULT_MATH || (ovr && strcmp(ovr, "0"))) { fprintf(stderr, "TF32 may be on\n"); return 1; }
  out_csv = fopen(argv[2], "w");
  fprintf(out_csv, "idx,mode,tile,status,us,tflops,max_rel,frob_rel\n");
  std::mt19937_64 rng(0);
  capture_check(argv[3], rng);
  Shape pk{-1, "peak", peak, peak, peak, 'R', peak, 4, 'R', peak, 4, 4, "RR_4_4_4", 0};
  printf("peak %d^3\n", peak);
  bench_shape(pk, std::max(10, repeats / 5), rng);  // as gemm_precision_bench: max(10, repeats // 5)
  std::ifstream in(argv[1]);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    Shape s;
    ss >> s.idx >> s.structure >> s.m >> s.k >> s.n >> s.la >> s.lda >> s.aa >> s.lb >> s.ldb >> s.ab >> s.ac >> s.combo >> s.inv;
    printf("[%d] %s %d x %d x %d  A %c ld %d  B %c ld %d  %s\n", s.idx, s.structure.c_str(), s.m, s.k, s.n, s.la, s.lda,
           s.lb, s.ldb, s.combo.c_str());
    bench_shape(s, repeats, rng);
  }
  fclose(out_csv);
  CB(cublasDestroy(handle));
  return 0;
}
