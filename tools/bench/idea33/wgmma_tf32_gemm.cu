// Idea 33 timing bound (H200 ideas plan, 6 November 2026): CUTLASS 3.x native Hopper TF32 wgmma GEMMs
// (CollectiveBuilder, sm_90a, TMA + warp-specialised mainloops), one TF32 pass per call, for comparison with
// fastmace/kernels/cutlass/tf32x3.py (sm_80 mma.sync 3xTF32). A lab library, not production code.
//
// D[M, N] (row-major, float) = A[M, K] (row-major, K-major) @ B[K, N], B given as B^T [N, K] row-major (K-major):
// TF32 wgmma reads both operands K-major from shared memory. alpha 1, beta 0, no C operand (ElementC void).
// The operands are read as float bits and the tensor cores use their TF32 part (single pass, ~1e-3 relative).
//
// C API (ctypes):  idea33_configs() -> int;  idea33_name(i) -> const char*;
//   idea33_create(config, M, N, K, A, Bt, D, &status) -> handle (initialised, workspace allocated; status 0 = ok,
//   1 = can_implement failed, 2 = initialize failed);  idea33_run(handle, stream) -> cutlass status (0 = ok);
//   idea33_run_n(handle, stream, n);  idea33_destroy(handle).
// Build: tools/bench/idea33/build.sh (nvcc 13.3, CUTLASS v4.8.1, -arch=sm_90a).

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

struct Runner {
  virtual ~Runner() {}
  virtual int run(cudaStream_t stream) = 0;
};

template <class TileShape, class ClusterShape, class KernelSchedule, class EpilogueSchedule>
struct Config {
  using ElementA = cutlass::tfloat32_t;
  using ElementB = cutlass::tfloat32_t;
  using ElementD = float;
  using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int Align = 4;  // 16 bytes

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, float, void, LayoutD, Align, ElementD, LayoutD,
      Align, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, Align, ElementB, LayoutB, Align,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Cfg>
struct Impl : Runner {
  using Gemm = typename Cfg::Gemm;
  Gemm gemm;
  void* workspace = nullptr;
  int status = 0;

  Impl(int M, int N, int K, const void* A, const void* Bt, void* D) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const typename Cfg::ElementA*>(A), sa,
         reinterpret_cast<const typename Cfg::ElementB*>(Bt), sb},
        {{1.0f, 0.0f}, nullptr, sc, reinterpret_cast<float*>(D), sd},
        hw};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      status = 1;
      return;
    }
    size_t bytes = Gemm::get_workspace_size(args);
    if (bytes) cudaMalloc(&workspace, bytes);
    if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) status = 2;
  }
  ~Impl() override {
    if (workspace) cudaFree(workspace);
  }
  int run(cudaStream_t stream) override { return static_cast<int>(gemm.run(stream)); }
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ECoop = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EPing = cutlass::epilogue::TmaWarpSpecialized;
using C1 = Shape<_1, _1, _1>;
using C2 = Shape<_2, _1, _1>;

using Cfg0 = Config<Shape<_128, _128, _32>, C1, Coop, ECoop>;
using Cfg1 = Config<Shape<_128, _64, _32>, C1, Coop, ECoop>;
using Cfg2 = Config<Shape<_64, _128, _32>, C1, Ping, EPing>;
using Cfg3 = Config<Shape<_64, _64, _32>, C1, Ping, EPing>;
using Cfg4 = Config<Shape<_128, _192, _32>, C1, Coop, ECoop>;
using Cfg5 = Config<Shape<_128, _128, _32>, C2, Coop, ECoop>;
using Cfg6 = Config<Shape<_64, _192, _32>, C1, Ping, EPing>;

static const char* NAMES[] = {
    "128x128x32 cooperative cluster1",  "128x64x32 cooperative cluster1",  "64x128x32 pingpong cluster1",
    "64x64x32 pingpong cluster1",       "128x192x32 cooperative cluster1", "128x128x32 cooperative cluster2x1",
    "64x192x32 pingpong cluster1",
};
static const int NCONFIGS = sizeof(NAMES) / sizeof(NAMES[0]);

// glibc 2.38's headers redirect strtoul to __isoc23_strtoul (GLIBC_2.38); the Modal image has glibc 2.36. Define
// it here, forwarding to the old strtoul symbol version, so the library needs nothing above GLIBC_2.36.
extern "C" unsigned long idea33_old_strtoul(const char*, char**, int);
__asm__(".symver idea33_old_strtoul, strtoul@GLIBC_2.2.5");
extern "C" unsigned long __isoc23_strtoul(const char* s, char** end, int base) {
  return idea33_old_strtoul(s, end, base);
}

extern "C" {

int idea33_configs() { return NCONFIGS; }

const char* idea33_name(int i) { return (i >= 0 && i < NCONFIGS) ? NAMES[i] : ""; }

void* idea33_create(int config, int M, int N, int K, const void* A, const void* Bt, void* D, int* status) {
  Runner* r = nullptr;
  int s = 0;
  switch (config) {
    case 0: { auto* p = new Impl<Cfg0>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    case 1: { auto* p = new Impl<Cfg1>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    case 2: { auto* p = new Impl<Cfg2>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    case 3: { auto* p = new Impl<Cfg3>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    case 4: { auto* p = new Impl<Cfg4>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    case 5: { auto* p = new Impl<Cfg5>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    case 6: { auto* p = new Impl<Cfg6>(M, N, K, A, Bt, D); s = p->status; r = p; break; }
    default: s = 3;
  }
  if (status) *status = s;
  if (s != 0 && r) {
    delete r;
    r = nullptr;
  }
  return r;
}

int idea33_run(void* h, void* stream) { return static_cast<Runner*>(h)->run(static_cast<cudaStream_t>(stream)); }

int idea33_run_n(void* h, void* stream, int n) {
  int s = 0;
  for (int i = 0; i < n && s == 0; ++i) s = static_cast<Runner*>(h)->run(static_cast<cudaStream_t>(stream));
  return s;
}

void idea33_destroy(void* h) { delete static_cast<Runner*>(h); }

}  // extern "C"
