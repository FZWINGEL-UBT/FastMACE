
extern "C" __device__ __forceinline__ float cheb_value(
    const float* __restrict__ coef, long base, int terms, int columns, int k, float x) {
  float v1 = 0.0f, v2 = 0.0f;
  for (int t = terms - 1; t >= 1; --t) {
    float c = coef[base + (long)t * columns + k];
    float v = fmaf(2.0f * x, v1, c - v2);
    v2 = v1; v1 = v;
  }
  return fmaf(x, v1, coef[base + k] - v2);
}

extern "C" __device__ __forceinline__ float cheb_scalar(
    const float* __restrict__ coef, long base, int terms, int columns, int k, float x) {
  return cheb_value(coef, base, terms, columns, k, x);
}

extern "C" __device__ __forceinline__ float edge_x(
    float r, const float* __restrict__ starts, const float* __restrict__ scales,
    const int* __restrict__ bins, int n_bins, float r_lo, float bin_scale, float r_max,
    int* piece_out, bool* inside_out) {
  float clamped = fmaxf(r, r_lo);
  int index = min((int)((clamped - r_lo) * bin_scale), n_bins - 1);
  int piece = bins[index];
  *piece_out = piece;
  *inside_out = r < r_max;
  return (clamped - starts[piece]) * scales[piece] - 1.0f;
}

// Receiver-owned forward.  One CTA writes every value of one receiver's
// [slots,D,K] tile; all output elements therefore have a single writer.
extern "C" __global__ void dmqe_outer_forward(
    const float* __restrict__ r, const int* __restrict__ pair, const long long* __restrict__ key,
    const float* __restrict__ coef, const float* __restrict__ starts,
    const float* __restrict__ scales, const int* __restrict__ bins,
    const float* __restrict__ y, const int* __restrict__ row_ptr,
    float* __restrict__ out, float* __restrict__ density_edge, float* __restrict__ density_receiver,
    int nodes, int slots, int pieces, int n_bins, int terms, int columns,
    float r_lo, float bin_scale, float r_max, int D, int K, int density_mode, int inner) {
  extern __shared__ float sm[];
  float* acc = sm;                         // slots * D * K
  float* aval = acc + slots * D * K;       // K
  float* yval = aval + K;                  // D
  int receiver = (int)blockIdx.x;
  if (receiver >= nodes) return;
  int tid = (int)threadIdx.x;
  int width = D * K;
  for (int i = tid; i < slots * width; i += blockDim.x) acc[i] = 0.0f;
  if (tid == 0 && density_mode) density_receiver[receiver] = 0.0f;
  __syncthreads();

  int begin = row_ptr[receiver], end = row_ptr[receiver + 1];
  for (int e = begin; e < end; ++e) {
    int slot = (int)(key[e] % slots);
    int p = pair[e];
    int piece; bool inside;
    float x = edge_x(r[e], starts, scales, bins, n_bins, r_lo, bin_scale, r_max, &piece, &inside);
    long base = ((long)p * pieces + piece) * terms * columns;
    if (tid < K) {
      aval[tid] = inside ? cheb_value(coef, base, terms, columns, tid, x) : 0.0f;
    }
    if (tid < D) yval[tid] = y[(long)e * D + tid];
    if (tid == 0) {
      float den = inside ? cheb_scalar(coef, base, terms, columns, columns - 1, x) : 0.0f;
      if (density_mode) density_receiver[receiver] += den;
      else density_edge[e] = den;
    }
    __syncthreads();
    for (int q = tid; q < width; q += blockDim.x) {
      int d = q / K, k = q - d * K;
      acc[slot * width + q] = fmaf(yval[d], aval[k], acc[slot * width + q]);
    }
    __syncthreads();
  }
  for (int i = tid; i < slots * width; i += blockDim.x) {
    int slot = i / width, q = i - slot * width;
    int d = q / K, k = q - d * K;
    long dst = inner == 1
        ? (long)(receiver * slots + slot) * width + q
        : (long)receiver * slots * width + (long)d * slots * K + (long)slot * K + k;
    out[dst] = acc[i];
  }
}

// Receiver-owned fused backward.  The four warps cover four lm components
// each; every eight-lane subgroup reduces its K=64 dot by shuffle butterfly.
extern "C" __global__ void dmqe_outer_backward(
    const float* __restrict__ grad, const float* __restrict__ r,
    const int* __restrict__ pair, const long long* __restrict__ key,
    const float* __restrict__ coef, const float* __restrict__ dcoef,
    const float* __restrict__ starts, const float* __restrict__ scales,
    const int* __restrict__ bins, const float* __restrict__ grad_density,
    const float* __restrict__ y, const int* __restrict__ row_ptr,
    float* __restrict__ dr, float* __restrict__ grad_y,
    int nodes, int slots, int pieces, int n_bins, int terms, int dterms, int columns,
    float r_lo, float bin_scale, float r_max, int D, int K, int density_mode, int inner) {
  extern __shared__ float sm[];
  float* g = sm;                       // D*K
  float* aval = g + D*K;               // K
  float* slope = aval + K;             // K
  float* yval = slope + K;             // D
  float* part = yval + D;              // 4 warp partials
  int receiver = (int)blockIdx.x, tid = (int)threadIdx.x;
  if (receiver >= nodes) return;
  int lane = tid & 31, warp = tid >> 5;
  int width = D * K;
  int begin = row_ptr[receiver], end = row_ptr[receiver + 1];
  long long current = -1;
  for (int e = begin; e < end; ++e) {
    long long edge_key = key[e];
    if (edge_key != current) {
      int slot = (int)(edge_key % slots);
      for (int i = tid; i < width; i += blockDim.x) {
        int d = i / K, k = i - d * K;
        long src = inner == 1
            ? (long)(receiver * slots + slot) * width + i
            : (long)receiver * slots * width + (long)d * slots * K + (long)slot * K + k;
        g[i] = grad[src];
      }
      current = edge_key;
    }
    int p = pair[e];
    int piece; bool inside;
    float x = edge_x(r[e], starts, scales, bins, n_bins, r_lo, bin_scale, r_max, &piece, &inside);
    long vbase = ((long)p * pieces + piece) * terms * columns;
    long dbase = ((long)p * pieces + piece) * dterms * columns;
    if (tid < K) {
      aval[tid] = inside ? cheb_value(coef, vbase, terms, columns, tid, x) : 0.0f;
      slope[tid] = inside ? cheb_value(dcoef, dbase, dterms, columns, tid, x) : 0.0f;
    }
    if (tid < D) yval[tid] = y[(long)e * D + tid];
    __syncthreads();

    // Each warp handles four d rows, with four independent 8-lane groups.
    float dr_local = 0.0f;
    for (int group = 0; group < 4; ++group) {
      int d = warp * 4 + group;
      if (d >= D) continue;
      int sub = lane >> 3, sublane = lane & 7;
      float gy = 0.0f, rd = 0.0f;
      for (int k = sublane; k < K; k += 8) {
        float gv = g[d * K + k];
        gy = fmaf(gv, aval[k], gy);
        rd = fmaf(gv * yval[d], slope[k], rd);
      }
      unsigned mask = 0xffu << (sub * 8);
      for (int off = 4; off > 0; off >>= 1) {
        gy += __shfl_down_sync(mask, gy, off, 8);
        rd += __shfl_down_sync(mask, rd, off, 8);
      }
      if (sublane == 0) {
        grad_y[(long)e * D + d] = gy;
        dr_local += rd;
      }
    }
    // One partial per warp, then warp 0 performs the final butterfly.
    if (lane == 0) part[warp] = dr_local;
    __syncthreads();
    if (warp == 0) {
      float v = lane < 4 ? part[lane] : 0.0f;
      unsigned mask = 0xffffffffu;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
      if (lane == 0) {
        float gd = density_mode ? grad_density[receiver] : grad_density[e];
        // The density column is the final table column.  Its derivative is
        // evaluated by lane 0 with the same Chebyshev recurrence.
        float ds = inside ? cheb_scalar(dcoef, dbase, dterms, columns, columns - 1, x) : 0.0f;
        // radial_table_backward suppresses the derivative below r_lo even
        // though the value is evaluated at the clamped lower endpoint.
        dr[e] = (inside && r[e] >= r_lo) ? v + gd * ds : 0.0f;
      }
    }
    __syncthreads();
  }
}

// Fixed OMAT-0 proposal variant.  D=16, K=64 and three sender-species slots
// are enforced by the Python harness.  Each thread owns eight (d,k) values
// per slot, so the 24 accumulators stay in registers instead of a 3*D*K
// shared-memory tile.  The ABI intentionally matches dmqe_outer_forward so
// the same pointer/shape checks can exercise both implementations.
extern "C" __global__ void dmqe_outer_forward_register(
    const float* __restrict__ r, const int* __restrict__ pair, const long long* __restrict__ key,
    const float* __restrict__ coef, const float* __restrict__ starts,
    const float* __restrict__ scales, const int* __restrict__ bins,
    const float* __restrict__ y, const int* __restrict__ row_ptr,
    float* __restrict__ out, float* __restrict__ density_edge, float* __restrict__ density_receiver,
    int nodes, int slots, int pieces, int n_bins, int terms, int columns,
    float r_lo, float bin_scale, float r_max, int D, int K, int density_mode, int inner) {
  constexpr int CD = 16, CK = 64, CS = 3, CT = 8;
  extern __shared__ float sm[];
  float* aval = sm;
  float* yval = aval + CK;
  int receiver = (int)blockIdx.x;
  if (receiver >= nodes) return;
  int tid = (int)threadIdx.x;
  // Keep the slot arrays statically addressable.  A dynamically indexed
  // accum[slot * CT + t] is liable to become a local-memory array under
  // NVRTC, invalidating the register-accumulator experiment.
  float accum0[CT], accum1[CT], accum2[CT];
  #pragma unroll
  for (int i = 0; i < CT; ++i) {
    accum0[i] = 0.0f;
    accum1[i] = 0.0f;
    accum2[i] = 0.0f;
  }
  if (tid < CK) aval[tid] = 0.0f;
  if (tid < CD) yval[tid] = 0.0f;
  if (tid == 0 && density_mode) density_receiver[receiver] = 0.0f;
  __syncthreads();

  int begin = row_ptr[receiver], end = row_ptr[receiver + 1];
  for (int e = begin; e < end; ++e) {
    int slot = (int)(key[e] % CS);
    int p = pair[e];
    int piece; bool inside;
    float x = edge_x(r[e], starts, scales, bins, n_bins, r_lo, bin_scale, r_max, &piece, &inside);
    long base = ((long)p * pieces + piece) * terms * columns;
    if (tid < CK) aval[tid] = inside ? cheb_value(coef, base, terms, columns, tid, x) : 0.0f;
    if (tid < CD) yval[tid] = y[(long)e * CD + tid];
    if (tid == 0) {
      float den = inside ? cheb_scalar(coef, base, terms, columns, columns - 1, x) : 0.0f;
      if (density_mode) density_receiver[receiver] += den;
      else density_edge[e] = den;
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < CT; ++t) {
      int q = tid + 128 * t;
      int d = q / CK, k = q - d * CK;
      if (slot == 0) accum0[t] = fmaf(yval[d], aval[k], accum0[t]);
      else if (slot == 1) accum1[t] = fmaf(yval[d], aval[k], accum1[t]);
      else accum2[t] = fmaf(yval[d], aval[k], accum2[t]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int t = 0; t < CT; ++t) {
    int q = tid + 128 * t;
    int d = q / CK, k = q - d * CK;
    long dst0 = inner == 1
        ? (long)(receiver * CS) * (CD * CK) + q
        : (long)receiver * CS * CD * CK + (long)d * CS * CK + k;
    long dst1 = inner == 1
        ? (long)(receiver * CS + 1) * (CD * CK) + q
        : (long)receiver * CS * CD * CK + (long)d * CS * CK + CK + k;
    long dst2 = inner == 1
        ? (long)(receiver * CS + 2) * (CD * CK) + q
        : (long)receiver * CS * CD * CK + (long)d * CS * CK + 2 * CK + k;
    out[dst0] = accum0[t];
    out[dst1] = accum1[t];
    out[dst2] = accum2[t];
  }
}

// Register-accumulator backward proposal.  One eight-lane subgroup owns one
// d row; unlike dmqe_outer_backward there is no redundant outer group loop.
extern "C" __global__ void dmqe_outer_backward_register(
    const float* __restrict__ grad, const float* __restrict__ r,
    const int* __restrict__ pair, const long long* __restrict__ key,
    const float* __restrict__ coef, const float* __restrict__ dcoef,
    const float* __restrict__ starts, const float* __restrict__ scales,
    const int* __restrict__ bins, const float* __restrict__ grad_density,
    const float* __restrict__ y, const int* __restrict__ row_ptr,
    float* __restrict__ dr, float* __restrict__ grad_y,
    int nodes, int slots, int pieces, int n_bins, int terms, int dterms, int columns,
    float r_lo, float bin_scale, float r_max, int D, int K, int density_mode, int inner) {
  constexpr int CD = 16, CK = 64, CS = 3;
  extern __shared__ float sm[];
  float* g = sm;
  float* aval = g + CD * CK;
  float* slope = aval + CK;
  float* yval = slope + CK;
  float* part = yval + CD;
  int receiver = (int)blockIdx.x, tid = (int)threadIdx.x;
  if (receiver >= nodes) return;
  int lane = tid & 31, warp = tid >> 5;
  int group = lane >> 3, sublane = lane & 7;
  int d = warp * 4 + group;
  int begin = row_ptr[receiver], end = row_ptr[receiver + 1];
  long long current = -1;
  for (int e = begin; e < end; ++e) {
    long long edge_key = key[e];
    if (edge_key != current) {
      int slot = (int)(edge_key % CS);
      for (int i = tid; i < CD * CK; i += blockDim.x) {
        int gd = i / CK, k = i - gd * CK;
        long src = inner == 1
            ? (long)(receiver * CS + slot) * (CD * CK) + i
            : (long)receiver * CS * CD * CK + (long)gd * CS * CK + (long)slot * CK + k;
        g[i] = grad[src];
      }
      current = edge_key;
    }
    int p = pair[e];
    int piece; bool inside;
    float x = edge_x(r[e], starts, scales, bins, n_bins, r_lo, bin_scale, r_max, &piece, &inside);
    long vbase = ((long)p * pieces + piece) * terms * columns;
    long dbase = ((long)p * pieces + piece) * dterms * columns;
    if (tid < CK) {
      aval[tid] = inside ? cheb_value(coef, vbase, terms, columns, tid, x) : 0.0f;
      slope[tid] = inside ? cheb_value(dcoef, dbase, dterms, columns, tid, x) : 0.0f;
    }
    if (tid < CD) yval[tid] = y[(long)e * CD + tid];
    __syncthreads();

    float gy = 0.0f, rd = 0.0f;
    for (int k = sublane; k < CK; k += 8) {
      float gv = g[d * CK + k];
      gy = fmaf(gv, aval[k], gy);
      rd = fmaf(gv * yval[d], slope[k], rd);
    }
    unsigned mask = 0xffu << (group * 8);
    for (int off = 4; off > 0; off >>= 1) {
      gy += __shfl_down_sync(mask, gy, off, 8);
      rd += __shfl_down_sync(mask, rd, off, 8);
    }
    if (sublane == 0) grad_y[(long)e * CD + d] = gy;
    float leader_rd = sublane == 0 ? rd : 0.0f;
    unsigned full = 0xffffffffu;
    leader_rd += __shfl_down_sync(full, leader_rd, 16);
    leader_rd += __shfl_down_sync(full, leader_rd, 8);
    if (lane == 0) part[warp] = leader_rd;
    __syncthreads();
    if (warp == 0) {
      float total = lane < 4 ? part[lane] : 0.0f;
      for (int off = 2; off > 0; off >>= 1) total += __shfl_down_sync(full, total, off);
      if (lane == 0) {
        float gd = density_mode ? grad_density[receiver] : grad_density[e];
        float ds = inside ? cheb_scalar(dcoef, dbase, dterms, columns, columns - 1, x) : 0.0f;
        dr[e] = (inside && r[e] >= r_lo) ? total + gd * ds : 0.0f;
      }
    }
    __syncthreads();
  }
}
