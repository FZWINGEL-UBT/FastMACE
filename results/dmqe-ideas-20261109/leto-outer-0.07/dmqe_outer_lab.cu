
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
        dr[e] = inside ? v + gd * ds : 0.0f;
      }
    }
    __syncthreads();
  }
}
