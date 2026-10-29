// OPT2-T01 gate 1 prototype: the receiver-streaming forward (vendored OEq PR 223 template, see
// fastmace/kernels/oeq_streaming/NOTICE.md) with the folded radial last layer computed inside the kernel.
//
// w[e, slot, c] = sum_k a[e, k] * Wf[slot_start + c, k] is never stored: a block owns R receivers and a slice of
// CS channels, keeps Wf for its slice in shared memory ([K][S][CS], loaded once), stages each receiver's next T
// edges' a rows in shared memory, and every thread (receiver, channel) forms the T x S weights it needs in
// registers before running the template's unrolled path code on them.

using int64_t = signed long long;
using int32_t = int;
using scalar_t = float;
using acc_t = float;

constexpr int INPUT_DIM = 512;
constexpr int EDGE_DIM = 16;
constexpr int OUTPUT_DIM = 5120;
constexpr int CHANNEL_DIM = 128;
constexpr int K = 65;
constexpr int CS = 16;
constexpr int R = 8;
constexpr int T = 4;
constexpr int S = 10;
__device__ __constant__ int SLOT_START[S] = { 0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152 };

extern "C" __global__ void __launch_bounds__(R * CS) fused_projected_forward(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ sh,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ wf,
    const int32_t* __restrict__ senders,
    const int32_t* __restrict__ row_ptr,
    scalar_t* __restrict__ out) {

    extern __shared__ scalar_t smem[];
    scalar_t* wf_s = smem;                  // [K][S][CS]
    scalar_t* a_s = smem + K * S * CS;      // [R][T][K]
    __shared__ int max_chunks;

    const int tid = threadIdx.x;
    const int rl = tid / CS;
    const int cl = tid - rl * CS;
    const int channel = int(blockIdx.y) * CS + cl;
    const int64_t node = int64_t(blockIdx.x) * R + rl;
    const bool active = node < node_count;

    for (int i = tid; i < K * S * CS; i += R * CS) {
        const int k = i / (S * CS);
        const int rest = i - k * (S * CS);
        const int s = rest / CS;
        const int c = rest - s * CS;
        wf_s[i] = wf[int64_t(SLOT_START[s] + int(blockIdx.y) * CS + c) * K + k];
    }
    if (tid == 0) max_chunks = 0;
    __syncthreads();

    int64_t begin = 0, end = 0;
    if (active) {
        begin = row_ptr[node] < 0 ? 0 : row_ptr[node];
        end = row_ptr[node + 1] < begin ? begin : row_ptr[node + 1];
        end = end > edge_count ? edge_count : end;
        if (cl == 0) atomicMax(&max_chunks, int((end - begin + T - 1) / T));
    }
    const int64_t output_base = node * OUTPUT_DIM;
        scalar_t output_0_0 = scalar_t(0);
        scalar_t output_128_0 = scalar_t(0);
        scalar_t output_256_0 = scalar_t(0);
        scalar_t output_256_1 = scalar_t(0);
        scalar_t output_256_2 = scalar_t(0);
        scalar_t output_640_0 = scalar_t(0);
        scalar_t output_640_1 = scalar_t(0);
        scalar_t output_640_2 = scalar_t(0);
        scalar_t output_1024_0 = scalar_t(0);
        scalar_t output_1024_1 = scalar_t(0);
        scalar_t output_1024_2 = scalar_t(0);
        scalar_t output_1408_0 = scalar_t(0);
        scalar_t output_1408_1 = scalar_t(0);
        scalar_t output_1408_2 = scalar_t(0);
        scalar_t output_1408_3 = scalar_t(0);
        scalar_t output_1408_4 = scalar_t(0);
        scalar_t output_2048_0 = scalar_t(0);
        scalar_t output_2048_1 = scalar_t(0);
        scalar_t output_2048_2 = scalar_t(0);
        scalar_t output_2048_3 = scalar_t(0);
        scalar_t output_2048_4 = scalar_t(0);
        scalar_t output_2688_0 = scalar_t(0);
        scalar_t output_2688_1 = scalar_t(0);
        scalar_t output_2688_2 = scalar_t(0);
        scalar_t output_2688_3 = scalar_t(0);
        scalar_t output_2688_4 = scalar_t(0);
        scalar_t output_3328_0 = scalar_t(0);
        scalar_t output_3328_1 = scalar_t(0);
        scalar_t output_3328_2 = scalar_t(0);
        scalar_t output_3328_3 = scalar_t(0);
        scalar_t output_3328_4 = scalar_t(0);
        scalar_t output_3328_5 = scalar_t(0);
        scalar_t output_3328_6 = scalar_t(0);
        scalar_t output_4224_0 = scalar_t(0);
        scalar_t output_4224_1 = scalar_t(0);
        scalar_t output_4224_2 = scalar_t(0);
        scalar_t output_4224_3 = scalar_t(0);
        scalar_t output_4224_4 = scalar_t(0);
        scalar_t output_4224_5 = scalar_t(0);
        scalar_t output_4224_6 = scalar_t(0);
    __syncthreads();
    const int chunks = max_chunks;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk) __syncthreads();
        for (int i = tid; i < R * T * K; i += R * CS) {
            const int r = i / (T * K);
            const int rest = i - r * (T * K);
            const int t = rest / K;
            const int k = rest - t * K;
            const int64_t n = int64_t(blockIdx.x) * R + r;
            scalar_t v = scalar_t(0);
            if (n < node_count) {
                int64_t b = row_ptr[n] < 0 ? 0 : row_ptr[n];
                int64_t f = row_ptr[n + 1] < b ? b : row_ptr[n + 1];
                f = f > edge_count ? edge_count : f;
                const int64_t e = b + int64_t(chunk) * T + t;
                if (e < f) v = a[e * K + k];
            }
            a_s[i] = v;
        }
        __syncthreads();
        if (!active) continue;

        acc_t w[T][S];
        #pragma unroll
        for (int t = 0; t < T; ++t)
            #pragma unroll
            for (int s = 0; s < S; ++s) w[t][s] = acc_t(0);
        #pragma unroll 5
        for (int k = 0; k < K; ++k) {
            scalar_t wk[S];
            #pragma unroll
            for (int s = 0; s < S; ++s) wk[s] = wf_s[(k * S + s) * CS + cl];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const scalar_t av = a_s[(rl * T + t) * K + k];
                #pragma unroll
                for (int s = 0; s < S; ++s) w[t][s] += av * wk[s];
            }
        }

        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int64_t e = begin + int64_t(chunk) * T + t;
            if (e >= end) break;
            const int32_t sender = senders[e];
            if (sender < 0 || sender >= node_count) continue;
            const int64_t input_base = int64_t(sender) * INPUT_DIM;
            const int64_t edge_base = e * EDGE_DIM;
            
                 {
                    const scalar_t radial_weight = w[t][0];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_0_0 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][1];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 2
                + 0 * 3;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 3
                + 0 * 3;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.279a74590331dp-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
            ;
                        output_128_0 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][2];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                        output_256_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 2
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_256_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 3
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                        output_256_2 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][3];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                        output_640_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_640_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                        output_640_2 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][4];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.43d136248490fp-2f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(-0x1.186f174f88471p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
            ;
                        output_1024_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.43d1362484910p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
            ;
                        output_1024_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(-0x1.43d136248490fp-2f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
            ;
                        output_1024_2 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][5];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_1408_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_1408_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000001p+0f)
            ;
                        output_1408_2 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_1408_3 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_1408_4 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][6];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 3
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 1
                + 0 * 3;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                        output_2048_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 2
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 1
                + 0 * 3;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
            ;
                        output_2048_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 2
                + 0 * 3;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 3
                + 0 * 3;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f)
            ;
                        output_2048_2 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 3
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 2
                + 0 * 3;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
            ;
                        output_2048_3 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 3
                + 0 * 3;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                        output_2048_4 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][7];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 13
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 15
                + 0 * 7;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 10
                + 0 * 7;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 9
                + 0 * 7;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];
            const int64_t angular_value_input_index_4 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_4 =
                edge_base + 11
                + 0 * 7;
            const scalar_t angular_value_input_value_4 =
                x[angular_value_input_index_4];
            const scalar_t angular_value_edge_value_4 =
                sh[angular_value_edge_index_4];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(0x1.31fa808c55b43p-1f)
                +
                angular_value_input_value_4
                    * angular_value_edge_value_4
                * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f)
            ;
                        output_2688_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 12
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 14
                + 0 * 7;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 11
                + 0 * 7;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 10
                + 0 * 7;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.3c03650e00e03p-1f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f)
            ;
                        output_2688_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 11
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 12
                + 0 * 7;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 13
                + 0 * 7;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.11acee560242ap-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.11acee560242ap-1f)
            ;
                        output_2688_2 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 10
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 13
                + 0 * 7;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 12
                + 0 * 7;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 14
                + 0 * 7;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.3c03650e00e03p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f)
            ;
                        output_2688_3 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 9
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 11
                + 0 * 7;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 14
                + 0 * 7;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 13
                + 0 * 7;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];
            const int64_t angular_value_input_index_4 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_4 =
                edge_base + 15
                + 0 * 7;
            const scalar_t angular_value_input_value_4 =
                x[angular_value_input_index_4];
            const scalar_t angular_value_edge_value_4 =
                sh[angular_value_edge_index_4];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.31fa808c55b43p-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.3c03650e00e03p-3f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f)
                +
                angular_value_input_value_4
                    * angular_value_edge_value_4
                * static_cast<scalar_t>(0x1.31fa808c55b43p-1f)
            ;
                        output_2688_4 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][8];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 9
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_3328_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 10
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_3328_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 11
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_3328_2 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 12
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000001p+0f)
            ;
                        output_3328_3 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 13
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_3328_4 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 14
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_3328_5 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 15
                + 0 * 7;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                        output_3328_6 += radial_weight * angular_value;
                    }
                }
                 {
                    const scalar_t radial_weight = w[t][9];
                     {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                        output_4224_0 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
            ;
                        output_4224_1 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(0x1.43d136248490fp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.75e9746a0b099p-3f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.75e9746a0b099p-1f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f)
            ;
                        output_4224_2 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f)
            ;
                        output_4224_3 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 6
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];
            const int64_t angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_3 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_3 =
                x[angular_value_input_index_3];
            const scalar_t angular_value_edge_value_3 =
                sh[angular_value_edge_index_3];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.75e9746a0b099p-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.43d136248490fp-1f)
                +
                angular_value_input_value_3
                    * angular_value_edge_value_3
                * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f)
            ;
                        output_4224_4 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];
            const int64_t angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_2 =
                edge_base + 7
                + 0 * 5;
            const scalar_t angular_value_input_value_2 =
                x[angular_value_input_index_2];
            const scalar_t angular_value_edge_value_2 =
                sh[angular_value_edge_index_2];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.279a74590331cp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                angular_value_input_value_2
                    * angular_value_edge_value_2
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
            ;
                        output_4224_5 += radial_weight * angular_value;
                    } {
                        
            const int64_t angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t angular_value_input_value_0 =
                x[angular_value_input_index_0];
            const scalar_t angular_value_edge_value_0 =
                sh[angular_value_edge_index_0];
            const int64_t angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t angular_value_input_value_1 =
                x[angular_value_input_index_1];
            const scalar_t angular_value_edge_value_1 =
                sh[angular_value_edge_index_1];

        const scalar_t angular_value =
                angular_value_input_value_0
                    * angular_value_edge_value_0
                * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f)
                +
                angular_value_input_value_1
                    * angular_value_edge_value_1
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                        output_4224_6 += radial_weight * angular_value;
                    }
                }
        }
    }
    if (!active) return;
     {
        const int64_t output_index =
            output_base + 0 + channel;
        out[output_index] = output_0_0;
    } {
        const int64_t output_index =
            output_base + 128 + channel;
        out[output_index] = output_128_0;
    } {
        const int64_t output_index =
            output_base + 256 + channel;
        out[output_index] = output_256_0;
    } {
        const int64_t output_index =
            output_base + 384 + channel;
        out[output_index] = output_256_1;
    } {
        const int64_t output_index =
            output_base + 512 + channel;
        out[output_index] = output_256_2;
    } {
        const int64_t output_index =
            output_base + 640 + channel;
        out[output_index] = output_640_0;
    } {
        const int64_t output_index =
            output_base + 768 + channel;
        out[output_index] = output_640_1;
    } {
        const int64_t output_index =
            output_base + 896 + channel;
        out[output_index] = output_640_2;
    } {
        const int64_t output_index =
            output_base + 1024 + channel;
        out[output_index] = output_1024_0;
    } {
        const int64_t output_index =
            output_base + 1152 + channel;
        out[output_index] = output_1024_1;
    } {
        const int64_t output_index =
            output_base + 1280 + channel;
        out[output_index] = output_1024_2;
    } {
        const int64_t output_index =
            output_base + 1408 + channel;
        out[output_index] = output_1408_0;
    } {
        const int64_t output_index =
            output_base + 1536 + channel;
        out[output_index] = output_1408_1;
    } {
        const int64_t output_index =
            output_base + 1664 + channel;
        out[output_index] = output_1408_2;
    } {
        const int64_t output_index =
            output_base + 1792 + channel;
        out[output_index] = output_1408_3;
    } {
        const int64_t output_index =
            output_base + 1920 + channel;
        out[output_index] = output_1408_4;
    } {
        const int64_t output_index =
            output_base + 2048 + channel;
        out[output_index] = output_2048_0;
    } {
        const int64_t output_index =
            output_base + 2176 + channel;
        out[output_index] = output_2048_1;
    } {
        const int64_t output_index =
            output_base + 2304 + channel;
        out[output_index] = output_2048_2;
    } {
        const int64_t output_index =
            output_base + 2432 + channel;
        out[output_index] = output_2048_3;
    } {
        const int64_t output_index =
            output_base + 2560 + channel;
        out[output_index] = output_2048_4;
    } {
        const int64_t output_index =
            output_base + 2688 + channel;
        out[output_index] = output_2688_0;
    } {
        const int64_t output_index =
            output_base + 2816 + channel;
        out[output_index] = output_2688_1;
    } {
        const int64_t output_index =
            output_base + 2944 + channel;
        out[output_index] = output_2688_2;
    } {
        const int64_t output_index =
            output_base + 3072 + channel;
        out[output_index] = output_2688_3;
    } {
        const int64_t output_index =
            output_base + 3200 + channel;
        out[output_index] = output_2688_4;
    } {
        const int64_t output_index =
            output_base + 3328 + channel;
        out[output_index] = output_3328_0;
    } {
        const int64_t output_index =
            output_base + 3456 + channel;
        out[output_index] = output_3328_1;
    } {
        const int64_t output_index =
            output_base + 3584 + channel;
        out[output_index] = output_3328_2;
    } {
        const int64_t output_index =
            output_base + 3712 + channel;
        out[output_index] = output_3328_3;
    } {
        const int64_t output_index =
            output_base + 3840 + channel;
        out[output_index] = output_3328_4;
    } {
        const int64_t output_index =
            output_base + 3968 + channel;
        out[output_index] = output_3328_5;
    } {
        const int64_t output_index =
            output_base + 4096 + channel;
        out[output_index] = output_3328_6;
    } {
        const int64_t output_index =
            output_base + 4224 + channel;
        out[output_index] = output_4224_0;
    } {
        const int64_t output_index =
            output_base + 4352 + channel;
        out[output_index] = output_4224_1;
    } {
        const int64_t output_index =
            output_base + 4480 + channel;
        out[output_index] = output_4224_2;
    } {
        const int64_t output_index =
            output_base + 4608 + channel;
        out[output_index] = output_4224_3;
    } {
        const int64_t output_index =
            output_base + 4736 + channel;
        out[output_index] = output_4224_4;
    } {
        const int64_t output_index =
            output_base + 4864 + channel;
        out[output_index] = output_4224_5;
    } {
        const int64_t output_index =
            output_base + 4992 + channel;
        out[output_index] = output_4224_6;
    }
}