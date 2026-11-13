// Vendored from OpenEquivariance PR 223 (see NOTICE.md); node offsets adapted for ir_mul.
// Receiver-owned forward kernels and edge-owned reverse kernels generated from
// sparse coupling paths.
using int64_t = signed long long;
using int32_t = int;
using scalar_t = float;
using acc_t = float;

constexpr int INPUT_DIM = 512;
constexpr int EDGE_DIM = 16;
constexpr int OUTPUT_DIM = 5120;
constexpr int WEIGHT_DIM = 1280;
constexpr int CHANNEL_DIM = 128;
constexpr int LOGICAL_GROUP_SIZE =
    32;
constexpr unsigned int FULL_MASK = 0xffffffffu;

struct EdgeRange {
    int64_t begin;
    int64_t end;
};

__device__ __forceinline__ EdgeRange safe_edge_range(
    const int32_t* row_ptr,
    const int64_t node,
    const int64_t edge_count) {
    const int64_t begin = row_ptr[node] < 0 ? 0 : row_ptr[node];
    int64_t end = row_ptr[node + 1] < begin ? begin : row_ptr[node + 1];
    end = end > edge_count ? edge_count : end;
    return {begin, end};
}












// One thread owns (receiver node, channel), scans that receiver's CSR edges,
// and directly writes all of its output components.
extern "C" __global__ void oeq_projected_forward(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    scalar_t* out) {

    const int64_t node_channel_index =
        int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t node_channel_count = node_count * CHANNEL_DIM;
    if (node_channel_index >= node_channel_count) return;

    const int64_t node = node_channel_index / CHANNEL_DIM;
    const int channel = int(node_channel_index - node * CHANNEL_DIM);
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

    const EdgeRange edge_range =
        safe_edge_range(row_ptr, node, edge_count);
    for (int64_t e = edge_range.begin; e < edge_range.end; ++e) {
        const int32_t sender = senders[e];
        if (sender < 0 || sender >= node_count) continue;

        const int64_t input_base = int64_t(sender) * INPUT_DIM;
        const int64_t edge_base = e * EDGE_DIM;
        const int64_t weight_base = e * WEIGHT_DIM;

        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_0_0 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_128_0 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_256_0 +=
                        radial_weight * angular_value;
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
                    output_256_1 +=
                        radial_weight * angular_value;
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
                    output_256_2 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_640_0 +=
                        radial_weight * angular_value;
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
                    output_640_1 +=
                        radial_weight * angular_value;
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
                    output_640_2 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_1024_0 +=
                        radial_weight * angular_value;
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
                    output_1024_1 +=
                        radial_weight * angular_value;
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
                    output_1024_2 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_1408_0 +=
                        radial_weight * angular_value;
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
                    output_1408_1 +=
                        radial_weight * angular_value;
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
                    output_1408_2 +=
                        radial_weight * angular_value;
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
                    output_1408_3 +=
                        radial_weight * angular_value;
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
                    output_1408_4 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_2048_0 +=
                        radial_weight * angular_value;
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
                    output_2048_1 +=
                        radial_weight * angular_value;
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
                    output_2048_2 +=
                        radial_weight * angular_value;
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
                    output_2048_3 +=
                        radial_weight * angular_value;
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
                    output_2048_4 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_2688_0 +=
                        radial_weight * angular_value;
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
                    output_2688_1 +=
                        radial_weight * angular_value;
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
                    output_2688_2 +=
                        radial_weight * angular_value;
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
                    output_2688_3 +=
                        radial_weight * angular_value;
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
                    output_2688_4 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_3328_0 +=
                        radial_weight * angular_value;
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
                    output_3328_1 +=
                        radial_weight * angular_value;
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
                    output_3328_2 +=
                        radial_weight * angular_value;
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
                    output_3328_3 +=
                        radial_weight * angular_value;
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
                    output_3328_4 +=
                        radial_weight * angular_value;
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
                    output_3328_5 +=
                        radial_weight * angular_value;
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
                    output_3328_6 +=
                        radial_weight * angular_value;
                }
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

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
                    output_4224_0 +=
                        radial_weight * angular_value;
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
                    output_4224_1 +=
                        radial_weight * angular_value;
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
                    output_4224_2 +=
                        radial_weight * angular_value;
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
                    output_4224_3 +=
                        radial_weight * angular_value;
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
                    output_4224_4 +=
                        radial_weight * angular_value;
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
                    output_4224_5 +=
                        radial_weight * angular_value;
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
                    output_4224_6 +=
                        radial_weight * angular_value;
                }
            }
    }
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

// The same (receiver node, channel) ownership computes the forward tangent.
extern "C" __global__ void oeq_projected_forward_jvp(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    const scalar_t* tx,
    const scalar_t* tsh,
    const scalar_t* tweights,
    scalar_t* out) {

    const int64_t node_channel_index =
        int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t node_channel_count = node_count * CHANNEL_DIM;
    if (node_channel_index >= node_channel_count) return;

    const int64_t node = node_channel_index / CHANNEL_DIM;
    const int channel = int(node_channel_index - node * CHANNEL_DIM);
    const int64_t output_base = node * OUTPUT_DIM;
        scalar_t tangent_output_0_0 = scalar_t(0);
        scalar_t tangent_output_128_0 = scalar_t(0);
        scalar_t tangent_output_256_0 = scalar_t(0);
        scalar_t tangent_output_256_1 = scalar_t(0);
        scalar_t tangent_output_256_2 = scalar_t(0);
        scalar_t tangent_output_640_0 = scalar_t(0);
        scalar_t tangent_output_640_1 = scalar_t(0);
        scalar_t tangent_output_640_2 = scalar_t(0);
        scalar_t tangent_output_1024_0 = scalar_t(0);
        scalar_t tangent_output_1024_1 = scalar_t(0);
        scalar_t tangent_output_1024_2 = scalar_t(0);
        scalar_t tangent_output_1408_0 = scalar_t(0);
        scalar_t tangent_output_1408_1 = scalar_t(0);
        scalar_t tangent_output_1408_2 = scalar_t(0);
        scalar_t tangent_output_1408_3 = scalar_t(0);
        scalar_t tangent_output_1408_4 = scalar_t(0);
        scalar_t tangent_output_2048_0 = scalar_t(0);
        scalar_t tangent_output_2048_1 = scalar_t(0);
        scalar_t tangent_output_2048_2 = scalar_t(0);
        scalar_t tangent_output_2048_3 = scalar_t(0);
        scalar_t tangent_output_2048_4 = scalar_t(0);
        scalar_t tangent_output_2688_0 = scalar_t(0);
        scalar_t tangent_output_2688_1 = scalar_t(0);
        scalar_t tangent_output_2688_2 = scalar_t(0);
        scalar_t tangent_output_2688_3 = scalar_t(0);
        scalar_t tangent_output_2688_4 = scalar_t(0);
        scalar_t tangent_output_3328_0 = scalar_t(0);
        scalar_t tangent_output_3328_1 = scalar_t(0);
        scalar_t tangent_output_3328_2 = scalar_t(0);
        scalar_t tangent_output_3328_3 = scalar_t(0);
        scalar_t tangent_output_3328_4 = scalar_t(0);
        scalar_t tangent_output_3328_5 = scalar_t(0);
        scalar_t tangent_output_3328_6 = scalar_t(0);
        scalar_t tangent_output_4224_0 = scalar_t(0);
        scalar_t tangent_output_4224_1 = scalar_t(0);
        scalar_t tangent_output_4224_2 = scalar_t(0);
        scalar_t tangent_output_4224_3 = scalar_t(0);
        scalar_t tangent_output_4224_4 = scalar_t(0);
        scalar_t tangent_output_4224_5 = scalar_t(0);
        scalar_t tangent_output_4224_6 = scalar_t(0);

    const EdgeRange edge_range =
        safe_edge_range(row_ptr, node, edge_count);
    for (int64_t e = edge_range.begin; e < edge_range.end; ++e) {
        const int32_t sender = senders[e];
        if (sender < 0 || sender >= node_count) continue;

        const int64_t input_base = int64_t(sender) * INPUT_DIM;
        const int64_t edge_base = e * EDGE_DIM;
        const int64_t weight_base = e * WEIGHT_DIM;
        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_0_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 2
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 3
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.279a74590331dp-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
            ;
                    tangent_output_128_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                    tangent_output_256_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 2
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_256_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 3
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                    tangent_output_256_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                    tangent_output_640_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_640_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 0
                + 0 * 1;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.fffffffffffffp-1f)
            ;
                    tangent_output_640_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.43d136248490fp-2f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(-0x1.186f174f88471p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
            ;
                    tangent_output_1024_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.43d1362484910p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
            ;
                    tangent_output_1024_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(-0x1.43d136248490fp-2f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(0x1.186f174f88471p-1f)
            ;
                    tangent_output_1024_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_1408_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_1408_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000001p+0f)
            ;
                    tangent_output_1408_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_1408_3 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_1408_4 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 3
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 1
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                    tangent_output_2048_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 2
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 1
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
            ;
                    tangent_output_2048_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 2
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 3
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f)
            ;
                    tangent_output_2048_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 3
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 2
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f)
            ;
                    tangent_output_2048_3 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 1
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 3
                + 0 * 3;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                    tangent_output_2048_4 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 13
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 15
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 10
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 9
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_4 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_4 =
                edge_base + 11
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_4 =
                x[tangent_angular_value_input_index_4];
            const scalar_t tangent_angular_value_edge_value_4 =
                sh[tangent_angular_value_edge_index_4];
                const scalar_t tangent_angular_value_tangent_input_value_4 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_4 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(0x1.31fa808c55b43p-1f)
                +
                (tangent_angular_value_tangent_input_value_4
                    * tangent_angular_value_edge_value_4
                    + tangent_angular_value_input_value_4
                    * tangent_angular_value_tangent_edge_value_4)
                * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f)
            ;
                    tangent_output_2688_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 12
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 14
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 11
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 10
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.3c03650e00e03p-1f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f)
            ;
                    tangent_output_2688_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 11
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 12
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 13
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.11acee560242ap-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.11acee560242ap-1f)
            ;
                    tangent_output_2688_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 10
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 13
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 12
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 14
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.3c03650e00e03p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f)
            ;
                    tangent_output_2688_3 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 9
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 11
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 14
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 13
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_4 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_4 =
                edge_base + 15
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_4 =
                x[tangent_angular_value_input_index_4];
            const scalar_t tangent_angular_value_edge_value_4 =
                sh[tangent_angular_value_edge_index_4];
                const scalar_t tangent_angular_value_tangent_input_value_4 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_4 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.31fa808c55b43p-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.3c03650e00e03p-3f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f)
                +
                (tangent_angular_value_tangent_input_value_4
                    * tangent_angular_value_edge_value_4
                    + tangent_angular_value_input_value_4
                    * tangent_angular_value_tangent_edge_value_4)
                * static_cast<scalar_t>(0x1.31fa808c55b43p-1f)
            ;
                    tangent_output_2688_4 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 9
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_3328_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 10
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_3328_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 11
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_3328_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 12
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000001p+0f)
            ;
                    tangent_output_3328_3 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 13
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_3328_4 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 14
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_3328_5 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 0 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 15
                + 0 * 7;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.0000000000000p+0f)
            ;
                    tangent_output_3328_6 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                    tangent_output_4224_0 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
            ;
                    tangent_output_4224_1 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(0x1.43d136248490fp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.75e9746a0b099p-3f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.75e9746a0b099p-1f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f)
            ;
                    tangent_output_4224_2 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f)
            ;
                    tangent_output_4224_3 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 6
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_3 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_3 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_3 =
                x[tangent_angular_value_input_index_3];
            const scalar_t tangent_angular_value_edge_value_3 =
                sh[tangent_angular_value_edge_index_3];
                const scalar_t tangent_angular_value_tangent_input_value_3 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_3 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.75e9746a0b099p-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.43d136248490fp-1f)
                +
                (tangent_angular_value_tangent_input_value_3
                    * tangent_angular_value_edge_value_3
                    + tangent_angular_value_input_value_3
                    * tangent_angular_value_tangent_edge_value_3)
                * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f)
            ;
                    tangent_output_4224_4 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 5
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 256 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_2 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_2 =
                edge_base + 7
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_2 =
                x[tangent_angular_value_input_index_2];
            const scalar_t tangent_angular_value_edge_value_2 =
                sh[tangent_angular_value_edge_index_2];
                const scalar_t tangent_angular_value_tangent_input_value_2 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_2 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.279a74590331cp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
                +
                (tangent_angular_value_tangent_input_value_2
                    * tangent_angular_value_edge_value_2
                    + tangent_angular_value_input_value_2
                    * tangent_angular_value_tangent_edge_value_2)
                * static_cast<scalar_t>(0x1.279a74590331cp-1f)
            ;
                    tangent_output_4224_5 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
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
                    
            const int64_t tangent_angular_value_input_index_0 =
                input_base + 128 + channel;
            const int64_t tangent_angular_value_edge_index_0 =
                edge_base + 4
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_0 =
                x[tangent_angular_value_input_index_0];
            const scalar_t tangent_angular_value_edge_value_0 =
                sh[tangent_angular_value_edge_index_0];
                const scalar_t tangent_angular_value_tangent_input_value_0 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_0 =
                    scalar_t(0);
            const int64_t tangent_angular_value_input_index_1 =
                input_base + 384 + channel;
            const int64_t tangent_angular_value_edge_index_1 =
                edge_base + 8
                + 0 * 5;
            const scalar_t tangent_angular_value_input_value_1 =
                x[tangent_angular_value_input_index_1];
            const scalar_t tangent_angular_value_edge_value_1 =
                sh[tangent_angular_value_edge_index_1];
                const scalar_t tangent_angular_value_tangent_input_value_1 =
                    scalar_t(0);
                const scalar_t tangent_angular_value_tangent_edge_value_1 =
                    scalar_t(0);

        const scalar_t tangent_angular_value =
                (tangent_angular_value_tangent_input_value_0
                    * tangent_angular_value_edge_value_0
                    + tangent_angular_value_input_value_0
                    * tangent_angular_value_tangent_edge_value_0)
                * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f)
                +
                (tangent_angular_value_tangent_input_value_1
                    * tangent_angular_value_edge_value_1
                    + tangent_angular_value_input_value_1
                    * tangent_angular_value_tangent_edge_value_1)
                * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f)
            ;
                    tangent_output_4224_6 +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                }
            }
    }
     {
        const int64_t output_index =
            output_base + 0 + channel;

        out[output_index] = tangent_output_0_0;
    } {
        const int64_t output_index =
            output_base + 128 + channel;

        out[output_index] = tangent_output_128_0;
    } {
        const int64_t output_index =
            output_base + 256 + channel;

        out[output_index] = tangent_output_256_0;
    } {
        const int64_t output_index =
            output_base + 384 + channel;

        out[output_index] = tangent_output_256_1;
    } {
        const int64_t output_index =
            output_base + 512 + channel;

        out[output_index] = tangent_output_256_2;
    } {
        const int64_t output_index =
            output_base + 640 + channel;

        out[output_index] = tangent_output_640_0;
    } {
        const int64_t output_index =
            output_base + 768 + channel;

        out[output_index] = tangent_output_640_1;
    } {
        const int64_t output_index =
            output_base + 896 + channel;

        out[output_index] = tangent_output_640_2;
    } {
        const int64_t output_index =
            output_base + 1024 + channel;

        out[output_index] = tangent_output_1024_0;
    } {
        const int64_t output_index =
            output_base + 1152 + channel;

        out[output_index] = tangent_output_1024_1;
    } {
        const int64_t output_index =
            output_base + 1280 + channel;

        out[output_index] = tangent_output_1024_2;
    } {
        const int64_t output_index =
            output_base + 1408 + channel;

        out[output_index] = tangent_output_1408_0;
    } {
        const int64_t output_index =
            output_base + 1536 + channel;

        out[output_index] = tangent_output_1408_1;
    } {
        const int64_t output_index =
            output_base + 1664 + channel;

        out[output_index] = tangent_output_1408_2;
    } {
        const int64_t output_index =
            output_base + 1792 + channel;

        out[output_index] = tangent_output_1408_3;
    } {
        const int64_t output_index =
            output_base + 1920 + channel;

        out[output_index] = tangent_output_1408_4;
    } {
        const int64_t output_index =
            output_base + 2048 + channel;

        out[output_index] = tangent_output_2048_0;
    } {
        const int64_t output_index =
            output_base + 2176 + channel;

        out[output_index] = tangent_output_2048_1;
    } {
        const int64_t output_index =
            output_base + 2304 + channel;

        out[output_index] = tangent_output_2048_2;
    } {
        const int64_t output_index =
            output_base + 2432 + channel;

        out[output_index] = tangent_output_2048_3;
    } {
        const int64_t output_index =
            output_base + 2560 + channel;

        out[output_index] = tangent_output_2048_4;
    } {
        const int64_t output_index =
            output_base + 2688 + channel;

        out[output_index] = tangent_output_2688_0;
    } {
        const int64_t output_index =
            output_base + 2816 + channel;

        out[output_index] = tangent_output_2688_1;
    } {
        const int64_t output_index =
            output_base + 2944 + channel;

        out[output_index] = tangent_output_2688_2;
    } {
        const int64_t output_index =
            output_base + 3072 + channel;

        out[output_index] = tangent_output_2688_3;
    } {
        const int64_t output_index =
            output_base + 3200 + channel;

        out[output_index] = tangent_output_2688_4;
    } {
        const int64_t output_index =
            output_base + 3328 + channel;

        out[output_index] = tangent_output_3328_0;
    } {
        const int64_t output_index =
            output_base + 3456 + channel;

        out[output_index] = tangent_output_3328_1;
    } {
        const int64_t output_index =
            output_base + 3584 + channel;

        out[output_index] = tangent_output_3328_2;
    } {
        const int64_t output_index =
            output_base + 3712 + channel;

        out[output_index] = tangent_output_3328_3;
    } {
        const int64_t output_index =
            output_base + 3840 + channel;

        out[output_index] = tangent_output_3328_4;
    } {
        const int64_t output_index =
            output_base + 3968 + channel;

        out[output_index] = tangent_output_3328_5;
    } {
        const int64_t output_index =
            output_base + 4096 + channel;

        out[output_index] = tangent_output_3328_6;
    } {
        const int64_t output_index =
            output_base + 4224 + channel;

        out[output_index] = tangent_output_4224_0;
    } {
        const int64_t output_index =
            output_base + 4352 + channel;

        out[output_index] = tangent_output_4224_1;
    } {
        const int64_t output_index =
            output_base + 4480 + channel;

        out[output_index] = tangent_output_4224_2;
    } {
        const int64_t output_index =
            output_base + 4608 + channel;

        out[output_index] = tangent_output_4224_3;
    } {
        const int64_t output_index =
            output_base + 4736 + channel;

        out[output_index] = tangent_output_4224_4;
    } {
        const int64_t output_index =
            output_base + 4864 + channel;

        out[output_index] = tangent_output_4224_5;
    } {
        const int64_t output_index =
            output_base + 4992 + channel;

        out[output_index] = tangent_output_4224_6;
    }
}

// One logical thread group owns an edge, including on HIP wavefront-64.
// Each lane owns channels separated by LOGICAL_GROUP_SIZE.
extern "C" __global__ void oeq_projected_backward(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    scalar_t* dx,
    scalar_t* dsh,
    scalar_t* dweights) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count)
        return;

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
            acc_t input_gradient_0_0 = acc_t(0);
            acc_t input_gradient_128_0 = acc_t(0);
            acc_t input_gradient_128_1 = acc_t(0);
            acc_t input_gradient_128_2 = acc_t(0);

        // Path weight intervals are disjoint. This lane directly owns each
        // dweights[e, weight] element generated below.
        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 0 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 128 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 256 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 384 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 512 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 640 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 768 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 896 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 1024 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1152 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1280 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 1408 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1536 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1664 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1792 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1920 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 2048 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2176 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2304 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2432 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2560 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 2688 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2816 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2944 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3072 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3200 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 3328 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3456 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3584 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3712 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3840 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3968 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4096 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 4224 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4352 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4480 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4608 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4736 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4864 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4992 + channel;
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }

        // Paths are combined locally for this (edge, channel, input component),
        // but different receiver-owned edges can contribute to the same sender.
            atomicAdd(
                dx + input_base + 0 + channel,
                scalar_t(input_gradient_0_0));
            atomicAdd(
                dx + input_base + 128 + channel,
                scalar_t(input_gradient_128_0));
            atomicAdd(
                dx + input_base + 256 + channel,
                scalar_t(input_gradient_128_1));
            atomicAdd(
                dx + input_base + 384 + channel,
                scalar_t(input_gradient_128_2));
    }

    // Reduce within the logical 32-thread edge owner. Lane 0 writes dsh[e, :].
    for (int offset = 16; offset > 0; offset >>= 1)
        for (int component = 0; component < EDGE_DIM; ++component)
            edge_gradient[component] +=
                __shfl_down_sync(FULL_MASK, edge_gradient[component], offset, 32);

    if (lane == 0)
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
}

// Added for the Langevin project: receiver-owned reverse pass. One thread owns
// (receiver node, channel) like the forward, keeps that node's output gradient
// in registers and scans the receiver's CSR edges: weight gradients are written
// directly (coalesced over channels), edge-feature gradients are reduced over
// the warp's channels when CHANNEL_DIM is a multiple of 32 (a warp never spans
// two nodes; otherwise one atomic per thread) and added atomically,
// sender-feature gradients are added
// atomically. dx and dsh must be zero on entry.
extern "C" __global__ void oeq_projected_backward_receiver(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    const scalar_t* dout,
    scalar_t* dx,
    scalar_t* dsh,
    scalar_t* dweights) {

    const int64_t node_channel_index =
        int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t node_channel_count = node_count * CHANNEL_DIM;
    if (node_channel_index >= node_channel_count) return;

    const int64_t node = node_channel_index / CHANNEL_DIM;
    const int channel = int(node_channel_index - node * CHANNEL_DIM);
    const int lane = threadIdx.x % 32;
    const int64_t output_base = node * OUTPUT_DIM;
        const scalar_t grad_output_0_0 =
            dout[output_base + 0 + channel];
        const scalar_t grad_output_128_0 =
            dout[output_base + 128 + channel];
        const scalar_t grad_output_256_0 =
            dout[output_base + 256 + channel];
        const scalar_t grad_output_256_1 =
            dout[output_base + 384 + channel];
        const scalar_t grad_output_256_2 =
            dout[output_base + 512 + channel];
        const scalar_t grad_output_640_0 =
            dout[output_base + 640 + channel];
        const scalar_t grad_output_640_1 =
            dout[output_base + 768 + channel];
        const scalar_t grad_output_640_2 =
            dout[output_base + 896 + channel];
        const scalar_t grad_output_1024_0 =
            dout[output_base + 1024 + channel];
        const scalar_t grad_output_1024_1 =
            dout[output_base + 1152 + channel];
        const scalar_t grad_output_1024_2 =
            dout[output_base + 1280 + channel];
        const scalar_t grad_output_1408_0 =
            dout[output_base + 1408 + channel];
        const scalar_t grad_output_1408_1 =
            dout[output_base + 1536 + channel];
        const scalar_t grad_output_1408_2 =
            dout[output_base + 1664 + channel];
        const scalar_t grad_output_1408_3 =
            dout[output_base + 1792 + channel];
        const scalar_t grad_output_1408_4 =
            dout[output_base + 1920 + channel];
        const scalar_t grad_output_2048_0 =
            dout[output_base + 2048 + channel];
        const scalar_t grad_output_2048_1 =
            dout[output_base + 2176 + channel];
        const scalar_t grad_output_2048_2 =
            dout[output_base + 2304 + channel];
        const scalar_t grad_output_2048_3 =
            dout[output_base + 2432 + channel];
        const scalar_t grad_output_2048_4 =
            dout[output_base + 2560 + channel];
        const scalar_t grad_output_2688_0 =
            dout[output_base + 2688 + channel];
        const scalar_t grad_output_2688_1 =
            dout[output_base + 2816 + channel];
        const scalar_t grad_output_2688_2 =
            dout[output_base + 2944 + channel];
        const scalar_t grad_output_2688_3 =
            dout[output_base + 3072 + channel];
        const scalar_t grad_output_2688_4 =
            dout[output_base + 3200 + channel];
        const scalar_t grad_output_3328_0 =
            dout[output_base + 3328 + channel];
        const scalar_t grad_output_3328_1 =
            dout[output_base + 3456 + channel];
        const scalar_t grad_output_3328_2 =
            dout[output_base + 3584 + channel];
        const scalar_t grad_output_3328_3 =
            dout[output_base + 3712 + channel];
        const scalar_t grad_output_3328_4 =
            dout[output_base + 3840 + channel];
        const scalar_t grad_output_3328_5 =
            dout[output_base + 3968 + channel];
        const scalar_t grad_output_3328_6 =
            dout[output_base + 4096 + channel];
        const scalar_t grad_output_4224_0 =
            dout[output_base + 4224 + channel];
        const scalar_t grad_output_4224_1 =
            dout[output_base + 4352 + channel];
        const scalar_t grad_output_4224_2 =
            dout[output_base + 4480 + channel];
        const scalar_t grad_output_4224_3 =
            dout[output_base + 4608 + channel];
        const scalar_t grad_output_4224_4 =
            dout[output_base + 4736 + channel];
        const scalar_t grad_output_4224_5 =
            dout[output_base + 4864 + channel];
        const scalar_t grad_output_4224_6 =
            dout[output_base + 4992 + channel];

    const EdgeRange edge_range =
        safe_edge_range(row_ptr, node, edge_count);
    for (int64_t e = edge_range.begin; e < edge_range.end; ++e) {
        const int32_t sender = senders[e];
        if (sender < 0 || sender >= node_count) continue;

        const int64_t input_base = int64_t(sender) * INPUT_DIM;
        const int64_t edge_base = e * EDGE_DIM;
        const int64_t weight_base = e * WEIGHT_DIM;
        acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
            acc_t input_gradient_0_0 = acc_t(0);
            acc_t input_gradient_128_0 = acc_t(0);
            acc_t input_gradient_128_1 = acc_t(0);
            acc_t input_gradient_128_2 = acc_t(0);

        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_0_0;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_128_0;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_256_0;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_256_1;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_256_2;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_640_0;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_640_1;
                     {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_640_2;
                     {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_1024_0;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_1024_1;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_1024_2;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_1408_0;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_1408_1;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_1408_2;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_1408_3;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_1408_4;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_2048_0;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2048_1;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2048_2;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2048_3;
                     {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2048_4;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_2688_0;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2688_1;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2688_2;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2688_3;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_2688_4;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_3328_0;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_3328_1;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_3328_2;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_3328_3;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_3328_4;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_3328_5;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_3328_6;
                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                 {
                    const scalar_t output_gradient = grad_output_4224_0;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_4224_1;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_4224_2;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_4224_3;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_4224_4;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_4224_5;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                } {
                    const scalar_t output_gradient = grad_output_4224_6;
                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_base + edge_component_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                }
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            }
            atomicAdd(
                dx + input_base + 0 + channel,
                scalar_t(input_gradient_0_0));
            atomicAdd(
                dx + input_base + 128 + channel,
                scalar_t(input_gradient_128_0));
            atomicAdd(
                dx + input_base + 256 + channel,
                scalar_t(input_gradient_128_1));
            atomicAdd(
                dx + input_base + 384 + channel,
                scalar_t(input_gradient_128_2));
        // A warp holds 32 channels of this node: reduce over them, one atomic per component.
        for (int offset = 16; offset > 0; offset >>= 1)
            for (int component = 0; component < EDGE_DIM; ++component)
                edge_gradient[component] +=
                    __shfl_down_sync(FULL_MASK, edge_gradient[component], offset);
        if (lane == 0)
            for (int component = 0; component < EDGE_DIM; ++component)
                atomicAdd(dsh + edge_base + component, scalar_t(edge_gradient[component]));
    }
}

// Mixed derivative of the edge-owned reverse pass, with the same logical
// 32-thread group and channel ownership as oeq_projected_backward.
extern "C" __global__ void oeq_projected_backward_jvp(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    const scalar_t* tx,
    const scalar_t* tsh,
    const scalar_t* tweights,
    const scalar_t* tdout,
    scalar_t* tdx,
    scalar_t* tdsh,
    scalar_t* tdweights) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count)
        return;

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    acc_t tangent_edge_gradient[EDGE_DIM] = {acc_t(0)};
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
            acc_t input_gradient_0_0 = acc_t(0);
            acc_t input_gradient_128_0 = acc_t(0);
            acc_t input_gradient_128_1 = acc_t(0);
            acc_t input_gradient_128_2 = acc_t(0);

        // Path weight intervals are disjoint. This lane directly owns each
        // tdweights[e, weight] element generated below.
        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 0 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 128 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331dp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 256 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 384 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 512 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 640 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 768 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 896 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 1024 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);

                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.186f174f88471p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 1152 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.43d1362484910p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 1280 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 1408 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 1536 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 1664 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 1792 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 1920 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 2048 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 2176 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 2304 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 2432 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 2560 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 2688 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);

                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 2816 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);

                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 2944 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3072 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3200 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 3328 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3456 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3584 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3712 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3840 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 3968 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4096 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 0 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_0_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    scalar_t(0);

                acc_t tangent_weight_gradient = acc_t(0);
                 {
                    const int64_t output_index =
                        output_base + 4224 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4352 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4480 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4608 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4736 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4864 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.279a74590331cp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 256 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_1 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);

                    }
                } {
                    const int64_t output_index =
                        output_base + 4992 + channel;
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        scalar_t(0);

                     {
                        const int64_t input_index =
                            input_base + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_0 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);

                    } {
                        const int64_t input_index =
                            input_base + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            scalar_t(0);
                        const scalar_t tangent_edge_value =
                            scalar_t(0);

                        input_gradient_128_2 +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);

                    }
                }

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            }

        // Paths are combined locally for this (edge, channel, input component),
        // but different receiver-owned edges can contribute to the same sender.
            atomicAdd(
                tdx + input_base + 0 + channel,
                scalar_t(input_gradient_0_0));
            atomicAdd(
                tdx + input_base + 128 + channel,
                scalar_t(input_gradient_128_0));
            atomicAdd(
                tdx + input_base + 256 + channel,
                scalar_t(input_gradient_128_1));
            atomicAdd(
                tdx + input_base + 384 + channel,
                scalar_t(input_gradient_128_2));
    }

    // Reduce within the logical 32-thread edge owner. Lane 0 writes tdsh[e, :].
    for (int offset = 16; offset > 0; offset >>= 1)
        for (int component = 0; component < EDGE_DIM; ++component)
            tangent_edge_gradient[component] +=
                __shfl_down_sync(FULL_MASK, tangent_edge_gradient[component], offset, 32);

    if (lane == 0)
        for (int component = 0; component < EDGE_DIM; ++component)
            tdsh[edge_base + component] =
                scalar_t(tangent_edge_gradient[component]);
}